"""The day nap's motion watch (SPEC 5, 7A, 13): motion in front of a still camera.

A day nap ends when something moves in front of the robot (a hand, a person walking past). The
watch reuses the object gate's scene-change test (``scene_change``: thumbnail, pixel share,
head still) with ``vision.object_gate``'s thresholds, but runs on its own, so it works with the
object stage off. ``VisionPipeline`` calls :meth:`SceneMotionWatch.offer` on every captured frame,
before the rate governor drops it; only while ``active`` (a day nap with
``circadian.nap_wake_on_motion``) does it do any work.

- **Rate.** One captured frame per ``vision.scene_motion.interval_s`` (capture time) is tested.
  Resting vision processes one frame per few seconds (``vision_rate_scale``), too slow to catch a
  hand or a person passing; the test is a strided slice and an 80x60 pool, so a few per second
  cost little next to a face detection.
- **Head still.** Every frame with a pose follows the head (``HeadStill``, ``head_still_deg``); a
  move drops the view and the confirmation, and nothing is tested until the head has been still
  for ``settle_s``. Arming (the nap starts) counts as a move: the robot's own move into the rest
  posture is not motion. A frame without a pose is never tested (no joint state or no
  ``FrameSpec``): the camera may be moving.
- **Reference.** The last sample that showed no change (the last still view), so slow daylight
  drift over a long nap never adds up to motion; a change has to differ from that view.
- **Confirmation.** Motion counts when ``confirm_samples`` samples in a row differ from the still
  view (``changed_share`` of the pixels by more than ``pixel_diff``): one odd frame or an exposure
  step does not wake the nap, a hand or a person stays in view longer than that.
- **Rate of reports.** After a report the watch waits for a new still view (as after a move), so
  a hand held still in view reports once. The nap ends on the first report and the watch goes
  inactive, so ``percept.scene_motion`` is rare.

Decisions use capture stamps only (a replay gives the same reports). One lock guards the state;
``snapshot`` and ``reset`` may run on another thread. The copies stay in RAM and are never logged
(SPEC 12); the report has counts only.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.schema import ObjectGateConfig, SceneMotionConfig
from .governor import EPSILON_S
from .scene_change import HeadStill, Pose, moved_share, thumbnail

COUNTS = ("samples", "head_moving", "no_pose", "changed", "confirmed")


@dataclass(frozen=True)
class ScenePercept:
    """Payload of ``percept.scene_motion``: a scalar, never the image."""

    share: float  # share of the copy's pixels that moved


class SceneMotionWatch:
    """See the module docstring. ``cfg`` and ``gate`` are replaced on hot reload."""

    def __init__(self, cfg: SceneMotionConfig, gate: ObjectGateConfig) -> None:
        self.cfg = cfg
        self.gate = gate
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(COUNTS, 0)
        self._disarm()

    def _disarm(self) -> None:
        self._armed = False
        self._head = HeadStill()
        self._reference: np.ndarray | None = None
        self._streak = 0
        self._last_sample: float | None = None

    def offer(
        self, stamp: float, image: np.ndarray, pose: Pose | None, active: bool
    ) -> float | None:
        """One captured frame at ``stamp`` with the head ``pose``; ``active``: a nap waits for
        motion. Returns the moved share when motion is confirmed now, else None."""
        if not active and not self._armed:  # no nap: one flag read, no lock (a stale read only
            return None  # delays the disarm by one frame)
        with self._lock:
            if not active:
                if self._armed:
                    self._disarm()
                return None
            if not self._armed:
                self._armed = True
                self._head.last_move = stamp  # the nap's start: the rest posture moves the head
            if pose is not None and self._head.observe(
                stamp, pose, math.radians(self.gate.head_still_deg)
            ):
                self._drop()
            last = self._last_sample
            if last is not None and stamp - last < self.cfg.interval_s - EPSILON_S:
                return None
            self._last_sample = stamp
            self._counts["samples"] += 1
            if pose is None:
                self._counts["no_pose"] += 1
                self._drop()
                return None
            if self._head.moving(stamp, self.gate.settle_s):
                self._counts["head_moving"] += 1
                return None
            return self._test(stamp, image)

    def _test(self, stamp: float, image: np.ndarray) -> float | None:
        """Compare with the last still view (caller holds the lock)."""
        small = thumbnail(image, self.gate.thumb_width_px, self.gate.thumb_height_px)
        ref = self._reference
        if small is None or ref is None or ref.shape != small.shape:
            self._reference, self._streak = small, 0
            return None
        share = moved_share(small, ref, self.gate.pixel_diff)
        if share <= self.gate.changed_share:
            self._reference, self._streak = small, 0
            return None
        self._counts["changed"] += 1
        self._streak += 1
        if self._streak < self.cfg.confirm_samples:
            return None
        self._counts["confirmed"] += 1
        self._drop()
        self._head.last_move = stamp  # wait for a new still view before the next report
        return share

    def _drop(self) -> None:
        self._reference, self._streak = None, 0

    def snapshot(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): numbers only."""
        with self._lock:
            return {"armed": self._armed, **self._counts}

    def reset(self) -> None:
        """Forget the view and the counts (new session or replay start)."""
        with self._lock:
            self._disarm()
            self._counts = dict.fromkeys(COUNTS, 0)
