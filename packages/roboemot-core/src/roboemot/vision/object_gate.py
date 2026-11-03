"""Scene-change gate of the object stage (SPEC 5, 13): run object detection only when the
scene changed, to save CPU.

The camera sits on the head, so a head move changes the image without any change in the scene.
The gate does not compensate for it: it trusts the motion test only while the head is still.

- ``observe`` runs on every frame with the head pose (yaw, pitch; ``track_stats.head_pose``).
  The head is still while the pose stays within ``head_still_deg`` of an anchor. A larger
  deviation is a move: the anchor follows the head, the reference copy is dropped and a run is
  owed once the head has been still for ``settle_s``.
- ``decide`` runs only on the ticks the governor allowed, so the gate can skip runs, never add
  one above ``objects_hz``. In order: no head pose (no joint state or no ``FrameSpec``) runs
  (``no_pose``, the fixed schedule); a live animal or object track, tentative ones included so
  a new thing gets its confirming hits, runs (``live_track``); ``max_interval_s`` since the last
  run, or no run yet, runs (``floor``); a head that moved less than ``settle_s`` ago skips
  (``head_moving``); the first tick after it settled runs (``settled``); otherwise the motion
  test runs (``motion``) or skips (``still_scene``).
- Motion test: a ``thumb_width_px`` x ``thumb_height_px`` grayscale copy, area pooled, against
  the copy taken at the last run (so a slow change adds up). The scene changed when more than
  ``changed_share`` of its pixels moved by more than ``pixel_diff``. A copy of another size (hot
  reload) counts as changed. Every run takes a fresh copy and anchors the head pose there.

Decisions use capture stamps only, so a replay gives the same decisions. The copies stay in RAM
and are never logged (SPEC 12); the debug report (SPEC 12A) gets counts only.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from ..config.schema import ObjectGateConfig
from .governor import EPSILON_S

Pose = tuple[float, float]
LUMA = (0.114, 0.587, 0.299)  # BGR weights of the Rec. 601 luma
SKIP_REASONS = ("still_scene", "head_moving")
RUN_REASONS = ("motion", "settled", "floor", "live_track", "no_pose")


def thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Grayscale ``height`` x ``width`` float32 copy of a BGR image, mean-pooled per cell
    (nearest sampling on an axis the image is smaller than), so sensor noise averages out."""
    gray = image.astype(np.float32) @ np.array(LUMA, np.float32)
    rows, cols = gray.shape
    if rows < height:
        gray = gray[np.linspace(0, rows - 1, height).astype(int)]
    else:
        ys = np.linspace(0, rows, height + 1).astype(int)
        gray = np.add.reduceat(gray, ys[:-1], axis=0) / np.diff(ys)[:, None]
    if cols < width:
        return gray[:, np.linspace(0, cols - 1, width).astype(int)].astype(np.float32)
    xs = np.linspace(0, cols, width + 1).astype(int)
    return (np.add.reduceat(gray, xs[:-1], axis=1) / np.diff(xs)[None, :]).astype(np.float32)


class ObjectGate:
    """Decides whether a governor-allowed object tick runs; see the module docstring.

    The vision thread calls ``observe`` and ``decide``; ``snapshot`` may run on another thread.
    ``cfg`` is replaced on hot reload and read at each call.
    """

    def __init__(self, cfg: ObjectGateConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._anchor: Pose | None = None
        self._last_move: float | None = None
        self._owed = False
        self._reference: np.ndarray | None = None
        self._last_run: float | None = None
        self._skipped = dict.fromkeys(SKIP_REASONS, 0)
        self._triggered = dict.fromkeys(RUN_REASONS, 0)

    def observe(self, stamp: float, pose: Pose | None) -> None:
        """Follow the head on every frame: a deviation past ``head_still_deg`` from the anchor
        is a move (drops the reference copy, owes a run once settled)."""
        if not self.cfg.enabled or pose is None:
            return
        if self._anchor is None:
            self._anchor = pose
            return
        limit = math.radians(self.cfg.head_still_deg)
        if max(abs(pose[0] - self._anchor[0]), abs(pose[1] - self._anchor[1])) > limit:
            self._anchor = pose
            self._last_move = stamp
            self._owed = True
            self._reference = None

    def decide(self, stamp: float, image: np.ndarray, pose: Pose | None, live_tracks: int) -> bool:
        """True when object detection should run on this governor-allowed tick. Counts the
        reason, and on a run takes a fresh reference copy anchored at ``pose``."""
        if not self.cfg.enabled:
            return True
        moving = self._last_move is not None and stamp - self._last_move < (
            self.cfg.settle_s - EPSILON_S
        )
        small: np.ndarray | None = None
        if pose is None:
            reason = "no_pose"
        elif live_tracks > 0:
            reason = "live_track"
        elif self._last_run is None or stamp - self._last_run >= (
            self.cfg.max_interval_s - EPSILON_S
        ):
            reason = "floor"
        elif moving:
            reason = "head_moving"
        elif self._owed:
            reason = "settled"
        else:
            small = self._thumbnail(image)
            reason = "motion" if self._changed(small) else "still_scene"
        run = reason in RUN_REASONS
        with self._lock:
            counts = self._triggered if run else self._skipped
            counts[reason] += 1
        if run:
            self._reference = self._thumbnail(image) if small is None else small
            self._last_run = stamp
            if pose is not None:
                self._anchor = pose
            self._owed = self._owed and moving
        return run

    def _thumbnail(self, image: np.ndarray) -> np.ndarray:
        return thumbnail(image, self.cfg.thumb_width_px, self.cfg.thumb_height_px)

    def _changed(self, small: np.ndarray) -> bool:
        """More than ``changed_share`` of the copy moved by more than ``pixel_diff``; a missing
        reference or one of another size counts as changed."""
        ref = self._reference
        if ref is None or ref.shape != small.shape:
            return True
        moved = np.abs(small - ref) > self.cfg.pixel_diff
        return float(moved.mean()) > self.cfg.changed_share

    def snapshot(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): skipped and triggered runs by reason."""
        with self._lock:
            return {
                "enabled": self.cfg.enabled,
                "skipped": dict(self._skipped),
                "triggered": dict(self._triggered),
            }

    def reset(self) -> None:
        """Forget the head, the reference copy and the counts (new session or replay start)."""
        with self._lock:
            self._anchor = self._last_move = self._reference = self._last_run = None
            self._owed = False
            self._skipped = dict.fromkeys(SKIP_REASONS, 0)
            self._triggered = dict.fromkeys(RUN_REASONS, 0)
