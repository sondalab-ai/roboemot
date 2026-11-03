"""Scene-change gate of the object stage (SPEC 5, 13): run object detection only when the
scene changed, to save CPU.

The camera sits on the head, so a head move changes the image without any change in the scene.
The gate does not compensate for it: it trusts the motion test only while the head is still.

- ``observe`` runs on every frame with the head pose (yaw, pitch; ``track_stats.head_pose``).
  The head is still while the pose stays within ``head_still_deg`` of an anchor. A larger
  deviation is a move: the anchor follows the head, the reference copy is dropped and a run is
  owed once the head has been still for ``settle_s``. Frames without a pose are skipped on
  purpose: they say nothing about the head, and ``decide`` handles the tick of such a frame.
  The pose is a sum of joint angles, not wrapped: differences are exact while the joints stay
  within +-pi, which head joints do.
- ``decide`` runs only on the ticks the governor allowed, so the gate can skip runs, never add
  one above ``objects_hz``; a skipped tick still used the governor's slot. In order: no head
  pose (no joint state or no ``FrameSpec``) runs (``no_pose``, the fixed schedule); a look at
  a shown object (``episode``: the governor's object stage is active, SPEC 5A) runs, before the
  head-moving skip, because the head moves on purpose while it follows the thing; a live
  track that needs a steady detector (``Tracker.holds``: an animal, a tentative track, so a new
  thing gets its confirming hits, or a recently moved one) runs (``live_track``, counted per
  hold in ``live_track_by``; confirmed still objects do not hold); ``max_interval_s`` since the
  last run, or no run yet, runs (``floor``, also while the head keeps moving); a head that moved
  less than ``settle_s`` ago skips (``head_moving``); the first tick after it settled runs
  (``settled``); otherwise the motion test runs (``motion``) or skips (``still_scene``).
- Motion test: a ``thumb_width_px`` x ``thumb_height_px`` grayscale copy against the copy taken
  at the last run (so a slow change adds up). The scene changed when more than
  ``changed_share`` of its pixels moved by more than ``pixel_diff``. The copy is cut from the
  uint8 frame by strided slicing first (``SAMPLES_PER_CELL`` per cell and axis) and area-pooled
  in float only at that size; numpy only, so a replay gives the same copy with or without cv2.
  Gray (HxW), BGR and BGRA frames are accepted; another shape, or a copy of another size (hot
  reload), counts as changed, so the gate never raises per frame. A run with a pose takes a
  fresh copy and anchors the head pose there; a ``no_pose`` run takes none and drops it.

Decisions use capture stamps only, so a replay gives the same decisions. One lock guards all
state, so ``reset`` and ``snapshot`` may run on another thread. The copies stay in RAM and are
never logged (SPEC 12); the debug report (SPEC 12A) gets counts only.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Collection
from typing import Any

import numpy as np

from ..config.schema import ObjectGateConfig
from .governor import EPSILON_S
from .tracker import HOLD_REASONS

Pose = tuple[float, float]
LUMA = (0.114, 0.587, 0.299)  # BGR weights of the Rec. 601 luma
SAMPLES_PER_CELL = 2  # strided samples per copy cell and axis before area pooling
GRAY_DIMS = 2  # HxW
COLOUR_DIMS = 3  # HxWxC
COLOUR_CHANNELS = (3, 4)  # BGR, BGRA (alpha ignored)
SKIP_REASONS = ("still_scene", "head_moving")
RUN_REASONS = ("motion", "settled", "floor", "live_track", "no_pose", "episode")


def _pool(gray: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Mean-pool ``gray`` to ``size`` along ``axis`` (nearest sampling when it is shorter)."""
    n = gray.shape[axis]
    if n < size:
        return np.take(gray, np.linspace(0, n - 1, size).astype(int), axis=axis)
    edges = np.linspace(0, n, size + 1).astype(int)
    counts = np.diff(edges).reshape((-1, 1) if axis == 0 else (1, -1))
    return np.add.reduceat(gray, edges[:-1], axis=axis) / counts


def thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Grayscale ``height`` x ``width`` float32 copy of a gray, BGR or BGRA uint8 frame, or None
    for another shape. Strided slicing cuts the frame to about ``SAMPLES_PER_CELL`` samples per
    cell first, so only that small image is converted to float and pooled."""
    if image.ndim == COLOUR_DIMS and image.shape[2] in COLOUR_CHANNELS:
        colour = True
    elif image.ndim == GRAY_DIMS:
        colour = False
    else:
        return None
    rows, cols = image.shape[:2]
    if rows == 0 or cols == 0:
        return None
    step_y = max(1, rows // (SAMPLES_PER_CELL * height))
    step_x = max(1, cols // (SAMPLES_PER_CELL * width))
    small = image[::step_y, ::step_x]
    if colour:
        gray = small[..., : len(LUMA)].astype(np.float32) @ np.array(LUMA, np.float32)
    else:
        gray = small.astype(np.float32)
    return _pool(_pool(gray, height, 0), width, 1).astype(np.float32)


class ObjectGate:
    """Decides whether a governor-allowed object tick runs; see the module docstring.

    ``cfg`` is replaced on hot reload and read at each call.
    """

    def __init__(self, cfg: ObjectGateConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._clear()

    def _clear(self) -> None:
        self._anchor: Pose | None = None
        self._last_move: float | None = None
        self._owed = False
        self._reference: np.ndarray | None = None
        self._last_run: float | None = None
        self._skipped = dict.fromkeys(SKIP_REASONS, 0)
        self._triggered = dict.fromkeys(RUN_REASONS, 0)
        self._held_by = dict.fromkeys(HOLD_REASONS, 0)

    def observe(self, stamp: float, pose: Pose | None) -> None:
        """Follow the head on every frame: a deviation past ``head_still_deg`` from the anchor
        is a move (drops the reference copy, owes a run once settled)."""
        if not self.cfg.enabled or pose is None:
            return
        limit = math.radians(self.cfg.head_still_deg)
        with self._lock:
            if self._anchor is None:
                self._anchor = pose
                return
            if max(abs(pose[0] - self._anchor[0]), abs(pose[1] - self._anchor[1])) > limit:
                self._anchor = pose
                self._last_move = stamp
                self._owed = True
                self._reference = None

    def decide(
        self,
        stamp: float,
        image: np.ndarray,
        pose: Pose | None,
        holds: Collection[str],
        *,
        episode: bool = False,
    ) -> bool:
        """True when object detection should run on this governor-allowed tick. ``holds`` are
        the reasons live tracks need a steady detector (``Tracker.holds``); ``episode``: a
        shown object is the attention target, so every allowed tick runs. Counts the reason,
        and on a run with a pose takes a fresh reference copy anchored at ``pose``."""
        if not self.cfg.enabled:
            return True
        with self._lock:
            moving = self._last_move is not None and stamp - self._last_move < (
                self.cfg.settle_s - EPSILON_S
            )
            small: np.ndarray | None = None
            if pose is None:
                reason = "no_pose"
            elif episode:
                reason = "episode"
            elif holds:
                reason = "live_track"
                for hold in holds:
                    self._held_by[hold] = self._held_by.get(hold, 0) + 1
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
            (self._triggered if run else self._skipped)[reason] += 1
            if run:
                self._last_run = stamp
                self._owed = self._owed and moving
                if pose is None:
                    self._reference = None
                else:
                    self._reference = self._thumbnail(image) if small is None else small
                    self._anchor = pose
            return run

    def _thumbnail(self, image: np.ndarray) -> np.ndarray | None:
        return thumbnail(image, self.cfg.thumb_width_px, self.cfg.thumb_height_px)

    def _changed(self, small: np.ndarray | None) -> bool:
        """More than ``changed_share`` of the copy moved by more than ``pixel_diff``; no copy, no
        reference or one of another size counts as changed."""
        ref = self._reference
        if small is None or ref is None or ref.shape != small.shape:
            return True
        moved = np.abs(small - ref) > self.cfg.pixel_diff
        return float(moved.mean()) > self.cfg.changed_share

    def snapshot(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): skipped and triggered runs by reason, and the
        ``live_track`` runs by the hold behind them (one run may count under several)."""
        with self._lock:
            return {
                "enabled": self.cfg.enabled,
                "skipped": dict(self._skipped),
                "triggered": dict(self._triggered),
                "live_track_by": dict(self._held_by),
            }

    def reset(self) -> None:
        """Forget the head, the reference copy and the counts (new session or replay start)."""
        with self._lock:
            self._clear()
