"""The scene-change test shared by the object gate and the nap's motion watch (SPEC 5, 7A, 13).

The camera sits on the head, so a head move changes the image without any change in the scene.
Neither user compensates for it: both trust the test only while the head is still.

- ``thumbnail``: a small grayscale copy of a frame. It is cut from the uint8 frame by strided
  slicing first (``SAMPLES_PER_CELL`` per cell and axis) and area-pooled in float only at that
  size; numpy only, so a replay gives the same copy with or without cv2. Gray (HxW), BGR and BGRA
  frames are accepted; another shape gives None.
- ``moved_share``: the share of a copy's pixels that moved by more than a pixel difference
  against a reference copy of the same size.
- ``HeadStill``: follows the head pose (yaw, pitch; ``track_stats.head_pose``). The head is still
  while the pose stays within a limit of an anchor; a larger deviation is a move, the anchor
  follows the head and the move's stamp is kept, so a caller can wait for a settle time. The pose
  is a sum of joint angles, not wrapped: differences are exact while the joints stay within
  +-pi, which head joints do.

The copies stay in RAM and are never logged (SPEC 12).
"""

from __future__ import annotations

import numpy as np

from .governor import EPSILON_S

Pose = tuple[float, float]
LUMA = (0.114, 0.587, 0.299)  # BGR weights of the Rec. 601 luma
SAMPLES_PER_CELL = 2  # strided samples per copy cell and axis before area pooling
GRAY_DIMS = 2  # HxW
COLOUR_DIMS = 3  # HxWxC
COLOUR_CHANNELS = (3, 4)  # BGR, BGRA (alpha ignored)


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


def moved_share(small: np.ndarray, reference: np.ndarray, pixel_diff: float) -> float:
    """Share of the pixels of ``small`` that moved by more than ``pixel_diff`` against
    ``reference`` (same shape; the caller checks it)."""
    return float((np.abs(small - reference) > pixel_diff).mean())


class HeadStill:
    """The head's anchor and its last move; see the module docstring. Not thread-safe: the
    caller holds its own lock."""

    def __init__(self) -> None:
        self.anchor: Pose | None = None
        self.last_move: float | None = None

    def observe(self, stamp: float, pose: Pose, limit_rad: float) -> bool:
        """Follow the head at ``stamp``: True when ``pose`` left the anchor by more than
        ``limit_rad`` (a move: the anchor follows the head). The first pose only anchors."""
        if self.anchor is None:
            self.anchor = pose
            return False
        if max(abs(pose[0] - self.anchor[0]), abs(pose[1] - self.anchor[1])) > limit_rad:
            self.anchor = pose
            self.last_move = stamp
            return True
        return False

    def anchor_to(self, pose: Pose) -> None:
        """Take ``pose`` as the still head (a fresh reference view was taken there)."""
        self.anchor = pose

    def moving(self, stamp: float, settle_s: float) -> bool:
        """Whether the last move is less than ``settle_s`` before ``stamp`` (capture time)."""
        return self.last_move is not None and stamp - self.last_move < settle_s - EPSILON_S
