"""Whether a face looks at the camera: a head-yaw proxy from the 5 detector landmarks (SPEC 8).

No model of its own and near-zero CPU: the face detector already gives the eyes, the nose tip
and the mouth corners. A rigid head seen from the front has the nose tip on the eye midpoint;
turned by ``yaw`` its horizontal offset from that midpoint, measured along the eye axis (so a
tilted head reads the same) and divided by the eye span, is ``NOSE_DEPTH_PER_EYE_SPAN *
tan(yaw)``. The yaw is relative to the line from the camera to the face, so a face anywhere in
the image that looks into the lens reads about 0. Sign: positive when the face turns toward the
robot's left (the azimuth convention), i.e. the nose moves toward the left of the image.

The vision pipeline keeps a short history per face track and publishes the median over
``vision.facing_window_s`` (``smoothed``). Landmarks never leave this module; only the angle
does (SPEC 12).
"""

from __future__ import annotations

import math
import statistics
from collections import deque

import numpy as np

RIGHT_EYE, LEFT_EYE, NOSE = 0, 1, 2  # YuNet landmark rows (the person's right eye first)
LANDMARKS = 5
# Nose tip depth in front of the eye centres over the distance between the eyes: about 27 mm
# over 63 mm for an adult (the common 3D face model used for head pose from 2D landmarks).
NOSE_DEPTH_PER_EYE_SPAN = 0.43
MIN_EYE_SPAN_PX = 1.0  # eyes closer than this on the image give no estimate (degenerate)


def head_yaw(landmarks: np.ndarray | None) -> float | None:
    """Head yaw (radians, positive toward the robot's left) from 5x2 landmarks in pixels; None
    when there are no landmarks or they are degenerate (eyes on top of each other, not finite)."""
    if landmarks is None:
        return None
    pts = np.asarray(landmarks, dtype=float)
    if pts.shape != (LANDMARKS, 2) or not np.all(np.isfinite(pts)):
        return None
    axis = pts[LEFT_EYE] - pts[RIGHT_EYE]  # points to the image right on an upright face
    span = float(np.hypot(*axis))
    if span < MIN_EYE_SPAN_PX:
        return None
    offset = float(np.dot(pts[NOSE] - (pts[LEFT_EYE] + pts[RIGHT_EYE]) / 2, axis)) / span**2
    return -math.atan(offset / NOSE_DEPTH_PER_EYE_SPAN)


def smoothed(history: deque[tuple[float, float]], stamp: float, window_s: float) -> float | None:
    """The median yaw of ``history`` (``(stamp, yaw)`` pairs, oldest first) over the last
    ``window_s`` before ``stamp``; older pairs are dropped. None when none is left."""
    while history and history[0][0] < stamp - window_s:
        history.popleft()
    return statistics.median(y for _, y in history) if history else None
