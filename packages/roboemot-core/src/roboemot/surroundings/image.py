"""Picture shaping (SPEC 8, Surroundings snapshots; designs 2026-11-24, 5.4; 2026-11-25).

On the vision thread each frame of a view is ``shrink``-ed to the sent size (boxes scaled with
it). On the service thread each view's chosen frame gets ``mask_faces`` with that view's face
boxes, the views are joined by ``tile`` and ``encode_jpeg`` runs once. Every function returns a
new array (``shrink`` may return its input unchanged): the frame is shared with the vision
pipeline and is never written. Nothing here logs an image (SPEC 12).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..vision.types import Box

MASK_BGR = (128, 128, 128)  # flat grey over a detected face
SEPARATOR_BGR = (0, 0, 0)  # the line between two views of the joined picture


def opencv_available() -> bool:
    """Whether OpenCV (the vision extra) can be imported: no JPEG without it."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


class NoEncoder(RuntimeError):
    """OpenCV (the vision extra) is not installed: no JPEG can be made."""


def _pixels(box: Box, height: int, width: int) -> tuple[int, int, int, int]:
    """``box`` as whole pixels, outward, clipped to the image."""
    x1, y1, x2, y2 = box
    return (
        min(max(math.floor(x1), 0), width),
        min(max(math.floor(y1), 0), height),
        min(max(math.ceil(x2), 0), width),
        min(max(math.ceil(y2), 0), height),
    )


def shrink(
    image: np.ndarray, boxes: Sequence[Box], max_side_px: int
) -> tuple[np.ndarray, list[Box]]:
    """A copy of ``image`` whose longest side is at most ``max_side_px``, and ``boxes`` scaled
    with it. An image already small enough comes back as is (never upscaled)."""
    height, width = image.shape[:2]
    if max(height, width) <= max_side_px:
        return image, list(boxes)
    scale = max_side_px / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    try:
        import cv2

        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    except ImportError:
        step = math.ceil(1 / scale)
        small = np.ascontiguousarray(image[::step, ::step])
    sy, sx = small.shape[0] / height, small.shape[1] / width
    return small, [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in boxes]


def mask_faces(image: np.ndarray, faces: Sequence[Box]) -> np.ndarray:
    """A copy of ``image`` with each face box, clipped to the image, filled with ``MASK_BGR``."""
    out = image.copy()
    height, width = out.shape[:2]
    for box in faces:
        x1, y1, x2, y2 = _pixels(box, height, width)
        out[y1:y2, x1:x2] = MASK_BGR
    return out


def _to_height(image: np.ndarray, height: int) -> np.ndarray:
    """``image`` scaled to ``height`` rows, its aspect kept (as is when already that high)."""
    rows, cols = image.shape[:2]
    if rows == height:
        return image
    width = max(1, round(cols * height / rows))
    try:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        ys = np.arange(height) * rows // height
        xs = np.arange(width) * cols // width
        return np.ascontiguousarray(image[ys][:, xs])


def tile(views: Sequence[np.ndarray], separator_px: int, height_px: int) -> np.ndarray:
    """One picture of ``views`` (BGR, at least one) joined left to right in the given order,
    each scaled to one height, the smallest view's height capped at ``height_px`` (never
    upscaled), with a ``separator_px`` wide ``SEPARATOR_BGR`` column between two views."""
    height = min(height_px, *(v.shape[0] for v in views))
    line = np.empty((height, separator_px, len(SEPARATOR_BGR)), np.uint8)
    line[:] = SEPARATOR_BGR
    parts: list[np.ndarray] = []
    for view in views:
        if parts:
            parts.append(line)
        parts.append(_to_height(view, height))
    return np.concatenate(parts, axis=1)


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    """``image`` (BGR) as JPEG bytes at ``quality``. Raises ``NoEncoder`` without OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise NoEncoder("opencv is not installed") from exc
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise NoEncoder("jpeg encode failed")
    return data.tobytes()
