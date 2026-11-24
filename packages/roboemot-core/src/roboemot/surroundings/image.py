"""Snapshot shaping (SPEC 8, Surroundings snapshots; design 2026-11-24, 5.4).

Order on the vision thread: ``shrink`` the frame to the sent size (boxes scaled with it),
``mask_faces`` with every detected face box, ``crop`` around a shown thing. ``encode_jpeg``
runs once, on the chosen frame. Every function returns a new array: the frame is shared with the
vision pipeline and is never written. Nothing here logs an image (SPEC 12).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..vision.types import Box

MASK_BGR = (128, 128, 128)  # flat grey over a detected face


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


def crop(image: np.ndarray, box: Box, margin: float) -> np.ndarray:
    """A copy of ``box`` grown by ``margin`` of its width and height on each side, clipped to
    the image."""
    x1, y1, x2, y2 = box
    gx, gy = (x2 - x1) * margin, (y2 - y1) * margin
    height, width = image.shape[:2]
    a, b, c, d = _pixels((x1 - gx, y1 - gy, x2 + gx, y2 + gy), height, width)
    return image[b:d, a:c].copy()


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
