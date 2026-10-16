"""Safe decoding of catalogue photos (SPEC 6 Catalogue import, 12 Data).

A photo arrives as bytes from an authenticated upload, is decoded in memory and is never
written to disk or logged. Only JPEG and PNG are accepted, recognised by their magic bytes
(the declared type is not trusted). The pixel count is read from the file header and checked
*before* the decoder runs, so a small file that expands to gigabytes (decompression bomb) is
refused without allocating anything. Decoding uses OpenCV (the ``vision`` extra); EXIF
orientation is applied (OpenCV's default), which matches MWG region coordinates: they refer to
the upright image.
"""

from __future__ import annotations

import struct

import numpy as np

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_IHDR_END = 24
JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
JPEG_NO_LENGTH = frozenset({0x01, *range(0xD0, 0xDA)})
JPEG_SCAN_START = 0xDA
JPEG_MARKER = 0xFF


class ImageError(ValueError):
    """The photo is refused; ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < PNG_IHDR_END or data[12:16] != b"IHDR":
        raise ImageError("image_corrupt")
    width, height = struct.unpack(">II", data[16:PNG_IHDR_END])
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Width and height from the first start-of-frame marker; refuses a file without one."""
    pos, end = 2, len(data)
    while pos + 4 <= end:
        if data[pos] != JPEG_MARKER:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == JPEG_MARKER:
            pos += 1
            continue
        if marker in JPEG_NO_LENGTH:
            pos += 2
            continue
        if marker == JPEG_SCAN_START:
            break
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if marker in JPEG_SOF:
            if pos + 9 > end:
                break
            height, width = struct.unpack(">HH", data[pos + 5 : pos + 9])
            return width, height
        pos += 2 + max(length, 2)
    raise ImageError("image_corrupt")


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the header of a JPEG or PNG; ``ImageError`` for anything else."""
    if data.startswith(PNG_MAGIC):
        return _png_size(data)
    if data.startswith(JPEG_MAGIC):
        return _jpeg_size(data)
    raise ImageError("image_format")


def decode_bgr(data: bytes, *, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Decode a JPEG or PNG to an HxWx3 uint8 BGR array, within the byte and pixel limits."""
    if len(data) > max_bytes:
        raise ImageError("image_too_large")
    width, height = image_size(data)
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ImageError("image_too_many_pixels")
    try:
        import cv2
    except ImportError as exc:
        raise ImageError("image_unsupported") from exc
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageError("image_corrupt") from exc
    if image is None or image.ndim != 3 or image.shape[0] * image.shape[1] > max_pixels:
        raise ImageError("image_corrupt")
    return image
