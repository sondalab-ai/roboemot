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
PNG_TYPE_AT = 12  # signature (8) + chunk length (4)
PNG_DATA_AT = 16
PNG_IHDR_END = 24
JPEG_MARKER_LEN = 2
JPEG_SEGMENT_HEAD = 4  # marker (2) + segment length (2)
JPEG_SOF_HEIGHT_AT = 5  # from the marker: marker (2), length (2), precision (1)
JPEG_SOF_END = 9
JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
JPEG_NO_LENGTH = frozenset({0x01, *range(0xD0, 0xDA)})
JPEG_SCAN_START = 0xDA
JPEG_MARKER = 0xFF
COLOR_NDIM = 3


class ImageError(ValueError):
    """The photo is refused; ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < PNG_IHDR_END or data[PNG_TYPE_AT:PNG_DATA_AT] != b"IHDR":
        raise ImageError("image_corrupt")
    width, height = struct.unpack(">II", data[PNG_DATA_AT:PNG_IHDR_END])
    return width, height


def _jpeg_size(data: bytes) -> tuple[int, int]:
    """Width and height from the first start-of-frame marker; refuses a file without one."""
    pos, end = JPEG_MARKER_LEN, len(data)
    while pos + JPEG_SEGMENT_HEAD <= end:
        if data[pos] != JPEG_MARKER:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == JPEG_MARKER:
            pos += 1
            continue
        if marker in JPEG_NO_LENGTH:
            pos += JPEG_MARKER_LEN
            continue
        if marker == JPEG_SCAN_START:
            break
        length = struct.unpack(">H", data[pos + JPEG_MARKER_LEN : pos + JPEG_SEGMENT_HEAD])[0]
        if marker in JPEG_SOF:
            if pos + JPEG_SOF_END > end:
                break
            height, width = struct.unpack(
                ">HH", data[pos + JPEG_SOF_HEIGHT_AT : pos + JPEG_SOF_END]
            )
            return width, height
        pos += JPEG_MARKER_LEN + max(length, JPEG_MARKER_LEN)
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
    if image is None or image.ndim != COLOR_NDIM or image.shape[0] * image.shape[1] > max_pixels:
        raise ImageError("image_corrupt")
    return image
