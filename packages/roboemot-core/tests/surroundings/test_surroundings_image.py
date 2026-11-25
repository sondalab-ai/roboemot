"""Picture shaping: faces masked, views tiled, JPEG (designs 2026-11-24, 5.4; 2026-11-25)."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.surroundings.image import MASK_BGR, encode_jpeg, mask_faces, shrink, tile


def noise(h: int = 100, w: int = 200) -> np.ndarray:
    return np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_every_face_box_is_filled_and_nothing_else_changes():
    img = noise()
    out = mask_faces(img, [(10, 10, 30, 40), (150, 50, 190, 90)])
    assert (out[10:40, 10:30] == MASK_BGR).all()
    assert (out[50:90, 150:190] == MASK_BGR).all()
    keep = np.ones(img.shape[:2], bool)
    keep[10:40, 10:30] = keep[50:90, 150:190] = False
    assert np.array_equal(out[keep], img[keep])
    assert not (img[10:40, 10:30] == MASK_BGR).all()  # the input is not changed


def test_shrink_scales_boxes_and_never_upscales():
    img = noise(100, 200)
    small, boxes = shrink(img, [(20, 10, 60, 50)], 100)
    assert small.shape[:2] == (50, 100)
    assert boxes == [(10.0, 5.0, 30.0, 25.0)]
    same, kept = shrink(img, [(20, 10, 60, 50)], 400)
    assert same.shape == img.shape and kept == [(20, 10, 60, 50)]


def test_jpeg_round_trips_size():
    cv2 = pytest.importorskip("cv2")
    data = encode_jpeg(noise(), 80)
    assert data[:2] == b"\xff\xd8"
    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == (100, 200, 3)


def solid(value: int, h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def test_tile_joins_views_left_to_right_at_one_height_with_a_separator():
    out = tile([solid(10, 60, 80), solid(20, 60, 80), solid(30, 60, 80)], 4, 30)
    assert out.shape == (30, 40 * 3 + 4 * 2, 3)
    assert (out[:, :40] == 10).all() and (out[:, 44:84] == 20).all() and (out[:, 88:] == 30).all()
    assert (out[:, 40:44] == 0).all() and (out[:, 84:88] == 0).all()


def test_tile_never_upscales_and_keeps_each_aspect():
    out = tile([solid(10, 40, 80), solid(20, 60, 60)], 2, 100)
    assert out.shape[0] == 40  # the smallest view's height
    assert out.shape[1] == 80 + 2 + 40
