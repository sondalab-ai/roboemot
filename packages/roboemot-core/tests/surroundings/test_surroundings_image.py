"""Snapshot shaping: faces masked, crop, JPEG (design 2026-11-24, 5.4)."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.surroundings.image import MASK_BGR, crop, encode_jpeg, mask_faces, shrink


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


def test_a_face_overlapping_the_thing_is_masked_inside_the_crop():
    img = noise()
    face, thing = (60, 10, 100, 50), (50, 30, 150, 90)
    out = crop(mask_faces(img, [face]), thing, 0.0)
    assert out.shape[:2] == (60, 100)  # the whole thing box is kept
    assert (out[0:20, 10:50] == MASK_BGR).all()  # the face part inside the crop


def test_crop_is_clipped_to_the_image():
    out = crop(noise(), (180, 80, 220, 120), 0.5)
    assert out.shape[:2] == (100 - 60, 200 - 160)


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
