"""Reachy camera source against a fake of the SDK media interface (SPEC 4A, 5)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from roboemot_reachymini.camera import ReachyCamera


def fake_mini(frame, k=None):
    media = SimpleNamespace(get_frame=lambda: frame, camera=SimpleNamespace(K=k))
    return SimpleNamespace(media=media)


def test_read_stamps_frame_and_attaches_joint_state():
    img = np.zeros((4, 6, 3), np.uint8)
    cam = ReachyCamera(fake_mini(img), lambda: {"head_yaw": 0.2}, clock=lambda: 12.5)
    frame = cam.read()
    assert frame.stamp == 12.5 and frame.joint_state == {"head_yaw": 0.2}
    assert frame.image.shape == (4, 6, 3)


def test_read_returns_none_without_a_frame():
    assert ReachyCamera(fake_mini(None)).read() is None


def test_camera_model_from_intrinsics():
    k = np.array([[400.0, 0, 320], [0, 410.0, 240], [0, 0, 1]])
    cam = ReachyCamera(fake_mini(None, k))
    model = cam.camera_model(640, 480)
    assert (model.fx, model.fy, model.cx, model.cy) == (400.0, 410.0, 320.0, 240.0)
    assert model.hfov_deg == math.degrees(2 * math.atan(320 / 400))
    assert model.bearing(320, 240) == (0.0, 0.0)
    assert cam.camera_model(100, 100) is None
    assert ReachyCamera(fake_mini(None, None)).camera_model(640, 480) is None
