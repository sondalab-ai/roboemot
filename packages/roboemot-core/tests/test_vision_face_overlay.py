"""SPEC 11A: ``VisionPipeline.face_overlay`` for the live preview: the visible tracks of the last
processed frame as (pixel box, owner id or None)."""

from __future__ import annotations

import numpy as np
from test_vision import CONSENT, FACE_MODEL, axis, face, rig, run


def test_no_faces_gives_an_empty_overlay(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [])
    run(pipe, 4)
    assert pipe.face_overlay() == []


def test_a_confirmed_track_gives_its_box_and_no_owner_until_recognised(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    [(box, owner)] = pipe.face_overlay()
    assert owner is None
    assert 240 < (box[0] + box[2]) / 2 < 290 and box[2] - box[0] > 40


def test_a_recognised_track_carries_the_owner_id(tmp_path):
    reg, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)])
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult")
    run(pipe, 24)
    [(_, owner)] = pipe.face_overlay()
    assert owner is not None


def test_clear_overlay_forgets_the_tracks(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    assert pipe.face_overlay()
    pipe.clear_overlay()
    assert pipe.face_overlay() == []
