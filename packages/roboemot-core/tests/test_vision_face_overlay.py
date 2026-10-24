"""SPEC 11A: ``VisionPipeline.face_overlay`` for the live preview: the visible tracks of the last
processed frame as (pixel box, owner id or None)."""

from __future__ import annotations

import numpy as np
from roboemot.vision.types import Frame
from test_vision import CIRCADIAN, CONSENT, FACE_MODEL, axis, face, image, rig, run


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
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    run(pipe, 24)
    [(_, owner)] = pipe.face_overlay()
    assert owner is not None


def test_clear_overlay_forgets_the_tracks(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    assert pipe.face_overlay()
    pipe.clear_overlay()
    assert pipe.face_overlay() == []


def test_the_preview_overlay_is_only_for_frames_faces_were_detected_on(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    detected = pipe.process(Frame(image(7), 10.0))
    assert detected.faces_detected and pipe.preview_overlay(detected) == pipe.face_overlay()
    skipped = pipe.process(Frame(image(8), 10.001))  # inside the detection period
    assert not skipped.faces_detected and pipe.preview_overlay(skipped) is None


def test_a_resting_pipeline_previews_frames_without_boxes(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    pipe.governor.set_scale(CIRCADIAN, 0.0)  # no detection at all: the picture still shows
    assert pipe.preview_overlay(pipe.process(Frame(image(7), 10.0))) == []
