"""SPEC 11A: ``VisionPipeline.face_overlay`` and ``object_overlay`` for the live preview: the
visible face tracks of the last processed frame as (pixel box, owner id or None), and the
confirmed animal and object tracks of the last object run as (pixel box, readable category,
owner id or None)."""

from __future__ import annotations

import numpy as np
from roboemot.vision.types import Detection, Frame, PreviewOverlay
from test_vision import (
    CIRCADIAN,
    CONSENT,
    FACE_MODEL,
    UNGATED,
    ObjectMock,
    axis,
    face,
    image,
    rig,
    run,
)


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
    assert detected.faces_detected
    assert pipe.preview_overlay(detected) == PreviewOverlay(tuple(pipe.face_overlay()), ())
    skipped = pipe.process(Frame(image(8), 10.001))  # inside the detection period
    assert not skipped.faces_detected and pipe.preview_overlay(skipped) is None


def test_a_resting_pipeline_previews_frames_without_boxes(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    pipe.governor.set_scale(CIRCADIAN, 0.0)  # no detection at all: the picture still shows
    assert pipe.preview_overlay(pipe.process(Frame(image(7), 10.0))) == PreviewOverlay()


PLANT = Detection((200.0, 100.0, 260.0, 160.0), 0.8, "potted_plant")


def object_rig(tmp_path, dets):
    cfg = UNGATED.model_copy(update={"object_categories": ["cup", "potted_plant"]})
    return rig(tmp_path, lambda i: [face(40, y=10, size=50)], vision=cfg,
               object_detector=ObjectMock(dets))  # fmt: skip


def test_a_confirmed_object_track_gives_its_box_and_readable_category(tmp_path):
    _, _, pipe, _, got = object_rig(tmp_path, [PLANT])
    run(pipe, 40)
    [(box, category, owner)] = pipe.object_overlay()
    seen = {m.payload.category for m in got["percept.entity"] if m.payload.kind == "object"}
    assert category == "potted plant" and seen == {"potted plant"}
    assert owner is None and 220 < (box[0] + box[2]) / 2 < 240


def test_the_preview_overlay_carries_the_object_boxes_with_the_faces(tmp_path):
    _, _, pipe, _, _ = object_rig(tmp_path, [PLANT])
    run(pipe, 40)
    overlays = []
    for i in range(8):  # what the runner's hook does: right after each frame
        overlays.append(pipe.preview_overlay(pipe.process(Frame(image(40 + i), 5.0 + i * 0.125))))
    overlay = next(o for o in overlays if o is not None)
    assert overlay.faces and overlay.objects
    assert overlay.objects == tuple(pipe.object_overlay())


def test_an_expired_object_track_leaves_the_overlay(tmp_path):
    _, _, pipe, _, _ = object_rig(tmp_path, [PLANT])
    run(pipe, 40)
    assert pipe.object_overlay()
    pipe.object_detector.dets = []
    run(pipe, 120, dt=0.5, start=10.0)  # long past the object track lifetime
    assert pipe.object_overlay() == []


def test_clear_overlay_and_reset_forget_the_object_tracks(tmp_path):
    _, _, pipe, _, _ = object_rig(tmp_path, [PLANT])
    run(pipe, 40)
    pipe.clear_overlay()
    assert pipe.object_overlay() == []
    run(pipe, 40, start=10.0)
    assert pipe.object_overlay()
    pipe.reset()
    assert pipe.object_overlay() == []
