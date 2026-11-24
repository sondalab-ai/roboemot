"""The vision pipeline's snapshot tap (SPEC 8, Surroundings snapshots): called with the frame,
the fresh face boxes and the shown track's box, only while a snapshot is wanted."""

from __future__ import annotations

import logging

from roboemot.vision import Frame
from test_vision_object_gate import gated, joint, scene


def test_tap_gets_frame_faces_and_shown_box_when_wanted(tmp_path):
    pipe, _, _ = gated(tmp_path)
    got = []
    pipe.snapshot_tap = lambda frame, faces, shown: got.append((frame, faces, shown))
    pipe.snapshot_wanted = lambda: True
    frame = Frame(scene(), 0.0, joint())
    pipe.process(frame)
    assert len(got) == 1
    seen, faces, shown = got[0]
    assert seen is frame and faces == pipe._fresh_face_boxes(0.0) and shown is None


def test_tap_is_not_called_when_not_wanted(tmp_path):
    pipe, _, _ = gated(tmp_path)
    got = []
    pipe.snapshot_tap = lambda *a: got.append(a)
    pipe.process(Frame(scene(), 0.0, joint()))
    assert got == []


def test_the_shown_box_is_the_followed_track_box(tmp_path):
    pipe, _, _ = gated(tmp_path)

    class Track:
        track_id, box = 7, (10.0, 20.0, 30.0, 40.0)

    pipe._object_visible = [Track()]
    pipe.follower._target = 7
    assert pipe._shown_box() == (10.0, 20.0, 30.0, 40.0)
    pipe.follower._target = 8
    assert pipe._shown_box() is None


def test_a_failing_tap_does_not_stop_the_pipeline_and_logs_no_data(tmp_path, caplog):
    pipe, _, _ = gated(tmp_path)

    def boom(frame, faces, shown):
        raise ValueError("secret pixel data")

    pipe.snapshot_tap = boom
    pipe.snapshot_wanted = lambda: True
    with caplog.at_level(logging.WARNING):
        pipe.process(Frame(scene(), 0.0, joint()))
        pipe.process(Frame(scene(), 0.125, joint()))
    assert "snapshot tap failed: ValueError" in caplog.text
    assert "secret pixel data" not in caplog.text


# ---- every face the robot detected is masked (review C1)

import numpy as np  # noqa: E402
from roboemot.vision import Detection  # noqa: E402
from test_vision import rig  # noqa: E402

FACE = (100.0, 100.0, 160.0, 160.0)


def faces_seen(tmp_path, script, frames):
    _, _, pipe, _, _ = rig(tmp_path, script)
    got = []
    pipe.snapshot_tap = lambda frame, faces, shown: got.append(faces)
    pipe.snapshot_wanted = lambda: True
    for i in range(frames):
        img = np.zeros((480, 640, 3), np.uint8)
        img[0, 0, 0] = i
        pipe.process(Frame(img, i * 0.25, None))
    return got


def covers(boxes, box, slack=8.0):
    return any(all(abs(a - b) <= slack for a, b in zip(f, box, strict=True)) for f in boxes)


def test_a_first_unconfirmed_face_is_masked(tmp_path):
    got = faces_seen(tmp_path, lambda i: [Detection(FACE, 0.95)], 1)
    assert covers(got[0], FACE)


def test_a_face_the_last_run_missed_stays_masked(tmp_path):
    got = faces_seen(tmp_path, lambda i: [Detection(FACE, 0.95)] if i < 3 else [], 4)
    assert covers(got[3], FACE)


def test_a_face_below_the_tracking_size_is_masked(tmp_path):
    small = (300.0, 300.0, 310.0, 310.0)
    got = faces_seen(tmp_path, lambda i: [Detection(small, 0.95)], 1)
    assert covers(got[0], small)


# ---- the shown box without the single-object follower (review I4)


def test_the_shown_box_falls_back_to_the_detector_run_shown_track(tmp_path):
    pipe, _, _ = gated(tmp_path)

    class Track:
        track_id, box = 7, (10.0, 20.0, 30.0, 40.0)

    pipe._object_visible = [Track()]
    pipe.follower._target = None
    pipe._shown_detected = 7
    assert pipe._shown_box() == (10.0, 20.0, 30.0, 40.0)
