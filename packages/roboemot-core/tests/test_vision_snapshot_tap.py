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
