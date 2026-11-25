"""The vision pipeline's picture tap (SPEC 8, Surroundings snapshots; design 2026-11-25): called
with the frame and the boxes of its own face run (None when none ran on it), only while a
picture is wanted."""

from __future__ import annotations

import logging

import numpy as np
from roboemot.vision import Detection, Frame
from test_vision import rig
from test_vision_object_gate import gated, joint, scene

FACE = (100.0, 100.0, 160.0, 160.0)


def test_tap_gets_the_frame_and_its_own_faces_when_wanted(tmp_path):
    pipe, _, _ = gated(tmp_path)
    got = []
    pipe.snapshot_tap = lambda frame, faces: got.append((frame, faces))
    pipe.snapshot_wanted = lambda: True
    frame = Frame(scene(), 0.0, joint())
    pipe.process(frame)
    assert len(got) == 1
    seen, faces = got[0]
    assert seen is frame and faces == pipe._snapshot_faces(0.0)


def test_tap_is_not_called_when_not_wanted(tmp_path):
    pipe, _, _ = gated(tmp_path)
    got = []
    pipe.snapshot_tap = lambda *a: got.append(a)
    pipe.process(Frame(scene(), 0.0, joint()))
    assert got == []


def test_a_failing_tap_does_not_stop_the_pipeline_and_logs_no_data(tmp_path, caplog):
    pipe, _, _ = gated(tmp_path)

    def boom(frame, faces):
        raise ValueError("secret pixel data")

    pipe.snapshot_tap = boom
    pipe.snapshot_wanted = lambda: True
    with caplog.at_level(logging.WARNING):
        pipe.process(Frame(scene(), 0.0, joint()))
        pipe.process(Frame(scene(), 0.125, joint()))
    assert "snapshot tap failed: ValueError" in caplog.text
    assert "secret pixel data" not in caplog.text


# ---- every face of the frame's own run is masked (review C1), never one of another frame


def faces_seen(tmp_path, script, frames, dt=0.25):
    _, _, pipe, _, _ = rig(tmp_path, script)
    got = []
    pipe.snapshot_tap = lambda frame, faces: got.append(faces)
    pipe.snapshot_wanted = lambda: True
    for i in range(frames):
        img = np.zeros((480, 640, 3), np.uint8)
        img[0, 0, 0] = i
        pipe.process(Frame(img, i * dt, None))
    return got


def covers(boxes, box, slack=8.0):
    return any(all(abs(a - b) <= slack for a, b in zip(f, box, strict=True)) for f in boxes)


def test_a_first_unconfirmed_face_is_masked(tmp_path):
    got = faces_seen(tmp_path, lambda i: [Detection(FACE, 0.95)], 1)
    assert covers(got[0], FACE)


def test_a_face_below_the_tracking_size_is_masked(tmp_path):
    small = (300.0, 300.0, 310.0, 310.0)
    got = faces_seen(tmp_path, lambda i: [Detection(small, 0.95)], 1)
    assert covers(got[0], small)


def test_a_box_of_an_earlier_frame_is_never_given(tmp_path):
    got = faces_seen(tmp_path, lambda i: [Detection(FACE, 0.95)] if i < 3 else [], 4)
    assert covers(got[2], FACE)
    assert got[3] is not None and not covers(got[3], FACE)  # the head may have turned since


def test_a_frame_without_a_face_run_gives_none(tmp_path):
    got = faces_seen(tmp_path, lambda i: [Detection(FACE, 0.95)], 4, dt=0.001)
    assert got[0] is not None and None in got[1:]
