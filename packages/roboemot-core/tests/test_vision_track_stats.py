"""Face track diagnostics (SPEC 5 Tracking, 12A): why tracks end and why a face comes back as a
new track. Synthetic detections through ``Tracker`` and the pipeline; measurement only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from roboemot.vision import CIRCADIAN, Frame, Tracker
from roboemot.vision.track_stats import TrackDiagnostics, head_pose
from roboemot.world import FrameSpec
from test_vision import CFG, face, image, rig

DT = 0.125
LOST = CFG.vision.tracker.lost_s
SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))


def observed() -> tuple[Tracker, TrackDiagnostics]:
    tracker = Tracker(CFG.vision.tracker)
    diag = TrackDiagnostics(CFG.vision)
    tracker.observer = diag
    return tracker, diag


def confirm(tracker: Tracker, x: float, start: float = 0.0) -> tuple[int, float]:
    """Two matched updates: the track is confirmed; returns its id and last stamp."""
    tracker.update([face(x)], start)
    res = tracker.update([face(x)], start + DT)
    return res.visible[0].track_id, start + DT


def until_expired(tracker: Tracker, track_id: int, stamp: float) -> tuple[float, int]:
    """Empty updates until ``track_id`` is removed; returns that stamp and the misses before."""
    misses = 0
    while True:
        stamp += DT
        if track_id in tracker.update([], stamp).removed_ids:
            return stamp, misses
        misses += 1


def test_observer_does_not_change_tracking():
    rng = np.random.default_rng(7)
    script = []
    for i in range(300):
        dets = []
        for base in (40.0, 200.0):
            if rng.random() < 0.7:
                x = base + 20 * math.sin(i / 15) + rng.normal(0, 3)
                dets.append(face(x, score=float(rng.choice([0.9, 0.7, 0.4, 0.2, 0.05]))))
        script.append(dets)
    plain, (watched, _) = Tracker(CFG.vision.tracker), observed()
    for i, dets in enumerate(script):
        a, b = plain.update(dets, i * DT), watched.update(dets, i * DT)
        assert [(t.track_id, t.box) for t in a.visible] == [(t.track_id, t.box) for t in b.visible]
        assert a.new_ids == b.new_ids and a.removed_ids == b.removed_ids


def test_expiry_counts_misses_and_the_lost_s_in_force():
    tracker, diag = observed()
    tid, last = confirm(tracker, 100)
    _, misses = until_expired(tracker, tid, last)
    snap = diag.snapshot()
    assert snap["ended_by"] == {"expired": 1, "reset": 0}
    assert snap["expiry"]["misses"]["p50"] == misses == int(LOST / DT)
    assert snap["expiry"]["lost_s"]["p50"] == LOST
    assert snap["expiry"]["after_refused_low"] == 0
    json.dumps(snap)


def test_low_score_refused_after_a_miss_and_unmatched_elsewhere():
    tracker, diag = observed()
    tid, stamp = confirm(tracker, 50)
    tracker.update([], stamp + DT)  # one miss: the low pass now skips the track
    res = tracker.update([face(50, score=0.3), face(250, score=0.3)], stamp + 2 * DT)
    assert res.visible == []
    until_expired(tracker, tid, stamp + 2 * DT)
    snap = diag.snapshot()
    assert snap["low_score"]["refused"] == 1
    assert snap["low_score"]["refused_score"]["p50"] == 0.3
    assert snap["low_score"]["unmatched"] == 1
    assert snap["expiry"]["after_refused_low"] == 1


def test_low_score_matched_after_a_hit_is_not_refused():
    tracker, diag = observed()
    _, stamp = confirm(tracker, 50)
    tracker.update([face(50, score=0.3)], stamp + DT)
    assert diag.snapshot()["low_score"] == {
        "refused": 0,
        "refused_score": {"count": 0},
        "unmatched": 0,
    }


def test_start_scores_and_tentative_drops():
    tracker, diag = observed()
    tracker.update([face(20, score=0.9), face(200, score=0.7)], 0.0)
    tracker.update([], DT)
    snap = diag.snapshot()
    assert snap["start_score"] == {"count": 2, "p50": 0.7, "p90": 0.9, "max": 0.9}
    assert snap["tentative_dropped"] == 2
    assert snap["ended_by"]["expired"] == 0


def test_rebirth_after_the_old_track_ended():
    tracker, diag = observed()
    tid, last = confirm(tracker, 100)
    stamp, _ = until_expired(tracker, tid, last)
    tracker.update([face(104)], stamp + DT)
    tracker.update([face(104)], stamp + 2 * DT)
    reb = diag.snapshot()["rebirths"]
    assert reb["count"] == 1 and reb["overlapped"] == 0
    assert reb["gap_s"]["p50"] == pytest.approx(stamp + DT - last)
    assert reb["distance_heights"]["p50"] == pytest.approx(4 / 60)
    assert reb["iou"]["p50"] == pytest.approx(56 * 60 / (2 * 3600 - 56 * 60))
    assert reb["head_yaw_change_deg"] == {"count": 0}


def test_rebirth_while_the_old_track_still_lived_is_overlapped():
    # the face jumps 40 px: IoU 0.2 is below match_iou, so a second track starts beside the first
    tracker, diag = observed()
    tracker.update([face(50)], 0.0)
    old = tracker.update([face(50)], DT).visible[0].track_id
    stamp = DT
    removed: list[int] = []
    while old not in removed:
        stamp += DT
        removed = tracker.update([face(90)], stamp).removed_ids
    reb = diag.snapshot()["rebirths"]
    assert reb["count"] == 1 and reb["overlapped"] == 1
    assert reb["gap_s"]["p50"] == pytest.approx(DT)
    assert reb["distance_heights"]["p50"] == pytest.approx(40 / 60, abs=0.05)


def test_far_or_late_tracks_are_not_rebirths():
    tracker, diag = observed()
    tid, last = confirm(tracker, 20)
    stamp, _ = until_expired(tracker, tid, last)
    confirm(tracker, 220, stamp + DT)  # 200 px away: more than one box height
    late = stamp + CFG.vision.rebirth_window_s + DT  # the window runs from the expiry
    tracker.update([], late)  # the far track expires too
    confirm(tracker, 20, late)  # near the first one, but past the window
    assert diag.snapshot()["rebirths"]["count"] == 0
    assert diag.snapshot()["ended_by"]["expired"] == 2


def test_rebirth_reports_the_head_pose_change():
    tracker, diag = observed()
    diag.pose = (0.0, 0.0)
    tid, last = confirm(tracker, 100)
    stamp, _ = until_expired(tracker, tid, last)
    diag.pose = (math.radians(12.0), math.radians(-3.0))
    confirm(tracker, 100, stamp + DT)
    reb = diag.snapshot()["rebirths"]
    assert reb["head_yaw_change_deg"]["p50"] == pytest.approx(12.0)
    assert reb["head_pitch_change_deg"]["p50"] == pytest.approx(3.0)


def test_head_pose_needs_a_spec_and_every_dof():
    assert head_pose({"head.yaw": 0.1, "head.pitch": 0.2}, SPEC) == (0.1, 0.2)
    assert head_pose({"head.yaw": 0.1}, SPEC) is None
    assert head_pose(None, SPEC) is None
    assert head_pose({"head.yaw": 0.1, "head.pitch": 0.2}, None) is None


# ------------------------------------------------------------------------ pipeline


def test_pipeline_reports_rebirths_with_head_pose(tmp_path):
    # face for frames 0-9, gone 2.5 s (past lost_s), back 4 px away at frames 30-39; the head
    # turned 0.1 rad in between
    _, _, pipe, _, _ = rig(
        tmp_path,
        lambda i: [face(100)] if i < 10 else [face(104)] if 30 <= i < 40 else [],
        frame_spec=SPEC,
    )
    for i in range(45):
        joint = {"head.yaw": 0.1 if i >= 20 else 0.0, "head.pitch": 0.0}
        pipe.process(Frame(image(i), i * DT, joint))
    tracks = pipe.stats()["tracks"]
    assert tracks["created"] == 2 and tracks["ended_by"]["expired"] == 1
    reb = tracks["rebirths"]
    assert reb["count"] == 1 and reb["gap_s"]["p50"] == pytest.approx(30 * DT - 9 * DT)
    assert reb["head_yaw_change_deg"]["p50"] == pytest.approx(math.degrees(0.1))
    assert reb["head_pitch_change_deg"]["p50"] == pytest.approx(0.0)
    json.dumps(pipe.stats())


def test_pipeline_without_frame_spec_reports_no_head_change(tmp_path):
    _, _, pipe, _, _ = rig(
        tmp_path, lambda i: [face(100)] if i < 10 else [face(104)] if 30 <= i < 40 else []
    )
    for i in range(45):
        pipe.process(Frame(image(i), i * DT, {"head.yaw": 0.1 * i, "head.pitch": 0.0}))
    reb = pipe.stats()["tracks"]["rebirths"]
    assert reb["count"] == 1 and reb["head_yaw_change_deg"] == {"count": 0}


def test_pipeline_reset_counts_in_ended_by(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(20), face(200)])
    for i in range(4):
        pipe.process(Frame(image(i), i * DT))
    pipe.reset()
    tracks = pipe.stats()["tracks"]
    assert tracks["ended_by"] == {"expired": 0, "reset": 2}
    assert sum(tracks["ended_by"].values()) == tracks["ended"]


def test_pipeline_expiry_records_the_effective_lost_s(tmp_path):
    # a slow detector (circadian 0.1: 1.25 s period) stretches the face lifetime past lost_s
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if i < 40 else [])
    pipe.governor.set_scale(CIRCADIAN, 0.1)
    for i in range(120):
        pipe.process(Frame(image(i), i * DT))
    tracks = pipe.stats()["tracks"]
    assert tracks["ended_by"]["expired"] == 1
    effective = pipe._face_lost_s()
    assert effective > LOST
    assert tracks["expiry"]["lost_s"]["p50"] == pytest.approx(effective)
    assert tracks["rebirths"]["window_s"] == CFG.vision.rebirth_window_s


def test_rebirth_window_starts_at_expiry_not_at_the_last_match(tmp_path):
    # a slow detector stretches lost_s to 3.75 s: a window from the last match (2 s) would miss
    # a face back right after the expiry; from the expiry it is a rebirth
    vision = CFG.vision.model_copy(update={"rebirth_window_s": 2.0})
    on = [True]
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if on[0] else [], vision=vision)
    pipe.governor.set_scale(CIRCADIAN, 0.1)
    i = 0
    for _ in range(40):
        pipe.process(Frame(image(i), i * DT))
        i += 1
    on[0] = False
    while pipe.stats()["tracks"]["ended_by"]["expired"] == 0:
        pipe.process(Frame(image(i), i * DT))
        i += 1
    on[0] = True
    for _ in range(30):
        pipe.process(Frame(image(i), i * DT))
        i += 1
    reb = pipe.stats()["tracks"]["rebirths"]
    assert reb["count"] == 1 and reb["overlapped"] == 0
    assert reb["lost_s"]["p50"] == pytest.approx(pipe._face_lost_s())
    assert reb["gap_s"]["p50"] > vision.rebirth_window_s
