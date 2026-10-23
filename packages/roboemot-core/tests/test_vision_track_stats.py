"""Face track diagnostics (SPEC 5 Tracking, 12A): why tracks end and why a face comes back as a
new track. Synthetic detections through ``Tracker`` and the pipeline; measurement only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig, TrackerConfig
from roboemot.vision import CIRCADIAN, Frame, Tracker
from roboemot.vision.track_stats import TrackDiagnostics, _iou, head_pose
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
    low = diag.snapshot()["low_score"]
    assert low["refused"] == low["unmatched"] == low["near_ended"] == 0


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


# ------------------------------------------------------------------ fix round 1


def test_a_refusal_followed_by_a_recovery_does_not_mark_the_expiry():
    # refused once, then matched again for a while: the later expiry is a plain absence
    tracker, diag = observed()
    tid, stamp = confirm(tracker, 50)
    tracker.update([], stamp + DT)
    tracker.update([face(50, score=0.3)], stamp + 2 * DT)
    for k in range(3, 14):
        tracker.update([face(50)], stamp + k * DT)
    until_expired(tracker, tid, stamp + 13 * DT)
    snap = diag.snapshot()
    assert snap["low_score"]["refused"] == 1
    assert snap["expiry"]["after_refused_low"] == 0


def test_a_low_detection_overlapping_a_matched_track_is_not_unmatched():
    tracker, diag = observed()
    _, stamp = confirm(tracker, 100)
    tracker.update([face(100), face(105, score=0.3)], stamp + DT)
    tracker.update([face(100), face(250, score=0.2)], stamp + 2 * DT)
    low = diag.snapshot()["low_score"]
    assert low["unmatched"] == 1 and low["unmatched_score"]["p50"] == 0.2
    assert low["near_ended"] == 0


def test_a_weak_face_where_a_track_just_ended_counts_near_ended():
    tracker, diag = observed()
    tid, last = confirm(tracker, 100)
    stamp, _ = until_expired(tracker, tid, last)
    tracker.update([face(104, score=0.4)], stamp + DT)  # near the ended track
    tracker.update([face(250, score=0.3)], stamp + 2 * DT)  # far from it
    late = stamp + CFG.vision.rebirth_window_s + DT
    tracker.update([face(104, score=0.45)], late)  # near, past the window
    low = diag.snapshot()["low_score"]
    assert low["unmatched"] == 3
    assert low["near_ended"] == 1 and low["near_ended_score"]["p50"] == 0.4


def test_overlapped_rebirth_iou_is_taken_at_the_birth_not_at_the_expiry():
    # a face moving 10 px per frame jumps 40 px: the old track coasts on its velocity until it
    # expires, but the IoU that failed is the one against its box predicted at the new birth
    tracker, diag = observed()
    x, stamp = 20.0, 0.0
    for _ in range(8):
        tracker.update([face(x)], stamp)
        x, stamp = x + 10, stamp + DT
    birth_box = face(x + 40).box
    old_id = tracker._tracks[0].track_id
    predicted = None
    removed: list[int] = []
    while old_id not in removed:
        res = tracker.update([face(x + 40)], stamp)
        removed = res.removed_ids
        if predicted is None:
            predicted = next(t.box for t in tracker._tracks if t.track_id == old_id)
        stamp += DT
    snap = diag.snapshot()
    reb = snap["rebirths"]
    assert reb["count"] == 1 and reb["overlapped"] == 1
    assert reb["iou"]["p50"] == pytest.approx(_iou(birth_box, predicted))
    assert snap["start_near_iou"] == {
        "count": 1,
        "p50": reb["iou"]["p50"],
        "p90": reb["iou"]["p50"],
        "max": reb["iou"]["p50"],
    }
    assert reb["iou"]["p50"] < CFG.vision.tracker.match_iou


def test_the_nearer_of_two_ended_tracks_is_paired_once():
    tracker, diag = observed()
    tracker.update([face(100), face(150)], 0.0)
    res = tracker.update([face(100), face(150)], DT)
    ids = [t.track_id for t in res.visible]
    stamp = DT
    removed: list[int] = []
    while len(removed) < 2:
        stamp += DT
        removed += tracker.update([], stamp).removed_ids
    assert sorted(removed) == sorted(ids)
    confirm(tracker, 140, stamp + DT)  # 10 px from the second, 40 px from the first
    reb = diag.snapshot()["rebirths"]
    assert reb["count"] == 1 and reb["distance_heights"]["p50"] == pytest.approx(10 / 60)
    # a second birth near the consumed track only (25 px from it, 75 px from the other)
    tracker.update([face(140), face(175)], stamp + 3 * DT)
    tracker.update([face(140), face(175)], stamp + 4 * DT)
    assert diag.snapshot()["rebirths"]["count"] == 1


def test_single_hit_confirmation_still_reports_rebirths():
    tracker = Tracker(TrackerConfig(confirm_hits=1))
    diag = TrackDiagnostics(CFG.vision)
    tracker.observer = diag
    tid = tracker.update([face(100)], 0.0).visible[0].track_id
    stamp, _ = until_expired(tracker, tid, 0.0)
    res = tracker.update([face(102)], stamp + DT)
    assert res.new_ids and diag.snapshot()["rebirths"]["count"] == 1


def test_rebirth_settings_hot_reload_through_apply_config(tmp_path):
    _, _, pipe, _, _ = rig(
        tmp_path, lambda i: [face(100)] if i < 10 else [face(150)] if 30 <= i < 40 else []
    )
    wide = CFG.vision.model_copy(update={"rebirth_distance_heights": 2.0})
    pipe.apply_config(wide, CFG.identity)
    for i in range(45):
        pipe.process(Frame(image(i), i * DT))
    reb = pipe.stats()["tracks"]["rebirths"]
    assert reb["distance_heights_max"] == 2.0
    assert reb["count"] == 1 and reb["distance_heights"]["p50"] == pytest.approx(50 / 60)


def test_ended_by_adds_up_to_ended_with_expiry_and_reset(tmp_path):
    on = [True]
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(20), face(200)] if on[0] else [face(200)])
    for i in range(4):
        pipe.process(Frame(image(i), i * DT))
    on[0] = False
    for i in range(4, 24):
        pipe.process(Frame(image(i), i * DT))
    pipe.reset()
    tracks = pipe.stats()["tracks"]
    assert tracks["ended_by"] == {"expired": 1, "reset": 1}
    assert sum(tracks["ended_by"].values()) == tracks["ended"] == 2


def test_faces_below_min_px_are_counted(tmp_path):
    small = CFG.vision.face_min_px - 1
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100), face(200, size=small)])
    for i in range(3):
        pipe.process(Frame(image(i), i * DT))
    assert pipe.stats()["tracks"]["below_min_px"] == 3


def test_rebirth_settings_have_defaults_and_ranges():
    v = CFG.vision
    assert v.rebirth_window_s == 5.0 and v.rebirth_distance_heights == 1.0
    for bad in (
        {"rebirth_window_s": 0.0},
        {"rebirth_window_s": 61.0},
        {"rebirth_distance_heights": 0.05},
        {"rebirth_distance_heights": 11.0},
    ):
        with pytest.raises(ValidationError):
            RoboEmotConfig(vision=bad)
