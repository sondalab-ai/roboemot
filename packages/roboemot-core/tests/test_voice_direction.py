"""Voice-at-face check (SPEC 8 stage 2, 5A): music from off any face needs the wake word."""

from __future__ import annotations

import json
import math

import pytest
from roboemot.config.schema import VoiceDirectionConfig
from roboemot.voice.direction import DirectionSample, VoiceDirection, face_offset

FACE = math.radians(10.0)
SEEN = DirectionSample(None, (FACE,), True)  # no reading, an attended face in view


def run(samples, doa=True, muted=False, **cfg):
    """A check over ``samples`` (robot with a sound direction when ``doa``) with the config
    overrides ``cfg``; every sample ``muted`` when asked."""
    feed = iter(samples)
    d = VoiceDirection(VoiceDirectionConfig(**cfg), lambda now: next(feed), doa_available=doa)
    d.begin()
    for k in range(len(samples)):
        d.sample(float(k), muted=muted)
    return d, d.check()


def heard(deg, faces=(FACE,), engaged=False):
    return DirectionSample(math.radians(deg), tuple(faces), engaged)


def test_music_from_60_deg_off_any_face_is_rejected():
    d, ok = run([heard(70.0)] * 10)
    assert not ok
    snap = d.snapshot()
    assert snap["checks"]["rejected"] == 1 and sum(snap["degraded"].values()) == 0
    assert snap["face_offset_deg"]["median"] == pytest.approx(60.0)


def test_speaking_from_your_face_is_accepted():
    d, ok = run([heard(14.0)] * 8 + [heard(70.0)] * 2)
    assert ok and d.snapshot()["checks"]["passed"] == 1


def test_offset_wraps_across_the_rear_and_picks_the_nearest_face():
    s = DirectionSample(math.radians(175.0), (math.radians(-175.0), 0.0), False)
    assert math.degrees(face_offset(s)) == pytest.approx(10.0)
    assert face_offset(DirectionSample(None, (0.0,), False)) is None
    assert face_offset(DirectionSample(0.0, (), False)) is None


def test_a_voice_with_no_face_in_view_is_rejected_and_counted():
    d, ok = run([heard(10.0, faces=())] * 5)
    assert not ok
    snap = d.snapshot()
    assert snap["no_face"] == 1 and snap["face_offset_deg"]["count"] == 0


@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({}, False),  # 40 deg off: outside the default tolerance
        ({"tolerance_deg": 45.0}, True),
        ({"min_match_share": 0.4}, True),  # the 4 readings at the face are enough
    ],
)
def test_thresholds_come_from_config(cfg, expected):
    near = [heard(50.0)] * 6 + [heard(12.0)] * 4  # 40 deg off for 6 readings, at the face for 4
    assert run(near, **cfg)[1] is expected


def test_sparse_readings_are_evidence_one_off_face_reading_rejects():
    off = [heard(70.0, engaged=True)] + [SEEN] * 9  # music the detector flagged once
    d, ok = run(off)
    assert not ok
    snap = d.snapshot()
    assert snap["checks"]["rejected"] == 1 and sum(snap["degraded"].values()) == 0
    on = [heard(14.0, engaged=True)] + [SEEN] * 9  # the owner, flagged once
    assert run(on)[1]


def test_min_doa_readings_comes_from_config():
    sparse = [heard(70.0, engaged=True)] + [SEEN] * 9
    d, ok = run(sparse, min_doa_readings=2, degraded_rule="engaged")
    assert ok and d.snapshot()["degraded"]["silent"] == 1


@pytest.mark.parametrize(
    "doa,muted,reason,passed",
    [
        (False, False, "no_source", True),  # robot without sound direction: engaged face
        (True, True, "muted", True),  # barge-in: the robot muted it itself
        (True, False, "silent", False),  # it could hear and gave nothing: wake word needed
    ],
)
def test_auto_rule_resolves_per_reason(doa, muted, reason, passed):
    d, ok = run([SEEN] * 5, doa=doa, muted=muted)
    assert ok is passed
    snap = d.snapshot()
    assert snap["degraded"][reason] == 1 and snap["doa_available"] is doa
    assert snap["checks"]["degraded_passed" if passed else "degraded_rejected"] == 1


def test_auto_engaged_needs_attention_on_a_visible_face():
    d, ok = run([DirectionSample(None, (FACE,), False)] * 5, doa=False)
    assert not ok and d.snapshot()["checks"]["degraded_rejected"] == 1


@pytest.mark.parametrize("rule,passed", [("engaged", True), ("reject", False)])
def test_explicit_rules_ignore_the_reason(rule, passed):
    assert run([SEEN] * 5, degraded_rule=rule)[1] is passed  # silent
    assert run([SEEN] * 5, doa=False, degraded_rule=rule)[1] is passed  # no source


def test_degraded_engaged_share_comes_from_config():
    half = [SEEN] * 4 + [DirectionSample(None, (FACE,), False)] * 6
    assert not run(half, doa=False)[1]  # 0.4 < 0.5
    assert run(half, doa=False, degraded_engaged_share=0.4)[1]


def test_no_samples_at_all_is_degraded_and_rejected():
    d, ok = run([], doa=False)
    assert not ok and d.snapshot()["checks"]["degraded_rejected"] == 1


def test_a_raising_lookup_counts_as_nothing_heard():
    def boom(now):
        raise RuntimeError

    d = VoiceDirection(VoiceDirectionConfig(), boom, doa_available=True)
    d.begin()
    d.sample(0.0)
    assert not d.check() and d.snapshot()["degraded"]["silent"] == 1


def test_begin_forgets_the_previous_utterance_and_reload_applies():
    d, _ = run([heard(70.0)] * 3)
    d.apply_config(VoiceDirectionConfig(tolerance_deg=80.0))
    d.begin()
    assert d.check() is False  # no samples left: degraded, silent
    d._lookup = lambda now: heard(70.0)
    d.begin()
    d.sample(0.0)
    assert d.check()


def test_snapshot_is_json_numbers_only():
    d, _ = run([heard(14.0)] * 3)
    snap = d.snapshot()
    assert json.loads(json.dumps(snap)) == snap
    assert set(snap["face_offset_deg"]) == {"count", "min", "median", "p90", "max", "last"}
    assert set(snap["degraded"]) == {"no_source", "silent", "muted"}
