"""Voice-at-face check (SPEC 8 stage 2, 5A): music from off any face needs the wake word."""

from __future__ import annotations

import json
import math

import pytest
from roboemot.config.schema import VoiceDirectionConfig
from roboemot.voice.direction import DirectionSample, VoiceDirection, face_offset

FACE = math.radians(10.0)


def run(samples, **cfg):
    """A check over ``samples`` with the config overrides ``cfg``."""
    feed = iter(samples)
    d = VoiceDirection(VoiceDirectionConfig(**cfg), lambda now: next(feed))
    d.begin()
    for k in range(len(samples)):
        d.sample(float(k))
    return d, d.check()


def heard(deg, faces=(FACE,), engaged=False):
    return DirectionSample(math.radians(deg), tuple(faces), engaged)


def test_music_from_60_deg_off_any_face_is_rejected():
    d, ok = run([heard(70.0)] * 10)
    assert not ok
    snap = d.snapshot()
    assert snap["checks"]["rejected"] == 1 and snap["degraded"] == 0
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


def test_no_reading_falls_back_to_an_engaged_face_and_counts_degraded():
    seen = DirectionSample(None, (FACE,), True)
    d, ok = run([seen] * 5)
    assert ok
    snap = d.snapshot()
    assert snap["checks"]["degraded_passed"] == 1 and snap["degraded"] == 1
    d, ok = run([DirectionSample(None, (FACE,), False)] * 5)  # a face, attention elsewhere
    assert not ok and d.snapshot()["checks"]["degraded_rejected"] == 1
    assert not run([seen] * 5, degraded_rule="reject")[1]


def test_too_few_readings_run_degraded():
    sparse = [heard(70.0, engaged=True)] + [DirectionSample(None, (FACE,), True)] * 9
    d, ok = run(sparse)  # 1 reading in 10 < min_doa_share 0.3
    assert ok and d.snapshot()["degraded"] == 1
    d, ok = run(sparse, min_doa_share=0.1)  # now the one reading decides: 60 deg off
    assert not ok and d.snapshot()["checks"]["rejected"] == 1


def test_no_samples_at_all_is_degraded_and_rejected():
    d, ok = run([])
    assert not ok and d.snapshot()["checks"]["degraded_rejected"] == 1


def test_a_raising_lookup_counts_as_nothing_heard():
    def boom(now):
        raise RuntimeError

    d = VoiceDirection(VoiceDirectionConfig(), boom)
    d.begin()
    d.sample(0.0)
    assert not d.check() and d.snapshot()["degraded"] == 1


def test_begin_forgets_the_previous_utterance_and_reload_applies():
    d, _ = run([heard(70.0)] * 3)
    d.apply_config(VoiceDirectionConfig(tolerance_deg=80.0))
    d.begin()
    assert d.check() is False  # no samples left: degraded, nothing engaged
    d._lookup = lambda now: heard(70.0)
    d.begin()
    d.sample(0.0)
    assert d.check()


def test_snapshot_is_json_numbers_only():
    d, _ = run([heard(14.0)] * 3)
    snap = d.snapshot()
    assert json.loads(json.dumps(snap)) == snap
    assert set(snap["face_offset_deg"]) == {"count", "min", "median", "max", "last"}
