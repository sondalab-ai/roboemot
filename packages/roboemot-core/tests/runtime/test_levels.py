"""SPEC 11A, 12: the Live tab's audio levels. The microphone level is the hub's own per-chunk RMS
(the reflex sink, forwarded unchanged); the output level is read from the playback reference. Levels
only, never samples; asleep or paused reports paused and reads nothing."""

from __future__ import annotations

import math

import numpy as np
from roboemot.runtime.levels import SILENCE_DBFS, AudioLevels, dbfs
from roboemot.voice.bargein import PlaybackReference

RATE = 16000
ARGS = {"max_age_s": 1.0, "window_ms": 500}


def levels(forward=None):
    now = [100.0]
    return AudioLevels(forward, clock=lambda: now[0]), now


def test_dbfs_of_known_levels():
    assert math.isclose(dbfs(1.0), 0.0)
    assert math.isclose(dbfs(0.1), -20.0)
    assert dbfs(0.0) == SILENCE_DBFS and dbfs(float("nan")) == SILENCE_DBFS
    assert dbfs(1e-9) == SILENCE_DBFS


def test_the_hub_sink_forwards_to_the_reflex_engine_unchanged_and_keeps_the_level():
    seen = []
    lv, _ = levels(lambda rms, stamp: seen.append((rms, stamp)))
    lv.offer(0.1, 99.8)
    assert seen == [(0.1, 99.8)]
    snap = lv.snapshot(paused=False, **ARGS)
    assert snap["mic"] == {"state": "ok", "db": -20.0}


def test_a_failing_forward_still_propagates_like_before():
    def boom(rms, stamp):
        raise RuntimeError("x")

    lv, _ = levels(boom)
    try:
        lv.offer(0.1, 99.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the reflex sink's error must not be swallowed")


def test_a_stale_microphone_level_is_no_data():
    lv, now = levels(lambda *a: None)
    assert lv.snapshot(paused=False, **ARGS)["mic"]["state"] == "no_data"
    lv.offer(0.1, 100.0)
    now[0] += 1.5
    assert lv.snapshot(paused=False, **ARGS)["mic"] == {"state": "no_data", "db": None}


def test_without_a_reflex_sink_the_microphone_is_unavailable():
    lv, _ = levels(None)
    assert lv.snapshot(paused=False, **ARGS)["mic"] == {"state": "unavailable", "db": None}
    assert lv.mic_fed is False


def test_the_noise_floor_is_in_dbfs_or_absent():
    lv, _ = levels(lambda *a: None)
    assert lv.snapshot(paused=False, **ARGS)["floor_db"] is None
    lv.noise_floor = lambda: 0.01
    assert lv.snapshot(paused=False, **ARGS)["floor_db"] == -40.0


def test_the_output_level_is_the_playback_over_the_window():
    lv, _ = levels(lambda *a: None)
    ref = PlaybackReference(RATE, 2.0)
    lv.reference = ref
    assert lv.snapshot(paused=False, **ARGS)["out"] == {"state": "ok", "db": SILENCE_DBFS}
    ref.add(99.6, np.full(RATE // 4, 0.1, np.float32))  # 250 ms of 0.1 inside the 500 ms window
    out = lv.snapshot(paused=False, **ARGS)["out"]
    assert out["state"] == "ok" and -24.0 < out["db"] < -22.0  # 0.1 over half the window


def test_no_playback_reference_means_unavailable():
    lv, _ = levels(lambda *a: None)
    assert lv.snapshot(paused=False, **ARGS)["out"] == {"state": "unavailable", "db": None}


def test_paused_reports_paused_and_reads_nothing():
    class Boom:
        def window(self, *a):
            raise AssertionError("must not read while paused")

    lv, _ = levels(lambda *a: None)
    lv.reference = Boom()
    lv.noise_floor = lambda: (_ for _ in ()).throw(AssertionError("must not read while paused"))
    lv.offer(0.5, 100.0)
    snap = lv.snapshot(paused=True, **ARGS)
    assert snap == {
        "mic": {"state": "paused", "db": None},
        "out": {"state": "paused", "db": None},
        "floor_db": None,
    }


def test_only_scalars_are_held():
    lv, _ = levels(lambda *a: None)
    lv.offer(0.1, 100.0)
    assert lv._mic == (0.1, 100.0)
    assert not any(isinstance(v, np.ndarray) for v in vars(lv).values())
