"""SPEC 8, 11A: the music detector on generated audio (click track, drum loop, speech, fan noise,
a ticking clock), its back-dated interval history, release, gaps, blind hold and flywheel phase."""

from __future__ import annotations

import math

import numpy as np
import pytest
from audio_signals import (
    click_track,
    clock_tick,
    drum_loop,
    fan_noise,
    low_thump,
    mix,
    silence,
    speech_like,
)
from pydantic import ValidationError
from roboemot.config.schema import MusicConfig, RoboEmotConfig
from roboemot.voice.music import MusicDetector, _Estimate

RATE = 16000
CHUNK = 320  # 20 ms: not a multiple of the hop, so hops span chunks
TOLERANCE = 0.04


def run(det, signal, t0=0.0):
    """Feed ``signal`` from ``t0`` in chunks; returns [(chunk end, event)] for each event and the
    first chunk end at which ``state`` reported playing (None if never)."""
    events, first = [], None
    for i in range(0, signal.size, CHUNK):
        part = signal[i : i + CHUNK]
        start = t0 + i / RATE
        event = det.feed(part, start)
        end = start + part.size / RATE
        if event is not None:
            events.append((end, event))
        if first is None and det.state(end).playing:
            first = end
    return events, first


def near_tempo(found, bpm):
    """``found`` within 4 % of ``bpm`` or of its double or half (octave errors allowed)."""
    return any(abs(found - want) <= TOLERANCE * want for want in (bpm, bpm * 2, bpm / 2))


def detector(**overrides):
    """A detector at the test rate with ``MusicConfig`` defaults plus ``overrides``."""
    return MusicDetector(MusicConfig(**overrides), RATE)


@pytest.mark.parametrize("bpm", [70, 100, 140, 170])
@pytest.mark.parametrize("make", [click_track, drum_loop])
def test_detects_click_and_drums(make, bpm):
    det = detector()
    cfg = MusicConfig()
    events, first = run(det, make(bpm, 12.0, RATE))
    assert first is not None and first <= cfg.confirm_s + cfg.estimate_interval_s
    assert events[0][1] == "started"
    state = det.state(12.0)
    assert state.playing and state.tempo_bpm is not None
    assert near_tempo(state.tempo_bpm, bpm), state.tempo_bpm
    assert 0 < state.confidence <= 1


def test_speech_is_not_music():
    for seed in range(3):
        events, first = run(detector(), speech_like(20.0, RATE, pauses=False, seed=seed))
        assert first is None and events == [], seed


def test_speech_with_pauses_is_not_music():
    for seed in range(3):
        events, first = run(detector(), speech_like(20.0, RATE, pauses=True, seed=seed))
        assert first is None and events == [], seed


def test_fan_noise_is_not_music():
    events, first = run(detector(), fan_noise(20.0, RATE))
    assert first is None and events == []


def test_clock_tick_is_not_music():
    events, first = run(detector(), clock_tick(1.0, 20.0, RATE))
    assert first is None and events == []


def test_back_dates_the_interval():
    det = detector()
    evidence_start = 3.0
    run(det, silence(evidence_start, RATE))
    assert not det.during(0.0, evidence_start)
    events, first = run(det, drum_loop(120, 8.0, RATE), t0=evidence_start)
    assert first is not None and first > evidence_start + 1.0
    assert det.during(evidence_start + 0.5, evidence_start + 1.0)
    assert not det.during(evidence_start - 2.5, evidence_start - 1.0)


def test_release_after_music_stops():
    det = detector()
    cfg = MusicConfig()
    stop = 10.0
    run(det, drum_loop(120, stop, RATE))
    assert det.state(stop).playing
    events, _ = run(det, silence(cfg.release_s * 2, RATE), t0=stop)
    assert [e for _, e in events] == ["stopped"]
    stopped_at = events[0][0]
    assert stopped_at - stop <= cfg.release_s + cfg.estimate_interval_s
    assert stopped_at - stop >= cfg.release_s - cfg.estimate_interval_s
    assert not det.state(stopped_at).playing
    assert det.during(stopped_at - 0.1, stopped_at - 0.05)  # the release counts as music
    assert not det.during(stopped_at + 0.5, stopped_at + 1.0)


def test_gap_clears_envelope_keeps_state():
    det = detector()
    cfg = MusicConfig()
    run(det, drum_loop(100, 8.0, RATE))
    assert det.state(8.0).playing
    det.gap(8.0)
    resume = 8.0 + cfg.max_gap_s + 2.0
    fresh = drum_loop(140, 3.0, RATE, seed=3)
    events, _ = run(det, fresh, t0=resume)
    assert events == []
    snap = det.snapshot()
    assert snap["resets"] == 1
    state = det.state(resume + 3.0)
    assert state.playing
    assert near_tempo(state.tempo_bpm, 140), state.tempo_bpm  # from the fresh audio only


def test_hole_does_not_count_toward_the_release():
    det = detector()
    cfg = MusicConfig()
    run(det, drum_loop(120, 8.0, RATE))
    assert det.state(8.0).playing
    det.gap(8.0)
    resume = 8.0 + cfg.release_s + 2.0  # longer than the release, shorter than the blind hold
    assert resume - 8.0 < cfg.blind_hold_s
    events, _ = run(det, drum_loop(120, 6.0, RATE, seed=4), t0=resume)
    assert events == []
    assert det.state(resume + 6.0).playing
    assert det.snapshot()["resets"] == 1


def test_short_hole_keeps_the_envelope():
    det = detector()
    cfg = MusicConfig()
    run(det, drum_loop(120, 8.0, RATE))
    det.gap(8.0)
    events, _ = run(det, drum_loop(120, 2.0, RATE, seed=5), t0=8.0 + cfg.max_gap_s / 2)
    assert events == [] and det.snapshot()["resets"] == 0


@pytest.mark.parametrize(
    "noise",
    [
        pytest.param(lambda: speech_like(12.0, RATE, pauses=True, seed=2), id="speech"),
        pytest.param(lambda: fan_noise(12.0, RATE, seed=2), id="fan"),
    ],
)
def test_music_under_noise_still_confirms(noise):
    cfg = MusicConfig()
    music = mix(drum_loop(120, 12.0, RATE), noise(), 0.0)
    events, first = run(detector(), music)
    assert first is not None and first <= cfg.confirm_s + cfg.estimate_interval_s
    assert [e for _, e in events] == ["started"]


@pytest.mark.parametrize("bpm", [70, 100, 140, 170])
def test_jittered_drums_confirm(bpm):
    det = detector()
    events, first = run(det, drum_loop(bpm, 12.0, RATE, seed=7, jitter_s=0.010))
    assert first is not None and [e for _, e in events] == ["started"]
    assert near_tempo(det.state(12.0).tempo_bpm, bpm)


def _estimate(det, bpm, corr_at_lags=None):
    """An estimate at ``bpm`` whose autocorrelation is ``corr_at_lags`` ({lag: value}, else 0)."""
    corr = np.zeros(det._lag_hi * 2 + 2)
    for lag, value in (corr_at_lags or {}).items():
        corr[lag] = value
    return _Estimate(60.0 / bpm, 0.0, 0.9, True, corr)


def test_octave_flip_backed_by_the_envelope_is_stable():
    det = detector()
    lag = 0.5 / det._hop_s  # 120 bpm: about 15.6 lags at the default hop
    at_120 = {math.floor(lag): 0.8, math.ceil(lag): 0.8}
    before = _estimate(det, 120.0)
    assert det._stable(before, _estimate(det, 120.0 * 1.01))
    assert det._stable(before, _estimate(det, 60.0, at_120))  # still periodic at 120
    assert not det._stable(before, _estimate(det, 60.0))  # a flip the envelope does not back
    assert not det._stable(before, _estimate(det, 90.0, at_120))  # not an octave


def test_one_band_beat_is_not_music():
    events, first = run(detector(), low_thump(120, 20.0, RATE))
    assert first is None and events == []


def test_band_rule_is_what_rejects_the_clock():
    events, first = run(detector(min_bands_active=1), clock_tick(1.0, 20.0, RATE))
    assert first is not None and events[0][1] == "started"


def test_blind_hold():
    det = detector()
    cfg = MusicConfig()
    end = 8.0
    run(det, drum_loop(120, end, RATE))
    assert det.state(end).playing
    assert det.state(end + cfg.release_s + 1.0).playing
    assert det.during(end + cfg.release_s, end + cfg.release_s + 1.0)
    late = end + cfg.blind_hold_s + 1.0
    assert not det.state(late).playing
    assert not det.during(late, late + 1.0)
    chunk = silence(CHUNK / RATE, RATE)
    assert det.feed(chunk, late) == "stopped"
    assert det.feed(chunk, late + CHUNK / RATE) is None
    assert det.snapshot()["stops"] == 1


def test_phase_runs_on_flywheel():
    det = detector()
    cfg = MusicConfig()
    hop_s = cfg.hop_ms / 1000.0
    bpm = 120.0
    run(det, drum_loop(bpm, 8.0, RATE))
    first = det.state(8.0)
    assert first.playing and first.beat_phase_s is not None
    period = 60.0 / first.tempo_bpm
    assert 0 <= first.beat_phase_s < period

    def circular(a, b):
        d = abs(a - b) % period
        return min(d, period - d)

    for k in range(1, 6):
        later = det.state(8.0 + k * period)
        assert circular(later.beat_phase_s, first.beat_phase_s) <= hop_s
    true_phase = 8.0 % (60.0 / bpm)  # beats at 0, 0.5, 1.0 s, ...
    assert circular(first.beat_phase_s % (60.0 / bpm), true_phase) <= 2 * hop_s


def test_not_playing_has_no_tempo_or_phase():
    det = detector()
    run(det, silence(3.0, RATE))
    state = det.state(3.0)
    assert not state.playing and state.tempo_bpm is None and state.beat_phase_s is None


def test_snapshot_has_no_audio():
    det = detector()
    run(det, drum_loop(120, 8.0, RATE))
    snap = det.snapshot()

    def check(value, path):
        if isinstance(value, dict):
            for key, inner in value.items():
                assert isinstance(key, str), path
                check(inner, f"{path}.{key}")
        else:
            ok = value is None or isinstance(value, (bool, int, float))
            assert ok and not isinstance(value, np.ndarray), (path, type(value))
            if isinstance(value, float):
                assert math.isfinite(value), path

    check(snap, "music")
    assert snap["playing"] is True
    assert snap["starts"] == 1 and snap["estimates"] > 0
    assert {"min", "median", "p90", "last"} <= set(snap["cost_ms_per_audio_s"])


def test_cost_uses_the_injected_timer():
    ticks = iter(range(1_000_000))
    det = MusicDetector(MusicConfig(), RATE, timer=lambda: float(next(ticks)) * 1e-4)
    run(det, silence(3.0, RATE))
    cost = det.snapshot()["cost_ms_per_audio_s"]
    per_chunk_ms = 0.1  # one timer step per feed
    assert cost["last"] == pytest.approx(per_chunk_ms * RATE / CHUNK, rel=0.05)


def test_disabled_detector_hears_nothing():
    det = detector(enabled=False)
    events, first = run(det, drum_loop(120, 8.0, RATE))
    assert first is None and events == []


def test_apply_config_takes_effect_on_the_next_feed():
    det = detector()
    run(det, drum_loop(120, 8.0, RATE))
    det.apply_config(MusicConfig(enabled=False))
    assert det.feed(silence(CHUNK / RATE, RATE), 8.0) == "stopped"
    assert not det.state(8.1).playing


def test_band_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        MusicDetector(MusicConfig(bands_hz=[(30.0, 150.0), (150.0, 9000.0)]), RATE)


@pytest.mark.parametrize(
    "fields",
    [
        {"bands_hz": []},
        {"bands_hz": [(200.0, 100.0), (300.0, 400.0)]},
        {"bands_hz": [(30.0, 150.0)], "min_bands_active": 2},
        {"tempo_bpm": (180.0, 60.0)},
        {"prior_bpm": (130.0, 100.0)},
        {"window_s": 2.0, "tempo_bpm": (40.0, 180.0)},
        {"prior_bpm": (50.0, 100.0)},
        {"tempo_bpm": (60.0, 120.0)},
    ],
)
def test_invalid_music_config_is_rejected(fields):
    with pytest.raises(ValidationError):
        MusicConfig(**fields)


def test_music_is_a_root_section_with_defaults_on():
    cfg = RoboEmotConfig()
    assert cfg.music.enabled is True
    assert cfg.music.tempo_bpm == (60.0, 180.0)
    assert cfg.music.blind_hold_s == 30.0
