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
from roboemot.motion.beat import room_beat
from roboemot.voice.music import (
    MusicDetector,
    _at,
    _autocorr,
    _autocorr_at,
    _detrend,
    _Estimate,
)

RATE = 16000
HOP_SAMPLES = 512  # the default music.hop_ms at RATE
HOLDS_NEEDED = 20  # of the 29 estimates of a 30 s signal
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


def test_music_holds_through_the_robots_speech():
    """Regression for the owner's rule (keep moving while it answers): the listener feeds nothing
    while the robot speaks, so ``music.release_s`` of speech must not end the music; the hold
    ends only on fresh audio with no music in it, within ``music.release_s`` after the speech."""
    det = detector()
    cfg = MusicConfig()
    run(det, drum_loop(120, 8.0, RATE))
    assert det.state(8.0).playing
    det.gap(8.0)
    speech_end = 8.0 + cfg.release_s + 2.0
    assert speech_end - 8.0 < cfg.blind_hold_s
    held = det.state(speech_end)
    assert held.playing and held.tempo_bpm is not None
    assert (beat := room_beat(det.state(speech_end))) is not None and beat.source == "room"
    events, _ = run(det, silence(cfg.release_s * 2, RATE), t0=speech_end)
    stops = [at for at, event in events if event == "stopped"]
    assert len(stops) == 1
    assert speech_end < stops[0] <= speech_end + cfg.release_s + cfg.estimate_interval_s


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
    det = detector(batch_ms=0.0)  # the stop is read at the very chunk after the hole
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
        cfg = MusicConfig(bands_hz=[(30.0, 150.0), (150.0, 9000.0)], min_bands_active=2)
        MusicDetector(cfg, RATE)


def _clicks_between_hops(hops, seconds, seed=2):
    """2.5 ms noise clicks centred on a hop boundary every ``hops`` hops, over a quiet floor."""
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    signal = 0.003 * rng.standard_normal(size)
    width = 40
    click = 0.5 * rng.standard_normal(width) * np.exp(-np.arange(width) / (width / 3))
    for at in range(4 * HOP_SAMPLES, size - width, hops * HOP_SAMPLES):
        signal[at - width // 2 : at + width // 2] += click
    return signal.astype(np.float32)


def test_a_beat_between_two_hops_holds():
    """QA 2026-11-09: without overlapping windows a click on a hop boundary fell under the
    Hann edges of both hops and the beat was not seen (0 holds of 29 before the fix)."""
    det = detector()
    run(det, _clicks_between_hops(16, 30.0))
    assert det.snapshot()["holds"] >= HOLDS_NEEDED


@pytest.mark.parametrize("seed", [7, 8])
def test_a_beat_with_strong_jitter_holds(seed):
    """QA 2026-11-09: 40 ms of timing jitter spreads a beat over two hops; the smoothing lines
    it up (at most 4 holds of 29 before the fix at this tempo)."""
    det = detector()
    run(det, drum_loop(120, 30.0, RATE, seed=seed, jitter_s=0.04))
    assert det.snapshot()["holds"] >= HOLDS_NEEDED


def test_autocorr_at_matches_the_full_autocorrelation():
    x = np.random.default_rng(5).standard_normal(180)
    full = _autocorr(x, 40)
    for lag in (7.0, 12.25, 31.5):
        assert _autocorr_at(x, lag) == pytest.approx(_at(full, lag))


def test_detrend_keeps_only_the_rises_above_the_local_level():
    steady = np.full(50, 2.0)
    assert _detrend(steady, 9) == pytest.approx(np.zeros(50))
    peak = steady.copy()
    peak[25] = 5.0
    out = _detrend(peak, 9)
    assert out[25] == pytest.approx(3.0 * 8 / 9)
    assert out[10] == pytest.approx(0.0)


def test_detrend_off_returns_the_input():
    values = np.arange(5.0)
    assert _detrend(values, 0) is values


@pytest.mark.parametrize(
    "fields",
    [
        {"bands_hz": []},
        {"bands_hz": [(200.0, 100.0), (300.0, 400.0)]},
        {"bands_hz": [(30.0, 150.0)], "min_bands_active": 2},
        {"hop_ms": 32.0, "fft_ms": 16.0},
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


def test_a_batch_runs_one_analysis_for_many_chunks(monkeypatch):
    """QA 2026-11-09: the cost is per NumPy call, so chunks are analysed together."""
    det = detector(batch_ms=256.0)
    calls = []
    energy = det._bands.energy
    monkeypatch.setattr(det._bands, "energy", lambda frames: calls.append(1) or energy(frames))
    run(det, silence(2.56, RATE))
    assert len(calls) == 9  # 128 chunks of 20 ms, 13 chunks (260 ms) to a batch


def test_batching_keeps_the_start_within_one_batch():
    signal = click_track(120, 12.0, RATE)
    batched = run(detector(batch_ms=256.0), signal)[1]
    each_chunk = run(detector(batch_ms=0.0), signal)[1]
    assert each_chunk is not None and batched is not None
    assert 0.0 <= batched - each_chunk <= 0.256 + CHUNK / RATE


def test_a_hole_analyses_the_waiting_batch_and_is_not_lost():
    """The audio before a hole is analysed first, and the hole still clears the envelope."""
    det = detector(batch_ms=256.0)
    run(det, click_track(120, 8.0, RATE))
    resets = det.snapshot()["resets"]
    det.gap(8.0)
    late = 8.0 + MusicConfig().max_gap_s + 1.0
    run(det, silence(0.5, RATE), t0=late)
    assert det.snapshot()["resets"] == resets + 1


def test_a_jump_in_the_stamps_does_not_join_the_batch():
    det = detector(batch_ms=256.0)
    det.feed(np.zeros(CHUNK, dtype=np.float32), 0.0)
    det.feed(np.zeros(CHUNK, dtype=np.float32), 5.0)
    assert det._batch_start == 5.0 and det._batch_n == CHUNK
