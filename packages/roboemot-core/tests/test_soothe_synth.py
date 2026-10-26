"""SPEC 7B: the self-soothing synth (seeded, click-free, bounded, on the scale)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig, SoundShape
from roboemot.soothe.synth import (
    BLIRP_COUNT,
    PENTATONIC_ST,
    hum_breakpoints,
    plan_blirp,
    plan_hum,
    synthesize,
)

RATE = 16000
SHAPES = {f"{v.sound}:{v.label}": v.shape for v in RoboEmotConfig().soothe.variants}
DEFAULT = SoundShape()
SEEDS = range(12)


EXTREME = SoundShape(tempo_bpm=(240, 240), pitch_hz=(2000, 2000), length_s=(0.3, 0.3), gain_db=0)
EXTREME_SLOW = SoundShape(tempo_bpm=(20, 20), pitch_hz=(80, 80), length_s=(10, 10), gain_db=-60)


def make(kind: str, seed: int, shape: SoundShape = DEFAULT, rate_hz: int = RATE) -> np.ndarray:
    return synthesize(kind, shape, random.Random(seed), rate_hz)


@pytest.mark.parametrize("kind", ["hum", "blirp"])
def test_the_same_seed_gives_the_same_samples_and_other_seeds_other_sounds(kind):
    a, b = make(kind, 1), make(kind, 1)
    assert a.dtype == np.float32 and a.ndim == 1
    assert np.array_equal(a, b)
    others = [make(kind, s) for s in range(2, 8)]
    assert len({w.tobytes() for w in [a, *others]}) == 7


@pytest.mark.parametrize("kind", ["hum", "blirp"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_peak_ends_and_length_follow_the_shape(kind, name):
    shape = SHAPES[name]
    gain = 10 ** (shape.gain_db / 20)
    for seed in SEEDS:
        wave = make(kind, seed, shape)
        assert wave.dtype == np.float32 and np.isfinite(wave).all()
        assert 0 < np.max(np.abs(wave)) <= gain
        assert wave[0] == 0.0 and wave[-1] == 0.0
        assert abs(wave[1]) < 0.01 * gain and abs(wave[-2]) < 0.01 * gain
        lo, hi = shape.length_s
        assert lo - 1 / RATE <= wave.size / RATE <= hi + 1 / RATE


def test_the_peak_reaches_the_gain_but_never_passes_it():
    shape = SoundShape(gain_db=-3.0)
    for kind in ("hum", "blirp"):
        peak = float(np.max(np.abs(make(kind, 4, shape))))
        assert 0.9 * 10 ** (-3 / 20) < peak <= 10 ** (-3 / 20)


def test_hum_notes_lie_on_the_pentatonic_scale_and_end_on_the_tonic():
    for seed in SEEDS:
        plan = plan_hum(DEFAULT, random.Random(seed))
        assert DEFAULT.pitch_hz[0] <= plan.base_hz <= DEFAULT.pitch_hz[1]
        for note in plan.notes:
            semitones = (12 * math.log2(note.freq_hz / plan.base_hz)) % 12
            off = min(min(abs(semitones - s), 12 - abs(semitones - s)) for s in PENTATONIC_ST)
            assert off < 1e-6
        last = plan.notes[-1]
        assert last.degree % 5 == 0
        assert last.freq_hz == pytest.approx(plan.base_hz * 2 ** (last.degree // 5))
        starts = [n.start_s for n in plan.notes]
        assert starts == sorted(starts) and starts[0] == 0.0
        assert sum(n.dur_s for n in plan.notes) == pytest.approx(plan.length_s)


def test_hum_moves_mostly_by_one_degree():
    steps = []
    for seed in range(40):
        degrees = [n.degree for n in plan_hum(DEFAULT, random.Random(seed)).notes[:-1]]
        steps += [abs(b - a) for a, b in zip(degrees, degrees[1:], strict=False)]
    assert sum(s <= 1 for s in steps) / len(steps) > 0.7
    assert any(s > 1 for s in steps)


def test_blirp_has_two_to_eight_sound_events_inside_the_length():
    counts = set()
    for seed in range(60):
        plan = plan_blirp(DEFAULT, random.Random(seed))
        counts.add(len(plan.events))
        assert BLIRP_COUNT[0] <= len(plan.events) <= BLIRP_COUNT[1]
        ends = [e.start_s + e.dur_s for e in plan.events]
        starts = [e.start_s for e in plan.events]
        assert all(s2 >= e1 for e1, s2 in zip(ends, starts[1:], strict=False))
        assert ends[-1] == pytest.approx(plan.length_s)
    assert len(counts) > 1


def test_every_blirp_kind_turns_up():
    kinds = {e.kind for s in range(80) for e in plan_blirp(DEFAULT, random.Random(s)).events}
    assert kinds == {"chirp_up", "chirp_down", "fm", "trill", "warble"}


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        synthesize("whistle", DEFAULT, random.Random(0), RATE)  # type: ignore[arg-type]


def test_the_synth_reads_no_global_random_state():
    random.seed(123)
    a = make("hum", 5)
    random.seed(456)
    assert np.array_equal(a, make("hum", 5))


@pytest.mark.parametrize("rate_hz", [8000, 16000, 48000])
@pytest.mark.parametrize("shape", [EXTREME, EXTREME_SLOW], ids=["short-fast-high", "long-slow-low"])
@pytest.mark.parametrize("kind", ["hum", "blirp"])
def test_extreme_shapes_stay_finite_bounded_and_click_free(kind, shape, rate_hz):
    gain = 10 ** (shape.gain_db / 20)
    for seed in range(6):
        wave = make(kind, seed, shape, rate_hz)
        assert np.isfinite(wave).all() and 0 < np.max(np.abs(wave)) <= gain
        assert wave[0] == 0.0 and wave[-1] == 0.0
        assert wave.size / rate_hz == pytest.approx(shape.length_s[0], abs=2 / rate_hz)


def test_no_click_in_a_hum_the_sample_to_sample_jump_stays_small():
    for seed in range(20):
        wave = make("hum", seed, DEFAULT)
        assert float(np.max(np.abs(np.diff(wave)))) < 0.5 * float(np.max(np.abs(wave)))


def test_no_click_in_a_blirp_at_the_joins_of_sound_and_silence():
    for seed in range(20):
        wave = make("blirp", seed, DEFAULT)
        edges = np.abs(np.diff(wave))
        assert (
            edges[:8].max() < 0.01 * np.abs(wave).max()
            and edges[-8:].max() < 0.01 * np.abs(wave).max()
        )


def test_hum_envelope_and_glide_times_strictly_increase_even_for_short_notes():
    for shape in (DEFAULT, EXTREME):
        for seed in SEEDS:
            plan = plan_hum(shape, random.Random(seed))
            pitch_x, _, env_x, env_y = hum_breakpoints(plan)
            assert all(b > a for a, b in zip(env_x, env_x[1:], strict=False))
            assert all(b > a for a, b in zip(pitch_x, pitch_x[1:], strict=False))
            assert env_y[0] == 0.0 and env_y[-1] == 0.0


def test_a_hum_drops_the_harmonics_that_would_pass_the_frequency_cap():
    wave = make("hum", 2, SoundShape(pitch_hz=(2000, 2000), gain_db=0), 8000)
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(wave.size, 1 / 8000)
    assert spectrum[freqs > 0.35 * 8000 * 1.2].max() < 0.05 * spectrum.max()
