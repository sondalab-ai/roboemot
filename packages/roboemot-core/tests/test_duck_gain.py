"""SPEC 8 stage 8: the duck gain, a playback gain keyed to the time each sample plays."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import BargeInDuckConfig
from roboemot.voice.duck import DuckGain

RATE = 1000  # one sample per ms: indices read as milliseconds


def gain_curve(duck: DuckGain, start: float, n: int) -> np.ndarray:
    return duck.apply(np.ones(n, np.float32), RATE, start)


def test_without_a_duck_the_samples_pass_unchanged():
    duck = DuckGain(BargeInDuckConfig())
    x = np.full(50, 0.5, np.float32)
    assert duck.apply(x, RATE, 10.0) is x
    assert not duck.ducking(10.0)


def test_a_duck_ramps_down_over_attack_holds_and_ramps_back_up_after_until():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.9, 100)  # playing at full gain up to the duck
    duck.duck(1.0, 1.1)
    g = gain_curve(duck, 1.0, 200)  # one push that spans the whole episode
    assert g[0] == pytest.approx(1.0)
    assert g[5] == pytest.approx(1.0 - 0.75 * 5 / 10)
    assert g[10:100] == pytest.approx(0.25)
    assert g[125] == pytest.approx(0.25 + 0.75 * 25 / 50)
    assert g[150:] == pytest.approx(1.0)
    assert np.all(np.abs(np.diff(g)) <= 0.75 / 10 + 1e-6)  # no step: no click


def test_the_level_is_continuous_across_pushes_already_past_the_duck_start():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.0, 500)  # queued earlier, at full gain
    duck.duck(0.2, 5.0)  # the candidate's stamp is long before the next push plays
    g = gain_curve(duck, 0.5, 30)  # contiguous with the last push
    assert g[0] == pytest.approx(1.0)  # starts where the last pushed sample was
    assert g[10:] == pytest.approx(0.25)


def test_silence_between_pushes_lets_the_level_move():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.0, 100)
    duck.duck(0.2, 5.0)
    g = gain_curve(duck, 0.5, 10)  # 400 ms of nothing queued in between
    assert g == pytest.approx(0.25)


def test_a_new_candidate_extends_the_episode():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=0, ramp_ms=0))
    duck.duck(1.0, 1.1)
    duck.duck(1.05, 1.3)
    g = gain_curve(duck, 1.0, 400)
    assert g[:300] == pytest.approx(0.25) and g[300:] == pytest.approx(1.0)
    assert duck.ducking(1.2) and not duck.ducking(1.3)


def test_release_ends_the_episode_early_and_clear_snaps_back_to_full_gain():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=0, ramp_ms=50))
    duck.duck(1.0, 9.0)
    duck.release(1.1)
    g = gain_curve(duck, 1.0, 200)
    assert g[:100] == pytest.approx(0.25)
    assert g[100] == pytest.approx(0.25) and g[150:] == pytest.approx(1.0)
    duck.duck(2.0, 9.0)
    assert gain_curve(duck, 2.0, 5) == pytest.approx(0.25)
    duck.clear()  # a stop flushed the player: the next push starts at full gain
    assert gain_curve(duck, 2.005, 5) == pytest.approx(1.0)


def test_a_hot_reload_changes_the_gain_and_the_ramps():
    duck = DuckGain(BargeInDuckConfig())
    duck.apply_config(BargeInDuckConfig(gain=0.5, attack_ms=0))
    duck.duck(0.0, 1.0)
    assert gain_curve(duck, 0.0, 10) == pytest.approx(0.5)


def test_a_first_push_after_silence_starts_at_the_level_the_episode_asks_for():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10))
    duck.duck(1.0, 2.0)
    assert gain_curve(duck, 1.5, 10) == pytest.approx(0.25)  # nothing before it to click against


def test_the_output_is_float32_and_the_input_is_not_modified():
    duck = DuckGain(BargeInDuckConfig(attack_ms=0))
    duck.duck(0.0, 1.0)
    x = np.full(20, 0.8, np.float32)
    y = duck.apply(x, RATE, 0.0)
    assert y.dtype == np.float32 and y == pytest.approx(0.2)
    assert x == pytest.approx(0.8)


def test_a_rateless_or_empty_chunk_passes_unchanged():
    duck = DuckGain(BargeInDuckConfig())
    duck.duck(0.0, 1.0)
    x = np.ones(10, np.float32)
    assert duck.apply(x, 0, 0.5) is x
