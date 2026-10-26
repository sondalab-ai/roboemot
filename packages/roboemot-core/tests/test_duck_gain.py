"""SPEC 8 stage 8: the duck gain, a playback gain keyed to the time each sample plays."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from roboemot.config.schema import BargeInDuckConfig
from roboemot.voice.duck import DuckGain

RATE = 1000  # one sample per ms: indices read as milliseconds
FAST = {"attack_ms": 1, "ramp_ms": 1}  # the shortest ramps: one sample at this rate


def gain_curve(duck: DuckGain, start: float, n: int) -> np.ndarray:
    return duck.apply(np.ones(n, np.float32), RATE, start)


def test_without_a_duck_the_samples_pass_unchanged():
    duck = DuckGain(BargeInDuckConfig())
    x = np.full(50, 0.5, np.float32)
    assert duck.apply(x, RATE, 10.0) is x
    assert not duck.ducking(10.0)


def test_a_duck_ramps_down_over_attack_holds_and_ramps_back_up_after_the_release():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.9, 100)  # playing at full gain up to the duck
    duck.duck(1.0)
    duck.release(1.1)
    g = gain_curve(duck, 1.0, 200)  # one push that spans the whole episode
    assert g[0] == pytest.approx(1.0)
    assert g[5] == pytest.approx(1.0 - 0.75 * 5 / 10)
    assert g[10:100] == pytest.approx(0.25)
    assert g[125] == pytest.approx(0.25 + 0.75 * 25 / 50)
    assert g[150:] == pytest.approx(1.0)
    assert np.all(np.abs(np.diff(g)) <= 0.75 / 10 + 1e-6)  # no step: no click


def test_an_open_episode_ducks_a_push_longer_than_it_for_its_whole_length():
    """The user is still talking: no release is decided, so nothing comes back up early."""
    duck = DuckGain(BargeInDuckConfig(gain=0.25, **FAST))
    duck.duck(10.0)
    g = gain_curve(duck, 10.0, 1000)  # a one-second push (a Piper message is 200 ms)
    assert g == pytest.approx(0.25)
    assert duck.ducking(10.999)


def test_an_episode_no_candidate_extends_releases_at_the_cap_with_the_ramp_and_is_counted():
    """Barge-in stopped getting frames: the episode must not duck every later answer."""
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=1, ramp_ms=50, max_episode_s=2.0))
    duck.duck(10.0)
    duck.duck(10.5)  # the last candidate: the cap counts from it
    assert duck.ducking(12.49) and not duck.ducking(12.5)
    g = gain_curve(duck, 12.4, 300)
    assert g[:100] == pytest.approx(0.25)
    assert g[100] == pytest.approx(0.25) and g[150:] == pytest.approx(1.0)  # the ramp, no step
    gain_curve(duck, 12.7, 10)  # the first push wholly after the cap counts it
    assert duck.capped == 1
    duck.duck(20.0)  # a new episode later: the capped one is counted once
    assert duck.capped == 1


def test_the_level_is_continuous_across_pushes_already_past_the_duck_start():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.0, 500)  # queued earlier, at full gain
    duck.duck(0.2)  # the candidate's stamp is long before the next push plays
    g = gain_curve(duck, 0.5, 30)  # contiguous with the last push
    assert g[0] == pytest.approx(1.0)  # starts where the last pushed sample was
    assert g[10:] == pytest.approx(0.25)


def test_silence_between_pushes_lets_the_level_move():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10, ramp_ms=50))
    gain_curve(duck, 0.0, 100)
    duck.duck(0.2)
    g = gain_curve(duck, 0.5, 10)  # 400 ms of nothing queued in between
    assert g == pytest.approx(0.25)


def test_a_candidate_while_open_keeps_the_start_and_one_after_a_release_reopens():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, **FAST))
    duck.duck(1.0)
    duck.duck(1.05)  # still open: the start stays
    duck.release(1.2)
    assert duck.ducking(1.0) and not duck.ducking(1.2)
    duck.duck(1.5)  # a new candidate after the release
    assert not duck.ducking(1.3) and duck.ducking(1.5)


def test_release_ends_the_episode_and_clear_snaps_back_to_full_gain():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=1, ramp_ms=50))
    duck.duck(1.0)
    duck.release(1.1)
    g = gain_curve(duck, 1.0, 200)
    assert g[1:100] == pytest.approx(0.25)
    assert g[100] == pytest.approx(0.25) and g[150:] == pytest.approx(1.0)
    duck.duck(2.0)
    assert gain_curve(duck, 2.0, 5)[1:] == pytest.approx(0.25)  # after a one-sample attack
    duck.clear()  # a stop flushed the player: the next push starts at full gain
    assert gain_curve(duck, 2.005, 5) == pytest.approx(1.0)


def test_a_hot_reload_changes_the_gain_and_the_ramps():
    duck = DuckGain(BargeInDuckConfig())
    duck.apply_config(BargeInDuckConfig(gain=0.5, attack_ms=1))
    duck.duck(0.0)
    assert gain_curve(duck, 0.0, 10) == pytest.approx(0.5)


def test_a_gain_of_one_passes_the_samples_unchanged():
    duck = DuckGain(BargeInDuckConfig(gain=1.0))
    duck.duck(0.0)
    x = np.ones(10, np.float32)
    assert duck.apply(x, RATE, 0.5) is x


def test_a_first_push_after_silence_starts_at_the_level_the_episode_asks_for():
    duck = DuckGain(BargeInDuckConfig(gain=0.25, attack_ms=10))
    duck.duck(1.0)
    assert gain_curve(duck, 1.5, 10) == pytest.approx(0.25)  # nothing before it to click against


def test_the_output_is_float32_and_the_input_is_not_modified():
    duck = DuckGain(BargeInDuckConfig(**FAST))
    duck.duck(0.0)
    x = np.full(20, 0.8, np.float32)
    y = duck.apply(x, RATE, 0.0)
    assert y.dtype == np.float32 and y == pytest.approx(0.2)
    assert x == pytest.approx(0.8)


def test_a_rateless_or_empty_chunk_passes_unchanged():
    duck = DuckGain(BargeInDuckConfig())
    duck.duck(0.0)
    x = np.ones(10, np.float32)
    assert duck.apply(x, 0, 0.5) is x


def test_a_clear_racing_an_apply_waits_and_the_next_push_is_never_scaled(monkeypatch):
    """The listener clears (the robot stopped speaking) while the output thread is inside
    ``apply`` for the last push: the clear lands after it, and any later push is unscaled."""
    duck = DuckGain(BargeInDuckConfig(**FAST))
    duck.duck(0.0)
    inside, go = threading.Event(), threading.Event()
    curve = DuckGain._curve

    def slow_curve(*args):
        inside.set()
        go.wait(2.0)
        return curve(*args)

    monkeypatch.setattr(DuckGain, "_curve", staticmethod(slow_curve))
    out: list[np.ndarray] = []
    worker = threading.Thread(target=lambda: out.append(gain_curve(duck, 0.0, 10)))
    worker.start()
    assert inside.wait(2.0)
    cleared = threading.Event()
    clearer = threading.Thread(target=lambda: (duck.clear(), cleared.set()))
    clearer.start()
    time.sleep(0.05)
    assert not cleared.is_set()  # it waits for the apply in flight
    go.set()
    worker.join(2.0)
    clearer.join(2.0)
    assert cleared.is_set() and out[0] == pytest.approx(0.25)
    monkeypatch.setattr(DuckGain, "_curve", staticmethod(curve))
    assert gain_curve(duck, 0.01, 10) == pytest.approx(1.0)
