"""Seeded synthetic audio for the music detector tests (SPEC 8): generated here, never recorded.

Every generator returns mono float32 at ``rate`` Hz with a low noise floor (``FLOOR_DB``), so a
band with no content still carries random onsets, as in a quiet room.
"""

from __future__ import annotations

import numpy as np

FLOOR_DB = -60.0


def _gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _floor(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n) * _gain(FLOOR_DB)


def _out(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _place(x: np.ndarray, at: int, sound: np.ndarray) -> None:
    end = min(x.size, at + sound.size)
    if at < end:
        x[at:end] += sound[: end - at]


def _decay(n: int, rate: int, tau_s: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (rate * tau_s))


def click_track(bpm: float, seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """A broadband click (5 ms of decaying white noise) on every beat at ``bpm``."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    x = _floor(n, rng)
    length = round(0.005 * rate)
    period = 60.0 / bpm
    for k in range(int(seconds / period) + 1):
        click = rng.standard_normal(length) * _decay(length, rate, 0.0015) * 0.5
        _place(x, round(k * period * rate), click)
    return _out(x)


def drum_loop(bpm: float, seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """Kick (a 120 to 50 Hz sweep, low band) on every beat, a hat (differenced noise, high band)
    on every eighth note and a snare (mid-band noise burst) on beats 2 and 4."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    x = _floor(n, rng)
    beat = 60.0 / bpm
    kick_n = round(0.15 * rate)
    t = np.arange(kick_n) / rate
    freq = 50.0 + 70.0 * np.exp(-t / 0.03)
    kick = np.sin(2 * np.pi * np.cumsum(freq) / rate) * _decay(kick_n, rate, 0.05) * 0.6
    hat_n = round(0.04 * rate)
    snare_n = round(0.12 * rate)
    snare_kernel = np.ones(4) / 4
    for k in range(int(seconds / beat) + 1):
        at = round(k * beat * rate)
        _place(x, at, kick)
        if k % 4 in (1, 3):
            noise = np.convolve(rng.standard_normal(snare_n), snare_kernel, "same")
            _place(x, at, noise * _decay(snare_n, rate, 0.04) * 0.25)
        for half in (0.0, 0.5):
            hat = np.diff(rng.standard_normal(hat_n + 1)) * _decay(hat_n, rate, 0.01) * 0.08
            _place(x, at + round(half * beat * rate), hat)
    return _out(x)


def speech_like(seconds: float, rate: int, pauses: bool, seed: int = 0) -> np.ndarray:
    """Syllables: voiced bursts (a random pitch with harmonics and some noise, under a Hann
    envelope) of random length with random gaps; ``pauses`` adds a longer random pause every few
    syllables, as between phrases."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    x = _floor(n, rng)
    at = 0
    until_pause = rng.integers(3, 9)
    while at < n:
        length = round(rng.uniform(0.08, 0.3) * rate)
        t = np.arange(length) / rate
        f0 = rng.uniform(110.0, 230.0)
        voiced = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h for h in (1, 2, 3, 4))
        noisy = voiced + 0.3 * rng.standard_normal(length)
        _place(x, at, noisy * np.hanning(length) * rng.uniform(0.05, 0.2))
        at += length + round(rng.uniform(0.02, 0.15) * rate)
        until_pause -= 1
        if pauses and until_pause <= 0:
            at += round(rng.uniform(0.3, 1.2) * rate)
            until_pause = rng.integers(3, 9)
    return _out(x)


def fan_noise(seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """Steady low-passed noise at about -30 dBFS, as a fan or an air conditioner."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    noise = np.convolve(rng.standard_normal(n + 16), np.ones(16) / 16, "same")[8 : 8 + n]
    noise *= _gain(-30.0) / float(np.sqrt(np.mean(noise * noise)))
    return _out(noise + _floor(n, rng))


def clock_tick(hz: float, seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """A narrow-band tick ``hz`` times a second: a 3.5 kHz tone with a 3 ms raised-cosine attack
    and a 20 ms decay, so its energy stays in one band."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    x = _floor(n, rng)
    length = round(0.1 * rate)
    t = np.arange(length) / rate
    attack = np.minimum(1.0, t / 0.003)
    envelope = 0.5 - 0.5 * np.cos(np.pi * attack)
    tick = np.sin(2 * np.pi * 3500.0 * t) * envelope * _decay(length, rate, 0.02) * 0.3
    for k in range(int(seconds * hz) + 1):
        _place(x, round(k / hz * rate), tick)
    return _out(x)


def silence(seconds: float, rate: int, seed: int = 0) -> np.ndarray:
    """The noise floor alone: a quiet room."""
    rng = np.random.default_rng(seed)
    return _out(_floor(round(seconds * rate), rng))


def mix(a: np.ndarray, b: np.ndarray, db: float) -> np.ndarray:
    """``a`` plus ``b`` at ``db`` relative gain, trimmed to the shorter one."""
    n = min(a.size, b.size)
    return _out(a[:n] + b[:n] * _gain(db))
