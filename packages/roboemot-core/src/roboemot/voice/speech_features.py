"""Prosody features of one utterance for the fast path (SPEC 8 stage 3).

Loudness (RMS in dBFS over the speech frames), pitch (median of the autocorrelation pitch of the
voiced frames) and speaking rate (words per second of speech). Prosody and keywords are the
fast path's whole view of emotion: there is no speech-emotion classifier. Pure functions over
the utterance audio, which stays in RAM and is never logged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config.schema import FastpathConfig

MS_PER_S = 1000.0
DB_PER_DECADE = 20.0
# Level reported for digital silence, so rms_db stays a finite number.
SILENCE_DB = -120.0
# Among the autocorrelation peaks, the shortest lag that reaches this share of the best one is
# the period (the best peak itself is often a multiple of it: an octave error).
PEAK_SHARE = 0.9


@dataclass(frozen=True)
class Prosody:
    """Features of one utterance. ``pitch_hz`` and ``words_per_s`` are None when not measurable."""

    rms_db: float
    pitch_hz: float | None
    words_per_s: float | None
    duration_s: float


def rms_db(samples: np.ndarray) -> float:
    """RMS level in dBFS (full-scale sine is about -3 dB); ``SILENCE_DB`` for silence."""
    if samples.size == 0:
        return SILENCE_DB
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return DB_PER_DECADE * math.log10(rms) if rms > 0 else SILENCE_DB


def frame_pitch_hz(frame: np.ndarray, rate_hz: int, cfg: FastpathConfig) -> float | None:
    """Pitch of one frame by normalised autocorrelation, None when it is not voiced."""
    x = frame.astype(np.float64) - float(np.mean(frame))
    n = x.size
    lo, hi = int(rate_hz / cfg.pitch_max_hz), int(rate_hz / cfg.pitch_min_hz)
    if n <= hi + 1 or lo < 1:
        return None
    spectrum = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum))[: hi + 1]
    if ac[0] <= 0:
        return None
    r = ac[lo - 1 : hi + 2] / ac[0]  # one lag either side, to tell peaks from slopes
    inner = r[1:-1]
    peaks = np.flatnonzero((inner > r[:-2]) & (inner >= r[2:]))
    if peaks.size == 0:
        return None
    best = float(inner[peaks].max())
    if best < cfg.voiced_threshold:
        return None
    first = peaks[np.argmax(inner[peaks] >= PEAK_SHARE * best)]
    return rate_hz / (lo + int(first))


def pitch_hz(samples: np.ndarray, rate_hz: int, cfg: FastpathConfig) -> float | None:
    """Median pitch over the voiced frames of ``samples`` (None when no frame is voiced)."""
    size = max(1, round(cfg.frame_ms * rate_hz / MS_PER_S))
    pitches = []
    for i in range(0, samples.size - size + 1, size):
        frame = samples[i : i + size]
        if rms_db(frame) < cfg.active_floor_db:
            continue
        p = frame_pitch_hz(frame, rate_hz, cfg)
        if p is not None:
            pitches.append(p)
    return float(np.median(pitches)) if pitches else None


def analyse(samples: np.ndarray, rate_hz: int, words: int, cfg: FastpathConfig) -> Prosody:
    """Features of the speech part of an utterance (``words`` from the ASR text)."""
    duration = samples.size / rate_hz
    rate = words / duration if words >= cfg.min_words_for_rate and duration > 0 else None
    return Prosody(rms_db(samples), pitch_hz(samples, rate_hz, cfg), rate, duration)
