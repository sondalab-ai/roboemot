"""Prosody features on synthetic signals (SPEC 8 stage 3)."""

import numpy as np
import pytest
from roboemot.config.schema import FastpathConfig
from roboemot.voice.speech_features import SILENCE_DB, analyse, pitch_hz, rms_db

RATE = 16000
CFG = FastpathConfig()


def tone(freq: float, amp: float, seconds: float = 0.5, harmonics: int = 1) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    y = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))
    return (amp * y).astype(np.float32)


def test_rms_db_of_a_sine_and_of_silence():
    assert rms_db(tone(200, 0.1, harmonics=1)) == pytest.approx(-23.0, abs=0.2)
    assert rms_db(np.zeros(100, np.float32)) == SILENCE_DB
    assert rms_db(np.empty(0, np.float32)) == SILENCE_DB


def test_louder_tone_reads_louder():
    assert rms_db(tone(200, 0.4)) - rms_db(tone(200, 0.1)) == pytest.approx(12.0, abs=0.2)


@pytest.mark.parametrize("freq", [110, 180, 220, 300])
def test_pitch_of_pure_and_harmonic_tones(freq):
    for harmonics in (1, 4):
        got = pitch_hz(tone(freq, 0.2, harmonics=harmonics), RATE, CFG)
        assert got == pytest.approx(freq, rel=0.03)


def test_noise_and_quiet_signals_have_no_pitch():
    rng = np.random.default_rng(0)
    assert pitch_hz(rng.normal(0, 0.1, RATE).astype(np.float32), RATE, CFG) is None
    assert pitch_hz(tone(200, 1e-4), RATE, CFG) is None  # below active_floor_db


def test_pitch_outside_the_range_is_not_reported():
    assert pitch_hz(tone(30, 0.2, seconds=1.0), RATE, CFG) is None


def test_speaking_rate_needs_enough_words():
    audio = tone(200, 0.1, seconds=2.0)
    assert analyse(audio, RATE, 8, CFG).words_per_s == pytest.approx(4.0)
    assert analyse(audio, RATE, 2, CFG).words_per_s is None
    assert analyse(audio, RATE, 8, CFG).duration_s == pytest.approx(2.0)
