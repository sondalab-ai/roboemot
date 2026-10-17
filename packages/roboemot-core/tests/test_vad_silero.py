"""Silero VAD on the real ONNX model (needs onnxruntime and `roboemot models fetch --voice`)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from roboemot.config.schema import VadConfig
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.vad import Endpointer, SileroVAD, SpeechEnd, SpeechStart, VADBackend

pytestmark = pytest.mark.models

RATE = 16000
FRAME = 512
# Formants (centre Hz, bandwidth Hz) of four vowels, enough to fool a speech detector.
VOWELS = [
    ((700, 80), (1220, 90), (2600, 120)),
    ((300, 60), (2300, 90), (3000, 120)),
    ((500, 70), (1000, 90), (2500, 120)),
    ((300, 60), (870, 90), (2250, 120)),
]


def resonate(x: np.ndarray, centre: float, bandwidth: float) -> np.ndarray:
    r = np.exp(-np.pi * bandwidth / RATE)
    a1, a2 = -2 * r * np.cos(2 * np.pi * centre / RATE), r * r
    out = np.zeros_like(x)
    y1 = y2 = 0.0
    for i, v in enumerate(x):
        y = v - a1 * y1 - a2 * y2
        out[i] = y
        y2, y1 = y1, y
    return out


def vowel(duration_s: float, formants) -> np.ndarray:
    n = round(duration_s * RATE)
    t = np.arange(n) / RATE
    phase = np.cumsum(120 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / RATE
    y = (np.diff(np.floor(phase), prepend=0) > 0).astype(np.float64)
    for centre, bandwidth in formants:
        y = resonate(y, centre, bandwidth)
    y *= 1 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


@cache
def speechlike(seconds_of_vowels: int = 3) -> np.ndarray:
    return np.concatenate([vowel(0.18, f) for f in VOWELS * seconds_of_vowels])


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    pytest.importorskip("onnxruntime")
    try:
        path = voice_models.model_file(voice_models.models_dir(), "silero-vad")
    except ModelError:
        pytest.skip("silero-vad not fetched: run `roboemot models fetch --voice`")
    return SileroVAD(path)


def probs(vad: SileroVAD, x: np.ndarray) -> list[float]:
    vad.reset()
    return [vad.speech_prob(x[i : i + FRAME]) for i in range(0, len(x) - FRAME + 1, FRAME)]


def test_is_a_backend_with_v5_frame_size(vad):
    assert isinstance(vad, VADBackend)
    assert (vad.rate_hz, vad.frame_samples) == (16000, 512)


def test_silence_and_noise_are_not_speech(vad):
    rng = np.random.default_rng(0)
    assert max(probs(vad, np.zeros(RATE, np.float32))) < 0.2
    assert max(probs(vad, (0.01 * rng.standard_normal(RATE)).astype(np.float32))) < 0.5


def test_voiced_signal_is_speech_and_state_is_carried(vad):
    p = probs(vad, speechlike())
    assert np.mean(np.array(p[10:]) > 0.5) > 0.9
    again = probs(vad, speechlike())
    assert again == pytest.approx(p)  # reset gives the same scores


def test_wrong_frame_size_is_rejected(vad):
    with pytest.raises(ValueError):
        vad.speech_prob(np.zeros(100, np.float32))


def test_endpointer_on_voiced_burst(vad):
    x = np.concatenate([np.zeros(RATE, np.float32), speechlike(), np.zeros(RATE, np.float32)])
    ep = Endpointer(vad, VadConfig())
    events = []
    for i in range(0, len(x), 1600):
        events += ep.feed(x[i : i + 1600], i / RATE)
    starts = [e for e in events if isinstance(e, SpeechStart)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert len(starts) == len(ends) == 1
    assert 0.9 < starts[0].stamp < 1.4
    speech_end_s = 1 + 12 * 0.18
    assert ends[0].last_speech_stamp == pytest.approx(speech_end_s, abs=0.15)
    assert ends[0].endpoint_stamp - ends[0].last_speech_stamp < 0.6
