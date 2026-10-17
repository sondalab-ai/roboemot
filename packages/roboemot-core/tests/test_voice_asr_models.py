"""Vosk + Silero on synthesized speech clips (needs the voice extra and fetched voice models).

The clips are macOS `say` output (Alice it_IT, Samantha en_US), 16 kHz mono, under
tests/data/voice. Latencies use the audio clock plus the measured compute of each chunk, so they
are what a real-time stream would see, not the speed of a file read.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import ASREvent, VoskASR
from roboemot.voice.fastpath import FastPath
from roboemot.voice.listener import VoiceListener
from roboemot.voice.vad import Endpointer, SileroVAD

pytestmark = pytest.mark.models

DATA = Path(__file__).parent / "data" / "voice"
RATE = 16000
CHUNK = 512
LEAD_S = 0.5
CFG = RoboEmotConfig()


def load(name: str) -> np.ndarray:
    with wave.open(str(DATA / f"{name}.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (RATE, 1)
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (pcm / 32768.0).astype(np.float32)


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("vosk")
    root = voice_models.models_dir()
    try:
        vad = voice_models.model_file(root, "silero-vad")
        asr = {
            lang: voice_models.model_file(root, name) for lang, name in CFG.voice.asr.models.items()
        }
    except ModelError:
        pytest.skip("voice models not fetched: run `roboemot models fetch --voice`")
    return vad, asr


@pytest.fixture(scope="module")
def vosk_asr(models):
    asr = VoskASR(models[1], RATE)
    asr.load(models[1])
    return asr


def run(models, vosk_asr, clip: str, language: str):
    """Play LEAD silence, the clip and 1.5 s of silence through the listener in real time."""
    vad_path, _ = models
    audio = np.concatenate([np.zeros(round(LEAD_S * RATE), np.float32), load(clip)])
    audio = np.concatenate([audio, np.zeros(round(1.5 * RATE), np.float32)])
    state = SimpleNamespace(end=0.0, began=time.perf_counter())
    order, utterances = [], []
    fast = FastPath(
        CFG.voice,
        lambda name, now, payload: order.append((name, now)),
        None,
        clock=lambda: state.end + (time.perf_counter() - state.began),
    )
    listener = VoiceListener(
        CFG,
        lambda t: None,
        Endpointer(SileroVAD(vad_path, 1), CFG.voice.vad),
        vosk_asr,
        SpeechActivity(),
        fast,
        utterances.append,
        speaker_language=lambda: language,
        clock=lambda: state.end + (time.perf_counter() - state.began),
    )
    for i in range(0, audio.size, CHUNK):
        part = audio[i : i + CHUNK]
        state.end, state.began = (i + part.size) / RATE, time.perf_counter()
        listener.process(SimpleNamespace(samples=part, rate_hz=RATE, stamp=state.end, gap=False))
    return utterances, order, listener.snapshot()["latency"]


CASES = [
    # clip, language, words in the transcript, events before the endpoint
    ("it_greeting_name", "it", {"ciao"}, {"speech.keyword.greeting", "speech.name_called"}),
    ("it_stop", "it", {"fermati"}, {"speech.keyword.stop"}),
    ("en_hello_name", "en", {"hello"}, {"speech.keyword.greeting", "speech.name_called"}),
    ("en_stop", "en", {"stop"}, {"speech.keyword.stop"}),
]


@pytest.mark.parametrize("clip,language,words,events", CASES)
def test_asr_transcript_and_fast_path_before_the_endpoint(
    models, vosk_asr, clip, language, words, events, capsys
):
    utterances, order, latency = run(models, vosk_asr, clip, language)
    assert len(utterances) == 1
    u = utterances[0]
    assert words <= set(u.text.split()), "transcript lost a key word"
    assert u.language == language and u.final
    fired = {name for name, _ in order}
    assert events <= fired
    end_silence_s = CFG.voice.vad.end_silence_ms / 1000
    for name, at in order:
        if name in events:
            assert at < u.last_speech_stamp + end_silence_s, f"{name} came after the end silence"
    kw = latency["keyword_vs_last_speech"]["last_ms"]
    assert kw < CFG.voice.vad.end_silence_ms
    # SPEC 13: end of speech to final transcript, within the end-silence plus ASR finish
    assert latency["final_vs_last_speech"]["last_ms"] < CFG.voice.vad.end_silence_ms + 300
    with capsys.disabled():
        print(
            f"\n[latency] {clip}: keyword {kw:+.0f} ms, "
            f"reaction {latency['reaction_vs_last_speech']['last_ms']:+.0f} ms, "
            f"final {latency['final_vs_last_speech']['last_ms']:+.0f} ms vs last speech"
        )


def test_asr_wake_phrases_are_in_the_vocabulary(vosk_asr, models):
    """`reachy` is out of vocabulary in both small models, so the wake phrases use common words."""
    vosk_asr.load(CFG.voice.asr.models)
    for language, ks in CFG.voice.keywords.items():
        model = vosk_asr._models[language]
        for phrase in ks.wake_phrases:
            for word in phrase.split():
                assert model.vosk_model_find_word(word) >= 0, f"{language}: {word!r} not in vocab"
        assert model.vosk_model_find_word(CFG.voice.wake_word) == -1


def test_asr_wake_word_alias_addresses_a_stranger(models, vosk_asr):
    (u,), _, _ = run(models, vosk_asr, "it_greeting_name", "it")
    assert u.wake_word and u.addressed
    (u,), _, _ = run(models, vosk_asr, "it_stop", "it")
    assert not u.wake_word and not u.addressed


def test_asr_prosody_of_a_real_voice(models, vosk_asr):
    (u,), _, _ = run(models, vosk_asr, "it_stop", "it")
    assert -40 < u.prosody.rms_db < -5
    assert u.prosody.pitch_hz is None or 60 < u.prosody.pitch_hz < 450


def test_asr_backend_accumulates_across_internal_pauses(models, vosk_asr):
    audio = load("it_stop")
    vosk_asr.start("it")
    last = None
    for i in range(0, audio.size, CHUNK):
        ev = vosk_asr.accept(audio[i : i + CHUNK])
        last = ev or last
    assert isinstance(last, ASREvent) and "fermati" in last.text
    assert "fermati" in vosk_asr.finish()
    assert vosk_asr.finish() == ""


def test_asr_languages_are_separate_models(vosk_asr):
    vosk_asr.start("en")
    vosk_asr.accept(load("en_stop"))
    assert "stop" in vosk_asr.finish()
    vosk_asr.start("it")
    vosk_asr.accept(load("it_stop"))
    assert "fermati" in vosk_asr.finish()
