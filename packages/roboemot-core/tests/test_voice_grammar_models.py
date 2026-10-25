"""Grammar pass with the real small Vosk models (SPEC 8 stage 3; needs the voice extra and
fetched voice models).

Clips are macOS `say` output (Alice it_IT, Samantha en_US), 16 kHz mono, under tests/data/voice;
noise is seeded white noise at a fixed SNR, so a run is deterministic on one machine. The
on-robot failure ("ciao neve" heard as "cadeva") does not reproduce on clean synthetic audio:
these clips show the same kind of miss (a wake phrase the free text lost) and the guards that
keep ordinary speech from turning into keywords.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest
from reachy_config import reachy_config
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.asr import VoskASR
from roboemot.voice.grammar import GrammarSpotter
from roboemot.voice.keywords import KeywordSpotter
from roboemot.voice.robot_name import RobotName

pytestmark = pytest.mark.models

DATA = Path(__file__).parent / "data" / "voice"
RATE = 16000
CFG = reachy_config()
SEEDS = (0, 1, 2)
SNR_DB = 5.0
RTF_MAX = 0.5  # the grammar pass must stay well inside real time even on a slow core


def load(name: str) -> np.ndarray:
    with wave.open(str(DATA / f"{name}.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (RATE, 1)
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    return (pcm / 32768.0).astype(np.float32)


def noisy(x: np.ndarray, seed: int, snr_db: float = SNR_DB) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=x.size).astype(np.float32)
    noise *= np.sqrt(np.mean(x**2) / np.mean(noise**2) / 10 ** (snr_db / 10))
    return x + noise


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("vosk")
    root = voice_models.models_dir()
    try:
        paths = {
            lang: voice_models.model_file(root, name) for lang, name in CFG.voice.asr.models.items()
        }
    except ModelError:
        pytest.skip("voice models not fetched: run `roboemot models fetch --voice`")
    asr = VoskASR(paths, RATE)
    asr.load(paths)
    names = RobotName(None, asr.missing_words)
    names.configure(CFG.voice)
    spotter = KeywordSpotter(CFG.voice.keywords, None, names)
    grammar = GrammarSpotter(CFG.voice.grammar, asr.grammar_words, asr.missing_words, RATE)
    return asr, spotter, grammar


def run(rig, audio: np.ndarray, language: str):
    """(free-text categories, categories after the grammar pass, pass seconds)."""
    asr, spotter, grammar = rig
    asr.start(language)
    asr.accept(audio)
    text = asr.finish()
    free = spotter.spot(text, language)
    t0 = time.perf_counter()
    both = free | grammar.check(spotter, language, audio, text, free)
    return free, both, time.perf_counter() - t0


def test_grammar_recovers_a_wake_phrase_the_free_text_lost(rig):
    free, both, _ = run(rig, load("it_ehi_robot_luce"), "it")  # free text: "i robot ..."
    assert "wake" not in free and "wake" in both


def test_grammar_recovers_wake_phrases_in_noise(rig):
    rescued = 0
    for seed in SEEDS:
        free, both, _ = run(rig, noisy(load("en_hello_robot"), seed), "en")
        assert free <= both
        rescued += "wake" in both and "wake" not in free
    assert rescued >= 1


@pytest.mark.parametrize(
    "clip,language",
    [("it_pizza", "it"), ("robot_it_stelle", "it"), ("robot_en_sunsets", "en")],
)
def test_ordinary_speech_does_not_become_stop_or_wake(rig, clip, language):
    for audio in (load(clip), *(noisy(load(clip), seed) for seed in SEEDS)):
        _, both, _ = run(rig, audio, language)
        assert not both & {"stop", "wake"}


def test_grammar_pass_cost_per_utterance(rig):
    clips = [("it_ciao_robot", "it"), ("it_stop", "it"), ("en_hello_robot", "en")]
    worst = 0.0
    for clip, language in clips:
        audio = load(clip)
        _, _, spent = run(rig, audio, language)
        worst = max(worst, spent * RATE / audio.size)
    print(f"grammar pass: worst real-time factor {worst:.4f}")
    assert worst < RTF_MAX
    assert rig[2].snapshot()["out_of_vocab"] == {"it": 0, "en": 0}
