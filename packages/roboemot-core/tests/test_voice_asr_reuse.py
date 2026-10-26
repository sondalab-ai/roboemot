"""VoskASR reuses one free-text recognizer per language (SPEC 8 stage 3, 13), with a fake vosk."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.voice.asr import VoskASR

RATE = 16000
AUDIO = np.zeros(1600, np.float32)


class FakeRecognizer:
    def __init__(self, log: list[str]):
        self.log = log

    def AcceptWaveform(self, data):  # noqa: N802 (the vosk API)
        self.log.append("accept")
        return False

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": "ciao"})

    def Result(self):  # noqa: N802
        return json.dumps({"text": ""})

    def FinalResult(self):  # noqa: N802
        self.log.append("final")
        return json.dumps({"text": "ciao"})

    def Reset(self):  # noqa: N802
        self.log.append("reset")


class FakeVosk:
    def __init__(self):
        self.built = 0
        self.log: list[str] = []

    def SetLogLevel(self, level):  # noqa: N802
        pass

    def Model(self, path):  # noqa: N802
        return SimpleNamespace(path=path)

    def KaldiRecognizer(self, model, rate, grammar=None):  # noqa: N802
        self.built += 1
        return FakeRecognizer(self.log)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    vosk = FakeVosk()
    monkeypatch.setitem(sys.modules, "vosk", vosk)
    asr = VoskASR({"it": tmp_path / "it", "en": tmp_path / "en"}, RATE)
    return vosk, asr


def test_load_builds_the_recognizer_and_start_reuses_it(fake):
    vosk, asr = fake
    asr.load(["it", "en"])
    assert vosk.built == 2
    for _ in range(3):
        asr.start("it")
        asr.accept(AUDIO)
        assert asr.finish() == "ciao"
    assert vosk.built == 2
    assert asr.stats() == {"recognizers_built": 2, "built_on_start": 0}


def test_finish_and_reset_end_the_decode_before_the_vosk_reset(fake):
    vosk, asr = fake
    asr.load(["it"])
    asr.start("it")
    asr.accept(AUDIO)
    asr.reset()  # an abort: Reset alone would keep this audio for the next utterance
    assert vosk.log == ["accept", "final", "reset"]
    vosk.log.clear()
    asr.start("it")
    asr.reset()  # nothing heard: no decode to end
    assert vosk.log == ["reset"]
    vosk.log.clear()
    asr.start("it")
    asr.accept(AUDIO)
    asr.finish()
    assert vosk.log == ["accept", "final", "reset"]


def test_a_failing_recognizer_is_dropped_and_rebuilt_on_start(fake):
    vosk, asr = fake
    asr.load(["it"])
    asr.start("it")
    asr.accept(AUDIO)
    rec = asr._recs["it"]

    def boom():
        raise RuntimeError("decoder")

    rec.FinalResult = boom
    with pytest.raises(RuntimeError):
        asr.reset()
    asr.start("it")
    assert asr.stats() == {"recognizers_built": 2, "built_on_start": 1}
