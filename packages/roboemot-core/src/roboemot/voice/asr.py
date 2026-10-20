"""Streaming speech recognition (SPEC 8 stages 3-4).

``ASRBackend`` is one utterance at a time: ``start(language)``, ``accept(samples)`` as audio
arrives (a result whenever the text so far changed), ``finish()`` for the final text, ``reset()``
to drop an utterance. The VAD endpointer decides where an utterance ends, so a backend that
finds a pause by itself (Vosk does) must keep accumulating: its text is everything since
``start``. Text is the user's speech: it is hidden from ``repr`` and never logged.

``VoskASR`` keeps one Vosk model per language, loaded by ``load`` (the app does this at start so
the first utterance does not pay for it) and one recognizer at a time, created by ``start``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

INT16_MAX = 32767
VOSK_QUIET = -1
VOSK_NO_WORD = -1  # vosk_model_find_word: the word is not in the vocabulary


@dataclass(frozen=True)
class ASREvent:
    """The text of the running utterance so far (``final``: the backend will not revise it)."""

    text: str = field(repr=False)
    final: bool = False


@runtime_checkable
class ASRBackend(Protocol):
    """One utterance at a time; mono float32 audio at ``rate_hz``."""

    rate_hz: int

    @property
    def languages(self) -> list[str]:
        """Languages whose model is loaded now (what ``start`` can serve without loading)."""
        ...

    def load(self, languages: Iterable[str]) -> None:
        """Load the models of ``languages`` now (raises when one is missing)."""
        ...

    def missing_words(self, language: str, words: list[str]) -> list[str] | None:
        """The ``words`` the ``language`` model's vocabulary lacks (it can never output them);
        None when the backend cannot tell (no vocabulary, or that model is not loaded)."""
        ...

    def start(self, language: str) -> None:
        """Begin an utterance in ``language``, dropping any one in progress."""
        ...

    def accept(self, samples: np.ndarray) -> ASREvent | None:
        """Feed audio; the text so far if it changed since the last call, else None."""
        ...

    def finish(self) -> str:
        """End the utterance and return its full text (empty when nothing was recognised)."""
        ...

    def reset(self) -> None:
        """Drop the utterance in progress without a result."""
        ...


def resolve_language(
    available: Iterable[str], speaker_language: str | None, default_language: str
) -> str:
    """The ASR language: the speaker's if a model exists for it, else ``default_language`` (the
    ``conversation.language``) if it has one, else the first available. Raises ``ValueError``
    with no models."""
    langs = list(available)
    if not langs:
        raise ValueError("no ASR model configured")
    for candidate in (speaker_language, default_language):
        if candidate in langs:
            return candidate
    return langs[0]


def to_int16(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] as little-endian 16-bit PCM, clipped."""
    return (np.clip(samples, -1.0, 1.0) * INT16_MAX).astype("<i2").tobytes()


class VoskASR:
    """Vosk (Kaldi) streaming recognizer. Needs the ``voice`` extra; word timings are off."""

    def __init__(self, model_paths: Mapping[str, Path], rate_hz: int) -> None:
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("VoskASR needs the `voice` extra (vosk)") from exc
        vosk.SetLogLevel(VOSK_QUIET)
        self._vosk = vosk
        self.rate_hz = rate_hz
        self._paths = dict(model_paths)
        self._models: dict[str, Any] = {}
        self._rec: Any = None
        self._committed: list[str] = []
        self._last = ""

    @property
    def languages(self) -> list[str]:
        """Languages whose model is loaded, in load order."""
        return list(self._models)

    def load(self, languages: Iterable[str]) -> None:
        for language in languages:
            if language not in self._models:
                self._models[language] = self._vosk.Model(str(self._paths[language]))

    def missing_words(self, language: str, words: list[str]) -> list[str] | None:
        """The ``words`` the ``language`` model's vocabulary lacks (it can never output them);
        None while that model is not loaded."""
        model = self._models.get(language)
        if model is None:
            return None
        return [w for w in words if model.vosk_model_find_word(w) == VOSK_NO_WORD]

    def start(self, language: str) -> None:
        self.load([language])
        self._rec = self._vosk.KaldiRecognizer(self._models[language], self.rate_hz)
        self._committed, self._last = [], ""

    def accept(self, samples: np.ndarray) -> ASREvent | None:
        if self._rec is None:
            raise RuntimeError("accept() before start()")
        if self._rec.AcceptWaveform(to_int16(samples)):
            text = json.loads(self._rec.Result()).get("text", "")
            if text:
                self._committed.append(text)
            partial = ""
        else:
            partial = json.loads(self._rec.PartialResult()).get("partial", "")
        return self._changed(partial)

    def _changed(self, partial: str) -> ASREvent | None:
        text = " ".join([*self._committed, partial] if partial else self._committed)
        if text == self._last:
            return None
        self._last = text
        return ASREvent(text)

    def finish(self) -> str:
        if self._rec is None:
            return ""
        tail = json.loads(self._rec.FinalResult()).get("text", "")
        text = " ".join([*self._committed, tail] if tail else self._committed)
        self.reset()
        return text

    def reset(self) -> None:
        self._rec, self._committed, self._last = None, [], ""
