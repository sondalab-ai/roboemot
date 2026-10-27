"""Streaming speech recognition (SPEC 8 stages 3-4).

``ASRBackend`` is one utterance at a time: ``start(language)``, ``accept(samples)`` as audio
arrives (a result whenever the text so far changed), ``finish()`` for the final text, ``reset()``
to drop an utterance. The VAD endpointer decides where an utterance ends, so a backend that
finds a pause by itself (Vosk does) must keep accumulating: its text is everything since
``start``. Text is the user's speech: it is hidden from ``repr`` and never logged.

``VoskASR`` keeps one Vosk model and one free-text recognizer per language, both built by
``load`` (the app does this at start, so no utterance pays for them). Building a recognizer
took 50 ms on a Mac and was 13 % of the listener's samples on the CM4 when ``start`` did it at
every utterance, and a new recognizer decodes about three times slower while its graph cache
fills (SPEC 13); ``start`` now reuses the language's recognizer. Vosk's ``Reset`` alone keeps the
audio of an unfinished utterance (the next one then decodes it as a prefix), so ``finish`` and
``reset`` end the decode with ``FinalResult`` before the ``Reset``: a reused recognizer then
decodes like a new one (checked on the test clips with the real models). ``grammar_words`` decodes a
finished piece of audio with a throwaway recognizer limited to a grammar (the grammar pass,
``roboemot.voice.grammar``): its grammar holds the words of each utterance's free text, so it
cannot be reused, and building it is cheap (the grammar is compiled during the decode).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np

INT16_MAX = 32767
VOSK_QUIET = -1
VOSK_NO_WORD = -1  # vosk_model_find_word: the word is not in the vocabulary
VOSK_UNKNOWN = "[unk]"  # Vosk's grammar token for any word outside the grammar
# A model can run a grammar only with a runtime graph (the small models); a static HCLG graph
# (the big models) makes Vosk ignore the grammar and decode free text.
RUNTIME_GRAPH = ("graph/HCLr.fst", "graph/Gr.fst")


class GrammarUnsupported(Exception):
    """The language's model has no runtime graph, so it cannot decode with a grammar."""


class HeardWord(NamedTuple):
    """One word of a grammar decode: the word, its confidence and its span in the audio (s)."""

    word: str
    conf: float
    start_s: float
    end_s: float


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
        None when the backend cannot tell (no vocabulary, or that model is not loaded). Every
        implementation declares this method; None means "not supported"."""
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

    def decode(self, language: str, samples: np.ndarray) -> str:
        """``samples`` (a whole utterance) decoded on a recognizer of its own, so the
        utterance in progress is not disturbed: the text, empty when nothing was recognised or
        the language is not loaded (the realtime route's fallback, SPEC 8)."""
        ...

    def stats(self) -> dict[str, int]:
        """Recognizer counts for the debug report (SPEC 13); empty when the backend has none."""
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
        self._grammar_graph: dict[str, bool] = {}
        self._recs: dict[str, Any] = {}  # the free-text recognizer of each loaded language
        self._rec: Any = None  # the one of the utterance in progress
        self._dirty = False  # it has audio not yet ended with FinalResult
        self._committed: list[str] = []
        self._last = ""
        self.recognizers_built = 0
        self.built_on_start = 0  # recognizers ``start`` had to build (none after a load)
        self.reset_failures = 0  # recognizers dropped because ending their decode raised
        self.decodes = 0  # whole utterances decoded on their own recognizer (``decode``)

    @property
    def languages(self) -> list[str]:
        """Languages whose model is loaded, in load order."""
        return list(self._models)

    def load(self, languages: Iterable[str]) -> None:
        for language in languages:
            if language not in self._models:
                root = Path(self._paths[language])
                self._grammar_graph[language] = all((root / f).is_file() for f in RUNTIME_GRAPH)
                self._models[language] = self._vosk.Model(str(root))
            if language not in self._recs:
                self._recs[language] = self._build(language)

    def _build(self, language: str) -> Any:
        """A new free-text recognizer for the loaded ``language`` (counted)."""
        self.recognizers_built += 1
        return self._vosk.KaldiRecognizer(self._models[language], self.rate_hz)

    def stats(self) -> dict[str, int]:
        """Recognizer builds for the debug report: all of them, and those on ``start``."""
        return {
            "recognizers_built": self.recognizers_built,
            "built_on_start": self.built_on_start,
            "reset_failures": self.reset_failures,
            "decodes": self.decodes,
        }

    def missing_words(self, language: str, words: list[str]) -> list[str] | None:
        """The ``words`` the ``language`` model's vocabulary lacks (it can never output them);
        None while that model is not loaded."""
        model = self._models.get(language)
        if model is None:
            return None
        return [w for w in words if model.vosk_model_find_word(w) == VOSK_NO_WORD]

    def grammar_words(
        self, language: str, samples: np.ndarray, phrases: list[str]
    ) -> list[HeardWord] | None:
        """``samples`` decoded with a recognizer limited to ``phrases`` (plus the unknown-word
        filler): the words heard with confidence and timing. None while that model is not loaded;
        raises ``GrammarUnsupported`` for a model without a runtime graph. Every phrase must be in
        the vocabulary (Vosk warns, naming the word, on one that is not)."""
        model = self._models.get(language)
        if model is None:
            return None
        if not self._grammar_graph.get(language, False):
            raise GrammarUnsupported(language)
        grammar = json.dumps([*phrases, VOSK_UNKNOWN], ensure_ascii=False)
        rec = self._vosk.KaldiRecognizer(model, self.rate_hz, grammar)
        rec.SetWords(True)
        rec.AcceptWaveform(to_int16(samples))
        words = json.loads(rec.FinalResult()).get("result", [])
        return [
            HeardWord(w["word"], float(w["conf"]), float(w["start"]), float(w["end"]))
            for w in words
        ]

    def decode(self, language: str, samples: np.ndarray) -> str:
        """See ``ASRBackend.decode``: a recognizer built for this call (counted ``decodes``)."""
        model = self._models.get(language)
        if model is None:
            return ""
        self.decodes += 1
        rec = self._vosk.KaldiRecognizer(model, self.rate_hz)
        rec.AcceptWaveform(to_int16(samples))
        return str(json.loads(rec.FinalResult()).get("text", ""))

    def start(self, language: str) -> None:
        """Begin an utterance on the language's recognizer, built by ``load``; one that is
        missing (a model loaded before, a failed decode dropped it) is built here, counted."""
        self.reset()
        if language not in self._models:
            self.load([language])
        rec = self._recs.get(language)
        if rec is None:  # never loaded, or dropped by a failed reset (just above, too)
            self.built_on_start += 1
            rec = self._recs[language] = self._build(language)
        self._rec, self._committed, self._last = rec, [], ""

    def accept(self, samples: np.ndarray) -> ASREvent | None:
        if self._rec is None:
            raise RuntimeError("accept() before start()")
        self._dirty = True
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
        self._dirty = False
        text = " ".join([*self._committed, tail] if tail else self._committed)
        self.reset()
        return text

    def reset(self) -> None:
        """Drop the utterance in progress; the recognizer is kept for the next one. Audio not
        yet ended is ended (and its result dropped) first, since ``Reset`` alone keeps it. A
        recognizer that fails here is dropped and counted (``reset_failures``), never raised:
        the next ``start`` builds a new one."""
        rec, self._rec, self._committed, self._last = self._rec, None, [], ""
        dirty, self._dirty = self._dirty, False
        if rec is None:
            return
        try:
            if dirty:
                rec.FinalResult()
            rec.Reset()
        except Exception:
            self._recs = {k: v for k, v in self._recs.items() if v is not rec}
            self.reset_failures += 1
