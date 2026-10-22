"""Text-to-speech contracts and the streaming sentence splitter (SPEC 8, stage 7).

A :class:`TTSBackend` turns ONE sentence plus its :class:`~roboemot.voice.prosody.Delivery` into
mono float32 chunks that carry their own sample rate (read from the voice or the provider, never
assumed; the speaker resamples). :class:`SentenceSplitter` cuts the LLM's text deltas into
sentences as soon as each one is complete, so speech starts from the first sentence of the
stream. :class:`FallbackTTS` tries the primary backend and falls back when it cannot start.

Privacy (SPEC 12): nothing here logs a sentence; only lengths, ids and exception types.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config.schema import VoiceConfig
from .prosody import TAG, Delivery

log = logging.getLogger(__name__)

TERMINATORS = ".!?…"
CLOSERS = "\"')]»”’"
TAG_PREFIX = "[mood="
TAG_MAX_CHARS = 40  # longest tag held back while it may still be completing
CUT_AFTER = ",;:"  # preferred places to cut an over-long sentence
# Abbreviations whose trailing dot does not end a sentence (lower case). Words that are also
# ordinary words (no, on, es, st, ed, art, fig, cap, etc) are left out: a wrong join is worse.
ABBREVIATIONS = frozenset(
    {
        "sig", "sigg", "ra", "ssa", "dott", "dr", "prof", "ing", "avv", "geom", "rag", "pag",
        "tel", "ca", "cfr", "vs", "mr", "mrs", "ms", "jr", "sr", "vol", "approx",
    }
)  # fmt: skip
WORD_END = re.compile(r"([^\W\d_]+)$")


class TTSError(RuntimeError):
    """Synthesis failed (worker crash, timeout, HTTP error)."""


class TTSUnavailable(TTSError):
    """The backend cannot be used right now (offline, minor present, route spent, restarting)."""


@dataclass
class AudioChunk:
    """Mono float32 samples in [-1, 1] recorded at ``rate_hz``."""

    samples: np.ndarray
    rate_hz: int


@dataclass(frozen=True)
class Sentence:
    text: str
    mood: str | None = None  # the LLM's tag for it (``mood_source: llm_tagged``)


class TTSBackend(Protocol):
    """One speech engine. ``controls`` names what it renders natively out of ``rate``,
    ``pitch``, ``volume``, ``variability`` and ``instructions`` (SPEC 8, TTS backends)."""

    name: str
    controls: frozenset[str]

    def available(self, *, minor_present: bool = False) -> bool:
        """Whether a request may be started now (minor present, route rules, restart backoff)."""
        ...

    def synthesize(
        self,
        text: str,
        delivery: Delivery,
        *,
        language: str,
        voice: VoiceConfig,
        minor_present: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        """Stream the audio of one sentence; raises :class:`TTSError` on failure."""
        ...

    def cancel(self) -> None:
        """Abandon whatever is in flight; called on the backend's event loop."""
        ...

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        """Load what the first sentence of ``language`` needs (may be a no-op)."""
        ...

    def apply_config(self, voice: VoiceConfig) -> None:
        """Hot reload of the settings the backend keeps itself (may be a no-op)."""
        ...

    async def aclose(self) -> None: ...


class FallbackTTS:
    """Primary backend with an offline fallback (SPEC 8: remote speech falls back to Piper).

    The fallback is used when the primary is not available or fails before its first chunk;
    after audio started, a failure ends the sentence (no switch of voice mid-sentence).
    """

    def __init__(self, primary: TTSBackend, fallback: TTSBackend | None = None) -> None:
        self.primary, self.fallback = primary, fallback
        self.name = primary.name
        self.controls = primary.controls
        self.last_used: str | None = None

    def available(self, *, minor_present: bool = False) -> bool:
        return any(b.available(minor_present=minor_present) for b in self._chain())

    def _chain(self) -> list[TTSBackend]:
        return [b for b in (self.primary, self.fallback) if b is not None]

    def apply_config(self, voice: VoiceConfig) -> None:
        """Hot reload: hand ``voice`` to each backend of the chain."""
        for backend in self._chain():
            backend.apply_config(voice)

    async def synthesize(
        self,
        text: str,
        delivery: Delivery,
        *,
        language: str,
        voice: VoiceConfig,
        minor_present: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        last: TTSError = TTSUnavailable("no TTS backend available")
        for backend in self._chain():
            if not backend.available(minor_present=minor_present):
                continue
            yielded = False
            try:
                async for chunk in backend.synthesize(
                    text, delivery, language=language, voice=voice, minor_present=minor_present
                ):
                    yielded = True
                    self.last_used = backend.name
                    yield chunk
                return
            except TTSError as exc:
                if yielded:
                    raise
                last = exc
                log.warning("tts %s failed before audio: %s", backend.name, type(exc).__name__)
        raise last

    def cancel(self) -> None:
        for backend in self._chain():
            backend.cancel()

    async def warm(self, language: str, voice: VoiceConfig) -> None:
        for backend in self._chain():
            await backend.warm(language, voice)

    async def aclose(self) -> None:
        for backend in self._chain():
            await backend.aclose()


class SentenceSplitter:
    """Cuts streamed LLM text into sentences, each emitted as soon as it is complete.

    A sentence ends at ``. ! ? …`` followed by whitespace (the look-ahead keeps decimals such as
    ``3.5``, initials and the abbreviations in ``ABBREVIATIONS`` whole; a dot before a lower-case
    word does not end one either) or at a newline. Text that reaches ``max_chars`` without an end
    is cut at the last comma, else the last space. With ``strip_tags`` (``mood_source:
    llm_tagged``) ``[mood=x]`` tags, even split across deltas, are removed, start a new sentence
    and set the mood of what follows.
    """

    def __init__(self, max_chars: int, *, strip_tags: bool = False) -> None:
        self.max_chars, self.strip_tags = max_chars, strip_tags
        self._pending = ""
        self._tail = ""  # raw text that may be the start of a tag
        self._mood: str | None = None

    def feed(self, delta: str) -> list[Sentence]:
        """Add a text delta; returns the sentences it completed (possibly none)."""
        out: list[Sentence] = []
        raw = self._tail + delta
        self._tail = ""
        if self.strip_tags:
            pos = 0
            for m in TAG.finditer(raw):
                self._add(raw[pos : m.start()], out)
                self._emit_pending(out)
                self._mood = m.group(1)
                pos = m.end()
            raw = raw[pos:]
            cut = self._partial_tag_start(raw)
            raw, self._tail = raw[:cut], raw[cut:]
        self._add(raw, out)
        return out

    def flush(self) -> list[Sentence]:
        """End of stream: whatever is left, even without a final stop. A tag cut off by the end
        of the stream is dropped."""
        out: list[Sentence] = []
        self._tail = ""
        self._scan(out, final=True)
        return out

    @staticmethod
    def _partial_tag_start(raw: str) -> int:
        i = raw.rfind("[")
        if i < 0:
            return len(raw)
        sub = raw[i:]
        if "]" in sub or len(sub) > TAG_MAX_CHARS:
            return len(raw)
        n = min(len(sub), len(TAG_PREFIX))
        return i if sub[:n] == TAG_PREFIX[:n] else len(raw)

    def _add(self, text: str, out: list[Sentence]) -> None:
        self._pending += text
        self._scan(out, final=False)

    def _emit(self, text: str, out: list[Sentence]) -> None:
        text = text.strip()
        if any(c.isalnum() for c in text):
            out.append(Sentence(text, self._mood))

    def _emit_pending(self, out: list[Sentence]) -> None:
        self._emit(self._pending, out)
        self._pending = ""

    def _scan(self, out: list[Sentence], *, final: bool) -> None:
        buf, i = self._pending, 0
        while i < len(buf):
            c = buf[i]
            if c == "\n":
                self._emit(buf[:i], out)
                buf, i = buf[i + 1 :], 0
                continue
            if c not in TERMINATORS:
                i += 1
                continue
            j = i + 1
            while j < len(buf) and (buf[j] in TERMINATORS or buf[j] in CLOSERS):
                j += 1
            k = j
            while k < len(buf) and buf[k].isspace() and buf[k] != "\n":
                k += 1
            if k >= len(buf):  # the look-ahead is not here yet
                break
            if k > j and not self._dot_continues(buf, i, j, k):
                self._emit(buf[:j], out)
                buf, i = buf[j:], 0
            else:
                i = j
        if final:
            self._emit(buf, out)
            buf = ""
        while len(buf) > self.max_chars:
            cut = self._cut_point(buf)
            self._emit(buf[:cut], out)
            buf = buf[cut:]
        self._pending = buf

    @staticmethod
    def _dot_continues(buf: str, i: int, j: int, k: int) -> bool:
        """A lone dot after an abbreviation or initial, or before a lower-case word."""
        if buf[i] != "." or j != i + 1:
            return False
        word = WORD_END.search(buf[:i])
        if word and (len(word.group(1)) == 1 or word.group(1).lower() in ABBREVIATIONS):
            return True
        return buf[k].islower()

    def _cut_point(self, buf: str) -> int:
        window = buf[: self.max_chars]
        for marks in (CUT_AFTER, " "):
            idx = max(window.rfind(m) for m in marks)
            if idx > 0:
                return idx + 1
        return self.max_chars
