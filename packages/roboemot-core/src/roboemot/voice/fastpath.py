"""Speech fast path (SPEC 8 stage 3): react to a voice before the transcript is final.

Everything here is cheap and runs on the listener thread, ahead of the conversation:

- when speech starts: point the head at the speaker (``SpeakerHint``) and emit ``speech.heard``;
- on every ASR partial: spot keywords (``stop``, ``no``, greetings, the robot's name) and emit
  one appraisal event per category per utterance (``speech.keyword.<category>``,
  ``speech.name_called``), so the reaction starts while the person is still talking, well before
  ``voice.vad.end_silence_ms`` runs out;
- when the utterance ends: prosody (loudness, speaking rate; pitch is reported on the
  utterance), the keywords and wake word of the *final* text (a partial that the final text
  revises away does not count), and, for addressed utterances only, the ``speech.loud`` /
  ``speech.fast`` impulses (a television is not shouting at the robot).

Events go to ``emit(event, now, payload)`` (the affect runner). Payloads carry the language only:
never audio or text. Latencies (SPEC 13) are counted from the start of the last speech frame:
negative means the reaction came before the person had finished speaking.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..attention import SpeakerHint
from ..config.schema import VoiceConfig
from .keywords import KeywordSpotter
from .robot_name import RobotName
from .speech_features import Prosody, analyse

MS_PER_S = 1000.0
Emit = Callable[[str, float, dict[str, Any]], None]
EVENT_FOR = {
    "stop": "speech.keyword.stop",
    "no": "speech.keyword.no",
    "greeting": "speech.keyword.greeting",
    "name": "speech.name_called",
    "wake": "speech.name_called",
}


class HintSink(Protocol):
    """The attention selector's side of the speaker hint."""

    def hint_speaker(self, hint: SpeakerHint) -> None: ...


@dataclass
class LatencyStats:
    """Count, last and worst value per named latency, in seconds; ``snapshot`` gives ms."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _n: dict[str, int] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)
    _max: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            self._n[name] = self._n.get(name, 0) + 1
            self._last[name] = seconds
            self._max[name] = max(self._max.get(name, seconds), seconds)

    def snapshot(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                name: {
                    "n": float(self._n[name]),
                    "last_ms": self._last[name] * MS_PER_S,
                    "max_ms": self._max[name] * MS_PER_S,
                }
                for name in self._n
            }


@dataclass(frozen=True)
class FastResult:
    """What the end of an utterance adds, from the final text: keyword categories, whether it
    wakes the robot, and prosody."""

    keywords: frozenset[str]
    wake_word: bool
    prosody: Prosody


class FastPath:
    """Per-utterance state machine: ``begin``, ``partial``\\*, ``end``, ``impulses`` / ``reset``."""

    def __init__(
        self,
        cfg: VoiceConfig,
        emit: Emit,
        selector: HintSink | None = None,
        *,
        bearing: Callable[[float], float | None] = lambda now: None,
        target_entry: Callable[[], str | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
        names: RobotName | None = None,
    ) -> None:
        self.names = names if names is not None else RobotName()
        self._emit = emit
        self._selector = selector
        self._bearing = bearing
        self._target_entry = target_entry
        self._clock = clock
        self.stats = LatencyStats()
        self.apply_config(cfg)
        self.reset()

    def apply_config(self, cfg: VoiceConfig) -> None:
        """Hot reload of keyword lists, wake word and prosody thresholds."""
        self.cfg = cfg
        self._spotter = KeywordSpotter(cfg.keywords, cfg.wake_word, self.names)

    def reset(self) -> None:
        """Forget the utterance in progress."""
        self._language = ""
        self._emitted: set[str] = set()  # events already sent this utterance (dedupe only)
        self._hint_at: float | None = None
        self._keyword_at: float | None = None

    def begin(self, start_stamp: float, audio_stamp: float, language: str) -> None:
        """Speech started (``start_stamp``: first speech frame; ``audio_stamp``: end of the audio
        that confirmed it): hint, ``speech.heard``.

        ``hint_lag`` is the processing lag (now minus ``audio_stamp``); ``hint_vs_speech_start``
        adds the ``voice.vad.min_speech_ms`` the VAD needed to be sure.
        """
        self.reset()
        self._language = language
        now = self._clock()
        if self._hint(now):
            self._hint_at = now
            self.stats.record("hint_lag", now - audio_stamp)
            self.stats.record("hint_vs_speech_start", now - start_stamp)
        self._emit("speech.heard", now, {"language": language})

    def _hint(self, now: float) -> bool:
        """Turn toward the voice: the sound bearing if a recent one exists, else the person the
        robot already attends to, else nothing (a hint needs a bearing or an entry)."""
        if self._selector is None:
            return False
        az = self._bearing(now)
        entry = None if az is not None else self._target_entry()
        if az is None and entry is None:
            return False
        self._selector.hint_speaker(SpeakerHint(now, entry_id=entry, azimuth_rad=az))
        return True

    def _fire(self, categories: set[str]) -> None:
        """Emit the events of ``categories`` not yet sent this utterance."""
        events = sorted({EVENT_FOR[c] for c in categories} - self._emitted)
        if not events:
            return
        now = self._clock()
        for event in events:
            self._emit(event, now, {"language": self._language})
        self._emitted.update(events)
        if self._keyword_at is None:
            self._keyword_at = now

    def partial(self, text: str, audio_stamp: float) -> None:
        """React to keywords in the text so far, once per event per utterance. Only the final
        text decides the utterance's keywords and wake word (``end``)."""
        before = self._keyword_at
        self._fire(self._spotter.spot(text, self._language))
        if before is None and self._keyword_at is not None:
            self.stats.record("keyword_lag", self._keyword_at - audio_stamp)

    def end(
        self,
        text: str,
        speech_audio: np.ndarray,
        rate_hz: int,
        *,
        last_speech_stamp: float,
        endpoint_stamp: float,
    ) -> FastResult:
        """The utterance ended with final ``text``; ``speech_audio`` spans its speech frames.

        Keywords only the final text has are emitted now. Records the first reaction (hint or
        keyword) and the first keyword against ``last_speech_stamp``. Call ``impulses`` next.
        """
        words = len(text.split())
        cats = self._spotter.spot(text, self._language)
        before = self._keyword_at
        self._fire(cats)
        if before is None and self._keyword_at is not None:
            self.stats.record("keyword_lag", self._keyword_at - endpoint_stamp)
        fp = self.cfg.fastpath
        prosody = analyse(speech_audio, rate_hz, words, fp)
        reactions = [t for t in (self._hint_at, self._keyword_at) if t is not None]
        if reactions:
            self.stats.record("reaction_vs_last_speech", min(reactions) - last_speech_stamp)
        if self._keyword_at is not None:
            self.stats.record("keyword_vs_last_speech", self._keyword_at - last_speech_stamp)
        wake = "wake" in cats or ("name" in cats and words <= fp.alias_wake_max_words)
        return FastResult(frozenset(cats), wake, prosody)

    def impulses(self, result: FastResult, addressed: bool) -> None:
        """Emit ``speech.loud`` / ``speech.fast`` for an addressed utterance; ends the utterance."""
        fp = self.cfg.fastpath
        p = result.prosody
        if addressed:
            now = self._clock()
            payload = {"language": self._language}
            if p.rms_db >= fp.loud_db:
                self._emit("speech.loud", now, payload)
            if p.words_per_s is not None and p.words_per_s >= fp.fast_words_per_s:
                self._emit("speech.fast", now, payload)
        self.reset()
