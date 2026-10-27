"""Speech fast path (SPEC 8 stage 3): react to a voice before the transcript is final.

Everything here is cheap and runs on the listener thread, ahead of the conversation:

- when speech starts: point the head at the speaker (``SpeakerHint``) and emit ``speech.heard``;
- on every ASR partial: spot keywords (``stop``, ``yes``, ``no``, greetings, the name) and emit
  one appraisal event per category per utterance (``speech.keyword.<category>``,
  ``speech.name_called``), so the reaction starts while the person is still talking, well before
  ``voice.vad.end_silence_ms`` runs out;
- when the utterance ends: prosody (loudness, speaking rate; pitch is reported on the
  utterance), the keywords and wake word of the *final* text (a partial that the final text
  revises away does not count), and, for addressed utterances only, the ``speech.loud`` /
  ``speech.fast`` impulses (a television is not shouting at the robot).

An utterance whose final text comes later (the realtime route's transcript of its audio, SPEC 8,
Realtime speech route) is ``hold``-ed at its end: ``end`` and ``impulses`` then take that held
state, so a newer utterance can begin meanwhile; its keywords, wake word and speaking rate come
from that text.

A short final text with exactly one yes or no keyword, in an addressed utterance, also goes to
``on_answer(category, now)`` (the nod / shake handler, ``roboemot.voice.yes_no``); partials do not.

With a ``grammar`` (``roboemot.voice.grammar``), the end of an utterance also gets a grammar pass
on its audio (``grammar_pass``, run on the grammar worker thread): the categories it hears
confidently join those of the final text in ``end``, before the wake and answer decisions.

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
from .grammar import GrammarSpotter
from .keywords import KeywordSpotter
from .robot_name import RobotName
from .speech_features import Prosody, analyse

MS_PER_S = 1000.0
Emit = Callable[[str, float, dict[str, Any]], None]
EVENT_FOR = {
    "stop": "speech.keyword.stop",
    "yes": "speech.keyword.yes",
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
class GrammarPass:
    """The grammar pass of one finished utterance, fixed on the listener thread with the
    utterance's spotter and language (``found``: the free text's categories); ``run`` it on any
    thread. Text and audio are hidden from repr."""

    grammar: GrammarSpotter
    spotter: KeywordSpotter
    language: str
    audio: np.ndarray = field(repr=False)
    text: str = field(repr=False)
    found: frozenset[str]

    def run(self) -> frozenset[str]:
        return self.grammar.check(self.spotter, self.language, self.audio, self.text, self.found)


@dataclass(frozen=True)
class FastResult:
    """What the end of an utterance adds, from the final text: keyword categories, whether it
    wakes the robot, and prosody."""

    keywords: frozenset[str]
    wake_word: bool
    prosody: Prosody
    answer: str | None = None  # "yes" or "no": the one such keyword of a short final text


@dataclass
class UtteranceState:
    """The fast path's state of one utterance: its language, the events already sent (dedupe
    only), and when the hint and the first keyword came."""

    language: str = ""
    emitted: set[str] = field(default_factory=set)
    hint_at: float | None = None
    keyword_at: float | None = None


class FastPath:
    """Per-utterance state machine: ``begin``, ``partial``\\*, ``end``, ``impulses`` / ``reset``
    (``hold`` keeps an ended utterance's state for a later ``end``)."""

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
        on_answer: Callable[[str, float], None] | None = None,
        grammar: GrammarSpotter | None = None,
    ) -> None:
        self._on_answer = on_answer
        self._grammar = grammar
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
        self.names.configure(cfg)
        self._spotter = KeywordSpotter(cfg.keywords, cfg.wake_word, self.names)
        if self._grammar is not None:
            self._grammar.apply_config(cfg.grammar)

    def wake_out_of_vocab(self) -> dict[str, int]:
        """Per language, the wake phrases the recognizer can never produce (report only)."""
        return self._spotter.wake_out_of_vocab()

    def grammar_snapshot(self) -> dict[str, Any] | None:
        """The grammar pass report (counts only), None without one."""
        return None if self._grammar is None else self._grammar.snapshot()

    def reset(self) -> None:
        """Forget the utterance in progress."""
        self._u = UtteranceState()

    def hold(self) -> UtteranceState:
        """The ended utterance's state, for an ``end`` and ``impulses`` that come later (its
        final text is still to come); the fast path starts afresh."""
        held, self._u = self._u, UtteranceState()
        return held

    def begin(self, start_stamp: float, audio_stamp: float, language: str) -> None:
        """Speech started (``start_stamp``: first speech frame; ``audio_stamp``: end of the audio
        that confirmed it): hint, ``speech.heard``.

        ``hint_lag`` is the processing lag (now minus ``audio_stamp``); ``hint_vs_speech_start``
        adds the ``voice.vad.min_speech_ms`` the VAD needed to be sure.
        """
        self.reset()
        self._u.language = language
        now = self._clock()
        if self._hint(now):
            self._u.hint_at = now
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

    def _fire(self, categories: set[str], u: UtteranceState) -> None:
        """Emit the events of ``categories`` not yet sent for utterance ``u``."""
        events = sorted({EVENT_FOR[c] for c in categories} - u.emitted)
        if not events:
            return
        now = self._clock()
        for event in events:
            self._emit(event, now, {"language": u.language})
        u.emitted.update(events)
        if u.keyword_at is None:
            u.keyword_at = now

    def partial(self, text: str, audio_stamp: float) -> None:
        """React to keywords in the text so far, once per event per utterance. Only the final
        text decides the utterance's keywords and wake word (``end``)."""
        u = self._u
        before = u.keyword_at
        self._fire(self._spotter.spot(text, u.language), u)
        if before is None and u.keyword_at is not None:
            self.stats.record("keyword_lag", u.keyword_at - audio_stamp)

    def grammar_pass(self, text: str, audio: np.ndarray) -> GrammarPass | None:
        """The grammar pass of the utterance that ended with final ``text``, on ``audio``
        (pre-roll to the last speech frame); None without a grammar spotter."""
        if self._grammar is None:
            return None
        spotter = self._spotter  # one spotter for the utterance, whatever a reload does meanwhile
        language = self._u.language
        found = frozenset(spotter.spot(text, language))
        return GrammarPass(self._grammar, spotter, language, audio, text, found)

    def wakes(
        self,
        text: str,
        *,
        grammar_pass: GrammarPass | None = None,
        heard: frozenset[str] = frozenset(),
        state: UtteranceState | None = None,
    ) -> bool:
        """Whether final ``text`` (with what the grammar pass ``heard``) wakes the robot, as
        ``end`` decides it, with no event sent and nothing recorded."""
        u = state if state is not None else self._u
        spotter = grammar_pass.spotter if grammar_pass is not None else self._spotter
        cats = spotter.spot(text, u.language) | heard
        words = len(text.split())
        return "wake" in cats or (
            "name" in cats and words <= self.cfg.fastpath.alias_wake_max_words
        )

    def end(
        self,
        text: str,
        speech_audio: np.ndarray,
        rate_hz: int,
        *,
        last_speech_stamp: float,
        endpoint_stamp: float,
        grammar_pass: GrammarPass | None = None,
        heard: frozenset[str] = frozenset(),
        state: UtteranceState | None = None,
    ) -> FastResult:
        """The utterance ended with final ``text``; ``speech_audio`` spans its speech frames.
        ``heard`` is what its ``grammar_pass`` heard (empty when it was not run or not waited
        for); they join the final text's categories. A yes or no only the grammar pass heard can
        make the answer: today that drives the nod or shake gesture only (SPEC 8 stage 3).

        Keywords only the final text has are emitted now. Records the first reaction (hint or
        keyword) and the first keyword against ``last_speech_stamp``. Call ``impulses`` next.
        ``state`` is a held utterance's (``hold``), else the one in progress.
        """
        u = state if state is not None else self._u
        words = len(text.split())
        spotter = grammar_pass.spotter if grammar_pass is not None else self._spotter
        cats = spotter.spot(text, u.language) | heard
        before = u.keyword_at
        self._fire(cats, u)
        if before is None and u.keyword_at is not None:
            self.stats.record("keyword_lag", u.keyword_at - endpoint_stamp)
        fp = self.cfg.fastpath
        prosody = analyse(speech_audio, rate_hz, words, fp)
        reactions = [t for t in (u.hint_at, u.keyword_at) if t is not None]
        if reactions:
            self.stats.record("reaction_vs_last_speech", min(reactions) - last_speech_stamp)
        if u.keyword_at is not None:
            self.stats.record("keyword_vs_last_speech", u.keyword_at - last_speech_stamp)
        wake = "wake" in cats or ("name" in cats and words <= fp.alias_wake_max_words)
        said = cats & {"yes", "no"}
        answer = next(iter(said)) if len(said) == 1 and words <= fp.gesture_max_words else None
        return FastResult(frozenset(cats), wake, prosody, answer)

    def impulses(
        self, result: FastResult, addressed: bool, state: UtteranceState | None = None
    ) -> None:
        """Emit ``speech.loud`` / ``speech.fast`` for an addressed utterance; ends the utterance
        (``state``: a held one, which leaves the one in progress alone)."""
        fp = self.cfg.fastpath
        p = result.prosody
        u = state if state is not None else self._u
        if addressed:
            now = self._clock()
            if result.answer is not None and self._on_answer is not None:
                self._on_answer(result.answer, now)
            payload = {"language": u.language}
            if p.rms_db >= fp.loud_db:
                self._emit("speech.loud", now, payload)
            if p.words_per_s is not None and p.words_per_s >= fp.fast_words_per_s:
                self._emit("speech.fast", now, payload)
        if state is None:
            self.reset()
