"""Grammar pass of wake and keyword spotting (SPEC 8 stage 3).

The small Vosk models get short commands and names wrong in free text: on the robot "ciao neve"
came out as "cadeva", although both words are in the Italian vocabulary. So at the end of each
utterance its audio is decoded a second time by a recognizer limited to a grammar, built at run
time from configuration and the robot's state, never from literals:

- the phrases of ``voice.keywords.<language>`` per category (``KeywordSpotter.grammar_phrases``):
  wake phrases, ``name_wake_templates`` rendered with the robot's name, the bare name and its
  aliases, stop, yes, no and greetings. A phrase with a word the recognizer vocabulary lacks can
  never be heard: it is left out and counted (``out_of_vocab``), never matched;
- the words of the utterance's free-text result, as competitors;
- the recognizer's unknown-word filler.

The competitors are needed: measured on synthetic clips with the small Italian model, a grammar
of keywords and the filler alone turned ordinary sentences into confident keywords ("mi piace
molto la pizza margherita" -> ``stop`` at confidence 1.0), its unknown-word filler is weak. They
are not enough: a multi-word keyword phrase still beats an article and a word, so "i robot sono
utili" came out as ``ehi robot`` at confidence 1.0, and confidence does not tell such a hit from
a true one. So, besides the confidence floor (``voice.grammar.wake_min_word_conf`` for wake
phrases, the name and aliases, ``keyword_min_word_conf`` for the others):

- a phrase counts only when its words come out consecutively *and* adjacent in the audio
  (``max_word_gap_s``, Vosk word timings), so "hello [to the] robot" with the middle words lost
  is not ``hello robot``;
- a wake phrase, the name or an alias the free text missed is added only when the free text holds
  none of its words (it lost them, as "cadeva" for "ciao neve", rather than heard them next to
  another word, as "i robot"); ``rescue_wake`` off adds none. Stop, yes, no and greetings are
  added on the floor and the gap alone.

An utterance gets no pass when its free text has no usable word, more distinct words than
``max_competitor_words`` (cutting the list would leave part of the audio to the keywords alone),
or lasts longer than ``max_audio_s`` (the hand-off to the conversation waits for the pass), or
when the model has no runtime graph (a big static-graph Vosk model ignores grammars). Hits only
add to what the free text found. The phrase part of the grammar is rebuilt when its inputs change
(a config hot reload, the robot's name, a model that finished loading); the report counts builds.

No transcript, phrase or word leaves this module: the report has counts and confidences only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config.schema import GrammarConfig
from .asr import VOSK_UNKNOWN, GrammarUnsupported, HeardWord
from .keywords import CATEGORIES, KeywordSpotter, fold
from .robot_name import MissingWords

log = logging.getLogger(__name__)

MS_PER_S = 1000.0
WAKE_CLASSES = frozenset({"wake", "name"})
SKIPS = (
    "no_vocabulary",
    "no_competitors",
    "too_many_words",
    "too_long",
    "no_model",
    "no_grammar_graph",
    "errors",
)
# (language, audio, phrases) -> the words heard, with confidence and timing; None: model not
# loaded; raises GrammarUnsupported for a model without a runtime graph.
Decode = Callable[[str, np.ndarray, list[str]], list[HeardWord] | None]
Target = tuple[str, tuple[str, ...]]  # (category, folded words of one phrase)


@dataclass(frozen=True)
class _Built:
    """The phrase part of one language's grammar, for one set of inputs."""

    key: tuple[tuple[str, tuple[str, ...]], ...]
    phrases: tuple[str, ...]  # folded, in vocabulary, distinct: the grammar entries
    targets: tuple[Target, ...]
    oov: int


@dataclass
class _Conf:
    """Count, lowest and mean of a stream of confidences."""

    n: int = 0
    total: float = 0.0
    low: float = 1.0

    def add(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.low = min(self.low, value)

    def snapshot(self) -> dict[str, float]:
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "min": self.low, "mean": self.total / self.n}


@dataclass
class _Match:
    """What one decode gave, per category (see ``GrammarSpotter._match``)."""

    hits: dict[str, float] = field(default_factory=dict)
    rejected: dict[str, float] = field(default_factory=dict)
    guarded: set[str] = field(default_factory=set)
    gapped: set[str] = field(default_factory=set)


@dataclass
class _Stats:
    passes: int = 0
    builds: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SKIPS, 0))
    hits: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    rejected: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    rescued: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    missed: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    guarded: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    gapped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    accepted_conf: _Conf = field(default_factory=_Conf)
    rejected_conf: _Conf = field(default_factory=_Conf)
    last_ms: float = 0.0
    max_ms: float = 0.0
    max_rtf: float = 0.0


class GrammarSpotter:
    """See the module docstring. ``check`` runs on one thread at a time (the listener's grammar
    worker, or the listener itself inline); ``snapshot`` anywhere. That thread is the only
    writer of the grammar cache, so it reads it without the lock; the report reads it under the
    lock."""

    def __init__(
        self,
        cfg: GrammarConfig,
        decode: Decode,
        missing: MissingWords,
        rate_hz: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._decode, self._missing, self._rate, self._clock = decode, missing, rate_hz, clock
        self._lock = threading.Lock()
        self._cfg = cfg
        self._built: dict[str, _Built] = {}
        self._stats = _Stats()

    def apply_config(self, cfg: GrammarConfig) -> None:
        """Hot reload of ``voice.grammar``; the phrases follow the keyword spotter."""
        self._cfg = cfg

    def check(
        self,
        spotter: KeywordSpotter,
        language: str,
        audio: np.ndarray,
        text: str,
        found: Iterable[str] = (),
    ) -> frozenset[str]:
        """The categories the grammar pass hears confidently in ``audio`` (``text``: its
        free-text result, ``found``: the categories the free text gave, for the report)."""
        cfg = self._cfg
        if not cfg.enabled:
            return frozenset()
        if audio.size > cfg.max_audio_s * self._rate:
            return self._skip("too_long")
        built = self._build(spotter, language)
        if built is None:
            return self._skip("no_vocabulary")
        free = self._free_words(language, text)
        if free is None:
            return self._skip("no_vocabulary")
        if not free:
            return self._skip("no_competitors")
        if len(free) > cfg.max_competitor_words:
            return self._skip("too_many_words")
        grammar = list(dict.fromkeys([*built.phrases, *free]))
        t0 = self._clock()
        try:
            words = self._decode(language, audio, grammar)
        except GrammarUnsupported:
            return self._skip("no_grammar_graph")
        except Exception as exc:
            # The type only: a recognizer error message can quote grammar words.
            log.warning("voice: grammar pass: %s", type(exc).__name__)
            return self._skip("errors")
        spent = self._clock() - t0
        if words is None:
            return self._skip("no_model")
        found = set(found)
        result = self._match(built.targets, words, set(free), found, cfg)
        self._record(spent, audio.size, result, found)
        return frozenset(result.hits)

    def snapshot(self) -> dict[str, Any]:
        """Counts, confidences and cost for the debug report (no phrase, word or text)."""
        with self._lock:
            s = self._stats
            built = dict(self._built)
            return {
                "enabled": self._cfg.enabled,
                "builds": s.builds,
                "phrases": {lang: len(b.phrases) for lang, b in built.items()},
                "out_of_vocab": {lang: b.oov for lang, b in built.items()},
                "passes": s.passes,
                "skipped": dict(s.skipped),
                "hits": dict(s.hits),
                "rejected": dict(s.rejected),
                "rescued": dict(s.rescued),
                "missed": dict(s.missed),
                "guarded": dict(s.guarded),
                "gapped": dict(s.gapped),
                "confidence": {
                    "accepted": s.accepted_conf.snapshot(),
                    "rejected": s.rejected_conf.snapshot(),
                },
                "cost": {"last_ms": s.last_ms, "max_ms": s.max_ms, "max_rtf": s.max_rtf},
            }

    # ---- building

    def _build(self, spotter: KeywordSpotter, language: str) -> _Built | None:
        """The phrase part for ``language``, rebuilt when the spotter's phrases changed; None
        while the vocabulary of ``language`` cannot be checked (its model is not loaded)."""
        phrases = spotter.grammar_phrases(language)
        key = tuple(sorted(phrases.items()))
        current = self._built.get(language)
        if current is not None and current.key == key:
            return current
        kept: list[str] = []
        targets: list[Target] = []
        oov = 0
        for category, items in key:
            for phrase in items:
                absent = self._missing(language, phrase.split())
                if absent is None:
                    return None
                if absent:
                    oov += 1
                    continue
                kept.append(phrase)
                targets.append((category, tuple(phrase.split())))
        built = _Built(key, tuple(dict.fromkeys(kept)), tuple(targets), oov)
        with self._lock:
            self._built[language] = built
            self._stats.builds += 1
        return built

    def _free_words(self, language: str, text: str) -> list[str] | None:
        """The distinct folded in-vocabulary words of the free text ``text`` (the competitors);
        None when the vocabulary cannot be checked."""
        words = list(dict.fromkeys(fold(text).split()))
        absent = self._missing(language, words)
        if absent is None:
            return None
        gone = set(absent)
        return [w for w in words if w not in gone]

    # ---- matching

    @staticmethod
    def _match(
        targets: tuple[Target, ...],
        words: list[HeardWord],
        free: set[str],
        found: set[str],
        cfg: GrammarConfig,
    ) -> _Match:
        """Per category, the best phrase confidence (lowest word confidence of one occurrence
        whose words are adjacent in the audio) that passed its floor, the best of those that did
        not, and the categories dropped by the gap check or the wake guard (module docstring)."""
        seq = [HeardWord("" if w.word == VOSK_UNKNOWN else fold(w.word), *w[1:]) for w in words]
        out = _Match()
        for category, target in targets:
            n = len(target)
            spans = [
                seq[i : i + n]
                for i in range(len(seq) - n + 1)
                if tuple(w.word for w in seq[i : i + n]) == target
            ]
            if not spans:
                continue
            close = [s for s in spans if _adjacent(s, cfg.max_word_gap_s)]
            if not close:
                out.gapped.add(category)
                continue
            wake = category in WAKE_CLASSES
            if wake and category not in found and not (cfg.rescue_wake and free.isdisjoint(target)):
                out.guarded.add(category)
                continue
            best = max(min(w.conf for w in span) for span in close)
            floor = cfg.wake_min_word_conf if wake else cfg.keyword_min_word_conf
            bucket = out.hits if best >= floor else out.rejected
            bucket[category] = max(bucket.get(category, 0.0), best)
        for category in out.hits:
            out.rejected.pop(category, None)
            out.guarded.discard(category)
            out.gapped.discard(category)
        return out

    # ---- report

    def _skip(self, reason: str) -> frozenset[str]:
        with self._lock:
            self._stats.skipped[reason] += 1
        return frozenset()

    def _record(self, spent: float, samples: int, result: _Match, found: set[str]) -> None:
        with self._lock:
            s = self._stats
            s.passes += 1
            s.last_ms = spent * MS_PER_S
            s.max_ms = max(s.max_ms, s.last_ms)
            if samples:
                s.max_rtf = max(s.max_rtf, spent * self._rate / samples)
            for category, conf in result.hits.items():
                s.hits[category] += 1
                s.accepted_conf.add(conf)
                if category not in found:
                    s.rescued[category] += 1
            for category, conf in result.rejected.items():
                s.rejected[category] += 1
                s.rejected_conf.add(conf)
            for category in result.guarded:
                s.guarded[category] += 1
            for category in result.gapped:
                s.gapped[category] += 1
            for category in found - result.hits.keys():
                if category in s.missed:
                    s.missed[category] += 1


def _adjacent(span: list[HeardWord], max_gap_s: float) -> bool:
    """Whether each word of ``span`` starts at most ``max_gap_s`` after the previous one ends."""
    return all(b.start_s - a.end_s <= max_gap_s for a, b in zip(span, span[1:], strict=False))
