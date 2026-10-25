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

The competitors are what keeps the pass honest. Measured on synthetic clips with the small
Italian model, a grammar of keywords and the filler alone turned ordinary sentences into
confident keywords ("mi piace molto la pizza margherita" -> ``stop`` at confidence 1.0): its
unknown-word filler is weak. With the free-text words in the grammar a keyword wins only where
it fits the audio better than what the free-text recognizer heard. So an utterance whose free
text has no usable word gets no pass.

A phrase counts when its words come out consecutively, each with a confidence of at least
``voice.grammar.wake_min_word_conf`` (wake phrases, the name and aliases) or
``keyword_min_word_conf`` (the others). Hits only add to what the free text found. The phrase
part of the grammar is rebuilt when its inputs change (a config hot reload, the robot's name, a
model that finished loading); the report counts the builds.

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
from .asr import VOSK_UNKNOWN
from .keywords import CATEGORIES, KeywordSpotter, fold, normalise
from .robot_name import MissingWords

log = logging.getLogger(__name__)

MS_PER_S = 1000.0
WAKE_CLASSES = frozenset({"wake", "name"})
SKIPS = ("no_vocabulary", "no_competitors", "no_model", "errors")
# (language, audio, phrases) -> the words heard with their confidence; None: model not loaded.
Decode = Callable[[str, np.ndarray, list[str]], list[tuple[str, float]] | None]
Target = tuple[str, tuple[str, ...]]  # (category, normalised words of one phrase)


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
class _Stats:
    passes: int = 0
    builds: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SKIPS, 0))
    hits: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    rejected: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    rescued: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    missed: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    accepted_conf: _Conf = field(default_factory=_Conf)
    rejected_conf: _Conf = field(default_factory=_Conf)
    last_ms: float = 0.0
    max_ms: float = 0.0
    max_rtf: float = 0.0


class GrammarSpotter:
    """See the module docstring. ``check`` runs on the listener thread; ``snapshot`` anywhere."""

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
        built = self._build(spotter, language)
        if built is None:
            return self._skip("no_vocabulary")
        competitors = self._competitors(language, text, cfg.max_competitor_words)
        if competitors is None:
            return self._skip("no_vocabulary")
        if not competitors:
            return self._skip("no_competitors")
        grammar = list(dict.fromkeys([*built.phrases, *competitors]))
        t0 = self._clock()
        try:
            words = self._decode(language, audio, grammar)
        except Exception as exc:
            log.warning("voice: grammar pass: %s", type(exc).__name__)
            return self._skip("errors")
        spent = self._clock() - t0
        if words is None:
            return self._skip("no_model")
        hits, rejected = self._match(built.targets, words, cfg)
        self._record(spent, audio.size, hits, rejected, set(found))
        return frozenset(hits)

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
                targets.append((category, tuple(normalise(phrase).split())))
        built = _Built(key, tuple(dict.fromkeys(kept)), tuple(targets), oov)
        with self._lock:
            self._built[language] = built
            self._stats.builds += 1
        return built

    def _competitors(self, language: str, text: str, cap: int) -> list[str] | None:
        """The distinct in-vocabulary words of ``text``, at most ``cap``; None when the
        vocabulary cannot be checked."""
        words = list(dict.fromkeys(fold(text).split()))
        absent = self._missing(language, words)
        if absent is None:
            return None
        gone = set(absent)
        return [w for w in words if w not in gone][:cap]

    # ---- matching

    @staticmethod
    def _match(
        targets: tuple[Target, ...], words: list[tuple[str, float]], cfg: GrammarConfig
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Per category, the best phrase confidence (lowest word confidence of one occurrence)
        that passed its threshold, and the best of those that did not."""
        seq = [("" if w == VOSK_UNKNOWN else normalise(w), c) for w, c in words]
        hits: dict[str, float] = {}
        rejected: dict[str, float] = {}
        for category, target in targets:
            n = len(target)
            best = max(
                (
                    min(c for _, c in seq[i : i + n])
                    for i in range(len(seq) - n + 1)
                    if tuple(w for w, _ in seq[i : i + n]) == target
                ),
                default=None,
            )
            if best is None:
                continue
            wake = category in WAKE_CLASSES
            floor = cfg.wake_min_word_conf if wake else cfg.keyword_min_word_conf
            bucket = hits if best >= floor else rejected
            bucket[category] = max(bucket.get(category, 0.0), best)
        for category in hits:
            rejected.pop(category, None)
        return hits, rejected

    # ---- report

    def _skip(self, reason: str) -> frozenset[str]:
        with self._lock:
            self._stats.skipped[reason] += 1
        return frozenset()

    def _record(
        self,
        spent: float,
        samples: int,
        hits: dict[str, float],
        rejected: dict[str, float],
        found: set[str],
    ) -> None:
        with self._lock:
            s = self._stats
            s.passes += 1
            s.last_ms = spent * MS_PER_S
            s.max_ms = max(s.max_ms, s.last_ms)
            if samples:
                s.max_rtf = max(s.max_rtf, spent * self._rate / samples)
            for category, conf in hits.items():
                s.hits[category] += 1
                s.accepted_conf.add(conf)
                if category not in found:
                    s.rescued[category] += 1
            for category, conf in rejected.items():
                s.rejected[category] += 1
                s.rejected_conf.add(conf)
            for category in found - hits.keys():
                if category in s.missed:
                    s.missed[category] += 1
