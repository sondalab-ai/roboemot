"""Speech timeline: what the robot is about to say, as scalars on the clock (SPEC 8; design ZT).

:class:`~roboemot.voice.speech.SpeechOutput` writes it at push time, so the timeline is known
before it is heard (audio is pushed ahead of playback). Per sentence it keeps:

- energy frames of ``speech_motion.envelope_frame_ms``, as dBFS, stamped on the playback clock
  (the first frame starts where the sink's queue ended when the sentence's first chunk was
  pushed). Chunk boundaries do not cut frames: the rest of a chunk is carried to the next one as
  a sum of squares and a count, never as samples;
- the sentence's start and end, the start of the pause pushed before it, its question flag (from
  that piece's own text), its mood label and whether it continues a sentence the splitter cut;
- events: ``sentence_start``, ``sentence_end``, ``phrase_gap`` (between two voiced frames of one
  sentence, at least ``phrase_gap_ms`` with no voiced frame, stamped where the last voiced frame
  ended; leading and trailing silence never count) and ``emphasis`` (a local maximum at least
  ``emphasis_db`` above the running median of the sentence's voiced frames so far, once the
  sentence has ``emphasis_warmup_ms`` of voiced audio, at most once per
  ``emphasis_floor_interval_s`` across sentences).

A phrase gap is written when the voice resumes: with whole-sentence pushes (Piper) that is ahead of
playback, but a streamed backend may write it at or after its ``at_s``, so a reader must accept
events stamped in its past.

A barge-in truncates it at the stop: nothing at or after the cut remains, and a phrase gap the
cut ends early is dropped (the speech after it never plays). It never holds text or samples
(SPEC 12); the debug report gets counts only. Entries older than ``history_s`` are dropped.

Lock order: the writer is called under the speech output's lock and takes this one inside it;
readers (the app tick) take only this one.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ..config.schema import SpeechMotionConfig

MS_PER_S = 1000.0
PHRASE_UTTERANCES_MAX = 64  # answers whose phrase-gap count is remembered (not by time)
POWER_DB = 10.0  # dB per decade of a mean square
FLOOR_DBFS = -120.0  # the level of digital silence (log of zero)

EventKind = Literal["sentence_start", "sentence_end", "phrase_gap", "emphasis"]
EVENT_KINDS: tuple[EventKind, ...] = ("sentence_start", "sentence_end", "phrase_gap", "emphasis")


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    """One event on the playback clock. ``until_s`` is where a phrase gap ends (the next voiced
    frame); it equals ``at_s`` for the other kinds. ``index`` counts the phrase gaps of the
    utterance (0 for the first), so a choice seeded by utterance and index repeats. ``dbfs`` is the
    level of an emphasis peak (``FLOOR_DBFS`` for the other kinds)."""

    kind: EventKind
    at_s: float
    until_s: float
    sentence: int
    utterance: int
    index: int = 0
    dbfs: float = FLOOR_DBFS


@dataclass(slots=True)
class SpeechSentence:
    """One sentence (or piece of a sentence the splitter cut) on the playback clock."""

    id: int
    utterance: int
    start_s: float
    end_s: float
    question: bool
    mood: str | None
    continued: bool
    pad_start_s: float | None = None
    cut: bool = False


@dataclass(frozen=True, slots=True)
class TimelineView:
    """One consistent read of the timeline for one reader tick, taken under one lock
    acquisition (copy-on-read): a truncation or push cannot land between its parts.

    ``sentence`` and ``level`` are at ``t``; ``frames`` start in ``[frames_from, t)``; ``events``
    are stamped in ``[events_from, events_to)``; ``sentences`` holds copies of the sentences
    those events name and of ``sentence``; ``sentence_dbfs`` is the level of every frame of
    ``sentence`` when the reader asked for it (None otherwise)."""

    t: float
    last_cut: float | None
    sentence: SpeechSentence | None
    level: float | None
    frames: tuple[tuple[float, float], ...]
    events: tuple[SpeechEvent, ...]
    sentences: dict[int, SpeechSentence]
    sentence_dbfs: tuple[float, ...] | None = None


@dataclass(slots=True)
class _Writer:
    """State of the sentence being written: scalars and the sorted voiced energies only."""

    sentence: SpeechSentence
    rate_hz: int
    frame_n: int
    frame_s: float
    carry_t0: float = 0.0
    carry_sq: float = 0.0
    carry_n: int = 0
    last_voiced_end: float | None = None
    voiced: list[float] | None = None
    cand_at: float | None = None  # the last frame, waiting for the next to be judged a peak
    cand_db: float = FLOOR_DBFS
    prev_db: float = FLOOR_DBFS


class SpeechTimeline:
    """Energy frames, sentences and events of the robot's own speech (see the module docstring)."""

    def __init__(
        self, config: SpeechMotionConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config, self.clock = config, clock
        self._lock = threading.Lock()
        self._starts: list[float] = []
        self._ends: list[float] = []
        self._dbfs: list[float] = []
        self._events: list[SpeechEvent] = []
        self._sentences: list[SpeechSentence] = []
        self._pads: dict[int, float] = {}
        self._phrases: dict[int, int] = {}  # utterance -> phrase gaps so far
        self._writer: _Writer | None = None
        self._last_cut: float | None = None
        self._last_emphasis = -math.inf  # held across sentences, reset by a cut
        self._counts = {"frames": 0, "sentences": 0, "cuts": 0} | dict.fromkeys(EVENT_KINDS, 0)

    def apply_config(self, config: SpeechMotionConfig) -> None:
        """Hot reload. A new frame length applies from the next sentence; the levels, the gap,
        the emphasis margin, warm-up and interval apply at once, mid-sentence too."""
        with self._lock:
            self.config = config

    # ----------------------------------------------------------------- writer

    def pad(self, sentence: int, start_s: float) -> None:
        """The pause before ``sentence`` starts playing at ``start_s``."""
        with self._lock:
            self._pads[sentence] = start_s

    def push(
        self,
        sentence: int,
        samples: np.ndarray,
        rate_hz: int,
        start_s: float,
        *,
        utterance: int,
        question: bool,
        mood: str | None,
        continued: bool,
    ) -> None:
        """Speech samples of ``sentence`` that start playing at ``start_s``."""
        if rate_hz <= 0 or samples.size == 0:
            return
        with self._lock:
            w = self._writer
            if w is not None and w.sentence.id != sentence:
                self._finish(w, w.sentence.end_s)
                w = None
            if w is None:
                if self._ends:  # a sink with no timeline: keep the frames in order
                    start_s = max(start_s, self._ends[-1])
                w = self._begin(sentence, rate_hz, start_s, utterance, question, mood, continued)
            elif w.rate_hz != rate_hz or start_s - (w.carry_t0 + w.carry_n / w.rate_hz) > (
                w.frame_s / 2
            ):
                resume = max(start_s, w.carry_t0 + w.carry_n / w.rate_hz)
                self._flush_carry(w)  # a new rate, or the sink ran dry: no frame spans the gap
                w.rate_hz, w.frame_n = rate_hz, max(1, round(rate_hz * w.frame_s))
                w.frame_s, w.carry_t0 = w.frame_n / rate_hz, resume
            self._feed(w, samples)
            w.sentence.end_s = w.carry_t0 + w.carry_n / w.rate_hz
            self._prune()

    def set_question(self, sentence: int) -> None:
        """``sentence`` turned out to end in a question after its audio began (a streamed reply
        whose transcript came late); unknown sentences are ignored."""
        with self._lock:
            for s in self._sentences:
                if s.id == sentence:
                    s.question = True

    def finish(self, sentence: int, end_s: float) -> None:
        """``sentence`` was pushed whole; its audio (with the sink's flushed tail) ends at
        ``end_s``. Unknown or already finished sentences are ignored."""
        with self._lock:
            w = self._writer
            if w is not None and w.sentence.id == sentence:
                self._finish(w, end_s)

    def truncate(self, now: float) -> None:
        """Barge-in at ``now``: drop everything at or after it and the unfinished writer state."""
        with self._lock:
            w = self._writer
            active = w is not None or any(s.end_s > now for s in self._sentences)
            self._writer = None
            self._pads.clear()
            self._last_emphasis = -math.inf
            keep = bisect.bisect_left(self._starts, now)
            del self._starts[keep:], self._ends[keep:], self._dbfs[keep:]
            if self._ends and self._ends[-1] > now:
                self._ends[-1] = now
            self._events = [e for e in self._events if e.at_s < now and e.until_s <= now]
            self._sentences = [s for s in self._sentences if s.start_s < now]
            for s in self._sentences:
                if s.end_s > now:
                    s.end_s, s.cut = now, True
            if active:  # a stop with nothing playing or pending is not a cut
                self._last_cut = now
                self._counts["cuts"] += 1

    def _begin(
        self,
        sentence: int,
        rate_hz: int,
        start_s: float,
        utterance: int,
        question: bool,
        mood: str | None,
        continued: bool,
    ) -> _Writer:
        s = SpeechSentence(
            sentence,
            utterance,
            start_s,
            start_s,
            question,
            mood,
            continued,
            self._pads.pop(sentence, None),
        )
        self._pads.clear()  # a pause whose sentence never spoke (it failed) is forgotten
        self._sentences.append(s)
        self._counts["sentences"] += 1
        self._event("sentence_start", start_s, start_s, s)
        frame_n = max(1, round(rate_hz * self.config.envelope_frame_ms / MS_PER_S))
        w = _Writer(s, rate_hz, frame_n, frame_n / rate_hz, carry_t0=start_s, voiced=[])
        self._writer = w
        return w

    def _feed(self, w: _Writer, samples: np.ndarray) -> None:
        """Frames from the carry plus ``samples``; one reshape-and-mean for the whole frames."""
        x = samples.astype(np.float32, copy=False)
        take = min(w.frame_n - w.carry_n, x.size) if w.carry_n else 0
        if take:
            w.carry_sq += float(np.dot(x[:take], x[:take]))
            w.carry_n += take
            x = x[take:]
            if w.carry_n == w.frame_n:
                self._frame(w, w.carry_t0, w.carry_sq / w.carry_n)
                w.carry_t0 += w.frame_s
                w.carry_sq, w.carry_n = 0.0, 0
        whole = x.size // w.frame_n
        if whole:
            sq = np.square(x[: whole * w.frame_n], dtype=np.float64)
            for i, ms in enumerate(sq.reshape(whole, w.frame_n).mean(axis=1)):
                self._frame(w, w.carry_t0 + i * w.frame_s, float(ms))
            w.carry_t0 += whole * w.frame_s
        rest = x[whole * w.frame_n :]
        if rest.size:
            w.carry_sq += float(np.dot(rest, rest))
            w.carry_n += rest.size

    def _flush_carry(self, w: _Writer) -> None:
        if w.carry_n:
            self._frame(w, w.carry_t0, w.carry_sq / w.carry_n, w.carry_n / w.rate_hz)
        w.carry_sq, w.carry_n = 0.0, 0

    def _frame(self, w: _Writer, at: float, mean_sq: float, length_s: float | None = None) -> None:
        """One frame: store it, judge the previous one as an emphasis peak, look for a gap."""
        cfg = self.config
        db = POWER_DB * math.log10(mean_sq) if mean_sq > 0 else FLOOR_DBFS
        db = max(db, FLOOR_DBFS)
        end = at + (w.frame_s if length_s is None else length_s)
        self._starts.append(at)
        self._ends.append(end)
        self._dbfs.append(db)
        self._counts["frames"] += 1
        self._judge(w, db)
        if db < cfg.silence_dbfs:
            w.cand_at = None
            w.prev_db = db
            return
        if w.last_voiced_end is not None and at - w.last_voiced_end >= (
            cfg.phrase_gap_ms / MS_PER_S
        ):
            utt = w.sentence.utterance
            index = self._phrases.pop(utt, 0)
            self._phrases[utt] = index + 1
            while len(self._phrases) > PHRASE_UTTERANCES_MAX:
                del self._phrases[next(iter(self._phrases))]
            self._event("phrase_gap", w.last_voiced_end, at, w.sentence, index=index)
        w.last_voiced_end = end
        assert w.voiced is not None
        bisect.insort(w.voiced, db)
        w.cand_at, w.cand_db = at, db

    def _judge(self, w: _Writer, next_db: float) -> None:
        """The candidate (last voiced frame) is a peak when it is at least as loud as the frame
        before it and louder than ``next_db``, and far enough above the running median."""
        if w.cand_at is None:
            return
        at, db = w.cand_at, w.cand_db
        prev, w.prev_db, w.cand_at = w.prev_db, db, None
        cfg = self.config
        if not (db >= prev and db > next_db) or at - self._last_emphasis < (
            cfg.emphasis_floor_interval_s
        ):
            return
        voiced = w.voiced or [db]
        if len(voiced) * w.frame_s * MS_PER_S < cfg.emphasis_warmup_ms:
            return
        if db - _median(voiced) >= cfg.emphasis_db:
            self._last_emphasis = at
            self._event("emphasis", at, at, w.sentence, dbfs=db)

    def _finish(self, w: _Writer, end_s: float) -> None:
        self._flush_carry(w)
        self._judge(w, FLOOR_DBFS)
        w.sentence.end_s = max(end_s, w.sentence.end_s)  # a sink without a timeline says "now"
        self._event("sentence_end", w.sentence.end_s, w.sentence.end_s, w.sentence)
        self._writer = None
        self._prune()

    def _event(
        self,
        kind: EventKind,
        at: float,
        until: float,
        s: SpeechSentence,
        *,
        index: int = 0,
        dbfs: float = FLOOR_DBFS,
    ) -> None:
        bisect.insort(
            self._events,
            SpeechEvent(kind, at, until, s.id, s.utterance, index, dbfs),
            key=lambda e: e.at_s,
        )
        self._counts[kind] += 1

    def _prune(self) -> None:
        """Drop what ended more than ``history_s`` ago; the lock is held."""
        horizon = self.clock() - self.config.history_s
        old = bisect.bisect_left(self._ends, horizon)
        if old:
            del self._starts[:old], self._ends[:old], self._dbfs[:old]
        self._events = [e for e in self._events if e.until_s >= horizon]
        self._sentences = [s for s in self._sentences if s.end_s >= horizon]

    # ----------------------------------------------------------------- reader

    def level_at(self, t: float) -> float | None:
        """dBFS of the frame playing at ``t``; None between sentences or outside the history."""
        with self._lock:
            return self._level_at(t)

    def frames_between(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """``(start_s, dbfs)`` of the frames that start in ``[t0, t1)``."""
        with self._lock:
            return self._frames_between(t0, t1)

    def events_between(self, t0: float, t1: float) -> list[SpeechEvent]:
        """Events stamped in ``[t0, t1)``, in time order."""
        with self._lock:
            return [e for e in self._events if t0 <= e.at_s < t1]

    def sentence_at(self, t: float) -> SpeechSentence | None:
        """The sentence playing at ``t`` (its pause before it included); a copy."""
        with self._lock:
            s = self._sentence_at(t)
            return replace(s) if s is not None else None

    def sentence(self, sentence_id: int) -> SpeechSentence | None:
        """The sentence ``sentence_id`` while it is in the history; a copy."""
        with self._lock:
            for s in reversed(self._sentences):
                if s.id == sentence_id:
                    return replace(s)
            return None

    def view(
        self,
        t: float,
        *,
        frames_from: float,
        events_from: float,
        events_to: float,
        known: tuple[int, float] | None = None,
    ) -> TimelineView:
        """Everything a reader tick needs, under one lock (see :class:`TimelineView`).
        ``sentence_dbfs`` is filled unless ``known`` is the current sentence's ``(id, end_s)``
        (the reader already has its levels)."""
        with self._lock:
            now = self._sentence_at(t)
            events = tuple(e for e in self._events if events_from <= e.at_s < events_to)
            ids = {e.sentence for e in events} | ({now.id} if now is not None else set())
            sentences = {s.id: replace(s) for s in self._sentences if s.id in ids}
            dbfs = None
            if now is not None and (now.id, now.end_s) != known:
                dbfs = tuple(db for _, db in self._frames_between(now.start_s, now.end_s))
            return TimelineView(
                t,
                self._last_cut,
                sentences.get(now.id) if now is not None else None,
                self._level_at(t),
                tuple(self._frames_between(frames_from, t)),
                events,
                sentences,
                dbfs,
            )

    def _level_at(self, t: float) -> float | None:
        i = bisect.bisect_right(self._starts, t) - 1
        if i >= 0 and t < self._ends[i]:
            return self._dbfs[i]
        return None

    def _frames_between(self, t0: float, t1: float) -> list[tuple[float, float]]:
        a, b = bisect.bisect_left(self._starts, t0), bisect.bisect_left(self._starts, t1)
        return list(zip(self._starts[a:b], self._dbfs[a:b], strict=True))

    def _sentence_at(self, t: float) -> SpeechSentence | None:
        for s in reversed(self._sentences):
            first = s.pad_start_s if s.pad_start_s is not None else s.start_s
            if first <= t < s.end_s:
                return s
        return None

    def last_cut(self) -> float | None:
        """Clock stamp of the last barge-in truncation, None before any."""
        with self._lock:
            return self._last_cut

    def snapshot(self) -> dict[str, int]:
        """Counts since start for the debug report (SPEC 12A): frames, sentences, cuts and
        events per kind. Never text, samples or energies."""
        with self._lock:
            return dict(self._counts)


def _median(values: list[float]) -> float:
    """Median of a sorted list."""
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2
