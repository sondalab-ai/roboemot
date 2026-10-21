"""The live transcript: recent turns of the conversation for the web UI's Live tab (SPEC 11A, 12).

An in-memory ring of ``{seq, t, who, text, language, interrupted}``, filled by the conversation
worker (the user's final utterance, the robot's sentences as they are queued to play) and read by
the Live tab behind the PIN. Opt-in (``webui.live.transcript``, with the Live tab itself on): while
either is off ``add`` keeps nothing at all, and turning it off clears what was kept. Bounded by
count, age and text length, all from the configuration. Never persisted, logged, repr-ed or put in
a debug report: this module has no logger and no ``__repr__`` that shows a turn. It is per robot,
not per login session.
"""

from __future__ import annotations

import collections
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from roboemot.config.schema import LiveConfig

NOT_RECORDED = -1  # the generation of an utterance that was never recorded: its reply never is
EPOCH_BYTES = 4  # random bytes naming one life of the ring (a restart gives the page a new one)


@dataclass(repr=False)
class _Turn:
    seq: int
    at: float  # monotonic, for the age bound
    wall: float  # epoch seconds, shown by the page
    who: str
    text: str
    language: str
    interrupted: bool = False
    rev: int = 0  # the ring's revision when this turn was last added to or changed


class TranscriptRing:
    """Thread-safe. ``add`` is called on the listener and conversation threads, so the "off" case
    returns before the lock is taken."""

    def __init__(
        self,
        cfg: LiveConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = cfg
        self._enabled = cfg.enabled and cfg.transcript
        self._clock, self._wall = clock, wall
        self._lock = threading.Lock()
        self._turns: collections.deque[_Turn] = collections.deque()
        self._seq = 0
        self._rev = 0  # bumped by every add, extend and settle: the page's cursor
        self._boot = secrets.token_hex(EPOCH_BYTES)
        self._generation = 0
        self._clears = 0  # generation fence: every clear bumps it
        self._asleep = False

    def __repr__(self) -> str:
        return f"TranscriptRing(enabled={self._enabled}, size={len(self._turns)})"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._turns)

    def attach(self, stack: Any) -> Callable[[], None]:
        """Follow ``webui.live`` on the configuration stack as a listener of its own, so turning the
        flag off stops the recording and clears the ring whatever the other listeners do."""
        return stack.subscribe(lambda change: self.apply_config(change.config.webui.live))

    def apply_config(self, cfg: LiveConfig) -> None:
        """Hot reload: new bounds apply at once; turning the flag off clears the ring."""
        with self._lock:
            had = self._enabled or bool(self._turns)
            self._cfg = cfg
            self._enabled = cfg.enabled and cfg.transcript
            if not self._enabled:
                if had:
                    self._clear_locked()
            else:
                self._trim_locked(self._clock())

    def set_asleep(self, asleep: bool) -> None:
        """The robot fell asleep (True: nothing is recorded while it sleeps and the ring is
        cleared, whether or not the rest of the runtime pauses) or woke up."""
        with self._lock:
            self._asleep = asleep
            if asleep:
                self._clear_locked()

    def trim(self) -> None:
        """Drop turns past the age bound. Called from the app tick, so text does not outlive
        ``transcript_max_age_s`` in memory when nobody polls or speaks."""
        if self._turns:
            with self._lock:
                self._trim_locked(self._clock())

    def begin(self, who: str, text: str, language: str) -> tuple[int | None, int]:
        """Record the user's turn and return ``(seq, generation)`` atomically. The reply to it
        passes the generation to ``add``: a clear in between makes the ring refuse it.
        ``NOT_RECORDED`` as the generation when nothing was kept."""
        with self._lock:
            seq = self._add_locked(who, text, language, False)
            return seq, (self._clears if seq is not None else NOT_RECORDED)

    def add(
        self,
        who: str,
        text: str,
        language: str,
        *,
        interrupted: bool = False,
        generation: int | None = None,
    ) -> int | None:
        """Keep one turn; returns its ``seq``, or None while the transcript is off or asleep,
        ``text`` is empty, or ``generation`` is not the current one (a clear came after the
        utterance this is the reply to). The text is cut to ``transcript_max_chars``."""
        if not self._enabled or not text:
            return None
        with self._lock:
            if generation is not None and generation != self._clears:
                return None
            return self._add_locked(who, text, language, interrupted)

    def _add_locked(self, who: str, text: str, language: str, interrupted: bool) -> int | None:
        if not self._enabled or self._asleep or not text:
            return None
        now = self._clock()
        self._seq += 1
        self._rev += 1
        self._turns.append(
            _Turn(
                self._seq,
                now,
                self._wall(),
                who,
                text[: self._cfg.transcript_max_chars],
                language,
                interrupted,
                self._rev,
            )
        )
        self._trim_locked(now)
        return self._seq

    def extend(self, seq: int | None, text: str) -> None:
        """Another sentence of the robot turn ``seq`` is queued to play; a turn that has left the
        ring is ignored."""
        if seq is None or not text or not self._enabled:
            return
        with self._lock:
            turn = self._find(seq)
            if turn is not None:
                joined = f"{turn.text} {text}"
                turn.text = joined[: self._cfg.transcript_max_chars]
                self._touch(turn)

    def settle(self, seq: int | None, heard: str, *, interrupted: bool) -> None:
        """The robot turn ``seq`` ended: its text becomes what was heard. Nothing heard leaves an
        empty turn (a page that has the old text then drops it; ``snapshot`` from the start never
        lists it); ``interrupted`` marks a reply cut by a barge-in or a stop."""
        if seq is None or not self._enabled:
            return
        with self._lock:
            turn = self._find(seq)
            if turn is None:
                return
            turn.text = heard[: self._cfg.transcript_max_chars]
            turn.interrupted = interrupted and bool(heard)
            self._touch(turn)

    def clear(self) -> None:
        """Forget everything (the Clear button, sleep, shutdown). Sequence numbers keep rising."""
        with self._lock:
            self._clear_locked()

    def snapshot(self, after: int = 0, epoch: str = "") -> dict[str, Any]:
        """``enabled``, ``epoch`` (changes on clear and restart: the page then starts over),
        ``rev`` (the cursor to send back), ``max_turns``, ``min_seq`` (the oldest kept, 0 if none)
        and the turns added or changed after revision ``after``: a reply that grew or was trimmed
        since the page last asked comes again with the same ``seq``. Everything is returned when
        ``after`` is ahead of the ring or ``epoch`` is given and is not the current one (the
        page's cursor is from another life). Empty turns are listed only to a page that has
        already seen them."""
        with self._lock:
            now = self._clock()
            self._trim_locked(now)
            current = f"{self._boot}.{self._generation}"
            stale = after > self._rev or bool(epoch and epoch != current)
            cursor = 0 if stale else after
            kept = [t for t in self._turns if t.rev > cursor and (t.text or cursor > 0)]
            kept = self._within_budget(kept)
            turns = [
                {
                    "seq": t.seq,
                    "t": t.wall,
                    "who": t.who,
                    "text": t.text,
                    "language": t.language,
                    "interrupted": t.interrupted,
                }
                for t in kept
            ]
            return {
                "enabled": self._enabled,
                "epoch": current,
                "rev": self._rev,
                "min_seq": self._turns[0].seq if self._turns else 0,
                "max_turns": self._cfg.transcript_max_turns,
                "turns": turns,
            }

    def _within_budget(self, turns: list[_Turn]) -> list[_Turn]:
        """The newest turns whose text fits ``transcript_response_max_chars`` (at least one):
        a full list larger than that sends its newest part."""
        budget, out = self._cfg.transcript_response_max_chars, []
        for turn in reversed(turns):
            budget -= len(turn.text)
            if budget < 0 and out:
                break
            out.append(turn)
        return out[::-1]

    def _find(self, seq: int) -> _Turn | None:
        for turn in reversed(self._turns):
            if turn.seq == seq:
                return turn
            if turn.seq < seq:
                break
        return None

    def _touch(self, turn: _Turn) -> None:
        self._rev += 1
        turn.rev = self._rev

    def _clear_locked(self) -> None:
        self._turns.clear()
        self._generation += 1
        self._clears += 1

    def _trim_locked(self, now: float) -> None:
        while len(self._turns) > self._cfg.transcript_max_turns:
            self._turns.popleft()
        horizon = now - self._cfg.transcript_max_age_s
        while self._turns and self._turns[0].at < horizon:
            self._turns.popleft()
