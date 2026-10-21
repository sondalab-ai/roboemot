"""The live transcript: recent turns of the conversation for the web UI's Live tab (SPEC 11A, 12).

An in-memory ring of ``{seq, t, who, text, language, interrupted}``, filled by the conversation
worker (the user's final utterance, the robot's sentences as they are queued to play) and read by
the Live tab behind the PIN. Opt-in (``webui.live.transcript``): while it is off ``add`` keeps
nothing at all, and turning it off clears what was kept. Bounded by count, age and text length,
all from the configuration. Never persisted, logged, repr-ed or put in a debug report: this module
has no logger and no ``__repr__`` that shows a turn. It is per robot, not per login session.
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

EPOCH_BYTES = 4  # random bytes naming one life of the ring (a restart gives the page a new one)


@dataclass
class _Turn:
    seq: int
    at: float  # monotonic, for the age bound
    wall: float  # epoch seconds, shown by the page
    who: str
    text: str
    language: str
    interrupted: bool = False


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
        self._enabled = cfg.transcript
        self._clock, self._wall = clock, wall
        self._lock = threading.Lock()
        self._turns: collections.deque[_Turn] = collections.deque()
        self._seq = 0
        self._boot = secrets.token_hex(EPOCH_BYTES)
        self._generation = 0

    def __repr__(self) -> str:
        return f"TranscriptRing(enabled={self._enabled}, size={len(self._turns)})"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._turns)

    def apply_config(self, cfg: LiveConfig) -> None:
        """Hot reload: new bounds apply at once; turning the flag off clears the ring."""
        with self._lock:
            self._cfg = cfg
            self._enabled = cfg.transcript
            if not cfg.transcript:
                self._clear_locked()
            else:
                self._trim_locked(self._clock())

    def add(self, who: str, text: str, language: str, *, interrupted: bool = False) -> int | None:
        """Keep one turn; returns its ``seq``, or None while the transcript is off or ``text`` is
        empty. The text is cut to ``transcript_max_chars``."""
        if not self._enabled or not text:
            return None
        now = self._clock()
        with self._lock:
            if not self._enabled:
                return None
            self._seq += 1
            self._turns.append(
                _Turn(
                    self._seq,
                    now,
                    self._wall(),
                    who,
                    text[: self._cfg.transcript_max_chars],
                    language,
                    interrupted,
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

    def settle(self, seq: int | None, heard: str, *, interrupted: bool) -> None:
        """The robot turn ``seq`` ended: its text becomes what was heard. Nothing heard removes
        the turn; ``interrupted`` marks a reply cut by a barge-in or a stop."""
        if seq is None or not self._enabled:
            return
        with self._lock:
            turn = self._find(seq)
            if turn is None:
                return
            if not heard:
                self._turns.remove(turn)
                return
            turn.text = heard[: self._cfg.transcript_max_chars]
            turn.interrupted = interrupted

    def clear(self) -> None:
        """Forget everything (the Clear button, sleep, shutdown). Sequence numbers keep rising."""
        with self._lock:
            self._clear_locked()

    def snapshot(self, after: int = 0) -> dict[str, Any]:
        """``enabled``, ``epoch`` (changes on clear and restart: the page then starts over),
        ``seq`` (the newest), ``max_turns``, ``min_seq`` (the oldest kept, 0 if none) and the
        turns newer than ``after`` (all of them when ``after`` is ahead of the ring: the page's
        cursor is from another life)."""
        with self._lock:
            now = self._clock()
            self._trim_locked(now)
            cursor = after if after <= self._seq else 0
            turns = [
                {
                    "seq": t.seq,
                    "t": t.wall,
                    "who": t.who,
                    "text": t.text,
                    "language": t.language,
                    "interrupted": t.interrupted,
                }
                for t in self._turns
                if t.seq > cursor
            ]
            return {
                "enabled": self._enabled,
                "epoch": f"{self._boot}.{self._generation}",
                "seq": self._seq,
                "min_seq": self._turns[0].seq if self._turns else 0,
                "max_turns": self._cfg.transcript_max_turns,
                "turns": turns,
            }

    def _find(self, seq: int) -> _Turn | None:
        for turn in reversed(self._turns):
            if turn.seq == seq:
                return turn
            if turn.seq < seq:
                break
        return None

    def _clear_locked(self) -> None:
        self._turns.clear()
        self._generation += 1

    def _trim_locked(self, now: float) -> None:
        while len(self._turns) > self._cfg.transcript_max_turns:
            self._turns.popleft()
        horizon = now - self._cfg.transcript_max_age_s
        while self._turns and self._turns[0].at < horizon:
            self._turns.popleft()
