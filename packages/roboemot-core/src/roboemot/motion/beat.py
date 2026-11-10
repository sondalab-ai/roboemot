"""The beat the dance follows (SPEC 7, Dance motion): one read-only interface over its sources.

Two providers, asked in this order: the robot's own sound (a self-soothing hum, whose beat grid is
known exactly, section 7B), then room music (the detector, section 8). Each returns None when it
has no beat. Accents (the stamps of a blirp's events) come through the same object, since a blirp
has no period to give.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..config.schema import SECONDS_PER_MINUTE

if TYPE_CHECKING:
    from ..voice.music import MusicState


@dataclass(frozen=True)
class Beat:
    """A beat grid. ``phase_s`` is the stamp of one beat on the monotonic clock (any beat of the
    grid: it may differ from read to read), ``strength`` a share in [0, 1] that scales the
    amplitude, ``source`` which provider it came from: ``free`` is room music with no readable
    beat, a slow grid that follows nothing."""

    period_s: float
    phase_s: float
    strength: float
    source: Literal["own", "room", "free"]


class BeatSource:
    """The dance layer's only view of the beat. ``own`` and ``room`` are the two providers;
    ``accents`` lists the accent stamps between two times (none by default)."""

    def __init__(
        self,
        own: Callable[[float], Beat | None],
        room: Callable[[float], Beat | None],
        *,
        accents: Callable[[float, float], list[float]] = lambda _since, _now: [],
    ) -> None:
        self._own, self._room, self._accents = own, room, accents

    def beat_at(self, now: float) -> Beat | None:
        """The beat at ``now``: the robot's own sound first, then room music, else None. The
        providers are asked on every tick, so each must be cheap and must not block."""
        beat = self._own(now)
        return beat if beat is not None else self._room(now)

    def accents(self, since: float, now: float) -> list[float]:
        """Accent stamps in ``(since, now]``, oldest first. ``now`` may lie ahead of the clock:
        the dance layer asks early enough for a flick to peak on its stamp."""
        return self._accents(since, now)


def room_beat(state: MusicState | None, free: tuple[float, float] | None = None) -> Beat | None:
    """The beat of room music from the detector's state; None without a state or while no music
    plays. Its strength is the detector's confidence. Music with no tempo gets a ``free`` grid
    from ``free`` (tempo in bpm, strength), else None.

    The detector gives the time since the last beat, so ``phase_s`` is that last beat: it moves
    forward one period per beat and jumps when the grid is re-estimated. It tells where a beat
    falls and nothing about which beat it is; the dance layer keeps its own place in a move."""
    if state is None or not state.playing:
        return None
    if state.tempo_bpm is None or state.beat_phase_s is None:
        if free is None:
            return None
        return Beat(SECONDS_PER_MINUTE / free[0], 0.0, free[1], "free")
    period_s = SECONDS_PER_MINUTE / state.tempo_bpm
    return Beat(period_s, state.stamp - state.beat_phase_s, state.confidence, "room")


__all__ = ["Beat", "BeatSource", "room_beat"]
