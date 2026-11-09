"""The beat the dance follows (SPEC 7, Dance motion): one read-only interface over its sources.

Two providers, asked in this order: the robot's own sound (a self-soothing hum, whose beat grid is
known exactly, section 7B), then room music (the detector, section 8). Each returns None when it
has no beat. Accents (the stamps of a blirp's events) come through the same object, since a blirp
has no period to give.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Beat:
    """A beat grid. ``phase_s`` is the stamp of one beat on the monotonic clock, ``strength`` a
    share in [0, 1] that scales the amplitude, ``source`` which provider it came from."""

    period_s: float
    phase_s: float
    strength: float
    source: Literal["own", "room"]


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


__all__ = ["Beat", "BeatSource"]
