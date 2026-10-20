"""Motion primitives: named keyframe sequences replayed through the arbiter (SPEC 10, 11A).

A primitive is data (configurable), sampled at the loop rate and submitted as a COMMANDED
(or EXPRESSION, if additive) command. Amplitude scales the deviation from the first keyframe
(absolute) or the offsets themselves (additive); speed scales time.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from .core import Command, Priority
from .curves import CURVES


@dataclass(frozen=True)
class Keyframe:
    t: float  # seconds from start
    values: dict[str, float]


@dataclass(frozen=True)
class Primitive:
    name: str
    keyframes: tuple[Keyframe, ...]
    method: str = "minjerk"
    additive: bool = False

    def __post_init__(self) -> None:
        if len(self.keyframes) < 2:
            raise ValueError(f"primitive {self.name}: needs at least two keyframes")
        keys = set(self.keyframes[0].values)
        times = [k.t for k in self.keyframes]
        if any(set(k.values) != keys for k in self.keyframes):
            raise ValueError(f"primitive {self.name}: every keyframe must set the same dofs")
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError(f"primitive {self.name}: keyframe times must increase")
        if self.method not in CURVES:
            raise ValueError(f"primitive {self.name}: unknown curve {self.method}")

    @property
    def duration(self) -> float:
        return self.keyframes[-1].t

    def sample(self, t: float) -> dict[str, float]:
        """Dof values at ``t`` seconds: the curve between the two keyframes around it (binary
        search, a recorded clip holds hundreds of keyframes); clamped to the first and last."""
        kfs = self.keyframes
        if t <= kfs[0].t:
            return dict(kfs[0].values)
        i = bisect_left(kfs, t, key=lambda k: k.t)  # first keyframe at or after t
        if i >= len(kfs):
            return dict(kfs[-1].values)
        a, b = kfs[i - 1], kfs[i]
        s = CURVES[self.method]((t - a.t) / (b.t - a.t))
        return {k: a.values[k] + (b.values[k] - a.values[k]) * s for k in a.values}


class Player:
    """Plays one primitive; call :meth:`command` every tick until it returns None."""

    def __init__(
        self,
        prim: Primitive,
        start: float,
        *,
        source: str | None = None,
        amplitude: float = 1.0,
        speed: float = 1.0,
        priority: Priority | None = None,
    ) -> None:
        self.prim, self.start, self.amplitude, self.speed = prim, start, amplitude, speed
        self.source = source or f"primitive:{prim.name}"
        self.priority = priority or (Priority.EXPRESSION if prim.additive else Priority.COMMANDED)
        self._origin = prim.keyframes[0].values

    def done(self, now: float) -> bool:
        return (now - self.start) * self.speed >= self.prim.duration

    def command(self, now: float) -> Command | None:
        if self.done(now):
            return None
        values = self.prim.sample((now - self.start) * self.speed)
        if self.prim.additive:
            values = {k: v * self.amplitude for k, v in values.items()}
        else:
            values = {
                k: self._origin.get(k, v) + (v - self._origin.get(k, v)) * self.amplitude
                for k, v in values.items()
            }
        return Command(self.source, self.priority, values, now, additive=self.prim.additive)
