"""The motion loop's movement, for diagnostics that run on other threads (SPEC 12A).

The loop's tick only stores raw values it already has (its stamp, the last two fresh measured
poses as one tuple, the arbiter's reflex flag); a reader takes a cheap :class:`MotionSample`
(``MotionProbe.motion_sample``: a copy of the limiter's velocities and those references) and
derives the per-group speeds later, off its own hot path (``MotionProbe.motion_state``). Neither
takes the loop's lock, which is held across driver calls. Robot-agnostic: speeds are keyed by
manifest pose group, labels the manifest owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# The last two fresh measured poses: (previous stamp, previous pose, last stamp, last pose).
MeasuredPair = tuple[float | None, dict[str, float] | None, float, dict[str, float]]


@dataclass(frozen=True)
class MotionSample:
    """Raw values at one instant: the last tick's stamp, the commanded angular velocities
    (rad/s, a copy), the last two fresh measured poses, external control, an active reflex."""

    stamp: float
    velocities: dict[str, float]
    measured: MeasuredPair | None
    external: bool
    reflex: bool


@dataclass(frozen=True)
class MotionState:
    """A sample's derived movement: the highest commanded and measured angular speed (deg/s) per
    manifest pose group (measured empty without two fresh poses), external control, reflex."""

    stamp: float
    commanded_dps: dict[str, float]
    measured_dps: dict[str, float]
    external: bool
    reflex: bool


class MotionProbe(Protocol):
    """Where diagnostics read the motion: ``motion_sample`` is cheap (None before the first
    tick); ``motion_state`` derives a sample, later and on the reader's thread."""

    def motion_sample(self) -> MotionSample | None: ...

    def motion_state(self, sample: MotionSample) -> MotionState: ...
