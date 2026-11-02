"""The motion loop's latest movement reading, for diagnostics that run on other threads (SPEC 12A).

The loop publishes a :class:`MotionState` at the end of every tick by a single attribute
assignment, so a reader (the barge-in detector on the listener thread) never takes the loop's
lock, which is held across driver calls. Robot-agnostic: the fastest joint is named by its
manifest group, a label the manifest owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MotionState:
    """One tick's movement: the highest commanded angular speed over the angle dofs (deg/s),
    the manifest group of that dof (None when still), whether the loop yields to external
    control, whether a reflex command is active, the tick's monotonic stamp, and the highest
    angular speed per manifest group (deg/s), so a fast group does not hide a slower one."""

    stamp: float
    speed_dps: float
    group: str | None
    external: bool
    reflex: bool
    group_speed_dps: dict[str, float] = field(default_factory=dict)


class MotionProbe(Protocol):
    """Where diagnostics read the motion state: None before the first tick."""

    def motion_state(self) -> MotionState | None: ...
