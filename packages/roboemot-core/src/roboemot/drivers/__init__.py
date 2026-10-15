"""Driver interfaces (SPEC 4, Driver interfaces) and a mock for tests without hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

MotorMode = Literal["active", "limp", "teach"]


class PoseGroupDriver(ABC):
    """For vendor SDKs that take poses, not servo angles (Reachy Mini). Values in rad and m,
    keyed by manifest dof name."""

    @abstractmethod
    def write_pose_targets(self, t: float, values: dict[str, float]) -> None: ...

    @abstractmethod
    def read_state(self) -> dict[str, float]: ...

    @abstractmethod
    def set_mode(self, mode: MotorMode) -> None: ...

    def reconnect(self, timeout_s: float) -> None:  # noqa: B027 - optional hook, not abstract
        """Re-open the link after a loss (SPEC 12A); raise if the robot is still unreachable.
        Drivers whose transport recovers by itself keep this no-op."""


@dataclass
class MockPoseDriver(PoseGroupDriver):
    """Records every write; ``read_state`` returns the last target (perfect tracking)."""

    home: dict[str, float]
    writes: list[tuple[float, dict[str, float]]] = field(default_factory=list)
    mode: MotorMode = "limp"
    modes: list[MotorMode] = field(default_factory=list)

    def write_pose_targets(self, t: float, values: dict[str, float]) -> None:
        if self.mode != "active":
            raise RuntimeError("targets written while motors are not active")
        self.writes.append((t, dict(values)))

    def read_state(self) -> dict[str, float]:
        return dict(self.writes[-1][1]) if self.writes else dict(self.home)

    def set_mode(self, mode: MotorMode) -> None:
        self.mode = mode
        self.modes.append(mode)
