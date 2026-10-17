"""World plugin interfaces and their entry-point loader (SPEC 5A, Observations and sources; 11).

A robot package registers one factory in the ``roboemot.world`` entry-point group, keyed by
the manifest ``robot.id`` (same pattern as ``roboemot.diagnostics`` and ``roboemot.installers``).
The factory returns a ``WorldPlugin``: the robot's own ``WorldSource`` list (sound direction,
proximity, ...) and its ``ScanPlanner``. Core stays robot-agnostic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol

from ..config.schema import RoboEmotConfig
from ..manifest import Manifest
from .types import Observation

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.world"

ObservationSink = Callable[[Observation, Mapping[str, float] | None], None]
"""Receives an observation in the sensor frame and the joint state at capture."""


class WorldSource(Protocol):
    """Produces observations until stopped; the sink is the bus (``percept.entity``)."""

    name: str

    def start(self, sink: ObservationSink) -> None:
        """Begin producing; ``sink`` may be called from the source's own thread."""
        ...

    def stop(self) -> None:
        """Stop producing and release the sensor; safe to call twice."""
        ...


@dataclass(frozen=True)
class ScanWaypoint:
    """One stop of a sweep: dof targets in the robot base frame and how long to stay.

    ``azimuth_rad`` is the base-frame bearing the camera looks at once the targets are reached.
    """

    azimuth_rad: float
    targets: dict[str, float]
    dwell_s: float


class ScanPlanner(Protocol):
    """How a robot looks around: waypoints within its manifest limits (SPEC 5A, Scanning)."""

    def plan(
        self, pose: Mapping[str, float], fov_h_rad: float, *, amplitude: float, dwell_s: float
    ) -> list[ScanWaypoint]:
        """Waypoints that cover ``amplitude`` (share of the full circle, centred on the front)
        with camera views of ``fov_h_rad`` that overlap; the path starts near ``pose``."""
        ...

    def glance(
        self,
        pose: Mapping[str, float],
        azimuth_rad: float,
        elevation_rad: float,
        *,
        dwell_s: float,
    ) -> ScanWaypoint:
        """A single waypoint that points the camera at a base-frame bearing."""
        ...

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (SPEC 11A): take the new configuration."""
        ...


@dataclass
class WorldPlugin:
    """What a robot package contributes to the world model."""

    planner: ScanPlanner | None = None
    sources: list[WorldSource] = field(default_factory=list)


JointStateFn = Callable[[], Mapping[str, float]]
WorldFactory = Callable[[Manifest, RoboEmotConfig, Any, JointStateFn], WorldPlugin]


def load_world_plugin(
    manifest: Manifest, cfg: RoboEmotConfig, robot: Any, joint_state: JointStateFn
) -> WorldPlugin:
    """The plugin registered for ``manifest.robot.id``, built with the robot handle and the
    app's measured-joint-state provider (plugins never open a second driver).

    A missing or broken plugin is logged (type and message only) and yields an empty plugin:
    the robot then has no scanning and no extra sources, and the app keeps running.
    """
    robot_id = manifest.robot.id
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == robot_id:
            try:
                return ep.load()(manifest, cfg, robot, joint_state)  # type: ignore[no-any-return]
            except Exception as exc:
                log.warning("world plugin %s: %s: %s", robot_id, type(exc).__name__, exc)
                return WorldPlugin()
    log.info("no world plugin installed for %s", robot_id)
    return WorldPlugin()
