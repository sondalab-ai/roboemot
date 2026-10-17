"""World model: egocentric map of people, animals and objects (SPEC 5A)."""

from .feed import feed_world
from .frame import FrameSpec, angular_distance, camera_in_view, from_base_frame, to_base_frame
from .model import InViewFn, WorldModel
from .scan import ScanContext, ScanController, ScanDecision, ScanPolicy, SweepRunner
from .sources import (
    ObservationSink,
    ScanPlanner,
    ScanWaypoint,
    WorldPlugin,
    WorldSource,
    load_world_plugin,
)
from .types import Entry, Observation, ObserveResult

__all__ = [
    "Entry",
    "FrameSpec",
    "InViewFn",
    "Observation",
    "ObservationSink",
    "ObserveResult",
    "ScanContext",
    "ScanController",
    "ScanDecision",
    "ScanPlanner",
    "ScanPolicy",
    "ScanWaypoint",
    "SweepRunner",
    "WorldModel",
    "WorldPlugin",
    "WorldSource",
    "angular_distance",
    "camera_in_view",
    "feed_world",
    "from_base_frame",
    "load_world_plugin",
    "to_base_frame",
]
