"""World model: egocentric map of people, animals and objects (SPEC 5A)."""

from .feed import feed_world
from .frame import FrameSpec, angular_distance, camera_in_view, from_base_frame, to_base_frame
from .model import InViewFn, WorldModel
from .types import Entry, Observation, ObserveResult

__all__ = [
    "Entry",
    "FrameSpec",
    "InViewFn",
    "Observation",
    "ObserveResult",
    "WorldModel",
    "angular_distance",
    "camera_in_view",
    "feed_world",
    "from_base_frame",
    "to_base_frame",
]
