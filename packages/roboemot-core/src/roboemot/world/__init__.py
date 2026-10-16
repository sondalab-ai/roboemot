"""World model: egocentric map of people, animals and objects (SPEC 5A)."""

from .frame import FrameSpec, angular_distance, to_base_frame
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
    "to_base_frame",
]
