"""Data types of the vision pipeline (SPEC 5, 5A, 12).

Images are ``HxWx3`` uint8 BGR arrays, the order YuNet and SFace expect. Boxes are
``(x1, y1, x2, y2)`` in pixels of the original frame. Published payloads (``FacePercept``,
``PersonRecognized``) never hold an embedding or an image (SPEC 12): embeddings live only in
the matcher call that consumes them, and animal or object appearance embeddings travel only
inside ``world.Observation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]
FACE_LABEL = "face"


@dataclass
class Frame:
    """One camera frame with its capture time (monotonic seconds) and the joint state then."""

    image: np.ndarray = field(repr=False)
    stamp: float
    joint_state: Mapping[str, float] | None = None


@dataclass
class Detection:
    """A detector output: box, score, class label and, for faces, 5 landmarks (5x2 pixels)."""

    box: Box
    score: float
    label: str = FACE_LABEL
    landmarks: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def min_side(self) -> float:
        """The shorter side of the box in pixels."""
        x1, y1, x2, y2 = self.box
        return min(x2 - x1, y2 - y1)


@dataclass(frozen=True)
class FacePercept:
    """Published on ``percept.face`` for each confirmed face track in a processed frame.

    ``box`` is normalised to the frame (0..1). ``owner_id`` is a registry ID once the track is
    recognised, else ``anonymous_id`` (session-only) is set. No age or gender is estimated
    (owner decision 2026-10-23).
    """

    track_id: int
    stamp: float
    box: Box
    score: float
    azimuth_rad: float
    elevation_rad: float
    owner_id: str | None
    anonymous_id: str | None
    new_track: bool


@dataclass(frozen=True)
class PersonRecognized:
    """Published on ``person.recognized`` when a track turns into a known person."""

    track_id: int
    person_id: str
    stamp: float
    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True)
class FrameReport:
    """What ``VisionPipeline.process`` did with a frame (for tests, replay and the governor)."""

    stamp: float
    faces_detected: bool = False
    objects_detected: bool = False
    embeddings: int = 0
