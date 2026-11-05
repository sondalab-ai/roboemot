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
class FaceMarks:
    """One face's landmarks for the live preview (SPEC 11A): the 5 detector points (eyes,
    nose, mouth corners) in camera pixels, None without them, and the published head yaw
    (degrees, None without one). Only for the local preview, never stored or logged (SPEC 12)."""

    points: tuple[tuple[float, float], ...] | None
    yaw_deg: float | None


@dataclass(frozen=True)
class PreviewOverlay:
    """The live preview's boxes for one frame (SPEC 11A): face tracks as ``(box, owner_id or
    None)``, confirmed animal and object tracks as ``(box, readable category, owner_id or
    None)``. ``faces_current`` is False when face detection did not run on the frame (its face
    boxes are unknown, ``faces`` is empty): ``LatestFrameRunner`` then holds the last frame
    with face boxes for a poll period. Object boxes are the tracker's latest estimate, never
    an exact match of the picture, so they never hold a frame. ``marks`` is empty unless
    ``webui.live.video_landmarks`` is on; then it has one ``FaceMarks`` per face, in ``faces``
    order."""

    faces: tuple[tuple[Box, str | None], ...] = ()
    objects: tuple[tuple[Box, str, str | None], ...] = ()
    faces_current: bool = True
    marks: tuple[FaceMarks, ...] = ()

    def __bool__(self) -> bool:
        """True when it holds a face box: what makes a kept frame worth holding."""
        return bool(self.faces)


@dataclass(frozen=True)
class FrameReport:
    """What ``VisionPipeline.process`` did with a frame (for tests, replay and the governor)."""

    stamp: float
    faces_detected: bool = False
    objects_detected: bool = False
    embeddings: int = 0  # face embeds run, plus appearance embeds run inline or queued (not run)
