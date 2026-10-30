"""World model types (SPEC 5A, Observations and sources; Entries).

Bearings are azimuth (positive left) and elevation (positive up) in radians. Observations
carry them in the sensor frame; entries store them in the robot base frame. Appearance
embeddings are biometric-adjacent data: they are excluded from ``repr`` and from every
``world.*`` payload (SPEC 12). Only an animal or object observation on ``percept.entity``
carries one, as input to matching; person observations never do.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, get_args

import numpy as np

ObservationKind = Literal["person", "animal", "object", "unknown"]
EntryKind = Literal["person", "animal", "object"]
UNKNOWN_IDENTITY = "unknown"


@dataclass
class Observation:
    """One detection or sound bearing from a source (published on ``percept.entity``)."""

    kind: ObservationKind
    stamp: float
    source: str
    azimuth_rad: float
    elevation_rad: float
    category: str = ""
    identity: str | None = None
    distance_m: float | None = None
    confidence: float = 1.0
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    facing_yaw_rad: float | None = None  # a face's head yaw, 0 looking at the camera (SPEC 8)

    def __post_init__(self) -> None:
        if self.kind not in get_args(ObservationKind):
            raise ValueError(f"unknown observation kind: {self.kind!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("observation confidence must be in [0, 1]")


@dataclass
class Entry:
    """A tracked person, animal or object, positioned in the robot base frame."""

    entry_id: str
    kind: EntryKind
    category: str
    identity: str
    azimuth_rad: float
    elevation_rad: float
    distance_m: float | None
    uncertainty_rad: float
    confidence: float
    first_seen: float
    last_seen: float
    sources: set[str] = field(default_factory=set)
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    facing_yaw_rad: float | None = None  # last head yaw a face observation gave (SPEC 8)
    facing_stamp: float | None = None  # its capture stamp

    def public(self) -> Entry:
        """A copy safe to publish: no embedding, own ``sources`` set."""
        return replace(self, embedding=None, sources=set(self.sources))


@dataclass
class ObserveResult:
    """Outcome of ``WorldModel.observe``.

    ``entry`` is the matched or created entry (None for an unexplained bearing).
    ``unexplained_bearing`` is the (azimuth, elevation) in the base frame of a bearing-only
    observation that matched nothing: the caller may trigger a glance (SPEC 5A, Scanning).
    ``rejected`` is True when a new entry would start at or below ``forget_confidence``.
    ``entry`` is always a public copy (no embedding), a snapshot at the time of the call.
    """

    entry: Entry | None
    created: bool = False
    rejected: bool = False
    unexplained_bearing: tuple[float, float] | None = None
