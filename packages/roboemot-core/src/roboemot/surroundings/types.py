"""Shared types of the surroundings pictures (SPEC 8, Surroundings snapshots)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..vision.types import Box


@dataclass(frozen=True)
class Candidate:
    """A still frame of one view, shrunk to the sent size, with its sharpness score and the
    face boxes of its own face run (in its pixels), which mask it. The image stays in RAM and
    is never logged (SPEC 12)."""

    image: np.ndarray = field(repr=False)
    stamp: float
    sharpness: float
    faces: tuple[Box, ...] = ()
