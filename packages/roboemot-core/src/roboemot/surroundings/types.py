"""Shared types of the surroundings pictures (SPEC 8, Surroundings snapshots)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A still frame of one view, shrunk to the sent size, with its sharpness score. The image
    stays in RAM and is never logged (SPEC 12)."""

    image: np.ndarray = field(repr=False)
    stamp: float
    sharpness: float
