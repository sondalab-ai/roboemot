"""Shared types of the surroundings snapshots (SPEC 8, Surroundings snapshots)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

Kind = Literal["shown", "room"]
Skip = Literal["head_moving", "object_moving", "too_blurry", "no_frame"]
SKIPS: tuple[str, ...] = get_args(Skip)


@dataclass(frozen=True)
class Candidate:
    """A still frame, already shaped (faces masked, cropped), with its sharpness score. The
    image stays in RAM and is never logged (SPEC 12)."""

    image: np.ndarray = field(repr=False)
    stamp: float
    sharpness: float
