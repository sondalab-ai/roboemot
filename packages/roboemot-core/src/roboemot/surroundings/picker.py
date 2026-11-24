"""The snapshot picker (SPEC 8, Surroundings snapshots; design 2026-11-24, 5.3).

A frame taken while the head moves, or while a shown thing moves, is skipped with that reason.
The other frames join a window of at most ``surroundings.candidates`` frames from
``surroundings.window_s``; when the window closes the sharpest frame wins if it scores at least
``surroundings.min_sharpness``, else the window clears (``too_blurry``) and the attempt goes on.
At ``wait_s`` the attempt ends with its last skip reason. Not thread-safe: the vision thread
offers frames, its owner reads ``result`` under its own lock.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..config.schema import SurroundingsConfig
from .types import Candidate, Skip

SHARPNESS_SIDE_PX = 256  # the grey copy is scored at most this long, so a big crop costs little
COLOUR_DIMS = 3  # HxWxC
LAPLACIAN_SIDE = 3  # the Laplacian needs at least this many pixels per side
LAPLACIAN_CENTRE = 4.0  # the centre weight of the 4-neighbour Laplacian


def sharpness(image: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of a grey copy of ``image`` (HxW or HxWxC),
    strided down to at most ``SHARPNESS_SIDE_PX`` on its longest side. Higher is sharper; a
    flat or blurred image scores near 0."""
    step = max(1, math.ceil(max(image.shape[:2]) / SHARPNESS_SIDE_PX))
    small = image[::step, ::step]
    gray = small.mean(axis=-1) if small.ndim == COLOUR_DIMS else small.astype(np.float32)
    if min(gray.shape) < LAPLACIAN_SIDE:
        return 0.0
    lap = (
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        - LAPLACIAN_CENTRE * gray[1:-1, 1:-1]
    )
    return float(lap.var())


class Picker:
    """One snapshot attempt. ``cfg`` gives the live surroundings settings, ``wait_s`` the
    attempt's longest wait (``shown.wait_s`` or ``room.wait_s``), ``now`` its start."""

    def __init__(self, cfg: Callable[[], SurroundingsConfig], wait_s: float, now: float) -> None:
        self.cfg, self.wait_s, self.start = cfg, wait_s, now
        self.window: list[Candidate] = []
        self.last_skip: Skip | None = None

    def offer(
        self, image: np.ndarray | None, stamp: float, *, head_moving: bool, object_moving: bool
    ) -> None:
        """Offer one shaped frame (None: no frame could be shaped)."""
        if image is None:
            self.last_skip = "no_frame"
        elif head_moving:
            self.last_skip = "head_moving"
        elif object_moving:
            self.last_skip = "object_moving"
        else:
            self.window.append(Candidate(image, stamp, sharpness(image)))

    def result(self, now: float) -> Candidate | Skip | None:
        """The chosen frame, the reason the attempt ended without one, or None while it goes
        on."""
        cfg = self.cfg()
        if self.window and (
            len(self.window) >= cfg.candidates or now - self.window[0].stamp >= cfg.window_s
        ):
            best = max(self.window, key=lambda c: c.sharpness)
            self.window = []
            if best.sharpness >= cfg.min_sharpness:
                return best
            self.last_skip = "too_blurry"
        if now - self.start >= self.wait_s:
            return self.last_skip or "no_frame"
        return None
