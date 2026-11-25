"""The view picker (SPEC 8, Surroundings snapshots; designs 2026-11-24, 5.3; 2026-11-25).

One picker per view of a photo sweep. A frame taken while the head moves is skipped. The other
frames join a window of at most ``surroundings.candidates`` frames from ``surroundings.window_s``;
when the window closes the sharpest frame wins if it scores at least
``surroundings.min_sharpness``, else the window clears and the view goes on. The dwell bounds the
view, not the picker: when it ends, ``best`` takes what the open window holds. Not thread-safe:
the vision thread offers frames under its owner's lock.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..config.schema import SurroundingsConfig
from .types import Candidate

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
    """The frames of one view. ``cfg`` gives the live surroundings settings."""

    def __init__(self, cfg: Callable[[], SurroundingsConfig]) -> None:
        self.cfg = cfg
        self.window: list[Candidate] = []

    def offer(self, image: np.ndarray, stamp: float, *, head_moving: bool) -> None:
        """Offer one shrunk frame; skipped while the head moves."""
        if not head_moving:
            self.window.append(Candidate(image, stamp, sharpness(image)))

    def result(self, now: float) -> Candidate | None:
        """The sharpest frame of a window that closed at ``now``, if it is sharp enough; None
        while the window is open or when it closed with every frame blurred (it clears)."""
        cfg = self.cfg()
        if not self.window or (
            len(self.window) < cfg.candidates and now - self.window[0].stamp < cfg.window_s
        ):
            return None
        best = self.best()
        self.window = []
        return best

    def best(self) -> Candidate | None:
        """The sharpest frame of the open window at or above ``min_sharpness``, or None (the
        window stays)."""
        if not self.window:
            return None
        best = max(self.window, key=lambda c: c.sharpness)
        return best if best.sharpness >= self.cfg().min_sharpness else None
