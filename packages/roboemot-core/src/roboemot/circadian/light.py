"""Ambient light from camera frames (SPEC 7A, Light sensing from the camera).

Auto-exposure normalises brightness, so mean luminance alone does not measure room light.
With exposure time and gain the estimate is ``k * luma / (exposure * gain)``; without them,
a dim image that is also noisy (high sensor gain) is classed as dark. Only aggregates leave
this module: no frame is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..config.schema import LightConfig

Level = Literal["dark", "dim", "bright"]
UINT8_MAX = 255.0
COLOR_NDIM = 3  # H x W x channels
# ITU-R BT.601 luma weights
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


@dataclass
class LightReading:
    level: Level
    luma: float  # normalised mean luminance of the frame, 0..1
    noise: float  # high-frequency noise estimate, 0..1
    index: float  # brightness index the thresholds were applied to


class LightEstimator:
    def __init__(self, cfg: LightConfig) -> None:
        self.cfg = cfg
        self.level: Level | None = None

    @staticmethod
    def stats(frame: np.ndarray) -> tuple[float, float]:
        """Mean luminance and noise of a frame (RGB channel order for a colour image)."""
        img = np.asarray(frame, dtype=np.float64) / UINT8_MAX
        luma_img = img @ LUMA_WEIGHTS if img.ndim == COLOR_NDIM else img
        luma = float(luma_img.mean())
        # Noise: robust spread of horizontal pixel differences (flat regions dominate).
        noise = float(np.median(np.abs(np.diff(luma_img, axis=1))))
        return luma, noise

    def measure(
        self, frame: np.ndarray, *, exposure_s: float | None = None, gain: float | None = None
    ) -> LightReading:
        luma, noise = self.stats(frame)
        return self.classify(luma, noise, exposure_s=exposure_s, gain=gain)

    def classify(
        self,
        luma: float,
        noise: float,
        *,
        exposure_s: float | None = None,
        gain: float | None = None,
    ) -> LightReading:
        """Classify already-measured statistics (a logged sample); ``measure`` ends here, so a
        replayed light log goes through the same thresholds and hysteresis as a live frame."""
        if exposure_s and gain and self.cfg.calibration_k:
            index = self.cfg.calibration_k * luma / (exposure_s * gain)
        else:
            index = luma
            if luma < self.cfg.bright_luma and noise > self._noise_edge():
                index = 0.0  # pumped sensor gain: the room is dark whatever the history
        self.level = self._classify(index)
        return LightReading(self.level, luma, noise, index)

    def _noise_edge(self) -> float:
        """The noise above which a dim image reads dark: lower while it already reads dark, so
        sensor noise hovering at the threshold does not toggle the level."""
        edge, h = self.cfg.noise_dark_threshold, self.cfg.noise_hysteresis
        return edge - h if self.level == "dark" else edge + h if self.level else edge

    def _classify(self, index: float) -> Level:
        c, h, prev = self.cfg, self.cfg.hysteresis, self.level
        dark_edge = c.dark_luma + (h if prev == "dark" else -h if prev else 0.0)
        bright_edge = c.bright_luma + (-h if prev == "bright" else h if prev else 0.0)
        if index < dark_edge:
            return "dark"
        if index > bright_edge:
            return "bright"
        return "dim"
