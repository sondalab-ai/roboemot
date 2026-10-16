"""Pixel to bearing for a pinhole camera (SPEC 5A, Frame).

Sensor frame follows REP-103: x forward, y left, z up. Azimuth is positive left and elevation
positive up, so a pixel right of the image centre has negative azimuth and one above it has
positive elevation. The camera looks along x. Intrinsics come from the camera source when it
has them (``from_intrinsics``), else from the horizontal field of view with square pixels and
the principal point at the image centre; the fallback fields are ``world.camera_fov_h_deg`` and
``camera_fov_v_deg``. Lens distortion is ignored. The base-frame rotation
with the joint state is done by the world model, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics: from the frame size and field of view, or explicit fx, fy, cx, cy."""

    width: int
    height: int
    hfov_deg: float
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    vfov_deg: float | None = None

    @classmethod
    def from_intrinsics(
        cls, width: int, height: int, fx: float, fy: float, cx: float, cy: float
    ) -> CameraModel:
        """Model from a calibrated camera matrix already scaled to ``width`` x ``height``.

        ``hfov_deg`` is derived from ``fx`` so both views of the model agree.
        """
        fov = math.degrees(2 * math.atan((width / 2) / fx))
        return cls(width, height, fov, fx, fy, cx, cy)

    @property
    def focal_px(self) -> float:
        """Horizontal focal length in pixels (from fx, else from the field of view)."""
        if self.fx is not None:
            return self.fx
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def focal_y_px(self) -> float:
        """Vertical focal length in pixels (fy, else from the vertical field of view if known)."""
        if self.fy is not None:
            return self.fy
        if self.vfov_deg is not None:
            return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)
        return self.focal_px

    def bearing(self, u: float, v: float) -> tuple[float, float]:
        """Sensor-frame ``(azimuth, elevation)`` in radians of the ray through pixel ``(u, v)``.

        Pixels are in the original frame (not a resized detector input). The centre gives
        (0, 0); right of centre gives negative azimuth, above centre positive elevation.
        """
        cx = self.cx if self.cx is not None else self.width / 2
        cy = self.cy if self.cy is not None else self.height / 2
        y = -(u - cx) / self.focal_px
        z = -(v - cy) / self.focal_y_px
        return math.atan2(y, 1.0), math.asin(z / math.sqrt(1.0 + y * y + z * z))
