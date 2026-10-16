"""Frame transform for the world model (SPEC 5A, Frame).

Bearings are captured in the sensor frame (azimuth positive to the left, elevation positive
up, radians). With the head or body turned, they are rotated into the robot base frame using
the joint state at capture time: pitch first, then yaw. Which dofs count as yaw or pitch is
data (``FrameSpec``), not hard-coded, so any robot manifest can supply it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    """Names of the manifest dofs that turn the sensor: their angles add up per axis."""

    yaw_dofs: tuple[str, ...] = ()
    pitch_dofs: tuple[str, ...] = ()


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    w = math.fmod(a + math.pi, 2 * math.pi)
    if w <= 0:
        w += 2 * math.pi
    return w - math.pi


def to_base_frame(
    azimuth_rad: float,
    elevation_rad: float,
    joint_state: Mapping[str, float] | None,
    spec: FrameSpec,
) -> tuple[float, float]:
    """Rotate a sensor-frame bearing into the base frame; missing dofs count as zero."""
    js = joint_state or {}
    yaw = sum(js.get(n, 0.0) for n in spec.yaw_dofs)
    pitch = sum(js.get(n, 0.0) for n in spec.pitch_dofs)
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, z = x * cp - z * sp, x * sp + z * cp
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y = x * cy - y * sy, x * sy + y * cy
    return wrap_angle(math.atan2(y, x)), math.asin(max(-1.0, min(1.0, z)))


def angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angle between two bearings, radians."""
    c = math.sin(el1) * math.sin(el2) + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    return math.acos(max(-1.0, min(1.0, c)))
