"""Frame transform for the world model (SPEC 5A, Frame).

Convention (REP-103): x forward, y left, z up, right-handed rotations. Azimuth is measured
from x toward y (positive left), elevation from the xy plane toward z (positive up). A
positive pitch is a right-handed rotation about y, so it points the nose down; positive
roll is about x, positive yaw about z (turn left). This matches the Reachy Mini driver,
which reads head angles as scipy Euler ``xyz`` (extrinsic: roll, then pitch, then yaw).

Bearings are captured in the sensor frame and rotated into the base frame with the joint
state at capture time: ``R = Rz(yaw) Ry(pitch) Rx(roll)``. Which dofs count per axis is
data (``FrameSpec``), not hard-coded; angles of several dofs on one axis add up. Limitation:
a sensor offset from the rotation centre (parallax) is ignored in v1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    """Names of the manifest dofs that turn the sensor, per axis."""

    yaw_dofs: tuple[str, ...] = ()
    pitch_dofs: tuple[str, ...] = ()
    roll_dofs: tuple[str, ...] = ()


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
    roll = sum(js.get(n, 0.0) for n in spec.roll_dofs)
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    y, z = y * cr - z * sr, y * sr + z * cr
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, z = x * cp + z * sp, -x * sp + z * cp
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y = x * cy - y * sy, x * sy + y * cy
    return wrap_angle(math.atan2(y, x)), math.asin(max(-1.0, min(1.0, z)))


def angular_distance(az1: float, el1: float, az2: float, el2: float) -> float:
    """Great-circle angle between two bearings, radians."""
    c = math.sin(el1) * math.sin(el2) + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    return math.acos(max(-1.0, min(1.0, c)))
