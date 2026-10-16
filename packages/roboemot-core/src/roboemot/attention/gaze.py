"""Gaze: attention target -> head and body pose commands (SPEC 10, Sources and priority).

Convention (REP-103, ``world/frame.py``): the target bearing is in the robot base frame,
azimuth positive left, elevation positive up; head yaw is a base-frame angle, and a positive
pitch points the nose down, so a target above the horizon gives a negative pitch.

Yaw split: the head takes the target azimuth (clamped to its own limits and to what the body
can back up). The body stays put while the head is within ``head_yaw_comfort_deg`` of it
(never more than the manifest's ``yaw_delta`` constraint, less the safety margin) and turns
only by the excess. Limits come from the manifest; the commands still go through the arbiter
and the safety limiter, which own speed and acceleration. Which dofs are the gaze dofs is data
(``GazeMap``) supplied by the robot plugin.

Body recentring: while the head is within the comfort band, the body drifts back towards its
home at ``body_recenter_dps`` (never so far that the head leaves the band). Straight behind
(``rear_band_deg`` around +-180 degrees) the gaze keeps the side it is on, so a bearing
jittering across the seam does not swing the head or body through 360 degrees.

Smoothing: an exponential approach with ``follow_gain`` capped at ``max_rate_dps``. Attention
tracking sits at idle priority (SPEC 10): any commanded, mimicry or expression motion on the
same dof wins. When there is no target for ``release_s`` the controller stops and reports
``active == False``; the caller then clears the source so idle motion takes over.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..config.schema import AttentionConfig
from ..manifest import Constraint, Dof
from ..motion import Command, Priority
from .selector import AttentionTarget

SOURCE = "attention"


@dataclass(frozen=True)
class GazeMap:
    """Names of the manifest dofs that aim the gaze; ``body_yaw`` is None without a body turn."""

    head_yaw: str
    head_pitch: str
    body_yaw: str | None = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def yaw_reach(gaze: GazeMap, constraints: list[Constraint], margin_rad: float = 0.0) -> float:
    """Largest head-minus-body yaw offset the manifest allows (less the margin); inf if none."""
    reach = math.inf
    for c in constraints:
        if c.type == "yaw_delta" and {c.a, c.b} == {gaze.head_yaw, gaze.body_yaw}:
            reach = min(reach, max(0.0, math.radians(c.max_deg) - margin_rad))
    return reach


def look_at(
    azimuth_rad: float,
    elevation_rad: float,
    gaze: GazeMap,
    dofs: Mapping[str, Dof],
    constraints: list[Constraint],
    *,
    comfort_rad: float,
    margin_rad: float = 0.0,
    body_now: float = 0.0,
) -> dict[str, float]:
    """Dof targets that point the head at a base-frame bearing, within the manifest limits.

    ``body_now`` is the body yaw the split starts from. Within ``comfort_rad`` of it the body
    target is the point of the comfort band nearest to the body's home (a recentring drift);
    beyond it the body turns by the excess. Targets beyond reach are clamped, never wrapped.
    """
    head, pitch = dofs[gaze.head_yaw], dofs[gaze.head_pitch]
    out = {gaze.head_pitch: _clamp(-elevation_rad, pitch.min, pitch.max)}
    if gaze.body_yaw is None:
        out[gaze.head_yaw] = _clamp(azimuth_rad, head.min, head.max)
        return out
    body = dofs[gaze.body_yaw]
    reach = yaw_reach(gaze, constraints, margin_rad)
    yaw = _clamp(azimuth_rad, max(head.min, body.min - reach), min(head.max, body.max + reach))
    comfort = min(comfort_rad, reach)
    offset = yaw - body_now
    if abs(offset) > comfort:
        target_body = yaw - math.copysign(comfort, offset)
    else:
        target_body = _clamp(body.home, yaw - comfort, yaw + comfort)
    out[gaze.head_yaw] = yaw
    out[gaze.body_yaw] = _clamp(target_body, max(body.min, yaw - reach), min(body.max, yaw + reach))
    return out


class GazeController:
    """Turns the attention target into smoothed ``Command``s for the motion loop."""

    def __init__(
        self,
        cfg: AttentionConfig,
        gaze: GazeMap,
        dofs: Mapping[str, Dof],
        constraints: list[Constraint],
        *,
        margin_rad: float = 0.0,
    ) -> None:
        names = self._names(gaze)
        if missing := [n for n in names if n not in dofs]:
            raise KeyError(f"gaze dofs not in the manifest: {missing}")
        self.cfg, self.gaze, self.dofs = cfg, gaze, dofs
        self.constraints, self.margin_rad = constraints, margin_rad
        self.active = False
        self._pos: dict[str, float] = {}
        self._aim: dict[str, float] = {}
        self._last = 0.0
        self._lost_since: float | None = None
        self._side = 0.0
        self._recentering = False

    @staticmethod
    def _names(gaze: GazeMap) -> list[str]:
        return [gaze.head_yaw, gaze.head_pitch] + ([gaze.body_yaw] if gaze.body_yaw else [])

    def step(
        self,
        now: float,
        target: AttentionTarget | None,
        joint_state: Mapping[str, float] | None = None,
    ) -> Command | None:
        """One tick: the command to feed, or None when idle (never started, or just released).

        ``joint_state`` seeds the start of a new acquisition from the measured pose, so the
        head never snaps; it is ignored while the gaze is active (callers may then pass None).
        """
        if target is None:
            if not self.active:
                return None
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since >= self.cfg.release_s:
                self.release()
                return None
        else:
            self._lost_since = None
            if not self.active:
                self._start(now, joint_state)
            comfort = math.radians(self.cfg.head_yaw_comfort_deg)
            body_now = self._pos.get(self.gaze.body_yaw or "", 0.0)
            self._aim = look_at(
                self._sticky_azimuth(target.azimuth_rad),
                target.elevation_rad,
                self.gaze,
                self.dofs,
                self.constraints,
                comfort_rad=comfort,
                margin_rad=self.margin_rad,
                body_now=body_now,
            )
            reach = yaw_reach(self.gaze, self.constraints, self.margin_rad)
            self._recentering = self.gaze.body_yaw is not None and abs(
                self._aim[self.gaze.head_yaw] - body_now
            ) <= min(comfort, reach)
        self._approach(now)
        return Command(
            SOURCE, Priority.IDLE, dict(self._pos), now, expires=now + self.cfg.command_ttl_s
        )

    def release(self) -> None:
        """Stop emitting and forget the smoothed pose."""
        self.active = False
        self._pos, self._aim, self._lost_since = {}, {}, None
        self._side = 0.0
        self._recentering = False

    def _sticky_azimuth(self, az: float) -> float:
        """Inside the rear band keep the side the head (else body) is on, not the bearing's."""
        if abs(az) <= math.pi - math.radians(self.cfg.rear_band_deg):
            self._side = 0.0
            return az
        if self._side == 0.0:
            body = self._pos.get(self.gaze.body_yaw or "", 0.0)
            ref = next((v for v in (self._pos[self.gaze.head_yaw], body) if v), az)
            self._side = math.copysign(1.0, ref)
        return math.copysign(abs(az), self._side)

    def _start(self, now: float, joint_state: Mapping[str, float] | None) -> None:
        js = joint_state or {}
        self._pos = {n: js.get(n, self.dofs[n].home) for n in self._names(self.gaze)}
        self._last = now
        self.active = True

    def _approach(self, now: float) -> None:
        """Move the smoothed pose towards the aim: exponential, capped at ``max_rate_dps``."""
        dt = min(max(0.0, now - self._last), self.cfg.command_ttl_s)
        self._last = now
        alpha = 1.0 - math.exp(-self.cfg.follow_gain * dt)
        cap = math.radians(self.cfg.max_rate_dps) * dt
        recenter_cap = math.radians(self.cfg.body_recenter_dps) * dt
        for name, goal in self._aim.items():
            limit = recenter_cap if self._recentering and name == self.gaze.body_yaw else cap
            self._pos[name] += _clamp((goal - self._pos[name]) * alpha, -limit, limit)
