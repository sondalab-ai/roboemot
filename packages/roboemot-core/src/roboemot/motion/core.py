"""Reference arbiter and safety limiter (SPEC 10, 12).

This is the executable specification of ``roboemot-rt``: the C++ core implements the same
semantics and is checked against golden vectors produced from this module
(``scripts/gen_golden.py``). Units: rad, m, s.

Arbitration (SPEC 10, Sources and priority):
- every command carries a source id, a priority and targets for some dofs;
- per dof, the absolute command with the highest priority (lowest number) wins, ties go to
  the most recent, then to the smallest source id (deterministic, same as the C++ core);
- additive commands (expression) are added on top of a winner of priority COMMANDED or lower
  precedence, never on top of SAFETY or REFLEX;
- when the winning source of a dof changes, the output cross-fades over ``crossfade_s`` with a
  min-jerk blend, except when the new winner is SAFETY (immediate);
- with no command, a dof holds its home position.

Limiting (SPEC 12, Physical safety), applied to the arbitrated target every tick:
position clamp, coupled constraints (yaw_delta), then velocity and acceleration caps with a
stopping-distance rule so motion decelerates before the target, then a hard position clamp
and rate-limited constraint coupling. While a minor is present, the velocity cap and the
amplitude around home are scaled by ``minor_factor``; acceleration (braking) never is.
Non-finite targets and non-positive ``dt`` are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from ..manifest import Constraint, Dof
from .curves import min_jerk

STOP_SEARCH_ITERATIONS = 40
# Relative slack when checking that a final landing respects the acceleration cap.
LAND_TOLERANCE = 1.0 + 1e-9


def stop_distance(v: float, adt: float, dt: float) -> float:
    """Distance covered while braking from speed ``v`` by ``adt`` per tick (this tick included)."""
    n = math.floor(v / adt)
    return dt * ((n + 1) * v - adt * n * (n + 1) / 2.0)


def stop_speed(distance: float, amax: float, dt: float) -> float:
    """Largest speed from which a discrete-time loop still stops within ``distance``.

    Exact for the discrete braking sequence (no overshoot), found by bisection on the
    monotonic :func:`stop_distance`; the C++ core uses the same search.
    """
    if amax <= 0.0 or dt <= 0.0 or distance <= 0.0:
        return 0.0
    adt = amax * dt
    lo, hi = 0.0, adt * (math.sqrt(2.0 * distance / (adt * dt) + 0.25) + 0.5)
    for _ in range(STOP_SEARCH_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if stop_distance(mid, adt, dt) <= distance:
            lo = mid
        else:
            hi = mid
    return lo


class Priority(IntEnum):
    SAFETY = 0
    REFLEX = 1
    COMMANDED = 2
    MIMICRY = 3
    EXPRESSION = 4
    IDLE = 5


@dataclass
class Command:
    source: str
    priority: Priority
    targets: dict[str, float]
    stamp: float
    expires: float | None = None  # absolute time; None = until replaced or cleared
    additive: bool = False


@dataclass
class _Fade:
    start: float
    from_value: float
    winner: str


class Arbiter:
    def __init__(self, dofs: dict[str, Dof], crossfade_s: float) -> None:
        self.dofs = dofs
        self.crossfade_s = crossfade_s
        self._cmds: dict[str, Command] = {}
        self._winner: dict[str, str | None] = {d: None for d in dofs}
        self._fade: dict[str, _Fade | None] = {d: None for d in dofs}
        self.last: dict[str, float] = {name: d.home for name, d in dofs.items()}

    def submit(self, cmd: Command) -> None:
        unknown = set(cmd.targets) - set(self.dofs)
        if unknown:
            raise KeyError(f"unknown dofs {sorted(unknown)}")
        _require_finite(cmd.targets)
        if not math.isfinite(cmd.stamp):
            raise ValueError("non-finite command stamp")
        self._cmds[cmd.source] = cmd

    def clear(self, source: str) -> None:
        self._cmds.pop(source, None)

    def active(self, now: float) -> list[Command]:
        for src in [s for s, c in self._cmds.items() if c.expires is not None and c.expires <= now]:
            del self._cmds[src]
        return list(self._cmds.values())

    def step(self, now: float, amplitude: float = 1.0) -> dict[str, float]:
        cmds = self.active(now)
        out: dict[str, float] = {}
        for name, dof in self.dofs.items():
            absolute = [c for c in cmds if not c.additive and name in c.targets]
            if absolute:
                best = min(absolute, key=lambda c: (c.priority, -c.stamp, c.source))
                target, winner, prio = best.targets[name], best.source, best.priority
            else:
                target, winner, prio = dof.home, None, Priority.IDLE
            if prio >= Priority.COMMANDED:
                target += sum(c.targets[name] for c in cmds if c.additive and name in c.targets)
            if prio != Priority.SAFETY:
                # Clamp the summed target before the minor scaling, so an additive offset on a
                # winner near a limit cannot leave the band as factor x overflow (SPEC 12).
                target = min(max(target, dof.min), dof.max)
                target = dof.home + (target - dof.home) * amplitude
            if winner != self._winner[name]:
                self._fade[name] = (
                    None if prio == Priority.SAFETY else _Fade(now, self.last[name], winner or "")
                )
                self._winner[name] = winner
            fade = self._fade[name]
            if fade is not None and self.crossfade_s > 0:
                s = (now - fade.start) / self.crossfade_s
                if s >= 1.0:
                    self._fade[name] = None
                else:
                    w = min_jerk(s)
                    target = fade.from_value + (target - fade.from_value) * w
            out[name] = target
        self.last = dict(out)
        return out


@dataclass
class Limiter:
    dofs: dict[str, Dof]
    constraints: list[Constraint] = field(default_factory=list)
    # Coupled constraints are planned this much inside their limit (rad), so the transient
    # lag of an acceleration-limited follower never exceeds the real limit.
    constraint_margin: float = 0.0

    def __post_init__(self) -> None:
        self.pos: dict[str, float] = {n: d.home for n, d in self.dofs.items()}
        self.vel: dict[str, float] = {n: 0.0 for n in self.dofs}

    def reset(self, positions: dict[str, float]) -> None:
        _require_finite(positions)
        for n, p in positions.items():
            if n in self.pos:
                self.pos[n] = p
                self.vel[n] = 0.0

    def clamp(self, targets: dict[str, float]) -> dict[str, float]:
        out = {n: min(max(v, self.dofs[n].min), self.dofs[n].max) for n, v in targets.items()}
        for c in self.constraints:
            if c.type == "yaw_delta" and c.a in out and c.b in out:
                lim = math.radians(c.max_deg)
                out[c.a] = min(max(out[c.a], out[c.b] - lim), out[c.b] + lim)
        return out

    def step(self, targets: dict[str, float], dt: float, factor: float = 1.0) -> dict[str, float]:
        """One tick. ``factor`` in (0, 1] scales the velocity cap only: braking always uses
        the full acceleration, so lowering the factor mid-motion (a minor appears) can never
        weaken deceleration."""
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        if not (math.isfinite(factor) and 0 < factor <= 1):
            raise ValueError(f"factor must be in (0, 1], got {factor}")
        _require_finite(targets)
        goal = self.clamp(targets)
        # Coupled constraints act on goals: each constrained dof aims inside the band around
        # the *current* position of the other, so normal (velocity-, acceleration- and
        # limit-respecting) motion keeps the pair together. No post-hoc position edits.
        for c in self.constraints:
            if c.type != "yaw_delta" or c.a not in self.pos or c.b not in self.pos:
                continue
            lim = max(0.0, math.radians(c.max_deg) - self.constraint_margin)
            for u, w in ((c.a, c.b), (c.b, c.a)):
                if u in goal:
                    du = self.dofs[u]
                    g = min(max(goal[u], self.pos[w] - lim), self.pos[w] + lim)
                    goal[u] = min(max(g, du.min), du.max)
        prev = dict(self.pos)
        for name, dof in self.dofs.items():
            if name not in goal:
                continue
            vmax, amax = dof.max_vel * factor, dof.max_acc
            x, v = self.pos[name], self.vel[name]
            err = goal[name] - x
            v_cap = min(vmax, stop_speed(abs(err), amax, dt))
            v_des = max(-v_cap, min(v_cap, err / dt))
            dv = max(-amax * dt, min(amax * dt, v_des - v))
            x_new = x + (v + dv) * dt
            # Land exactly on the goal only if that stop is within the acceleration cap;
            # a goal that appears inside the braking distance is overshot and braked back to.
            if (goal[name] - x_new) * err < 0 and abs(err / dt - v) <= amax * dt * LAND_TOLERANCE:
                x_new = goal[name]
            self.pos[name] = x_new
        # Hard position limits (limits win over smoothness). A dof already inside never leaves;
        # one that starts outside (pose left by another app) is never pushed further out and
        # comes in at limited speed, never snapped.
        for name, dof in self.dofs.items():
            x0, x = prev[name], self.pos[name]
            if dof.min <= x0 <= dof.max:
                x = min(max(x, dof.min), dof.max)
            elif x0 > dof.max:
                x = min(x, x0)
            else:
                x = max(x, x0)
            self.pos[name] = x
        for name in self.dofs:
            self.vel[name] = (self.pos[name] - prev[name]) / dt
        return dict(self.pos)


def _require_finite(values: dict[str, float]) -> None:
    bad = [k for k, v in values.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite targets for {sorted(bad)}")


class MotionCore:
    """Arbiter + limiter + watchdog, one tick per call (SPEC 10, Motor loop)."""

    def __init__(
        self,
        dofs: dict[str, Dof],
        constraints: list[Constraint],
        *,
        crossfade_s: float,
        watchdog_s: float,
        minor_factor: float,
        watchdog_return_factor: float,
        constraint_margin: float = 0.0,
    ) -> None:
        self.arbiter = Arbiter(dofs, crossfade_s)
        self.limiter = Limiter(dofs, constraints, constraint_margin)
        self.watchdog_s = watchdog_s
        self.minor_factor = minor_factor
        self.watchdog_return_factor = watchdog_return_factor
        self.minor_present = False
        self._last_feed: float | None = None
        self.holding = False

    def feed(self, cmd: Command) -> None:
        self.arbiter.submit(cmd)
        self._last_feed = cmd.stamp

    def tick(self, now: float, dt: float) -> dict[str, float]:
        if self._last_feed is not None and now - self._last_feed > self.watchdog_s:
            # Watchdog (SPEC 10): return home slowly, at SAFETY priority.
            if not self.holding:
                home = {n: d.home for n, d in self.arbiter.dofs.items()}
                self.arbiter.submit(Command("watchdog", Priority.SAFETY, home, now))
                self.holding = True
        elif self.holding:
            self.arbiter.clear("watchdog")
            self.holding = False
        factor = self.minor_factor if self.minor_present else 1.0
        targets = self.arbiter.step(now, amplitude=factor)
        speed = min(factor, self.watchdog_return_factor) if self.holding else factor
        return self.limiter.step(targets, dt, speed)
