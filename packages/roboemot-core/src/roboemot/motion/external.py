"""External control of the robot (SPEC 4A, 12A): the measured pose stops following the commands.

The motion loop owns one :class:`ExternalControl`. While it writes, every *new* state sample (the
driver's ``state_seq``; a repeated sample is no evidence) is compared, joint by joint, with the
range of the values commanded over the last ``lag_window_s x loop_hz`` ticks, excluding the
command written in the same tick (a sample read just after a write cannot show it yet). Rules,
from review R1 (B1) of the robot run where an overloaded CPU and an old state cache made the loop
yield falsely:

- **Direction.** A robot that lags, even behind an old state cache, sits on the commanded path
  behind the commands: on the side the path came from when it left the window. Outside the range
  on that side is lag, however far. Outside it on any other side by more than the tolerance
  counts. While the path holds still (it moved less than ``still_band`` over a window) both sides
  count, so a hand pushing a still head is caught; one pushing against a moving head is caught
  once the commands settle.
- **Samples, not time.** Divergence must hold on ``confirm_s x loop_hz`` new samples in a row. A
  tick later than ``late_tick_periods`` periods after the previous one (an overloaded CPU) starts
  the count again: a stall is no evidence.
- **Yield units.** A manifest group yields on its own, merged with every group a constraint ties
  it to (writing one side of a coupled pair while the other is pushed would fight the
  constraint). On Reachy Mini an antenna held by a hand stops the antennas only; head and body
  yaw yield together.

While a unit yields the loop writes none of its dofs and follows its measured pose. The yielding
units resume together, from the measured pose, once each one's pose has stayed inside
``still_band_*`` of an anchor for ``motion.platform_move_grace_s`` (the loop checks the motors
first; off means asleep). The recent events, with the largest tick gap and the overruns of the
confirming run, stay in the debug report.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from ..config.schema import ExternalControlConfig
from ..manifest import Constraint, Dof

PEAKS = (
    "err_rad", "err_m", "latest_rad", "latest_m",
    "diverged_s", "diverged_samples", "speed_rad", "speed_m",
)  # fmt: skip


@dataclass
class Unit:
    """Dofs that yield together, and their divergence count and stillness while yielding."""

    name: str
    dofs: tuple[str, ...]
    yielding: bool = False
    run: int = 0  # new divergent samples in a row
    run_since: float = 0.0
    run_gap_s: float = 0.0  # the largest tick gap during the run
    run_overruns: int = 0  # the loop's overrun count when the run started
    still_since: float | None = None
    anchor: dict[str, float] | None = None


def yield_units(dofs: dict[str, Dof], constraints: list[Constraint]) -> list[Unit]:
    """One unit per manifest group, merging the groups a constraint ties together."""
    parent = {d.group: d.group for d in dofs.values()}

    def root(group: str) -> str:
        while parent[group] != group:
            group = parent[group]
        return group

    for c in constraints:
        if c.a in dofs and c.b in dofs:
            parent[root(dofs[c.a].group)] = root(dofs[c.b].group)
    members: dict[str, list[str]] = {}
    for name, dof in dofs.items():
        members.setdefault(root(dof.group), []).append(name)
    return [
        Unit("+".join(sorted({dofs[n].group for n in names})), tuple(names))
        for names in members.values()
    ]


class _Window:
    """The values commanded for one dof over the last ticks, and the last two that left it."""

    __slots__ = ("values", "came_from", "before")

    def __init__(self, size: int) -> None:
        self.values: deque[float] = deque(maxlen=size)
        self.came_from: float | None = None
        self.before: float | None = None

    def push(self, value: float) -> None:
        if len(self.values) == self.values.maxlen:
            self.before, self.came_from = self.came_from, self.values[0]
        self.values.append(value)

    def reset(self, start: float | None) -> None:
        """Empty; a path that starts at ``start`` (the measured pose), not moving."""
        self.values.clear()
        self.came_from = self.before = start


class ExternalControl:
    """Detection and yield state for :class:`~roboemot.motion.loop.MotionLoop` (not thread-safe:
    the loop calls it under its lock)."""

    def __init__(
        self,
        dofs: dict[str, Dof],
        constraints: list[Constraint],
        cfg: ExternalControlConfig,
        rate_hz: float,
    ) -> None:
        self.units = yield_units(dofs, constraints)
        self.rate_hz = rate_hz
        self._kind = {n: "m" if d.kind == "translation" else "rad" for n, d in dofs.items()}
        self.events = 0
        self.recent: deque[dict[str, Any]] = deque(maxlen=cfg.recent_events)
        self.held: frozenset[str] = frozenset()
        self._last_tick: float | None = None
        self._last_seen: tuple[float, dict[str, float]] | None = None
        self.cfg = cfg
        self._derive(cfg)
        self.reset_peaks()

    # ---- configuration

    def configure(self, cfg: ExternalControlConfig) -> None:
        """Hot reload (SPEC 11A). The peaks are measured against these values, so they reset
        when any changes; the event record stays."""
        if cfg == self.cfg:
            return
        self.cfg = cfg
        self._derive(cfg)
        self.reset_peaks()
        self.recent = deque(self.recent, maxlen=cfg.recent_events)

    def _derive(self, cfg: ExternalControlConfig) -> None:
        size = max(1, math.ceil(cfg.lag_window_s * self.rate_hz))
        self._confirm = max(1, math.ceil(cfg.confirm_s * self.rate_hz))
        self._late_s = cfg.late_tick_periods / self.rate_hz
        self._tol = {
            n: cfg.tolerance_m if k == "m" else cfg.tolerance_rad for n, k in self._kind.items()
        }
        self._band = {
            n: cfg.still_band_m if k == "m" else cfg.still_band_rad for n, k in self._kind.items()
        }
        self._drift = {n: band / size for n, band in self._band.items()}  # per tick: "moving"
        self._windows = {n: _Window(size) for n in self._kind}
        for unit in self.units:
            unit.run = 0

    def reset_peaks(self) -> None:
        self.peak = dict.fromkeys(PEAKS, 0.0)

    # ---- state

    @property
    def yielding(self) -> bool:
        return bool(self.held)

    def reset(self, measured: dict[str, float]) -> None:
        """The loop re-aligned to the measured pose: commands start from there."""
        for name, window in self._windows.items():
            window.reset(measured.get(name))
        for unit in self.units:
            unit.run = 0

    def clear(self) -> None:
        """Motors off: asleep, nothing yields (the loop writes nothing anyway)."""
        for unit in self.units:
            unit.yielding, unit.run, unit.still_since, unit.anchor = False, 0, None, None
        for window in self._windows.values():
            window.reset(None)
        self._update_held()

    def _update_held(self) -> None:
        self.held = frozenset(n for u in self.units if u.yielding for n in u.dofs)

    # ---- one tick

    def tick(self, now: float) -> None:
        """Start of a writing tick: a late tick restarts every divergence count."""
        last, self._last_tick = self._last_tick, now
        if last is None:
            return
        gap = now - last
        for unit in self.units:
            if gap > self._late_s:
                unit.run = 0
            elif unit.run:
                unit.run_gap_s = max(unit.run_gap_s, gap)

    def push(self, out: dict[str, float]) -> None:
        """Record this tick's commands (after :meth:`judge`, which must not see them)."""
        for name, value in out.items():
            window = self._windows.get(name)
            if window is not None and name not in self.held:
                window.push(value)

    def judge(self, now: float, measured: dict[str, float], overruns: int) -> list[Unit]:
        """Compare a new sample with the windowed commands; returns the units that yield now."""
        started: list[Unit] = []
        for unit in self.units:
            if unit.yielding:
                continue
            worst = self._worst(unit, measured)
            if worst is None:
                unit.run = 0
                continue
            if unit.run == 0:
                unit.run_since, unit.run_gap_s, unit.run_overruns = now, 0.0, overruns
            unit.run += 1
            run_s = now - unit.run_since
            self.peak["diverged_s"] = max(self.peak["diverged_s"], run_s)
            self.peak["diverged_samples"] = max(self.peak["diverged_samples"], unit.run)
            if unit.run >= self._confirm:
                joint, error, tolerance = worst
                self._start_yield(unit, now, measured)
                self.recent.append(
                    {
                        "t": now,
                        "group": unit.name,
                        "joint": joint,
                        "error": error,
                        "tolerance": tolerance,
                        "samples": unit.run,
                        "run_s": run_s,
                        "max_gap_s": unit.run_gap_s,
                        "overruns": overruns - unit.run_overruns,
                    }
                )
                unit.run = 0
                started.append(unit)
        return started

    def _worst(self, unit: Unit, measured: dict[str, float]) -> tuple[str, float, float] | None:
        """The joint most beyond its tolerance, off the commanded path (not behind it)."""
        worst: tuple[str, float, float] | None = None
        for name in unit.dofs:
            value, window = measured.get(name), self._windows[name]
            if value is None or not window.values:
                continue
            kind, tol = self._kind[name], self._tol[name]
            latest = abs(value - window.values[-1])
            self.peak[f"latest_{kind}"] = max(self.peak[f"latest_{kind}"], latest)
            lo, hi = min(window.values), max(window.values)
            error = value - hi if value > hi else lo - value
            if error <= 0.0 or self._behind(name, window, value):
                continue
            self.peak[f"err_{kind}"] = max(self.peak[f"err_{kind}"], error)
            if error > tol and (worst is None or error / tol > worst[1] / worst[2]):
                worst = (name, error, tol)
        return worst

    def _behind(self, name: str, window: _Window, value: float) -> bool:
        """On the side the path came from when it left the window (where a lagging robot is)."""
        if window.came_from is None or window.before is None:
            return False
        step = window.came_from - window.before
        if abs(step) <= self._drift[name]:
            return False  # the path was still: no side is lag
        return (value - window.came_from) * step < 0.0

    def _start_yield(self, unit: Unit, now: float, measured: dict[str, float]) -> None:
        unit.yielding, unit.still_since, unit.anchor = True, None, None
        for name in unit.dofs:
            self._windows[name].reset(None)
        self._update_held()
        self.events += 1
        self._last_seen = (now, dict(measured))

    # ---- while yielding

    def watch(self, now: float, measured: dict[str, float], grace_s: float) -> list[Unit]:
        """Stillness of the yielding units. Each unit's pose must stay inside the band of its
        anchor (the pose where its count started; a joint beyond the band restarts it from the
        new pose, so encoder jitter or a slow cache does not). Returns every yielding unit once
        *all* of them have been still for ``grace_s``, counted from the latest start: one
        platform move drives several units, and the one that settles first must not resume
        while the move runs on (its targets would be accepted the moment the move ends, before
        the motors-off that follows it is seen). The caller checks the motors first."""
        last = self._last_seen
        self._last_seen = (now, dict(measured))
        if last is not None and now > last[0]:
            dt = now - last[0]
            for name in self.held:
                if name in measured and name in last[1]:
                    key = f"speed_{self._kind[name]}"
                    speed = abs(measured[name] - last[1][name]) / dt
                    self.peak[key] = max(self.peak[key], speed)
        yielding = [u for u in self.units if u.yielding]
        for unit in yielding:
            anchor = unit.anchor
            if anchor is not None and any(
                name in measured and abs(measured[name] - anchor[name]) > self._band[name]
                for name in unit.dofs
                if name in anchor
            ):
                anchor = None
            if anchor is None:
                unit.anchor = {n: measured[n] for n in unit.dofs if n in measured}
                unit.still_since = now
        starts = [u.still_since for u in yielding if u.still_since is not None]
        if not starts or now - max(starts) < grace_s:
            return []
        return yielding

    def resume(self, unit: Unit, measured: dict[str, float]) -> None:
        unit.yielding, unit.still_since, unit.anchor, unit.run = False, None, None, 0
        for name in unit.dofs:
            self._windows[name].reset(measured.get(name))
        self._update_held()

    # ---- report (SPEC 12A)

    def snapshot(self, now: float) -> dict[str, object]:
        yielding = [u for u in self.units if u.yielding]
        still = [now - u.still_since for u in yielding if u.still_since is not None]
        return {
            "external": bool(yielding),
            "external_groups": sorted(u.name for u in yielding),
            "external_events": self.events,
            "external_still_s": max(0.0, min(still)) if still else 0.0,
            "external_peak_err_rad": self.peak["err_rad"],
            "external_peak_err_m": self.peak["err_m"],
            "external_peak_err_latest_rad": self.peak["latest_rad"],
            "external_peak_err_latest_m": self.peak["latest_m"],
            "external_peak_diverged_s": self.peak["diverged_s"],
            "external_peak_diverged_samples": self.peak["diverged_samples"],
            "external_peak_speed_rad_s": self.peak["speed_rad"],
            "external_peak_speed_m_s": self.peak["speed_m"],
            "external_last_event": dict(self.recent[-1]) if self.recent else None,
            "external_recent_events": [dict(e) for e in self.recent],
        }
