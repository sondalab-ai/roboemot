"""External control of the robot (SPEC 4A, 12A): the measured pose stops following the commands.

The motion loop owns one :class:`ExternalControl`. While it writes, every *new* state sample (the
driver's ``state_seq``; a repeated sample is no evidence) is compared, joint by joint, with the
range of the values commanded over the last ``lag_window_s x loop_hz`` ticks, excluding the
command written in the same tick (a sample read just after a write cannot show it yet). Rules,
from review R1 (B1) of the robot run where an overloaded CPU and an old state cache made the loop
yield falsely:

- **Lag.** A sample outside the range by more than the tolerance is still lag when (a) the joint
  moved toward the range by more than ``still_band`` since it last moved (closing in, behind a
  slow servo or an old state cache: the daemon's queue alone can hold ~0.67 s); (b) it lies
  within the tolerance of the range commanded over ``lag_window_s + max_state_age_s`` (an old
  sample may still show the robot there); or (c) it is behind the commands, on the side the path
  came from when it left the window, and moved within ``max_state_age_s``. A hand, or a platform
  move, holds the joint still or takes it away. The bound: a joint held still where the commands
  have not been for ``lag_window_s + max_state_age_s`` counts, so a push against a moving head is
  caught within about ``lag_window_s + max_state_age_s + confirm_s``.
- **Samples, not time.** Divergence must hold on ``confirm_s x loop_hz`` new samples in a row. A
  sample read in a tick later than ``late_tick_periods`` periods after the previous one (an
  overloaded CPU) does not count, so a stall cannot bridge the confirmation, but it does not
  restart the count either: under steady overruns a push is still caught.
- **Yield units.** A manifest group yields on its own, merged with every group a constraint ties
  it to (writing one side of a coupled pair while the other is pushed would fight the
  constraint). An antenna held by a hand stops the antennas only; a head and a body yaw tied by
  a yaw-delta constraint yield together.

While a unit yields the loop writes none of its dofs and follows its measured pose. Units that
started yielding within ``motion.platform_move_grace_s`` of each other (one platform move) resume
together, from the measured pose, once each one's pose has stayed inside ``still_band_*`` of an
anchor for the grace (the loop checks the motors first; off means asleep); a unit that yielded on
its own resumes on its own. The recent events, with the largest tick gap and the overruns of the
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
    run_gap_s: float = 0.0  # the largest tick gap during the run, late ticks included (report only)
    run_overruns: int = 0  # the loop's overrun count when the run started
    yield_t: float = 0.0  # when it started yielding
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
    """The values commanded for one dof over the last ticks, the last two that left it, and the
    values over a longer span that an old sample may still show (``older``)."""

    __slots__ = ("values", "came_from", "before", "older")

    def __init__(self, size: int, older: int) -> None:
        self.values: deque[float] = deque(maxlen=size)
        self.older: deque[float] = deque(maxlen=older)
        self.came_from: float | None = None
        self.before: float | None = None

    def push(self, value: float) -> None:
        if len(self.values) == self.values.maxlen:
            self.before, self.came_from = self.came_from, self.values[0]
        self.values.append(value)
        self.older.append(value)

    def reset(self, start: float | None) -> None:
        """Empty; a path that starts at ``start`` (the measured pose), not moving."""
        self.values.clear()
        self.older.clear()
        self.came_from = self.before = start
        if start is not None:
            self.older.append(start)


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
        self._late = False  # this tick came late: its sample does not count
        self._last_seen: tuple[float, dict[str, float]] | None = None
        self._ref: dict[str, float | None] = dict.fromkeys(self._kind)  # value at its last move
        self._moved_t: dict[str, float] = {}  # when it last moved by more than still_band
        self._keys = {n: (f"latest_{k}", f"err_{k}", f"speed_{k}") for n, k in self._kind.items()}
        self._windows: dict[str, _Window] = {}
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
        older = size + math.ceil(cfg.max_state_age_s * self.rate_hz)
        for name in self._kind:  # resized, keeping the commands and where the path came from
            old = self._windows.get(name)
            if old is None or (old.values.maxlen, old.older.maxlen) != (size, older):
                window = _Window(size, older)
                if old is not None:
                    window.values.extend(old.values)
                    window.older.extend(old.older)
                    window.came_from, window.before = old.came_from, old.before
                self._windows[name] = window
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
            self._ref[name] = None
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
        """Start of a writing tick: the sample of a late tick will not count."""
        last, self._last_tick = self._last_tick, now
        gap = 0.0 if last is None else now - last
        self._late = gap > self._late_s
        for unit in self.units:
            if unit.run:
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
            worst = self._worst(unit, now, measured)
            if worst is None:
                unit.run = 0
                continue
            if self._late:
                continue  # no evidence either way: a stall must not bridge the confirmation
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

    def _worst(
        self, unit: Unit, now: float, measured: dict[str, float]
    ) -> tuple[str, float, float] | None:
        """The joint most beyond its tolerance and not lagging (see the module docstring)."""
        worst: tuple[str, float, float] | None = None
        for name in unit.dofs:
            value, window = measured.get(name), self._windows[name]
            if value is None:
                continue
            step = self._moved(name, now, value)
            if not window.values:
                continue
            latest_key, err_key, _ = self._keys[name]
            self.peak[latest_key] = max(self.peak[latest_key], abs(value - window.values[-1]))
            lo, hi = min(window.values), max(window.values)
            side = 1.0 if value > hi else -1.0
            error = value - hi if value > hi else lo - value
            if error <= 0.0 or step * side < 0.0:
                continue  # inside the range, or closing in on it: lag
            self.peak[err_key] = max(self.peak[err_key], error)
            tol = self._tol[name]
            if error <= tol:
                continue
            if (min(window.older) - tol <= value <= max(window.older) + tol) or (
                self._behind(name, window, value)
                and now - self._moved_t[name] < self.cfg.max_state_age_s
            ):
                continue  # where the commands were up to max_state_age_s ago, or behind them
            if worst is None or error / tol > worst[1] / worst[2]:
                worst = (name, error, tol)
        return worst

    def _moved(self, name: str, now: float, value: float) -> float:
        """How far the joint moved since it last moved by more than ``still_band`` (0.0 if it
        has not since): a lagging joint keeps moving, a held one stays inside the band."""
        ref = self._ref[name]
        if ref is not None and abs(value - ref) <= self._band[name]:
            return 0.0
        self._ref[name], self._moved_t[name] = value, now
        return 0.0 if ref is None else value - ref

    def _behind(self, name: str, window: _Window, value: float) -> bool:
        """On the side the path came from when it left the window (where a lagging robot is)."""
        if window.came_from is None or window.before is None:
            return False
        step = window.came_from - window.before
        if abs(step) <= self._drift[name]:
            return False  # the path was still: no side is lag
        return (value - window.came_from) * step < 0.0

    def _start_yield(self, unit: Unit, now: float, measured: dict[str, float]) -> None:
        unit.yielding, unit.still_since, unit.anchor, unit.yield_t = True, None, None, now
        for name in unit.dofs:
            self._windows[name].reset(None)
        self._update_held()
        self.events += 1
        self._last_seen = (now, dict(measured))

    # ---- while yielding

    def watch(self, now: float, measured: dict[str, float], grace_s: float) -> list[Unit]:
        """Stillness of the yielding units. Each unit's pose must stay inside the band of its
        anchor (the pose where its count started; a joint beyond the band restarts it from the
        new pose, so encoder jitter or a slow cache does not). Returns the units that may resume:
        a unit is ready once it and every unit that started yielding within ``grace_s`` of it
        have been still for ``grace_s``, counted from the latest start. One platform move
        drives several units, and the one that settles first must not resume while the move
        runs on (its targets would be accepted the moment the move ends, before the motors-off
        that follows it is seen); units that yielded apart do not hold each other. The caller
        checks the motors first."""
        last = self._last_seen
        self._last_seen = (now, dict(measured))
        if last is not None and now > last[0]:
            dt = now - last[0]
            for name in self.held:
                if name in measured and name in last[1]:
                    key = self._keys[name][2]
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
        ready: list[Unit] = []
        for unit in yielding:
            peers = [v for v in yielding if abs(v.yield_t - unit.yield_t) <= grace_s]
            starts = [v.still_since for v in peers]
            if None not in starts and now - max(s for s in starts if s is not None) >= grace_s:
                ready.append(unit)
        return ready

    def resume(self, unit: Unit, measured: dict[str, float]) -> None:
        unit.yielding, unit.still_since, unit.anchor, unit.run = False, None, None, 0
        for name in unit.dofs:
            self._windows[name].reset(measured.get(name))
            self._ref[name] = None
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
