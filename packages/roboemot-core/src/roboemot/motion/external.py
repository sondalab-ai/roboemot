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
- **Yield units.** A unit is a manifest group, merged with every group a constraint ties it to
  (writing one side of a coupled pair while the other is pushed would fight the constraint). A
  diverging unit yields every unit, unless all its groups are in ``solo_groups``: then it
  yields alone. A platform's own move can show first on any unit (in the MuJoCo sim the daemon's
  goto_sleep moves the antennas ~0.6 s before the head), and a unit still written would have its
  targets accepted in the move's pauses and as it ends; a hand on one part cannot be told from
  it. A robot whose platform moves never move a group first may list that group (an antenna held
  by a hand then stops the antennas only).

While a unit yields the loop writes none of its dofs and follows its measured pose. Units that
started yielding within ``motion.platform_move_grace_s`` of each other (one platform move) resume
together, from the measured pose, once each one's pose has stayed inside ``still_band_*`` of an
anchor for the grace (the loop checks the motors first; off means asleep); a unit that yielded on
its own resumes on its own. The recent events, with the largest tick gap and the overruns of the
confirming run, stay in the debug report.

**Unreachable target** (robot run 2026-11-05: the rest posture drove the head into the body, the
loop yielded, resumed after the grace and pushed again, every ~6.8 s all night). A unit that
diverges again within ``repeat_window_s`` of its resume, on the same joint, toward the same arbiter
target and stopped at the same place (both within ``same_target_*``), repeats; after
``unreachable_repeats`` repeats it holds: it writes the measured pose once, clamped by the limiter
(the platform's goal leaves the obstacle; a failed write is retried), then nothing, and never
resumes on a timer. It takes control back once still for the grace, when the blocked joint's
target moves further than ``same_target_*`` from the blocked one and not further past the stop
(never while the watchdog holds: that target is transient). The blocked record stays: the same
target coming back is held at its first divergence. A settled hold is not external control
(``yielding``) and the other units resume as usual; a held unit that moves after it settled (a
hand) becomes a plain yield. Motors off forgets it all.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
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
    groups: frozenset[str] = frozenset()
    yielding: bool = False
    run: int = 0  # new divergent samples in a row
    run_since: float = 0.0
    run_gap_s: float = 0.0  # the largest tick gap during the run, late ticks included (report only)
    run_overruns: int = 0  # the loop's overrun count when the run started
    yield_t: float = 0.0  # when it started yielding
    still_since: float | None = None
    anchor: dict[str, float] | None = None
    holding: bool = False  # yielding because its target cannot be reached (no resume on a timer)
    holds: dict[str, Stop] = field(default_factory=dict)  # the stops it holds on, by joint
    sticky: bool = False  # held again at a known stop: only a target off that stop releases
    hold_pending: bool = False  # the hold pose is still to be written (a failed write retries)
    settled: bool = False  # still for the grace while yielding (a moving held unit is external)
    resumed_t: float | None = None
    repeats: int = 0


@dataclass(frozen=True)
class Stop:
    """Where a joint stopped short of the target it was driven to: the place, the side the
    target lay on (+1 or -1), that target and what found it (``repeat``, ``steady_offset``)."""

    joint: str
    target: float
    measured: float
    side: float
    cause: str = "repeat"

    def past(self, value: float) -> float:
        """How far ``value`` lies beyond the stop, toward the target side (negative: back)."""
        return self.side * (value - self.measured)


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
    units = []
    for names in members.values():
        groups = frozenset(dofs[n].group for n in names)
        units.append(Unit("+".join(sorted(groups)), tuple(names), groups))
    return units


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
        self.unreachable = 0
        self.unreachable_last: dict[str, Any] | None = None
        self._stops: dict[str, Stop] = {}  # each joint's last divergence (robot-wide)
        self._progress: dict[str, float] = {}  # how far past that stop it got since
        self._blocked: dict[str, Stop] = {}  # stops found unreachable, until passed or motors off
        self._steady: dict[str, tuple[float, float, float]] = {}  # since, command, measured
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
        self._same = {
            n: cfg.same_target_m if k == "m" else cfg.same_target_rad for n, k in self._kind.items()
        }
        self._offset = {
            n: cfg.steady_offset_m if k == "m" else cfg.steady_offset_rad
            for n, k in self._kind.items()
        }
        # a resumed push is caught within lag_window_s + max_state_age_s + confirm_s plus the
        # travel: the window scales with them, so no in-range value switches the guard off
        self.repeat_window_s = (
            cfg.lag_window_s + cfg.max_state_age_s + cfg.confirm_s + cfg.repeat_margin_s
        )
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
        """External control: a unit yields to something else. A hold on an unreachable target
        counts only until its pose has been still for the grace (a hand may be moving it); its
        dofs stay in ``held`` throughout. Read without the loop's lock by diagnostics and Live
        (booleans, set by the loop thread only): a stale value lasts one tick."""
        return any(self._external(u) for u in self.units)

    @property
    def holding(self) -> bool:
        """A unit holds on an unreachable target (same lock exception as ``yielding``)."""
        return any(u.holding for u in self.units)

    @staticmethod
    def _external(unit: Unit) -> bool:
        return unit.yielding and not (unit.holding and unit.settled)

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
            unit.holding, unit.holds, unit.resumed_t, unit.repeats = False, {}, None, 0
            unit.sticky = False
            unit.hold_pending, unit.settled = False, False
        self._stops.clear()
        self._progress.clear()
        self._blocked.clear()
        self._steady.clear()
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

    def judge(
        self,
        now: float,
        measured: dict[str, float],
        overruns: int,
        goal: dict[str, float] | None = None,
    ) -> list[Unit]:
        """Compare a new sample with the windowed commands; returns the units that yield now.
        ``goal`` is the arbiter's target this tick (before the fade and the limiter): a
        divergence that repeats at a known stop, or a steady offset from a still command,
        makes its unit hold (module docstring, Unreachable target)."""
        started: list[Unit] = []
        goal = goal or {}
        for unit in self.units:
            if unit.yielding:
                continue
            self._track(unit, measured)
            diverging = self._diverging(unit, now, measured)
            if not diverging:
                unit.run = 0
                if not self._late and self._steady_offset(unit, now, measured, goal):
                    started.append(unit)
                continue
            worst = max(diverging, key=lambda d: d[1] / d[2])
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
                alone = unit.groups <= set(self.cfg.solo_groups)
                others = [] if alone else [u for u in self.units if not u.yielding]
                for each in [unit, *[u for u in others if u is not unit]]:
                    self._start_yield(each, now, measured)
                    started.append(each)
                target = goal.get(joint)
                held = self._repeat(unit, now, [d[0] for d in diverging], goal, measured)
                self.events += 1
                self.recent.append(
                    {
                        "t": now,
                        "group": unit.name,
                        "joint": joint,
                        "error": error,
                        "tolerance": tolerance,
                        "target": target,
                        "measured": measured[joint],
                        "unreachable": held,
                        "samples": unit.run,
                        "run_s": run_s,
                        "max_gap_s": unit.run_gap_s,
                        "overruns": overruns - unit.run_overruns,
                    }
                )
                unit.run = 0
        return started

    def _repeat(
        self,
        unit: Unit,
        now: float,
        joints: list[str],
        goal: dict[str, float],
        measured: dict[str, float],
    ) -> bool:
        """Record the diverging ``joints`` of ``unit``; True when it now holds. A joint repeats
        when it stopped again where it stopped before (within ``same_target_*``), with the
        target beyond on the same side (whatever the target is: a slowly varying one too),
        within the repeat window of the unit's resume and without having got past that stop
        since: a hand that moves a head which had reached its target is no repeat. A joint at a
        stop found unreachable before holds at once. Records are per joint, robot-wide: a unit
        that yielded for another one keeps its own."""
        recent = unit.resumed_t is not None and now - unit.resumed_t <= self.repeat_window_s
        stops, known, repeated = {}, False, False
        for joint in joints:
            if joint not in goal or joint not in measured:
                continue
            stop = self._stop(joint, goal[joint], measured[joint], "repeat")
            stops[joint] = stop
            known = known or self._same_stop(self._blocked.get(joint), stop)
            last = self._stops.get(joint)
            stuck = self._progress.get(joint, 0.0) <= self._same[joint]
            repeated = repeated or (recent and stuck and self._same_stop(last, stop))
            self._stops[joint], self._progress[joint] = stop, 0.0
        unit.repeats = unit.repeats + 1 if repeated else 0
        if not stops or (not known and unit.repeats < self.cfg.unreachable_repeats):
            return False
        self._hold(unit, now, stops, sticky=known)
        return True

    @staticmethod
    def _stop(joint: str, target: float, value: float, cause: str) -> Stop:
        return Stop(joint, target, value, 1.0 if target >= value else -1.0, cause)

    def _hold(
        self, unit: Unit, now: float, stops: dict[str, Stop], *, sticky: bool = False
    ) -> None:
        unit.holding, unit.holds, unit.repeats, unit.hold_pending = True, dict(stops), 0, True
        unit.sticky = sticky
        self._blocked.update(stops)
        self.unreachable += 1
        first = next(iter(stops.values()))
        self.unreachable_last = {
            "t": now,
            "sticky": sticky,
            "group": unit.name,
            "joint": first.joint,
            "joints": sorted(stops),
            "target": first.target,
            "measured": first.measured,
            "cause": first.cause,
        }

    def _steady_offset(
        self, unit: Unit, now: float, measured: dict[str, float], goal: dict[str, float]
    ) -> bool:
        """The blind band under the tolerance: a joint whose written command and measured pose
        have both stayed inside ``still_band_*`` for ``steady_offset_s`` while they sit more
        than ``steady_offset_*`` apart presses on something. The unit holds (alone: no platform
        move does this). True when it starts holding."""
        stops = {}
        for name in unit.dofs:
            window, value = self._windows[name], measured.get(name)
            if value is None or not window.values:
                self._steady.pop(name, None)
                continue
            command, band = window.values[-1], self._band[name]
            since, ref_cmd, ref_val = self._steady.get(name, (now, command, value))
            if abs(command - ref_cmd) > band or abs(value - ref_val) > band:
                since, ref_cmd, ref_val = now, command, value
            self._steady[name] = (since, ref_cmd, ref_val)
            if (
                abs(command - value) > self._offset[name]
                and now - since >= self.cfg.steady_offset_s
            ):
                stops[name] = self._stop(name, goal.get(name, command), value, "steady_offset")
        if not stops:
            return False
        known = any(self._same_stop(self._blocked.get(n), stop) for n, stop in stops.items())
        for name, stop in stops.items():
            self._stops[name], self._progress[name] = stop, 0.0
        self._start_yield(unit, now, measured)
        self._hold(unit, now, stops, sticky=known)
        return True

    def _track(self, unit: Unit, measured: dict[str, float]) -> None:
        """While written: how far each joint got past its last stop; a stop found unreachable
        that the joint got past by more than ``same_target_*`` was reachable after all and is
        dropped."""
        for name in unit.dofs:
            value = measured.get(name)
            if value is None:
                continue
            last = self._stops.get(name)
            if last is not None:
                self._progress[name] = max(self._progress.get(name, 0.0), last.past(value))
            blocked = self._blocked.get(name)
            if blocked is not None and blocked.past(value) > self._same[name]:
                del self._blocked[name]

    def _same_stop(self, a: Stop | None, b: Stop) -> bool:
        """Same joint, stopped at the same place, the target beyond it on the same side (a
        stop's side is the side its target lay on)."""
        return (
            a is not None
            and a.joint == b.joint
            and a.side == b.side
            and abs(a.measured - b.measured) <= self._same[b.joint]
        )

    def _diverging(
        self, unit: Unit, now: float, measured: dict[str, float]
    ) -> list[tuple[str, float, float]]:
        """The joints beyond their tolerance and not lagging (see the module docstring), as
        (joint, error, tolerance)."""
        out: list[tuple[str, float, float]] = []
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
            out.append((name, error, tol))
        return out

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
        unit.settled = False
        for name in unit.dofs:
            self._windows[name].reset(None)
            self._steady.pop(name, None)
        self._update_held()
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
                if unit.holding and unit.settled:  # moved after it settled (a hand): a yield
                    unit.holding, unit.holds, unit.sticky = False, {}, False
                unit.anchor = {n: measured[n] for n in unit.dofs if n in measured}
                unit.still_since = now
            unit.settled = unit.still_since is not None and now - unit.still_since >= grace_s
        ready: list[Unit] = []
        for unit in yielding:
            if unit.holding:
                continue  # never on a timer: it waits for a new target (``release``)
            peers = [v for v in yielding if abs(v.yield_t - unit.yield_t) <= grace_s]
            starts = [v.still_since for v in peers]
            if None not in starts and now - max(s for s in starts if s is not None) >= grace_s:
                ready.append(unit)
        return ready

    def release(self, goal: dict[str, float]) -> list[Unit]:
        """The holding units that may take control back: still for the grace (as ``watch``
        requires; a hand may be moving it), and the target of a held joint no longer lies
        beyond its stop or (owner decision 2026-11-06) changed by more than ``same_target_*``,
        in either direction. A target still beyond the same stop is then held again at its
        first divergence (``_blocked``: one push), and that sticky hold takes only a target off
        the stop: a target that keeps drifting beyond it would otherwise buy a push per
        drift (review item 5)."""
        return [
            u
            for u in self.units
            if u.holding
            and u.settled
            and any(
                name in goal
                and (
                    stop.past(goal[name]) <= 0.0
                    or (not u.sticky and abs(goal[name] - stop.target) > self._same[name])
                )
                for name, stop in u.holds.items()
            )
        ]

    def hold_pose(self, unit: Unit, measured: dict[str, float]) -> dict[str, float]:
        """The pose to write once for a holding unit: the measured pose, each held joint backed
        off by ``still_band_*`` away from its target, so no press stays."""
        pose = {k: measured[k] for k in unit.dofs if k in measured}
        for name, stop in unit.holds.items():
            if name in pose:
                pose[name] -= stop.side * self._band[name]
        return pose

    def resume(self, unit: Unit, measured: dict[str, float], now: float) -> None:
        unit.yielding, unit.still_since, unit.anchor, unit.run = False, None, None, 0
        unit.holding, unit.holds, unit.sticky, unit.resumed_t = False, {}, False, now
        unit.hold_pending, unit.settled = False, False
        for name in unit.dofs:
            self._windows[name].reset(measured.get(name))
            self._ref[name] = None
            self._steady.pop(name, None)
        self._update_held()

    # ---- report (SPEC 12A)

    def snapshot(self, now: float) -> dict[str, object]:
        yielding = [u for u in self.units if self._external(u)]
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
            "unreachable": self.unreachable,
            "unreachable_held": sorted(u.name for u in self.units if u.holding),
            "unreachable_last": dict(self.unreachable_last) if self.unreachable_last else None,
        }
