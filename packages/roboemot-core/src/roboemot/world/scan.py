"""Scanning: when to look around, and how a sweep is run (SPEC 5A, Scanning; 10).

- ``ScanPolicy`` decides *when*: idle for ``idle_interval_s`` (jittered, stretched by the
  persona's energy), wake-up from rest, and a glance toward an unexplained bearing. Never while
  resting or in quiet hours; never a sweep while engaged with a person (one glance per
  engagement is allowed). It reads no clock and no global RNG: stamps and the RNG are passed
  in, so a replay with the same seed gives the same decisions.
- ``SweepRunner`` walks the planner's waypoints with a rate-limited target, waits for the
  measured pose to arrive, dwells, and moves on.
- ``ScanController`` ties both to a ``ScanPlanner``. Its command is fed by the app only while
  attention is not active: attention wins by construction (SPEC 10, Idle row). A sweep is
  cancelled the tick a person (or a speaker hint) engages the robot. An engaged glance is the
  one exception: it is fed at expression priority for ``glance_hold_s`` so it can show over
  the idle-tier attention command.

Pace and amplitude come from the persona's energy effects (``idle_period_x``,
``motion_amplitude_x``); a present minor only slows the pace, because the arbiter already scales
amplitude and speed around home (SPEC 12) and scaling twice would stop the waypoints being
reached.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ..config.schema import WorldScanConfig
from ..manifest import Dof
from ..motion import Command, Priority
from ..persona import PersonaEffects
from .sources import ScanPlanner, ScanWaypoint

SOURCE = "scan"
Reason = Literal["idle", "wakeup", "unexplained"]


@dataclass(frozen=True)
class ScanContext:
    """What the policy needs to know at one tick (all injected, nothing read from a clock)."""

    now: float
    engaged: bool  # attention has a person target or a speaker hint
    attending: bool = False  # attention owns the head, for any target (yields to it always)
    circadian: str = "awake"  # awake | drowsy | resting
    quiet_hours: bool = False
    effects: PersonaEffects = PersonaEffects()
    minor_factor: float = 1.0  # safety.minor_motion_factor while a minor is present, else 1

    @property
    def asleep(self) -> bool:
        """Resting or in quiet hours: no scanning at all."""
        return self.circadian == "resting" or self.quiet_hours


@dataclass(frozen=True)
class ScanDecision:
    """A request to look around: a sweep, or a glance at a base-frame bearing."""

    kind: Literal["sweep", "glance"]
    reason: Reason
    amplitude: float  # share of the full circle (sweeps)
    pace_x: float  # speed multiplier; dwell is divided by it
    dwell_s: float
    azimuth_rad: float = 0.0
    elevation_rad: float = 0.0
    engaged: bool = False  # a glance made while engaged with a person


class ScanPolicy:
    """Decides when to scan; pure given the contexts, the RNG and ``note_unexplained`` calls."""

    def __init__(self, cfg: WorldScanConfig, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self._rng = rng or random.Random()
        self._lock = threading.Lock()  # note_unexplained comes from a source's thread
        self._next_at: float | None = None
        self._running = False
        self._was_asleep = False
        self._wake_pending = False
        self._pending: tuple[float, float, float] | None = None  # azimuth, elevation, stamp
        self._glanced_while_engaged = False
        self._last_glance = -math.inf

    def note_unexplained(self, azimuth_rad: float, elevation_rad: float, stamp: float) -> None:
        """Remember the newest bearing no entry explains (``ObserveResult.unexplained_bearing``).

        Thread-safe. It becomes a glance at the next ``update`` that allows one, unless it is
        older than ``glance_max_age_s`` by then.
        """
        with self._lock:
            self._pending = (azimuth_rad, elevation_rad, stamp)

    def finished(self) -> None:
        """The decided scan ended (done or cancelled); the idle clock restarts at the next tick."""
        self._running = False
        self._next_at = None

    def update(self, ctx: ScanContext) -> ScanDecision | None:
        """The scan to start now, or None. At most one decision until ``finished`` is called."""
        woke = self._was_asleep and not ctx.asleep
        self._was_asleep = ctx.asleep
        if woke:
            self._wake_pending = True
        if ctx.asleep:
            self._drop_pending()
            self._wake_pending = False
            self._next_at = None
            return None
        if ctx.engaged or ctx.attending:
            self._wake_pending = False
            self._next_at = None  # the idle clock only runs while attention is idle
            if self._running or not ctx.engaged:
                return None
            return self._engaged_glance(ctx)
        self._glanced_while_engaged = False
        if self._running:
            return None
        if self._next_at is None:
            self._next_at = ctx.now + self._interval(ctx)
        if self._wake_pending:
            return self._start(ctx, "wakeup")
        glance = self._glance(ctx, engaged=False)
        if glance is not None:
            return glance
        if ctx.now >= self._next_at:
            return self._start(ctx, "idle")
        return None

    def _interval(self, ctx: ScanContext) -> float:
        base = self.cfg.idle_interval_s * ctx.effects.idle_period_x
        return base + self._rng.uniform(-self.cfg.jitter_s, self.cfg.jitter_s)

    def _pace(self, ctx: ScanContext) -> float:
        return ctx.minor_factor / ctx.effects.idle_period_x

    def _start(self, ctx: ScanContext, reason: Reason) -> ScanDecision:
        self._running = True
        self._wake_pending = False
        self._drop_pending()  # the sweep passes every bearing anyway
        pace = self._pace(ctx)
        amplitude = min(1.0, max(self.cfg.min_amplitude, ctx.effects.motion_amplitude_x))
        return ScanDecision("sweep", reason, amplitude, pace, self.cfg.dwell_s / pace)

    def _engaged_glance(self, ctx: ScanContext) -> ScanDecision | None:
        if self._glanced_while_engaged:
            return None
        return self._glance(ctx, engaged=True)

    def _glance(self, ctx: ScanContext, *, engaged: bool) -> ScanDecision | None:
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        az, el, stamp = pending
        if ctx.now - stamp > self.cfg.glance_max_age_s:
            self._drop_pending()
            return None
        if ctx.now - self._last_glance < self.cfg.glance_cooldown_s:
            return None
        self._drop_pending()
        self._running = True
        self._last_glance = ctx.now
        self._glanced_while_engaged = self._glanced_while_engaged or engaged
        pace = self._pace(ctx)
        return ScanDecision(
            "glance", "unexplained", 0.0, pace, self.cfg.glance_hold_s / pace, az, el, engaged
        )

    def _drop_pending(self) -> None:
        with self._lock:
            self._pending = None


class SweepRunner:
    """Walks waypoints: slewed target, arrival check on the measured pose, dwell, next."""

    def __init__(self, dofs: Mapping[str, Dof], cfg: WorldScanConfig) -> None:
        self.dofs, self.cfg = dofs, cfg
        self._wps: list[ScanWaypoint] = []
        self._i = 0
        self._cur: dict[str, float] = {}
        self._pace = 1.0
        self._cap = 1.0
        self._priority = Priority.IDLE
        self._last = 0.0
        self._wp_start = 0.0
        self._arrived_at: float | None = None

    @property
    def active(self) -> bool:
        """Whether a sweep is in progress."""
        return self._i < len(self._wps)

    def start(
        self,
        now: float,
        pose: Mapping[str, float],
        waypoints: list[ScanWaypoint],
        *,
        pace_x: float = 1.0,
        cap: float = 1.0,
        priority: Priority = Priority.IDLE,
    ) -> None:
        """Begin a sweep from ``pose``. ``cap`` is the arbiter's amplitude factor around home
        (minor present), so arrival is judged against the pose the arbiter will really allow."""
        self._wps, self._i = list(waypoints), 0
        names = {n for wp in self._wps for n in wp.targets}
        self._cur = {n: pose.get(n, self.dofs[n].home) for n in names}
        self._pace, self._cap, self._priority = pace_x, cap, priority
        self._last = self._wp_start = now
        self._arrived_at = None

    def cancel(self) -> None:
        """Abandon the sweep."""
        self._wps, self._i, self._cur = [], 0, {}

    def step(self, now: float, pose: Mapping[str, float]) -> Command | None:
        """One tick: the command to feed, or None when the sweep is over (or not running)."""
        if not self.active:
            return None
        wp = self._wps[self._i]
        self._slew(now, wp)
        if self._arrived_at is None and self._reached(wp, pose):
            self._arrived_at = now
        timed_out = now - self._wp_start >= self.cfg.waypoint_timeout_s
        if self._arrived_at is not None and now - self._arrived_at >= wp.dwell_s or timed_out:
            self._i += 1
            self._arrived_at, self._wp_start = None, now
            if not self.active:
                self.cancel()
                return None
        return Command(
            SOURCE, self._priority, dict(self._cur), now, expires=now + self.cfg.command_ttl_s
        )

    def _slew(self, now: float, wp: ScanWaypoint) -> None:
        dt = min(max(0.0, now - self._last), self.cfg.command_ttl_s)
        self._last = now
        step = math.radians(self.cfg.speed_dps) * self._pace * dt
        for name, goal in wp.targets.items():
            gap = goal - self._cur[name]
            self._cur[name] = (
                goal if abs(gap) <= step else self._cur[name] + math.copysign(step, gap)
            )

    def _reached(self, wp: ScanWaypoint, pose: Mapping[str, float]) -> bool:
        tol = math.radians(self.cfg.arrive_tol_deg)
        for name, goal in wp.targets.items():
            if abs(self._cur[name] - goal) > 0.0:
                return False
            home = self.dofs[name].home
            expected = home + (goal - home) * self._cap
            if name in pose and abs(pose[name] - expected) > tol:
                return False
        return True


class ScanController:
    """Policy + planner + runner: one ``tick`` per app cycle returns the command to feed."""

    def __init__(
        self,
        cfg: WorldScanConfig,
        planner: ScanPlanner,
        dofs: Mapping[str, Dof],
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.planner = planner
        self.policy = ScanPolicy(cfg, rng)
        self.runner = SweepRunner(dofs, cfg)
        self._engaged_glance = False

    @property
    def active(self) -> bool:
        """Whether a sweep or glance is in progress."""
        return self.runner.active

    def apply_config(self, cfg: WorldScanConfig) -> None:
        """Hot reload (SPEC 11A): the new scan parameters apply from the next tick."""
        self.cfg = self.policy.cfg = self.runner.cfg = cfg

    def reset(self) -> None:
        """Drop any sweep in progress (after a failure); the idle clock restarts."""
        self.runner.cancel()
        self.policy.finished()
        self._engaged_glance = False

    def tick(self, ctx: ScanContext, pose: Mapping[str, float], fov_h_rad: float) -> Command | None:
        """Advance one tick and return the scan command, or None when nothing should move.

        The caller feeds a command and clears ``SOURCE`` on None. A running sweep is cancelled
        when the robot falls asleep or engages with a person (attention takes the head).
        """
        if self.runner.active and (
            ctx.asleep or ((ctx.engaged or ctx.attending) and not self._engaged_glance)
        ):
            self.runner.cancel()
            self.policy.finished()
        decision = self.policy.update(ctx)
        if decision is not None:
            self._begin(ctx, decision, pose, fov_h_rad)
        if not self.runner.active:
            return None
        cmd = self.runner.step(ctx.now, pose)
        if cmd is None:
            self.policy.finished()
            self._engaged_glance = False
        return cmd

    def _begin(
        self, ctx: ScanContext, d: ScanDecision, pose: Mapping[str, float], fov_h_rad: float
    ) -> None:
        if d.kind == "sweep":
            wps = self.planner.plan(pose, fov_h_rad, amplitude=d.amplitude, dwell_s=d.dwell_s)
        else:
            wps = [self.planner.glance(pose, d.azimuth_rad, d.elevation_rad, dwell_s=d.dwell_s)]
        if not wps:
            self.policy.finished()
            return
        self._engaged_glance = d.engaged
        priority = Priority.EXPRESSION if d.engaged else Priority.IDLE
        self.runner.start(
            ctx.now, pose, wps, pace_x=d.pace_x, cap=ctx.minor_factor, priority=priority
        )
