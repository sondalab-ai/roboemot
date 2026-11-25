"""Scanning: when to look around, and how a sweep is run (SPEC 5A, Scanning; 10).

- ``ScanPolicy`` decides *when*: idle for ``idle_interval_s`` (jittered, stretched by the
  persona's energy), or only ``target_lost_delay_s`` (same scaling, own jitter) after a person
  target or speaker hint is lost, wake-up from rest, and a glance toward an unexplained bearing.
  Never while resting or in quiet hours, nor while the robot is alone (``AloneWatch``: no face and
  no sound for ``alone_after_s``; the idle clock restarts when that ends); never a sweep while
  engaged with a person (one glance per engagement is allowed). It reads no clock and no global
  RNG: stamps and the RNG are passed in, so a replay with the same seed gives the same decisions.
- ``SweepRunner`` walks the planner's waypoints with a rate-limited target, waits for the
  measured pose to arrive, dwells, and moves on.
- ``ScanController`` ties both to a ``ScanPlanner``. A sweep ends with a waypoint at home, at
  scan pace. It is cancelled the tick a person (or a speaker hint) engages the robot, and
  the idle clock only runs while nobody is engaged (SPEC 5A). Attention on an animal or object
  holds a due scan off for at most ``attention.object_max_hold_s``; then the sweep runs and the
  app hides that attention command while it does (SPEC 10).
  An engaged glance is the one exception: it is fed at expression priority for
  ``glance_hold_s`` so it can show over the idle-tier attention command.
- Photo sweeps (SPEC 5A, 8; design 2026-11-25): while a surroundings picture is due
  (``ScanContext.picture_due``), any sweep that starts is a photo sweep: exactly
  ``picture_amplitude`` of the circle, whatever the energy says (owner, 2026-11-25), and at
  least ``picture_dwell_s`` at each waypoint. When no sweep
  starts within ``picture_wait_s`` of the picture becoming due, the policy starts one itself
  (reason ``picture``). The first picture owed at start or switch-on (``ScanContext.picture_first``)
  starts its photo sweep at once, even while engaged or alone (owner, 2026-11-25); asleep, music
  and a sweep already running still hold it. It is one attempt: once that sweep ended the
  surroundings service clears the flag and a picture still due follows the rules above. A photo
  sweep is fed at expression priority and only sleep, rest and quiet hours cancel it;
  ``ScanController.view`` names the waypoint the head dwells at.

Pace and amplitude come from the persona's energy effects (``idle_period_x``,
``motion_amplitude_x``).
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple

from ..config.schema import WorldScanConfig
from ..manifest import Dof
from ..motion import Command, Priority
from ..persona import PersonaEffects
from .sources import ScanPlanner, ScanWaypoint

SOURCE = "scan"
# a sweep ran to its end; payload {"photo": bool, "sweep": int} (SPEC 8, Surroundings snapshots)
SCAN_FINISHED = "world.scan_finished"
SLEW_EPS = 1e-9
FULL_CIRCLE_DEG = 360.0  # a photo sweep's span is a share of it (surroundings.room.span_deg)
Reason = Literal["idle", "wakeup", "unexplained", "picture"]


class ScanView(NamedTuple):
    """The waypoint a photo sweep dwells at: the sweep's id (one per photo sweep, so views of
    a cancelled sweep never join the next), the coverage index and the base-frame azimuth."""

    sweep: int
    index: int
    azimuth_rad: float


@dataclass(frozen=True)
class ScanContext:
    """What the policy needs to know at one tick (all injected, nothing read from a clock)."""

    now: float
    engaged: bool  # attention has a person target or a speaker hint
    object_target: str | None = None  # key of an animal or object attention target, if any
    object_max_hold_s: float = math.inf  # attention.object_max_hold_s
    circadian: str = "awake"  # awake | drowsy | resting
    quiet_hours: bool = False
    alone: bool = False  # no face and no sound for world.scan.alone_after_s (AloneWatch)
    music: bool = False  # the robot dances to room music: hold the scans (SPEC 7, Dance motion)
    effects: PersonaEffects = PersonaEffects()
    picture_due: bool = False  # a surroundings picture is due (SPEC 8): sweeps are photo sweeps
    picture_wait_s: float = math.inf  # surroundings.room.sweep_wait_s
    # the due picture is the first one owed at start or switch-on, its one attempt unused:
    # it sweeps at once, even engaged or alone (owner, 2026-11-25)
    picture_first: bool = False
    picture_amplitude: float = 0.0  # share of the circle: surroundings.room.span_deg / 360
    picture_dwell_s: float = 0.0  # surroundings.view_dwell_s

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
    photo: bool = False  # a photo sweep for the surroundings picture (SPEC 8)


class ScanPolicy:
    """Decides when to scan; pure given the contexts, the RNG and ``note_unexplained`` calls."""

    def __init__(self, cfg: WorldScanConfig, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self._rng = rng or random.Random()
        self._lock = threading.Lock()  # note_unexplained comes from a source's thread
        self._next_at: float | None = None
        self._running = False
        self._was_asleep = False
        self._was_engaged = False
        self._wake_pending = False
        self._pending: tuple[float, float, float] | None = None  # azimuth, elevation, stamp
        self._glanced_while_engaged = False
        self._last_glance = -math.inf
        self._due_since: float | None = None
        self._picture_since: float | None = None  # when the picture became due (not running)
        self._photo = False  # the running decision is a photo sweep

    def note_unexplained(self, azimuth_rad: float, elevation_rad: float, stamp: float) -> None:
        """Remember the newest bearing no entry explains (``ObserveResult.unexplained_bearing``).

        Thread-safe. It becomes a glance at the next ``update`` that allows one, unless it is
        older than ``glance_max_age_s`` by then.
        """
        with self._lock:
            self._pending = (azimuth_rad, elevation_rad, stamp)

    def finished(self) -> None:
        """The decided scan ended (done or cancelled); the idle clock restarts at the next tick.
        After a photo sweep the picture's wait starts again too, so a picture still due (a view
        was missing) never starts sweeps back to back."""
        self._running = False
        self._next_at = None
        self._due_since = None
        if self._photo:
            self._picture_since = None
        self._photo = False

    def update(self, ctx: ScanContext) -> ScanDecision | None:
        """The scan to start now, or None. At most one decision until ``finished`` is called.
        The first picture owed (``picture_first``) skips the alone, engaged, object-hold and
        ``picture_wait_s`` rules; only sleep, music and a running scan hold it."""
        if not ctx.picture_due:
            self._picture_since = None
        woke = self._was_asleep and not ctx.asleep
        self._was_asleep = ctx.asleep
        if woke:
            self._wake_pending = True
        if ctx.asleep:
            self._drop_pending()
            self._wake_pending = False
            self._next_at = None
            self._was_engaged = False
            return None
        if ctx.picture_due and ctx.picture_first and not ctx.music:
            return None if self._running else self._start(ctx, "picture")
        if ctx.alone or ctx.music:  # hold still; the idle clock restarts when this ends
            self._drop_pending()
            self._wake_pending = False
            self._next_at = None
            self._was_engaged = False
            return None
        if ctx.engaged:
            self._was_engaged = True
            self._wake_pending = False
            self._next_at = None  # the idle clock only runs while attention is idle
            return None if self._running else self._engaged_glance(ctx)
        self._glanced_while_engaged = False
        if self._running:
            return None
        if self._next_at is None:
            self._next_at = ctx.now + (
                self._lost_delay(ctx) if self._was_engaged else self._interval(ctx)
            )
            self._was_engaged = False
        if ctx.picture_due and self._picture_since is None:
            self._picture_since = ctx.now
        if self._held_by_object(ctx):
            return None
        if self._wake_pending:
            return self._start(ctx, "wakeup")
        if self._picture_ready(ctx):
            return self._start(ctx, "picture")
        glance = self._glance(ctx, engaged=False)
        if glance is not None:
            return glance
        if ctx.now >= self._next_at:
            return self._start(ctx, "idle")
        return None

    def _held_by_object(self, ctx: ScanContext) -> bool:
        """Whether an animal or object target still holds off a scan that is due.

        The hold timer starts when a scan first becomes due while such a target is held and
        runs for ``object_max_hold_s``; it resets only when the object target is lost (a switch
        to another object keeps it, so the total hold stays bounded) or the scan starts.
        """
        if ctx.object_target is None:
            self._due_since = None
            return False
        due = (
            self._wake_pending
            or self._picture_ready(ctx)
            or self._glance_ready(ctx)
            or (self._next_at is not None and ctx.now >= self._next_at)
        )
        if not due:
            return False
        if self._due_since is None:
            self._due_since = ctx.now
        return ctx.now - self._due_since < ctx.object_max_hold_s

    def _picture_ready(self, ctx: ScanContext) -> bool:
        """Whether the picture has been due for ``picture_wait_s`` with no sweep started."""
        since = self._picture_since
        return ctx.picture_due and since is not None and ctx.now - since >= ctx.picture_wait_s

    def _glance_ready(self, ctx: ScanContext) -> bool:
        with self._lock:
            pending = self._pending
        return (
            pending is not None
            and ctx.now - pending[2] <= self.cfg.glance_max_age_s
            and ctx.now - self._last_glance >= self.cfg.glance_cooldown_s
        )

    def _interval(self, ctx: ScanContext) -> float:
        base = self.cfg.idle_interval_s * ctx.effects.idle_period_x
        return base + self._rng.uniform(-self.cfg.jitter_s, self.cfg.jitter_s)

    def _lost_delay(self, ctx: ScanContext) -> float:
        """The short wait after a person target or speaker hint went away (SPEC 5A). The base is
        scaled by ``idle_period_x``; the jitter is not, so it stays a few seconds."""
        base = self.cfg.target_lost_delay_s * ctx.effects.idle_period_x
        jitter = self.cfg.target_lost_jitter_s
        return max(0.0, base + self._rng.uniform(-jitter, jitter))

    def _pace(self, ctx: ScanContext) -> float:
        return 1.0 / ctx.effects.idle_period_x

    def _start(self, ctx: ScanContext, reason: Reason) -> ScanDecision:
        self._running = True
        self._wake_pending = False
        self._drop_pending()  # the sweep passes every bearing anyway
        pace = self._pace(ctx)
        amplitude = min(1.0, max(self.cfg.min_amplitude, ctx.effects.motion_amplitude_x))
        dwell_s = self.cfg.dwell_s / pace
        self._photo = ctx.picture_due
        if self._photo:
            amplitude = min(1.0, ctx.picture_amplitude)
            dwell_s = max(dwell_s, ctx.picture_dwell_s)
        return ScanDecision("sweep", reason, amplitude, pace, dwell_s, photo=self._photo)

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
        self._drop_pending(pending)
        self._running = True
        self._last_glance = ctx.now
        self._glanced_while_engaged = self._glanced_while_engaged or engaged
        pace = self._pace(ctx)
        return ScanDecision(
            "glance", "unexplained", 0.0, pace, self.cfg.glance_hold_s / pace, az, el, engaged
        )

    def _drop_pending(self, only: tuple[float, float, float] | None = None) -> None:
        """Forget the pending bearing; with ``only``, just if it is still that very value (a
        newer one set from the source's thread meanwhile is kept)."""
        with self._lock:
            if only is None or self._pending == only:
                self._pending = None


class SweepRunner:
    """Walks waypoints: slewed target, arrival check on the measured pose, dwell, next."""

    def __init__(self, dofs: Mapping[str, Dof], cfg: WorldScanConfig) -> None:
        self.dofs, self.cfg = dofs, cfg
        self._wps: list[ScanWaypoint] = []
        self._i = 0
        self._cur: dict[str, float] = {}
        self._pace = 1.0
        self._priority = Priority.IDLE
        self._last = 0.0
        self._slew_done_at: float | None = None
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
        priority: Priority = Priority.IDLE,
    ) -> None:
        """Begin a sweep from ``pose``."""
        self._wps, self._i = list(waypoints), 0
        names = {n for wp in self._wps for n in wp.targets}
        self._cur = {n: pose.get(n, self.dofs[n].home) for n in names}
        self._pace, self._priority = pace_x, priority
        self._last = now
        self._arrived_at = self._slew_done_at = None

    def dwelling(self) -> tuple[int, float] | None:
        """The index and azimuth of the waypoint the pose has reached and is dwelling at, None
        while travelling or with no sweep."""
        if not self.active or self._arrived_at is None:
            return None
        return self._i, self._wps[self._i].azimuth_rad

    def cancel(self) -> None:
        """Abandon the sweep."""
        self._wps, self._i, self._cur = [], 0, {}

    def step(self, now: float, pose: Mapping[str, float]) -> Command | None:
        """One tick: the command to feed, or None when the sweep is over (or not running)."""
        if not self.active:
            return None
        wp = self._wps[self._i]
        slewed = self._slew(now, wp)
        if slewed and self._slew_done_at is None:
            self._slew_done_at = now
        if self._arrived_at is None and slewed and self._reached(wp, pose):
            self._arrived_at = now
        dwelled = self._arrived_at is not None and now - self._arrived_at >= wp.dwell_s
        # Travel takes as long as the pace needs; only the wait for the pose to catch up after
        # the target arrived is bounded, and the dwell never is.
        stuck = (
            self._arrived_at is None
            and self._slew_done_at is not None
            and now - self._slew_done_at >= self.cfg.waypoint_timeout_s
        )
        if dwelled or stuck:
            self._i += 1
            self._arrived_at = self._slew_done_at = None
            if not self.active:
                self.cancel()
                return None
        return Command(
            SOURCE, self._priority, dict(self._cur), now, expires=now + self.cfg.command_ttl_s
        )

    def _slew(self, now: float, wp: ScanWaypoint) -> bool:
        """Move the target towards the waypoint at the pace; True once it is there."""
        dt = min(max(0.0, now - self._last), self.cfg.command_ttl_s)
        self._last = now
        step = math.radians(self.cfg.speed_dps) * self._pace * dt
        for name, goal in wp.targets.items():
            gap = goal - self._cur[name]
            self._cur[name] = (
                goal if abs(gap) <= step else self._cur[name] + math.copysign(step, gap)
            )
        return all(math.isclose(self._cur[n], g, abs_tol=SLEW_EPS) for n, g in wp.targets.items())

    def _reached(self, wp: ScanWaypoint, pose: Mapping[str, float]) -> bool:
        tol = math.radians(self.cfg.arrive_tol_deg)
        for name, goal in wp.targets.items():
            if name in pose and abs(pose[name] - goal) > tol:
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
        self._kind: str | None = None  # the running decision's kind
        self._finished_sweep = False
        self._photo = False  # the running sweep is a photo sweep
        self._coverage = 0  # coverage waypoints of the current or last photo sweep (home excluded)
        self._view: ScanView | None = None  # one attribute: read from the vision thread
        self._sweep = 0  # the id of the current or last photo sweep
        self.last_sweep_photo = False  # whether the sweep last reported finished was a photo one

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
        self._photo = False
        self._view = None

    @property
    def photo(self) -> bool:
        """Whether a photo sweep is in progress."""
        return self._photo and self.runner.active

    def view(self) -> ScanView | None:
        """The sweep id, index and base-frame azimuth of the coverage waypoint where the head
        dwells now (from the pose's arrival until the dwell ends), only during a photo sweep;
        None while travelling, on the glide home and outside a photo sweep. Safe from any
        thread."""
        return self._view

    def views_planned(self) -> int:
        """How many views the current (or last) photo sweep plans; 0 before any."""
        return self._coverage

    @property
    def sweep_id(self) -> int:
        """The id of the current (or last) photo sweep; 0 before any."""
        return self._sweep

    def tick(self, ctx: ScanContext, pose: Mapping[str, float], fov_h_rad: float) -> Command | None:
        """Advance one tick and return the scan command, or None when nothing should move.

        The caller feeds a command and clears ``SOURCE`` on None. A running sweep is cancelled
        when the robot falls asleep, is left alone or engages with a person (attention takes the
        head). A photo sweep always completes: only sleep, rest and quiet hours cancel it.
        """
        cmd = self._advance(ctx, pose, fov_h_rad)
        dwelling = self.runner.dwelling() if self._photo else None
        self._view = (
            ScanView(self._sweep, *dwelling)
            if dwelling is not None and dwelling[0] < self._coverage
            else None
        )
        return cmd

    def _advance(
        self, ctx: ScanContext, pose: Mapping[str, float], fov_h_rad: float
    ) -> Command | None:
        if self.runner.active and self._cancelled(ctx):
            self.runner.cancel()
            self.policy.finished()
            self._photo = False
        decision = self.policy.update(ctx)
        if decision is not None:
            self._begin(ctx, decision, pose, fov_h_rad)
        if not self.runner.active:
            return None
        cmd = self.runner.step(ctx.now, pose)
        if cmd is None:
            self.policy.finished()
            self._engaged_glance = False
            self._finished_sweep = self._kind == "sweep"
            self.last_sweep_photo = self._finished_sweep and self._photo
            self._photo = False
        return cmd

    def _cancelled(self, ctx: ScanContext) -> bool:
        """Whether the running sweep or glance stops this tick."""
        if self._photo:
            return ctx.asleep
        still = ctx.asleep or ctx.alone or ctx.music
        return still or (ctx.engaged and not self._engaged_glance)

    def take_finished_sweep(self) -> bool:
        """Whether a sweep ran to its end since the last call (a cancelled sweep and a glance
        do not count): a new view for the surroundings room snapshot (SPEC 8)."""
        done, self._finished_sweep = self._finished_sweep, False
        return done

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
        self._photo = d.photo
        if d.photo:
            self._coverage = len(wps)
            self._sweep += 1
        if not d.engaged:  # glide home at scan pace instead of snapping at the limiter's speed
            names = {n for wp in wps for n in wp.targets}
            home = {n: self.runner.dofs[n].home for n in names}
            wps = [*wps, ScanWaypoint(0.0, home, 0.0)]
        self._engaged_glance = d.engaged
        self._kind = d.kind
        priority = Priority.EXPRESSION if d.engaged or d.photo else Priority.IDLE
        self.runner.start(ctx.now, pose, wps, pace_x=d.pace_x, priority=priority)
