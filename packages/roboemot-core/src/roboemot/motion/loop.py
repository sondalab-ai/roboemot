"""Fixed-rate motion loop: MotionCore -> PoseGroupDriver (SPEC 10; 4A real-time path).

In production the arbiter runs in roboemot-rt (C++). Until the C++ process and the bus
transport land (M0), this Python loop drives the robot with the reference MotionCore.

Safety properties (independent review fixes):
- the limiter is re-aligned to the measured pose on start and on every return to ``active``;
  while ``limp`` or ``teach`` it tracks the measured pose, so re-enabling never jumps;
- safety parameters are re-read on ``config.changed`` (``apply_config``); the manifest's
  ``safety.watchdog_ms`` is a ceiling the config cannot raise;
- an exception in a tick (bad input, driver error) never kills the thread: the tick is
  skipped, the error is counted and kept in ``last_error``, and the watchdog takes over.

Robot link (SPEC 12A): ``connectivity.robot.failures_to_offline`` failed ticks in a row mark
the link lost (``link.robot.lost``). The loop then stops writing and retries ``read_state``
with backoff, calling ``driver.reconnect`` when a check fails; on recovery it re-aligns the
limiter and the arbiter to the measured pose and only then reports ``link.robot.restored``, so
motion resumes from where the robot really is, under the velocity and acceleration limits.
Only driver failures count against the link, not faults in RoboEmot's own code.

Reflexes (SPEC 10): ``tick_hooks`` run at the start of every active tick, so a reflex engine
reacts within one loop period instead of waiting for the app's slower tick.

Motors (SPEC 4A, 12A): the loop never turns the motors on after start-up. It polls
``driver.motors_enabled`` every ``probe_initial_s``; while the platform keeps them off (robot
asleep, stopped by its owner, a backend restarted without wake-up) it follows the measured pose
without writing, and resumes from that pose once they are on again. Sources keep feeding the
arbiter meanwhile. A robot unreachable at ``start`` begins lost.

External control (SPEC 4A, 12A): a platform that runs a move of its own (the owner's go-to-sleep)
drops every target streamed meanwhile and accepts them again the moment it ends, which pulled the
robot back from the pose the move had reached. The loop sees this without any platform signal:
the measured pose stops following what it commands (``external.ExternalControl`` has the rules).
The diverging yield unit (a manifest group, with the groups a constraint ties to it) is no longer
written and follows the measured pose; the motor status is read at the platform's own status
period meanwhile (``driver.status_period_s``) and once more before resuming. A unit resumes, from
the measured pose, once its pose has been still for ``motion.platform_move_grace_s`` with the
motors on (that grace must outlast the platform's delay before it switches the motors off after a
sleep move, plus one status period); motors off at any point means asleep. A hand pushing the
head is the same: the loop yields and resumes after it settles.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from .. import shutdown
from ..config.schema import RoboEmotConfig
from ..drivers import MotorMode, PoseGroupDriver
from ..health import Link, OnChange
from ..manifest import Manifest
from .core import Command, MotionCore
from .external import ExternalControl

MS_PER_S = 1000.0
log = logging.getLogger(__name__)


class DriverError(RuntimeError):
    """A driver call failed: counts against the robot link (SPEC 12A)."""


class MotionLoop:
    def __init__(
        self,
        manifest: Manifest,
        driver: PoseGroupDriver,
        cfg: RoboEmotConfig,
        *,
        rate_hz: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_link_change: OnChange | None = None,
    ) -> None:
        self.driver = driver
        self.link = Link("robot", cfg.connectivity.robot, on_link_change)
        self.manifest = manifest
        self.rate_hz = rate_hz or cfg.motion.arbiter.loop_hz
        self.clock = clock
        self.core = MotionCore(
            manifest.dofs,
            manifest.constraints,
            crossfade_s=0.0,
            watchdog_s=0.0,
            watchdog_return_factor=1.0,
        )
        self._lock = threading.Lock()
        self.external_ctl = ExternalControl(
            manifest.dofs, manifest.constraints, cfg.motion.external, self.rate_hz
        )
        self.apply_config(cfg)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mode: MotorMode = "limp"
        self.ticks = 0
        self.overruns = 0
        self.errors = 0
        self.last_error: str | None = None
        self.max_jitter_s = 0.0
        self.motors_off = False  # the platform holds the motors off: track, do not write
        self._seen_seq: int | None = None  # the driver's state_seq at the last judged sample
        self._next_motor_check = -math.inf
        # Called every active tick with ``(now, core)``, under the loop's lock and before the
        # arbiter runs: they may feed or clear ``core`` directly (never ``feed`` or ``clear`` of
        # this class, which take the lock). A raising hook is counted and skipped, the tick goes on.
        self.tick_hooks: list[Callable[[float, MotionCore], None]] = []
        self._hook_error: str | None = None

    # ---- configuration (SPEC 11A hot reload)

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        with self._lock:  # called from the config watcher thread
            self._apply_config(cfg)

    def _apply_config(self, cfg: RoboEmotConfig) -> None:
        watchdog_ms = min(cfg.safety.watchdog_ms, self.manifest.model.safety.watchdog_ms)
        self.core.arbiter.crossfade_s = cfg.motion.arbiter.crossfade_ms / MS_PER_S
        self.core.watchdog_s = watchdog_ms / MS_PER_S
        self.core.watchdog_return_factor = cfg.safety.watchdog_return_factor
        self.core.limiter.constraint_margin = math.radians(cfg.safety.constraint_margin_deg)
        self.link.policy = cfg.connectivity.robot
        self.platform_move_grace_s = cfg.motion.platform_move_grace_s
        self.external_ctl.configure(cfg.motion.external)

    @property
    def external(self) -> bool:
        """Something else controls part of the robot: those dofs are followed, not written."""
        return self.external_ctl.yielding

    @property
    def external_events(self) -> int:
        return self.external_ctl.events

    # ---- inputs

    def feed(self, cmd: Command) -> None:
        with self._lock:
            self.core.feed(cmd)

    def clear(self, source: str) -> None:
        """Drop a source's command so lower-priority motion (idle) takes the dofs back."""
        with self._lock:
            self.core.arbiter.clear(source)

    def pose(self) -> dict[str, float]:
        """The limiter's current pose (what the loop last commanded), by dof name."""
        with self._lock:
            return dict(self.core.limiter.pos)

    def set_mode(self, mode: MotorMode) -> None:
        with self._lock:
            self.driver.set_mode(mode)
            if mode == "active":
                self._resync()  # the robot may have been moved by hand: start from there
            self.mode = mode

    def snapshot(self) -> dict[str, object]:
        """Loop health for diagnostics (SPEC 12A); safe to call from any thread."""
        now = self.clock()
        with self._lock:
            out: dict[str, object] = {
                "mode": self.mode,
                "motors_off": self.motors_off,
                "asleep": self.motors_off and self.link.online,
                **self.external_ctl.snapshot(now),
                "holding": self.core.holding,
                "watchdog_engagements": self.core.watchdog_engagements,
                "rate_hz": self.rate_hz,
                "ticks": self.ticks,
                "overruns": self.overruns,
                "errors": self.errors,
                "last_error": self.last_error,
                "max_jitter_s": self.max_jitter_s,
                "running": self._thread is not None and self._thread.is_alive(),
            }
        return out | {"robot_link": self.link.snapshot(self.clock())}

    # ---- lifecycle

    def start(self) -> None:
        try:
            self.set_mode("active")
        except Exception as exc:  # robot not reachable yet: recover in the loop
            self.mode = "active"
            self._note_error(exc)
            self.link.set(False, self.clock())
        thread = threading.Thread(target=self._run, name="motion-loop", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            shutdown.join(self._thread, 2)

    def step_once(self, now: float, dt: float) -> dict[str, float]:
        with self._lock:
            if self.mode != "active" or self.motors_off:
                # limp / teach / motors held off by the platform: follow the measured pose so
                # re-enabling never jumps
                state = self._resync()
                self.ticks += 1
                return state
            ext = self.external_ctl
            ext.tick(now)
            failed = False
            for hook in self.tick_hooks:
                try:
                    hook(now, self.core)
                except Exception as exc:
                    failed = True
                    self._note_hook_error(exc)
            if not failed:
                self._hook_error = None
            out = self.core.tick(now, dt)
            held = ext.held
            targets = {k: v for k, v in out.items() if k not in held} if held else out
            if targets:
                self._call(self.driver.write_pose_targets, now, targets)
            measured = self._read_fresh()
            if measured is not None:
                if held:
                    self._follow(measured, held)
                    self._watch_external(now, measured)
                started = ext.judge(now, measured, self.overruns)
                if started:
                    self._follow(measured, ext.held)
                    names = ", ".join(u.name for u in started)
                    log.info("motion: external control of the robot detected (%s), yielding", names)
            ext.push(out)
            self.ticks += 1
            return out

    def _read_fresh(self) -> dict[str, float] | None:
        """The measured pose if a new sample arrived since the last call (always, for a driver
        that cannot tell), else None: a repeated sample is no evidence and costs a read."""
        seq = self._call(self.driver.state_seq)
        if seq is not None and seq == self._seen_seq:
            return None
        self._seen_seq = seq
        return self._call(self.driver.read_state)

    def _follow(self, measured: dict[str, float], names: frozenset[str]) -> None:
        """The limiter and the arbiter of the dofs not written take the measured pose."""
        state = {k: measured[k] for k in names if k in measured and k in self.core.limiter.pos}
        self.core.limiter.reset(state)
        self.core.arbiter.last = {**self.core.arbiter.last, **state}

    def _watch_external(self, now: float, measured: dict[str, float]) -> None:
        """Resume the yield units that have been still for the grace, after one more look at
        the motor status (the cached one: the platform may have switched them off since)."""
        ready = self.external_ctl.watch(now, measured, self.platform_move_grace_s)
        if not ready:
            return
        if self._call(self.driver.motors_enabled) is False:
            self._motors_now(False)
            return
        for unit in ready:
            self.external_ctl.resume(unit, measured)
            self._follow(measured, frozenset(unit.dofs))
            log.info("motion: %s still, resuming control", unit.name)

    def _motors_now(self, enabled: bool) -> None:
        """Track the motor state the platform reports (caller holds the lock)."""
        if enabled and self.motors_off:
            self._resync()  # resume from wherever the platform's wake-up left the robot
        if not enabled:
            self.external_ctl.clear()
        self.motors_off = not enabled

    def _resync(self) -> dict[str, float]:
        measured = self._call(self.driver.read_state)
        state = {k: v for k, v in measured.items() if k in self.core.limiter.pos}
        self.core.limiter.reset(state)
        self.core.arbiter.last = {**self.core.arbiter.last, **state}
        self.external_ctl.reset(state)
        return state

    @staticmethod
    def _call(fn: Callable[..., Any], *args: Any) -> Any:
        """Driver call; its failures count against the robot link, ours do not."""
        try:
            return fn(*args)
        except Exception as exc:
            raise DriverError(f"{type(exc).__name__}: {exc}") from exc

    def cycle(self, now: float, dt: float) -> None:
        """One loop period: a tick while the robot link is up, else a due recovery check."""
        if self.link.online:
            try:
                self._check_motors(now)
                self.step_once(now, dt)
                self.link.report(True, now)
            except DriverError as exc:  # the watchdog covers missed ticks
                self._note_error(exc)
                self.link.report(False, now)
            except Exception as exc:  # a fault in RoboEmot itself: not a robot link problem
                self._note_error(exc)
        elif self.link.due(now):
            self._try_recover(now)

    def _check_motors(self, now: float) -> None:
        """Track whether the platform keeps the motors off (asleep, stopped by its owner).
        The loop never turns them on itself (SPEC 4A, 12A); it resumes when they are on. While
        external control is detected the status is read as often as the platform refreshes it
        (``driver.status_period_s``), no more: a faster read sees nothing new."""
        if self.mode != "active" or now < self._next_motor_check:
            return
        period = self.link.policy.probe_initial_s
        status_period = self.driver.status_period_s
        if self.external and status_period is not None:
            period = status_period
        self._next_motor_check = now + period
        enabled = self._call(self.driver.motors_enabled)
        if enabled is None:
            return
        with self._lock:
            self._motors_now(enabled)

    def _note_hook_error(self, exc: Exception) -> None:
        """Count a failing hook every tick but log it once per distinct message until a clean tick
        (a hook at 50 Hz would otherwise flood the log)."""
        msg = f"{type(exc).__name__}: {exc}"
        self.errors += 1
        self.last_error = msg
        if msg != self._hook_error:
            self._hook_error = msg
            log.warning("motion: %s", msg)

    def _note_error(self, exc: Exception) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        log.warning("motion: %s", self.last_error)

    def _try_recover(self, now: float) -> None:
        """One recovery check; the confirming one re-aligns to the measured pose."""
        try:
            self._probe()
            ok = True
        except Exception as exc:
            self._note_error(exc)
            ok = False
        self.link.report(ok, now)

    def _probe(self) -> None:
        try:
            with self._lock:
                self.driver.read_state()
        except Exception:
            # A dead transport may need a new connection (outside the lock: it can block).
            self.driver.reconnect(self.link.policy.probe_timeout_s)
        with self._lock:
            self.driver.read_state()
            if self.link.would_restore():
                self._resync()
                self._next_motor_check = -math.inf  # learn the motor state on the next tick

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            self.max_jitter_s = max(self.max_jitter_s, abs(now - next_t))
            self.cycle(now, period)
            next_t += period
            delay = next_t - self.clock()
            if delay < 0:
                self.overruns += 1
                next_t = self.clock()
            else:
                self._stop.wait(delay)
