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

Motors (SPEC 4A, 12A): the loop never turns the motors on after start-up. It polls
``driver.motors_enabled`` every ``probe_initial_s``; while the platform keeps them off (robot
asleep, stopped by its owner, a backend restarted without wake-up) it follows the measured pose
without writing, and resumes from that pose once they are on again. Sources keep feeding the
arbiter meanwhile. A robot unreachable at ``start`` begins lost.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from ..config.schema import RoboEmotConfig
from ..drivers import MotorMode, PoseGroupDriver
from ..health import Link, OnChange
from ..manifest import Manifest
from .core import Command, MotionCore

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
            minor_factor=1.0,
            watchdog_return_factor=1.0,
        )
        self._lock = threading.Lock()
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
        self._next_motor_check = -math.inf

    # ---- configuration (SPEC 11A hot reload)

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        with self._lock:  # called from the config watcher thread
            self._apply_config(cfg)

    def _apply_config(self, cfg: RoboEmotConfig) -> None:
        watchdog_ms = min(cfg.safety.watchdog_ms, self.manifest.model.safety.watchdog_ms)
        self.core.arbiter.crossfade_s = cfg.motion.arbiter.crossfade_ms / MS_PER_S
        self.core.watchdog_s = watchdog_ms / MS_PER_S
        self.core.minor_factor = cfg.safety.minor_motion_factor
        self.core.watchdog_return_factor = cfg.safety.watchdog_return_factor
        self.core.limiter.constraint_margin = math.radians(cfg.safety.constraint_margin_deg)
        self.link.policy = cfg.connectivity.robot

    # ---- inputs

    def feed(self, cmd: Command) -> None:
        with self._lock:
            self.core.feed(cmd)

    def set_minor_present(self, present: bool) -> None:
        with self._lock:
            self.core.minor_present = present

    def set_mode(self, mode: MotorMode) -> None:
        with self._lock:
            self.driver.set_mode(mode)
            if mode == "active":
                self._resync()  # the robot may have been moved by hand: start from there
            self.mode = mode

    def snapshot(self) -> dict[str, object]:
        """Loop health for diagnostics (SPEC 12A); safe to call from any thread."""
        with self._lock:
            out: dict[str, object] = {
                "mode": self.mode,
                "motors_off": self.motors_off,
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
        self._thread = threading.Thread(target=self._run, name="motion-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def step_once(self, now: float, dt: float) -> dict[str, float]:
        with self._lock:
            if self.mode != "active" or self.motors_off:
                # limp / teach / motors held off by the platform: follow the measured pose so
                # re-enabling never jumps
                state = self._resync()
                self.ticks += 1
                return state
            out = self.core.tick(now, dt)
            self._call(self.driver.write_pose_targets, now, out)
            self.ticks += 1
            return out

    def _resync(self) -> dict[str, float]:
        measured = self._call(self.driver.read_state)
        state = {k: v for k, v in measured.items() if k in self.core.limiter.pos}
        self.core.limiter.reset(state)
        self.core.arbiter.last = {**self.core.arbiter.last, **state}
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
        The loop never turns them on itself (SPEC 4A, 12A); it resumes when they are on."""
        if self.mode != "active" or now < self._next_motor_check:
            return
        self._next_motor_check = now + self.link.policy.probe_initial_s
        enabled = self._call(self.driver.motors_enabled)
        if enabled is None:
            return
        if enabled and self.motors_off:
            with self._lock:
                self._resync()  # resume from wherever the platform's wake-up left the robot
        self.motors_off = not enabled

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
