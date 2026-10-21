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
the measured pose stops following what it commands. While it writes, the measured pose is
compared with the poses commanded within ``motion.external.lag_window_s`` (tracking lag stays
inside); a joint further than the tolerance from all of them for ``confirm_s`` makes the loop yield:
it follows the measured pose without writing, like ``motors_off``, and reads the motor status every
tick. It resumes, from the measured pose, once the pose has been still for
``motion.platform_move_grace_s`` with the motors on (that grace must outlast the platform's delay
before it switches the motors off after a sleep move); motors off at any point means asleep. A hand
pushing the head is the same: the loop yields and resumes after it settles.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from .. import shutdown
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
        self.external = False  # the measured pose left our commands: track, do not write
        self.external_events = 0
        self._commanded: deque[tuple[float, dict[str, float]]] = deque()
        self._diverged_since: float | None = None
        self._still_since: float | None = None
        self._last_measured: tuple[float, dict[str, float]] | None = None
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
        self.core.minor_factor = cfg.safety.minor_motion_factor
        self.core.watchdog_return_factor = cfg.safety.watchdog_return_factor
        self.core.limiter.constraint_margin = math.radians(cfg.safety.constraint_margin_deg)
        self.link.policy = cfg.connectivity.robot
        self.platform_move_grace_s = cfg.motion.platform_move_grace_s
        self.ext = cfg.motion.external

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

    def set_minor_present(self, present: bool) -> None:
        """Engage or release the minor motion caps (SPEC 12); fed by ``MinorPresence``."""
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
        now = self.clock()
        with self._lock:
            out: dict[str, object] = {
                "mode": self.mode,
                "motors_off": self.motors_off,
                "asleep": self.motors_off and self.link.online,
                "external": self.external,
                "external_events": self.external_events,
                "external_still_s": (
                    0.0 if self._still_since is None else max(0.0, now - self._still_since)
                ),
                "minor_present": self.core.minor_present,
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
            if self.mode != "active" or self.motors_off or self.external:
                # limp / teach / motors held off by the platform / external control: follow the
                # measured pose so re-enabling never jumps
                state = self._resync()
                if self.external and not self.motors_off:
                    self._watch_external(now, state)
                self.ticks += 1
                return state
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
            self._call(self.driver.write_pose_targets, now, out)
            self._detect_external(now, out)
            self.ticks += 1
            return out

    def _tolerance(self, name: str) -> float:
        dof = self.core.limiter.dofs[name]
        return self.ext.tolerance_m if dof.kind == "translation" else self.ext.tolerance_rad

    def _detect_external(self, now: float, out: dict[str, float]) -> None:
        """After a write: compare the measured pose with the poses commanded within the lag
        window; a measured joint further than its tolerance from all of them, for ``confirm_s``
        in a row, means something else controls the robot (see the module docstring)."""
        self._commanded.append((now, dict(out)))
        while self._commanded and now - self._commanded[0][0] > self.ext.lag_window_s:
            self._commanded.popleft()
        measured = self._call(self.driver.read_state)
        diverged = False
        for name, value in measured.items():
            dof = self.core.limiter.dofs.get(name)
            if dof is None:
                continue
            errors = [abs(value - cmd[name]) for _, cmd in self._commanded if name in cmd]
            if errors and min(errors) > self._tolerance(name):
                diverged = True
                break
        if not diverged:
            self._diverged_since = None
            return
        if self._diverged_since is None:
            self._diverged_since = now
        if now - self._diverged_since >= self.ext.confirm_s:
            self.external, self._still_since = True, None
            self._diverged_since = None
            self._commanded.clear()
            self._last_measured = (now, dict(measured))
            self.external_events += 1
            log.info("motion: external control of the robot detected, yielding")

    def _watch_external(self, now: float, measured: dict[str, float]) -> None:
        """While yielding: once the measured pose has been still for ``platform_move_grace_s``
        (the motors are on: the caller checked), take control again from where it is."""
        last = self._last_measured
        self._last_measured = (now, dict(measured))
        if last is None or now <= last[0]:
            return
        dt = now - last[0]
        moving = False
        for name, value in measured.items():
            dof = self.core.limiter.dofs.get(name)
            if dof is None or name not in last[1]:
                continue
            limit = (
                self.ext.still_speed_m_s
                if dof.kind == "translation"
                else self.ext.still_speed_rad_s
            )
            if abs(value - last[1][name]) / dt > limit:
                moving = True
                break
        if moving:
            self._still_since = None
            return
        if self._still_since is None:
            self._still_since = now
        if now - self._still_since >= self.platform_move_grace_s:
            self.external, self._still_since = False, None
            self._commanded.clear()
            log.info("motion: robot still, resuming control")

    def _resync(self) -> dict[str, float]:
        measured = self._call(self.driver.read_state)
        state = {k: v for k, v in measured.items() if k in self.core.limiter.pos}
        self.core.limiter.reset(state)
        self.core.arbiter.last = {**self.core.arbiter.last, **state}
        self._commanded.clear()
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
        The loop never turns them on itself (SPEC 4A, 12A); it resumes when they are on. While a
        external control is detected the status is read every tick (a cached value)."""
        if self.mode != "active" or (now < self._next_motor_check and not self.external):
            return
        self._next_motor_check = now + self.link.policy.probe_initial_s
        enabled = self._call(self.driver.motors_enabled)
        if enabled is None:
            return
        if enabled and self.motors_off:
            with self._lock:
                self._resync()  # resume from wherever the platform's wake-up left the robot
        if not enabled:
            self.external, self._still_since = False, None
        self.motors_off = not enabled
        if self.motors_off:
            self._commanded.clear()

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
