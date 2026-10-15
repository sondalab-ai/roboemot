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
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

from ..config.schema import RoboEmotConfig
from ..drivers import MotorMode, PoseGroupDriver
from ..manifest import Manifest
from .core import Command, MotionCore

MS_PER_S = 1000.0
log = logging.getLogger(__name__)


class MotionLoop:
    def __init__(
        self,
        manifest: Manifest,
        driver: PoseGroupDriver,
        cfg: RoboEmotConfig,
        *,
        rate_hz: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.driver = driver
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

    # ---- lifecycle

    def start(self) -> None:
        self.set_mode("active")
        self._thread = threading.Thread(target=self._run, name="motion-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def step_once(self, now: float, dt: float) -> dict[str, float]:
        with self._lock:
            if self.mode != "active":
                # limp / teach: follow the measured pose so re-enabling never jumps
                state = self._resync()
                self.ticks += 1
                return state
            out = self.core.tick(now, dt)
            self.driver.write_pose_targets(now, out)
            self.ticks += 1
            return out

    def _resync(self) -> dict[str, float]:
        state = {k: v for k, v in self.driver.read_state().items() if k in self.core.limiter.pos}
        self.core.limiter.reset(state)
        self.core.arbiter.last = {**self.core.arbiter.last, **state}
        return state

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            self.max_jitter_s = max(self.max_jitter_s, abs(now - next_t))
            try:
                self.step_once(now, period)
            except Exception as exc:  # keep the loop alive; the watchdog covers missed ticks
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("motion tick failed")
            next_t += period
            delay = next_t - self.clock()
            if delay < 0:
                self.overruns += 1
                next_t = self.clock()
            else:
                self._stop.wait(delay)
