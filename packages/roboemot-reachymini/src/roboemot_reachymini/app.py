"""Reachy Mini app wrapper (SPEC 4A, Packaging as a Reachy Mini app).

The daemon starts this as a subprocess and passes a connected ``ReachyMini``; this process is
the SDK bridge (the only owner of the SDK client and its media). It never wakes or sleeps the
robot: the platform does. M0 scope: load config and manifest, run the motion loop with idle
motion, stop cleanly on ``stop_event``.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

from roboemot.config import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority

from .driver import ReachyMiniDriver

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

DATA = Path(__file__).resolve().parent / "data"


def idle_command(cfg: Any, t: float) -> Command:
    """Breathing on head pitch (SPEC 10, Idle; parameters from motion.idle)."""
    idle = cfg.motion.idle
    pitch = idle.breathing_amplitude * math.sin(2 * math.pi * t / idle.breathing_period_s)
    return Command("idle", Priority.IDLE, {"head.pitch": pitch}, t)


class RoboEmotApp(ReachyMiniApp):  # type: ignore[misc,valid-type]
    custom_app_url: str | None = None  # web UI arrives with enrollment (M1)

    def run(self, reachy_mini: Any, stop_event: threading.Event) -> None:
        stack = ConfigStack.from_files(
            app=_optional(DATA / "app.yaml"), preset="default", preset_dirs=[DATA / "presets"]
        )
        cfg = stack.effective()
        # Unmeasured dynamics only when the owner enabled it in the app config (SPEC 12).
        manifest = load_manifest(
            DATA / "reachy-mini-wireless.yaml",
            allow_unmeasured_dynamics=cfg.safety.allow_unmeasured_dynamics,
            safety=cfg.safety,
        )
        loop = MotionLoop(manifest, ReachyMiniDriver(reachy_mini), cfg, rate_hz=50.0)
        stack.subscribe(lambda change: loop.apply_config(change.config))
        stop_watch = stack.start_watcher()
        loop.start()
        try:
            while not stop_event.is_set():
                loop.feed(idle_command(stack.effective(), time.monotonic()))
                stop_event.wait(0.05)
        finally:
            stop_watch.set()
            loop.stop()


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
