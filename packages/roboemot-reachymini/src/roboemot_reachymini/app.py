"""Reachy Mini app wrapper (SPEC 4A, Packaging as a Reachy Mini app).

The daemon starts this as a subprocess and passes a connected ``ReachyMini``; this process is
the SDK bridge (the only owner of the SDK client and its media). It never wakes or sleeps the
robot: the platform does. M0 scope: load config and manifest, run the motion loop with idle
motion, track connectivity (SPEC 12A: robot link in the loop, LLM routes when configured) with
the offline responder's gestures, stop cleanly on ``stop_event``.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from roboemot.affect import AffectEngine
from roboemot.config import ConfigStack
from roboemot.diagnostics import Diagnostics, FnSource, Target, platform_sources
from roboemot.diagnostics.system import local_sources
from roboemot.health.network import NetworkMonitor
from roboemot.health.responder import OfflineResponder
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.persona import effects as persona_effects

from .driver import ReachyMiniDriver

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

DATA = Path(__file__).resolve().parent / "data"


def idle_command(cfg: Any, t: float) -> Command:
    """Breathing on head pitch (SPEC 10, Idle; parameters from motion.idle, period scaled by
    the persona's energy, SPEC 11A)."""
    idle = cfg.motion.idle
    period = idle.breathing_period_s * persona_effects(cfg.persona).idle_period_x
    pitch = idle.breathing_amplitude * math.sin(2 * math.pi * t / period)
    return Command("idle", Priority.IDLE, {"head.pitch": pitch}, t)


class RoboEmotApp(ReachyMiniApp):  # type: ignore[misc,valid-type]
    custom_app_url: str | None = None  # web UI arrives with enrollment (M1)
    diagnostics: Diagnostics | None = None

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
        responder = OfflineResponder(
            cfg.connectivity,
            set(manifest.dofs),
            cfg.conversation.language,
            affect=AffectEngine(cfg.affect, cfg.appraisal),
        )
        loop = MotionLoop(
            manifest,
            ReachyMiniDriver(reachy_mini),
            cfg,
            rate_hz=50.0,
            on_link_change=responder.on_event,
        )
        # The conversation pipeline (M3) will share this monitor with its Router.
        network = NetworkMonitor(
            cfg.llm_routes, cfg.connectivity.network, on_change=responder.on_event
        )

        affect = responder.affect
        self.diagnostics = Diagnostics().add(
            *local_sources(process_pid=os.getpid()),
            *platform_sources(manifest.robot.id, Target()),
            FnSource("roboemot.motion", loop.snapshot),
            FnSource("roboemot.network", network.snapshot),
            FnSource(
                "roboemot.app",
                lambda: {
                    "manifest": manifest.robot.id,
                    "language": stack.effective().conversation.language,
                    "valence": affect.valence if affect else None,
                    "arousal": affect.arousal if affect else None,
                },
            ),
        )

        def reload(change: Any) -> None:
            loop.apply_config(change.config)
            responder.cfg = change.config.connectivity
            network.set_policy(change.config.connectivity.network)

        stack.subscribe(reload)
        stop_watch = stack.start_watcher()
        loop.start()
        network.start()
        try:
            while not stop_event.is_set():
                now = time.monotonic()
                loop.feed(idle_command(stack.effective(), now))
                for cmd in responder.commands(now):
                    loop.feed(cmd)
                stop_event.wait(0.05)
        finally:
            stop_watch.set()
            network.stop()
            loop.stop()


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
