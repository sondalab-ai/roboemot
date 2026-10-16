"""Reachy Mini app wrapper (SPEC 4A, Packaging as a Reachy Mini app).

The daemon starts this as a subprocess and passes a connected ``ReachyMini``; this process is
the SDK bridge (the only owner of the SDK client and its media). It never wakes or sleeps the
robot: the platform does. M0 scope: load config and manifest, run the motion loop with idle
motion, look at the attention target chosen from the world model (SPEC 5, 10), track
connectivity (SPEC 12A: robot link in the loop, LLM routes when configured) with
the offline responder's gestures, stop cleanly on ``stop_event``.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from roboemot.affect import AffectEngine
from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.attention import AttentionSelector, GazeController
from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.diagnostics import Diagnostics, FnSource, Target, platform_sources
from roboemot.diagnostics.system import local_sources
from roboemot.health.network import NetworkMonitor
from roboemot.health.responder import OfflineResponder
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.persona import effects as persona_effects
from roboemot.world import WorldModel, camera_in_view

from .driver import ReachyMiniDriver
from .gaze import FRAME, GAZE

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)
ADDITIVE_IDLE = "idle-additive"
DATA = Path(__file__).resolve().parent / "data"


def idle_command(cfg: Any, t: float) -> Command:
    """Breathing on head pitch (SPEC 10, Idle; parameters from motion.idle, period scaled by
    the persona's energy, SPEC 11A)."""
    idle = cfg.motion.idle
    period = idle.breathing_period_s * persona_effects(cfg.persona).idle_period_x
    pitch = idle.breathing_amplitude * math.sin(2 * math.pi * t / period)
    return Command("idle", Priority.IDLE, {"head.pitch": pitch}, t)


def feed_idle(loop: MotionLoop, cfg: Any, now: float, attending: bool) -> None:
    """Feed idle breathing. While attention owns the head it rides on top as an additive layer
    (additive commands stack on any winner of idle priority or lower precedence); otherwise it
    is the plain idle command."""
    cmd = idle_command(cfg, now)
    if attending:
        loop.feed(
            replace(
                cmd, source=ADDITIVE_IDLE, additive=True, expires=now + cfg.attention.command_ttl_s
            )
        )
    else:
        loop.clear(ADDITIVE_IDLE)
        loop.feed(cmd)


def attention_tick(
    now: float,
    world: WorldModel,
    selector: AttentionSelector,
    gaze: GazeController,
    loop: MotionLoop,
) -> bool:
    """One attention step: decay the world, pick the target, feed or release the gaze command.

    Returns whether attention owns the head. A failure inside is logged (type and message
    only), the attention command is dropped and idle motion takes over; the app keeps running.
    """
    try:
        wc = world.cfg
        in_view = camera_in_view(
            loop.pose(), FRAME, math.radians(wc.camera_fov_h_deg), math.radians(wc.camera_fov_v_deg)
        )
        world.tick(now, in_view)
        target = selector.update(now, world.entries)
        was_active = gaze.active
        cmd = gaze.step(now, target, None if gaze.active else loop.pose())
        if cmd is not None:
            loop.feed(cmd)
        elif was_active:
            loop.clear(ATTENTION_SOURCE)
        return gaze.active
    except Exception as exc:
        log.warning("attention: %s: %s", type(exc).__name__, exc)
        gaze.release()
        loop.clear(ATTENTION_SOURCE)
        return False


def apply_attention_config(
    cfg: Any, world: WorldModel, selector: AttentionSelector, gaze: GazeController
) -> None:
    """Hot reload (SPEC 11A): hand the new world and attention config to their consumers."""
    world.cfg = cfg.world
    selector.cfg = gaze.cfg = cfg.attention
    gaze.margin_rad = math.radians(cfg.safety.constraint_margin_deg)


class RoboEmotApp(ReachyMiniApp):  # type: ignore[misc,valid-type]
    custom_app_url: str | None = None  # web UI arrives with enrollment (M1)
    diagnostics: Diagnostics | None = None
    bus: Bus | None = None
    world: WorldModel | None = None  # vision (M1) and the speaker hint (M3) feed these
    selector: AttentionSelector | None = None

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
        bus = self.bus = Bus()
        world = self.world = WorldModel(cfg.world, FRAME, bus)
        selector = self.selector = AttentionSelector(cfg.attention, bus)
        gaze = GazeController(
            cfg.attention,
            GAZE,
            manifest.dofs,
            manifest.constraints,
            margin_rad=math.radians(cfg.safety.constraint_margin_deg),
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
            apply_attention_config(change.config, world, selector, gaze)
            responder.cfg = change.config.connectivity
            network.set_policy(change.config.connectivity.network)

        stack.subscribe(reload)
        stop_watch = stack.start_watcher()
        loop.start()
        network.start()
        try:
            while not stop_event.is_set():
                now = time.monotonic()
                attending = attention_tick(now, world, selector, gaze, loop)
                feed_idle(loop, stack.effective(), now, attending)
                for cmd in responder.commands(now):
                    loop.feed(cmd)
                stop_event.wait(0.05)
        finally:
            stop_watch.set()
            network.stop()
            loop.stop()


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
