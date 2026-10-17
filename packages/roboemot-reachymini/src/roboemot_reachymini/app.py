"""Reachy Mini app wrapper (SPEC 4A, Packaging as a Reachy Mini app).

The daemon starts this as a subprocess and passes a connected ``ReachyMini``; this process is
the SDK bridge (the only owner of the SDK client and its media). It never wakes or sleeps the
robot: the platform does. M0 scope: load config and manifest, run the motion loop with idle
motion, look at the attention target chosen from the world model (SPEC 5, 10), look around
when idle (SPEC 5A, Scanning: a sweep source fed only while attention is not active, so
attention always wins), track connectivity (SPEC 12A: robot link in the loop, LLM routes when
configured) with the offline responder's gestures, stop cleanly on ``stop_event``.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from roboemot import paths
from roboemot.affect import AffectEngine
from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.attention import AttentionSelector, GazeController
from roboemot.bus import Bus
from roboemot.circadian import CircadianModel, in_window
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics, FnSource, Target, platform_sources
from roboemot.diagnostics.system import local_sources
from roboemot.health.network import NetworkMonitor
from roboemot.health.responder import OfflineResponder
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.persona import effects as persona_effects
from roboemot.vision import CIRCADIAN, FrameSource
from roboemot.vision.service import Perception, default_stages, open_perception, world_snapshot
from roboemot.webui import Server, make_server
from roboemot.webui.catalogue import open_catalogue
from roboemot.world import (
    Observation,
    ObserveResult,
    ScanContext,
    ScanController,
    WorldModel,
    camera_in_view,
    feed_world,
    load_world_plugin,
)
from roboemot.world.autoenroll import AutoEnroller, open_autoenroll, purge_stored
from roboemot.world.scan import SOURCE as SCAN_SOURCE

from .camera import ReachyCamera
from .driver import ReachyMiniDriver
from .gaze import FRAME, GAZE

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)
ADDITIVE_IDLE = "idle-additive"
MINUTES_PER_HOUR = 60.0
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


def scan_context(
    now: float,
    cfg: Any,
    selector: AttentionSelector,
    *,
    hour: float,
    effects: Any = None,
    circadian: CircadianModel | None = None,
    minor_present: bool = False,
) -> ScanContext:
    """What the scan policy needs this tick: engagement from attention, the circadian state and
    quiet hours (SPEC 7A) from the local ``hour``, the persona's energy effects (``effects``,
    computed from the persona when the caller has not cached them) and the minor factor
    (SPEC 12)."""
    quiet = cfg.circadian.enabled and in_window(hour, cfg.circadian.quiet_hours)
    target = selector.target
    return ScanContext(
        now=now,
        engaged=selector.engaged,
        object_target=None if target is None or selector.engaged else target.key,
        object_max_hold_s=cfg.attention.object_max_hold_s,
        circadian=circadian.state if circadian else "awake",
        quiet_hours=quiet,
        effects=effects if effects is not None else persona_effects(cfg.persona),
        minor_factor=cfg.safety.minor_motion_factor if minor_present else 1.0,
    )


def scan_tick(ctx: ScanContext, scan: ScanController, loop: MotionLoop, cfg: Any) -> bool:
    """One scan step, run right after ``attention_tick``: feed the sweep command or release it.

    The policy cancels a sweep the tick attention engages with a person or a speaker, and this
    clears the ``scan`` source in the same tick, so that attention wins within one tick (SPEC
    10). Attention on an animal or object holds a due scan off for ``attention.object_max_hold_s``
    (the policy); once the sweep runs, that command is dropped here (it is also idle tier), so
    the sweep wins deterministically. Returns
    whether a sweep or glance is in progress. A failure is logged (type and message only), the
    sweep is dropped and idle motion carries on; the app keeps running.
    """
    try:
        was_active = scan.active
        fov = math.radians(cfg.world.camera_fov_h_deg)
        cmd = scan.tick(ctx, loop.pose(), fov)
        if cmd is not None:
            if not ctx.engaged:
                loop.clear(ATTENTION_SOURCE)
            loop.feed(cmd)
        elif was_active:
            loop.clear(SCAN_SOURCE)
        return scan.active
    except Exception as exc:
        log.warning("scan: %s: %s", type(exc).__name__, exc)
        scan.reset()
        loop.clear(SCAN_SOURCE)
        return False


def circadian_vision_scale(cfg: Any, circadian: CircadianModel | None) -> float:
    """The vision rate multiplier of the circadian state (SPEC 7A): the running model's state,
    or ``awake`` while no model is attached (the circadian runner arrives in M2)."""
    effects = circadian.effects() if circadian is not None else cfg.circadian.states["awake"]
    return effects.vision_rate_scale


def apply_attention_config(
    cfg: Any, world: WorldModel, selector: AttentionSelector, gaze: GazeController
) -> None:
    """Hot reload (SPEC 11A): hand the new world and attention config to their consumers."""
    world.cfg = cfg.world
    selector.cfg = gaze.cfg = cfg.attention
    gaze.margin_rad = math.radians(cfg.safety.constraint_margin_deg)


class RoboEmotApp(ReachyMiniApp):  # type: ignore[misc,valid-type]
    # The enrollment and registry web UI (SPEC 4A, 6): the daemon reads this class attribute to
    # show the page; the port is webui.port (change both together). The app serves the page
    # itself (roboemot.webui), so the SDK must not start its own server on that port.
    custom_app_url: str | None = f"http://0.0.0.0:{RoboEmotConfig().webui.port}"
    dont_start_webserver: bool = True
    webui: Server | None = None
    diagnostics: Diagnostics | None = None
    bus: Bus | None = None
    world: WorldModel | None = None  # vision (M1) and the speaker hint (M3) feed these
    selector: AttentionSelector | None = None
    circadian: CircadianModel | None = None  # the circadian runner (M2) sets this
    minor_present: bool = False  # the registry's age estimates (M1) set this
    perception: Perception | None = None
    enrollment: Any = None  # EnrollmentService: the web UI enrolls through it (SPEC 6)
    autoenroll: AutoEnroller | None = None  # animals and objects that keep coming back (SPEC 5A)
    catalogue: Any = None  # CatalogueService: XMP import, confirm "this is Mia" (SPEC 6)

    def make_frame_source(self, mini: Any, joint_state: Any) -> FrameSource:
        """The camera the vision pump reads (overridable: tests inject a planted scene)."""
        return ReachyCamera(mini, joint_state)

    def load_vision_stages(self, cfg: Any, profile: str | None) -> Any:
        """The vision backends (overridable: tests inject fakes)."""
        return default_stages(cfg, profile)

    def start_webui(
        self, stack: ConfigStack, cfg: Any, manifest: Any, cleanup: ExitStack
    ) -> Server | None:
        """Serve the web UI over the app's enrollment service and diagnostics registry.

        The server is stopped by ``cleanup`` (registered before it starts). A port that is
        taken or an unusable data folder is logged (type only) and the app runs without the
        page: enrollment is a side channel, not a reason to stop the robot.
        """
        try:
            server = make_server(
                self.perception.enrollment,
                self.diagnostics,
                stack.effective,
                paths.data_dir(cfg.data_dir),
                robot_hosted=manifest.robot.profile is not None,
                catalogue=self.catalogue,
            )
        except Exception as exc:
            log.error("web UI not started: %s", type(exc).__name__)
            return None
        cleanup.callback(server.stop)
        server.start()
        self.webui = server
        log.info("web UI on %s:%d", server.host, server.port)
        return server

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
        plugin = load_world_plugin(manifest, cfg, reachy_mini, loop.driver.read_state)
        scan = (
            ScanController(cfg.world.scan, plugin.planner, manifest.dofs)
            if plugin.planner is not None
            else None
        )

        def on_result(obs: Observation, result: ObserveResult) -> None:
            if scan is not None and result.unexplained_bearing is not None:
                scan.policy.note_unexplained(*result.unexplained_bearing, obs.stamp)
            if self.autoenroll is not None:  # runs on the vision thread: never let it raise
                try:
                    self.autoenroll.observe(obs, result)
                except Exception as exc:
                    log.warning("auto-enrollment: %s", type(exc).__name__)

        feed_world(bus, world, on_result)
        source = self.make_frame_source(reachy_mini, loop.driver.read_state)
        perception = self.perception = open_perception(
            cfg,
            bus,
            source,
            profile=manifest.robot.profile,
            camera_model=getattr(source, "camera_model", None),
            load_stages=self.load_vision_stages,
        )
        self.enrollment = perception.enrollment
        appearance = getattr(perception.pipeline, "appearance_embedder", None)
        self.autoenroll = open_autoenroll(
            perception.registry,
            appearance,
            cfg.world.autoenroll,
            world=world,
            language=lambda: stack.effective().conversation.language,
        )
        self.catalogue = open_catalogue(
            perception.registry,
            stack.effective,
            embedder=lambda: getattr(perception.pipeline, "appearance_embedder", None),
            world=world,
            purge_candidates=(
                self.autoenroll.purge
                if self.autoenroll
                else (lambda: purge_stored(perception.registry.store))
                if perception.registry is not None
                else None
            ),
        )

        def publish(obs: Observation, joint_state: Any) -> None:
            bus.publish("percept.entity", obs, obs.stamp, joint_state=joint_state)

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
            FnSource("roboemot.vision", perception.vision_snapshot),
            FnSource("roboemot.world", lambda: world_snapshot(world.entries, time.monotonic())),
            FnSource("roboemot.registry", perception.registry_snapshot),
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
            if scan is not None:
                scan.apply_config(change.config.world.scan)
                plugin.planner.apply_config(change.config)
            perception.apply_config(change.config)
            if self.autoenroll is not None:
                self.autoenroll.apply_config(change.config.world.autoenroll)
            if self.catalogue is not None:
                self.catalogue.cfg = change.config.catalogue
            responder.cfg = change.config.connectivity
            network.set_policy(change.config.connectivity.network)

        stack.subscribe(reload)
        fx_for, fx = None, None
        # Everything started below is stopped in reverse order whatever fails: ExitStack runs
        # every callback even when one raises, so the motion loop always stops and the camera
        # thread is quiet before the SDK client goes away.
        with ExitStack() as cleanup:
            cleanup.callback(stack.start_watcher().set)
            loop.start()
            cleanup.callback(loop.stop)
            network.start()
            cleanup.callback(network.stop)
            if self.autoenroll is not None:  # runs after perception.stop: no new sightings
                cleanup.callback(self.autoenroll.flush)
            cleanup.callback(perception.stop)
            perception.start()
            self.start_webui(stack, cfg, manifest, cleanup)
            for world_source in plugin.sources:
                cleanup.callback(world_source.stop)
                world_source.start(publish)
            while not stop_event.is_set():
                now = time.monotonic()
                attending = attention_tick(now, world, selector, gaze, loop)
                live = stack.effective()
                perception.enrollment.expire()
                if self.autoenroll is not None:
                    self.autoenroll.tick()  # retention does not wait for a sighting
                if perception.pipeline is not None:
                    perception.pipeline.governor.set_scale(
                        CIRCADIAN, circadian_vision_scale(live, self.circadian)
                    )
                if scan is not None:
                    local = datetime.now()
                    hour = local.hour + local.minute / MINUTES_PER_HOUR
                    if live.persona is not fx_for:  # recompute only after a reload
                        fx_for, fx = live.persona, persona_effects(live.persona)
                    ctx = scan_context(
                        now,
                        live,
                        selector,
                        effects=fx,
                        hour=hour,
                        circadian=self.circadian,
                        minor_present=self.minor_present,
                    )
                    scan_tick(ctx, scan, loop, live)
                feed_idle(loop, live, now, attending)
                for cmd in responder.commands(now):
                    loop.feed(cmd)
                stop_event.wait(0.05)


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
