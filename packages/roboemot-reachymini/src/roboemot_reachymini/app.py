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
from pathlib import Path
from typing import Any

from roboemot import paths
from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import RegisterTracker
from roboemot.affect.runner import AffectRunner
from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.attention import AttentionSelector, GazeController
from roboemot.bus import Bus
from roboemot.circadian import (
    CircadianModel,
    CircadianRunner,
    IdleLightReader,
    SystemClock,
    in_window,
)
from roboemot.circadian.motion import REST_SOURCE
from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation import ConversationWorker
from roboemot.conversation.preferences import RegistryPreferences
from roboemot.diagnostics import Diagnostics, FnSource, Target, platform_sources
from roboemot.diagnostics.system import local_sources
from roboemot.health.network import NetworkMonitor
from roboemot.health.responder import OfflineResponder
from roboemot.llm import Router
from roboemot.llm.hub import RouterHub
from roboemot.llm.router import SECRET
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.persona import effects as persona_effects
from roboemot.persona.llm_traits import TraitCache, TraitsWorker
from roboemot.reflex import ReflexEngine
from roboemot.skills.defaults import default_skills
from roboemot.vision import CIRCADIAN, FrameSource, LockedSource
from roboemot.vision.minor import MinorPresence
from roboemot.vision.service import Perception, default_stages, open_perception, world_snapshot
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import BargeIn, PlaybackReference
from roboemot.voice.factory import build_speech
from roboemot.voice.listener import Utterance, VoiceListener
from roboemot.voice.speech import SpeechOutput
from roboemot.webui import Server, make_server
from roboemot.webui.catalogue import open_catalogue
from roboemot.webui.routes import RouteChecker
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

from .audio import ReachyAudio
from .camera import ReachyCamera
from .conversation import profile_source, scene_source, speaker_source
from .driver import ReachyMiniDriver
from .gaze import FRAME, GAZE, REFLEX
from .skills import SkillBridge
from .speaker import ReachySpeaker
from .voice import build_listener

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)
ADDITIVE_IDLE = "idle-additive"
MS_PER_S = 1000.0
DATA = Path(__file__).resolve().parent / "data"


def idle_command(cfg: Any, t: float, scale: float = 1.0, phase_t: float | None = None) -> Command:
    """Breathing on head pitch (SPEC 10, Idle; parameters from motion.idle, period scaled by
    the persona's energy, SPEC 11A). ``scale`` is the circadian state's idle motion scale
    (SPEC 7A); ``phase_t`` is the breathing phase clock (default ``t``): the caller integrates it
    at the state's slower pace so the head does not jump when the period changes."""
    idle = cfg.motion.idle
    period = idle.breathing_period_s * persona_effects(cfg.persona).idle_period_x
    phase = t if phase_t is None else phase_t
    pitch = scale * idle.breathing_amplitude * math.sin(2 * math.pi * phase / period)
    return Command("idle", Priority.IDLE, {"head.pitch": pitch}, t)


def feed_idle(
    loop: MotionLoop,
    cfg: Any,
    now: float,
    attending: bool,
    scale: float = 1.0,
    phase_t: float | None = None,
) -> None:
    """Feed idle breathing. While attention owns the head it rides on top as an additive layer
    (additive commands stack on any winner of idle priority or lower precedence); otherwise it
    is the plain idle command. A ``scale`` of 0 (circadian resting) feeds none: the rest posture
    replaces it."""
    if scale <= 0:
        loop.clear("idle")
        loop.clear(ADDITIVE_IDLE)
        return
    cmd = idle_command(cfg, now, scale, phase_t)
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
    or ``awake`` while no model is attached (circadian disabled)."""
    effects = circadian.effects() if circadian is not None else cfg.circadian.states["awake"]
    return effects.vision_rate_scale


def affect_report(engine: Any, now: float) -> dict[str, Any]:
    """Affect fields of the debug report: the numbers and the label, nothing about who or why."""
    state = engine.state(now)
    return {
        "valence": state.valence,
        "arousal": state.arousal,
        "label": state.label,
    }


def register_report(tracker: RegisterTracker) -> dict[str, Any]:
    """Register fields of the debug report: the group and its parameters, no person ids."""
    return tracker.active.summary() if tracker.active is not None else {}


def _register_overrides(registry: Any, person_id: str) -> dict[str, Any]:
    """A registered person's ``register_overrides`` ({} without a registry). A missing record or
    a registry error raises: the tracker then resolves that person as ``unknown``."""
    if registry is None:
        return {}
    return dict(registry.get_person(person_id).register_overrides)


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
    circadian: CircadianModel | None = None  # the circadian runner's model; None while disabled
    circadian_runner: CircadianRunner | None = None  # light, presence, state, motion (SPEC 7A)
    light_reader: IdleLightReader | None = None  # reads a frame when vision gave none
    affect_runner: AffectRunner | None = None  # owns the one affect engine
    minor_presence: MinorPresence | None = None  # the one source of the minor-present signal
    register_tracker: RegisterTracker | None = None  # people in view -> register.active
    expression: ExpressionRenderer | None = None  # affect label -> additive antenna/head motion
    reflexes: ReflexEngine | None = None  # runs on the motion loop's tick (SPEC 10)
    audio: ReachyAudio | None = None  # the one reader of the microphone (reflex level, voice)
    speaker: ReachySpeaker | None = None  # playback; only when voice.enabled
    speech: SpeechActivity | None = None  # "the robot is speaking" gate (reflex, DoA)
    bargein: BargeIn | None = None  # user speech over the robot's voice (SPEC 8 stage 8)
    voice_listener: VoiceListener | None = None  # mic -> VAD -> ASR -> utterances; voice.enabled
    speech_output: SpeechOutput | None = None  # TTS queue onto the speaker; only when voice.enabled
    traits_worker: TraitsWorker | None = None  # rates persona traits when traits_source is llm
    router_hub: RouterHub | None = None  # the one router: traits, conversation, speech, checks
    conversation: ConversationWorker | None = None  # answers addressed utterances (voice.enabled)

    def make_router(self, cfg: Any, monitor: NetworkMonitor) -> Router:
        """The app's LLM router over the configured routes (overridable: tests inject fakes)."""
        return Router(cfg.llm_routes, cfg.safety, monitor=monitor)

    def on_utterance(self, utterance: Utterance) -> None:
        """Where finished utterances go (SPEC 8): the conversation worker, which answers the
        addressed ones; overridable, tests inject a collector."""
        if self.conversation is not None:
            self.conversation.submit(utterance)

    def make_circadian_clock(self) -> Any:
        """The clock the circadian rhythm runs on: monotonic seconds and the local hour of day
        (overridable: tests inject a fixed or accelerated clock)."""
        return SystemClock()

    def _forget_circadian(self) -> None:
        """The app stopped: hold and wake report that it is not running."""
        self.circadian_runner = None

    def hold(self, state: str, minutes: float) -> None:
        """``robot.circadian.hold``: keep the robot ``awake`` (or ``drowsy`` or ``resting``) for
        ``minutes``, e.g. during a party (SPEC 7A, API)."""
        runner = self.circadian_runner
        if runner is None:
            raise RuntimeError("the app is not running")
        runner.hold(state, minutes)

    def wake(self) -> None:
        """Wake the robot as the wake word or a direct address does (SPEC 7A, Waking)."""
        runner = self.circadian_runner
        if runner is None:
            raise RuntimeError("the app is not running")
        runner.wake()

    def detection_period(self) -> float:
        """Seconds between face-stage runs now (infinite while vision is off)."""
        pipeline = self.perception.pipeline if self.perception is not None else None
        return math.inf if pipeline is None else pipeline.governor.period("detection")

    @property
    def minor_present(self) -> bool:
        """Whether a possible minor is in view or was within the hold time (SPEC 5, 12)."""
        return self.minor_presence.present() if self.minor_presence is not None else False

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

    def route_check_router(self) -> Router | None:
        """The router a web UI route check is charged to: the app-wide one shared with the
        traits and the conversation (SPEC 8), None before it exists, which gives the checker a
        private one."""
        return self.router_hub.router if self.router_hub else None

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
                routes=RouteChecker(
                    stack.effective,
                    paths.route_checks_path(cfg.data_dir),
                    router=self.route_check_router,
                ),
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
        stack.set_manifest_behaviour(manifest.model.behaviour)  # the expression map (SPEC 11A)
        cfg = stack.effective()
        responder = OfflineResponder(
            cfg.connectivity,
            set(manifest.dofs),
            cfg.conversation.language,
        )
        loop = MotionLoop(
            manifest,
            ReachyMiniDriver(reachy_mini),
            cfg,
            rate_hz=50.0,
            on_link_change=responder.on_event,
        )
        bus = self.bus = Bus()
        # Reflexes run in the motion loop's own tick (not the 20 Hz app tick below): audio level,
        # voice bearing and face size in, REFLEX-priority commands to the arbiter (SPEC 10).
        reflexes = self.reflexes = ReflexEngine(
            cfg.reflexes,
            REFLEX,
            manifest.dofs,
            manifest.constraints,
            margin_rad=math.radians(cfg.safety.constraint_margin_deg),
            bus=bus,
        )
        loop.tick_hooks.append(reflexes.on_tick)
        # The robot's own voice must not startle it or draw its gaze: the speaker feeds this
        # window (playback end + echo tail) and the reflex engine and the DoA source read it.
        speech = self.speech = SpeechActivity(cfg.voice.echo_tail_ms)
        reflexes.speaking = speech.speaking
        reflex_audio = cfg.reflexes.enabled and cfg.reflexes.audio.enabled
        audio = self.audio = ReachyAudio(
            reachy_mini,
            reflexes.offer_audio_level if reflex_audio else None,
            cfg.reflexes.audio.poll_hz,
            max_gap_s=cfg.reflexes.max_age_ms / MS_PER_S,
            voice=cfg.voice.enabled,
            voice_rate_hz=cfg.voice.sample_rate_hz,
            voice_queue_s=cfg.voice.capture_queue_s,
        )
        reflexes.audio_status = audio.snapshot
        reference = (
            PlaybackReference(cfg.voice.sample_rate_hz, cfg.voice.barge_in.echo_reference_s)
            if cfg.voice.enabled
            else None
        )
        speaker = self.speaker = (
            ReachySpeaker(reachy_mini, cfg.voice.playback_chunk_ms, speech, reference=reference)
            if cfg.voice.enabled
            else None
        )
        # The one affect engine: appraisal events from the bus and the responder feed it, the
        # app tick decays it and publishes affect.state (SPEC 7).
        runner = self.affect_runner = AffectRunner(
            cfg.affect, cfg.appraisal, bus, time.monotonic(), self.detection_period
        )
        responder.affect = runner.engine
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
        # No face frames (vision off or dead) keep the caps on: release needs evidence (SPEC 12).
        presence = self.minor_presence = MinorPresence(
            bus, cfg.safety, sink=loop.set_minor_present, period=lambda: self.detection_period()
        )
        presence.apply(time.monotonic())  # the loop's caps are on from its first tick
        reflexes.attach(bus)  # after MinorPresence: it can never come before the minor latch
        tracker = self.register_tracker = RegisterTracker(
            bus,
            cfg.expression,
            overrides_for=lambda person_id: _register_overrides(perception.registry, person_id),
            detection_period=self.detection_period,
        )
        expression = self.expression = ExpressionRenderer(
            cfg.expression,
            cfg.expression_map,
            manifest.dofs,
            cfg.motion.arbiter.crossfade_ms / MS_PER_S,
        )
        clock = self.make_circadian_clock()
        circ = self.circadian_runner = CircadianRunner(
            cfg,
            bus,
            runner,
            clock=clock,
            dofs=manifest.dofs,
            detection_period=self.detection_period,
            engaged=lambda: selector.engaged,
        )
        expression.circadian_factor = circ.amplitude_factor
        self.circadian = circ.model if cfg.circadian.enabled else None
        # One lock for the vision pump and the idle light reader: the SDK media is read by one
        # thread at a time.
        source = LockedSource(self.make_frame_source(reachy_mini, loop.driver.read_state))
        perception = self.perception = open_perception(
            cfg,
            bus,
            source,
            profile=manifest.robot.profile,
            camera_model=getattr(source, "camera_model", None),
            load_stages=self.load_vision_stages,
        )
        if perception.pipeline is not None:  # light from the frames vision already captures
            perception.pipeline.light_tap = circ.sampler.offer
        light_reader = self.light_reader = IdleLightReader(circ.sampler, source)
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

        # The one router (traits, conversation, speech, route checks) reports to this monitor.
        network = NetworkMonitor(
            cfg.llm_routes, cfg.connectivity.network, on_change=responder.on_event
        )

        # Trait rating (SPEC 11A, Persona) runs on its own thread; its cache is in the data folder.
        for route in cfg.llm_routes:  # the app has no secret store yet (TODO.md)
            if SECRET.match(route.api_key):
                log.warning(
                    "llm route %s: api_key is a secret reference and no secret store is wired: "
                    "requests go out with an empty key",
                    route.name,
                )
        try:
            paths.data_dir(cfg.data_dir)  # created private (0700) when missing, SPEC 12
        except OSError as exc:  # the cache then fails to write and the rating is just not kept
            log.warning("data folder: %s", type(exc).__name__)
        hub = self.router_hub = RouterHub(
            self.make_router(cfg, network), lambda c: self.make_router(c, network)
        )
        worker = self.traits_worker = TraitsWorker(
            stack, None, TraitCache(paths.traits_cache_path(cfg.data_dir)), None, hub=hub
        )

        voice_pipeline = (
            build_listener(
                cfg,
                audio,
                speech,
                bus,
                runner.on_event,
                selector,
                world,
                perception.registry,
                FRAME,
                lambda u: self.on_utterance(u),
                reference,
            )
            if cfg.voice.enabled
            else None
        )
        listener = self.voice_listener = voice_pipeline[0] if voice_pipeline else None
        voice_close = voice_pipeline[1] if voice_pipeline else None
        bargein = self.bargein = voice_pipeline[2] if voice_pipeline else None
        if cfg.voice.enabled and listener is None:
            audio.set_voice(False)  # no consumer: the hub must not queue audio for nobody
        affect = runner.engine
        conv_speaker = speaker_source(selector, world)

        def value_cap() -> int:
            return stack.effective().conversation.summary_value_max_chars

        self.diagnostics = Diagnostics().add(
            *local_sources(process_pid=os.getpid()),
            *platform_sources(manifest.robot.id, Target()),
            FnSource("roboemot.motion", loop.snapshot),
            FnSource("roboemot.network", network.snapshot),
            FnSource("roboemot.vision", perception.vision_snapshot),
            FnSource("roboemot.world", lambda: world_snapshot(world.entries, time.monotonic())),
            FnSource("roboemot.registry", perception.registry_snapshot),
            FnSource("roboemot.minor", presence.snapshot),
            FnSource("roboemot.register", lambda: register_report(tracker)),
            FnSource("roboemot.circadian", circ.snapshot),
            FnSource("roboemot.reflexes", reflexes.snapshot),
            FnSource(
                "roboemot.voice",
                lambda: (
                    {
                        **listener.snapshot(),
                        **({"conversation": c.snapshot()} if (c := self.conversation) else {}),
                        **({"bargein": bargein.snapshot()} if bargein is not None else {}),
                    }
                    if listener is not None
                    else {"state": "off"}
                ),
            ),
            FnSource(
                "roboemot.app",
                lambda: {
                    "manifest": manifest.robot.id,
                    "language": stack.effective().conversation.language,
                    **affect_report(affect, time.monotonic()),
                },
            ),
        )

        def reload(change: Any) -> None:
            hub.sync(change.config)  # the web UI check and the speech backend see it at once
            loop.apply_config(change.config)
            presence.configure(change.config.safety)
            apply_attention_config(change.config, world, selector, gaze)
            if scan is not None:
                scan.apply_config(change.config.world.scan)
                plugin.planner.apply_config(change.config)
            perception.apply_config(change.config)
            if self.autoenroll is not None:
                self.autoenroll.apply_config(change.config.world.autoenroll)
            if self.catalogue is not None:
                self.catalogue.cfg = change.config.catalogue
            runner.apply_config(change.config.affect, change.config.appraisal)
            circ.apply_config(change.config)
            speech.set_echo_tail_ms(change.config.voice.echo_tail_ms)
            if listener is not None:
                listener.apply_config(change.config)
            if bargein is not None:
                bargein.apply_config(change.config)
            if self.speech_output is not None:
                self.speech_output.config = change.config
            if self.conversation is not None:
                self.conversation.apply_config(change.config)
            reflexes.apply_config(
                change.config.reflexes, math.radians(change.config.safety.constraint_margin_deg)
            )
            tracker.cfg = change.config.expression
            expression.apply_config(
                change.config.expression,
                change.config.expression_map,
                change.config.motion.arbiter.crossfade_ms / MS_PER_S,
            )
            responder.cfg = change.config.connectivity
            network.set_policy(change.config.connectivity.network)

        stack.subscribe(reload)
        stack.subscribe(worker.on_change)
        fx_for, fx = None, None
        idle_last = idle_t = time.monotonic()  # idle breathing phase clock (SPEC 7A, Drowsy)
        # Everything started below is stopped in reverse order whatever fails: ExitStack runs
        # every callback even when one raises, so the motion loop always stops and the camera
        # thread is quiet before the SDK client goes away.
        with ExitStack() as cleanup:
            cleanup.callback(presence.close)
            cleanup.callback(tracker.close)
            cleanup.callback(runner.close)
            cleanup.callback(circ.close)
            cleanup.callback(reflexes.close)
            cleanup.callback(stack.start_watcher().set)
            loop.start()
            cleanup.callback(loop.stop)
            network.start()
            cleanup.callback(network.stop)
            cleanup.callback(worker.close)  # before the network monitor its router reports to
            worker.start()
            if self.autoenroll is not None:  # runs after perception.stop: no new sightings
                cleanup.callback(self.autoenroll.flush)
            cleanup.callback(perception.stop)
            perception.start()
            if reflex_audio or listener is not None:
                cleanup.callback(audio.stop)  # before the SDK client goes away
                audio.start()  # degrades to "unavailable" in the debug report without a mic
            if speaker is not None:
                cleanup.callback(speaker.close)  # flushes the player before the SDK goes away
                speaker.start_playing()
                # Closed before the speaker (callbacks run last in, first out).
                output = self.speech_output = build_speech(
                    cfg, speaker, lambda: hub.router, bus, lambda: self.minor_present
                )
                cleanup.callback(output.close)
                output.start()
                if listener is not None:
                    # Shutdown order: listener (no new utterances), conversation (cancels the
                    # stream in flight), speech output, speaker.
                    conv = self.conversation = ConversationWorker(
                        stack,
                        hub,
                        output,
                        affect=runner.engine,
                        responder=responder,
                        register=lambda: tracker.active.params if tracker.active else None,
                        circadian=lambda: (
                            circ.model.effects()
                            if self.circadian
                            else stack.effective().circadian.states["awake"]
                        ),
                        minor_present=lambda: self.minor_present,
                        profile=profile_source(selector, world, perception.registry, value_cap),
                        scene=scene_source(
                            selector,
                            world,
                            perception.registry,
                            lambda: stack.effective().attention.stale_s,
                            value_cap,
                        ),
                        speaker=conv_speaker,
                        skills=default_skills(
                            SkillBridge(
                                expression, selector, world, perception.registry, stack.effective
                            ).env(runner.engine)
                        ),
                        preferences=(
                            RegistryPreferences(
                                perception.registry,
                                lambda: stack.effective().conversation.max_preferences,
                            )
                            if perception.registry is not None
                            else None
                        ),
                        bus=bus,
                    )
                    cleanup.callback(conv.close)
                    conv.start()
                    if bargein is not None:  # stop the speech first, then the stream, then listen
                        bargein.bind(
                            output.stop, conv.interrupt, listener.resume, conv.mark_interrupting
                        )
            if listener is not None and voice_close is not None:
                cleanup.callback(voice_close)
                cleanup.callback(listener.stop)  # before the hub stops feeding its queue
                listener.start()
            cleanup.callback(light_reader.stop)  # stopped before the camera thread
            cleanup.callback(self._forget_circadian)
            self.start_webui(stack, cfg, manifest, cleanup)
            for world_source in plugin.sources:
                if hasattr(world_source, "speaking"):  # no turning towards its own voice
                    world_source.speaking = speech.speaking
                cleanup.callback(world_source.stop)
                world_source.start(publish)
            while not stop_event.is_set():
                now = time.monotonic()
                circ.tick(now)  # before the affect tick: its events and baseline apply to it
                state = runner.tick(now)
                minor = presence.apply(now)
                attending = attention_tick(now, world, selector, gaze, loop)
                if self.conversation is not None and selector.engaged:
                    self.conversation.note_presence(
                        now, conv_speaker()
                    )  # keeps the session alive (SPEC 8)
                live = stack.effective()
                self.circadian = circ.model if live.circadian.enabled else None
                if (
                    not light_reader.running
                    and live.circadian.enabled
                    and live.circadian.light_source == "camera"
                ):
                    light_reader.start()  # lazily: never for a clock-only or disabled rhythm
                if live.persona is not fx_for:  # recompute only after a reload
                    fx_for, fx = live.persona, persona_effects(live.persona)
                register = tracker.tick(now, live.register, fx, minor_present=minor)
                expression.set_register(register, live.safety.minor_motion_factor if minor else 1.0)
                if state is not None:
                    expression.update(state, register, now)
                perception.enrollment.expire()
                if self.autoenroll is not None:
                    self.autoenroll.tick()  # retention does not wait for a sighting
                if perception.pipeline is not None:
                    perception.pipeline.governor.set_scale(
                        CIRCADIAN, circadian_vision_scale(live, self.circadian)
                    )
                if scan is not None:
                    ctx = scan_context(
                        now,
                        live,
                        selector,
                        effects=fx,
                        hour=clock.hour(),
                        circadian=self.circadian,
                        minor_present=minor,
                    )
                    scan_tick(ctx, scan, loop, live)
                state_fx = (
                    circ.model.effects() if self.circadian else live.circadian.states["awake"]
                )
                idle_t += (now - idle_last) / state_fx.idle_period_scale
                idle_last = now
                feed_idle(loop, live, now, attending, state_fx.idle_motion_scale, idle_t)
                if circ.motion is not None:
                    circ.motion.set_minor_factor(live.safety.minor_motion_factor if minor else 1.0)
                    circ.motion.set_register(
                        register.params.motion_amplitude, live.expression.max_amplitude
                    )
                posture = circ.commands(now, attending)
                if not any(c.source == REST_SOURCE for c in posture):
                    loop.clear(REST_SOURCE)
                for cmd in posture:
                    loop.feed(cmd)
                for cmd in responder.commands(now):
                    loop.feed(cmd)
                for cmd in expression.commands(now):
                    loop.feed(cmd)
                stop_event.wait(0.05)


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
