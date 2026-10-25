"""Builds the runtime's parts from the manifest, the platform and the bound session (SPEC 4A, 11).

``assemble`` constructs every part and wires them together without starting a thread; ``Pipeline``
is the bundle ``startup``, ``reload`` and ``loop`` work on. The parts are also published on the
host (``RuntimeHost``) as they are built. Nothing here names a robot: the robot's joints, camera,
microphone and speaker enter through the platform and the session.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from roboemot import paths
from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import RegisterTracker
from roboemot.affect.runner import AffectRunner
from roboemot.attention import AttentionSelector, GazeController
from roboemot.bus import Bus
from roboemot.circadian import CircadianRunner, IdleLightReader
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationPhase, TranscriptRing
from roboemot.conversation.sources import registry_names, speaker_source
from roboemot.health.network import NetworkMonitor
from roboemot.health.responder import OfflineResponder
from roboemot.llm.hub import RouterHub
from roboemot.llm.pseudonyms import NameSource
from roboemot.manifest import Manifest
from roboemot.motion import MotionLoop
from roboemot.persona.llm_traits import TraitCache, TraitsWorker
from roboemot.reflex import ReflexEngine
from roboemot.robot_platform import AudioHub, RobotPlatform, RobotSession, Speaker
from roboemot.vision import LockedSource
from roboemot.vision.service import Perception, open_perception
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import BargeIn, PlaybackReference
from roboemot.voice.listener import VoiceListener
from roboemot.voice.pipeline import build_listener
from roboemot.voice.robot_name import RobotName
from roboemot.voice.yes_no import YesNoGestures
from roboemot.webui.catalogue import open_catalogue
from roboemot.world import (
    Observation,
    ObserveResult,
    ScanController,
    WorldModel,
    feed_world,
    load_world_plugin,
)
from roboemot.world.autoenroll import open_autoenroll, purge_stored

from .host import RuntimeHost
from .levels import AudioLevels
from .reports import register_overrides
from .sleep import SleepGate

log = logging.getLogger(__name__)
MS_PER_S = 1000.0


@dataclass
class Pipeline:
    """The parts of one run, built by :func:`assemble` and shared by the start, reload and loop
    steps. ``cfg`` is the effective config at build time (the start-up values)."""

    cfg: Any
    expression: ExpressionRenderer
    responder: OfflineResponder
    loop: MotionLoop
    bus: Bus
    reflexes: ReflexEngine
    speech: SpeechActivity
    audio: AudioHub
    reflex_audio: bool
    reference: PlaybackReference | None
    speaker: Speaker | None
    runner: AffectRunner
    world: WorldModel
    selector: AttentionSelector
    gaze: GazeController
    plugin: Any
    scan: ScanController | None
    tracker: RegisterTracker
    clock: Any
    circ: CircadianRunner
    perception: Perception
    light_reader: IdleLightReader
    publish: Callable[[Observation, Any], None]
    network: NetworkMonitor
    hub: RouterHub
    worker: TraitsWorker
    yes_no: YesNoGestures
    names: RobotName
    listener: VoiceListener | None
    voice_close: Callable[[], None] | None
    bargein: BargeIn | None
    conv_speaker: Callable[[], str | None]
    value_cap: Callable[[], int]
    pitch_dof: str
    sleep: SleepGate


def make_yes_no(
    host: RuntimeHost, expression: ExpressionRenderer, speech: SpeechActivity, fastpath: Any
) -> YesNoGestures:
    """The nod / shake handler. The robot's last question is read from ``host.speech_output``
    when it exists (it is built after the listener), and the window comes from the config the
    reload keeps pushing in, never from a per-event config validation."""
    return YesNoGestures(
        expression.request,
        lambda: out.question_end() if (out := host.speech_output) is not None else None,
        speech.speaking,
        fastpath,
    )


def assemble(
    host: RuntimeHost,
    stack: ConfigStack,
    manifest: Manifest,
    platform: RobotPlatform,
    session: RobotSession,
) -> Pipeline:
    """Build every part of the runtime and publish it on ``host``; nothing is started."""
    gaze_map, frame, reflex_map = (
        platform.gaze_map(),
        platform.frame_spec(),
        platform.reflex_map(),
    )
    # the expression map and its channels (SPEC 11A)
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels)
    cfg = stack.effective()
    expression = host.expression = ExpressionRenderer(
        cfg.expression,
        cfg.expression_map,
        manifest.dofs,
        cfg.motion.arbiter.crossfade_ms / MS_PER_S,
        cfg.gesture_map,
        body_yaw=gaze_map.body_yaw,
    )
    responder = OfflineResponder(cfg.connectivity, expression, cfg.conversation.language)
    loop = MotionLoop(
        manifest,
        session.driver(),
        cfg,
        on_link_change=responder.on_event,
    )
    bus = host.bus = Bus()
    # Reflexes run in the motion loop's own tick (not the 20 Hz app tick below): audio level,
    # voice bearing and face size in, REFLEX-priority commands to the arbiter (SPEC 10).
    reflexes = host.reflexes = ReflexEngine(
        cfg.reflexes,
        reflex_map,
        manifest.dofs,
        manifest.constraints,
        margin_rad=math.radians(cfg.safety.constraint_margin_deg),
        bus=bus,
    )
    loop.tick_hooks.append(reflexes.on_tick)
    # The robot's own voice must not startle it or draw its gaze: the speaker feeds this
    # window (playback end + echo tail) and the reflex engine and the DoA source read it.
    speech = host.speech = SpeechActivity(cfg.voice.echo_tail_ms)
    reflexes.speaking = speech.speaking
    phase = host.phase = ConversationPhase(speech.speaking)
    # The microphone opens only for a consumer: the reflex engine with its audio on, or voice. An
    # engine turned on later runs at once, but its loud-onset part needs a restart (the reload
    # warns); turned off, it stops reading the level at once (it checks ``reflexes.enabled``).
    reflex_audio = cfg.reflexes.enabled and cfg.reflexes.audio.enabled
    # The Live tab's levels ride on the hub's reflex sink (one tuple store more, no new work on
    # the microphone thread); the transcript ring is cleared whenever the robot falls asleep.
    levels = host.levels = AudioLevels(reflexes.offer_audio_level if reflex_audio else None)
    levels.noise_floor = lambda: reflexes.noise_floor
    transcript = host.transcript = TranscriptRing(cfg.webui.live)
    bus.subscribe("robot.asleep", lambda msg: transcript.set_asleep(True))
    bus.subscribe("robot.awake", lambda msg: transcript.set_asleep(False))
    transcript.attach(stack)  # its own listener: runs before the reload fan-out is subscribed
    audio = host.audio = session.audio(cfg, levels.offer if reflex_audio else None)
    reflexes.audio_status = audio.snapshot
    reference = (
        PlaybackReference(cfg.voice.sample_rate_hz, cfg.voice.barge_in.echo_reference_s)
        if cfg.voice.enabled
        else None
    )
    levels.reference = reference
    speaker = host.speaker = session.speaker(cfg, speech, reference) if cfg.voice.enabled else None
    # The one affect engine: appraisal events from the bus and the responder feed it, the
    # app tick decays it and publishes affect.state (SPEC 7).
    runner = host.affect_runner = AffectRunner(
        cfg.affect, cfg.appraisal, bus, time.monotonic(), host.detection_period
    )
    responder.affect = runner.engine
    world = host.world = WorldModel(cfg.world, frame, bus)
    selector = host.selector = AttentionSelector(cfg.attention, bus)
    gaze = GazeController(
        cfg.attention,
        gaze_map,
        manifest.dofs,
        manifest.constraints,
        margin_rad=math.radians(cfg.safety.constraint_margin_deg),
    )
    plugin = load_world_plugin(manifest, cfg, session, loop.driver.read_state)
    scan = (
        ScanController(cfg.world.scan, plugin.planner, manifest.dofs)
        if plugin.planner is not None
        else None
    )

    def on_result(obs: Observation, result: ObserveResult) -> None:
        if scan is not None and result.unexplained_bearing is not None:
            scan.policy.note_unexplained(*result.unexplained_bearing, obs.stamp)
        if host.autoenroll is not None:  # runs on the vision thread: never let it raise
            try:
                host.autoenroll.observe(obs, result)
            except Exception as exc:
                log.warning("auto-enrollment: %s", type(exc).__name__)

    feed_world(bus, world, on_result)
    reflexes.attach(bus)
    tracker = host.register_tracker = RegisterTracker(
        bus,
        cfg.expression,
        overrides_for=lambda person_id: register_overrides(perception.registry, person_id),
        detection_period=host.detection_period,
    )
    clock = host.make_circadian_clock()
    circ = host.circadian_runner = CircadianRunner(
        cfg,
        bus,
        runner,
        clock=clock,
        dofs=manifest.dofs,
        body_yaw=gaze_map.body_yaw,
        detection_period=host.detection_period,
        engaged=lambda: selector.engaged,
    )
    expression.circadian_factor = circ.amplitude_factor
    host.circadian = circ.model if cfg.circadian.enabled else None
    # One lock for the vision pump and the idle light reader: the camera is read by one
    # thread at a time.
    source = LockedSource(host.make_frame_source(session, loop.driver.read_state))
    perception = host.perception = open_perception(
        cfg,
        bus,
        source,
        profile=manifest.robot.profile,
        camera_model=source.camera_model,
        load_stages=host.load_vision_stages,
        frame_spec=frame,
    )
    if perception.pipeline is not None:  # light from the frames vision already captures
        perception.pipeline.light_tap = circ.sampler.offer
    light_reader = host.light_reader = IdleLightReader(circ.sampler, source)
    host.enrollment = perception.enrollment
    pipeline = perception.pipeline
    appearance = pipeline.appearance_embedder if pipeline is not None else None
    host.autoenroll = open_autoenroll(
        perception.registry,
        appearance,
        cfg.world.autoenroll,
        world=world,
        language=lambda: stack.effective().conversation.language,
    )
    host.catalogue = open_catalogue(
        perception.registry,
        stack.effective,
        embedder=lambda: pipeline.appearance_embedder if pipeline is not None else None,
        world=world,
        purge_candidates=(
            host.autoenroll.purge
            if host.autoenroll
            else (lambda: purge_stored(perception.registry.store))
            if perception.registry is not None
            else None
        ),
    )

    def publish(obs: Observation, joint_state: Any) -> None:
        bus.publish("percept.entity", obs, obs.stamp, joint_state=joint_state)

    # The one router (traits, conversation, speech, route checks) reports to this monitor.
    network = NetworkMonitor(cfg.llm_routes, cfg.connectivity.network, on_change=responder.on_event)

    # Trait rating (SPEC 11A, Persona) runs on its own thread; its cache is in the data folder.
    try:
        paths.data_dir(cfg.data_dir)  # created private (0700) when missing, SPEC 12
    except OSError as exc:  # the cache then fails to write and the rating is just not kept
        log.warning("data folder: %s", type(exc).__name__)
    hub = host.router_hub = RouterHub(
        host.make_router(cfg, network), lambda c: host.make_router(c, network)
    )
    # The same registry names the conversation hides from cloud routes (SPEC 8, Pseudonyms)
    # are hidden in the persona prompt of the rating and in the route checks.
    host.name_source = NameSource(
        registry_names(
            perception.registry, lambda: stack.effective().conversation.summary_value_max_chars
        )
    )
    worker = host.traits_worker = TraitsWorker(
        stack,
        None,
        TraitCache(paths.traits_cache_path(cfg.data_dir)),
        None,
        hub=hub,
        names=host.name_source,
    )

    # Spoken yes / no (SPEC 8): a short addressed answer nods or shakes through the renderer,
    # unless the robot is speaking or just asked a question.
    yes_no = host.yes_no = make_yes_no(host, expression, speech, cfg.voice.fastpath)

    # The robot's name: voice.wake_word, else the owner's name in the daemon setup (SPEC 8).
    names = host.names = RobotName(source=loop.driver.robot_name)
    bearing = session.bearing_source(bus) if cfg.voice.enabled else None
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
            bearing,
            lambda u: host.on_utterance(u),
            reference,
            names,
            yes_no,
            on_activity=phase.set_listening,
            doa_available="doa" in manifest.capabilities(),
        )
        if bearing is not None
        else None
    )
    if bearing is not None and voice_pipeline is None:
        bearing.close()  # no listener will consume it
    listener = host.voice_listener = voice_pipeline[0] if voice_pipeline else None
    voice_close = voice_pipeline[1] if voice_pipeline else None
    bargein = host.bargein = voice_pipeline[2] if voice_pipeline else None
    if cfg.voice.enabled and listener is None:
        audio.set_voice(False)  # no consumer: the hub must not queue audio for nobody

    def value_cap() -> int:
        return stack.effective().conversation.summary_value_max_chars

    return Pipeline(
        cfg=cfg,
        expression=expression,
        responder=responder,
        loop=loop,
        bus=bus,
        reflexes=reflexes,
        speech=speech,
        audio=audio,
        reflex_audio=reflex_audio,
        reference=reference,
        speaker=speaker,
        runner=runner,
        world=world,
        selector=selector,
        gaze=gaze,
        plugin=plugin,
        scan=scan,
        tracker=tracker,
        clock=clock,
        circ=circ,
        perception=perception,
        light_reader=light_reader,
        publish=publish,
        network=network,
        hub=hub,
        worker=worker,
        yes_no=yes_no,
        names=names,
        listener=listener,
        voice_close=voice_close,
        bargein=bargein,
        conv_speaker=speaker_source(selector, world),
        value_cap=value_cap,
        pitch_dof=gaze_map.head_pitch,
        sleep=SleepGate(
            host,
            bus,
            loop,
            perception,
            light_reader,
            listener,
            gaze,
            scan,
            lambda: stack.effective().runtime.pause_when_asleep,
        ),
    )
