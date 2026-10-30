"""Starts the runtime's threads in order and registers their stop in ``cleanup`` (SPEC 4A, 8, 11).

Everything started here is stopped in reverse order whatever fails: the budgeted exit stack runs
every callback even when one raises, so the motion loop always stops and the camera thread is
quiet before the robot link goes away.
"""

from __future__ import annotations

import gc
import random
import time
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from roboemot import paths
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationWorker
from roboemot.conversation.preferences import RegistryPreferences
from roboemot.conversation.sources import profile_source, registry_names, scene_source
from roboemot.health.speech_cue import Delivery, SpeechCue
from roboemot.persona import apply_to_register, effects
from roboemot.realtime.input import RealtimeInput
from roboemot.realtime.route import RealtimeRoute
from roboemot.skills.bridge import SkillBridge
from roboemot.skills.defaults import default_skills
from roboemot.soothe import SootheRunner
from roboemot.voice.cues import Cues, CueSink
from roboemot.voice.factory import build_speech
from roboemot.voice.listener import AUDIO_NOT_HEARD, AUDIO_SENT
from roboemot.voice.tts import NO_VOICE

from .assembly import Pipeline
from .host import RuntimeHost
from .webui import WebAdmin, start_webui


def offline_cue(host: RuntimeHost, stack: ConfigStack, p: Pipeline) -> SpeechCue:
    """The responder's cue, speaking on ``host.speech_output`` (SPEC 12A, Behaviour while
    offline). Installed once that output exists; the register, circadian state and affect are
    read when the phrase is said. Muted while asleep (the speaker is quiet then). Without a
    local voice the phrase the route cannot say is the non-verbal ``offline`` cue (SPEC 8)."""

    def delivery() -> Delivery:
        cfg, active = stack.effective(), p.tracker.active
        register = (
            active.params
            if active
            else apply_to_register(cfg.register.params, effects(cfg.persona))
        )
        circadian = p.circ.model.effects() if host.circadian else cfg.circadian.states["awake"]
        state = p.runner.engine.state(time.monotonic())
        return register, circadian, state.label, state.intensity

    def line(text: str, language: str) -> bool:
        conv = host.conversation
        return conv is not None and conv.say_line(text, language)

    def cue() -> bool:
        cues = host.cues
        return cues is not None and cues.play("offline")

    return SpeechCue(
        lambda: host.speech_output,
        delivery,
        lambda: p.sleep.asleep or p.sleep.paused,
        line,
        no_voice=lambda: (out := host.speech_output) is not None and out.backend.name == NO_VOICE,
        cue=cue,
    )


def build_cues(host: RuntimeHost, stack: ConfigStack, p: Pipeline, output: Any) -> Cues:
    """The non-verbal cues (SPEC 8, Non-verbal cues): the manifest's ``cue_<kind>`` gestures and
    the robot's cue sounds on ``output``, at the circadian volume, muted while asleep or paused;
    the listening cue ends when the robot's speech plays (not a raw sound)."""
    return Cues(
        p.cfg.voice.cues,
        gesture=p.expression.gesture,
        sounds=p.cue_sounds,
        output=lambda: host.speech_output,
        volume_db=lambda: effects_now(host, stack, p).volume_db,
        muted=lambda: p.sleep.asleep or p.sleep.paused,
        speaking=lambda now: output.timeline.sentence_at(now) is not None,
    )


def effects_now(host: RuntimeHost, stack: ConfigStack, p: Pipeline) -> Any:
    """The circadian state's effects now (the awake ones while the rhythm is off)."""
    return p.circ.model.effects() if host.circadian else stack.effective().circadian.states["awake"]


def start_voice_output(
    host: RuntimeHost, stack: ConfigStack, p: Pipeline, cleanup: ExitStack
) -> None:
    """Start the speaker, the speech output and, with a listener, the conversation worker."""
    speaker, listener, bargein = p.speaker, p.listener, p.bargein
    if speaker is None:
        raise RuntimeError("no speaker to start")
    cleanup.callback(speaker.close)  # flushes the player before the robot link goes away
    speaker.start_playing()
    # Closed before the speaker (callbacks run last in, first out).
    output = host.speech_output = build_speech(p.cfg, speaker, lambda: p.hub.router, p.bus)
    cleanup.callback(output.close)
    output.start()
    soothe = host.soothe_runner = SootheRunner(
        p.cfg.soothe,
        output,
        on_event=p.runner.on_event,
        rng=random.Random(),  # noqa: S311 (a pick and a jitter, not security)
        volume_db=lambda: effects_now(host, stack, p).volume_db,
    )
    cleanup.callback(soothe.close)  # before the output goes away
    cues = host.cues = build_cues(host, stack, p, output)
    p.responder.cue = offline_cue(host, stack, p)
    if listener is None:
        return
    realtime = host.realtime_route = (
        RealtimeRoute(
            stack,
            lambda: p.hub.router,
            p.network,
            p.realtime_provider,
            upload_path=paths.realtime_upload_path(p.cfg.data_dir),
        )
        if p.cfg.voice.realtime_route.enabled
        else None
    )
    # Shutdown order: listener (no new utterances), conversation (cancels the
    # stream in flight), speech output, speaker.
    registry = p.perception.registry
    conv = host.conversation = ConversationWorker(
        stack,
        p.hub,
        output,
        affect=p.runner.engine,
        responder=p.responder,
        register=lambda: p.tracker.active.params if p.tracker.active else None,
        circadian=lambda: effects_now(host, stack, p),
        profile=profile_source(p.selector, p.world, registry, p.value_cap),
        scene=scene_source(
            p.selector,
            p.world,
            registry,
            lambda: stack.effective().attention.stale_s,
            p.value_cap,
        ),
        speaker=p.conv_speaker,
        names=p.names,
        known_names=registry_names(registry, p.value_cap),
        skills=default_skills(
            SkillBridge(p.expression, p.selector, p.world, registry, stack.effective).env(
                p.runner.engine
            )
        ),
        preferences=(
            RegistryPreferences(
                registry,
                lambda: stack.effective().conversation.max_preferences,
            )
            if registry is not None
            else None
        ),
        bus=p.bus,
        phase=host.phase,
        transcript=host.transcript,
        realtime=realtime,
        cues=cues,
    )
    if realtime is not None:  # closed after the conversation (no turn uses it any more)
        cleanup.callback(realtime.close)
        bind_realtime_input(host, realtime, listener, cleanup, p.sleep, cues)
    cleanup.callback(conv.close)
    conv.start()
    if realtime is not None:
        realtime.start()
    if bargein is not None:
        bind_barge_in(bargein, output, conv, listener)


def bind_realtime_input(
    host: RuntimeHost,
    realtime: RealtimeRoute,
    listener: Any,
    cleanup: ExitStack,
    sleep: Any,
    cues: CueSink | None = None,
) -> None:
    """Addressed utterances' audio to the realtime route (SPEC 8, Realtime speech route): the
    listener hands it to a ``RealtimeInput``, which sends nothing while the robot speaks or
    sleeps (``sleep`` is the ``SleepGate``: asleep counts whatever
    ``runtime.pause_when_asleep`` says) and only through the route's send gate
    (``input: audio`` with ``privacy.realtime_audio``). Closed before the route. What becomes
    of the audio drives ``cues`` (the listening and did-not-get-it cues, SPEC 8)."""
    audio_in = RealtimeInput(
        realtime,
        speaking=lambda: host.speech.speaking(time.monotonic()),
        asleep=lambda: bool(sleep.asleep),
    )
    listener.bind_realtime(audio_in, audio_cues(cues) if cues is not None else None)
    cleanup.callback(audio_in.close)


def audio_cues(cues: CueSink) -> Callable[[str, float], None]:
    """The listener's audio events as cues, keyed by the utterance's start stamp: audio sent
    begins the listening cue; audio that did not leave ends it; audio without a transcript ends
    it and shows ``not_heard``."""

    def on_audio(event: str, key: float) -> None:
        if event == AUDIO_SENT:
            cues.listening_begin(key)
            return
        cues.listening_end(key)
        if event == AUDIO_NOT_HEARD:
            cues.play("not_heard")

    return on_audio


def bind_barge_in(bargein: Any, output: Any, conv: Any, listener: Any) -> None:
    """Connect barge-in (SPEC 8 stage 8): its candidate frames duck the speech output's voice,
    and a cut-off stops the speech first, then the stream, then the listener restarts on the
    user's audio."""
    output.duck = bargein.duck
    bargein.bind(output.stop, conv.interrupt, listener.resume, conv.mark_interrupting)


def start_all(
    host: RuntimeHost,
    stack: ConfigStack,
    manifest: Any,
    p: Pipeline,
    cleanup: ExitStack,
    admin: WebAdmin | None = None,
) -> None:
    """Start every part in the order that lets the reverse order stop them safely. ``admin`` is
    the web UI's settings editor, restored before the pipeline was built."""
    if (
        host.transcript is not None
    ):  # registered first, so it runs last: a late sentence is wiped too
        cleanup.callback(host.transcript.clear)
    cleanup.callback(p.tracker.close)
    cleanup.callback(p.runner.close)
    cleanup.callback(p.circ.close)
    cleanup.callback(p.reflexes.close)
    cleanup.callback(stack.start_watcher().set)
    p.loop.start()
    cleanup.callback(p.loop.stop)
    p.network.start()
    cleanup.callback(p.network.stop)
    cleanup.callback(p.worker.close)  # before the network monitor its router reports to
    p.worker.start()
    if host.autoenroll is not None:  # runs after perception.stop: no new sightings
        cleanup.callback(host.autoenroll.flush)
    cleanup.callback(p.perception.stop)
    p.perception.start()
    if p.reflex_audio or p.listener is not None:
        cleanup.callback(p.audio.stop)  # before the robot link goes away
        p.audio.start()  # degrades to "unavailable" in the debug report without a mic
    if p.speaker is not None:
        start_voice_output(host, stack, p, cleanup)
    if p.listener is not None and p.voice_close is not None:
        cleanup.callback(p.voice_close)
        cleanup.callback(p.listener.stop)  # before the hub stops feeding its queue
        p.listener.start()
    cleanup.callback(p.light_reader.stop)  # stopped before the camera thread
    cleanup.callback(host.forget_circadian)
    start_webui(host, stack, p.cfg, manifest, cleanup, p, admin)
    for world_source in p.plugin.sources:
        world_source.speaking = p.speech.speaking  # no turning towards its own voice
        cleanup.callback(world_source.stop)
        world_source.start(p.publish)
    if stack.effective().runtime.gc_freeze:  # SPEC 8 stage 8: no full collection walks the models
        gc.collect()
        gc.freeze()
        cleanup.callback(gc.unfreeze)
