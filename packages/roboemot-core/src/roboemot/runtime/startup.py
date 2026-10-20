"""Starts the runtime's threads in order and registers their stop in ``cleanup`` (SPEC 4A, 8, 11).

Everything started here is stopped in reverse order whatever fails: the budgeted exit stack runs
every callback even when one raises, so the motion loop always stops and the camera thread is
quiet before the robot link goes away.
"""

from __future__ import annotations

import gc
from contextlib import ExitStack
from typing import Any

from roboemot.config import ConfigStack
from roboemot.conversation import ConversationWorker
from roboemot.conversation.preferences import RegistryPreferences
from roboemot.conversation.sources import profile_source, registry_names, scene_source
from roboemot.skills.bridge import SkillBridge
from roboemot.skills.defaults import default_skills
from roboemot.voice.factory import build_speech

from .assembly import Pipeline
from .host import RuntimeHost
from .webui import start_webui


def start_voice_output(
    host: RuntimeHost, stack: ConfigStack, p: Pipeline, cleanup: ExitStack
) -> None:
    """Start the speaker, the speech output and, with a listener, the conversation worker."""
    speaker, listener, bargein = p.speaker, p.listener, p.bargein
    assert speaker is not None
    cleanup.callback(speaker.close)  # flushes the player before the robot link goes away
    speaker.start_playing()
    # Closed before the speaker (callbacks run last in, first out).
    output = host.speech_output = build_speech(
        p.cfg, speaker, lambda: p.hub.router, p.bus, lambda: host.minor_present
    )
    cleanup.callback(output.close)
    output.start()
    if listener is None:
        return
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
        circadian=lambda: (
            p.circ.model.effects()
            if host.circadian
            else stack.effective().circadian.states["awake"]
        ),
        minor_present=lambda: host.minor_present,
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
    )
    cleanup.callback(conv.close)
    conv.start()
    if bargein is not None:  # stop the speech first, then the stream, then listen
        bargein.bind(output.stop, conv.interrupt, listener.resume, conv.mark_interrupting)


def start_all(
    host: RuntimeHost,
    stack: ConfigStack,
    manifest: Any,
    p: Pipeline,
    cleanup: ExitStack,
) -> None:
    """Start every part in the order that lets the reverse order stop them safely."""
    cleanup.callback(p.presence.close)
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
    start_webui(host, stack, p.cfg, manifest, cleanup)
    for world_source in p.plugin.sources:
        world_source.speaking = p.speech.speaking  # no turning towards its own voice
        cleanup.callback(world_source.stop)
        world_source.start(p.publish)
    if stack.effective().runtime.gc_freeze:  # SPEC 8 stage 8: no full collection walks the models
        gc.collect()
        gc.freeze()
        cleanup.callback(gc.unfreeze)
