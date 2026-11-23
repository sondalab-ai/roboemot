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
from datetime import UTC, datetime, tzinfo
from typing import Any

from roboemot import paths, timezones
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationWorker
from roboemot.conversation.phase import SPEAKING, THINKING
from roboemot.conversation.preferences import RegistryPreferences
from roboemot.conversation.sources import profile_source, scene_source
from roboemot.health.speech_cue import Delivery, SpeechCue
from roboemot.media.player import MediaPlayer
from roboemot.persona import apply_to_register, effects
from roboemot.realtime.input import RealtimeInput
from roboemot.realtime.route import RealtimeRoute
from roboemot.skills.bridge import SkillBridge
from roboemot.skills.host import http_client, tool_host
from roboemot.skills.plugins import (
    build_skill_registry,
    load_plugin_skill_sets,
    warn_hidden_core_skills,
)
from roboemot.skills.registry import SkillRegistry
from roboemot.soothe import SootheRunner
from roboemot.tools.announce import Announcer
from roboemot.tools.scheduler import Scheduler, SchedulerRunner
from roboemot.voice.clip_cache import ClipCache, clip_folder
from roboemot.voice.cues import Cues, CueSink
from roboemot.voice.factory import build_speech
from roboemot.voice.listener import (
    AUDIO_DROPPED,
    AUDIO_NOT_HEARD,
    AUDIO_NOT_HEARD_SPOKEN,
    AUDIO_SENT,
)
from roboemot.voice.speech import MediaSink
from roboemot.voice.tts import NO_VOICE

from .assembly import Pipeline
from .clips import VoiceClips
from .expressor import Expressor
from .host import RuntimeHost
from .radio_stop import stop_and_nod
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
        return cues is not None and cues.play("offline", "offline")

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
    the listening cue ends when the robot's speech plays (not a raw sound). The listening blip
    plays only in a quiet moment: nobody's speech in progress and no answer in progress (a turn
    in flight or the robot speaking: the gaps between its sentences included). A one-shot cue's
    sound does not play while someone's speech is in progress or starting (its gesture shows)."""

    def quiet() -> bool:
        listener, phase = p.listener, host.phase
        if listener is not None and listener.hearing():
            return False
        return phase is None or phase.current()[0] not in (THINKING, SPEAKING)

    return Cues(
        p.cfg.voice.cues,
        gesture=p.expression.gesture,
        sounds=p.cue_sounds,
        output=lambda: host.speech_output,
        volume_db=lambda: effects_now(host, stack, p).volume_db,
        muted=lambda: p.sleep.asleep or p.sleep.paused,
        speaking=lambda now: output.timeline.sentence_at(now) is not None,
        quiet=quiet,
        hearing=lambda: (listener := p.listener) is not None and listener.voice_active(),
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
    cues.events = host.events
    p.responder.cue = offline_cue(host, stack, p)
    start_media(host, stack, output, listener, cleanup, echo_cancelled=p.echo_cancelled)
    if listener is None:
        return
    cut_own = own_sound_cut(output, soothe)
    listener.bind_own_sound(cut_own)
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
    if realtime is not None:
        realtime.events = host.events
        realtime.prompts = host.prompts  # each client it builds taps its frames (Live tab)
    start_scheduler(host, stack, p, cleanup)  # stops after the conversation that sets items
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
        name_source=host.name_source,  # one alias per name with traits and route checks
        skills=skill_registry(stack, p, registry, host),
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
        emit=p.runner.on_event,  # the tone tool's speech.tone events, as the fast path's
    )
    conv.events = host.events
    if realtime is not None:  # closed after the conversation (no turn uses it any more)
        cleanup.callback(realtime.close)
        bind_realtime_input(host, realtime, listener, cleanup, p.sleep, cues)
    cleanup.callback(conv.close)
    conv.start()
    start_announcer(host, stack, p, cleanup)
    start_expressor(host, stack, p, cleanup)
    start_voice_clips(host, stack, cleanup)
    if realtime is not None:
        realtime.start()
    if bargein is not None:
        bind_barge_in(bargein, output, conv, listener, cut_own)


def start_media(
    host: RuntimeHost,
    stack: ConfigStack,
    output: Any,
    listener: Any,
    cleanup: ExitStack,
    *,
    echo_cancelled: bool = False,
) -> MediaPlayer | None:
    """The media player the radio plays through (SPEC 8, Radio playback; SPEC 11, Media
    player), only when the speaker can play media outside the speaking window (``MediaSink``):
    never through the speech path, which would make the listener deaf to the wake word. It
    listens for a nap and ``robot.asleep`` on the bus and for config changes (radio turned off,
    station removed), closes before the speech output, and
    tells the listener when the music can be heard (``audible_at``: the lyrics gate and
    ``tools.radio.dance``), lets it record a turn's hold (``hold_from``) and hands it the
    pieces that played (``on_piece``: the dance's audio behind a hardware echo canceller), and
    gives it each utterance's start and end (``duck_from``, ``release_from``: the duck over
    speech, only with the manifest's ``echo_cancelled``), and lets it stop the station by voice
    without the name (``stop_and_nod``). Sets ``host.media``; None (and no radio) otherwise."""
    if not isinstance(output.sink, MediaSink):
        return None
    player = host.media = MediaPlayer(output, host, stack.effective, echo_cancelled=echo_cancelled)
    cleanup.callback(player.close)  # before the output goes away (callbacks run last in first)
    cleanup.callback(stack.subscribe(player.on_config))  # radio off or station gone: stop
    player.start()
    if listener is not None:
        listener.media_audible = player.audible_at
        listener.media_hold = player.hold_from
        listener.media_set = player.playing
        listener.media_stop = lambda reason: stop_and_nod(host, player, reason)
        listener.media_speech_start = player.duck_from
        listener.media_speech_end = player.release_from
        player.on_piece = listener.hear_media
    return player


def skill_registry(
    stack: ConfigStack, p: Pipeline, registry: Any, runtime: RuntimeHost
) -> SkillRegistry:
    """The conversation's skills (SPEC 11): the default skills over the robot's pipeline plus
    every ``roboemot.skills`` plugin, each plugin loaded once. The tools' host gets the run's
    parts (``nap``: ``runtime.request_nap``, SPEC 7A; ``scheduler``: ``runtime.scheduler``,
    SPEC 11; ``http``: ``http_client``, a fresh client per call, SPEC 11, 12; ``media``:
    ``runtime.media``, SPEC 11). Warns once when the merged ``conversation.llm_skills`` hides one
    of core's own voice tools."""
    host = tool_host(stack.effective, p.bus)
    host.nap = runtime.request_nap
    host.scheduler = runtime.scheduler
    host.http = http_client
    host.media = runtime.media
    plugins = load_plugin_skill_sets(host)
    warn_hidden_core_skills(plugins, stack.effective().conversation.llm_skills)
    env = SkillBridge(p.expression, p.selector, p.world, registry, stack.effective).env(
        p.runner.engine
    )
    return build_skill_registry(env, host, plugins)


def start_scheduler(
    host: RuntimeHost, stack: ConfigStack, p: Pipeline, cleanup: ExitStack
) -> Scheduler:
    """The run's timers and reminders (SPEC 11, Voice tools): the store in the data folder,
    loaded with the restart grace, and its runner publishing ``tools.due``, stopped by
    ``cleanup``. The runner is not started here: ``start_announcer`` starts it once the
    announcer has attached, so no tick marks an item due at start fired unannounced (review C1);
    ``start_scheduler_runner`` starts it at the end of startup when no announcer was built.
    Recurrence uses ``circadian.timezone`` (this machine's zone when null), read live. Sets
    ``host.scheduler`` and ``host.scheduler_runner``."""

    def now() -> datetime:
        return datetime.now(UTC)

    def zone() -> tzinfo:
        return timezones.zone_or_local(stack.effective().circadian.timezone)

    scheduler = host.scheduler = Scheduler(
        paths.schedule_path(p.cfg.data_dir),
        lambda: stack.effective().tools.scheduler,
        now,
        zone,
    )
    scheduler.load(now())
    runner = host.scheduler_runner = SchedulerRunner(scheduler, p.bus, scheduler.cfg)
    cleanup.callback(runner.close)  # registered now: it stops after the conversation
    return scheduler


def start_scheduler_runner(host: RuntimeHost) -> None:
    """Start the scheduler runner when the run has one and it is not running yet: after the
    announcer attached, or at the end of startup with none (each due item is then logged and
    marked fired, SPEC 11, Due items)."""
    runner = host.scheduler_runner
    if runner is not None and not runner.alive:
        runner.start()


def start_announcer(host: RuntimeHost, stack: ConfigStack, p: Pipeline, cleanup: ExitStack) -> None:
    """The announcer of due timers and reminders (SPEC 11, Voice tools; SPEC 8, robot-initiated
    turn), once the conversation runs: it attaches to the scheduler runner (which then stops
    marking items fired itself), then the runner starts, and it is closed before the
    conversation. Sets ``host.announcer``; nothing without a scheduler or a conversation."""
    if host.scheduler is None or host.conversation is None:
        return
    announcer = host.announcer = Announcer(
        host.scheduler,
        host.conversation,
        host,
        stack.effective,
        asleep=lambda: p.sleep.asleep,
    )
    cleanup.callback(announcer.close)  # before the conversation it hands turns to
    announcer.start()
    start_scheduler_runner(host)  # attached first: its first tick reaches the announcer


def start_expressor(host: RuntimeHost, stack: ConfigStack, p: Pipeline, cleanup: ExitStack) -> None:
    """The robot's own reactions (SPEC 8, Robot expression), once the conversation runs. A
    route can word a line while the realtime session is up or the chat routes are online; the
    platform's sleep drops them. The listener and the conversation offer the "did not catch it"
    reaction (``offer_not_heard``). Closed before the conversation it hands turns to. Sets
    ``host.expressor``; nothing without a conversation."""
    conv = host.conversation
    if conv is None:
        return
    rt = host.realtime_route
    expressor = host.expressor = Expressor(
        conv,
        host,
        stack.effective,
        online=lambda: (rt is not None and rt.up()) or not p.hub.router.offline,
        asleep=lambda: p.sleep.asleep,
        registry=p.perception.registry,  # a returning person's name (SPEC 6, 8)
    )
    expressor.events = host.events
    cleanup.callback(expressor.close)  # before the conversation it hands turns to
    expressor.start()
    conv.on_not_heard = expressor.offer_not_heard
    if p.listener is not None:
        p.listener.on_not_heard = expressor.offer_not_heard


def start_voice_clips(host: RuntimeHost, stack: ConfigStack, cleanup: ExitStack) -> None:
    """The robot's status lines in its cloud voice (SPEC 8, Lost reply retry): recorded while
    the conversation is idle, played while the cloud does not answer. Closed before the
    conversation it records through. Sets ``host.clips``; nothing without a conversation and a
    realtime route."""
    conv = host.conversation
    if conv is None or host.realtime_route is None:
        return
    cache = ClipCache(clip_folder(stack.effective()))
    clips = host.clips = VoiceClips(conv, host, stack.effective, cache, time.monotonic)
    conv.play_clip = clips.play  # a lost reply's retry announces itself, or apologises
    cleanup.callback(clips.close)
    clips.start()


def own_sound_cut(output: Any, soothe: SootheRunner) -> Callable[[], bool]:
    """Someone speaks while only the robot's own sounds that are not speech play (a
    self-soothing hum, a cue sound): cut them (``SpeechOutput.cut_raw``: checked and cut under
    the output's lock, so robot speech queued meanwhile is kept), count the hum's cut on its
    runner (``cut_by_speech``) and say so (SPEC 7B, 8)."""

    def cut() -> bool:
        if not output.cut_raw():
            return False
        soothe.cut_by_speech(time.monotonic())
        return True

    return cut


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
    it and shows ``not_heard``, its gesture alone when the robot asks again in its own words
    (``AUDIO_NOT_HEARD_SPOKEN``, SPEC 8, Robot expression); a barge-in's segment the server
    never confirmed (dropped, S3) shows ``not_heard`` (no listening cue began for it)."""

    def on_audio(event: str, key: float) -> None:
        if event == AUDIO_SENT:
            cues.listening_begin(key)
            return
        if event == AUDIO_DROPPED:  # no listening cue began for it (S3)
            cues.play("not_heard", "playback_unconfirmed")
            return
        cues.listening_end(key)
        if event == AUDIO_NOT_HEARD:
            cues.play("not_heard", "no_transcript")
        elif event == AUDIO_NOT_HEARD_SPOKEN:  # the robot asks in its own words instead
            cues.play("not_heard", "no_transcript", sound=False)

    return on_audio


def bind_barge_in(
    bargein: Any,
    output: Any,
    conv: Any,
    listener: Any,
    cut_own: Callable[[], bool] | None = None,
) -> None:
    """Connect barge-in (SPEC 8 stage 8): its candidate frames duck the speech output's voice,
    and a cut-off stops the speech first, then the stream, then the listener restarts on the
    user's audio. Over the robot's own sounds alone, ``cut_own`` cuts them instead. With server
    turns (S2) the cut-off point streams the speech over the robot's voice instead, and the
    server's ``speech_started`` for that stream (the realtime input's ``on_confirm``) stops it."""
    output.duck = bargein.duck
    audio_in = listener.realtime_input
    route = audio_in.route if audio_in is not None else None  # each stop by mode (SPEC 13)
    bargein.bind(
        output.stop,
        conv.interrupt,
        listener.resume,
        conv.mark_interrupting,
        cut_own=cut_own,
        playback=listener.stream_playback,
        end_playback=listener.end_playback,
        server_mode=listener.server_turns,
        playback_job=listener.playback_job,
        on_stop=route.note_barge_in if route is not None else None,
    )
    if audio_in is not None:
        audio_in.on_confirm = bargein.confirm


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
    if host.prompts is not None:  # last too: a request sent during shutdown is wiped
        cleanup.callback(host.prompts.clear)
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
    start_scheduler_runner(host)  # no announcer was built: the log-only fallback
    if p.listener is not None:  # the face rule over the radio needs that reading (SPEC 8)
        p.listener.direction_over_media = p.echo_cancelled
    for world_source in p.plugin.sources:
        world_source.speaking = p.speech.sounding  # no turning towards its own sound
        # the radio does not reach a hardware echo canceller's microphone: the sound direction
        # is read over it for the listener's voice-at-face check alone (SPEC 8, Radio playback)
        world_source.direction_speaking = (
            p.speech.sounding_except_media if p.echo_cancelled else None
        )
        cleanup.callback(world_source.stop)
        world_source.start(p.publish)
    if stack.effective().runtime.gc_freeze:  # SPEC 8 stage 8: no full collection walks the models
        gc.collect()
        gc.freeze()
        cleanup.callback(gc.unfreeze)
