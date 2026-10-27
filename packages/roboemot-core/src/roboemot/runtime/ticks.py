"""Per-tick steps of the runtime's main loop (SPEC 5, 5A, 7A, 10): idle breathing, attention,
scanning. Each step is a function of its inputs so it can be driven without a running robot."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.attention import AttentionSelector, GazeController
from roboemot.circadian import CircadianModel, in_window
from roboemot.conversation.phase import SPEAKING, THINKING, ConversationPhase
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.persona import effects as persona_effects
from roboemot.soothe import SootheRunner
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.speech import SpeechOutput
from roboemot.world import ScanContext, ScanController, WorldModel, camera_in_view
from roboemot.world.scan import SOURCE as SCAN_SOURCE

log = logging.getLogger(__name__)
ADDITIVE_IDLE = "idle-additive"


def idle_command(
    cfg: Any, t: float, scale: float = 1.0, phase_t: float | None = None, *, dof: str
) -> Command:
    """Breathing on the gaze pitch dof ``dof`` (SPEC 10, Idle; parameters from motion.idle,
    period scaled by the persona's energy, SPEC 11A). ``scale`` is the circadian state's idle
    motion scale (SPEC 7A); ``phase_t`` is the breathing phase clock (default ``t``): the caller
    integrates it at the state's slower pace so the head does not jump when the period changes."""
    idle = cfg.motion.idle
    period = idle.breathing_period_s * persona_effects(cfg.persona).idle_period_x
    phase = t if phase_t is None else phase_t
    pitch = scale * idle.breathing_amplitude * math.sin(2 * math.pi * phase / period)
    return Command("idle", Priority.IDLE, {dof: pitch}, t)


def feed_idle(
    loop: MotionLoop,
    cfg: Any,
    now: float,
    attending: bool,
    scale: float = 1.0,
    phase_t: float | None = None,
    *,
    dof: str,
) -> None:
    """Feed idle breathing. While attention owns the head it rides on top as an additive layer
    (additive commands stack on any winner of idle priority or lower precedence); otherwise it
    is the plain idle command. A ``scale`` of 0 (circadian resting) feeds none: the rest posture
    replaces it."""
    if scale <= 0:
        loop.clear("idle")
        loop.clear(ADDITIVE_IDLE)
        return
    cmd = idle_command(cfg, now, scale, phase_t, dof=dof)
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
            loop.pose(),
            world.frame,
            math.radians(wc.camera_fov_h_deg),
            math.radians(wc.camera_fov_v_deg),
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
) -> ScanContext:
    """What the scan policy needs this tick: engagement from attention, the circadian state and
    quiet hours (SPEC 7A) from the local ``hour``, the persona's energy effects (``effects``,
    computed from the persona when the caller has not cached them)."""
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


def conversation_vision_scale(
    cfg: Any, phase: ConversationPhase | None, other_scale: float = 1.0
) -> float:
    """The vision rate multiplier of the conversation (SPEC 5, 8, 13): ``vision.rates.
    conversation_scale`` while the robot thinks or speaks, so the TTS keeps its CPU, and
    ``listening_scale`` while an utterance is in progress, so the listener keeps up (the lower of
    the two when both hold); 1 when idle or with no conversation. ``other_scale`` is the product
    of the governor's other scales (circadian, load): the listening scale is raised so that the
    face detection period at the resulting rate leaves ``listening_face_detections`` detections
    within the age the voice-at-face check accepts, and is 1 when even the full rate cannot."""
    if phase is None:
        return 1.0
    name, listening = phase.current()
    rates = cfg.vision.rates
    talking = name in (THINKING, SPEAKING)
    if not listening:
        return rates.conversation_scale if talking else 1.0
    scale = (
        min(rates.listening_scale, rates.conversation_scale) if talking else rates.listening_scale
    )
    age = cfg.voice.direction.face_max_age_s
    age = cfg.attention.stale_s if age is None else age
    full = rates.detection_hz * other_scale * age  # detections within the age at scale 1
    if full <= 0:
        return 1.0
    return min(1.0, max(scale, rates.listening_face_detections / full))


@dataclass(frozen=True)
class SootheContext:
    """What the idle test of the self-soothing sound needs this tick (SPEC 7B)."""

    in_session: bool  # a conversation session is open
    voice_recent: bool  # someone spoke: a speaker hint or an utterance within the quiet window
    other_queued: int  # speech jobs queued besides our own sound
    speaking: bool  # the speaker is in its speaking window (our own sound included)
    ours: bool  # our sound is queued or playing, or ended within the echo tail


def soothe_context(
    now: float,
    *,
    selector: AttentionSelector,
    speech: SpeechActivity,
    output: SpeechOutput,
    soothe: SootheRunner,
    in_session: bool,
    last_voice: float,
    echo_tail_s: float,
) -> SootheContext:
    """Gather the idle inputs: only interaction counts, a person who is merely in view does not
    (``selector.target``, attention, presence and scans are deliberately not read). A voice is
    recent when the selector holds an active speaker hint or an utterance was heard less than
    ``soothe.speech_quiet_s`` ago; ``last_voice`` is that utterance's time."""
    return SootheContext(
        in_session=in_session,
        voice_recent=selector.speaker_active(now) or now - last_voice < soothe.cfg.speech_quiet_s,
        other_queued=output.waiting(exclude=soothe.job),
        speaking=speech.speaking(now),
        ours=soothe.sounding(now, echo_tail_s),
    )


def soothe_idle(ctx: SootheContext) -> bool:
    """Whether nobody is interacting with the robot (SPEC 7B): no conversation session, no recent
    voice, no other queued sentence and no audio but the sound's own. The sound opens the
    speaker's speaking window and holds a place in the speech queue; both are ignored while it
    plays and for the echo tail after it, so it never cancels itself."""
    if ctx.in_session or ctx.voice_recent or ctx.other_queued > 0:
        return False
    return ctx.ours or not ctx.speaking


def soothe_tick(
    soothe: SootheRunner,
    now: float,
    context: Callable[[], SootheContext],
    circadian_state: str,
    quiet: bool,
) -> None:
    """One step of the self-soothing runner, run after ``scan_tick``. A failure (type and message
    only, never audio) is logged, the sound is cut and the runtime keeps running."""
    try:
        soothe.tick(now, soothe_idle(context()), circadian_state, quiet)
    except Exception as exc:
        log.warning("soothe: %s: %s", type(exc).__name__, exc)
        try:
            soothe.close()
        except Exception as close_exc:
            log.warning("soothe: close: %s", type(close_exc).__name__)
