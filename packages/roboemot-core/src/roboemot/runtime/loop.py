"""The runtime's main loop (SPEC 5, 7, 7A, 10): one tick every ``motion.app_tick_hz``."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from roboemot.circadian.motion import REST_SOURCE
from roboemot.config import ConfigStack
from roboemot.persona import effects as persona_effects
from roboemot.vision import CIRCADIAN

from .assembly import Pipeline
from .host import RuntimeHost
from .ticks import (
    attention_tick,
    circadian_vision_scale,
    feed_idle,
    scan_context,
    scan_tick,
)


@dataclass
class LoopClock:
    """The idle breathing phase clock (SPEC 7A, Drowsy): ``phase_s`` advances at the circadian
    state's slower pace so the head does not jump when the period changes."""

    last_s: float = field(default_factory=time.monotonic)
    phase_s: float = field(default_factory=time.monotonic)


def run_loop(
    host: RuntimeHost,
    stack: ConfigStack,
    p: Pipeline,
    clock: LoopClock,
    pitch_dof: str,
    stop_event: threading.Event,
) -> None:
    """Tick until ``stop_event`` is set: circadian, affect, attention, expression, scanning,
    idle breathing and the commands of every layer into the motion loop."""
    fx_for, fx = None, None
    while not stop_event.is_set():
        now = time.monotonic()
        if host.transcript is not None:
            host.transcript.trim()  # the age bound holds when nobody polls or speaks
        if host.live_video is not None:
            host.live_video.expire()  # the same for the preview's frame and JPEG
        if p.sleep.tick(now):  # asleep (SPEC 12A): only the clocks run; the work waits for the wake
            p.circ.tick(now)
            p.runner.tick(now)
            clock.last_s = now
            stop_event.wait(1.0 / stack.effective().motion.app_tick_hz)
            continue
        p.circ.tick(now)  # before the affect tick: its events and baseline apply to it
        state = p.runner.tick(now)
        minor = p.presence.apply(now)
        attending = attention_tick(now, p.world, p.selector, p.gaze, p.loop)
        if host.conversation is not None and p.selector.engaged:
            host.conversation.note_presence(
                now, p.conv_speaker()
            )  # keeps the session alive (SPEC 8)
        live = stack.effective()
        host.circadian = p.circ.model if live.circadian.enabled else None
        if (
            not p.light_reader.running
            and live.circadian.enabled
            and live.circadian.light_source == "camera"
        ):
            p.light_reader.start()  # lazily: never for a clock-only or disabled rhythm
        if live.persona is not fx_for:  # recompute only after a reload
            fx_for, fx = live.persona, persona_effects(live.persona)
        register = p.tracker.tick(now, live.register, fx, minor_present=minor)
        p.expression.set_register(register, live.safety.minor_motion_factor if minor else 1.0)
        if state is not None:
            p.expression.update(state, register, now)
        p.perception.enrollment.expire()
        if host.autoenroll is not None:
            host.autoenroll.tick()  # retention does not wait for a sighting
        if p.perception.pipeline is not None:
            p.perception.pipeline.governor.set_scale(
                CIRCADIAN, circadian_vision_scale(live, host.circadian)
            )
        if p.scan is not None:
            ctx = scan_context(
                now,
                live,
                p.selector,
                effects=fx,
                hour=p.clock.hour(),
                circadian=host.circadian,
                minor_present=minor,
            )
            scan_tick(ctx, p.scan, p.loop, live)
        state_fx = p.circ.model.effects() if host.circadian else live.circadian.states["awake"]
        clock.phase_s += (now - clock.last_s) / state_fx.idle_period_scale
        clock.last_s = now
        feed_idle(
            p.loop,
            live,
            now,
            attending,
            state_fx.idle_motion_scale,
            clock.phase_s,
            dof=pitch_dof,
        )
        if p.circ.motion is not None:
            p.circ.motion.set_minor_factor(live.safety.minor_motion_factor if minor else 1.0)
            p.circ.motion.set_register(
                register.params.motion_amplitude, live.expression.max_amplitude
            )
        posture = p.circ.commands(now, attending)
        if not any(c.source == REST_SOURCE for c in posture):
            p.loop.clear(REST_SOURCE)
        for cmd in posture:
            p.loop.feed(cmd)
        for cmd in p.responder.commands(now):
            p.loop.feed(cmd)
        for cmd in p.expression.commands(now):
            p.loop.feed(cmd)
        stop_event.wait(1.0 / live.motion.app_tick_hz)
