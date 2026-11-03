"""SPEC 5A, 10: alone stillness. No face and no sound for world.scan.alone_after_s: no scans."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationPhase
from roboemot.runtime.ticks import company, in_conversation
from roboemot.world import AloneWatch

AFTER = 120.0


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def config(after: float = AFTER):
    return ConfigStack(app={"world": {"scan": {"alone_after_s": after}}}).effective()


def rig(after: float = AFTER, period: float = math.inf):
    clock, bus, events = Clock(), Bus(), []
    watch = AloneWatch(config(after), bus, clock, detection_period=lambda: period)
    watch.events = lambda kind, reason, **f: events.append((kind, reason))
    return SimpleNamespace(clock=clock, bus=bus, watch=watch, events=events)


def run(r, seconds: float, busy: bool = False, step: float = 1.0) -> bool:
    end = r.clock.t + seconds
    while r.clock.t < end:
        r.clock.t += step
        r.watch.tick(r.clock.t, busy)
    return r.watch.alone


def test_not_alone_at_start_up_and_alone_after_the_delay():
    r = rig()
    assert r.watch.tick(r.clock.t) is False  # the delay counts from the first tick
    assert run(r, AFTER - 2) is False
    assert run(r, 2) is True
    assert r.events == [("alone", "start")]


@pytest.mark.parametrize(
    ("topic", "cause"),
    [
        ("percept.face", "face"),
        ("person.recognized", "face"),
        ("percept.utterance", "sound"),
        ("sound.loud", "sound"),
    ],
)
def test_a_face_or_a_sound_ends_it_at_once(topic, cause):
    r = rig()
    r.watch.tick(r.clock.t)
    assert run(r, AFTER + 5)
    r.bus.publish(topic, None, r.clock.t)
    assert r.watch.tick(r.clock.t) is False  # the very next tick
    assert r.events == [("alone", "start"), ("alone", cause)]


def test_busy_keeps_it_from_starting_and_ends_it():
    r = rig()
    r.watch.tick(r.clock.t)
    assert run(r, 3 * AFTER, busy=True) is False
    assert run(r, AFTER + 1) is True
    r.watch.tick(r.clock.t, busy=True)
    assert r.watch.alone is False
    assert r.events[-1] == ("alone", "sound")


def test_signs_of_company_restart_the_delay():
    r = rig()
    r.watch.tick(r.clock.t)
    run(r, AFTER - 10)
    r.bus.publish("percept.face", None, r.clock.t)
    assert run(r, AFTER - 5) is False
    assert run(r, 10) is True


def test_a_hot_reload_changes_the_delay():
    r = rig()
    r.watch.tick(r.clock.t)
    run(r, 30)
    r.watch.apply_config(config(20.0))
    assert r.watch.tick(r.clock.t) is True


def test_the_report_section():
    r = rig()
    assert r.watch.snapshot(r.clock.t)["quiet_for_s"] is None
    r.watch.tick(r.clock.t)
    run(r, AFTER + 30)
    snap = r.watch.snapshot(r.clock.t)
    assert snap == {
        "alone": True,
        "alone_for_s": 30.0,
        "quiet_for_s": AFTER + 30,
        "alone_after_s": AFTER,
        "changes": 1,
    }


def test_close_detaches_from_the_bus():
    r = rig()
    r.watch.tick(r.clock.t)
    run(r, AFTER + 1)
    r.watch.close()
    r.bus.publish("percept.face", None, r.clock.t)
    assert r.watch.tick(r.clock.t) is True


def test_the_robot_speaking_alone_is_no_conversation():
    """A self-soothing sound opens the speaking window: it must not end the stillness."""
    speaking = ConversationPhase(lambda now: True)
    assert in_conversation(speaking) is False
    listening = ConversationPhase()
    listening.set_listening(True)
    assert in_conversation(listening) is True
    thinking = ConversationPhase()
    thinking.begin_thinking()
    assert in_conversation(thinking) is True
    session = SimpleNamespace(in_session=lambda: True)
    assert in_conversation(ConversationPhase(), session) is True
    assert in_conversation(None) is False

    assert in_conversation(listening, hearing=False) is False  # the VAD: a television too
    assert in_conversation(thinking, hearing=False) is True


def selector(engaged=False, speaker=False):
    return SimpleNamespace(engaged=engaged, speaker_active=lambda now: speaker)


@pytest.mark.parametrize(
    ("sel", "phase", "conversation", "busy"),
    [
        (selector(), None, None, False),
        (selector(engaged=True), None, None, True),
        (selector(speaker=True), None, None, True),
        (selector(), "listening", None, True),
        (selector(), None, SimpleNamespace(in_session=lambda: True), True),
        (selector(), "speaking", None, False),  # the robot alone speaking (a soothe sound)
    ],
)
def test_what_reaches_the_watch_as_busy(sel, phase, conversation, busy):
    """The loop's wiring (``company``): attention, a speaker hint and a conversation."""
    p = ConversationPhase(lambda now: phase == "speaking")
    if phase == "listening":
        p.set_listening(True)
    assert company(0.0, sel, p, conversation) is busy


def test_a_slow_detector_lengthens_the_delay():
    """A still face seen once every 30 s (drowsy or resting vision) is not an empty room."""
    r = rig(after=10.0, period=30.0)
    periods = r.watch.cfg.circadian.presence_periods
    assert r.watch.after_s() == periods * 30.0
    r.watch.tick(r.clock.t)
    assert run(r, periods * 30.0 - 2) is False
    assert run(r, 2) is True


def test_waking_from_platform_sleep_restarts_the_delay():
    r = rig()
    r.watch.tick(r.clock.t)
    assert run(r, AFTER + 5)
    r.bus.publish("robot.asleep", None, r.clock.t)
    r.clock.t += 3 * AFTER  # the loop does not tick the watch while asleep
    r.bus.publish("robot.awake", None, r.clock.t)
    assert r.watch.alone is False
    assert r.watch.tick(r.clock.t) is False  # not alone on the first tick after the wake
    assert run(r, AFTER - 2) is False
    assert run(r, 2) is True


def test_with_reflexes_off_loud_talk_alone_does_not_count():
    """SPEC 5A: sound.loud comes from the reflex engine. With reflexes.enabled off it never
    comes, so people talking count only through the listener (an utterance, the VAD): a robot
    without voice goes alone while they talk."""
    from test_reflex import LOUD, Rig, cfg_with, rule

    reflex = Rig(cfg_with(rule(), enabled=False))
    clock, events = Clock(), []
    watch = AloneWatch(config(), reflex.bus, clock)
    watch.events = lambda kind, reason, **f: events.append(reason)
    watch.tick(clock.t)
    reflex.settle_audio()
    for _ in range(int(AFTER) + 1):
        reflex.engine.offer_audio_level(LOUD, reflex.t)
        reflex.tick()
        clock.t += 1.0
        watch.tick(clock.t)
    assert watch.alone and events == ["start"]
