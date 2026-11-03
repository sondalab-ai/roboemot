"""SPEC 5A, 10: alone stillness. No face and no sound for world.scan.alone_after_s: no scans."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from roboemot.bus import Bus
from roboemot.config.schema import WorldScanConfig
from roboemot.conversation import ConversationPhase
from roboemot.runtime.ticks import in_conversation
from roboemot.world import AloneWatch

AFTER = 120.0


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def rig(after: float = AFTER):
    clock, bus, events = Clock(), Bus(), []
    watch = AloneWatch(WorldScanConfig(alone_after_s=after), bus, clock)
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
    r.watch.apply_config(WorldScanConfig(alone_after_s=20.0))
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
