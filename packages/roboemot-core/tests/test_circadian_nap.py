"""SPEC 7A, Day nap: with no confirmed face and no addressed speech for circadian.nap_after_s
outside quiet hours, the robot rests; a confirmed face or addressed speech wakes it for
circadian.wake_hold_s. A single detection blip or a television never starts or ends a nap, a stale
idle clock never fires one, and the resting reason says why the robot rests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from roboemot.bus import Bus
from roboemot.circadian import CircadianRunner
from roboemot.circadian.replay import RecordingSink, ReplayClock
from roboemot.config import ConfigStack

STEP = 10.0


class FakeClock(ReplayClock):
    def advance(self, dt: float) -> None:
        self.t += dt
        self._hour = (self._hour + dt / 3600) % 24


def config(**circadian):
    return ConfigStack(app={"circadian": circadian}).effective()


def rig(hour: float = 10.0, *, conversing=None, engaged=None, **circadian):
    cfg = config(**circadian)
    clock, bus, events = FakeClock(), Bus(), []
    clock.set(0.0, hour)
    kw = {}
    if conversing is not None:
        kw["conversing"] = conversing
    if engaged is not None:
        kw["engaged"] = engaged
    r = CircadianRunner(
        cfg, bus, RecordingSink(), clock=clock, body_yaw=None, affect_intensity=lambda: 0.0, **kw
    )
    r.events = lambda kind, reason, **f: events.append((kind, reason))
    r.tick(clock.t)
    return SimpleNamespace(r=r, clock=clock, bus=bus, events=events, cfg=cfg)


def run(x, seconds: float, step: float = STEP) -> list[str]:
    """Tick for ``seconds``; the states seen, one per tick."""
    seen = []
    end = x.clock.t + seconds
    while x.clock.t < end:
        x.clock.advance(step)
        x.r.tick(x.clock.t)
        seen.append(x.r.model.state)
    return seen


def face(x, stamp: float | None = None) -> None:
    x.bus.publish("percept.face", object(), x.clock.t if stamp is None else stamp)


def napping(x) -> SimpleNamespace:
    """Run until the nap starts (nap_after_s and a few ticks)."""
    run(x, x.cfg.circadian.nap_after_s + 3 * STEP)
    assert x.r.model.state == "resting" and x.r.model.reason == "nap"
    return x


def test_it_naps_after_nap_after_s_without_a_face_or_speech():
    x = rig()
    after = x.cfg.circadian.nap_after_s
    assert after == 1800.0
    assert set(run(x, after - 2 * STEP)) == {"awake"}  # never at start-up: counts from the 1st tick
    run(x, 3 * STEP)
    assert x.r.model.state == "resting" and x.r.model.reason == "nap"
    assert x.events == [("nap", "start")]
    snap = x.r.snapshot()
    assert snap["reason"] == "nap" and snap["nap"]["napping"] is True
    assert snap["nap"]["after_s"] == after and snap["nap"]["naps"] == 1
    assert x.r.last.reason == "nap"  # the circadian.state payload carries it


def test_quiet_hours_rest_with_their_own_reason_and_no_nap():
    x = rig(hour=23.0)
    run(x, 3600)
    assert x.r.model.state == "resting" and x.r.model.reason == "quiet_hours"
    assert x.r.snapshot()["nap"]["napping"] is False and x.events == []


def test_the_night_does_not_count_towards_a_morning_nap():
    x = rig(hour=6.5)
    run(x, 0.5 * 3600 + STEP)  # quiet hours end at 07:00
    assert x.r.model.reason != "nap"
    run(x, x.cfg.circadian.nap_after_s - 3 * STEP)
    assert x.r.model.reason != "nap"  # the clock started at 07:00, not at 06:30
    run(x, 4 * STEP)
    assert x.r.model.state == "resting" and x.r.model.reason == "nap"


def test_a_nap_ends_when_quiet_hours_begin():
    x = napping(rig(hour=21.5))
    run(x, 3600)  # past 22:30
    assert x.r.model.state == "resting" and x.r.model.reason == "quiet_hours"
    assert x.r.snapshot()["nap"]["napping"] is False


def test_a_confirmed_face_wakes_the_nap_for_wake_hold_s():
    x = napping(rig())
    face(x, stamp=1.0)
    run(x, STEP)
    assert x.r.model.reason == "nap"  # one message is a blip
    face(x, stamp=2.0)
    run(x, STEP)
    assert x.r.model.state == "awake" and x.r.model.reason is None
    assert x.events == [("nap", "start"), ("nap", "face")]
    held = x.r.model.hold_remaining()
    assert held is not None and held[0] == "awake"
    assert held[1] > x.cfg.circadian.wake_hold_s - 2 * STEP


def test_two_messages_of_the_same_frame_are_one_sighting():
    x = napping(rig())
    face(x, stamp=5.0)
    x.bus.publish("person.recognized", object(), 5.0)
    run(x, STEP)
    assert x.r.model.reason == "nap"


def test_faces_further_apart_than_the_presence_window_never_wake_it():
    x = napping(rig())
    window = x.cfg.circadian.presence_window_s
    seen = []
    for i in range(10):
        face(x, stamp=float(i))
        seen += run(x, window + 2 * STEP)
    assert set(seen) == {"resting"} and x.r.model.reason == "nap"  # no flap
    assert x.events == [("nap", "start")]


def say(x, **payload) -> None:
    x.bus.publish("percept.utterance", SimpleNamespace(**payload), x.clock.t)


def test_addressed_speech_wakes_it():
    x = napping(rig())
    say(x, addressed=True)
    run(x, STEP)
    assert x.r.model.state == "awake"
    assert x.events[-1] == ("nap", "speech")


def test_television_speech_neither_holds_a_nap_off_nor_wakes_it():
    """Owner decision 2026-11-03: no wake word, no face in view: not addressed, not activity."""
    x = rig()
    after = x.cfg.circadian.nap_after_s
    for _ in range(int(after // 60) + 2):
        say(x, addressed=False)
        run(x, 60)
    assert x.r.model.reason == "nap"
    for _ in range(30):
        say(x, addressed=False)
        assert set(run(x, 60)) == {"resting"}
    assert x.events == [("nap", "start")]


def test_after_the_wake_hold_it_naps_again_only_after_nap_after_s():
    x = napping(rig())
    say(x, addressed=True)
    run(x, STEP)
    c = x.cfg.circadian
    assert set(run(x, c.wake_hold_s + c.nap_after_s - 3 * STEP)) == {"awake"}  # clock: hold end
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap"


def test_a_conversation_keeps_it_awake_and_wakes_it():
    talking = [True]
    x = rig(conversing=lambda: talking[0])
    assert set(run(x, 3 * x.cfg.circadian.nap_after_s)) == {"awake"}
    talking[0] = False
    napping(x)
    talking[0] = True
    run(x, STEP)
    assert x.r.model.state == "awake" and x.events[-1] == ("nap", "speech")


def test_attention_on_a_person_keeps_it_awake():
    x = rig(engaged=lambda: True)
    assert "resting" not in run(x, 2 * x.cfg.circadian.nap_after_s)


def test_a_face_blip_delays_the_start_while_presence_keeps_awake():
    x = rig()
    run(x, x.cfg.circadian.nap_after_s - STEP)
    face(x)  # one sighting: not activity, but presence for presence_window_s
    run(x, x.cfg.circadian.presence_window_s - STEP)
    assert x.r.model.state == "awake"
    run(x, 3 * STEP)
    assert x.r.model.reason == "nap"
    y = rig(presence_keeps_awake=False)
    run(y, y.cfg.circadian.nap_after_s - STEP)
    face(y)
    run(y, 3 * STEP)
    assert y.r.model.reason == "nap"


def test_an_app_hold_prevents_the_nap():
    x = rig()
    x.r.hold("awake", 60)
    assert set(run(x, 3000)) == {"awake"}


def test_null_turns_naps_off():
    x = rig(nap_after_s=None)
    assert set(run(x, 4 * 3600)) == {"awake"}
    assert x.r.snapshot()["nap"]["after_s"] is None


def test_turning_the_rhythm_off_ends_the_nap():
    x = napping(rig())
    off = config(enabled=False)
    x.r.apply_config(off)
    assert x.r.model.state == "awake" and x.r.snapshot()["nap"]["napping"] is False


def test_turning_naps_off_ends_the_nap_in_progress():
    x = napping(rig())
    x.r.apply_config(config(nap_after_s=None))
    run(x, STEP)
    assert x.r.model.state != "resting" and x.r.snapshot()["nap"]["napping"] is False


def percept(track: int, stamp: float):
    from roboemot.vision.types import FacePercept

    return FacePercept(track, stamp, (0.4, 0.4, 0.6, 0.6), 0.9, 0.0, 0.0, None, "anon", False)


def test_two_faces_of_one_frame_count_once_and_the_next_frame_confirms():
    x = napping(rig())
    frame = x.clock.t
    for track in (1, 2):  # the vision pipeline publishes one percept per track, frame stamp
        x.bus.publish("percept.face", percept(track, frame), frame)
    run(x, STEP)
    assert x.r.model.reason == "nap"
    x.bus.publish("percept.face", percept(1, frame + 0.2), frame + 0.2)
    run(x, STEP)
    assert x.r.model.state == "awake" and x.events[-1] == ("nap", "face")


@pytest.mark.parametrize("step", [1.0, 60.0])
def test_idle_time_before_quiet_hours_adds_up_across_midnight(step):
    """Idle from 22:15, quiet hours 22:30 to 07:00: 15 min before, 15 min after: nap at 07:15."""
    x = rig(hour=22.25)
    start_hour = None
    while start_hour is None:
        x.clock.advance(step)
        x.r.tick(x.clock.t)
        if x.r.model.reason == "nap":
            start_hour = x.clock.hour()
    tolerance_h = (step + x.cfg.circadian.update_period_s) / 3600
    assert start_hour == pytest.approx(7.25, abs=tolerance_h)
    assert x.r.nap.naps == 1


def test_no_nap_while_the_platform_sleeps_and_the_clock_restarts_on_wake():
    x = rig()
    after = x.cfg.circadian.nap_after_s
    run(x, after - 5 * STEP)
    x.bus.publish("robot.asleep", None, x.clock.t)
    assert "resting" not in run(x, 2 * after)
    x.bus.publish("robot.awake", None, x.clock.t)
    assert set(run(x, after - 3 * STEP)) == {"awake"}  # not within 10 s of the wake
    run(x, 4 * STEP)
    assert x.r.model.reason == "nap"


@pytest.mark.parametrize("call", [lambda r: r.wake(), lambda r: r.hold("awake", 10)])
def test_a_wake_or_hold_from_outside_ends_the_nap_and_restarts_the_clock(call):
    x = napping(rig())
    call(x.r)
    run(x, STEP)
    assert x.r.model.state == "awake" and x.r.snapshot()["nap"]["napping"] is False
    held = x.r.model.hold_remaining()[1]
    assert set(run(x, held + x.cfg.circadian.nap_after_s - 3 * STEP)) == {"awake"}
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap"


def test_a_resting_hold_ends_the_nap_and_no_nap_comes_right_after_it():
    x = napping(rig())
    x.r.hold("resting", 20)
    run(x, STEP)
    assert x.r.model.reason == "hold"
    run(x, 20 * 60)
    assert x.r.model.reason != "nap"
    assert "resting" not in run(x, x.cfg.circadian.nap_after_s - 3 * STEP)


def test_no_nap_right_after_a_rest_of_the_rhythm_ends():
    x = rig(hour=15.0, peak_hour=3.0, initial_sleep_pressure=1.0, nap_after_s=300.0)
    run(x, STEP)
    assert x.r.model.reason == "sleepiness"
    while x.r.model.state == "resting":  # S decays until the rhythm lets go
        run(x, STEP)
    assert x.r.model.reason is None
    assert "resting" not in run(x, 300.0 - 3 * STEP)
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap"


@pytest.mark.parametrize(
    ("first", "then"),
    [
        ({"nap_after_s": None}, {"nap_after_s": 1800.0}),  # null to a value
        ({}, {"nap_after_s": 600.0}),  # a lower value
        ({}, {"enabled": False}),  # off, then on again (below)
    ],
)
def test_a_reload_never_fires_a_nap_from_a_stale_clock(first, then):
    x = rig(**first)
    run(x, 1700)
    x.r.apply_config(config(**{**first, **then}))
    if then.get("enabled") is False:
        run(x, 1000)
        x.r.apply_config(config(**first))
    after = x.r.cfg.circadian.nap_after_s
    assert "resting" not in run(x, after - 3 * STEP)
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap"
