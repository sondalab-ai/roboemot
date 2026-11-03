"""SPEC 7A, Day nap: with no face and no speech for circadian.nap_after_s, outside quiet hours,
the robot rests; a confirmed face or speech wakes it for circadian.wake_hold_s. A single detection
blip never starts or ends a nap, and the resting reason says why the robot rests."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_speech_wakes_it_even_unaddressed():
    x = napping(rig())
    x.bus.publish("percept.utterance", SimpleNamespace(addressed=False), x.clock.t)
    run(x, STEP)
    assert x.r.model.state == "awake"
    assert x.events[-1] == ("nap", "speech")


def test_after_the_wake_it_naps_again_only_after_nap_after_s():
    x = napping(rig())
    x.bus.publish("percept.utterance", SimpleNamespace(addressed=False), x.clock.t)
    run(x, STEP)
    after = x.cfg.circadian.nap_after_s
    assert set(run(x, after - 3 * STEP)) == {"awake"}
    run(x, 4 * STEP)
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
