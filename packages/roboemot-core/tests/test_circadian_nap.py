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
from roboemot.reflex.engine import SoundLoud
from roboemot.vision.scene_motion import ScenePercept

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


# ------------------------------------------------- light, motion and loud wakes (2026-11-03)

DIM, BRIGHT, DARK = 0.2, 0.6, 0.03  # normalised luma of each light level (LightConfig defaults)


def lit(x, seconds: float, luma: float) -> list[str]:
    """Tick for ``seconds`` with one light sample per tick; the states seen."""
    seen = []
    end = x.clock.t + seconds
    while x.clock.t < end:
        x.clock.advance(STEP)
        x.r.sampler.offer_stats(luma, 0.0)
        x.r.tick(x.clock.t)
        seen.append(x.r.model.state)
    return seen


def loud(x) -> None:
    x.bus.publish("sound.loud", SoundLoud(ratio=6.0, level=0.3), x.clock.t)


def motion(x) -> None:
    x.bus.publish("percept.scene_motion", ScenePercept(0.2), x.clock.t)


STIMULI = {"loud": loud, "motion": motion}


def napping_lit(x, luma: float = DIM) -> SimpleNamespace:
    lit(x, x.cfg.circadian.nap_after_s + 3 * STEP, luma)
    assert x.r.model.reason == "nap"
    return x


def past_grace(x, luma: float = DIM) -> None:
    lit(x, x.cfg.circadian.nap_light_grace_s + STEP, luma)
    assert x.r.model.reason == "nap"


def woke(x, reason: str) -> None:
    run(x, STEP)
    assert x.r.model.state == "awake" and x.r.model.reason is None
    assert x.events[-1] == ("nap", reason)
    assert x.r.snapshot()["nap"]["wakes"][reason] == 1
    held = x.r.model.hold_remaining()
    assert held is not None and held[0] == "awake"


def test_new_wake_switches_default_on():
    c = config()
    assert c.circadian.nap_wake_on_light and c.circadian.nap_wake_on_motion
    assert c.circadian.nap_wake_on_loud and c.circadian.nap_light_grace_s == 300.0


def test_light_rising_from_dim_to_bright_wakes_the_nap():
    """The robot was in dim when the owner's test failed (2026-11-03): dim to bright counts."""
    x = napping_lit(rig())
    past_grace(x)
    lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)
    assert x.r.model.state == "awake" and x.events[-1] == ("nap", "light")
    assert x.r.snapshot()["nap"]["wakes"]["light"] == 1


def test_light_rising_from_dark_to_dim_wakes_the_nap():
    x = napping_lit(rig(), DARK)
    past_grace(x, DARK)
    lit(x, x.cfg.light.confirm_s + 2 * STEP, DIM)
    assert x.r.model.state == "awake" and x.events[-1] == ("nap", "light")


def test_light_falling_does_not_wake_the_nap():
    x = napping_lit(rig(), BRIGHT)
    past_grace(x, BRIGHT)
    assert set(lit(x, x.cfg.light.confirm_s + 6 * STEP, DARK)) == {"resting"}
    assert x.r.model.reason == "nap"


def test_a_light_rise_within_the_grace_after_the_nap_start_is_the_rest_posture():
    x = rig()
    lit(x, x.cfg.circadian.nap_after_s + STEP, DIM)
    assert x.r.model.reason == "nap"
    # the rest posture shows the camera a brighter view: confirmed light.confirm_s later
    assert x.cfg.light.confirm_s + 2 * STEP < x.cfg.circadian.nap_light_grace_s
    assert set(lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)) == {"resting"}
    assert set(lit(x, x.cfg.circadian.nap_light_grace_s, BRIGHT)) == {"resting"}
    assert x.r.model.reason == "nap"  # bright is now the nap's level: no rise, no wake


@pytest.mark.parametrize("name", sorted(STIMULI))
def test_motion_or_a_loud_sound_wakes_the_nap(name):
    x = napping(rig())
    STIMULI[name](x)
    woke(x, name)


@pytest.mark.parametrize("name", ["light", "motion", "loud"])
def test_each_switch_turns_its_stimulus_off(name):
    x = napping_lit(rig(**{f"nap_wake_on_{name}": False}))
    past_grace(x)
    if name == "light":
        assert set(lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)) == {"resting"}
    else:
        STIMULI[name](x)
        run(x, STEP)
    assert x.r.model.reason == "nap"
    assert x.r.snapshot()["nap"]["wakes"][name] == 0


def test_none_of_them_ends_a_quiet_hours_rest():
    x = rig(hour=23.0)
    lit(x, 3600, DIM)
    assert x.r.model.reason == "quiet_hours"
    loud(x)
    motion(x)
    lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)
    assert x.r.model.state == "resting" and x.r.model.reason == "quiet_hours"
    assert x.r.woken_at_night == 0 and x.events == []


@pytest.mark.parametrize("name", sorted(STIMULI))
def test_a_stimulus_between_ticks_after_quiet_hours_began_does_not_wake(name):
    """Sound and frames arrive on their own threads: the nap flag is cleared only at the next
    update after 22:30, so the handler checks the quiet hours itself."""
    x = napping(rig(hour=21.5))
    while x.clock.hour() < 22.5 - STEP / 3600:
        run(x, STEP)
    assert x.r.model.reason == "nap"
    x.clock.advance(STEP)  # into the quiet hours, no update yet
    STIMULI[name](x)
    run(x, STEP)
    assert x.r.model.reason == "quiet_hours" and x.r.woken_at_night == 0
    assert x.events == [("nap", "start")]


@pytest.mark.parametrize("name", sorted(STIMULI))
def test_none_of_them_ends_a_rest_held_by_the_app(name):
    x = napping(rig())
    x.r.hold("resting", 20)
    run(x, STEP)
    STIMULI[name](x)
    run(x, STEP)
    assert x.r.model.reason == "hold"


def test_while_awake_none_of_them_restarts_the_idle_clock():
    """Not people engaging (a television's bang, a car, clouds, the robot's own moves): a busy
    room would otherwise never let the robot nap."""
    x = rig()
    after, start = x.cfg.circadian.nap_after_s, x.clock.t
    lit(x, after / 2, DARK)  # then the light rises while awake
    for _ in range(int(after / 2 // 60) - 1):
        loud(x)
        motion(x)
        assert set(lit(x, 60, BRIGHT)) == {"awake"}
    lit(x, start + after + 3 * STEP - x.clock.t, BRIGHT)
    assert x.r.model.reason == "nap"  # on time: the clock ran from the start
    assert sum(x.r.snapshot()["nap"]["wakes"].values()) == 0


def test_the_motion_watch_is_wanted_only_during_a_nap_with_its_switch():
    x = rig()
    assert x.r.wants_scene_motion() is False
    napping(x)
    assert x.r.wants_scene_motion() is True
    x.r.apply_config(config(nap_wake_on_motion=False))
    assert x.r.wants_scene_motion() is False
    x.r.apply_config(config())
    motion(x)
    run(x, STEP)
    assert x.r.wants_scene_motion() is False  # awake


def test_wakes_are_counted_per_reason():
    x = napping(rig())
    face(x, stamp=1.0)
    face(x, stamp=2.0)
    run(x, STEP)
    wakes = x.r.snapshot()["nap"]["wakes"]
    assert wakes == {"face": 1, "speech": 0, "light": 0, "motion": 0, "loud": 0}


def test_none_of_them_wakes_a_nap_while_the_platform_sleeps():
    x = napping_lit(rig())
    past_grace(x)
    x.bus.publish("robot.asleep", None, x.clock.t)
    assert x.r.wants_scene_motion() is False
    loud(x)
    motion(x)
    lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)
    assert x.r.model.state == "resting"
    assert x.r.model.hold_remaining() is None
    assert sum(x.r.snapshot()["nap"]["wakes"].values()) == 0
    assert x.events == [("nap", "start")]


def test_motion_within_the_grace_after_the_nap_start_is_ignored():
    x = rig()
    grace = x.cfg.circadian.nap_motion_grace_s
    assert grace == 10.0
    while x.r.model.reason != "nap":
        run(x, STEP)
    start = x.r.nap.started
    assert x.clock.t == start
    x.clock.advance(grace - 1.0)  # inside the grace, between ticks
    motion(x)
    run(x, STEP)
    assert x.r.model.reason == "nap"
    motion(x)  # past it
    woke(x, "motion")


@pytest.mark.parametrize(("offset", "wakes"), [(-STEP, False), (0.0, True)])
def test_the_light_grace_boundary(offset, wakes):
    """A rise first seen at an update exactly nap_light_grace_s after the start wakes the nap;
    one an update earlier does not, and is not looked at again (the level stays bright)."""
    x = rig()
    level = {"now": "dim"}
    x.r.sampler.level = lambda: level["now"]
    while x.r.model.reason != "nap":
        run(x, STEP)
    rise_at = x.r.nap.started + x.cfg.circadian.nap_light_grace_s + offset
    run(x, rise_at - STEP - x.clock.t)
    assert x.clock.t == rise_at - STEP and x.r.model.reason == "nap"
    level["now"] = "bright"
    run(x, STEP)
    assert (x.r.model.reason is None) is wakes
    run(x, 3 * STEP)
    assert (x.r.model.reason is None) is wakes  # the earlier rise is not re-evaluated


def test_scene_motion_from_the_pipeline_ends_a_nap_end_to_end(tmp_path):
    from roboemot.vision import Frame
    from test_vision import rig as vision_rig
    from test_vision_scene_motion import DT, GATE, MOTION, SPEC, joint, scene

    x = napping(rig())
    _, vbus, pipe, _, _ = vision_rig(tmp_path, lambda i: [], frame_spec=SPEC)
    vbus.subscribe("percept.scene_motion", lambda m: x.bus.publish(m.topic, m.payload, m.stamp))
    pipe.motion_wanted = x.r.wants_scene_motion
    span = GATE.settle_s + 2 * MOTION.interval_s
    n, m = round(span / DT), round(3 * MOTION.interval_s / DT)
    for i in range(n + m):
        pipe.process(Frame(scene(100 if i >= n else None), i * DT, joint()))
    woke(x, "motion")
    assert x.r.wants_scene_motion() is False
