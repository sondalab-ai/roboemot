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
from roboemot.config import ConfigError, ConfigStack
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


def test_a_loud_sound_during_the_robots_own_motion_does_not_wake_the_nap():
    """SPEC 10 (own-motion gate): the reflex engine holds sound.loud back while the robot moves,
    so its own rattle never ends a nap; the same sound once it is still does."""
    from roboemot.motion import Command, Priority
    from test_reflex import LOUD, YAW, Rig, cfg_with, rule

    x = napping(rig())
    reflexes = Rig(cfg_with(rule(enabled=False)))
    reflexes.engine.bus = x.bus
    reflexes.settle_audio()
    for i in range(10):  # an expression swings the head fast
        target = 0.6 if i % 2 == 0 else -0.6
        reflexes.loop.feed(Command("expr", Priority.COMMANDED, {YAW: target}, reflexes.t))
        reflexes.tick(5)
    reflexes.engine.offer_audio_level(LOUD, reflexes.t)
    run(x, STEP)
    assert x.r.model.reason == "nap"
    reflexes.settle_audio(2.0)  # still again, past the tail
    reflexes.engine.offer_audio_level(LOUD, reflexes.t)
    woke(x, "loud")


# ------------------------------------------------- resume after a stimulus wake (2026-11-05)


def test_the_resume_default_is_a_few_minutes_and_valid_for_every_nap_after_s():
    assert config().circadian.nap_resume_after_s == 180.0
    assert config(nap_after_s=300.0).circadian.nap_resume_after_s < 300.0


@pytest.mark.parametrize("resume", [300.0, 600.0])
def test_the_resume_must_be_below_nap_after_s(resume):
    with pytest.raises(ConfigError, match="nap_resume_after_s"):
        config(nap_after_s=300.0, nap_resume_after_s=resume)
    assert config(nap_after_s=None, nap_resume_after_s=resume).circadian.nap_after_s is None


@pytest.mark.parametrize("name", sorted(STIMULI))
def test_a_stimulus_wake_with_nobody_following_naps_again_after_the_resume_time(name):
    """2026-11-05: a loud onset woke a nap with nobody in the room and the robot stayed awake for
    the full nap_after_s. The idle clock counts from the end of the wake hold, as for activity."""
    x = napping(rig())
    STIMULI[name](x)
    woke(x, name)
    c = x.cfg.circadian
    assert set(run(x, c.wake_hold_s + c.nap_resume_after_s - 3 * STEP)) == {"awake"}
    run(x, 5 * STEP)
    assert x.r.model.state == "resting" and x.r.model.reason == "nap"
    snap = x.r.snapshot()["nap"]
    assert snap["naps"] == 2 and snap["resumes"] == 1
    assert snap["resume_after_s"] == c.nap_resume_after_s
    assert x.events[-1] == ("nap", "start")


def test_a_light_wake_with_nobody_following_also_resumes_soon():
    x = napping_lit(rig())
    past_grace(x)
    lit(x, x.cfg.light.confirm_s + 2 * STEP, BRIGHT)
    assert x.r.model.state == "awake" and x.events[-1] == ("nap", "light")
    c = x.cfg.circadian
    assert set(lit(x, c.wake_hold_s + c.nap_resume_after_s - 3 * STEP, BRIGHT)) == {"awake"}
    lit(x, 5 * STEP, BRIGHT)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 1


def confirmed_face(x) -> None:
    """Two sightings from two frames one second apart: a confirmed face."""
    face(x, stamp=x.clock.t)
    x.clock.advance(1.0)
    face(x, stamp=x.clock.t)


@pytest.mark.parametrize("person", ["face", "speech", "conversation"])
def test_a_person_after_the_stimulus_wake_brings_back_nap_after_s(person):
    talking = [False]
    x = napping(rig(conversing=lambda: talking[0]))
    loud(x)
    woke(x, "loud")
    run(x, 6 * STEP)  # inside the wake hold
    if person == "face":
        confirmed_face(x)
    elif person == "speech":
        say(x, addressed=True)
    else:
        talking[0] = True
        run(x, STEP)
        talking[0] = False
    held = x.r.model.hold_remaining()
    assert held is not None
    assert "resting" not in run(x, held[1] + x.cfg.circadian.nap_after_s - 3 * STEP)
    run(x, x.cfg.circadian.presence_window_s + 5 * STEP)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 0


def test_a_face_during_the_resume_idle_time_brings_back_nap_after_s():
    x = napping(rig())
    motion(x)
    woke(x, "motion")
    c = x.cfg.circadian
    run(x, c.wake_hold_s + c.nap_resume_after_s / 2)
    assert x.r.model.state == "awake"
    confirmed_face(x)
    assert "resting" not in run(x, c.nap_after_s - 3 * STEP)
    run(x, c.presence_window_s + 5 * STEP)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 0


@pytest.mark.parametrize("call", [lambda r: r.wake(), lambda r: r.hold("awake", 10)])
def test_a_wake_or_hold_from_outside_after_a_stimulus_wake_brings_back_nap_after_s(call):
    x = napping(rig())
    loud(x)
    woke(x, "loud")
    call(x.r)
    run(x, STEP)
    held = x.r.model.hold_remaining()[1]
    assert "resting" not in run(x, held + x.cfg.circadian.nap_after_s - 3 * STEP)


def test_the_night_ends_a_pending_resume():
    """A stimulus wake shortly before the quiet hours, the resume time not yet over at 22:30:
    the next morning uses nap_after_s (the idle time before 22:30 still counts, as always)."""
    x = napping(rig(hour=21.75, nap_resume_after_s=900.0))
    loud(x)
    woke(x, "loud")
    while x.clock.hour() < 7.0 or x.clock.hour() > 22.0:
        run(x, 60, step=60)
    assert "resting" not in run(x, x.cfg.circadian.nap_resume_after_s)
    while x.r.model.reason != "nap":
        run(x, STEP)
    assert x.r.snapshot()["nap"]["resumes"] == 0


def test_repeated_loud_onsets_never_loop_faster_than_the_resume_time():
    """A loud onset every two minutes for two hours: each wake holds the robot awake for
    wake_hold_s, then it naps again after nap_resume_after_s; never a tight wake/nap loop."""
    x = napping(rig())
    c = x.cfg.circadian
    starts = [x.r.nap.started]
    for _ in range(60):
        loud(x)
        run(x, 120)
        if x.r.nap.started != starts[-1]:
            starts.append(x.r.nap.started)
    assert len(starts) >= 5
    for a, b in zip(starts, starts[1:], strict=False):
        assert b - a >= c.wake_hold_s + c.nap_resume_after_s
    snap = x.r.snapshot()["nap"]
    assert snap["naps"] == len(starts) and snap["resumes"] == len(starts) - 1


def test_lowering_the_resume_time_while_resuming_never_fires_from_a_stale_clock():
    x = napping(rig(nap_resume_after_s=900.0))
    loud(x)
    woke(x, "loud")
    c = x.cfg.circadian
    run(x, c.wake_hold_s + 600)
    x.r.apply_config(config(nap_resume_after_s=300.0))
    assert "resting" not in run(x, 300 - 3 * STEP)
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 1


def stimulus_woken(x) -> SimpleNamespace:
    """A nap, then a loud onset ends it: a resume is pending."""
    napping(x)
    loud(x)
    woke(x, "loud")
    assert x.r.snapshot()["nap"]["resuming"] is True
    return x


def full_idle_time_again(x) -> None:
    """The next nap is not a resumed one: none within nap_after_s - nap_resume_after_s of now
    (well past the resume time; the clock may have started a few ticks ago)."""
    c = x.cfg.circadian
    assert "resting" not in run(x, c.nap_after_s - c.nap_resume_after_s)
    while x.r.model.reason != "nap":
        run(x, STEP)
    assert x.r.snapshot()["nap"]["resumes"] == 0


def test_the_end_of_a_platform_sleep_ends_a_pending_resume():
    x = stimulus_woken(rig())
    x.bus.publish("robot.asleep", None, x.clock.t)
    run(x, 3600)
    x.bus.publish("robot.awake", None, x.clock.t)
    assert x.r.snapshot()["nap"]["resuming"] is False
    full_idle_time_again(x)


def test_a_rest_of_the_rhythms_own_ends_a_pending_resume():
    x = stimulus_woken(rig(hour=15.0, peak_hour=3.0, initial_sleep_pressure=0.0))
    x.r.model.s = 1.0  # sleep pressure up: the rhythm rests the robot after the wake hold
    while x.r.model.reason != "sleepiness":
        run(x, STEP)
    assert x.r.snapshot()["nap"]["resuming"] is False
    x.r.model.s = 0.0
    while x.r.model.state == "resting":
        run(x, STEP)
    full_idle_time_again(x)


def test_turning_naps_off_ends_a_pending_resume():
    x = stimulus_woken(rig())
    x.r.apply_config(config(nap_after_s=None))
    run(x, STEP)
    assert x.r.snapshot()["nap"]["resuming"] is False
    x.r.apply_config(config())
    run(x, x.cfg.circadian.wake_hold_s)
    full_idle_time_again(x)


def test_presence_holds_a_due_resumed_nap_off_until_it_ends():
    x = stimulus_woken(rig())
    c = x.cfg.circadian
    run(x, c.wake_hold_s + c.nap_resume_after_s - 3 * STEP)
    face(x)  # one sighting: presence for the window, not activity
    assert set(run(x, c.presence_window_s - STEP)) == {"awake"}
    assert x.r.snapshot()["nap"]["resuming"] is True
    run(x, 3 * STEP)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 1


def test_attention_on_a_person_keeps_a_pending_resume_armed():
    """Owner decision 2026-11-05: attention (and an unconfirmed face) holds a nap off but does
    not bring back nap_after_s."""
    engaged = [False]
    x = stimulus_woken(rig(engaged=lambda: engaged[0]))
    c = x.cfg.circadian
    engaged[0] = True
    assert set(run(x, c.wake_hold_s + 2 * c.nap_resume_after_s)) == {"awake"}
    assert x.r.snapshot()["nap"]["resuming"] is True
    engaged[0] = False
    assert set(run(x, c.nap_resume_after_s - 3 * STEP)) == {"awake"}
    run(x, 5 * STEP)
    assert x.r.model.reason == "nap" and x.r.snapshot()["nap"]["resumes"] == 1
