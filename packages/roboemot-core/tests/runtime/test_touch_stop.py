"""SPEC 4A, 11 Radio, 12A: a hand on the robot (external control) stops the radio, then a nod."""

from __future__ import annotations

import logging
import math

import pytest
from roboemot import diag
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import Dof, Manifest, ManifestModel, RobotInfo
from roboemot.motion import Command, MotionLoop, Priority
from roboemot.motion.external import YieldView
from roboemot.runtime.touch_stop import TouchStop, touch_stop_tick
from roboemot.webui.strings import STRINGS
from test_media_player import JAZZ
from test_media_player import Rig as PlayerRig

TICK = 0.05  # one app tick at 20 Hz
HAND = YieldView(units=2, away=1, unreachable=False)


class FakeRadio:
    def __init__(self, audible: bool = True) -> None:
        self.on = audible
        self.stops: list[str] = []
        self.fail = False

    def audible(self) -> bool:
        return self.on

    def stop(self, reason: str) -> None:
        if self.fail:
            raise RuntimeError("decoder")
        self.stops.append(reason)
        self.on = False


def config(**radio):
    return ConfigStack(app={"tools": {"radio": radio}}).effective()


class Rig:
    def __init__(self, view: YieldView | None = HAND, accept: bool = True, **radio) -> None:
        self.radio: FakeRadio | None = FakeRadio()
        self.view = view
        self.gestures: list[tuple[str, float]] = []
        self.events: list[tuple[str, str, dict]] = []
        self.accept = accept
        self.touch = TouchStop(lambda: self.radio, lambda: self.view, self._gesture)
        self.touch.events = lambda kind, reason, **f: self.events.append((kind, reason, f))
        self.cfg = config(**radio)
        self.check_s = self.cfg.tools.radio.touch_stop_check_s
        self.t = 100.0

    def _gesture(self, label: str, now: float) -> bool:
        self.gestures.append((label, now))
        return self.accept

    def run(self, seconds: float, external: bool, motors_off: bool = False) -> None:
        for _ in range(round(seconds / TICK)):
            self.touch.tick(self.t, external, motors_off, self.cfg)
            self.t += TICK

    @property
    def stops(self) -> list[str]:
        return self.radio.stops if self.radio is not None else []


def test_defaults():
    radio = config().tools.radio
    assert radio.touch_stop is True
    assert radio.touch_stop_check_s == 1.0
    assert radio.touch_stop_check_s < config().motion.platform_move_grace_s


def test_a_hand_holding_one_part_at_the_check_stops_the_radio_once():
    r = Rig()
    r.run(r.check_s - TICK, external=True)
    assert r.stops == []
    r.run(3.0, external=True)
    assert r.stops == ["touch"] and r.touch.stops == 1
    assert [(k, why) for k, why, _ in r.events] == [("media", "touch_stop")]
    assert r.events[0][2]["held_s"] == pytest.approx(r.check_s, abs=2 * TICK)
    assert r.touch.snapshot()["last_held_s"] == pytest.approx(r.check_s, abs=2 * TICK)


@pytest.mark.parametrize(
    ("view", "why"),
    [
        (None, "settled"),  # the yield ended or has no sample
        (YieldView(2, 0, False), "settled"),  # a lag that caught up: the pose is where we sent it
        (YieldView(2, 1, True), "unreachable"),  # our own target meeting the body
        (YieldView(2, 2, False), "platform"),  # the whole robot moves: a platform move
    ],
)
def test_a_yield_that_is_no_hand_does_not_stop_the_radio(view, why):
    r = Rig(view)
    r.run(3.0, external=True)
    assert r.stops == [] and r.events == []
    assert r.touch.skipped[why] == 1  # looked at once per yield
    r.run(TICK, external=False)
    assert r.gestures == []


def test_a_robot_of_one_part_takes_any_away_part_for_a_hand():
    r = Rig(YieldView(1, 1, False))
    r.run(2.0, external=True)
    assert r.stops == ["touch"]


def test_no_stop_while_the_radio_is_not_audible_or_absent():
    r = Rig()
    r.radio.on = False
    r.run(3.0, external=True)
    assert r.stops == [] and r.events == []
    r.radio = None
    r.run(3.0, external=True)
    assert r.events == [] and r.touch.stops == 0


def test_only_audible_time_counts():
    r = Rig()
    r.radio.on = False
    r.run(2.0, external=True)  # the music held (robot speech): the check waits
    r.radio.on = True
    r.run(r.check_s - TICK, external=True)
    assert r.stops == []
    r.run(3 * TICK, external=True)
    assert r.stops == ["touch"]


def test_no_stop_with_the_opt_in_off():
    r = Rig(touch_stop=False)
    r.run(3.0, external=True)
    r.run(TICK, external=False)
    assert r.stops == [] and r.events == [] and r.gestures == []


def test_the_nod_waits_for_the_loop_to_take_control_back_and_carries_the_time():
    r = Rig()
    r.run(4.0, external=True)  # the hand stays, then the grace runs
    assert r.stops == ["touch"] and r.gestures == []
    assert r.touch.snapshot()["nod_pending"] is True
    resumed = r.t
    r.run(TICK, external=False)
    assert r.gestures == [("nod", resumed)]  # with the time: a late nod is refused, not queued
    r.run(1.0, external=False)
    assert len(r.gestures) == 1
    snap = r.touch.snapshot()
    assert (snap["stops"], snap["nods"], snap["nod_pending"]) == (1, 1, False)


def test_a_refused_nod_is_counted():
    r = Rig(accept=False)
    r.run(2.0, external=True)
    r.run(TICK, external=False)
    assert r.touch.nods == 0 and r.touch.nods_refused == 1


def test_motors_off_in_the_same_read_as_the_yield_end_drops_the_nod():
    # the loop sets motors_off before it clears the yield: external False with motors off is
    # the platform's sleep, never a resume
    r = Rig()
    r.run(2.0, external=True)
    r.run(TICK, external=False, motors_off=True)
    r.run(1.0, external=False)
    assert r.gestures == [] and r.touch.nods_dropped == 1
    assert r.touch.stops_then_asleep == 1


def test_motors_off_long_after_a_stop_is_not_counted_as_a_platform_move():
    r = Rig()
    r.run(2.0, external=True)
    r.run(TICK, external=False)  # resumed and nodded
    r.run(r.cfg.motion.platform_move_grace_s + 1.0, external=False)
    r.run(TICK, external=False, motors_off=True)
    assert r.touch.stops_then_asleep == 0 and r.touch.nods == 1


def test_a_stop_that_raises_leaves_no_half_state():
    r = Rig()
    r.radio.fail = True
    with pytest.raises(RuntimeError):
        r.run(2.0, external=True)
    snap = r.touch.snapshot()
    assert (snap["stops"], snap["nod_pending"], r.events) == (0, False, [])


def test_the_main_loop_step_logs_a_failure_by_type_and_goes_on(caplog):
    class Loop:
        external, motors_off = True, False

    r = Rig()
    r.radio.fail = True
    caplog.set_level(logging.WARNING, logger="roboemot.runtime.touch_stop")
    for _ in range(round(2.0 / TICK)):
        touch_stop_tick(r.touch, r.t, Loop(), r.cfg)  # never raises
        r.t += TICK
    assert "touch stop: RuntimeError" in caplog.text and "decoder" not in caplog.text
    touch_stop_tick(None, r.t, Loop(), r.cfg)  # a run without the part


def test_the_main_loop_step_reads_external_before_motors_off():
    reads: list[str] = []

    class Loop:
        @property
        def external(self):
            reads.append("external")
            return False

        @property
        def motors_off(self):
            reads.append("motors_off")
            return False

    touch_stop_tick(Rig().touch, 0.0, Loop(), config())
    assert reads == ["external", "motors_off"]


def test_the_event_is_a_declared_one_and_has_strings_in_both_languages():
    assert "touch_stop" in diag.REASONS["media"]
    cleaned = diag.clean({"held_s": 1.004, "station": "Jazz"})
    assert cleaned == {"held_s": 1.0}
    for lang in ("en", "it"):
        assert "{held_s}" in STRINGS[lang]["live_ev_media_touch_stop"]


# --------------------------------------------------- a real motion loop and media player

DT = 0.02  # the motion loop at 50 Hz; the app tick runs every DT here too
DOFS = {
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
    "antenna": Dof("antenna", "antennas", -3.0, 3.0, 6.0, 40.0),
}
TAU_S = 0.05


class Robot(MockPoseDriver):
    """Position-controlled servos that lag their last accepted target by ``TAU_S``. ``hand``
    holds dofs where it puts them; ``platform`` (a move of its own) drops every target and
    takes each dof to its goal."""

    def setup(self) -> None:
        self.pose = dict(self.home)
        self.target = dict(self.home)
        self.hand: dict[str, float] = {}
        self.platform: dict[str, float] = {}

    def write_pose_targets(self, t, values):
        super().write_pose_targets(t, values)
        if not self.platform:
            self.target.update(values)

    def read_state(self):
        return dict(self.pose)

    def advance(self, dt: float) -> None:
        for k in self.pose:
            if k in self.hand:
                self.pose[k] = self.hand[k]
            elif self.platform:
                self.pose[k] = self.target[k] = self.platform[k]
            else:
                self.pose[k] += (self.target[k] - self.pose[k]) * (1 - math.exp(-dt / TAU_S))

    def motors_enabled(self):
        return self.mode == "active"


class World:
    def __init__(self) -> None:
        self.cfg = ConfigStack().effective()
        self.now = 0.0
        self.robot = Robot(home={"head.yaw": 0.0, "antenna": 0.0})
        self.robot.setup()
        manifest = Manifest(ManifestModel(robot=RobotInfo(id="test")), DOFS, [])
        self.loop = MotionLoop(manifest, self.robot, self.cfg, rate_hz=1 / DT, clock=self.clock)
        self.loop.set_mode("active")
        self.player = PlayerRig()
        assert self.player.player.play(JAZZ) == "ok"
        self.player.events.clear()
        self.gestures: list[str] = []
        self.touch = TouchStop(
            lambda: self.player.player,
            self.loop.yield_view,
            lambda label, now: self.gestures.append(label) or True,
        )

    def clock(self) -> float:
        return self.now

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            self.loop.feed(
                Command("app", Priority.COMMANDED, {"head.yaw": 0.2, "antenna": 0.1}, self.now)
            )
            self.loop.cycle(self.now, DT)
            self.robot.advance(DT)
            touch_stop_tick(self.touch, self.now, self.loop, self.cfg)
            self.now += DT

    @property
    def stopped(self) -> list[str]:
        return [p["reason"] for topic, p in self.player.events if topic == "media.stopped"]


def test_a_hand_on_the_head_stops_the_real_player_and_the_nod_comes_after_the_grace():
    w = World()
    w.run(1.0)
    w.robot.hand = {"head.yaw": 0.9}  # held well away from the commands
    w.run(0.5)
    assert w.loop.external and w.stopped == []
    w.run(1.0)
    assert w.stopped == ["touch"] and w.player.player.now_playing() is None
    w.robot.hand = {}  # let go: the servo goes back to its last target, then the grace runs
    w.run(w.cfg.motion.platform_move_grace_s - 1.0)
    assert w.loop.external and w.gestures == []
    w.run(2.0)
    assert not w.loop.external and w.gestures == ["nod"]


def test_a_platform_move_of_the_whole_robot_does_not_stop_the_real_player():
    w = World()
    w.run(1.0)
    w.robot.platform = {"head.yaw": -1.0, "antenna": 2.5}
    w.run(2.0)
    assert w.loop.external and w.stopped == [] and w.touch.skipped["platform"] == 1


def test_a_hand_gone_before_the_check_does_not_stop_the_real_player():
    w = World()
    w.run(1.0)
    w.robot.hand = {"head.yaw": 0.9}
    w.run(0.4)
    assert w.loop.external
    w.robot.hand = {}  # a brush: the servo is back on its target at the check
    w.run(1.5)
    assert w.stopped == [] and w.touch.skipped["settled"] == 1


def test_the_loop_sets_motors_off_before_it_clears_the_yield():
    w = World()
    seen: list[bool] = []
    clear = w.loop.external_ctl.clear

    def spy() -> None:
        seen.append(w.loop.motors_off)
        clear()

    w.loop.external_ctl.clear = spy
    w.run(1.0)
    w.robot.hand = {"head.yaw": 0.9}
    w.run(2.0)
    w.robot.mode = "limp"  # the platform turns the motors off while the loop yields
    w.run(w.cfg.motion.platform_move_grace_s)
    assert seen and all(seen)
    assert w.gestures == [] and w.touch.nods_dropped == 1
