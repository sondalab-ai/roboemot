"""The motion loop yields when something else controls the robot (SPEC 4A, 12A): the measured
pose stops following the commands (the platform runs a move of its own and drops our targets, or
a hand moves the head). No write while that lasts and during the grace after the pose is still;
a motors-off seen then is never written over."""

from __future__ import annotations

import math

import pytest
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import Constraint, Dof, Manifest, ManifestModel, RobotInfo
from roboemot.motion import Command, MotionLoop, Priority

DOFS = {
    "head.pitch": Dof("head.pitch", "head", -0.7, 0.7, 2.0, 15.0),
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
    "head.z": Dof("head.z", "head", -0.05, 0.05, 0.2, 2.0, kind="translation"),
}
CFG = ConfigStack().effective()
GRACE_S = CFG.motion.platform_move_grace_s
DT = 0.02
TAU_S = 0.1  # the simulated servos' first-order tracking lag
NOW = [0.0]


def manifest() -> Manifest:
    model = ManifestModel(robot=RobotInfo(id="test"))
    return Manifest(
        model, DOFS, [Constraint(type="yaw_delta", a="head.yaw", b="head.pitch", max_deg=180)]
    )


class ServoDriver(MockPoseDriver):
    """A robot whose measured pose lags the last accepted target by ``TAU_S``. While ``external``
    is set the platform owns it: targets are dropped and the pose follows ``goal`` at ``speed``."""

    external = False
    goal: dict[str, float] = {}
    speed = 1.0

    def setup(self) -> None:
        self.pose = dict(self.home)
        self.target = dict(self.home)
        self.accepted: list[float] = []

    def write_pose_targets(self, t, values):
        super().write_pose_targets(t, values)
        if not self.external:
            self.target = dict(values)
            self.accepted.append(t)

    def read_state(self):
        return dict(self.pose)

    def advance(self, dt: float) -> None:
        for k in self.pose:
            if self.external:
                step = self.speed * dt
                d = self.goal.get(k, self.pose[k]) - self.pose[k]
                self.pose[k] += max(-step, min(step, d))
                self.target[k] = self.pose[k]  # the platform leaves the targets where it ended
            else:
                self.pose[k] += (self.target[k] - self.pose[k]) * (1 - math.exp(-dt / TAU_S))

    def motors_enabled(self):
        return self.mode == "active"


def make() -> tuple[MotionLoop, ServoDriver]:
    NOW[0] = 0.0
    driver = ServoDriver(home={"head.pitch": 0.0, "head.yaw": 0.0, "head.z": 0.0})
    driver.setup()
    loop = MotionLoop(manifest(), driver, CFG, rate_hz=50, clock=lambda: NOW[0])
    loop.set_mode("active")
    return loop, driver


def drive(loop, driver, t, seconds, yaw=0.3):
    for _ in range(round(seconds / DT)):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": yaw}, t))
        NOW[0] = t
        loop.cycle(t, DT)
        driver.advance(DT)
        t += DT
    NOW[0] = t
    return t


def test_default_grace_outlasts_the_platforms_two_second_wait_before_motors_off():
    assert GRACE_S > 2.0


def test_our_own_lagging_motion_never_triggers_it():
    loop, driver = make()
    t = 0.0
    for yaw in (0.3, -0.5, 1.2, -1.0, 0.0):  # hard reversals at the limiter's speed limit
        t = drive(loop, driver, t, 1.5, yaw=yaw)
    assert not loop.external and loop.external_events == 0
    assert len(driver.accepted) == len(driver.writes)


def test_a_platform_move_is_noticed_and_nothing_is_written_during_it_or_the_grace():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal, driver.speed = True, {"head.pitch": 0.5, "head.yaw": -1.0}, 1.0
    t = drive(loop, driver, t, 0.5)  # a fraction of a second: the confirmation
    assert loop.external and loop.snapshot()["external"] is True
    n = len(driver.writes)
    t = drive(loop, driver, t, 1.6)  # the move runs on, then the pose sits still
    assert driver.pose["head.yaw"] == pytest.approx(-1.0)
    t = drive(loop, driver, t, GRACE_S - 1.5)
    assert len(driver.writes) == n and loop.external
    driver.external = False  # the platform lets go; the pose has been still for the grace
    t = drive(loop, driver, t, 1.5)
    assert not loop.external and len(driver.writes) > n


def test_motors_off_in_the_grace_means_asleep_with_zero_writes():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.pitch": 0.5, "head.yaw": -1.0}
    t = drive(loop, driver, t, 2.5)
    n = len(driver.writes)
    driver.mode = "limp"  # the platform disables the motors 2 s after the move
    t = drive(loop, driver, t, 0.2)
    assert loop.motors_off and loop.snapshot()["asleep"] is True and not loop.external
    t = drive(loop, driver, t, GRACE_S + 3.0)
    assert len(driver.writes) == n
    driver.mode, driver.external = "active", False  # woken
    drive(loop, driver, t, 3.0)
    assert len(driver.writes) > n and not loop.motors_off


def test_grace_end_with_motors_on_resumes_from_the_measured_pose_without_a_jump():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.yaw": -0.8}
    t = drive(loop, driver, t, 2.0)
    n = len(driver.writes)
    driver.external = False  # the move is over, motors stay on
    t = drive(loop, driver, t, GRACE_S + 0.5)
    resumed = [v["head.yaw"] for _, v in driver.writes[n:]]
    vmax = DOFS["head.yaw"].max_vel
    assert resumed and resumed[0] == pytest.approx(-0.8, abs=vmax * DT * 2)
    assert max(abs(b - a) for a, b in zip(resumed, resumed[1:], strict=False)) <= vmax * DT + 1e-9
    assert driver.modes == ["active"]  # never turned the motors on


def test_a_hand_pushing_the_head_yields_and_control_returns_after_it_settles():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.pose["head.yaw"] = 0.3 + 0.5  # shoved well past the tolerance, held
    driver.target = dict(driver.pose)
    driver.external, driver.goal = True, {"head.yaw": 0.8}
    t = drive(loop, driver, t, 0.6)
    assert loop.external
    driver.external = False
    t = drive(loop, driver, t, GRACE_S + 0.5)
    assert not loop.external


def test_a_move_that_is_seen_again_in_the_grace_keeps_it_going():
    loop, driver = make()
    t = drive(loop, driver, 0.0, 1.0)
    driver.external, driver.goal = True, {"head.pitch": 0.3}
    t = drive(loop, driver, t, 1.0)
    t = drive(loop, driver, t, GRACE_S - 1.0)  # nearly out of the grace
    driver.goal = {"head.pitch": -0.3}  # a second move (go-to-sleep is two)
    n = len(driver.writes)
    t = drive(loop, driver, t, 1.5)
    assert loop.external and len(driver.writes) == n
