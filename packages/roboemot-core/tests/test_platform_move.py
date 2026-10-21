"""The motion loop yields to the platform's own moves (SPEC 4A, 12A): no write while one runs
or during the grace after it, and a motors-off seen in the grace is never written over."""

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
}
CFG = ConfigStack().effective()
GRACE_S = CFG.motion.platform_move_grace_s
HOME = {"head.pitch": 0.0, "head.yaw": 0.0}
DT = 0.02
NOW = [0.0]  # the loop's clock: ``drive`` advances it with the simulated time


def manifest() -> Manifest:
    model = ManifestModel(robot=RobotInfo(id="test"))
    return Manifest(
        model, DOFS, [Constraint(type="yaw_delta", a="head.yaw", b="head.pitch", max_deg=180)]
    )


class PlatformDriver(MockPoseDriver):
    """``read_state`` follows ``home`` (the platform moves the robot); ``mode`` is the motors."""

    def read_state(self):
        return dict(self.home)


def make(grace_s: float | None = None) -> tuple[MotionLoop, PlatformDriver]:
    cfg = CFG
    if grace_s is not None:
        cfg = ConfigStack().effective().model_copy(deep=True)
        cfg.motion.platform_move_grace_s = grace_s
    NOW[0] = 0.0
    driver = PlatformDriver(home=dict(HOME))
    loop = MotionLoop(manifest(), driver, cfg, rate_hz=50, clock=lambda: NOW[0])
    loop.set_mode("active")
    return loop, driver


def drive(loop: MotionLoop, t: float, seconds: float, yaw: float = 0.3) -> float:
    for _ in range(round(seconds / DT)):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": yaw}, t))
        NOW[0] = t
        loop.cycle(t, DT)
        t += DT
    NOW[0] = t
    return t


def test_no_write_while_a_platform_move_runs_nor_during_the_grace():
    loop, driver = make()
    t = drive(loop, 0.0, 0.5)
    n = len(driver.writes)
    assert n > 0
    driver.platform_move = True
    driver.home = {"head.pitch": 0.2, "head.yaw": -0.4}  # the sleep move carries the robot
    t = drive(loop, t, 1.0)
    assert len(driver.writes) == n and loop.snapshot()["platform_move"] is True
    driver.platform_move = False
    t = drive(loop, t, GRACE_S - 0.2)
    assert len(driver.writes) == n  # the grace: still nothing, the incident's 1 s of pull-back
    assert loop.snapshot()["yielding"] is True
    t = drive(loop, t, 0.5 + 0.2)
    assert len(driver.writes) > n and loop.snapshot()["yielding"] is False


def test_grace_end_with_motors_on_resumes_from_the_measured_pose():
    loop, driver = make()
    t = drive(loop, 0.0, 0.3)
    driver.platform_move = True
    driver.home = {"head.pitch": 0.1, "head.yaw": -0.4}
    t = drive(loop, t, 0.5)
    driver.platform_move = False
    n = len(driver.writes)
    t = drive(loop, t, GRACE_S + 1.0)
    resumed = [v["head.yaw"] for _, v in driver.writes[n:]]
    vmax = DOFS["head.yaw"].max_vel
    assert resumed[0] == pytest.approx(-0.4, abs=vmax * DT + 1e-9)  # no jump from our old pose
    assert max(abs(b - a) for a, b in zip(resumed, resumed[1:], strict=False)) <= vmax * DT + 1e-9
    assert resumed[-1] == pytest.approx(0.3, abs=1e-6)
    assert not loop.motors_off and driver.modes == ["active"]  # never turned the motors on


def test_motors_off_inside_the_grace_means_asleep_with_zero_writes():
    loop, driver = make()
    t = drive(loop, 0.0, 0.3)
    n = len(driver.writes)
    driver.platform_move = True
    t = drive(loop, t, 0.5)
    driver.platform_move = False
    t = drive(loop, t, 0.1)
    driver.mode = "limp"  # the app disables the motors right after the sleep move
    t = drive(loop, t, 0.2)
    assert loop.motors_off and loop.snapshot()["asleep"] is True
    t = drive(loop, t, GRACE_S + 3.0)  # past the grace: still asleep, still nothing written
    assert len(driver.writes) == n and loop.motors_off
    driver.mode = "active"  # woken
    drive(loop, t, 3.0)
    assert len(driver.writes) > n and not loop.motors_off


def test_the_signal_going_unknown_mid_move_counts_as_the_move_ending():
    loop, driver = make()
    t = drive(loop, 0.0, 0.3)
    driver.platform_move = True
    t = drive(loop, t, 0.3)
    driver.platform_move = None  # a stale cache: the move is over as far as we can tell
    n = len(driver.writes)
    t = drive(loop, t, GRACE_S - 0.3)
    assert len(driver.writes) == n  # still inside the grace, not writing right after the move
    drive(loop, t, 1.0)
    assert len(driver.writes) > n


def test_a_second_move_in_the_grace_restarts_it():
    loop, driver = make()
    t = drive(loop, 0.0, 0.3)
    driver.platform_move = True
    t = drive(loop, t, 0.3)
    driver.platform_move = False  # goto_sleep: to the init pose, then to the sleep pose
    t = drive(loop, t, 0.3)
    driver.platform_move = True
    t = drive(loop, t, 0.3)
    driver.platform_move = False
    n = len(driver.writes)
    t = drive(loop, t, GRACE_S - 0.3)
    assert len(driver.writes) == n
    drive(loop, t, 1.0)
    assert len(driver.writes) > n


def test_a_zero_grace_resumes_as_the_move_ends():
    loop, driver = make(grace_s=0.0)
    t = drive(loop, 0.0, 0.3)
    driver.platform_move = True
    t = drive(loop, t, 0.3)
    driver.platform_move = False
    n = len(driver.writes)
    drive(loop, t, 0.2)
    assert len(driver.writes) > n


def test_a_driver_without_the_signal_never_yields():
    loop, driver = make()
    assert driver.platform_move_running() is None
    drive(loop, 0.0, 0.5)
    snap = loop.snapshot()
    assert snap["platform_move"] is None and snap["yielding"] is False
    assert len(driver.writes) == 25


def test_stop_closes_the_driver():
    loop, driver = make()
    closed: list[bool] = []
    driver.close = lambda: closed.append(True)  # type: ignore[method-assign]
    loop.stop()
    assert closed == [True]
