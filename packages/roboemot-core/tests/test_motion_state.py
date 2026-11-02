"""SPEC 12A: the motion loop keeps raw movement values each tick for diagnostics on other threads
(the barge-in trigger records), without its lock; the reader derives per-group speeds."""

from __future__ import annotations

import math
import time

import pytest
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import Dof, Manifest, ManifestModel, RobotInfo
from roboemot.motion import Command, MotionLoop, Priority

DOFS = {
    "neck.yaw": Dof("neck.yaw", "neck", -math.pi, math.pi, 3.0, 20.0),
    "ears.left": Dof("ears.left", "ears", -1.0, 1.0, 2.0, 20.0),
    "neck.z": Dof("neck.z", "neck", -0.05, 0.05, 5.0, 50.0, kind="translation"),
}
CFG = ConfigStack().effective()
DT = 0.02
TICK_BUDGET_S = 0.002  # generous: a tick is ~65 us on a laptop; the period is 20 ms


def make_loop() -> MotionLoop:
    driver = MockPoseDriver(home={n: 0.0 for n in DOFS})
    loop = MotionLoop(Manifest(ManifestModel(robot=RobotInfo(id="test")), DOFS), driver, CFG)
    loop.set_mode("active")
    return loop


def state_of(loop):
    return loop.motion_state(loop.motion_sample())


def test_no_sample_before_the_first_tick_and_still_when_nothing_moves():
    loop = make_loop()
    assert loop.motion_sample() is None
    loop.step_once(0.0, DT)
    state = state_of(loop)
    assert state.stamp == 0.0 and not state.external and not state.reflex
    assert state.commanded_dps == {"neck": 0.0, "ears": 0.0}
    assert state.measured_dps == {}  # one fresh pose only: no measured speed yet


def test_a_turn_reports_commanded_and_measured_speed_per_group():
    loop = make_loop()
    loop.feed(Command("gaze", Priority.COMMANDED, {"neck.yaw": 1.0, "ears.left": 0.1}, 0.0))
    for i in range(10):
        loop.step_once(i * DT, DT)
    vel = loop.core.limiter.vel
    sample = loop.motion_sample()
    loop.step_once(10 * DT, DT)  # the sample is a copy: a later tick does not change it
    state = loop.motion_state(sample)
    assert state.commanded_dps == {
        "neck": math.degrees(abs(vel["neck.yaw"])),  # the translation is not an angular speed
        "ears": math.degrees(abs(vel["ears.left"])),
    }
    # the mock tracks perfectly: the measured speed is the commanded step over one tick
    assert state.measured_dps["neck"] > 0.0
    assert state.measured_dps["neck"] == pytest.approx(state.commanded_dps["neck"])
    assert not state.reflex


def test_a_translation_is_not_an_angular_speed():
    loop = make_loop()
    loop.feed(Command("lift", Priority.COMMANDED, {"neck.z": 0.04}, 0.0))
    for i in range(5):
        loop.step_once(i * DT, DT)
    state = state_of(loop)
    assert set(state.commanded_dps.values()) == {0.0} and set(state.measured_dps.values()) == {0.0}


def test_a_reflex_that_drives_a_dof_is_reported():
    loop = make_loop()
    loop.feed(Command("reflex.x", Priority.REFLEX, {"ears.left": 0.5}, 0.0, expires=0.1))
    loop.step_once(0.0, DT)
    assert state_of(loop).reflex
    loop.step_once(0.2, DT)  # expired
    assert not state_of(loop).reflex


def test_not_driving_clears_the_measured_poses():
    loop = make_loop()
    for i in range(3):
        loop.step_once(i * DT, DT)
    loop.set_mode("limp")
    loop.step_once(3 * DT, DT)
    sample = loop.motion_sample()
    assert sample.stamp == 3 * DT and sample.measured is None


def test_the_tick_stays_cheap():
    loop = make_loop()
    loop.feed(Command("gaze", Priority.COMMANDED, {"neck.yaw": 1.0, "ears.left": 0.5}, 0.0))
    n = 500
    t0 = time.perf_counter()
    for i in range(n):
        loop.step_once(i * DT, DT)
    assert (time.perf_counter() - t0) / n < TICK_BUDGET_S
