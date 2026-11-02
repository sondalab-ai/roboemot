"""SPEC 12A: the motion loop publishes each tick's movement for diagnostics on other threads
(the barge-in trigger records), without its lock."""

from __future__ import annotations

import math

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


def make_loop() -> MotionLoop:
    driver = MockPoseDriver(home={n: 0.0 for n in DOFS})
    loop = MotionLoop(Manifest(ManifestModel(robot=RobotInfo(id="test")), DOFS), driver, CFG)
    loop.set_mode("active")
    return loop


def test_no_reading_before_the_first_tick_and_still_when_nothing_moves():
    loop = make_loop()
    assert loop.motion_state() is None
    loop.step_once(0.0, DT)
    state = loop.motion_state()
    assert state is not None and state.stamp == 0.0
    assert (state.speed_dps, state.group, state.external, state.reflex) == (0.0, None, False, False)


def test_a_turn_reports_the_fastest_angle_dof_and_its_group():
    loop = make_loop()
    loop.feed(Command("gaze", Priority.COMMANDED, {"neck.yaw": 1.0, "ears.left": 0.1}, 0.0))
    for i in range(10):
        loop.step_once(i * DT, DT)
    state = loop.motion_state()
    assert state.group == "neck" and state.speed_dps > 0.0
    assert state.speed_dps == math.degrees(abs(loop.core.limiter.vel["neck.yaw"]))
    vel = loop.core.limiter.vel
    assert state.group_speed_dps == {
        "neck": math.degrees(abs(vel["neck.yaw"])),  # the translation is not an angular speed
        "ears": math.degrees(abs(vel["ears.left"])),
    }
    assert not state.reflex


def test_a_translation_is_not_an_angular_speed():
    loop = make_loop()
    loop.feed(Command("lift", Priority.COMMANDED, {"neck.z": 0.04}, 0.0))
    for i in range(5):
        loop.step_once(i * DT, DT)
    assert loop.motion_state().speed_dps == 0.0


def test_an_active_reflex_is_reported():
    loop = make_loop()
    loop.feed(Command("reflex.x", Priority.REFLEX, {"ears.left": 0.5}, 0.0, expires=0.1))
    loop.step_once(0.0, DT)
    assert loop.motion_state().reflex
    loop.step_once(0.2, DT)  # expired
    assert not loop.motion_state().reflex
