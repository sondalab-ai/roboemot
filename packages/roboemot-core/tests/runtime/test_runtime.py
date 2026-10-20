"""SPEC 4A, 11, 12A: the runtime is robot-agnostic. A fake platform and session built for the
``desk-head-v1`` manifest (fake driver, a camera without frames, no microphone) run the whole
pipeline for a few ticks, and it stops inside the shutdown budget."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.robot_platform import RobotPlatform, RobotSession
from roboemot.runtime import Runtime, RuntimeHost
from test_robot_platform import DeskPlatform, DeskSession

CONFIGS = Path(__file__).resolve().parents[3].parent / "configs"
TICKS = 3


class RecordingSession(DeskSession):
    """The desk session, keeping the driver the runtime's motion loop was given."""

    def driver(self):
        self.built = MockPoseDriver(home={"neck_pan": 0.0, "neck_tilt": 0.0})
        return self.built


class RecordingPlatform(DeskPlatform):
    def bind(self, client):
        self.session = RecordingSession(client)
        return self.session


def wait_for(predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_the_runtime_runs_a_non_reachy_robot_and_stops_within_the_budget():
    stack = ConfigStack.from_files(preset="default", preset_dirs=[CONFIGS / "presets"])
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    platform = RecordingPlatform()
    assert isinstance(platform, RobotPlatform)
    session = platform.bind(object())
    assert isinstance(session, RobotSession)
    host = RuntimeHost()
    stop, errors = threading.Event(), []
    ticks = []

    def run() -> None:
        try:
            Runtime(stack, manifest, platform, session, host=host).run(stop)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: host.diagnostics is not None)
        assert wait_for(lambda: host.expression is not None and host.circadian_runner is not None)
        t0 = time.monotonic()
        period_s = 1.0 / stack.effective().motion.app_tick_hz
        while time.monotonic() - t0 < TICKS * period_s + 0.5 and thread.is_alive():
            time.sleep(period_s)
            ticks.append(1)
    finally:
        started = time.monotonic()
        stop.set()
        thread.join(timeout=30)
    budget_s = stack.effective().runtime.shutdown_budget_s
    assert not errors, errors
    assert not thread.is_alive()
    assert time.monotonic() - started < budget_s
    assert len(ticks) >= TICKS
    report = host.diagnostics.report()["sections"]
    assert report["roboemot.app"]["data"]["manifest"] == "desk-head-v1"
    assert report["roboemot.motion"]["ok"]
    assert platform.session.built.writes  # the motion loop drove the fake joints
    assert host.circadian_runner is None  # stopped: hold and wake report it
