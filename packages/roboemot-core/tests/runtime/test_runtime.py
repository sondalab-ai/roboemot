"""SPEC 4A, 11, 12A: the runtime is robot-agnostic. A fake platform and session built for the
``desk-head-v1`` manifest (fake driver, a camera without frames, no microphone) run the whole
pipeline: the main loop's idle breathing reaches the desk head's pitch joint (from the platform's
gaze map), and it stops inside the shutdown budget."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboemot import paths
from roboemot.config import ConfigStack
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.robot_platform import RobotPlatform, RobotSession
from roboemot.runtime import Runtime, RuntimeHost
from roboemot.runtime import runtime as runtime_module
from test_robot_platform import DeskPlatform, DeskSession

CONFIGS = Path(__file__).resolve().parents[3].parent / "configs"


class DaytimeHost(RuntimeHost):
    """Real monotonic time at a fixed noon: quiet hours would put the robot to rest."""

    def make_circadian_clock(self):
        return SimpleNamespace(monotonic=time.monotonic, hour=lambda: 12.0)


class RecordingSession(DeskSession):
    """The desk session, keeping the driver the runtime's motion loop was given."""

    built: MockPoseDriver | None = None

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
    host = DaytimeHost()

    def breathing() -> bool:
        """The idle command moved the desk head's pitch joint (the gaze map's dof)."""
        driver = platform.session.built
        return driver is not None and any(
            abs(w.get("neck_tilt", 0.0)) > 0 for _, w in driver.writes
        )

    stop, errors = threading.Event(), []

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
        assert host.speaking is not None  # idle: the desk head ships no speech_motion_map
        assert wait_for(breathing)
    finally:
        started = time.monotonic()
        stop.set()
        thread.join(timeout=30)
    budget_s = stack.effective().runtime.shutdown_budget_s
    assert not errors, errors
    assert not thread.is_alive()
    assert time.monotonic() - started < budget_s
    report = host.diagnostics.report()["sections"]
    assert report["roboemot.app"]["data"]["manifest"] == "desk-head-v1"
    assert report["roboemot.motion"]["ok"]
    assert breathing()
    assert host.circadian_runner is None  # stopped: hold and wake report it


def test_web_ui_settings_are_replayed_before_the_pipeline_is_built(tmp_path, monkeypatch):
    """Regression (robot QA 2026-10-28): a restart-only edit saved from the web UI (here
    ``voice.realtime_route.mode``) is in the config the pipeline is built from, so it takes
    effect at the next start. The overrides file was replayed only when the web UI started,
    after the realtime provider had been chosen from the config without it."""
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    store = paths.settings_overrides_path(str(data))
    store.write_text(json.dumps({"v": 1, "overrides": {"voice.realtime_route.mode": "platform"}}))
    store.chmod(0o600)
    stack = ConfigStack(app={"data_dir": str(data)}, audit_path=tmp_path / "audit.jsonl")
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    platform = RecordingPlatform()
    seen: list[str] = []

    class Assembled(Exception):
        """Stops the run once the pipeline would be built."""

    def assemble(host, stack, *args):
        seen.append(stack.effective().voice.realtime_route.mode)
        raise Assembled

    monkeypatch.setattr(runtime_module, "assemble", assemble)
    with pytest.raises(Assembled):
        Runtime(stack, manifest, platform, platform.bind(object())).run(threading.Event())
    assert seen == ["platform"]
