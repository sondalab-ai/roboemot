"""App wrapper (SPEC 4A packaging; SPEC 12A diagnostics registry)."""

from __future__ import annotations

import json
import math
import runpy
import shutil
import sys
import threading
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from roboemot_reachymini import app as app_module

ROOT = Path(__file__).resolve().parents[3]


def test_entry_point_module_runs_the_app_when_executed(monkeypatch):
    """Regression: the daemon runs ``python -m <entry point module>``; app.py has no runner, so
    the entry point must name main.py or the app process exits at once on the robot."""
    (ep,) = [e for e in entry_points(group="reachy_mini_apps") if e.name == "roboemot"]
    assert ep.load() is app_module.RoboEmotApp
    ran = []
    monkeypatch.setattr(app_module.RoboEmotApp, "__init__", lambda self: None)
    monkeypatch.setattr(
        app_module.RoboEmotApp, "wrapped_run", lambda self: ran.append(1), raising=False
    )
    monkeypatch.delitem(sys.modules, ep.module, raising=False)  # fresh run, as in a subprocess
    runpy.run_module(ep.module, run_name="__main__")
    assert ran == [1]


def fake_mini() -> MagicMock:
    mini = MagicMock()
    mini.client._is_alive = True
    mini.client._last_status = None
    mini.get_current_head_pose.return_value = np.eye(4)
    mini.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return mini


def test_run_fills_a_json_ready_diagnostics_registry(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(app_module.DATA, data)
    shutil.copy2(ROOT / "configs/dev-unmeasured.yaml", data / "app.yaml")
    monkeypatch.setattr(app_module, "DATA", data)
    stop = threading.Event()
    stop.set()
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)  # skip the SDK's daemon probe
    app.run(fake_mini(), stop)

    report = app.diagnostics.report()
    assert json.loads(json.dumps(report)) == report
    sections = report["sections"]
    for name in ("system", "temperatures", "battery", "process", "roboemot.motion"):
        assert sections[name]["ok"], (name, sections[name])
    assert {"robot.daemon", "robot.motors", "robot.state", "robot.app"} <= set(sections)
    assert sections["roboemot.network"]["data"] == {"online": True, "routes": {}}
    assert sections["roboemot.app"]["data"]["manifest"] == "reachy-mini-wireless"
    assert sections["process"]["data"]["processes"][0]["pid"] > 0


def test_attention_tick_turns_the_head_then_releases_to_idle():
    """SPEC 5, 10: a person at az +30, el +10 pulls the head left and up; once the entry goes
    stale and the release time passes, the attention command is cleared (idle takes over)."""
    from roboemot.attention import SOURCE, AttentionSelector, GazeController
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.drivers import MockPoseDriver
    from roboemot.manifest import load_manifest
    from roboemot.motion import MotionLoop
    from roboemot.world import Observation, WorldModel
    from roboemot_reachymini.gaze import FRAME, GAZE

    cfg = RoboEmotConfig()
    manifest = load_manifest(
        app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
    )
    driver = MockPoseDriver(home={n: d.home for n, d in manifest.dofs.items()})
    clock = [0.0]
    loop = MotionLoop(manifest, driver, cfg, rate_hz=50, clock=lambda: clock[0])
    loop.set_mode("active")
    world = WorldModel(cfg.world, FRAME)
    selector = AttentionSelector(cfg.attention)
    gaze = GazeController(cfg.attention, GAZE, manifest.dofs, manifest.constraints)

    def run(seconds: float, observe: bool) -> bool:
        active = False
        for _ in range(round(seconds * 50)):
            if observe and round(clock[0] * 50) % 5 == 0:
                world.observe(
                    Observation(
                        "person", clock[0], "test", math.radians(30), math.radians(10), "face"
                    )
                )
            if round(clock[0] * 50) % 2 == 0:
                active = app_module.attention_tick(clock[0], world, selector, gaze, loop)
            loop.cycle(clock[0], 0.02)
            clock[0] += 0.02
        return active

    assert run(6.0, observe=True)
    state = driver.read_state()
    assert state["head.yaw"] == pytest.approx(math.radians(30), abs=0.05)
    assert state["head.pitch"] == pytest.approx(-math.radians(10), abs=0.05)

    cfg_a = cfg.attention
    assert not run(cfg_a.stale_s + cfg_a.release_s + 2.0, observe=False)
    assert SOURCE not in {c.source for c in loop.core.arbiter.active(clock[0])}
