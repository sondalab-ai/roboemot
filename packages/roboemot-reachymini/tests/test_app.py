"""App wrapper (SPEC 4A packaging; SPEC 12A diagnostics registry)."""

from __future__ import annotations

import json
import runpy
import shutil
import sys
import threading
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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
