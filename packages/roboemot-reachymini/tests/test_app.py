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


def _attention_rig():
    from roboemot.attention import AttentionSelector, GazeController
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.drivers import MockPoseDriver
    from roboemot.manifest import load_manifest
    from roboemot.motion import MotionLoop
    from roboemot.world import WorldModel
    from roboemot_reachymini.gaze import FRAME, GAZE

    cfg = RoboEmotConfig()
    manifest = load_manifest(
        app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
    )
    driver = MockPoseDriver(home={n: d.home for n, d in manifest.dofs.items()})
    loop = MotionLoop(manifest, driver, cfg, rate_hz=50, clock=lambda: 0.0)
    loop.set_mode("active")
    world = WorldModel(cfg.world, FRAME)
    selector = AttentionSelector(cfg.attention)
    gaze = GazeController(cfg.attention, GAZE, manifest.dofs, manifest.constraints)
    return cfg, loop, world, selector, gaze


def test_attention_tick_failure_is_logged_and_falls_back_to_idle(caplog):
    from roboemot.attention import SOURCE
    from roboemot.world import Observation

    cfg, loop, world, selector, gaze = _attention_rig()
    world.observe(Observation("person", 0.0, "t", 0.5, 0.0, "face"))
    assert app_module.attention_tick(0.0, world, selector, gaze, loop)
    assert SOURCE in {c.source for c in loop.core.arbiter.active(0.0)}

    def boom(now, entries):
        raise RuntimeError("selector exploded")

    selector.update = boom
    with caplog.at_level("WARNING"):
        assert app_module.attention_tick(0.1, world, selector, gaze, loop) is False
    assert "RuntimeError" in caplog.text and "selector exploded" in caplog.text
    assert SOURCE not in {c.source for c in loop.core.arbiter.active(0.1)}
    assert not gaze.active


def test_attention_tick_uses_the_camera_view_for_world_decay():
    from roboemot.world import Observation

    cfg, loop, world, selector, gaze = _attention_rig()
    world.observe(Observation("person", 0.0, "t", 0.0, 0.0, "face"))
    world.observe(Observation("person", 0.0, "t", math.radians(120), 0.0, "face"))
    app_module.attention_tick(30.0, world, selector, gaze, loop)  # remembers the view state
    app_module.attention_tick(60.0, world, selector, gaze, loop)
    conf = {round(e.azimuth_rad, 2): e.confidence for e in world.entries}
    assert conf[0.0] < conf[round(math.radians(120), 2)]  # in view fades faster than out of view


def test_reload_updates_attention_consumers():
    cfg, loop, world, selector, gaze = _attention_rig()
    new = cfg.model_copy(deep=True)
    new.attention.release_s = 9.0
    new.world.match_gate_deg = 33.0
    new.safety.constraint_margin_deg = 7.0
    app_module.apply_attention_config(new, world, selector, gaze)
    assert selector.cfg.release_s == 9.0 and gaze.cfg.release_s == 9.0
    assert world.cfg.match_gate_deg == 33.0
    assert gaze.margin_rad == pytest.approx(math.radians(7.0))


def test_app_exposes_its_bus():
    assert "bus" in app_module.RoboEmotApp.__annotations__


def test_breathing_rides_on_attention_as_an_additive_layer():
    from roboemot.attention import SOURCE
    from roboemot.world import Observation

    cfg, loop, world, selector, gaze = _attention_rig()
    world.observe(Observation("person", 0.0, "t", 0.0, 0.0, "face"))
    t = 0.0
    for _ in range(100):
        t += 0.05
        world.observe(Observation("person", t, "t", 0.0, 0.0, "face"))
        attending = app_module.attention_tick(t, world, selector, gaze, loop)
        app_module.feed_idle(loop, cfg, t, attending)
    srcs = {c.source: c for c in loop.core.arbiter.active(t)}
    assert SOURCE in srcs and srcs["idle-additive"].additive
    loop.core.arbiter.crossfade_s = 0.0
    out = loop.core.arbiter.step(t)
    breathing = srcs["idle-additive"].targets["head.pitch"]
    assert out["head.pitch"] == pytest.approx(srcs[SOURCE].targets["head.pitch"] + breathing)

    app_module.feed_idle(loop, cfg, t, False)  # attention released: plain idle again
    assert "idle-additive" not in {c.source for c in loop.core.arbiter.active(t)}


def test_a_failing_source_start_does_not_leak_the_loop_or_network(tmp_path, monkeypatch):
    """The sources start inside the try block, so the loop and network monitor still stop."""
    from roboemot.world import WorldPlugin

    class Boom:
        name = "boom"
        stopped = False

        def start(self, sink):
            raise RuntimeError("no sensor")

        def stop(self):
            Boom.stopped = True

    data = tmp_path / "data"
    shutil.copytree(app_module.DATA, data)
    shutil.copy2(ROOT / "configs/dev-unmeasured.yaml", data / "app.yaml")
    monkeypatch.setattr(app_module, "DATA", data)
    monkeypatch.setattr(
        app_module, "load_world_plugin", lambda *a, **k: WorldPlugin(None, [Boom()])
    )
    threads = []
    real_start = app_module.MotionLoop.start

    def tracking_start(self):
        real_start(self)
        threads.append(self)

    monkeypatch.setattr(app_module.MotionLoop, "start", tracking_start)
    app = app_module.RoboEmotApp.__new__(app_module.RoboEmotApp)
    with pytest.raises(RuntimeError, match="no sensor"):
        app.run(fake_mini(), threading.Event())
    assert Boom.stopped
    assert not threads[0]._thread.is_alive()
