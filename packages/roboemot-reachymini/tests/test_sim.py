"""Integration tests against the MuJoCo Reachy Mini daemon (``scripts/sim-daemon.sh``).

Run with ``pytest -m sim``. They answer, in simulation, items of the SPEC 15 "verify in M0"
list and measure command-to-state latency through the real SDK and daemon.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

reachy_mini = pytest.importorskip("reachy_mini")

from roboemot.config import ConfigStack  # noqa: E402
from roboemot.manifest import load_manifest  # noqa: E402
from roboemot.motion import Command, MotionLoop, Priority  # noqa: E402
from roboemot_reachymini import ReachyMiniDriver  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "src/roboemot_reachymini/data"


@pytest.fixture(scope="module")
def mini():
    m = reachy_mini.ReachyMini(
        connection_mode="localhost_only", media_backend="no_media", timeout=10
    )
    yield m


@pytest.fixture()
def loop(mini):
    cfg = ConfigStack().effective()
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    lp = MotionLoop(manifest, ReachyMiniDriver(mini), cfg, rate_hz=50)
    lp.start()
    yield lp
    lp.stop()


def _hold(lp: MotionLoop, targets: dict[str, float], seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        lp.feed(Command("test", Priority.COMMANDED, dict(targets), time.monotonic()))
        time.sleep(0.02)


def test_streamed_pose_reaches_target(loop):
    target = {
        "head.yaw": 0.3,
        "head.pitch": 0.15,
        "body_yaw": 0.2,
        "antenna_right": -0.6,
        "antenna_left": 0.6,
    }
    _hold(loop, target, 8.0)  # from the sleep pose (antennas at ±3.05 rad) at fallback speed
    state = loop.driver.read_state()
    for k, v in target.items():
        assert state[k] == pytest.approx(v, abs=0.05), k
    assert loop.overruns == 0


def test_yaw_delta_constraint_holds_on_the_robot(loop):
    _hold(loop, {"head.yaw": 1.8, "body_yaw": 0.0}, 5.0)
    state = loop.driver.read_state()
    assert abs(state["head.yaw"] - state["body_yaw"]) <= math.radians(65) + 0.05


def test_command_to_state_latency(loop):
    """First visible movement after a step command; informs the 100 ms reflex budget."""
    _hold(loop, {"head.yaw": 0.0}, 1.5)
    start_yaw = loop.driver.read_state()["head.yaw"]
    t0 = time.monotonic()
    latency = None
    stop = threading.Event()

    def feeder() -> None:
        while not stop.is_set():
            loop.feed(Command("test", Priority.REFLEX, {"head.yaw": 0.4}, time.monotonic()))
            time.sleep(0.01)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    samples = []
    while time.monotonic() - t0 < 1.0:
        s = time.monotonic()
        yaw = loop.driver.read_state()["head.yaw"]
        samples.append(time.monotonic() - s)
        if latency is None and abs(yaw - start_yaw) > math.radians(1):
            latency = time.monotonic() - t0
    stop.set()
    th.join()
    print(
        f"\ncommand-to-state latency: {latency * 1000:.0f} ms; "
        f"state read: {statistics.median(samples) * 1000:.1f} ms median"
    )
    assert latency is not None and latency < 0.25


def _daemon_post(path: str) -> None:
    import urllib.request

    req = urllib.request.Request(f"http://localhost:8000/api{path}", method="POST")
    urllib.request.urlopen(req, timeout=30).read()


def test_robot_link_recovers_after_daemon_backend_restart(mini):
    """SPEC 12A: backend stop -> link.robot.lost; restart -> reconnect, link.robot.restored."""
    from roboemot_reachymini.app import idle_command

    events: list[str] = []
    cfg = ConfigStack().effective()
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    loop = MotionLoop(
        manifest,
        ReachyMiniDriver(mini),
        cfg,
        rate_hz=50,
        on_link_change=lambda e, n: events.append(e),
    )
    stop = threading.Event()

    def feed() -> None:
        while not stop.is_set():
            loop.feed(idle_command(cfg, time.monotonic()))
            stop.wait(0.05)

    loop.start()
    threading.Thread(target=feed, daemon=True).start()
    try:
        time.sleep(1.0)
        _daemon_post("/daemon/stop?goto_sleep=false")
        deadline = time.monotonic() + 10
        while "link.robot.lost" not in events and time.monotonic() < deadline:
            time.sleep(0.1)
        assert events == ["link.robot.lost"]
        _daemon_post("/daemon/start?wake_up=false")
        deadline = time.monotonic() + 30
        while "link.robot.restored" not in events and time.monotonic() < deadline:
            time.sleep(0.2)
        assert events == ["link.robot.lost", "link.robot.restored"]
        # The MuJoCo backend ignores motor-mode changes (always "enabled"), so the motors-off
        # path is covered by unit tests (test_health.py); here streaming must simply resume.
        assert loop.driver.motors_enabled() is True
        ticks = loop.ticks
        time.sleep(1.0)
        assert loop.ticks > ticks
        assert not loop.motors_off and loop.link.online
    finally:
        stop.set()
        loop.stop()


def test_debug_report_against_the_simulated_daemon():
    """SPEC 12A, Diagnostics: every daemon section answers; IMU is null in simulation."""
    from roboemot.diagnostics import Target, collect, platform_sources

    report = collect(platform_sources("reachy-mini-wireless", Target()))
    s = report["sections"]
    assert all(sec["ok"] for sec in s.values()), s
    assert s["robot.daemon"]["data"]["simulation_enabled"] is True
    assert s["robot.motors"]["data"]["mode"] in ("enabled", "disabled", "gravity_compensation")
    assert s["robot.state"]["data"]["imu"] is None


def test_attention_turns_the_head_towards_a_world_entry(loop):
    """SPEC 5, 10: a person injected at az +30 deg, el +10 deg makes the head turn left and up.

    Runs the app's own attention step against the simulated robot; the fallback dynamics are
    slow, so the timeout is generous.
    """
    from roboemot.attention import AttentionSelector, GazeController
    from roboemot.world import Observation, WorldModel
    from roboemot_reachymini.app import attention_tick
    from roboemot_reachymini.gaze import FRAME, GAZE

    cfg = ConfigStack().effective()
    manifest = loop.manifest
    world = WorldModel(cfg.world, FRAME)
    selector = AttentionSelector(cfg.attention)
    gaze = GazeController(cfg.attention, GAZE, manifest.dofs, manifest.constraints)
    az, el = math.radians(30), math.radians(10)
    deadline = time.monotonic() + 20.0
    state: dict[str, float] = {}
    while time.monotonic() < deadline:
        now = time.monotonic()
        world.observe(Observation("person", now, "sim-test", az, el, "face"))
        assert attention_tick(now, world, selector, gaze, loop)
        state = loop.driver.read_state()
        if state["head.yaw"] > 0.9 * az and state["head.pitch"] < -0.9 * el:
            break
        time.sleep(0.05)
    assert state["head.yaw"] > 0.9 * az, state  # turned left
    assert state["head.pitch"] < -0.9 * el, state  # looking up (nose up is negative pitch)


def test_full_sweep_keeps_the_world_map_correct(loop):
    """SPEC 5A (M1 exit): two objects planted aside and behind the robot are found by one full
    sweep, at the right base-frame bearings, with no duplicate entries.

    A stand-in detector reports each planted object whenever it is inside the camera's view of
    the measured pose; it passes that joint state so the world model rotates the bearing back
    into the base frame. Dwell is short because the fallback dynamics are slow.
    """
    import random

    from roboemot.attention import AttentionSelector, GazeController
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.world import (
        Observation,
        ScanController,
        WorldModel,
        angular_distance,
        from_base_frame,
    )
    from roboemot_reachymini.app import attention_tick, scan_context, scan_tick
    from roboemot_reachymini.gaze import FRAME, GAZE
    from roboemot_reachymini.world import ReachyScanPlanner

    cfg = RoboEmotConfig.model_validate(
        {"world": {"scan": {"idle_interval_s": 10, "jitter_s": 2, "dwell_s": 0.4}}}
    )
    manifest = loop.manifest
    world = WorldModel(cfg.world, FRAME)
    selector = AttentionSelector(cfg.attention)
    gaze = GazeController(cfg.attention, GAZE, manifest.dofs, manifest.constraints)
    scan = ScanController(
        cfg.world.scan,
        ReachyScanPlanner(manifest.dofs, manifest.constraints, cfg),
        manifest.dofs,
        random.Random(1),
    )
    planted = [math.radians(100), math.radians(-170)]
    half_fov = math.radians(cfg.world.camera_fov_h_deg) / 2 * 0.9
    half_fov_v = math.radians(cfg.world.camera_fov_v_deg) / 2 * 0.9

    scanned = False
    deadline = time.monotonic() + 240.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        state = loop.driver.read_state()
        for az in planted:
            s_az, s_el = from_base_frame(az, 0.0, state, FRAME)
            if abs(s_az) <= half_fov and abs(s_el) <= half_fov_v:
                world.observe(Observation("object", now, "sim-detector", s_az, s_el, "cup"), state)
        attention_tick(now, world, selector, gaze, loop)
        ctx = scan_context(now, cfg, selector, hour=12.0)
        scanning = scan_tick(ctx, scan, loop, cfg)
        if scanned and not scanning:
            break
        scanned |= scanning
        time.sleep(0.05)
    assert scanned and not scan.active, "the sweep did not finish in time"
    for _ in range(40):  # let the pose settle after the glide home
        time.sleep(0.1)
    state = loop.driver.read_state()
    assert abs(state["head.yaw"]) < math.radians(8), state
    assert abs(state["body_yaw"]) < math.radians(8), state

    objects = [e for e in world.entries if e.kind == "object"]
    assert len(objects) == len(planted), objects  # found both, no duplicates
    gate = math.radians(cfg.world.match_gate_deg)
    for az in planted:
        nearest = min(angular_distance(e.azimuth_rad, e.elevation_rad, az, 0.0) for e in objects)
        assert nearest <= gate, (math.degrees(az), math.degrees(nearest))
