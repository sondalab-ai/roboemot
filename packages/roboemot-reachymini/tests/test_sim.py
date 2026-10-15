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
        ticks = loop.ticks
        time.sleep(1.0)
        assert loop.ticks > ticks  # streaming again
    finally:
        stop.set()
        loop.stop()
