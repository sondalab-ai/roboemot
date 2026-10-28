"""Reachy Mini driver against a fake SDK object (no robot, no simulator)."""

from __future__ import annotations

import filecmp
import math
from pathlib import Path

import numpy as np
import pytest
from roboemot.config import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionLoop, Priority
from roboemot_reachymini import ReachyMiniDriver, head_matrix, head_values

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "packages/roboemot-reachymini/src/roboemot_reachymini/data"


class FakeClient:
    """The SDK's WSClient surface the driver relies on."""

    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self.host, self.port = host, port
        self._is_alive = True
        self.disconnected = False
        self.fail_connect = False
        self._last_head_pose: object = object()  # replaced on every message, as the SDK does
        self._last_joint_positions: object = object()

    def wait_for_connection(self, timeout: float = 5.0) -> None:
        if self.fail_connect:
            raise TimeoutError("Timeout while waiting for connection with the server.")

    def disconnect(self) -> None:
        self.disconnected = True


class FakeMini:
    def __init__(self) -> None:
        self.client = FakeClient()
        self.calls: list[tuple[str, dict]] = []
        self.auto_body_yaw = True
        self.pose = np.eye(4)
        self.joints = ([0.1, 0, 0, 0, 0, 0, 0], [-0.17, 0.17])

    def set_automatic_body_yaw(self, enabled: bool) -> None:
        self.auto_body_yaw = enabled

    def set_target(self, **kw) -> None:
        self.calls.append(("set_target", kw))
        # tracks perfectly once told where to go (a static pose reads as external control)
        if "head" in kw:
            self.pose = np.asarray(kw["head"])
        joints = list(self.joints[0])
        if "body_yaw" in kw:
            joints[0] = kw["body_yaw"]
        antennas = list(kw["antennas"]) if "antennas" in kw else self.joints[1]
        self.joints = (joints, antennas)
        self.client._last_head_pose, self.client._last_joint_positions = object(), object()

    def goto_target(self, **kw) -> None:  # must never be called (blocking in SDK 1.11)
        raise AssertionError("goto_target used")

    def enable_motors(self) -> None:
        self.calls.append(("enable_motors", {}))

    def disable_motors(self) -> None:
        self.calls.append(("disable_motors", {}))

    def enable_gravity_compensation(self) -> None:
        self.calls.append(("gravity_on", {}))

    def disable_gravity_compensation(self) -> None:
        self.calls.append(("gravity_off", {}))

    def get_current_head_pose(self):
        return self.pose

    def get_current_joint_positions(self):
        return self.joints


def test_head_pose_roundtrip():
    vals = {
        "head.roll": 0.1,
        "head.pitch": -0.2,
        "head.yaw": 0.3,
        "head.x": 0.001,
        "head.y": -0.002,
        "head.z": 0.005,
    }
    back = head_values(head_matrix(vals))
    for k, v in vals.items():
        assert back[k] == pytest.approx(v, abs=1e-12)


def test_driver_disables_auto_body_yaw_and_orders_antennas():
    mini = FakeMini()
    drv = ReachyMiniDriver(mini)
    assert mini.auto_body_yaw is False
    drv.write_pose_targets(0.0, {"antenna_right": -0.5, "antenna_left": 0.5})
    assert mini.calls == []  # not active yet: nothing streamed
    drv.set_mode("active")
    drv.write_pose_targets(
        0.0, {"head.yaw": 0.2, "antenna_right": -0.5, "antenna_left": 0.5, "body_yaw": 0.1}
    )
    name, kw = mini.calls[-1]
    assert name == "set_target" and kw["antennas"] == [-0.5, 0.5] and kw["body_yaw"] == 0.1
    assert mini.calls[0][0] == "enable_motors"  # enable precedes any set_target


def test_teach_mode_switches_gravity_compensation():
    mini = FakeMini()
    drv = ReachyMiniDriver(mini)
    drv.set_mode("teach")
    drv.set_mode("active")
    assert [c[0] for c in mini.calls] == ["gravity_on", "gravity_off", "enable_motors"]


def test_read_state_maps_joints():
    st = ReachyMiniDriver(FakeMini()).read_state()
    assert st["body_yaw"] == 0.1 and st["antenna_right"] == -0.17 and st["head.yaw"] == 0


@pytest.mark.parametrize("head_yaw", [-3.1435, -math.pi, 3.1435, math.pi, -2.6, 2.9])
def test_read_state_head_yaw_stays_next_to_the_body_at_the_back(head_yaw):
    """Robot run 2026-10-28: with the body at -130 deg and the head at -180 deg, the pose's yaw
    read +pi; the 2*pi jump looked like a hand on the head and froze the motion loop. The head
    yaw is read within pi of the body yaw, where the head always is (head-body delta 65 deg)."""
    mini = FakeMini()
    body = math.copysign(2.267, head_yaw)
    mini.pose = head_matrix({"head.yaw": head_yaw})
    mini.joints = ([body, 0, 0, 0, 0, 0, 0], [-0.17, 0.17])
    st = ReachyMiniDriver(mini).read_state()
    assert st["head.yaw"] == pytest.approx(head_yaw, abs=1e-9)
    assert abs(st["head.yaw"] - st["body_yaw"]) <= math.pi


def test_motion_loop_streams_limited_targets_from_measured_pose():
    mini = FakeMini()
    cfg = ConfigStack().effective()
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    loop = MotionLoop(manifest, ReachyMiniDriver(mini), cfg, rate_hz=50)
    loop.set_mode("active")
    t = 0.0
    for _ in range(100):
        loop.feed(Command("app", Priority.COMMANDED, {"head.yaw": 0.5}, t))
        loop.step_once(t, 0.02)
        t += 0.02
    targets = [kw for name, kw in mini.calls if name == "set_target"]
    assert len(targets) == 100
    yaws = [head_values(kw["head"])["head.yaw"] for kw in targets]
    vmax = manifest.dofs["head.yaw"].max_vel
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:], strict=False)) / 0.02 <= vmax + 1e-9
    # started from the measured pose (0.1), then eases home with no jump
    assert targets[0]["body_yaw"] == pytest.approx(0.1, abs=0.01)
    assert yaws[-1] == pytest.approx(0.5, abs=1e-6)
    assert not math.isnan(yaws[-1])


def test_package_data_in_sync_with_configs():
    assert filecmp.cmp(
        DATA / "reachy-mini-wireless.yaml",
        REPO / "configs/manifests/reachy-mini-wireless.yaml",
        shallow=False,
    )
    for p in (REPO / "configs/presets").glob("*.yaml"):
        assert filecmp.cmp(DATA / "presets" / p.name, p, shallow=False), (
            "run scripts/sync-package-data.sh"
        )


def test_return_from_teach_mode_does_not_jump():
    """Review finding 5: body moved by hand in teach mode, then motors re-enabled."""
    mini = FakeMini()
    cfg = ConfigStack().effective()
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    loop = MotionLoop(manifest, ReachyMiniDriver(mini), cfg, rate_hz=50)
    loop.set_mode("active")
    t = 0.0
    for _ in range(50):
        loop.feed(Command("app", Priority.COMMANDED, {"body_yaw": 0.0}, t))
        loop.step_once(t, 0.02)
        t += 0.02
    loop.set_mode("teach")
    mini.joints = ([-1.0, 0, 0, 0, 0, 0, 0], [-0.17, 0.17])  # moved by hand
    for _ in range(10):
        loop.step_once(t, 0.02)  # tracking, no writes
        t += 0.02
    n_before = len([c for c in mini.calls if c[0] == "set_target"])
    loop.set_mode("active")
    loop.feed(Command("app", Priority.COMMANDED, {"body_yaw": 0.0}, t))
    loop.step_once(t, 0.02)
    first = [kw for name, kw in mini.calls if name == "set_target"][n_before]
    vmax = manifest.dofs["body_yaw"].max_vel
    assert abs(first["body_yaw"] - (-1.0)) <= vmax * 0.02 + 1e-9


def test_loop_survives_bad_tick_and_applies_config():
    mini = FakeMini()
    stack = ConfigStack()
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    loop = MotionLoop(manifest, ReachyMiniDriver(mini), stack.effective(), rate_hz=50)
    stack.subscribe(lambda ch: loop.apply_config(ch.config))
    stack.set("safety.watchdog_return_factor", 0.2)
    assert loop.core.watchdog_return_factor == 0.2
    assert loop.core.watchdog_s <= manifest.model.safety.watchdog_ms / 1000
    loop.set_mode("active")
    with pytest.raises(ValueError):
        loop.step_once(0.0, 0.0)  # step_once raises; the thread wrapper counts and continues


def test_read_state_raises_on_dead_connection():
    """SPEC 12A: the SDK serves cached state on a dead link; the driver must not."""
    mini = FakeMini()
    driver = ReachyMiniDriver(mini)
    mini.client._is_alive = False
    with pytest.raises(ConnectionError):
        driver.read_state()


# The SDK is installed only in the sim job (`-m sim`); without the mark these never ran in CI.
@pytest.mark.sim
def test_reconnect_swaps_client_and_restores_body_yaw(monkeypatch):
    ws = pytest.importorskip("reachy_mini.io.ws_client")

    made: list[FakeClient] = []

    def factory(host, port):
        made.append(FakeClient(host, port))
        return made[-1]

    monkeypatch.setattr(ws, "WSClient", factory)
    mini = FakeMini()
    old = mini.client
    driver = ReachyMiniDriver(mini)
    mini.auto_body_yaw = True  # the restarted daemon is back to its default
    driver.reconnect(0.5)
    assert mini.client is made[0] and old.disconnected and not mini.auto_body_yaw
    assert (made[0].host, made[0].port) == ("localhost", 8000)


# The SDK is installed only in the sim job (`-m sim`); without the mark these never ran in CI.
@pytest.mark.sim
def test_reconnect_failure_keeps_old_client(monkeypatch):
    ws = pytest.importorskip("reachy_mini.io.ws_client")

    def factory(host, port):
        c = FakeClient(host, port)
        c.fail_connect = True
        return c

    monkeypatch.setattr(ws, "WSClient", factory)
    mini = FakeMini()
    old = mini.client
    with pytest.raises(TimeoutError):
        ReachyMiniDriver(mini).reconnect(0.5)
    assert mini.client is old and not old.disconnected


# ---------------------------------------------------------------- diagnostics (SPEC 12A)


def test_diagnostics_sources_read_the_daemon_api():
    import httpx
    from roboemot.diagnostics import Target, collect, platform_sources
    from roboemot_reachymini import diagnostics
    from roboemot_reachymini.daemon import DaemonClient

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        if request.url.path == "/api/apps/current-app-status":
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"path": request.url.path})

    client = DaemonClient(Target("robot"), transport=httpx.MockTransport(handler))
    report = collect(diagnostics.sources(Target("robot"), client))
    s = report["sections"]
    assert s["robot.daemon"]["data"] == {"path": "/api/daemon/status", "hardware_id_present": False}
    assert s["robot.motors"]["ok"] and s["robot.state"]["ok"]
    assert s["robot.app"]["ok"] is False and "500" in s["robot.app"]["error"]
    state_params = dict(seen)["/api/state/full"]
    assert state_params["with_imu"] == "true"
    assert {sec["scope"] for sec in s.values()} == {"robot"}
    # registered under the manifest id (entry point installed with the package)
    names = [src.name for src in platform_sources("reachy-mini-wireless", Target())]
    assert names == ["robot.daemon", "robot.motors", "robot.state", "robot.app", "robot.wifi"]


class StatusClient(FakeClient):
    """A client whose daemon status carries the owner's name for the robot."""

    def __init__(self, robot_name, status_arrived=True, error=None) -> None:
        super().__init__()
        self._last_status = object() if status_arrived else None
        self._name, self._error = robot_name, error

    def get_status(self, wait: bool = True):
        if self._error:
            raise self._error
        return type("Status", (), {"robot_name": self._name})()


@pytest.mark.parametrize(
    "client,expected",
    [
        (StatusClient("Neve"), "Neve"),
        (StatusClient("  reachy_mini "), "reachy_mini"),
        (StatusClient("", True), None),
        (StatusClient(None), None),
        (StatusClient(42), None),
        (StatusClient("Neve", status_arrived=False), None),
        (StatusClient("Neve", error=ConnectionError("lost")), None),
    ],
)
def test_robot_name_is_the_daemon_status_name_and_never_raises(client, expected):
    mini = FakeMini()
    mini.client = client
    assert ReachyMiniDriver(mini).robot_name() == expected


# ---------------------------------------------------------------- review R1 (B1, S2, S4)


def test_head_pose_conversions_equal_scipy_over_random_poses():
    """The closed forms replace scipy's Rotation on the 50 Hz path; same values (SDK order xyz)."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    for _ in range(2000):
        rpy = rng.uniform([-math.pi, -1.5, -math.pi], [math.pi, 1.5, math.pi])
        vals = dict(zip(("head.roll", "head.pitch", "head.yaw"), rpy, strict=True))
        vals.update(zip(("head.x", "head.y", "head.z"), rng.uniform(-0.05, 0.05, 3), strict=True))
        m = head_matrix(vals)
        np.testing.assert_allclose(
            m[:3, :3], Rotation.from_euler("xyz", rpy).as_matrix(), rtol=0, atol=1e-12
        )
        ref = Rotation.from_matrix(m[:3, :3]).as_euler("xyz")
        got = head_values(m)
        for k, v in zip(("head.roll", "head.pitch", "head.yaw"), ref, strict=True):
            assert got[k] == pytest.approx(float(v), abs=1e-9)
        assert got["head.z"] == vals["head.z"]


def test_head_values_of_a_slightly_skewed_matrix_stay_close_to_scipy():
    """The daemon's pose comes from its kinematics in floats: scipy projects onto the nearest
    rotation first, the closed form reads the entries; a 1e-6 skew moves them by about as much."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(11)
    for _ in range(500):
        rpy = rng.uniform([-0.7, -0.7, -math.pi], [0.7, 0.7, math.pi])
        m = np.eye(4)
        m[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix() + rng.normal(0, 1e-6, (3, 3))
        ref = Rotation.from_matrix(m[:3, :3]).as_euler("xyz")
        got = head_values(m)
        for k, v in zip(("head.roll", "head.pitch", "head.yaw"), ref, strict=True):
            assert got[k] == pytest.approx(float(v), abs=1e-5)


def test_state_seq_counts_new_messages_only():
    mini = FakeMini()
    driver = ReachyMiniDriver(mini)
    first = driver.state_seq()
    assert driver.state_seq() == first  # nothing arrived: a repeat
    mini.client._last_head_pose = object()
    assert driver.state_seq() == first  # the joints are still the old message
    mini.client._last_joint_positions = object()
    second = driver.state_seq()
    assert second != first and driver.state_seq() == second


def test_state_seq_raises_on_dead_connection():
    mini = FakeMini()
    mini.client._is_alive = False
    with pytest.raises(ConnectionError):
        ReachyMiniDriver(mini).state_seq()


def test_motors_enabled_does_not_consume_the_sdk_status_event():
    """``get_status(wait=False)`` clears ``status_received``, which the SDK's own waits rely on
    (review R1 S4): the driver reads the cached status without it."""
    import threading
    from types import SimpleNamespace

    mini = FakeMini()
    mini.client._last_status = SimpleNamespace(
        backend_status=SimpleNamespace(motor_control_mode="enabled")
    )
    mini.client.status_received = threading.Event()
    mini.client.status_received.set()
    mini.client.get_status = lambda wait=True, timeout=5.0: pytest.fail("get_status called")
    assert ReachyMiniDriver(mini).motors_enabled() is True
    assert mini.client.status_received.is_set()


STATUS_BUILD_S = 0.3  # the status thread builds the status before its 1 s sleep (loaded CM4)
MIN_JERK_TAIL_S = 0.5  # a short min-jerk goto reads as still this long before it ends
GRACE_MARGIN_S = 0.5  # kept in hand for scheduling on a saturated CPU


def test_grace_outlasts_the_sleep_wait_a_slow_status_and_old_state():
    """SPEC 4A, 12A: motors-off comes ``SLEEP_MOTORS_OFF_DELAY_S`` after the sleep move and is
    in the status up to a period (plus its build time) later, through a queue up to
    ``max_state_age_s`` old, while stillness may start in the move's min-jerk tail. The pose
    comes through the same queue as the status, so that age delays both alike and cancels."""
    from roboemot_reachymini.driver import SLEEP_MOTORS_OFF_DELAY_S

    grace = ConfigStack().effective().motion.platform_move_grace_s
    status = ReachyMiniDriver.status_period_s + STATUS_BUILD_S
    assert grace > SLEEP_MOTORS_OFF_DELAY_S + status + MIN_JERK_TAIL_S + GRACE_MARGIN_S


def test_head_values_round_trip_near_gimbal_lock():
    rng = np.random.default_rng(3)
    for pitch in (math.pi / 2 - 1e-9, -math.pi / 2 + 1e-9, math.pi / 2 - 1e-4):
        vals = {
            "head.roll": rng.uniform(-1, 1),
            "head.pitch": pitch,
            "head.yaw": rng.uniform(-1, 1),
        }
        m = head_matrix(vals)
        np.testing.assert_allclose(head_matrix(head_values(m)), m, atol=1e-8)
