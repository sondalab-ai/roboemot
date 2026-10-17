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
        # The new client reports None until its first daemon status arrives.
        deadline = time.monotonic() + 5
        while loop.driver.motors_enabled() is None and time.monotonic() < deadline:
            time.sleep(0.1)
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


PLANTED_AZ = math.radians(25)


class Scene:
    """The whole app on the simulated robot, with ONNX faked at the stage boundary.

    Only the detector, the embedder and the camera are stand-ins: the real pump thread,
    tracker, matcher, registry, world model, attention and motion run. The camera draws a face
    where the measured head pose sees the planted person at ``PLANTED_AZ`` (nothing when she is
    out of view or when ``plant`` is false); the embedder returns her enrolled embedding.
    """

    def __init__(self, mini, tmp_path, monkeypatch, *, enroll: bool, plant: bool):
        import shutil
        from datetime import UTC, datetime
        from types import SimpleNamespace

        import numpy as np
        from roboemot.config.schema import RoboEmotConfig
        from roboemot.registry import Consent, FileStore, Registry
        from roboemot.vision import CameraModel, Detection, Frame
        from roboemot.world import from_base_frame
        from roboemot_reachymini import app as app_module
        from roboemot_reachymini.gaze import FRAME

        root = Path(__file__).resolve().parents[3]
        data = tmp_path / "app-data"
        shutil.copytree(DATA, data)
        shutil.copy2(root / "configs/dev-unmeasured.yaml", data / "app.yaml")
        monkeypatch.setattr(app_module, "DATA", data)
        cfg = RoboEmotConfig()
        face_vec = np.eye(8, dtype=np.float32)[0]
        self.ada_id = None
        if enroll:
            reg = Registry(
                FileStore(tmp_path / "roboemot-data" / "registry", allow_plaintext=True),
                cfg.identity,
            )
            self.ada_id = reg.enroll(
                "Ada",
                [face_vec * (1 + 0.01 * i) for i in range(5)],
                "sim-face",
                consent=Consent("Ada", datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh")),
                age_group="adult",
            ).person_id
        width, height, box = 320, 240, 80
        camera = CameraModel(
            width, height, cfg.world.camera_fov_h_deg, vfov_deg=cfg.world.camera_fov_v_deg
        )
        half_h = math.radians(cfg.world.camera_fov_h_deg) / 2 * 0.9
        half_v = math.radians(cfg.world.camera_fov_v_deg) / 2 * 0.9

        class PlantedCamera:
            def __init__(self, joint_state):
                self.joint_state, self.n = joint_state, 0

            def read(self):
                self.n += 1
                state = dict(self.joint_state())
                az, el = from_base_frame(PLANTED_AZ, 0.0, state, FRAME)
                img = np.zeros((height, width, 3), np.uint8)
                img[0, 0, 1], img[0, 0, 2] = self.n % 250, (self.n // 250) % 250  # never repeats
                if plant and abs(az) <= half_h and abs(el) <= half_v:
                    y = -math.tan(az)
                    u = width / 2 - camera.focal_px * math.tan(az)
                    v = height / 2 - camera.focal_y_px * math.tan(el) * math.sqrt(1 + y * y)
                    img[0, 0, 0] = 255
                    img[0, 1, 0], img[0, 1, 1] = int(u) // 256, int(u) % 256
                    img[1, 1, 0], img[1, 1, 1] = int(v) // 256, int(v) % 256
                return Frame(img, time.monotonic(), state)

        class PlantedDetector:
            def detect(self, image):
                if image[0, 0, 0] != 255:
                    return []
                u = int(image[0, 1, 0]) * 256 + int(image[0, 1, 1])
                v = int(image[1, 1, 0]) * 256 + int(image[1, 1, 1])
                h = box / 2
                return [Detection((u - h, v - h, u + h, v + h), 0.95, landmarks=np.zeros((5, 2)))]

        class PlantedEmbedder:
            model_id = "sim-face"

            def embed(self, image, face):
                return face_vec.copy()

        class SimApp(app_module.RoboEmotApp):
            def make_frame_source(self, mini_, joint_state):
                return PlantedCamera(joint_state)

            def load_vision_stages(self, cfg_, profile):
                return SimpleNamespace(
                    face_detector=PlantedDetector(),
                    face_embedder=PlantedEmbedder(),
                    attribute_estimator=None,
                    appearance_embedder=None,
                    object_detector=None,
                    objects_unavailable=None,
                )

        self.mini, self.app = mini, SimApp.__new__(SimApp)
        self.stop_event, self.errors, self.recognized = threading.Event(), [], []
        self.reader = ReachyMiniDriver(mini)
        self.thread = threading.Thread(target=self._run, name="sim-app", daemon=True)

    def _run(self) -> None:
        try:
            self.app.run(self.mini, self.stop_event)
        except Exception as exc:
            self.errors.append(exc)

    def yaw(self) -> float:
        return self.reader.read_state()["head.yaw"]

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 60
        # diagnostics is set last in RoboEmotApp.run, so the app is fully started once it exists.
        while self.app.diagnostics is None and not self.errors and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not self.errors, self.errors
        assert self.app.diagnostics is not None, "the app did not start within 60 s"
        self.app.bus.subscribe(
            "person.recognized", lambda m: self.recognized.append(m.payload.person_id)
        )
        return self

    def __exit__(self, *exc) -> None:
        self.stop_event.set()
        self.thread.join(30)
        assert not self.errors, self.errors
        assert not self.thread.is_alive(), "the app did not stop within 30 s"


def test_robot_turns_to_an_enrolled_person_and_names_them(mini, tmp_path, monkeypatch):
    """SPEC 5, 6, 10 (M1 exit): the app against the simulated robot; Ada is enrolled and
    planted 25 deg to its left (see ``Scene``: only ONNX is faked, at the stage boundary).
    The head turns to her and ``person.recognized`` names her."""
    with Scene(mini, tmp_path, monkeypatch, enroll=True, plant=True) as scene:
        deadline, yaw = time.monotonic() + 60, 0.0
        while time.monotonic() < deadline:
            yaw = scene.yaw()
            if scene.recognized and yaw > 0.9 * PLANTED_AZ:
                break
            time.sleep(0.1)
        app = scene.app
        entries = [e for e in app.world.entries if e.kind == "person"]
        sections = app.diagnostics.report()["sections"]
        name = app.perception.registry.get_person(scene.recognized[0]).display_name
    assert scene.recognized[0] == scene.ada_id and name == "Ada"
    assert yaw > 0.9 * PLANTED_AZ, math.degrees(yaw)  # turned left, to her
    assert [e.identity for e in entries] == [scene.ada_id]  # one entry, named
    assert sections["roboemot.vision"]["data"]["frames"]["processed"] > 0


def test_nobody_in_view_the_robot_stays_put_and_names_nobody(mini, tmp_path, monkeypatch):
    """Negative control for the exit test: Ada is enrolled but not in the room."""
    with Scene(mini, tmp_path, monkeypatch, enroll=True, plant=False) as scene:
        time.sleep(8)
        yaw = scene.yaw()
        persons = [e for e in scene.app.world.entries if e.kind == "person"]
        target = scene.app.selector.target
        processed = scene.app.diagnostics.report()["sections"]["roboemot.vision"]["data"]
    assert processed["frames"]["processed"] > 0  # vision ran, it just saw nobody
    assert scene.recognized == [] and persons == []
    assert target is None or target.kind != "person"
    assert abs(yaw) < 0.4 * PLANTED_AZ, math.degrees(yaw)


def test_an_unenrolled_face_is_followed_but_never_named(mini, tmp_path, monkeypatch):
    """The same face with nobody enrolled: tracked as a person entry, identity unknown."""
    with Scene(mini, tmp_path, monkeypatch, enroll=False, plant=True) as scene:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if any(e.kind == "person" for e in scene.app.world.entries):
                break
            time.sleep(0.1)
        time.sleep(3)  # long enough for recognition, were she enrolled
        persons = [e for e in scene.app.world.entries if e.kind == "person"]
    assert persons and all(e.identity == "unknown" for e in persons)
    assert scene.recognized == []


def test_an_unknown_age_face_engages_the_minor_caps(mini, tmp_path, monkeypatch):
    """SPEC 5, 12: no attribute model, so the planted face's age is unknown (a possible minor);
    the motion loop's minor caps engage."""
    with Scene(mini, tmp_path, monkeypatch, enroll=False, plant=True) as scene:
        deadline, engaged = time.monotonic() + 30, False
        while time.monotonic() < deadline and not engaged:
            motion = scene.app.diagnostics.report()["sections"]["roboemot.motion"]["data"]
            engaged = motion["minor_present"]
            time.sleep(0.1)
    assert engaged
