"""M2 exit (SPEC 14), app level: a known adult, a known child and a stranger each walk up to the
robot and it reacts distinctly, with no LLM. Mocked SDK client; the same probe as the simulated
robot test (``test_sim.py``), reading the driver's commanded targets, not the robot's state."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from exit_support import BODY_MAP_YAML, Run, check_exit, format_run, react
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.registry import Consent, FileStore, Registry
from roboemot.vision import CameraModel, Detection
from roboemot_reachymini import app as app_module
from roboemot_reachymini.driver import ReachyMiniDriver
from test_app_perception import (  # noqa: F401  (data is a fixture)
    CONSENT,
    MODEL,
    PLANTED,
    ROOT,
    FakeEmbedder,
    PlantedSource,
    data,
    new_app,
    start,
    wait_for,
)

GUARDIAN = Consent("Mum", datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"), guardian="Mum")


class Room:
    """The planted person is in the frame only when ``present`` is set."""

    present = False


class RoomDetector:
    def detect(self, image):
        if not Room.present:
            return []
        return [Detection((100, 60, 180, 140), 0.95, landmarks=np.zeros((5, 2)))]


class ExitApp(app_module.RoboEmotApp):
    def make_frame_source(self, mini, joint_state):
        return PlantedSource()

    def load_vision_stages(self, cfg, profile):
        return SimpleNamespace(
            face_detector=RoomDetector(),
            face_embedder=FakeEmbedder(),
            attribute_estimator=None,
            appearance_embedder=None,
            object_detector=None,
            objects_unavailable=None,
        )


@pytest.fixture()
def commanded(monkeypatch):
    """The latest pose the motion loop wrote to the driver (after the limiter)."""
    last: dict[str, float] = {}
    real = ReachyMiniDriver.write_pose_targets

    def write(self, t, values):
        last.update(values)
        real(self, t, values)

    monkeypatch.setattr(ReachyMiniDriver, "write_pose_targets", write)
    return last


def run_case(data, tmp_path, monkeypatch, commanded, case: str) -> Run:  # noqa: F811
    """One fresh app and registry per person: Ada the adult, a child, or nobody enrolled."""
    (data / "app.yaml").write_text(
        (ROOT / "configs/dev-unmeasured.yaml").read_text() + BODY_MAP_YAML
    )
    root = tmp_path / case
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(root))
    if case != "stranger":
        reg = Registry(
            FileStore(root / "registry", allow_plaintext=True), RoboEmotConfig().identity
        )
        reg.enroll(
            "Ada" if case == "adult" else "Kid",
            [PLANTED * (1 + 0.01 * i) for i in range(5)],
            MODEL,
            consent=CONSENT if case == "adult" else GUARDIAN,
            age_group=case,
        )
    manifest = load_manifest(
        app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
    )
    home = {n: d.home for n, d in manifest.dofs.items()}
    Room.present = False
    app = new_app(ExitApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None, 30)
        result = react(
            app, lambda: dict(commanded), lambda: setattr(Room, "present", True), case, home
        )
    finally:
        stop.set()
        thread.join(20)
    assert not errors and not thread.is_alive()
    print(format_run(result))
    return result


def test_the_robot_reacts_distinctly_to_an_adult_a_child_and_a_stranger_without_an_llm(
    data,  # noqa: F811
    tmp_path,
    monkeypatch,
    commanded,
):
    runs = {
        c: run_case(data, tmp_path, monkeypatch, commanded, c)
        for c in ("adult", "child", "stranger")
    }
    # the fake detector's face sits at pixel (140, 100): the commanded head yaw is its bearing,
    # so unlike the simulated servos there is nothing to overshoot with
    cfg = RoboEmotConfig()
    bearing = CameraModel(320, 240, cfg.world.camera_fov_h_deg).bearing(140, 100)[0]
    check_exit(runs["adult"], runs["child"], runs["stranger"], abs(bearing) + math.radians(1))
