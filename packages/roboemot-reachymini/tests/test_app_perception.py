"""Perception wired into the app (SPEC 4A, 5, 6, 12, 12A), against a mocked SDK client."""

from __future__ import annotations

import json
import shutil
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from roboemot.circadian import CircadianModel
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.phase import ConversationPhase
from roboemot.motion import MotionLoop
from roboemot.registry import Consent, FileStore, Registry
from roboemot.registry.enrollment import EnrollmentError
from roboemot.runtime import ticks
from roboemot.vision import CIRCADIAN, CONVERSATION, Detection, Frame
from roboemot.vision.service import Perception
from roboemot_reachymini import app as app_module

ROOT = Path(__file__).resolve().parents[3]
DIM = 8
MODEL = "mock-face"
PLANTED = np.eye(DIM, dtype=np.float32)[0]
CONSENT = Consent("Ada", datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"))


def fake_mini() -> MagicMock:
    mini = MagicMock()
    mini.client._is_alive = True
    mini.client._last_status = None
    # a robot that tracks its targets perfectly: the motion loop yields to a pose that does not
    # follow its commands (external control, SPEC 12A), so a static fake would silence it
    pose: dict = {"head": np.eye(4), "antennas": [0.0, 0.0], "body_yaw": 0.0}

    def set_target(head=None, antennas=None, body_yaw=None) -> None:
        if head is not None:
            pose["head"] = np.asarray(head)
        if antennas is not None:
            pose["antennas"] = [float(a) for a in antennas]
        if body_yaw is not None:
            pose["body_yaw"] = float(body_yaw)

    mini.set_target.side_effect = set_target
    mini.get_current_head_pose.side_effect = lambda: pose["head"]
    mini.get_current_joint_positions.side_effect = lambda: (
        [pose["body_yaw"], *[0.0] * 6],
        list(pose["antennas"]),
    )
    return mini


@pytest.fixture()
def data(tmp_path, monkeypatch):
    folder = tmp_path / "app-data"
    shutil.copytree(app_module.DATA, folder)
    shutil.copy2(ROOT / "configs/dev-unmeasured.yaml", folder / "app.yaml")
    monkeypatch.setattr(app_module, "DATA", folder)
    monkeypatch.setattr(app_module, "app_config_path", lambda: folder / "app.yaml")
    return folder


class FakeDetector:
    empty = False  # a test can empty the room without stopping the camera

    def detect(self, image):
        if self.empty:
            return []
        return [Detection((100, 60, 180, 140), 0.95, landmarks=np.zeros((5, 2)))]


class FakeEmbedder:
    model_id = MODEL

    def embed(self, image, face):
        return PLANTED.copy()


class PlantedSource:
    """A camera that sees one person: a new frame on every read."""

    def __init__(self):
        self.n = 0

    def camera_model(self, width, height):
        return None

    def read(self):
        self.n += 1
        img = np.zeros((240, 320, 3), np.uint8)
        img[0, :4, :] = self.n % 250
        img[5, :4, :] = (self.n // 250) % 250
        return Frame(img, time.monotonic(), None)


class EmptySource:
    def camera_model(self, width, height):
        return None

    def read(self):
        return None


class AppWithFakes(app_module.RoboEmotApp):
    source_class: type = PlantedSource

    def make_frame_source(self, mini, joint_state):
        return self.source_class()

    def load_vision_stages(self, cfg, profile):
        self.profile_seen = profile
        return SimpleNamespace(
            face_detector=FakeDetector(),
            face_embedder=FakeEmbedder(),
            attribute_estimator=None,
            appearance_embedder=None,
            object_detector=None,
            objects_unavailable=None,
        )


def new_app(cls=app_module.RoboEmotApp):
    return cls.__new__(cls)  # skip the SDK's daemon probe


def start(app) -> tuple[threading.Event, threading.Thread, list]:
    stop, errors = threading.Event(), []

    def target():
        try:
            app.run(fake_mini(), stop)
        except Exception as exc:  # surfaced by the test
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return stop, thread, errors


def wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_app_starts_without_models_and_reports_vision_unavailable(data, tmp_path):
    stop = threading.Event()
    stop.set()
    app = new_app()
    app.run(fake_mini(), stop)  # the real loader finds no models in the empty data folder

    sections = app.diagnostics.report()["sections"]
    vision = sections["roboemot.vision"]
    assert vision["ok"] and vision["data"]["state"] == "unavailable"
    assert vision["data"]["reason"]
    assert sections["roboemot.world"]["data"] == {"count": 0, "entries": []}
    assert sections["roboemot.registry"]["data"]["records"] == {}
    folder = tmp_path / "roboemot-data"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700 and (folder / "registry").is_dir()
    with pytest.raises(EnrollmentError, match="vision unavailable"):
        app.enrollment.start("s")
    assert app.enrollment is app.perception.enrollment


def test_app_with_fake_stages_names_an_enrolled_person_and_reports(data, tmp_path):
    reg = Registry(FileStore(tmp_path / "roboemot-data" / "registry", allow_plaintext=True),
                   RoboEmotConfig().identity)  # fmt: skip
    person = reg.enroll("Ada", [PLANTED * (1 + 0.01 * i) for i in range(5)], MODEL,
                        consent=CONSENT, age_group="adult")  # fmt: skip
    app = new_app(AppWithFakes)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.bus is not None)
        recognized = []
        app.bus.subscribe("person.recognized", lambda m: recognized.append(m.payload.person_id))
        assert wait_for(lambda: recognized), "the enrolled person was never recognised"
        assert wait_for(lambda: any(e.identity == person.person_id for e in app.world.entries))
        report = app.diagnostics.report()["sections"]
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
    assert recognized[0] == person.person_id
    assert app.profile_seen == "cm4"  # the manifest profile reaches load_stages
    vision = report["roboemot.vision"]["data"]
    assert vision["state"] == "running" and vision["frames"]["processed"] > 0
    world = report["roboemot.world"]["data"]
    assert world["count"] == 1 and world["entries"][0]["enrolled"] is True
    assert report["roboemot.registry"]["data"]["records"] == {"person": 1}
    text = json.dumps(report["roboemot.vision"]) + json.dumps(report["roboemot.world"])
    assert person.person_id not in text and "Ada" not in text and "embedding" not in text
    assert not [t for t in threading.enumerate() if t.name.startswith("roboemot-")]
    reloaded = Registry(FileStore(tmp_path / "roboemot-data" / "registry", allow_plaintext=True),
                        RoboEmotConfig().identity)  # fmt: skip
    assert reloaded.get_person(person.person_id).last_seen is not None  # flushed at shutdown


def test_app_auto_enrolls_a_cat_it_keeps_seeing_and_the_catalogue_sees_it(data, tmp_path):
    (data / "app.yaml").write_text(
        "schema_version: 1\n"
        "safety: {allow_unmeasured_dynamics: true}\n"
        "vision: {objects_enabled: true}\n"
        "world:\n"
        "  autoenroll: {min_days: 1, min_sightings: 3, sighting_min_interval_s: 1}\n"
    )
    app = new_app(AppWithObjects)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.perception is not None and app.perception.registry)
        reg = app.perception.registry
        assert wait_for(lambda: len(reg.entities()) == 2, timeout=30), "nothing was auto-enrolled"
        found = {r.category: r for r in reg.entities()}
        assert set(found) == {"cat", "potted plant"}  # the person is ignored
        assert {r.origin for r in found.values()} == {"auto"}
        assert found["cat"].display_name == "gatto 1"
        assert found["potted plant"].display_name == "pianta 1"  # the detector's display name
        cat = found["cat"]
        assert wait_for(lambda: any(e.identity == cat.entity_id for e in app.world.entries))
        assert sorted(s.display_name for s in app.catalogue.list_entities()) == [
            "gatto 1",
            "pianta 1",
        ]
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
    assert app.autoenroll.stats()["clusters"] == 0


def seed_candidate(store, key="left-behind"):
    meta = {
        "schema_version": 2,
        "cluster_id": key,
        "kind": "animal",
        "category": "cat",
        "model_id": "x",
    }
    store.put("candidates", key, meta, {"members": np.eye(2, DIM), "days": np.array([1, 2])})


def test_without_the_appearance_model_candidates_are_erased_at_startup_and_on_request(
    data, tmp_path
):
    store = FileStore(tmp_path / "roboemot-data" / "registry", allow_plaintext=True)
    seed_candidate(store)
    app = new_app(AppWithFakes)
    stop = threading.Event()
    stop.set()
    app.run(fake_mini(), stop)
    assert app.autoenroll is None
    assert store.keys("candidates") == []  # deleted at startup
    seed_candidate(app.perception.registry.store)
    assert app.catalogue.purge_candidates() == 1  # the endpoint's action needs no enroller
    assert app.perception.registry.store.keys("candidates") == []


def test_app_without_the_appearance_model_has_no_auto_enrollment(data):
    app = new_app(AppWithFakes)
    stop = threading.Event()
    stop.set()
    app.run(fake_mini(), stop)
    assert app.autoenroll is None and app.catalogue is not None


def test_circadian_state_scales_the_vision_rate(data):
    """The runner's model is what the governor reads: a hold on the app moves the rate."""
    app = new_app(AppWithFakes)
    app.source_class = EmptySource
    stop, thread, errors = start(app)
    want = RoboEmotConfig().circadian.states["resting"].vision_rate_scale
    try:
        assert wait_for(lambda: app.perception is not None and app.perception.pipeline is not None)
        assert wait_for(lambda: app.circadian_runner is not None and app.circadian is not None)
        assert isinstance(app.circadian, CircadianModel)
        governor = app.perception.pipeline.governor
        app.hold("resting", 5)
        assert wait_for(lambda: governor.scale == want)
        app.hold("awake", 5)
        assert wait_for(lambda: governor.scale == 1.0)
    finally:
        stop.set()
        thread.join(15)
    assert not errors and CIRCADIAN == "circadian"


def test_conversation_phase_scales_the_vision_rate(data):
    """SPEC 5, 8, 13: thinking or speaking lowers the vision rate; idle brings it back to 1."""
    app = new_app(AppWithFakes)
    app.source_class = EmptySource
    stop, thread, errors = start(app)
    want = RoboEmotConfig().vision.rates.conversation_scale
    try:
        assert wait_for(lambda: app.perception is not None and app.perception.pipeline is not None)
        governor = app.perception.pipeline.governor
        if app.phase is None:  # no voice stack with the fakes: the tick reads whatever is set
            app.phase = ConversationPhase()
        app.phase.begin_thinking()
        assert wait_for(lambda: governor.scales().get(CONVERSATION) == want)
        app.phase.end_thinking()
        assert wait_for(lambda: governor.scales().get(CONVERSATION) == 1.0)
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_circadian_scale_is_awake_while_no_model_runs():
    cfg = RoboEmotConfig()
    assert (
        ticks.circadian_vision_scale(cfg, None) == cfg.circadian.states["awake"].vision_rate_scale
    )


@pytest.fixture()
def stops(monkeypatch):
    """Record the order in which the app tears things down."""
    order: list[str] = []
    real_loop_stop = MotionLoop.stop

    def loop_stop(self):
        order.append("loop")
        real_loop_stop(self)

    monkeypatch.setattr(MotionLoop, "stop", loop_stop)
    return order


def test_the_motion_loop_stops_even_when_perception_fails_to_stop(data, monkeypatch, stops):
    def failing_stop(self):
        stops.append("perception")
        raise RuntimeError("camera stuck")

    monkeypatch.setattr(Perception, "stop", failing_stop)
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="camera stuck"):
        new_app().run(fake_mini(), stop)
    assert stops == ["perception", "loop"]  # the camera side is quiet before the SDK goes


def test_the_motion_loop_stops_when_perception_fails_to_start(data, monkeypatch, stops):
    def failing_start(self):
        raise OSError("no camera")

    monkeypatch.setattr(Perception, "start", failing_start)
    stop = threading.Event()
    stop.set()
    with pytest.raises(OSError, match="no camera"):
        new_app().run(fake_mini(), stop)
    assert stops == ["loop"]


def test_the_app_keeps_its_motion_when_the_data_folder_is_unusable(data, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a folder")
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(blocker / "data"))
    stop = threading.Event()
    stop.set()
    app = new_app(AppWithFakes)
    app.run(fake_mini(), stop)
    sections = app.diagnostics.report()["sections"]
    assert sections["roboemot.motion"]["ok"]
    assert sections["roboemot.vision"]["data"]["state"] == "unavailable"
    assert sections["roboemot.registry"]["data"]["state"] == "unavailable"
    assert "data folder" in sections["roboemot.registry"]["data"]["reason"]


class FakeObjects:
    """A detector that always sees a cat, a plant and a person (the person must be ignored)."""

    def detect(self, image):
        return [
            Detection((20, 100, 80, 160), 0.8, "cat"),
            Detection((200, 100, 260, 160), 0.8, "potted_plant"),
            Detection((120, 20, 180, 80), 0.9, "person"),
        ]


class FakeAppearance:
    model_id = "mock-appearance"

    def embed(self, image, box):
        return PLANTED.copy()


class AppWithObjects(AppWithFakes):
    objects_unavailable: str | None = None

    def load_vision_stages(self, cfg, profile):
        stages = super().load_vision_stages(cfg, profile)
        stages.appearance_embedder = FakeAppearance()
        stages.object_detector = None if self.objects_unavailable else FakeObjects()
        stages.objects_unavailable = self.objects_unavailable
        return stages


def test_an_animal_and_an_object_reach_the_world_model_through_the_app(data):
    app = new_app(AppWithObjects)
    stop, thread, errors = start(app)
    try:
        assert wait_for(
            lambda: (
                app.world is not None
                and {e.kind for e in app.world.entries} >= {"animal", "object"}
            )
        ), "the animal and the object never reached the world model"
        report = app.diagnostics.report()["sections"]
        entries = {e.kind: e for e in app.world.entries}
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
    assert entries["animal"].category == "cat"
    assert entries["object"].category == "potted plant"  # readable, not the class name
    assert "person" not in {e.category for e in app.world.entries if e.kind != "person"}
    assert report["roboemot.vision"]["data"]["objects"] == {"state": "running"}


def test_a_missing_object_detector_leaves_the_rest_of_vision_running(data):
    app = new_app(AppWithObjects)
    app.objects_unavailable = "ModelError: nanodet-plus-m-320: not verified"
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.perception is not None and app.perception.service is not None)
        assert wait_for(lambda: app.perception.service.runner.processed > 3)
        report = app.diagnostics.report()["sections"]
        kinds = {e.kind for e in app.world.entries}
    finally:
        stop.set()
        thread.join(15)
    assert not errors
    vision = report["roboemot.vision"]["data"]
    assert vision["state"] == "running"
    assert vision["objects"] == {"state": "unavailable", "reason": app.objects_unavailable}
    assert "animal" not in kinds and "object" not in kinds


def test_a_possible_minor_face_engages_the_motion_caps_and_releases_after_the_hold(data):
    (data / "app.yaml").write_text(
        "schema_version: 1\nsafety: {allow_unmeasured_dynamics: true, minor_hold_s: 1}\n"
    )
    FakeDetector.empty = False
    app = new_app(AppWithFakes)  # no attribute model: the face's age stays unknown (SPEC 5)
    stop, thread, errors = start(app)

    def caps() -> bool:
        return app.diagnostics.report()["sections"]["roboemot.motion"]["data"]["minor_present"]

    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert wait_for(caps), "an unknown-age face did not engage the minor caps"
        assert app.minor_present
        FakeDetector.empty = True  # frames keep coming, with nobody in them: positive evidence
        assert wait_for(lambda: not caps(), timeout=30), "the caps never released"
        assert not app.minor_present
    finally:
        FakeDetector.empty = False
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()


def test_the_minor_caps_stay_on_without_vision_and_when_frames_stop(data):
    (data / "app.yaml").write_text(
        "schema_version: 1\nsafety: {allow_unmeasured_dynamics: true, minor_hold_s: 1}\n"
    )
    app = new_app()  # the real loader finds no models: perception.pipeline is None
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.perception is not None and app.diagnostics is not None)
        assert app.perception.pipeline is None

        def caps() -> bool:
            sections = app.diagnostics.report()["sections"]
            return sections["roboemot.motion"]["data"]["minor_present"]

        assert wait_for(caps), "the caps were never engaged without vision"
        time.sleep(2)  # longer than the hold
        assert app.minor_present and caps()
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()

    app = new_app(AppWithFakes)
    app.source_class = EmptySource  # the camera gives no frames at all
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.perception is not None and app.perception.pipeline)
        time.sleep(2)
        assert app.minor_present
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_a_known_child_gives_joy_on_affect_state_and_in_the_debug_report(data):
    from roboemot.vision.types import PersonRecognized

    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    states: list = []
    try:
        assert wait_for(lambda: app.bus is not None and app.diagnostics is not None)
        app.bus.subscribe("affect.state", lambda m: states.append(m.payload))
        assert wait_for(lambda: states), "affect.state was never published"
        assert states[-1].valence == pytest.approx(0.0, abs=0.01)
        app.bus.publish(
            "person.recognized",
            PersonRecognized(1, "p-1", time.monotonic(), 0.0, 0.0, "child", True),
            time.monotonic(),
        )
        assert wait_for(lambda: states[-1].valence > 0.5), "no positive valence after the child"
        assert states[-1].label == "joy"
        report = app.diagnostics.report()["sections"]["roboemot.app"]["data"]
        assert report["valence"] > 0 and report["label"] in app.affect_runner.engine.cfg.labels
        assert "p-1" not in json.dumps(report)
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
