"""SPEC 4A, 5, 6, 12, 12A: perception in an app (``roboemot.vision.service``)."""

from __future__ import annotations

import json
import logging
import math
import stat
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.registry import Consent, FileStore, Registry
from roboemot.registry.enrollment import EnrollmentError
from roboemot.vision import Detection, Frame, LatestFrameRunner
from roboemot.vision.models import ModelError
from roboemot.vision.service import open_perception, world_snapshot
from roboemot.world import Entry

CFG = RoboEmotConfig()
DIM = 8
MODEL = "mock-face"
NAME = "Grace Hopper Private"
CONSENT = Consent(NAME, datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"))
PLANTED = np.eye(DIM, dtype=np.float32)[0]


class Detector:
    """One face whose nose swings left and right, so enrollment sees varied angles."""

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        eye_l, eye_r, y = 120.0, 160.0, 90.0
        nose = 140.0 + 0.3 * (eye_r - eye_l) * math.sin(self.calls / 3)
        pts = [[eye_l, y], [eye_r, y], [nose, y + 20], [eye_l, y + 40], [eye_r, y + 40]]
        return [Detection((100, 60, 180, 140), 0.95, landmarks=np.array(pts))]


class Embedder:
    model_id = MODEL

    def embed(self, image, face):
        return PLANTED.copy()


class Source:
    """A camera that returns a different frame on every read (``pump`` skips repeats)."""

    def __init__(self):
        self.n = 0

    def read(self):
        self.n += 1
        img = np.zeros((240, 320, 3), np.uint8)
        img[0, :4, :] = self.n % 250
        img[5, :4, :] = (self.n // 250) % 250
        return Frame(img, time.monotonic(), {})


def fake_stages(cfg, profile):
    return SimpleNamespace(
        face_detector=Detector(),
        face_embedder=Embedder(),
        appearance_embedder=None,
        object_detector=None,
        objects_unavailable=None,
    )


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("ROBOEMOT_MODELS_DIR", raising=False)
    return tmp_path / "data"


def test_registry_lives_under_a_private_data_folder(data_dir):
    open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    assert (data_dir / "registry").is_dir()
    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700


def test_startup_warns_that_the_registry_is_plaintext(caplog):
    with caplog.at_level(logging.WARNING):
        open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    assert "without encryption (pre-M5" in caplog.text


def test_missing_models_run_without_vision_and_say_why(caplog):
    def broken(cfg, profile):
        raise ModelError("yunet: missing, run `roboemot models fetch`")

    with caplog.at_level(logging.WARNING):
        p = open_perception(CFG, Bus(), Source(), load_stages=broken)
    assert p.pipeline is None and p.service is None
    p.start()
    p.stop()  # harmless
    assert p.vision_snapshot() == {"state": "unavailable", "reason": p.unavailable}
    assert "ModelError" in p.unavailable and "yunet" in p.unavailable
    assert sum("vision unavailable" in r.message for r in caplog.records) == 1
    with pytest.raises(EnrollmentError, match="vision unavailable"):
        p.enrollment.start("s")
    assert p.registry.list() == []  # the registry still works


def test_real_loader_without_models_is_unavailable_not_a_crash(tmp_path):
    p = open_perception(CFG, Bus(), Source())  # nothing fetched in the empty data folder
    assert p.pipeline is None and p.unavailable


def test_frames_flow_to_percepts_and_the_enrolled_person_is_named(data_dir):
    bus = Bus()
    p = open_perception(CFG, bus, Source(), load_stages=fake_stages)
    person = p.registry.enroll(
        NAME,
        [PLANTED * (1 + 0.01 * i) for i in range(5)],
        MODEL,
        consent=CONSENT,
    )
    names, entities = [], []
    bus.subscribe("person.recognized", lambda m: names.append(m.payload.person_id))
    bus.subscribe("percept.entity", lambda m: entities.append(m.payload))
    p.start()
    try:
        deadline = time.monotonic() + 15
        while not names and time.monotonic() < deadline:
            time.sleep(0.05)
        snap = p.vision_snapshot()
    finally:
        p.stop()
    assert names and names[0] == person.person_id
    assert any(e.identity == person.person_id for e in entities)
    assert snap["state"] == "running" and snap["camera"]["frames"] > 0
    assert snap["frames"]["processed"] > 0 and snap["frames"]["last_error"] is None
    assert snap["face_tracks"] == 1 and snap["rate_scale"] == 1.0
    # shutdown flushed the buffered last_seen to disk
    reloaded = Registry(FileStore(data_dir / "registry", allow_plaintext=True), CFG.identity)
    assert reloaded.get_person(person.person_id).last_seen is not None
    assert not [t for t in threading.enumerate() if t.name.startswith("roboemot-")]


def test_diagnostics_sections_carry_no_embeddings_or_names():
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    p.registry.enroll(NAME, [PLANTED] * 5, MODEL, consent=CONSENT)
    p.registry.enroll_entity("animal", "Rex Private", [PLANTED], "dino", category="dog")
    entry = Entry("e1", "person", "person", "some-registry-id", 0.5, 0.1, None, 0.1, 0.9, 1.0, 3.0)
    entry.embedding = PLANTED
    payloads = [
        p.registry_snapshot(),
        p.vision_snapshot(),
        world_snapshot([entry], now=5.0),
    ]
    text = json.dumps(payloads)
    for secret in (NAME, "Rex Private", "some-registry-id", "embedding"):
        assert secret not in text, secret
    reg = payloads[0]
    assert reg["records"] == {"person": 1, "animal": 1} and reg["quarantined"] == 0
    (row,) = payloads[2]["entries"]
    assert (
        row["enrolled"] is True
        and row["age_s"] == 2.0
        and row["azimuth_deg"] == pytest.approx(28.6, abs=0.1)
    )


def test_runner_counts_frames_and_keeps_only_the_error_type(caplog):
    class Module:
        name = "m"

        def process(self, frame):
            if frame.stamp == 1.0:
                raise ValueError("secret detail")

    runner = LatestFrameRunner(Module())
    runner.start()
    try:
        for stamp in (0.0, 1.0, 2.0):
            runner.submit(Frame(np.zeros((2, 2, 3), np.uint8), stamp))
            deadline = time.monotonic() + 5
            while runner.processed + runner.errors < int(stamp) + 1 and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        runner.stop()
    assert (runner.processed, runner.errors, runner.last_error) == (2, 1, "ValueError")
    assert runner.rates()["processed_hz"] > 0
    assert "secret detail" not in caplog.text


class TexturedSource(Source):
    """Frames with enough texture to pass the enrollment sharpness check."""

    def read(self):
        frame = super().read()
        noise = np.random.default_rng(self.n).integers(0, 255, frame.image.shape, dtype=np.uint8)
        return Frame(noise, frame.stamp, {})


def test_enroll_from_the_live_pipeline_then_the_person_is_named():
    """The path the web UI takes: start, wait for samples, complete, and the next frames name
    the person (SPEC 6)."""
    cfg = RoboEmotConfig.model_validate(
        {
            "vision": {"rates": {"embedding_hz": 20}, "burst_min_gap_s": 0.05},
            "enrollment": {"min_gap_s": 0.05},
        }
    )
    bus = Bus()
    p = open_perception(cfg, bus, TexturedSource(), load_stages=fake_stages)
    names = []
    bus.subscribe("person.recognized", lambda m: names.append(m.payload.person_id))
    p.start()
    try:
        p.enrollment.start("session-1")
        deadline = time.monotonic() + 20
        while not p.enrollment.status()["can_complete"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert p.enrollment.status()["collected"] >= cfg.identity.enroll_min_embeddings
        assert not names  # nobody is enrolled yet
        person_id = p.enrollment.complete(NAME, CONSENT)
        while not names and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        p.stop()
    assert names and names[0] == person_id
    assert p.registry.get_person(person_id).display_name == NAME


def test_an_unenrolled_face_is_tracked_but_stays_unnamed():
    bus = Bus()
    p = open_perception(CFG, bus, Source(), load_stages=fake_stages)
    names, entities = [], []
    bus.subscribe("person.recognized", lambda m: names.append(m))
    bus.subscribe("percept.entity", lambda m: entities.append(m.payload))
    p.start()
    try:
        deadline = time.monotonic() + 15
        while len(entities) < 8 and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        p.stop()
    assert len(entities) >= 8 and names == []
    assert all(e.kind == "person" and e.identity is None for e in entities)


def test_perception_stop_runs_every_step_even_when_the_service_raises(caplog):
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    calls = []

    def boom():
        raise RuntimeError("camera stuck")

    p.service.stop = boom
    p.registry.flush = lambda: calls.append("flush")
    p.enrollment.start("s1")
    with caplog.at_level(logging.WARNING):
        p.stop()
    assert calls == ["flush"]
    assert p.enrollment.status()["state"] == "cancelled"
    assert "RuntimeError" in caplog.text and "camera stuck" not in caplog.text


def test_a_failing_camera_read_is_counted_logged_once_and_retried(monkeypatch, caplog):
    import roboemot.vision.service as service

    monkeypatch.setattr(service, "PUMP_RESTART_S", 0.05)

    class Flaky(Source):
        def read(self):
            if self.n < 3:
                self.n += 1
                raise OSError("sdk: media gone")
            return super().read()

    p = open_perception(CFG, Bus(), Flaky(), load_stages=fake_stages)
    with caplog.at_level(logging.WARNING):
        p.start()
        try:
            deadline = time.monotonic() + 15
            while p.vision_snapshot()["frames"]["processed"] < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            snap = p.vision_snapshot()
        finally:
            p.stop()
    assert snap["camera"]["errors"] == 3 and snap["camera"]["last_error"] == "OSError"
    assert snap["frames"]["processed"] >= 3
    assert sum("camera read failed: OSError" in r.message for r in caplog.records) == 1
    assert "media gone" not in caplog.text


def test_a_raising_face_tap_does_not_break_the_frame():
    bus = Bus()
    p = open_perception(CFG, bus, Source(), load_stages=fake_stages)
    faces = []
    bus.subscribe("percept.face", lambda m: faces.append(m.payload))

    def bad_tap(frame, tracks):
        raise RuntimeError("tap exploded")

    p.pipeline.face_tap = bad_tap
    for i in range(4):
        img = np.zeros((240, 320, 3), np.uint8)
        img[0, 0, 0] = i
        p.pipeline.process(Frame(img, i * 0.2))
    assert len(faces) >= 2  # confirmed after two hits, published despite the failing tap


def test_runner_rates_cover_more_than_256_samples_and_a_partial_window():
    now = [0.0]

    class Module:
        name = "m"

        def process(self, frame):
            pass

    runner = LatestFrameRunner(Module(), clock=lambda: now[0])
    runner.start()
    try:
        frame = Frame(np.zeros((2, 2, 3), np.uint8), 0.0)
        for _ in range(200):  # 2 s of a 100 Hz camera
            runner.submit(frame)
            now[0] += 0.01
        assert runner.rates()["submitted_hz"] == pytest.approx(100, rel=0.05)  # not 200/10 s
        for _ in range(1000):  # 12 s in all: the window is full and holds ~1000 samples
            runner.submit(frame)
            now[0] += 0.01
        assert runner.rates()["submitted_hz"] == pytest.approx(100, rel=0.05)
    finally:
        runner.stop()


def test_hot_reload_reaches_the_pipeline_trackers_governor_and_enrollment():
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    new = RoboEmotConfig.model_validate(
        {
            "vision": {"rates": {"detection_hz": 3}, "tracker": {"lost_s": 7}},
            "enrollment": {"min_gap_s": 2},
        }
    )
    p.apply_config(new)
    assert p.pipeline.governor.rates.detection_hz == 3
    assert p.pipeline.face_tracker.cfg.lost_s == 7 and p.pipeline.object_tracker.cfg.lost_s == 5.0
    assert p.enrollment.cfg.min_gap_s == 2


def test_an_unusable_data_folder_degrades_instead_of_stopping_the_app(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the data folder should be")
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(blocker / "data"))
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    assert p.registry is None and p.pipeline is None and "data folder" in p.unavailable
    assert p.registry_snapshot() == {"state": "unavailable", "reason": p.unavailable}
    assert p.vision_snapshot()["state"] == "unavailable"
    with pytest.raises(EnrollmentError, match="unavailable"):
        p.enrollment.start("s")
    p.start()
    p.stop()  # harmless


def test_models_list_points_at_weights_left_in_the_old_default_folder(
    tmp_path, monkeypatch, capsys
):
    from roboemot import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / ".cache" / "roboemot" / "models"
    legacy.mkdir(parents=True)
    (legacy / "yunet.onnx").write_bytes(b"x")
    assert cli.main(["models", "list", "--cache-dir", str(tmp_path / "new")]) == 0
    out = capsys.readouterr().out
    assert "old default folder" in out and str(legacy) in out
    assert cli.main(["models", "list", "--cache-dir", str(legacy)]) == 0
    assert "old default folder" not in capsys.readouterr().out


def test_the_pipeline_gets_the_stage_detector_and_reports_objects_running():
    objs = SimpleNamespace(detect=lambda image: [])
    stages = fake_stages(CFG, None)
    stages.object_detector = objs
    p = open_perception(CFG, Bus(), Source(), load_stages=lambda cfg, profile: stages)
    assert p.pipeline.object_detector is objs
    assert p.vision_snapshot()["objects"] == {"state": "running"}
    snap = p.vision_snapshot()
    assert snap["tracks"]["created"] == 0 and "lifetime_s" in snap["tracks"]
    assert "attributes" not in snap  # no age model, no attribute worker (2026-10-23)
    json.dumps(snap)


def test_objects_report_off_when_the_flag_is_off_even_with_the_model_loaded():
    stages = fake_stages(CFG, None)
    stages.object_detector = SimpleNamespace(detect=lambda image: [])
    p = open_perception(CFG, Bus(), Source(), load_stages=lambda cfg, profile: stages)
    off = CFG.model_copy(
        update={"vision": CFG.vision.model_copy(update={"objects_enabled": False})}
    )
    p.apply_config(off)
    assert p.vision_snapshot()["objects"] == {"state": "off"}


def test_hot_reload_reaches_the_camera_field_of_view_and_the_identity_settings():
    """Sweep (SPEC 11A): the pipeline's world settings (camera field of view, for the bearings)
    and the matcher's identity settings were start-up copies."""
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    try:
        new = RoboEmotConfig.model_validate(
            {"world": {"camera_fov_h_deg": 66.0}, "identity": {"match_threshold": 0.5}}
        )
        p.apply_config(new)
        assert p.pipeline.world.camera_fov_h_deg == 66.0
        assert p.pipeline.matcher.config is new.identity
    finally:
        p.stop()
