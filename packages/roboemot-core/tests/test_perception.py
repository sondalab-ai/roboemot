"""SPEC 4A, 5, 6, 12, 12A: perception in an app (``roboemot.vision.service``)."""

from __future__ import annotations

import json
import logging
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
    def detect(self, image):
        return [Detection((100, 60, 180, 140), 0.95, landmarks=np.zeros((5, 2)))]


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
        attribute_estimator=None,
        appearance_embedder=None,
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
        age_group="adult",
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
    p.registry.enroll(NAME, [PLANTED] * 5, MODEL, consent=CONSENT, age_group="adult")
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
        person_id = p.enrollment.complete(NAME, CONSENT, "adult", "female")
        while not names and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        p.stop()
    assert names and names[0] == person_id
    assert p.registry.get_person(person_id).display_name == NAME
