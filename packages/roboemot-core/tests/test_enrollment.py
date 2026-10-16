"""SPEC 6, 12 Data: enrollment from the live pipeline (``roboemot.registry.enrollment``)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.registry import Consent, ConsentError, FileStore, Registry
from roboemot.registry.enrollment import (
    CANCELLED,
    COLLECTING,
    COMPLETED,
    READY,
    TIMED_OUT,
    EnrollmentError,
    EnrollmentService,
    sharpness,
)
from roboemot.vision import Detection, Frame
from roboemot.vision.tracker import Track

CFG = RoboEmotConfig()
W, H = 320, 240
DIM = 8
MODEL = "mock-face"
NAME = "Ada Lovelace Secret"
ADULT = Consent(NAME, datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"))


def axis(i: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    if noise:
        v += np.random.default_rng(seed).normal(0, noise, DIM).astype(np.float32)
    return v


def sharp_image(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def blurred_image() -> np.ndarray:
    return np.full((H, W, 3), 128, np.uint8)


def track(track_id: int = 1, x: float = 100, y: float = 60, size: float = 100, score: float = 0.95):
    det = Detection((x, y, x + size, y + size), score, landmarks=np.zeros((5, 2)))
    t = Track(track_id, "face", score, 0.0, 0.0, detection=det)
    t._start(det, 0.0)
    return t


class FakeEmbedder:
    model_id = MODEL

    def __init__(self, vectors=None):
        self.calls = 0
        self.boxes: list[tuple] = []
        self.vectors = vectors or (lambda n: axis(0, 0.05, n))

    def embed(self, image, face):
        self.calls += 1
        self.boxes.append(face.box)
        return self.vectors(self.calls)


@pytest.fixture()
def rig(tmp_path):
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), CFG.identity)
    clock = [0.0]
    svc = EnrollmentService(reg, CFG.enrollment, CFG.identity, clock=lambda: clock[0])
    embedder = FakeEmbedder()
    pipeline = SimpleNamespace(face_embedder=embedder, face_tap=None)
    svc.attach(pipeline)
    return SimpleNamespace(reg=reg, svc=svc, embedder=embedder, pipeline=pipeline, clock=clock)


def feed(rig, n: int, *, tracks=None, image=None, start: float = 0.0, dt: float = 0.5) -> None:
    for i in range(n):
        frame = Frame(sharp_image(i) if image is None else image, start + i * dt)
        rig.pipeline.face_tap(frame, [track()] if tracks is None else tracks)


def test_the_pipeline_tap_is_set_when_attached(rig):
    assert rig.pipeline.face_tap == rig.svc.on_frame


def test_collects_the_target_number_of_samples_then_is_ready(rig):
    rig.svc.start("s1")
    feed(rig, 30)
    st = rig.svc.status()
    assert st["state"] == READY and st["session_id"] == "s1"
    assert st["collected"] == st["target"] == CFG.enrollment.target_samples
    assert st["can_complete"] and rig.embedder.calls == st["target"]


def test_samples_are_spaced_by_the_minimum_gap(rig):
    rig.svc.start("s1")
    feed(rig, 20, dt=0.1)  # 2 s of frames, gap 0.4 s
    assert rig.svc.status()["collected"] == 5


def test_rejects_low_score_small_and_blurred_frames(rig):
    rig.svc.start("s1")
    feed(rig, 3, tracks=[track(score=0.7)])
    feed(rig, 3, tracks=[track(size=40)])
    feed(rig, 3, image=blurred_image())
    st = rig.svc.status()
    assert st["collected"] == 0 and rig.embedder.calls == 0
    assert st["rejected"] == {"score": 3, "small": 3, "blurred": 3, "inconsistent": 0}
    feed(rig, 3)  # a good frame still works afterwards
    assert rig.svc.status()["collected"] >= 1


def test_chooses_the_largest_central_face_and_sticks_to_it(rig):
    big_centre = track(1, x=110, y=70, size=100)
    big_corner = track(2, x=0, y=0, size=100)
    small_centre = track(3, x=140, y=100, size=60)
    rig.svc.start("s1")
    feed(rig, 2, tracks=[small_centre, big_corner, big_centre])
    assert rig.embedder.boxes[0] == big_centre.detection.box
    # a bigger face appears: the session stays with its track while it is visible
    bigger = track(4, x=100, y=40, size=150)
    feed(rig, 2, tracks=[bigger, big_centre], start=10.0)
    assert rig.embedder.boxes[-1] == big_centre.detection.box


def test_a_different_person_does_not_mix_into_the_session(rig):
    rig.embedder.vectors = lambda n: axis(0, 0.02, n) if n <= 3 else axis(5, 0.02, n)
    rig.svc.start("s1")
    feed(rig, 12)
    st = rig.svc.status()
    assert st["collected"] == 3 and st["rejected"]["inconsistent"] >= 1


def test_complete_enrolls_with_consent_and_zeroes_the_samples(rig):
    rig.svc.start("s1")
    feed(rig, 30)
    samples = list(rig.svc._samples)
    person_id = rig.svc.complete(NAME, ADULT, "adult", "female")
    person = rig.reg.get_person(person_id)
    assert person.display_name == NAME and person.status == "active"
    assert len(person.face_embeddings) == CFG.enrollment.target_samples
    assert person.embedding_model_id == MODEL
    assert person.age_group.value == "adult" and person.age_group.source == "declared"
    assert not any(s.any() for s in samples)
    assert rig.svc.status()["state"] == COMPLETED and rig.svc.status()["collected"] == 0


def test_complete_with_too_few_samples_or_no_session_raises(rig):
    with pytest.raises(EnrollmentError, match="no enrollment session"):
        rig.svc.complete(NAME, ADULT, "adult")
    rig.svc.start("s1")
    feed(rig, 2)
    with pytest.raises(EnrollmentError, match="only 2 of 5"):
        rig.svc.complete(NAME, ADULT, "adult")


def test_a_refused_consent_keeps_the_samples_for_a_retry(rig):
    rig.svc.start("s1")
    feed(rig, 30)
    with pytest.raises(ConsentError, match="guardian"):  # age not declared: possible minor
        rig.svc.complete(NAME, ADULT)
    assert rig.svc.status()["collected"] == CFG.enrollment.target_samples
    assert rig.reg.list() == []
    rig.svc.complete(NAME, ADULT, "adult")
    assert len(rig.reg.list()) == 1


def test_cancel_discards_everything(rig):
    rig.svc.start("s1")
    feed(rig, 10)
    samples = list(rig.svc._samples)
    assert samples
    rig.svc.cancel()
    assert not any(s.any() for s in samples)
    st = rig.svc.status()
    assert st["state"] == CANCELLED and st["collected"] == 0 and st["session_id"] is None
    feed(rig, 5)  # frames after a cancel collect nothing
    assert rig.svc.status()["collected"] == 0
    rig.svc.start("s2")  # and a new session can begin
    assert rig.svc.status()["state"] == COLLECTING


def test_one_session_at_a_time(rig):
    rig.svc.start("s1")
    with pytest.raises(EnrollmentError, match="already running"):
        rig.svc.start("s2")


def test_a_session_times_out(rig):
    rig.svc.start("s1")
    feed(rig, 4)
    rig.clock[0] += CFG.enrollment.timeout_s + 1
    st = rig.svc.status()
    assert st["state"] == TIMED_OUT and st["collected"] == 0
    with pytest.raises(EnrollmentError):
        rig.svc.complete(NAME, ADULT, "adult")


def test_unavailable_vision_refuses_to_start(tmp_path):
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), CFG.identity)
    svc = EnrollmentService(reg, CFG.enrollment, CFG.identity)
    svc.set_unavailable("ModelError: yunet missing")
    with pytest.raises(EnrollmentError, match="vision unavailable: ModelError: yunet missing"):
        svc.start("s1")
    st = svc.status()
    assert st["state"] == "unavailable" and "yunet" in st["reason"]
    svc.on_frame(Frame(sharp_image(), 0.0), [track()])  # harmless without a session


def test_nothing_secret_is_logged(rig, caplog):
    with caplog.at_level(logging.DEBUG):
        rig.svc.start("s1")
        feed(rig, 30)
        rig.svc.status()
        rig.svc.complete(NAME, ADULT, "adult", "male")
        rig.svc.start("s2")
        rig.svc.cancel()
    text = caplog.text
    assert NAME not in text and "array" not in text.lower()
    assert not any(isinstance(a, np.ndarray) for r in caplog.records for a in (r.args or ()))


def test_status_has_no_embeddings_or_names(rig):
    rig.svc.start("s1")
    feed(rig, 30)
    flat = repr(rig.svc.status())
    assert "array" not in flat and NAME not in flat


def test_passthroughs_list_rename_merge_delete(rig):
    reg, svc = rig.reg, rig.svc
    pid = reg.enroll(
        NAME, [axis(0, 0.01, i) for i in range(5)], MODEL, consent=ADULT, age_group="adult"
    ).person_id
    cat_a = reg.enroll_entity("animal", "Tom", [axis(1), axis(1, 0.01)], "dino", category="cat")
    cat_b = reg.enroll_entity("animal", "Tommy", [axis(1, 0.02, 3)], "dino", category="cat")
    assert {r.kind for r in svc.list_records()} == {"person", "animal"}
    svc.rename(pid, "Ada")
    svc.rename(cat_a.entity_id, "Tomcat")
    assert reg.get_person(pid).display_name == "Ada"
    assert reg.get_entity(cat_a.entity_id).display_name == "Tomcat"
    svc.merge(cat_a.entity_id, cat_b.entity_id)
    assert [r.id for r in svc.list_records() if r.kind == "animal"] == [cat_a.entity_id]
    with pytest.raises(ValueError, match="people cannot be merged"):
        svc.merge(pid, cat_a.entity_id)
    svc.delete(cat_a.entity_id)
    svc.delete(pid)
    assert svc.list_records() == [] and svc.quarantined() == []
    with pytest.raises(KeyError):
        svc.delete("nope")
    with pytest.raises(KeyError):
        svc.rename("nope", "x")


def test_sharpness_tells_texture_from_blur():
    assert sharpness(sharp_image()[:100, :100], 64) > CFG.enrollment.min_sharpness
    assert sharpness(blurred_image()[:100, :100], 64) < 1.0
    assert sharpness(np.zeros((0, 0, 3), np.uint8), 64) == 0.0
