"""SPEC 6, 12 Data: enrollment from the live pipeline (``roboemot.registry.enrollment``)."""

from __future__ import annotations

import logging
import math
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
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
    yaw_proxy,
)
from roboemot.vision import Detection, Frame
from roboemot.vision.tracker import Track

CFG = RoboEmotConfig()
W, H = 320, 240
DIM = 8
MODEL = "mock-face"
NAME = "Ada Lovelace Secret"
ADULT = Consent(NAME, datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"))
TARGET = CFG.enrollment.target_samples


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


def landmarks(x: float, y: float, size: float, yaw: float) -> np.ndarray:
    """YuNet order: right eye (image left), left eye, nose, mouth corners; ``yaw`` shifts the
    nose in inter-eye distances."""
    eye_l, eye_r = x + 0.3 * size, x + 0.7 * size
    mid = (eye_l + eye_r) / 2
    nose = mid + yaw * (eye_r - eye_l)
    return np.array(
        [
            [eye_l, y + 0.4 * size],
            [eye_r, y + 0.4 * size],
            [nose, y + 0.6 * size],
            [eye_l, y + 0.8 * size],
            [eye_r, y + 0.8 * size],
        ]
    )


def track(track_id=1, x=100.0, y=60.0, size=100.0, score=0.95, yaw=0.0):
    det = Detection((x, y, x + size, y + size), score, landmarks=landmarks(x, y, size, yaw))
    t = Track(track_id, "face", score, 0.0, 0.0, detection=det)
    t._start(det, 0.0)
    return t


def swing(i: int) -> float:
    """Head yaw proxy that sweeps -0.25 .. 0.25 over successive samples."""
    return 0.25 * math.sin(i)


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


def feed(rig, n, *, tracks=None, image=None, start=0.0, dt=0.5, varied=True):
    for i in range(n):
        frame = Frame(sharp_image(i) if image is None else image, start + i * dt)
        found = tracks if tracks is not None else [track(yaw=swing(i) if varied else 0.0)]
        rig.pipeline.face_tap(frame, found)


def plain(value) -> bool:
    """Only JSON-like values, recursively: no arrays, no numpy scalars."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain(v) for v in value)
    return value is None or type(value) in (bool, int, float, str)


def test_the_pipeline_tap_is_set_when_attached(rig):
    assert rig.pipeline.face_tap == rig.svc.on_frame


def test_collects_varied_samples_until_the_target_then_is_ready(rig):
    rig.svc.start("s1")
    feed(rig, 40)
    st = rig.svc.status()
    assert st["state"] == READY and st["session_id"] == "s1"
    assert TARGET <= st["collected"] <= CFG.identity.max_embeddings_per_person
    assert st["can_complete"] and st["pose_varied"]


def test_frontal_only_samples_keep_collecting_up_to_the_cap(rig):
    """SPEC 6, varied angles: ten identical poses are not enough; the cap ends the session."""
    rig.svc.start("s1")
    feed(rig, 12, varied=False)
    st = rig.svc.status()
    assert st["collected"] == 12 and st["state"] == COLLECTING
    assert not st["can_complete"] and st["hint"]
    with pytest.raises(EnrollmentError, match="too little change of head angle"):
        rig.svc.complete(NAME, ADULT, "adult")
    feed(rig, 20, varied=False, start=100.0)
    st = rig.svc.status()
    assert st["state"] == READY and st["collected"] == CFG.identity.max_embeddings_per_person
    assert st["can_complete"] and not st["pose_varied"]


def test_samples_are_spaced_by_the_minimum_gap(rig):
    rig.svc.start("s1")
    feed(rig, 20, dt=0.1)  # 2 s of frames, gap 0.4 s
    assert rig.svc.status()["collected"] == 5


def test_rejects_low_score_small_cropped_and_blurred_frames(rig):
    rig.svc.start("s1")
    feed(rig, 3, tracks=[track(score=0.7)])
    feed(rig, 3, tracks=[track(size=40)])
    feed(rig, 3, tracks=[track(x=-10)])  # box leaves the frame
    feed(rig, 3, tracks=[track(x=W - 50)])
    feed(rig, 3, image=blurred_image())
    st = rig.svc.status()
    assert st["collected"] == 0 and rig.embedder.calls == 0
    assert st["rejected"] == {"score": 3, "small": 3, "cropped": 6, "blurred": 3, "inconsistent": 0}
    feed(rig, 3)  # a good frame still works afterwards
    assert rig.svc.status()["collected"] >= 1


def test_chooses_the_largest_central_face_and_sticks_to_it(rig):
    big_centre = track(1, x=110, y=70, size=100)
    big_corner = track(2, x=0, y=0, size=100)
    small_centre = track(3, x=140, y=100, size=60)
    rig.svc.start("s1")
    feed(rig, 2, tracks=[small_centre, big_corner, big_centre])
    assert rig.embedder.boxes[0] == big_centre.detection.box
    bigger = track(4, x=100, y=40, size=150)  # a bigger face appears: stay with the session's
    feed(rig, 2, tracks=[bigger, big_centre], start=10.0)
    assert rig.embedder.boxes[-1] == big_centre.detection.box


def test_the_lock_on_a_track_ends_with_the_track(rig):
    rig.svc.start("s1")
    feed(rig, 1, tracks=[track(1, x=110, y=70, size=100)])
    feed(rig, 1, tracks=[], start=5.0)  # the face is gone
    feed(rig, 1, tracks=[track(2, x=20, y=20, size=120), track(1, x=110, y=70, size=80)], start=9.0)
    assert rig.embedder.boxes[-1] == (20, 20, 140, 140)  # chosen afresh, not stuck on track 1


def test_a_wrong_first_sample_does_not_lock_the_session(rig):
    """The first sample is a candidate: a second one that disagrees replaces it."""
    rig.embedder.vectors = lambda n: axis(5, 0.02, n) if n == 1 else axis(0, 0.02, n)
    rig.svc.start("s1")
    feed(rig, 12)
    st = rig.svc.status()
    assert st["rejected"]["inconsistent"] == 1
    assert st["collected"] >= TARGET - 1
    assert all(float(s @ axis(0)) > 0.9 for s in rig.svc._samples)


def test_a_different_person_later_does_not_mix_in(rig):
    rig.embedder.vectors = lambda n: axis(0, 0.02, n) if n <= 3 else axis(5, 0.02, n)
    rig.svc.start("s1")
    feed(rig, 12)
    st = rig.svc.status()
    assert st["collected"] == 3 and st["rejected"]["inconsistent"] >= 1


def test_complete_enrolls_with_consent_and_zeroes_the_samples(rig):
    rig.svc.start("s1")
    feed(rig, 40)
    samples = list(rig.svc._samples)
    person_id = rig.svc.complete(NAME, ADULT, "adult", "female")
    person = rig.reg.get_person(person_id)
    assert person.display_name == NAME and person.status == "active"
    assert len(person.face_embeddings) == len(samples)
    assert person.embedding_model_id == MODEL
    assert person.age_group.value == "adult" and person.age_group.source == "declared"
    assert not any(s.any() for s in samples)
    st = rig.svc.status()
    assert st["state"] == COMPLETED and st["collected"] == 0


def test_complete_with_too_few_samples_or_no_session_raises(rig):
    with pytest.raises(EnrollmentError, match="no enrollment session"):
        rig.svc.complete(NAME, ADULT, "adult")
    rig.svc.start("s1")
    feed(rig, 2)
    with pytest.raises(EnrollmentError, match="only 2 of 5"):
        rig.svc.complete(NAME, ADULT, "adult")


def test_a_refused_consent_keeps_the_samples_for_a_retry(rig):
    rig.svc.start("s1")
    feed(rig, 40)
    n = rig.svc.status()["collected"]
    with pytest.raises(ConsentError, match="guardian"):  # age not declared: possible minor
        rig.svc.complete(NAME, ADULT)
    st = rig.svc.status()
    assert st["collected"] == n and st["state"] == READY
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


def test_two_threads_starting_at_once_get_one_session(rig):
    results, barrier = [], threading.Barrier(2)

    def go(name):
        barrier.wait()
        try:
            rig.svc.start(name)
            results.append("ok")
        except EnrollmentError:
            results.append("busy")

    threads = [threading.Thread(target=go, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == ["busy", "ok"]


def test_two_threads_completing_at_once_enroll_one_person(rig, monkeypatch):
    rig.svc.start("s1")
    feed(rig, 40)
    real = rig.reg.enroll
    gate = threading.Event()

    def slow(*a, **kw):
        gate.wait(5)
        return real(*a, **kw)

    monkeypatch.setattr(rig.reg, "enroll", slow)
    outcomes, barrier = [], threading.Barrier(2)

    def go():
        barrier.wait()
        try:
            outcomes.append(rig.svc.complete(NAME, ADULT, "adult"))
        except EnrollmentError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    while len([o for o in outcomes if "no enrollment session" in o]) < 1:
        threading.Event().wait(0.01)  # the loser is refused while the winner is storing
    assert rig.svc.status()["state"] == "completing"
    rig.svc.cancel()  # does not interrupt a session that is being stored
    gate.set()
    [t.join(10) for t in threads]
    assert len(rig.reg.list()) == 1
    assert sum(1 for o in outcomes if len(o) == 36) == 1  # one uuid


def test_an_embedding_in_flight_when_the_session_is_cancelled_is_discarded(rig):
    entered, release = threading.Event(), threading.Event()

    class Blocking(FakeEmbedder):
        def embed(self, image, face):
            entered.set()
            release.wait(5)
            return super().embed(image, face)

    rig.svc._embedder = Blocking()
    rig.svc.start("s1")
    worker = threading.Thread(target=lambda: feed(rig, 1))
    worker.start()
    assert entered.wait(5)
    rig.svc.cancel()
    rig.svc.start("s2")  # a new session starts while the old frame is still being embedded
    release.set()
    worker.join(5)
    assert rig.svc.status()["collected"] == 0 and rig.svc.status()["state"] == COLLECTING


def test_a_session_times_out_on_the_next_call(rig):
    rig.svc.start("s1")
    feed(rig, 4)
    rig.clock[0] += CFG.enrollment.timeout_s + 1
    st = rig.svc.status()
    assert st["state"] == TIMED_OUT and st["collected"] == 0
    with pytest.raises(EnrollmentError):
        rig.svc.complete(NAME, ADULT, "adult")


def test_expire_ends_a_session_without_any_frame_or_call(rig):
    rig.svc.start("s1")
    feed(rig, 4)
    samples = list(rig.svc._samples)
    rig.clock[0] += CFG.enrollment.timeout_s + 1
    rig.svc.expire()  # what the app's main loop does
    assert rig.svc._state == TIMED_OUT and rig.svc._samples == []
    assert not any(s.any() for s in samples)


def test_unavailable_vision_refuses_to_start(tmp_path):
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), CFG.identity)
    svc = EnrollmentService(reg, CFG.enrollment, CFG.identity)
    svc.set_unavailable("ModelError: yunet missing")
    with pytest.raises(EnrollmentError, match="vision unavailable: ModelError: yunet missing"):
        svc.start("s1")
    st = svc.status()
    assert st["state"] == "unavailable" and "yunet" in st["reason"]
    svc.on_frame(Frame(sharp_image(), 0.0), [track()])  # harmless without a session


def test_without_a_registry_every_registry_call_is_refused():
    svc = EnrollmentService(None, CFG.enrollment, CFG.identity)
    svc.set_unavailable("data folder: PermissionError")
    for call in (svc.list_records, svc.quarantined, lambda: svc.rename("x", "y")):
        with pytest.raises(EnrollmentError, match="registry unavailable"):
            call()


def test_nothing_secret_is_logged(rig, caplog):
    with caplog.at_level(logging.DEBUG):
        rig.svc.start("s1")
        feed(rig, 40)
        rig.svc.status()
        rig.svc.complete(NAME, ADULT, "adult", "male")
        rig.svc.start("s2")
        rig.svc.cancel()
    assert NAME not in caplog.text
    assert all(isinstance(r.msg, str) and not r.args for r in caplog.records if "enroll" in r.msg)
    assert not any(isinstance(a, np.ndarray) for r in caplog.records for a in (r.args or ()))


def test_status_holds_only_plain_values_never_embeddings_or_names(rig):
    rig.svc.start("s1")
    feed(rig, 40)
    status = rig.svc.status()
    assert plain(status)
    assert NAME not in repr(status)


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


def test_rename_person_persists_and_refuses_unknown_or_forgotten_ids(tmp_path):
    store_dir = tmp_path / "reg"
    reg = Registry(FileStore(store_dir, allow_plaintext=True), CFG.identity)
    pid = reg.enroll(
        NAME, [axis(0, 0.01, i) for i in range(5)], MODEL, consent=ADULT, age_group="adult"
    ).person_id
    reg.rename_person(pid, "Ada")
    reloaded = Registry(FileStore(store_dir, allow_plaintext=True), CFG.identity)
    assert reloaded.get_person(pid).display_name == "Ada"
    assert reloaded.get_person(pid).face_embeddings.shape == (5, DIM)
    with pytest.raises(KeyError):
        reg.rename_person("unknown", "x")
    reg.forget(pid)
    with pytest.raises(KeyError):
        reg.rename_person(pid, "again")
    assert reg.list() == []


def test_sharpness_tells_texture_from_blur_and_area_pooling_ignores_pixel_noise():
    assert sharpness(sharp_image()[:100, :100], 64) > CFG.enrollment.min_sharpness
    assert sharpness(blurred_image()[:100, :100], 64) < 1.0
    assert sharpness(np.zeros((0, 0, 3), np.uint8), 64) == 0.0
    # fine sensor noise on a large face is averaged down by pooling (point sampling keeps it)
    rng = np.random.default_rng(1)
    noisy = np.clip(128 + rng.normal(0, 8, (256, 256, 3)), 0, 255).astype(np.uint8)
    assert sharpness(noisy, 64) < 0.2 * sharpness(noisy[:64, :64], 64)
    assert sharpness(noisy[:40, :40], 64) > 0  # smaller than the target size: no crash


def test_yaw_proxy():
    assert yaw_proxy(landmarks(0, 0, 100, 0.0)) == pytest.approx(0.0)
    assert yaw_proxy(landmarks(0, 0, 100, 0.3)) == pytest.approx(0.3)
    assert yaw_proxy(None) is None
    assert yaw_proxy(np.zeros((5, 2))) is None


def test_target_samples_outside_the_identity_bounds_is_rejected():
    with pytest.raises(ValidationError, match="target_samples"):
        RoboEmotConfig.model_validate(
            {"identity": {"enroll_min_embeddings": 8}, "enrollment": {"target_samples": 6}}
        )
    with pytest.raises(ValidationError, match="target_samples"):
        RoboEmotConfig.model_validate(
            {"identity": {"max_embeddings_per_person": 8}, "enrollment": {"target_samples": 12}}
        )
    assert RoboEmotConfig.model_validate({"enrollment": {"target_samples": 20}})
