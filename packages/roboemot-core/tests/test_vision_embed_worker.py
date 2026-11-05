"""SPEC 5, 12A, 13, 14: animal and object appearance embeddings on the embed worker.

Mock stages. The worker is driven step by step with ``run_one`` (deterministic), or started as
a thread with an embedder held on an event; waits are bounded, never sleeps.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.bus import Bus
from roboemot.config.schema import VisionConfig
from roboemot.registry import Matcher
from roboemot.vision import Detection, Frame
from roboemot.vision.embed_worker import EmbedJob, EmbedWorker, cut
from roboemot.vision.pipeline import VisionPipeline
from roboemot.vision.replay import load_session, replay, summarize
from roboemot.vision.service import open_perception
from test_vision import (
    APP_MODEL,
    CFG,
    FixedFaceEmbedder,
    ObjectMock,
    ScriptedDetector,
    axis,
    crowd,
    face,
    image,
    registry,
    write_session,
)

WAIT_S = 5.0
CUP = Detection((100, 100, 160, 160), 0.8, "cup")
NO_GATE = CFG.vision.object_gate.model_copy(update={"enabled": False})


class Appearance:
    """Returns ``vector`` for every box; records the image shape and box it was given."""

    model_id = APP_MODEL

    def __init__(self, vector=None):
        self.vector = axis(5) if vector is None else vector
        self.calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def embed(self, img, box):
        self.calls.append((img.shape, tuple(box)))
        return self.vector.copy()


class Held(Appearance):
    """Blocks inside ``embed`` until ``release`` is set; ``started`` is set on entry."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, img, box):
        self.started.set()
        assert self.release.wait(WAIT_S), "test never released the embedder"
        return super().embed(img, box)


class CountingDetector(ObjectMock):
    def __init__(self, dets):
        super().__init__(dets)
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return self.dets


def pipeline(tmp_path, app, *, worker=True, dets=(CUP,), record=True, **update):
    """A face and the given things in view; ``record`` enrolls a cup that ``Appearance``
    matches (axis 5). Every frame is an object run (gate off)."""
    cfg = CFG.vision.model_copy(
        update={"object_gate": NO_GATE, "entity_embed_min_age_s": 0.0, **update}
    )
    reg = registry(tmp_path)
    if record:
        reg.enroll_entity("object", "Mug", np.stack([axis(5)] * 5), APP_MODEL, category="cup")
    embed_worker = EmbedWorker(app) if worker else None
    pipe = VisionPipeline(
        cfg,
        CFG.identity,
        Bus(),
        Matcher(reg, CFG.identity),
        face_detector=ScriptedDetector(lambda i: [face(100)]),
        face_embedder=FixedFaceEmbedder(axis(0)),
        appearance_embedder=app,
        object_detector=CountingDetector(list(dets)),
        embed_worker=embed_worker,
    )
    return pipe, embed_worker


def observed(pipe):
    """Wrap ``matcher.observe`` to record the entity votes: (track, capture stamp, owner)."""
    calls = []
    inner = pipe.matcher.observe

    def observe(track_id, embedding, model_id, stamp, **kw):
        identity = inner(track_id, embedding, model_id, stamp, **kw)
        if kw.get("kind") == "entity":
            calls.append((track_id, stamp, identity.owner_id))
        return identity

    pipe.matcher.observe = observe
    return calls


def entity(pipe):
    (state,) = pipe._entities.values()
    return state


def frame(i):
    return Frame(image(i), float(i))


def worker_stats(pipe):
    return pipe.stats()["embed_worker"]


# ------------------------------------------------------------------- the vision thread


def test_process_returns_while_the_embed_runs_and_detection_goes_on(tmp_path):
    app = Held()
    pipe, worker = pipeline(tmp_path, app)
    worker.start()
    try:
        for i in range(2):  # confirmed at the second object run: queued there
            pipe.process(frame(i))
        assert app.started.wait(WAIT_S)
        faces, objects = pipe.face_detector.calls, pipe.object_detector.calls
        for i in range(2, 6):
            report = pipe.process(frame(i))
            assert report.faces_detected and report.objects_detected
        assert pipe.face_detector.calls == faces + 4
        assert pipe.object_detector.calls == objects + 4
        state = entity(pipe)
        assert state.pending and state.last_embed is None
        stats = worker_stats(pipe)
        assert stats["busy"] is True and stats["queued"] == 1 and stats["finished"] == 0
        assert pipe.stats()["entity_embeds"]["skips"]["pending"] == 4
        app.release.set()
        assert worker.wait_idle(WAIT_S)
        got = []
        pipe.bus.subscribe("percept.entity", got.append)
        pipe.process(frame(6))
        assert state.last_embed == 1.0 and state.votes == 1  # at the capture stamp of the crop
        cup = [m.payload for m in got if m.payload.kind == "object"]
        assert cup and cup[-1].embedding is not None
        stats = worker_stats(pipe)
        assert stats["finished"] == 1 and stats["queued"] == 2  # the next burst vote is due
    finally:
        app.release.set()
        worker.stop()


def test_the_worker_gets_a_copy_of_the_box_never_the_frame(tmp_path):
    app = Appearance()
    pipe, worker = pipeline(tmp_path, app)
    for i in range(2):
        pipe.process(frame(i))
    worker.run_one()
    ((shape, box),) = app.calls
    assert shape[:2] == (60, 60)  # the cup's 60 x 60 box, not the 240 x 320 frame
    assert box[0] < 1 and box[1] < 1


def test_cut_covers_the_pixels_a_backend_takes_from_the_frame():
    img = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    crop, (x1, y1, x2, y2) = cut(img, (10.7, -3.2, 70.4, 20.9))
    expected = img[0:20, 10:64]  # int edges, clipped to the image
    assert np.array_equal(crop, expected) and not np.shares_memory(crop, img)
    again = crop[max(0, int(y1)) : min(crop.shape[0], int(y2)), max(0, int(x1)) : int(x2)]
    assert np.array_equal(again, expected)  # the same crop as from the whole frame


def run_burst(tmp_path, worker_mode):
    app = Appearance()
    pipe, worker = pipeline(tmp_path, app, worker=worker_mode)
    votes = observed(pipe)
    known = []
    for i in range(12):
        pipe.process(frame(i))
        if worker is not None:
            worker.run_one()  # the embedding finishes before the next frame
        known.append(bool(pipe._entities) and entity(pipe).owner_id is not None)
    owners = {o for *_, o in votes} - {None}
    return [(t, s, o is None) for t, s, o in votes], known, pipe.stats()["entity_embeds"], owners


def test_a_vote_burst_on_the_worker_reaches_the_inline_identity(tmp_path):
    inline_votes, inline_known, inline_counts, inline_owners = run_burst(tmp_path / "a", False)
    votes, known, counts, owners = run_burst(tmp_path / "b", True)
    # The same votes, at the same capture stamps, name the record at the same vote (the last
    # frame's vote is still to be applied on the worker)...
    assert votes == inline_votes[:-1] and inline_votes[-1][1] == 11.0
    assert len(owners) == len(inline_owners) == 1
    # ...applied at the start of the next frame; the runs and skips by reason are the same.
    assert known[1:] == inline_known[:-1] and any(known)
    assert counts == inline_counts and counts["skips"]["pending"] == 0


def test_a_pending_track_is_never_queued_twice(tmp_path):
    pipe, worker = pipeline(tmp_path, Appearance())
    for i in range(8):
        pipe.process(frame(i))
    stats = worker_stats(pipe)
    assert stats["queued"] == 1 and stats["waiting"] == 1 and stats["replaced"] == 0
    assert pipe.stats()["entity_embeds"]["skips"]["pending"] == 6


def test_a_job_dropped_at_the_cap_lets_its_track_queue_again(tmp_path):
    pipe, worker = pipeline(
        tmp_path, Appearance(), dets=crowd(3), record=False, embed_queue_max=2,
        max_entity_embeds_per_frame=3,
    )  # fmt: skip
    for i in range(2):
        pipe.process(frame(i))
    stats = worker_stats(pipe)
    assert stats["queued"] == 3 and stats["dropped_cap"] == 1 and stats["waiting"] == 2
    first = pipe._entities[min(pipe._entities)]  # served first, so the oldest job
    assert first.pending
    pipe.process(frame(2))  # the drop comes back: that track is queued again
    assert first.pending and worker_stats(pipe)["queued"] == 4


def test_a_dropped_job_gives_the_reacquired_flag_back(tmp_path):
    pipe, worker = pipeline(tmp_path, Appearance(), record=False, embed_queue_max=1)
    for i in range(2):
        pipe.process(frame(i))
    worker.run_one()
    pipe.process(frame(2))  # the first embedding applied
    pipe.object_detector.dets = []
    pipe.process(frame(3))  # an object run without the cup
    pipe.object_detector.dets = [CUP]
    pipe.process(frame(4))  # back: queued as reacquired, the flag spent
    state = entity(pipe)
    assert state.pending and not state.reacquired
    assert pipe.stats()["entity_embeds"]["runs"]["reacquired"] == 1
    worker.submit(job(999), cap=1)  # pushes the cup's job out
    pipe.process(frame(5))  # the drop comes back with the flag: queued as reacquired again
    assert state.pending and pipe.stats()["entity_embeds"]["runs"]["reacquired"] == 2


def until_gone(pipe, start):
    pipe.object_detector.dets = []
    i = start
    while pipe._entities:
        pipe.process(frame(i))
        i += 1
        assert i < start + 100
    return i


def test_a_result_for_a_track_that_ended_is_discarded(tmp_path):
    app = Held()
    pipe, worker = pipeline(tmp_path, app)
    votes = observed(pipe)
    worker.start()
    try:
        for i in range(2):
            pipe.process(frame(i))
        assert app.started.wait(WAIT_S)  # running: no longer withdrawable
        i = until_gone(pipe, 2)
        app.release.set()
        assert worker.wait_idle(WAIT_S)
        pipe.process(frame(i))
        assert votes == [] and worker_stats(pipe)["dropped_ended"] == 1
    finally:
        app.release.set()
        worker.stop()


def test_a_waiting_job_of_a_track_that_ended_is_withdrawn(tmp_path):
    app = Appearance()
    pipe, worker = pipeline(tmp_path, app)
    for i in range(2):
        pipe.process(frame(i))
    until_gone(pipe, 2)
    stats = worker_stats(pipe)
    assert stats["waiting"] == 0 and stats["dropped_ended"] == 1
    assert not worker.run_one() and app.calls == []


def test_a_result_in_flight_across_a_reset_never_reaches_the_reused_track_id(tmp_path):
    app = Held()
    pipe, worker = pipeline(tmp_path, app)
    votes = observed(pipe)
    worker.start()
    try:
        for i in range(2):
            pipe.process(frame(i))
        assert app.started.wait(WAIT_S)
        (old,) = pipe._entities
        pipe.reset()
        for i in range(10, 12):  # the same ID again, a new state with a job of its own
            pipe.process(frame(i))
        assert tuple(pipe._entities) == (old,)
        app.release.set()
        assert worker.wait_idle(WAIT_S)
        pipe.process(frame(12))
        assert [stamp for _, stamp, _ in votes] == [11.0]
        assert entity(pipe).last_embed == 11.0
        assert worker_stats(pipe)["dropped_ended"] == 1
    finally:
        app.release.set()
        worker.stop()


# ------------------------------------------------------------------------ the queue


def job(track_id, stamp=0.0, value=0):
    crop = np.full((2, 2, 3), value, np.uint8)
    return EmbedJob(track_id, object(), crop, (0, 0, 2, 2), stamp, "new", False, False,
                    (0, 0, 2, 2), False)  # fmt: skip


class Echo:
    model_id = APP_MODEL

    def embed(self, img, box):
        return np.full(2, float(img[0, 0, 0]), np.float32)


def test_a_newer_job_of_a_track_replaces_the_waiting_one():
    worker = EmbedWorker(Echo())
    worker.submit(job(1, value=1), cap=4)
    worker.submit(job(1, value=2), cap=4)
    assert worker.run_one() and not worker.run_one()
    (outcome,) = worker.drain()
    assert outcome.embedding[0] == 2
    stats = worker.snapshot(30.0)
    assert stats["queued"] == 2 and stats["replaced"] == 1 and stats["finished"] == 1


def test_the_cap_drops_the_oldest_waiting_job_and_hands_it_back():
    worker = EmbedWorker(Echo())
    for track in (1, 2, 3):
        worker.submit(job(track), cap=2)
    (dropped,) = worker.drain()
    assert dropped.dropped and dropped.job.track_id == 1 and dropped.embedding is None
    assert worker.snapshot(30.0)["dropped_cap"] == 1 and worker.snapshot(30.0)["waiting"] == 2


def test_a_failing_embed_is_counted_and_handed_back_without_an_embedding(caplog):
    class Broken:
        model_id = APP_MODEL

        def embed(self, img, box):
            raise ValueError("secret detail")

    worker = EmbedWorker(Broken())
    worker.submit(job(1), cap=4)
    assert worker.run_one()
    (outcome,) = worker.drain()
    assert outcome.error == "ValueError" and outcome.embedding is None
    assert worker.snapshot(30.0)["failed"] == 1
    assert "secret detail" not in caplog.text


def test_the_queue_wait_is_reported_from_the_worker_clock():
    now = [0.0]
    worker = EmbedWorker(Echo(), clock=lambda: now[0], max_window_s=60.0)
    worker.submit(job(1), cap=4)
    now[0] = 0.25
    worker.run_one()
    stats = worker.snapshot(30.0)
    assert stats["queue_wait_ms"] == {"p50": 250.0, "p90": 250.0}
    assert "embedding" not in json.dumps(stats)


def test_the_inline_pipeline_reports_its_mode(tmp_path):
    pipe, _ = pipeline(tmp_path, Appearance(), worker=False)
    assert pipe.stats()["embed_worker"] == {"state": "inline"}


# ------------------------------------------------------------------ replay, the app


def test_replay_runs_inline_even_with_an_embed_worker(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    write_session(root, 12)
    session = load_session(root)
    results = []
    for name, worker in (("a", False), ("b", True)):
        pipe, embed_worker = pipeline(tmp_path / name, Appearance(), worker=worker)
        results.append(summarize(replay(session, pipe, pipe.bus).messages))
        assert pipe.embed_worker is embed_worker
        if embed_worker is not None:
            assert embed_worker.snapshot(30.0)["queued"] == 0
    assert results[0] and results[0] == results[1]


def test_open_perception_gives_the_live_pipeline_an_embed_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "data"))
    app = Appearance()

    def stages(cfg, profile):
        return SimpleNamespace(
            face_detector=ScriptedDetector(lambda i: []),
            face_embedder=FixedFaceEmbedder(axis(0)),
            appearance_embedder=app,
            object_detector=ObjectMock([]),
            objects_unavailable=None,
        )

    class Source:
        def read(self):
            return None

        def camera_model(self, w, h):
            return None

    p = open_perception(CFG, Bus(), Source(), load_stages=stages)
    worker = p.pipeline.embed_worker
    assert worker is not None and p.service.embed_worker is worker
    assert worker.timer is p.pipeline.timer
    p.start()
    try:
        assert p.vision_snapshot()["embed_worker"]["state"] == "running"
    finally:
        p.stop()
    assert p.vision_snapshot()["embed_worker"]["state"] == "idle"


# ------------------------------------------------------------------ the input size


@pytest.mark.parametrize("px", [112, 126, 168, 224])
def test_the_input_size_takes_whole_patches_in_range(px):
    assert VisionConfig(appearance_input_px=px).appearance_input_px == px


@pytest.mark.parametrize("px", [98, 120, 238, 230])
def test_the_input_size_refuses_partial_patches_and_out_of_range(px):
    with pytest.raises(ValidationError):
        VisionConfig(appearance_input_px=px)


def test_the_defaults_keep_the_224_px_model_and_two_threads():
    cfg = VisionConfig()
    assert cfg.appearance_input_px == 224 and cfg.appearance_threads == 2
    assert cfg.embed_queue_max == 4


def test_the_model_id_names_any_size_but_224():
    onnx = pytest.importorskip("roboemot.vision.onnx")
    assert onnx.dino_model_id(224) == "dinov2-small"
    assert onnx.dino_model_id(112) == "dinov2-small-112px"
    assert onnx.dino_model_id(126) != onnx.dino_model_id(112)
