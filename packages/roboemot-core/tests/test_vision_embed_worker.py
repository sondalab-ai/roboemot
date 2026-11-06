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
from roboemot import priority
from roboemot.bus import Bus
from roboemot.config.schema import VisionConfig
from roboemot.registry import Matcher
from roboemot.vision import Detection, Frame
from roboemot.vision.embed_worker import THREAD_NAME, EmbedJob, EmbedWorker, cut
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
    pipe.appearance_embedder = None  # objects off: no DINOv2 anywhere
    assert pipe.stats()["embed_worker"] == {"state": "off"}


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
            object_tracker=None,
            object_track_unavailable=None,
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


# ------------------------------------------------------------------ robustness (review)


class Raising:
    model_id = APP_MODEL

    def __init__(self, exc=ValueError):
        self.exc = exc
        self.calls = 0

    def embed(self, img, box):
        self.calls += 1
        raise self.exc("secret detail")


class BrokenTimer:
    """A stage timer whose ``record`` raises: a failure outside ``embedder.embed``."""

    def start(self):
        return 0.0, 0.0

    def record(self, stage, wall0, cpu0):
        raise RuntimeError("timer")


def test_a_failure_outside_the_embedder_still_hands_the_job_back(tmp_path):
    app = Appearance()
    pipe, _ = pipeline(tmp_path, app, worker=False)
    worker = EmbedWorker(app, BrokenTimer())
    pipe.embed_worker = worker
    for i in range(2):
        pipe.process(frame(i))
    assert entity(pipe).pending
    assert worker.run_one()
    (outcome,) = worker.drain()
    assert outcome.error == "RuntimeError" and outcome.embedding is None
    worker._done.append(outcome)
    pipe.process(frame(2))
    state = entity(pipe)
    assert not state.pending and state.failed_at == 1.0  # handed back, then held off
    assert worker_stats(pipe)["queued"] == 1


def test_the_worker_thread_survives_a_failing_step(tmp_path):
    worker = EmbedWorker(Echo(), BrokenTimer())
    worker.start()
    try:
        for track in (1, 2):
            worker.submit(job(track), cap=4)
            assert worker.wait_idle(WAIT_S)
        assert [o.error for o in worker.drain()] == ["RuntimeError", "RuntimeError"]
        assert worker.snapshot(30.0)["state"] == "running"
    finally:
        worker.stop()


def test_no_outcome_dropped_job_or_report_keeps_pixels(tmp_path, caplog):
    worker = EmbedWorker(Echo())
    for track in (1, 2, 3):
        worker.submit(job(track, value=7), cap=2)  # 1 dropped at the cap
    worker.run_one()  # 2 finishes
    worker.cancel(3)  # 3 withdrawn
    failing = EmbedWorker(Raising())
    failing.submit(job(4, value=7), cap=4)
    failing.run_one()
    outcomes = worker.drain() + failing.drain()
    assert len(outcomes) == 3
    for outcome in outcomes:
        assert outcome.job.crop.size == 0
        assert (outcome.embedding is not None) == (not outcome.dropped and outcome.error is None)
    assert not worker._queue and not worker._done
    worker.submit(job(5, value=7), cap=4)
    worker.clear()
    assert not worker._queue and not worker._done
    worker.submit(job(6, value=7), cap=4)
    worker.stop()  # a waiting job comes back without its crop
    worker.submit(job(7, value=7), cap=4)  # after stop: nothing kept
    late = worker.drain()
    assert [o.job.track_id for o in late] == [6, 7] and all(o.dropped for o in late)
    assert all(o.job.crop.size == 0 for o in late) and not worker._queue
    text = json.dumps(worker.snapshot(30.0))
    assert "embedding" not in text and "secret detail" not in caplog.text


def test_a_failing_box_waits_embed_retry_s_before_it_is_queued_again(tmp_path):
    app = Raising()
    pipe, worker = pipeline(tmp_path, app, record=False)
    for i in range(25):
        pipe.process(frame(i))
        worker.run_one()
    retry = int(pipe.cfg.embed_retry_s)
    assert app.calls == 3  # at 1 s, then 1 + 10 and 1 + 20 s, not at every object run
    assert pipe.stats()["entity_embeds"]["skips"]["retry"] >= 2 * (retry - 2)
    assert worker_stats(pipe)["failed"] == 3


def test_a_success_clears_the_retry_wait(tmp_path):
    pipe, worker = pipeline(tmp_path, Appearance(), record=False)
    for i in range(2):
        pipe.process(frame(i))
    state = entity(pipe)
    state.failed_at = 0.5
    worker.run_one()
    pipe.process(frame(2))
    assert state.failed_at is None and state.last_embed == 1.0


def test_start_after_a_stop_that_timed_out_keeps_one_thread(tmp_path):
    app = Held()
    worker = EmbedWorker(app, join_timeout_s=0.05)
    worker.start()
    try:
        worker.submit(job(1), cap=4)
        assert app.started.wait(WAIT_S)
        old = worker._thread
        worker.stop()  # bounded: the stuck run is left, not waited for
        assert old.is_alive()
        worker.start()
        assert worker._thread is old  # not doubled
        app.release.set()
        worker.submit(job(2), cap=4)
        assert worker.wait_idle(WAIT_S)
        assert [o.job.track_id for o in worker.drain()] == [1, 2]
    finally:
        app.release.set()
        worker.stop()
    assert not old.is_alive()


def test_stop_then_start_runs_jobs_on_a_new_thread():
    worker = EmbedWorker(Echo())
    worker.start()
    first = worker._thread
    worker.stop()
    assert not first.is_alive()
    worker.start()
    try:
        assert worker._thread is not first
        worker.submit(job(1, value=3), cap=4)
        assert worker.wait_idle(WAIT_S)
        (outcome,) = worker.drain()
        assert outcome.embedding[0] == 3
    finally:
        worker.stop()


@pytest.fixture
def lowering(monkeypatch):
    monkeypatch.setattr(priority, "is_supported", lambda: True)
    yield
    priority.configure(0, ())


def test_a_user_thread_list_without_the_worker_is_warned_about(lowering, caplog):
    priority.configure(5, ["roboemot-vision"])
    assert priority.left_out(THREAD_NAME)
    worker = EmbedWorker(Echo())
    worker.start()
    worker.stop()
    assert f"{THREAD_NAME} is not in runtime.background_threads" in caplog.text
    priority.configure(5, ["roboemot-vision", THREAD_NAME])
    assert not priority.left_out(THREAD_NAME)
    priority.configure(0, ["roboemot-vision"])
    assert not priority.left_out(THREAD_NAME)  # nothing is lowered anyway


# --------------------------------------------- results late by a random number of frames


class LateWorker(EmbedWorker):
    """Runs every job at once but hands each outcome to the vision thread 0 to ``max_lag``
    frames later (seeded), as a slow worker would."""

    def __init__(self, embedder, seed, max_lag):
        super().__init__(embedder)
        self.rng = np.random.default_rng(seed)
        self.max_lag = max_lag
        self.frame = 0.0
        self.held = []

    def drain(self):
        for outcome in super().drain():
            self.held.append((self.frame + int(self.rng.integers(0, self.max_lag + 1)), outcome))
        ready = [o for at, o in self.held if at <= self.frame]
        self.held = [(at, o) for at, o in self.held if at > self.frame]
        return ready


SPACED = {
    "entity_burst_min_gap_s": 5.0,  # longer than any lag below: the next decision sees
    "rates": CFG.vision.rates.model_copy(update={"appearance_hz": 0.2}),  # the result
    "max_entity_embeds_per_frame": 1,
}


def scripted(pipe, worker, frames):
    """Two cups and a third that comes and goes; a shown-object look from 12 s to 20 s."""
    for i in range(frames):
        if worker is not None:
            worker.frame = i
        pipe.object_detector.dets = crowd(3) if i % 9 < 6 else crowd(2)
        pipe.governor.set_active("objects", 12 <= i < 20)
        pipe.process(frame(i))
        while worker is not None and worker.run_one():
            pass
    if worker is not None:
        worker.frame = float("inf")
        pipe._apply_embeds()


def final(pipe, reg_names):
    name = lambda owner: None if owner is None else reg_names(owner)  # noqa: E731
    states = {
        i: (s.votes, name(s.owner_id), s.last_embed, s.box) for i, s in pipe._entities.items()
    }
    history = {
        i: [(name(v), stamp) for v, stamp in t.face] for i, t in pipe.matcher._tracks.items()
    }
    return states, history


@pytest.mark.parametrize("seed", range(4))
def test_late_results_reach_the_inline_outcome(tmp_path, seed):
    results = []
    for mode in ("inline", "late"):
        app = Appearance()
        pipe, _ = pipeline(tmp_path / mode, app, worker=False, dets=crowd(3), **SPACED)
        worker = None
        if mode == "late":
            worker = pipe.embed_worker = LateWorker(app, seed, max_lag=3)
        votes = observed(pipe)
        scripted(pipe, worker, 30)
        assert pipe.stats()["entity_embeds"]["skips"]["episode"] > 0  # the look took effect
        reg = pipe.matcher.registry
        results.append((final(pipe, lambda o, r=reg: r.get_entity(o).display_name), votes))
    (inline_final, inline_votes), (late_final, late_votes) = results
    assert inline_final == late_final
    assert sorted((t, s) for t, s, _ in inline_votes) == sorted((t, s) for t, s, _ in late_votes)
    assert any(owner for *_, owner in inline_votes)  # the record was recognised


def test_a_late_result_of_an_ended_track_leaves_its_rebirth_alone(tmp_path):
    app = Held()
    pipe, worker = pipeline(tmp_path, app, record=False, object_lost_s=0.5, object_lost_periods=1.0)
    worker.start()
    try:
        for i in range(2):
            pipe.process(frame(i))
        assert app.started.wait(WAIT_S)  # the first track's job (stamp 1 s) runs
        (old,) = pipe._entities
        i = until_gone(pipe, 2)
        pipe.object_detector.dets = [CUP]
        for k in range(i, i + 2):  # reborn at the same spot: it carries age and fields
            pipe.process(frame(k))
        (new,) = pipe._entities
        reborn = entity(pipe)
        assert new != old and reborn.born == 0.0 and reborn.last_embed is None
        app.release.set()
        assert worker.wait_idle(WAIT_S)
        pipe.process(frame(i + 2))
        assert worker_stats(pipe)["dropped_ended"] == 1  # the old state's result
        # Only the reborn track's own result (its own capture stamp) reached it.
        assert reborn.last_embed == i + 1.0 and reborn.born == 0.0
    finally:
        app.release.set()
        worker.stop()


# ------------------------------------------------------------------ the DINOv2 rewrite


def pos_graph(*, side=2, scales=None, resizes=1, constant=True, attr="value", table=None):
    """A tiny graph shaped like the export's position path: table -> Reshape -> Transpose ->
    Resize(scales); ``side`` x ``side`` positions of width 8."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    width = 8
    dims = table or [1, side * side, width]
    pos = numpy_helper.from_array(np.zeros(dims, np.float32), "pos")
    shape = numpy_helper.from_array(np.array([1, side, side, width], np.int64), "shape")
    trained = (224 // 14 + 0.1) / side
    values = np.array(scales if scales is not None else [1, 1, trained, trained], np.float32)
    nodes = [
        helper.make_node("Reshape", ["pos", "shape"], ["grid"]),
        helper.make_node("Transpose", ["grid"], ["nchw"], perm=[0, 3, 1, 2]),
    ]
    inits = [pos, shape]
    if constant:
        tensor = numpy_helper.from_array(values)
        nodes.append(helper.make_node("Constant", [], ["scales"], **{attr: tensor}))
    else:
        inits.append(numpy_helper.from_array(values, "scales"))
    for k in range(resizes):
        nodes.append(helper.make_node("Resize", ["nchw", "", "scales"], [f"out{k}"]))
    outputs = [
        helper.make_tensor_value_info(f"out{k}", TensorProto.FLOAT, None) for k in range(resizes)
    ]
    if not resizes:
        outputs = [helper.make_tensor_value_info("nchw", TensorProto.FLOAT, None)]
    graph = helper.make_graph(nodes, "pos", [], outputs, inits)
    return onnx, helper.make_model(graph)


def scales_of(model):
    from onnx import numpy_helper

    (node,) = [n for n in model.graph.node if n.op_type == "Constant"]
    return numpy_helper.to_array(node.attribute[0].t)


def test_the_rewrite_sets_the_scales_for_the_new_grid():
    pytest.importorskip("onnxruntime")
    from roboemot.vision.onnx import rescale_positions

    _, model = pos_graph(side=37)
    rescale_positions(model, 112, "m")
    assert scales_of(model) == pytest.approx([1, 1, 8.1 / 37, 8.1 / 37])


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"resizes": 0}, "one position-embedding Resize"),
        ({"resizes": 2}, "one position-embedding Resize"),
        ({"constant": False}, "no constant scales"),
        ({"attr": "value_floats"}, "not a constant tensor"),
        ({"scales": [1, 1, 8.05]}, "not the 224 px ones"),
        ({"scales": [1, 1, 2.0, 2.0]}, "not the 224 px ones"),
        ({"table": [1, 5, 8]}, "no square position table"),
    ],
)
def test_the_rewrite_refuses_a_graph_of_another_shape(kw, message):
    pytest.importorskip("onnxruntime")
    from roboemot.vision.models import ModelError
    from roboemot.vision.onnx import rescale_positions

    _, model = pos_graph(**kw)
    with pytest.raises(ModelError, match=message):
        rescale_positions(model, 112, "m")


def test_a_wrong_token_count_is_refused():
    pytest.importorskip("onnxruntime")
    from roboemot.vision.models import ModelError
    from roboemot.vision.onnx import check_tokens

    class Session:
        def __init__(self, tokens):
            self.tokens = tokens

        def run(self, outputs, feed):
            return [np.zeros((1, self.tokens, 384), np.float32)]

    check_tokens(Session(65), "x", 112, "m")  # 8 x 8 patches plus CLS
    with pytest.raises(ModelError, match="257 tokens at 112 px"):
        check_tokens(Session(257), "x", 112, "m")


def test_the_embedder_session_does_not_spin_when_asked(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx as backends

    seen = []

    def session(model, options, providers):
        seen.append(options.get_session_config_entry(backends.NO_SPINNING))
        raise RuntimeError("stop here")

    monkeypatch.setattr(ort, "InferenceSession", session)
    with pytest.raises(RuntimeError):
        backends._session(b"", 1, spinning=False)
    assert seen == ["0"]
