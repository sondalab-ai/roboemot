"""SPEC 5, 6, 12, 13 vision pipeline with mock stages: no models, no camera."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, TrackerConfig
from roboemot.registry import Consent, FileStore, Matcher, Registry
from roboemot.vision import (
    CIRCADIAN,
    AttributeSample,
    CameraModel,
    Detection,
    Frame,
    InlineExecutor,
    LatestFrameRunner,
    RateGovernor,
    Tracker,
    VisionPipeline,
)
from roboemot.vision.replay import ReplayError, load_session, replay, summarize
from roboemot.world import FrameSpec, WorldModel, feed_world

CFG = RoboEmotConfig()
W, H = 320, 240
DIM = 8
FACE_MODEL = "mock-face"
APP_MODEL = "mock-app"
CONSENT = Consent("Test", datetime(2026, 1, 1, tzinfo=UTC), ("face", "refresh"))


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def face(x: float, y: float = 80, size: float = 60, score: float = 0.9) -> Detection:
    return Detection((x, y, x + size, y + size), score, landmarks=np.zeros((5, 2)))


def image(index: int = 0) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[0, 0, 0] = index
    return img


class ScriptedDetector:
    """Detections as a function of the call index (image[0,0,0] carries the frame index)."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return self.script(int(img[0, 0, 0]))


class FixedFaceEmbedder:
    model_id = FACE_MODEL

    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.stamps: list[float] = []
        self.stamp_of = None

    def embed(self, img, face_det):
        self.stamps.append(self.stamp_of() if self.stamp_of else math.nan)
        return self.vector.copy()


class CountingEstimator:
    def __init__(self, age="adult", gender="female", conf=0.95):
        self.sample = AttributeSample(age, conf, gender, conf)
        self.crops: list[tuple[int, ...]] = []

    def estimate(self, crop):
        self.crops.append(crop.shape)
        return self.sample


class CountingExecutor(InlineExecutor):
    def __init__(self):
        self.jobs = 0

    def submit(self, fn, /, *args, **kwargs):
        self.jobs += 1
        return super().submit(fn, *args, **kwargs)


class AppearanceMock:
    model_id = APP_MODEL

    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    def embed(self, img, box):
        self.calls += 1
        return self.vector.copy() if box[0] < W / 2 else axis(5)


class ObjectMock:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


def registry(tmp_path) -> Registry:
    return Registry(FileStore(tmp_path / "reg", allow_plaintext=True), CFG.identity)


def rig(tmp_path, script, *, vector=None, estimator=None, executor=None, vision=None, **kw):
    reg = registry(tmp_path)
    bus = Bus()
    matcher = Matcher(reg, CFG.identity)
    embedder = FixedFaceEmbedder(axis(0) if vector is None else vector)
    pipe = VisionPipeline(
        CFG.vision if vision is None else vision,
        CFG.identity,
        bus,
        matcher,
        face_detector=ScriptedDetector(script),
        face_embedder=embedder,
        attribute_estimator=estimator,
        executor=executor or InlineExecutor(),
        **kw,
    )
    got: dict[str, list] = {}
    for topic in ("percept.face", "percept.entity", "person.recognized"):
        got[topic] = []
        bus.subscribe(topic, lambda m, t=topic: got[t].append(m))
    return reg, bus, pipe, embedder, got


def run(pipe, n: int, dt: float = 0.125, start: float = 0.0, joint=None):
    for i in range(n):
        pipe.process(Frame(image(i), start + i * dt, joint))


# ------------------------------------------------------------------------ geometry


def test_bearing_center_right_top():
    cam = CameraModel(W, H, 90)
    assert cam.bearing(W / 2, H / 2) == (0.0, 0.0)
    az_right, el_right = cam.bearing(W, H / 2)
    assert az_right == pytest.approx(-math.radians(45))
    assert el_right == pytest.approx(0.0)
    az_left, _ = cam.bearing(0, H / 2)
    assert az_left == pytest.approx(math.radians(45))
    az_top, el_top = cam.bearing(W / 2, 0)
    assert az_top == pytest.approx(0.0)
    assert el_top > 0
    assert cam.bearing(W / 2, H)[1] == pytest.approx(-el_top)


def test_bearing_uses_the_ray_not_the_pixel_ratio():
    cam = CameraModel(W, H, 90)
    az, el = cam.bearing(W, 0)
    assert el < math.atan2(H / 2, W / 2)
    assert az == pytest.approx(-math.radians(45))


# ------------------------------------------------------------------------- tracker


def test_tracker_keeps_ids_while_faces_move():
    t = Tracker(CFG.vision.tracker)
    ids = {}
    for i in range(10):
        stamp = i * 0.125
        res = t.update([face(20 + 4 * i), face(220 - 4 * i)], stamp)
        for tr in res.visible:
            ids.setdefault(round(tr.box[0] / 100), set()).add(tr.track_id)
    assert all(len(v) == 1 for v in ids.values())
    assert len({next(iter(v)) for v in ids.values()}) == 2


def test_tracker_confirms_after_hits_and_reports_new_once():
    t = Tracker(CFG.vision.tracker)
    first = t.update([face(50)], 0.0)
    assert first.visible == [] and first.new_ids == set()
    second = t.update([face(51)], 0.125)
    assert len(second.visible) == 1 and second.new_ids == {second.visible[0].track_id}
    assert t.update([face(52)], 0.25).new_ids == set()


def test_tracker_drops_unconfirmed_flicker_and_removes_lost_by_stamp():
    t = Tracker(CFG.vision.tracker)
    t.update([face(50)], 0.0)
    t.update([], 0.125)
    res = t.update([face(50)], 0.25)
    assert res.visible == []
    t2 = Tracker(CFG.vision.tracker)
    t2.update([face(50)], 0.0)
    tid = t2.update([face(50)], 0.125).visible[0].track_id
    lost = CFG.vision.tracker.lost_s
    assert t2.update([], 0.125 + lost - 0.01).removed_ids == []
    assert t2.update([], 0.125 + lost + 0.01).removed_ids == [tid]


def test_tracker_low_score_detection_keeps_track_and_id_survives_a_gap():
    t = Tracker(CFG.vision.tracker)
    t.update([face(50)], 0.0)
    tid = t.update([face(52)], 0.125).visible[0].track_id
    weak = t.update([face(54, score=0.3)], 0.25)
    assert [tr.track_id for tr in weak.visible] == [tid]
    t.update([], 0.375)
    back = t.update([face(58)], 0.5)
    assert [tr.track_id for tr in back.visible] == [tid]


def test_tracker_low_score_alone_never_starts_a_track():
    t = Tracker(TrackerConfig(confirm_hits=1))
    assert t.update([face(50, score=0.3)], 0.0).visible == []


def test_tracker_config_rejects_inverted_scores():
    with pytest.raises(ValidationError):
        TrackerConfig(low_score=0.9, high_score=0.5)


# ------------------------------------------------------------------------- governor


def test_governor_drops_instead_of_queueing():
    g = RateGovernor(CFG.vision.rates)
    allowed = [i for i in range(60) if g.allow("detection", i / 30)]
    assert len(allowed) == pytest.approx(16, abs=1)
    assert g.dropped["detection"] == 60 - len(allowed)


def test_governor_scales_and_zero_turns_off():
    g = RateGovernor(CFG.vision.rates)
    g.set_scale(CIRCADIAN, 0.5)
    assert g.period("detection") == pytest.approx(2 / CFG.vision.rates.detection_hz)
    g.set_scale(CIRCADIAN, 0.0)
    assert math.isinf(g.period("detection"))
    assert not g.allow("detection", 0.0)
    g.set_scale(CIRCADIAN, 5)
    assert g.scale == 1.0


def test_pipeline_runs_detector_at_the_governed_rate(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [])
    run(pipe, 90, dt=1 / 30)
    assert pipe.face_detector.calls == pytest.approx(24, abs=2)
    pipe.governor.set_scale(CIRCADIAN, 0.0)
    before = pipe.face_detector.calls
    run(pipe, 10, start=10.0)
    assert pipe.face_detector.calls == before


# ------------------------------------------------------------------------- pipeline


def test_new_track_gets_a_burst_then_the_steady_rate(tmp_path):
    clock = {"t": 0.0}
    _, _, pipe, emb, _ = rig(tmp_path, lambda i: [face(100)])
    emb.stamp_of = lambda: clock["t"]
    for i in range(80):
        clock["t"] = i * 0.125
        pipe.process(Frame(image(i), clock["t"]))
    burst = CFG.vision.embedding_burst
    gap = CFG.vision.burst_min_gap_s
    assert emb.stamps[0] == 0.125
    assert np.diff(emb.stamps[:burst]).min() >= gap - 1e-9
    assert np.diff(emb.stamps[:burst]).max() <= gap + 0.125 + 1e-9
    period = 1 / CFG.vision.rates.embedding_hz
    gaps = np.diff(emb.stamps[burst - 1 :])
    assert (gaps >= period - 1e-9).all() and (gaps <= period + 0.125 + 1e-9).all()


def test_identity_decision_inside_the_cm4_budget(tmp_path):
    reg, _, pipe, emb, got = rig(tmp_path, lambda i: [face(100)])
    person = reg.enroll(
        "Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult"
    )
    run(pipe, 24)
    rec = got["person.recognized"]
    assert len(rec) == 1
    assert rec[0].payload.person_id == person.person_id
    assert rec[0].stamp - 0.0 <= 2.5
    assert rec[0].stamp == pytest.approx(0.625)


def test_published_payloads_carry_no_embeddings(tmp_path):
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=CountingEstimator())
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult")
    run(pipe, 24)

    def arrays(obj):
        if isinstance(obj, np.ndarray):
            return True
        if is_dataclass(obj):
            return any(arrays(getattr(obj, f.name)) for f in fields(obj))
        if isinstance(obj, (list, tuple)):
            return any(arrays(x) for x in obj)
        return False

    for topic in ("percept.face", "person.recognized", "percept.entity"):
        assert got[topic], topic
        assert not any(arrays(m.payload) for m in got[topic]), topic
    assert all(m.payload.embedding is None for m in got["percept.entity"])


def test_face_percept_has_bearing_and_anonymous_id_until_recognised(tmp_path):
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(250, y=10, size=50)])
    run(pipe, 6)
    p = got["percept.face"][0].payload
    assert p.owner_id is None and p.anonymous_id.startswith("anon-")
    assert p.azimuth_rad < 0 and p.elevation_rad > 0
    assert p.new_track and not got["percept.face"][-1].payload.new_track
    assert got["person.recognized"] == []
    assert p.age_group == "unknown" and p.possible_minor


def test_small_faces_are_ignored(tmp_path):
    _, _, pipe, emb, got = rig(tmp_path, lambda i: [face(100, size=CFG.vision.face_min_px - 1)])
    run(pipe, 8)
    assert got["percept.face"] == [] and emb.stamps == []


def test_attribute_job_runs_once_per_track_on_n_crops(tmp_path):
    est, ex = CountingEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=est, executor=ex)
    run(pipe, 40)
    needed = CFG.vision.attribute_crops
    assert needed == 3
    assert ex.jobs == 1
    assert len(est.crops) == needed
    state = next(iter(pipe._faces.values()))
    assert state.crops == []
    early, late = got["percept.face"][0].payload, got["percept.face"][-1].payload
    assert early.age_group == "unknown" and early.possible_minor
    assert late.age_group == "adult" and not late.possible_minor
    assert late.gender == "female" and late.attributes_stamp is not None


@pytest.mark.parametrize("crops", [3, 5])
def test_attribute_crops_sets_the_crops_per_track(tmp_path, crops):
    vision = CFG.vision.model_copy(update={"attribute_crops": crops})
    est, ex = CountingEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)], estimator=est, executor=ex, vision=vision
    )
    run(pipe, 60)
    assert ex.jobs == 1 and len(est.crops) == crops
    assert pipe.worker.config is pipe.identity
    late = got["percept.face"][-1].payload
    assert (late.age_group, late.gender) == ("adult", "female")


def test_single_crop_with_frames_min_one_gives_a_confident_value(tmp_path):
    cfg = RoboEmotConfig(vision={"attribute_crops": 1}, identity={"estimate_frames_min": 1})
    est, ex = CountingEstimator(conf=0.9), CountingExecutor()
    reg = registry(tmp_path)
    pipe = VisionPipeline(
        cfg.vision,
        cfg.identity,
        Bus(),
        Matcher(reg, cfg.identity),
        face_detector=ScriptedDetector(lambda i: [face(100)]),
        face_embedder=FixedFaceEmbedder(axis(0)),
        attribute_estimator=est,
        executor=ex,
    )
    got = []
    pipe.bus.subscribe("percept.face", got.append)
    run(pipe, 20)
    assert ex.jobs == 1 and len(est.crops) == 1
    last = got[-1].payload
    assert (last.age_group, last.gender, last.possible_minor) == ("adult", "female", False)


def test_single_crop_still_applies_the_confidence_rule(tmp_path):
    cfg = RoboEmotConfig(vision={"attribute_crops": 1}, identity={"estimate_frames_min": 1})
    pipe = VisionPipeline(
        cfg.vision,
        cfg.identity,
        Bus(),
        Matcher(registry(tmp_path), cfg.identity),
        face_detector=ScriptedDetector(lambda i: [face(100)]),
        face_embedder=FixedFaceEmbedder(axis(0)),
        attribute_estimator=CountingEstimator(conf=0.5),
        executor=InlineExecutor(),
    )
    got = []
    pipe.bus.subscribe("percept.face", got.append)
    run(pipe, 20)
    assert got[-1].payload.age_group == "unknown" and got[-1].payload.possible_minor


def test_track_shorter_than_the_crops_needed_runs_no_job(tmp_path):
    est, ex = CountingEstimator(), CountingExecutor()
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)] if i < 3 else [], estimator=est, executor=ex
    )
    run(pipe, 60)
    assert ex.jobs == 0 and est.crops == []
    assert all(m.payload.age_group == "unknown" for m in got["percept.face"])


def test_attribute_crops_and_frames_min_validation():
    assert CFG.vision.attribute_crops == 3 and CFG.identity.estimate_frames_min == 3
    for bad in (0, 11):
        with pytest.raises(ValueError):
            RoboEmotConfig(vision={"attribute_crops": bad})
    with pytest.raises(ValueError, match="attribute_crops must be at least"):
        RoboEmotConfig(vision={"attribute_crops": 2})
    with pytest.raises(ValueError, match="attribute_crops must be at least"):
        RoboEmotConfig(identity={"estimate_frames_min": 5})
    RoboEmotConfig(vision={"attribute_crops": 5}, identity={"estimate_frames_min": 5})


def test_second_track_gets_its_own_job(tmp_path):
    est, ex = CountingEstimator(), CountingExecutor()

    def script(i):
        return [face(20)] + ([face(200)] if i >= 10 else [])

    _, _, pipe, _, _ = rig(tmp_path, script, estimator=est, executor=ex)
    run(pipe, 50)
    assert ex.jobs == 2


def test_low_confidence_attributes_stay_unknown_and_minor(tmp_path):
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)], estimator=CountingEstimator(conf=0.5))
    run(pipe, 40)
    last = got["percept.face"][-1].payload
    assert last.age_group == "unknown" and last.possible_minor


def test_declared_age_overrides_the_estimate(tmp_path):
    reg, _, pipe, _, got = rig(
        tmp_path, lambda i: [face(100)], estimator=CountingEstimator(age="child")
    )
    reg.enroll("Kid", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="senior")
    run(pipe, 40)
    last = got["percept.face"][-1].payload
    assert (last.age_group, last.age_source, last.possible_minor) == ("senior", "declared", False)


def test_worker_queue_is_bounded(tmp_path):
    release = threading.Event()

    class Slow(CountingEstimator):
        def estimate(self, crop):
            release.wait(5)
            return super().estimate(crop)

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(1)
    est = Slow()
    try:

        def script(i):
            return [face(20 + 60 * k) for k in range(4)]

        _, _, pipe, _, _ = rig(tmp_path, script, estimator=est, executor=pool)
        pipe.worker.queue_max = 2
        run(pipe, 30)
        assert pipe.worker._pending == 2
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_ended_tracks_release_matcher_state(tmp_path):
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)] if i < 6 else [])
    run(pipe, 6)
    tid = got["percept.face"][0].payload.track_id
    assert tid in pipe.matcher._tracks
    run(pipe, 40, start=1.0)
    assert tid not in pipe.matcher._tracks and tid not in pipe._faces


def test_animals_and_objects_publish_with_appearance_embedding(tmp_path):
    objs = ObjectMock(
        [
            Detection((20, 100, 80, 160), 0.8, "cat"),
            Detection((200, 100, 260, 160), 0.8, "cup"),
            Detection((120, 20, 180, 80), 0.9, "person"),
        ]
    )
    app = AppearanceMock(axis(1))
    cfg = CFG.vision.model_copy(update={"objects_enabled": True})
    reg = registry(tmp_path)
    cat = reg.enroll_entity("animal", "Tom", np.stack([axis(1)] * 5), APP_MODEL, category="cat")
    bus = Bus()
    pipe = VisionPipeline(
        cfg,
        CFG.identity,
        bus,
        Matcher(reg, CFG.identity),
        face_detector=ScriptedDetector(lambda i: []),
        face_embedder=FixedFaceEmbedder(axis(0)),
        appearance_embedder=app,
        object_detector=objs,
    )
    seen = []
    bus.subscribe("percept.entity", lambda m: seen.append(m.payload))
    run(pipe, 40, dt=0.25)
    kinds = {o.category: o for o in seen}
    assert set(kinds) == {"cat", "cup"}
    assert kinds["cat"].kind == "animal" and kinds["cup"].kind == "object"
    assert kinds["cat"].identity == cat.entity_id and kinds["cup"].identity is None
    assert kinds["cat"].embedding is not None
    assert app.calls < len(seen)


def test_objects_can_be_switched_off(tmp_path):
    objs = ObjectMock([Detection((20, 100, 80, 160), 0.8, "cat")])
    off = CFG.vision.model_copy(update={"objects_enabled": False})
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], vision=off, object_detector=objs)
    run(pipe, 20)
    assert got["percept.entity"] == []


def test_objects_are_on_by_default_and_defaults_cover_the_household_list():
    v = CFG.vision
    assert v.objects_enabled and v.rates.objects_hz == 1.0
    assert set(v.animal_categories) == {"bird", "cat", "dog", "horse", "sheep", "cow"}
    assert {"cup", "potted_plant", "cell_phone", "teddy_bear", "tv"} <= set(v.object_categories)
    assert "person" not in v.animal_categories + v.object_categories


def test_class_mapping_drops_unlisted_classes_and_person(tmp_path):
    objs = ObjectMock(
        [
            Detection((20, 100, 80, 160), 0.8, "cat"),
            Detection((200, 100, 260, 160), 0.8, "cup"),
            Detection((120, 20, 180, 80), 0.9, "person"),
            Detection((300, 20, 340, 80), 0.9, "car"),
        ]
    )
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], object_detector=objs)
    run(pipe, 20)
    kinds = {m.payload.category: m.payload.kind for m in got["percept.entity"]}
    assert kinds == {"cat": "animal", "cup": "object"}


def test_class_mapping_is_configurable(tmp_path):
    objs = ObjectMock(
        [Detection((20, 100, 80, 160), 0.8, "cat"), Detection((200, 100, 260, 160), 0.8, "cup")]
    )
    cfg = CFG.vision.model_copy(update={"animal_categories": ["cup"], "object_categories": []})
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], vision=cfg, object_detector=objs)
    run(pipe, 20)
    kinds = {m.payload.category: m.payload.kind for m in got["percept.entity"]}
    assert kinds == {"cup": "animal"}


def test_class_in_both_lists_is_rejected():
    with pytest.raises(ValidationError, match="both animal and object"):
        RoboEmotConfig.model_validate(
            {"vision": {"animal_categories": ["cat"], "object_categories": ["cat"]}}
        )


def test_object_score_threshold_is_applied(tmp_path):
    objs = ObjectMock(
        [Detection((20, 100, 80, 160), 0.39, "cat"), Detection((200, 100, 260, 160), 0.41, "cup")]
    )
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], object_detector=objs)
    run(pipe, 20)
    assert {m.payload.category for m in got["percept.entity"]} == {"cup"}


def test_world_model_gets_observations_over_the_bus_with_joint_state(tmp_path):
    _, bus, pipe, _, _ = rig(tmp_path, lambda i: [face(W / 2 - 30, H / 2 - 30)])
    world = WorldModel(CFG.world, FrameSpec(yaw_dofs=("head_yaw",)), bus)
    feed_world(bus, world)
    run(pipe, 6, joint={"head_yaw": 0.5})
    (entry,) = world.entries
    assert entry.kind == "person"
    assert entry.azimuth_rad == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------- runner


def test_runner_is_latest_wins():
    gate, done, started = threading.Event(), threading.Event(), threading.Event()
    seen: list[float] = []

    class Slow:
        name = "slow"

        def process(self, frame):
            seen.append(frame.stamp)
            if len(seen) == 1:
                started.set()
                gate.wait(5)
            if frame.stamp == 3.0:
                done.set()

    runner = LatestFrameRunner(Slow())
    runner.start()
    try:
        runner.submit(Frame(image(), 0.0))
        assert started.wait(5)
        for s in (1.0, 2.0, 3.0):
            runner.submit(Frame(image(), s))
        gate.set()
        assert done.wait(5)
    finally:
        runner.stop()
    assert seen == [0.0, 3.0] and runner.dropped == 2


# --------------------------------------------------------------------------- replay


def write_session(root, n: int) -> None:
    frames = []
    for i in range(n):
        np.save(root / f"{i:06d}.npy", image(i))
        frames.append(
            {
                "file": f"{i:06d}.npy",
                "stamp": i * 0.125,
                "joint_state": {"head_yaw": 0.1 * i},
                "labels": {"faces": [{"person_id": None}]},
            }
        )
    (root / "session.json").write_text(json.dumps({"version": 1, "frames": frames}))


def script(i):
    return [face(40 + i), face(200 - i)] if i < 20 else [face(100)]


def run_replay(tmp_path, session, name):
    _, bus, pipe, _, _ = rig(tmp_path / name, script, estimator=CountingEstimator())
    return replay(session, pipe, bus)


def test_replay_is_deterministic_and_stamped_from_the_recording(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    write_session(root, 40)
    session = load_session(root)
    a = run_replay(tmp_path, session, "a")
    b = run_replay(tmp_path, session, "b")
    assert a.messages and summarize(a.messages) == summarize(b.messages)
    stamps = {m.stamp for m in a.messages}
    assert stamps <= {f.stamp for f in session.frames}
    anon = {s["anonymous_id"] for s in summarize(a.messages) if s.get("anonymous_id")}
    assert anon <= {"anon-1", "anon-2", "anon-3"}
    assert a.frames[3].labels == {"faces": [{"person_id": None}]}


def test_replay_rejects_bad_sessions(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    with pytest.raises(ReplayError):
        load_session(root)
    (root / "session.json").write_text(
        json.dumps(
            {
                "version": 1,
                "frames": [{"file": "a.npy", "stamp": 1.0}, {"file": "b.npy", "stamp": 1.0}],
            }
        )
    )
    with pytest.raises(ReplayError):
        load_session(root)


def test_replay_against_one_registry_does_not_change_it(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    write_session(root, 40)
    session = load_session(root)
    reg, bus, pipe, _, _ = rig(tmp_path / "r", script)
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT, age_group="adult")
    before = reg.gallery("face", FACE_MODEL)
    first = summarize(replay(session, pipe, bus).messages)
    second = summarize(replay(session, pipe, bus).messages)
    after = reg.gallery("face", FACE_MODEL)
    assert any(m["topic"] == "person.recognized" for m in first)
    assert first == second
    assert {k: v.shape for k, v in before.items()} == {k: v.shape for k, v in after.items()}


def test_object_track_lifetime_follows_the_governor_scale(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [], object_detector=ObjectMock([]))
    cfg = CFG.vision
    assert pipe.object_tracker.cfg.lost_s == cfg.object_lost_s  # 1 Hz: 3 periods < 5 s
    pipe.governor.set_scale("circadian", 0.1)
    pipe.process(Frame(image(0), 0.0))
    period = 1 / (cfg.rates.objects_hz * 0.1)
    assert pipe.object_tracker.cfg.lost_s == pytest.approx(cfg.object_lost_periods * period)
    assert pipe.object_tracker.cfg.lost_s > cfg.object_lost_s
    pipe.governor.set_scale("circadian", 1.0)
    pipe.process(Frame(image(1), 1.0))
    assert pipe.object_tracker.cfg.lost_s == cfg.object_lost_s
    pipe.apply_config(cfg.model_copy(update={"object_lost_s": 9.0}), CFG.identity)
    assert pipe.object_tracker.cfg.lost_s == 9.0 and pipe.object_tracker.cfg.high_score == 0.4


def test_an_object_track_survives_a_missed_frame_at_a_tenth_of_the_rate(tmp_path):
    objs = ObjectMock([Detection((20, 100, 80, 160), 0.8, "cat")])
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], object_detector=objs)
    pipe.governor.set_scale("circadian", 0.1)  # one object frame every 10 s
    for stamp in (0.0, 10.0):
        pipe.process(Frame(image(int(stamp)), stamp))
    assert {m.payload.category for m in got["percept.entity"]} == {"cat"}
    objs.dets = []
    pipe.process(Frame(image(20), 20.0))  # missed frame: 10 s gap, within 3 periods
    assert len(pipe._entities) == 1
    objs.dets = [Detection((20, 100, 80, 160), 0.8, "cat")]
    pipe.process(Frame(image(30), 30.0))
    assert len(pipe._entities) == 1 and len(pipe.object_tracker._tracks) == 1


def test_the_object_tracker_has_no_low_score_pass(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [], object_detector=ObjectMock([]))
    cfg = pipe.object_tracker.cfg
    assert cfg.low_score == cfg.high_score == CFG.vision.object_score_min


def test_categories_are_readable_names(tmp_path):
    objs = ObjectMock(
        [
            Detection((20, 100, 80, 160), 0.8, "potted_plant"),
            Detection((200, 100, 260, 160), 0.8, "hair_drier"),
        ]
    )
    cfg = CFG.vision.model_copy(update={"object_categories": ["potted_plant", "hair_drier"]})
    _, _, pipe, _, got = rig(tmp_path, lambda i: [], vision=cfg, object_detector=objs)
    run(pipe, 20)
    assert {m.payload.category for m in got["percept.entity"]} == {"potted plant", "hair drier"}
