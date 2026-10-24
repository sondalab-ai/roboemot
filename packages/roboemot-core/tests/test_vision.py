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
    CameraModel,
    Detection,
    Frame,
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


def rig(tmp_path, script, *, vector=None, vision=None, **kw):
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
    person = reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    run(pipe, 24)
    rec = got["person.recognized"]
    assert len(rec) == 1
    assert rec[0].payload.person_id == person.person_id
    assert rec[0].stamp - 0.0 <= 2.5
    assert rec[0].stamp == pytest.approx(0.625)


def test_published_payloads_carry_no_embeddings(tmp_path):
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)])
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
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


def test_small_faces_are_ignored(tmp_path):
    _, _, pipe, emb, got = rig(tmp_path, lambda i: [face(100, size=CFG.vision.face_min_px - 1)])
    run(pipe, 8)
    assert got["percept.face"] == [] and emb.stamps == []


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
    _, bus, pipe, _, _ = rig(tmp_path / name, script)
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
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
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
    pipe.process(Frame(image(1), 1.0))  # the longer lifetime is kept until this frame is folded in
    pipe.process(Frame(image(2), 2.0))
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


class RecordingAppearance:
    model_id = APP_MODEL

    def __init__(self):
        self.boxes = []

    def embed(self, img, box):
        self.boxes.append(tuple(box))
        return axis(3)


def crowd(n, label="cup"):
    return [Detection((10 + 50 * k, 100, 40 + 50 * k, 140), 0.8, label) for k in range(n)]


def entity_rig(tmp_path, dets, **update):
    cfg = CFG.vision.model_copy(update=update)
    app = RecordingAppearance()
    reg = registry(tmp_path)
    bus = Bus()
    pipe = VisionPipeline(
        cfg,
        CFG.identity,
        bus,
        Matcher(reg, CFG.identity),
        face_detector=ScriptedDetector(lambda i: [face(100)]),
        face_embedder=FixedFaceEmbedder(axis(0)),
        appearance_embedder=app,
        object_detector=ObjectMock(dets),
    )
    return pipe, app


def test_entities_have_their_own_embed_budget_and_never_take_the_faces_share(tmp_path):
    pipe, app = entity_rig(
        tmp_path, crowd(4), max_embeds_per_frame=1, max_entity_embeds_per_frame=1
    )
    reports = [pipe.process(Frame(image(i), float(i))) for i in range(10)]
    per_frame = [r.embeddings for r in reports]
    assert max(per_frame) == 2  # one face + one entity in the same frame
    assert len(app.boxes) <= len(reports)  # never more than one entity embed per frame
    assert len(app.boxes) >= 4


def test_a_crowd_of_entities_takes_turns_oldest_embedding_first(tmp_path):
    pipe, app = entity_rig(tmp_path, crowd(6))
    for i in range(8):
        pipe.process(Frame(image(i), float(i)))
    first_six = app.boxes[:6]
    assert len(set(first_six)) == 6, "a track was embedded twice before every track had a turn"
    assert all(s.last_embed is not None for s in pipe._entities.values())
    assert len(pipe._entities) == 6


def test_a_new_entity_is_not_starved_by_old_ones(tmp_path):
    objs = ObjectMock(crowd(5))
    cfg = CFG.vision
    reg = registry(tmp_path)
    app = RecordingAppearance()
    pipe = VisionPipeline(
        cfg,
        CFG.identity,
        Bus(),
        Matcher(reg, CFG.identity),
        face_detector=ScriptedDetector(lambda i: []),
        face_embedder=FixedFaceEmbedder(axis(0)),
        appearance_embedder=app,
        object_detector=objs,
    )
    for i in range(12):
        pipe.process(Frame(image(i), float(i)))
    objs.dets = [*crowd(5), Detection((400, 100, 440, 140), 0.8, "cup")]
    seen = len(app.boxes)
    for i in range(12, 16):
        pipe.process(Frame(image(i), float(i)))
    assert (400.0, 100.0, 440.0, 140.0) in app.boxes[seen : seen + 4]  # confirmed, then served


def test_entity_burst_evaluations_are_spaced_by_the_minimum_gap(tmp_path):
    rates = CFG.vision.rates.model_copy(update={"objects_hz": 4.0, "appearance_hz": 0.05})
    pipe, app = entity_rig(tmp_path, crowd(1), rates=rates, entity_burst_min_gap_s=1.0)
    stamps = []
    for i in range(24):
        before = len(app.boxes)
        pipe.process(Frame(image(i), i * 0.25))
        if len(app.boxes) > before:
            stamps.append(i * 0.25)
    assert len(stamps) >= 3
    assert all(b - a >= 1.0 for a, b in zip(stamps, stamps[1:], strict=False))


def test_unconfirmed_object_tracks_publish_nothing_and_cost_no_embedding(tmp_path):
    pipe, app = entity_rig(tmp_path, crowd(1))
    pipe.object_detector.dets = crowd(1)
    pipe.process(Frame(image(0), 0.0))
    pipe.object_detector.dets = []
    pipe.process(Frame(image(1), 1.0))
    pipe.process(Frame(image(2), 2.0))
    assert app.boxes == [] and pipe._entities == {}


def test_object_confirm_hits_one_publishes_at_the_first_sighting(tmp_path):
    pipe, app = entity_rig(tmp_path, crowd(1), object_confirm_hits=1)
    pipe.process(Frame(image(0), 0.0))
    assert len(app.boxes) == 1 and len(pipe._entities) == 1


def test_a_track_survives_the_return_to_full_rate_for_one_frame(tmp_path):
    objs = ObjectMock(crowd(1))
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [], object_detector=objs)
    pipe.governor.set_scale("circadian", 0.1)
    for stamp in (0.0, 10.0):
        pipe.process(Frame(image(int(stamp)), stamp))
    (track_id,) = pipe._entities
    pipe.governor.set_scale("circadian", 1.0)  # lifetime would drop from 30 s to 5 s
    pipe.process(Frame(image(25), 25.0))  # 15 s after the last sighting
    assert list(pipe._entities) == [track_id]
    pipe.process(Frame(image(26), 26.0))
    assert pipe.object_tracker.cfg.lost_s == CFG.vision.object_lost_s
    assert list(pipe._entities) == [track_id]


# ------------------------------------------------- report counters (SPEC 12A)


def test_track_churn_counts_created_ended_lifetime_and_rate(tmp_path):
    # a face for frames 0-9, gone until it expires, back as a new track at frames 30-39
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if i < 10 or 30 <= i < 40 else [])
    run(pipe, 45)
    tracks = pipe.stats()["tracks"]
    assert tracks["created"] == 2 and tracks["ended"] == 1
    assert tracks["rate_window_s"] == CFG.vision.report_rate_window_s
    assert tracks["created_per_min"] == pytest.approx(2 * 60 / CFG.vision.report_rate_window_s)
    # first stamp 0 (tentative), last matched at frame 9 (1.125 s); the expiry tail is excluded
    assert tracks["lifetime_s"] == {"count": 1, "p50": 1.125, "p90": 1.125, "max": 1.125}
    json.dumps(pipe.stats())


def test_no_age_or_gender_anywhere_in_the_pipeline(tmp_path):
    """Owner decision 2026-10-23: no age model, no attribute worker, nothing estimated."""
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)])
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    run(pipe, 24)
    removed = {"age_group", "age_source", "gender", "gender_source", "attribute_confidence"}
    for topic in ("percept.face", "person.recognized"):
        assert got[topic], topic
        payload_fields = {f.name for m in got[topic] for f in fields(m.payload)}
        assert not removed & payload_fields, topic
    assert "attributes" not in pipe.stats() and not hasattr(pipe, "worker")


def test_track_rate_counts_only_the_window_before_the_last_frame(tmp_path):
    vision = CFG.vision.model_copy(update={"report_rate_window_s": 10.0})
    on = [True]
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if on[0] else [], vision=vision)
    run(pipe, 6)
    assert pipe.stats()["tracks"]["created_per_min"] == pytest.approx(6.0)
    on[0] = False
    run(pipe, 2, start=20.0)
    tracks = pipe.stats()["tracks"]
    assert tracks["created_per_min"] == 0.0 and tracks["created"] == 1 and tracks["ended"] == 1


def test_reset_ends_the_live_tracks_in_the_report(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(20), face(200)])
    run(pipe, 4)
    pipe.reset()
    tracks = pipe.stats()["tracks"]
    assert tracks["created"] == 2 and tracks["ended"] == 2
    assert tracks["lifetime_s"]["count"] == 2


def test_lifetime_summary_keeps_the_last_report_window_tracks(tmp_path):
    vision = CFG.vision.model_copy(update={"report_window": 10})
    on = [True]
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)] if on[0] else [], vision=vision)
    for k in range(13):  # 3 frames with the face, then 2.5 s without: 12 tracks end
        on[0] = True
        run(pipe, 3, start=k * 3.0)
        on[0] = False
        run(pipe, 2, start=k * 3.0 + 0.5, dt=1.25)
    tracks = pipe.stats()["tracks"]
    assert tracks["ended"] == 12 and tracks["lifetime_s"]["count"] == 10


def test_report_window_settings_have_defaults_and_ranges():
    v = CFG.vision
    assert v.report_window == 200 and v.report_rate_window_s == 60.0
    for bad in ({"report_window": 5}, {"report_rate_window_s": 1.0}):
        with pytest.raises(ValidationError):
            RoboEmotConfig(vision=bad)
