"""SPEC 5 (Hand check), 11A, 12, 12A: the fist rule on the hand model's 21 points, the hold that
turns a held fist into one percept.hand event, the hand check that runs only while asked for, at
vision.hands.rate_hz, and the hand model only on a palm, with counts only in the report."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.bus import TOPICS
from roboemot.config.schema import HandsConfig, RoboEmotConfig
from roboemot.vision import Frame, models
from roboemot.vision import hands as hands_module
from roboemot.vision.governor import CIRCADIAN
from roboemot.vision.hands import (
    FIST,
    OPEN,
    UNKNOWN,
    FistHold,
    HandGesture,
    HandWatch,
    classify,
    finger_ratios,
)
from roboemot.vision.models import ModelError
from roboemot.vision.stage_timing import HAND_PALM, HAND_POSE, StageTimer
from roboemot.vision.types import HandPose, Palm
from test_vision import H, W, rig

HANDS = HandsConfig(enabled=True)
GAP = 1 / HANDS.rate_hz + HANDS.max_gap_s
CACHE = models.default_dir()  # read at import: the autouse fixture moves the data folder

# A synthetic right hand in its own frame (palm length 1): finger bases across the palm, each
# finger straight up (+y) from its base, bending towards the palm side (-z) at each joint.
BASES = {1: (-0.35, 0.25), 5: (-0.2, 1.0), 9: (0.0, 1.0), 13: (0.18, 0.95), 17: (0.33, 0.85)}
BONES = (0.45, 0.28, 0.22)  # proximal, middle, distal
THUMB_BONES = (0.35, 0.3, 0.25)


def hand(bend: tuple[float, float, float], fingers: tuple[int, ...] = (5, 9, 13, 17)) -> np.ndarray:
    """The 21 points with the ``fingers`` (by base index) bent by ``bend`` degrees at their base,
    middle and end joints, the others straight; the thumb lies along the palm."""
    pts = np.zeros((21, 3))
    for base, (x, y) in BASES.items():
        angles = bend if base in fingers else (0.0, 0.0, 0.0)
        bones = THUMB_BONES if base == 1 else BONES
        p = np.array([x, y, 0.0])
        pts[base] = p
        theta = 0.0
        for k, (bone, angle) in enumerate(zip(bones, angles, strict=True)):
            theta += math.radians(angle)
            p = p + bone * np.array([0.0, math.cos(theta), -math.sin(theta)])
            pts[base + 1 + k] = p
    return pts


OPEN_HAND = hand((0, 0, 0))
FIST_HAND = hand((90, 100, 70))
HALF_CLOSED = hand((0, 90, 0))  # fingers bent at the middle joint only (a claw)
VICTORY = hand((90, 100, 70), fingers=(13, 17))  # index and middle up


def rotation(seed: int) -> np.ndarray:
    """A random 3D rotation (QR of a Gaussian matrix, sign-fixed to det +1)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def moved(points: np.ndarray, seed: int, scale: float) -> np.ndarray:
    return scale * points @ rotation(seed).T + np.random.default_rng(seed).normal(size=3)


# ------------------------------------------------------------------------ the fist rule


def test_a_fist_an_open_hand_and_in_between():
    assert classify(FIST_HAND, 0.99, HANDS) == FIST
    assert classify(OPEN_HAND, 0.99, HANDS) == OPEN
    assert classify(HALF_CLOSED, 0.99, HANDS) == UNKNOWN
    assert classify(VICTORY, 0.99, HANDS) == UNKNOWN
    ratios = finger_ratios(HALF_CLOSED)
    assert np.all((ratios > HANDS.fist_ratio_max) & (ratios < HANDS.open_ratio_min))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("scale", [0.01, 1.0, 250.0])
def test_the_rule_does_not_depend_on_rotation_scale_or_position(seed, scale):
    for points in (FIST_HAND, OPEN_HAND, HALF_CLOSED):
        assert np.allclose(finger_ratios(moved(points, seed, scale)), finger_ratios(points))
    assert classify(moved(FIST_HAND, seed, scale), 0.99, HANDS) == FIST
    assert classify(moved(OPEN_HAND, seed, scale), 0.99, HANDS) == OPEN
    assert classify(moved(HALF_CLOSED, seed, scale), 0.99, HANDS) == UNKNOWN


def test_the_thumb_is_ignored():
    thumb_out = FIST_HAND.copy()
    thumb_out[2:5] = thumb_out[1] + np.array([[0, 0.4, 0], [0, 0.8, 0], [0, 1.2, 0]])
    assert classify(thumb_out, 0.99, HANDS) == FIST


def test_an_unsure_model_or_bad_points_are_unknown():
    assert classify(FIST_HAND, HANDS.presence_min - 0.01, HANDS) == UNKNOWN
    assert classify(FIST_HAND, HANDS.presence_min, HANDS) == FIST
    assert classify(FIST_HAND[:, :2], 0.99, HANDS) == UNKNOWN
    bad = FIST_HAND.copy()
    bad[8, 0] = np.nan
    assert classify(bad, 0.99, HANDS) == UNKNOWN
    assert classify(np.zeros((21, 3)), 0.99, HANDS) == UNKNOWN  # every PIP on the wrist


def test_the_thresholds_are_config_fields():
    loose = HANDS.model_copy(update={"fist_ratio_max": 1.1, "open_ratio_min": 1.5})
    assert classify(HALF_CLOSED, 0.99, loose) == FIST  # 1.06: curled for the looser rule
    assert classify(OPEN_HAND, 0.99, loose) == UNKNOWN  # 1.34 is not open any more


# ------------------------------------------------------------------------ the hold


def test_a_fist_held_for_fist_hold_s_fires_once_and_rearms_after_a_non_fist():
    hold, period = FistHold(), 1 / HANDS.rate_hz
    seen = []
    for k, fist in enumerate([True] * 5 + [False] + [True] * 3):
        held = hold.observe(k * period, fist, HANDS.fist_hold_s, GAP)
        if held is not None:
            seen.append((k, held))
    assert seen == [(2, pytest.approx(HANDS.fist_hold_s)), (8, pytest.approx(HANDS.fist_hold_s))]


def test_a_break_in_the_fist_starts_the_hold_again():
    hold = FistHold()
    assert hold.observe(0.0, True, 1.0, 2.0) is None
    assert hold.observe(0.5, True, 1.0, 2.0) is None
    assert hold.observe(0.9, False, 1.0, 2.0) is None  # unknown, no palm or a failure
    assert hold.observe(1.0, True, 1.0, 2.0) is None
    assert hold.observe(1.5, True, 1.0, 2.0) is None
    assert hold.observe(2.0, True, 1.0, 2.0) == pytest.approx(1.0)


def test_fist_checks_further_apart_than_the_hold_are_not_one_hold():
    hold = FistHold()
    assert hold.observe(0.0, True, 1.0, 2.0) is None
    assert hold.observe(600.0, True, 1.0, 2.0) is None  # the camera paused in between
    assert hold.observe(600.5, True, 1.0, 2.0) is None
    assert hold.observe(601.0, True, 1.0, 2.0) == pytest.approx(1.0)


# ------------------------------------------------------------------------ the check


class FakePalm:
    def __init__(self, score: float | None = 0.9, raises: bool = False):
        self.score, self.raises, self.calls = score, raises, 0
        self.images: list[np.ndarray] = []

    def best(self, image):
        self.calls += 1
        self.images.append(image)
        if self.raises:
            raise RuntimeError("backend down")
        if self.score is None:
            return None
        return Palm((10.0, 10.0, 60.0, 60.0), self.score, np.zeros((7, 2)))


class FakePose:
    def __init__(self, points=FIST_HAND, presence: float = 0.99):
        self.points, self.presence, self.calls = points, presence, 0

    def estimate(self, image, palm):
        self.calls += 1
        return HandPose(self.presence, self.points)


def image() -> np.ndarray:
    return np.zeros((H, W, 3), np.uint8)


def feed(watch, seconds: float, wanted=True, start: float = 0.0, dt: float = 1 / 30):
    """Frames at ``dt`` for ``seconds``; the events with their frame stamps."""
    events = []
    for i in range(round(seconds / dt)):
        stamp = start + i * dt
        events += watch.step(stamp, image(), wanted)
    return events


def test_no_palm_means_no_hand_model_run():
    timer = StageTimer(30)
    palm, pose = FakePalm(score=None), FakePose()
    watch = HandWatch(HANDS, palm, pose, timer)
    feed(watch, 2.0)
    snap = watch.snapshot(2.0)
    assert snap["checks"] == palm.calls == 4 and pose.calls == 0
    assert (snap["palms"], snap["pose_runs"], snap["fists"]) == (0, 0, 0)
    timing = timer.snapshot(30)
    assert timing[HAND_PALM]["runs"] == 4 and timing[HAND_POSE]["runs"] == 0


def test_a_palm_below_palm_score_min_is_no_palm():
    palm, pose = FakePalm(score=HANDS.palm_score_min - 0.01), FakePose()
    watch = HandWatch(HANDS, palm, pose)
    assert feed(watch, 3.0) == [] and pose.calls == 0 and palm.calls == 6


def test_checks_run_at_rate_hz_only_while_wanted():
    palm = FakePalm(score=None)
    watch = HandWatch(HANDS, palm, FakePose())
    feed(watch, 2.0, wanted=False)
    assert palm.calls == 0 and watch.snapshot(2.0)["state"] == "idle"
    feed(watch, 2.0, start=2.0)
    assert palm.calls == round(2.0 * HANDS.rate_hz)
    assert watch.snapshot(4.0)["state"] == "watching"
    faster = HANDS.model_copy(update={"rate_hz": 5.0})
    watch.cfg = faster  # hot reload
    feed(watch, 2.0, start=4.0)
    assert palm.calls == round(2.0 * HANDS.rate_hz) + round(2.0 * faster.rate_hz)


def test_a_held_fist_makes_one_event_with_the_hold_time():
    timer = StageTimer(30)
    palm, pose = FakePalm(), FakePose(FIST_HAND)
    watch = HandWatch(HANDS, palm, pose, timer)
    events = feed(watch, 3.0)
    assert len(events) == 1
    stamp, event = events[0]
    assert event == HandGesture(FIST, pytest.approx(HANDS.fist_hold_s))
    assert stamp == pytest.approx(HANDS.fist_hold_s)
    snap = watch.snapshot(3.0)
    assert (snap["checks"], snap["palms"], snap["pose_runs"], snap["fists"]) == (6, 6, 6, 6)
    assert snap["events"] == 1
    assert timer.snapshot(30)[HAND_POSE]["runs"] == 6


def test_an_open_hand_or_an_unsure_model_never_fires():
    for pose in (FakePose(OPEN_HAND), FakePose(HALF_CLOSED), FakePose(FIST_HAND, presence=0.1)):
        assert feed(HandWatch(HANDS, FakePalm(), pose), 3.0) == []


def test_turning_the_check_off_ends_the_hold():
    watch = HandWatch(HANDS, FakePalm(), FakePose(FIST_HAND))
    assert feed(watch, 0.9) == []  # two fist checks
    feed(watch, 0.5, wanted=False, start=0.9)
    assert feed(watch, 0.9, start=1.4) == []  # a new run: two more checks are not enough
    assert len(feed(watch, 0.5, start=2.3)) == 1


def test_off_without_models_or_with_the_switch_off():
    off = HandWatch(HANDS)
    assert feed(off, 1.0) == [] and off.snapshot(1.0)["state"] == "off"
    palm = FakePalm()
    disabled = HandWatch(HandsConfig(enabled=False), palm, FakePose())
    assert feed(disabled, 1.0) == [] and palm.calls == 0
    missing = HandWatch(HANDS, unavailable="ModelError: mediapipe-palm: not verified")
    snap = missing.snapshot(None)
    assert snap["state"] == "unavailable" and "mediapipe-palm" in snap["reason"]


def test_a_failing_backend_is_a_failed_check_logged_by_type(caplog):
    watch = HandWatch(HANDS, FakePalm(raises=True), FakePose())
    assert feed(watch, 1.0) == []
    snap = watch.snapshot(1.0)
    assert snap["checks"] == 2 and snap["failed"] == 2
    assert "RuntimeError" in caplog.text and "backend down" not in caplog.text


def test_the_threaded_check_hands_over_a_copy_and_never_waits():
    palm = FakePalm()
    watch = HandWatch(HANDS, palm, FakePose(FIST_HAND), threaded=True)
    frame = image()
    assert watch.step(0.0, frame, True) == []
    assert palm.calls == 0  # nothing ran on the calling thread
    assert watch.step(0.5, frame, True) == []  # the first still waits: skipped
    assert watch.snapshot(0.5)["busy"] == 1
    assert watch.run_one() and palm.images[0] is not frame
    events = []
    for stamp in (0.5 + 1 / 30, 1.0, 1.5):  # checks at 0.53 and 1.5 (1.0 is not due yet)
        events += watch.step(stamp, frame, True)
        watch.run_one()
    assert events == [] and watch.snapshot(1.5)["checks"] == 3
    events += watch.step(1.6, frame, True)  # the 1.5 check lands: held from 0.0
    assert events == [(1.5, HandGesture(FIST, pytest.approx(1.5)))]


def test_a_result_after_the_check_was_turned_off_is_discarded():
    watch = HandWatch(HANDS, FakePalm(), FakePose(FIST_HAND), threaded=True)
    watch.step(0.0, image(), True)
    watch.run_one()
    watch.step(0.1, image(), False)  # off: a new generation
    watch.step(0.2, image(), True)
    snap = watch.snapshot(0.2)
    assert snap["discarded"] == 1 and snap["last_result_age_s"] is None


def test_the_thread_starts_runs_and_stops():
    watch = HandWatch(HANDS, FakePalm(), FakePose(FIST_HAND), threaded=True)
    watch.start()
    try:
        events = []
        for stamp in (0.0, 0.5, 1.0):
            events += watch.step(stamp, image(), True)
            assert watch.wait_idle(5.0)  # the result is ready once the thread is idle
        assert watch.snapshot(1.0)["checks"] == 3
        events += watch.step(1.1, image(), True)
        assert events == [(1.0, HandGesture(FIST, pytest.approx(1.0)))]
    finally:
        watch.stop()
    HandWatch(HANDS, threaded=True).start()  # without models: no thread, no error


# ------------------------------------------------------------------------ the pipeline


def hands_pipeline(tmp_path, palm, pose, **vision):
    cfg = RoboEmotConfig().vision.model_copy(
        update={"hands": HANDS.model_copy(update=vision), "objects_enabled": False}
    )
    _, bus, pipe, _, _ = rig(tmp_path, lambda i: [], vision=cfg, palm_detector=palm, hand_pose=pose)
    got = []
    bus.subscribe("percept.hand", got.append)
    return pipe, got


def run_frames(pipe, seconds: float, start: float = 0.0, dt: float = 1 / 30):
    for i in range(round(seconds / dt)):
        pipe.process(Frame(image(), start + i * dt))


def test_percept_hand_is_a_bus_topic():
    assert "percept.hand" in TOPICS


def test_the_pipeline_publishes_a_held_fist_only_while_hands_wanted(tmp_path):
    palm = FakePalm()
    pipe, got = hands_pipeline(tmp_path, palm, FakePose(FIST_HAND))
    run_frames(pipe, 2.0)
    assert got == [] and palm.calls == 0  # the seam defaults to never
    pipe.hands_wanted = lambda: True
    run_frames(pipe, 2.0, start=2.0)
    assert len(got) == 1
    payload = got[0].payload
    assert payload == HandGesture(FIST, pytest.approx(HANDS.fist_hold_s))
    assert set(vars(payload)) == {"gesture", "held_s"}  # no image, box or point


def test_the_hand_check_ignores_the_governor_rates_but_stops_at_scale_zero(tmp_path):
    palm = FakePalm(score=None)
    pipe, _ = hands_pipeline(tmp_path, palm, FakePose())
    pipe.hands_wanted = lambda: True
    pipe.governor.set_scale(CIRCADIAN, 0.05)  # resting vision: few face frames
    run_frames(pipe, 2.0)
    assert palm.calls == round(2.0 * HANDS.rate_hz)
    pipe.governor.set_scale(CIRCADIAN, 0.0)
    run_frames(pipe, 2.0, start=2.0)
    assert palm.calls == round(2.0 * HANDS.rate_hz)


def test_a_failing_seam_never_costs_the_frame(tmp_path, caplog):
    pipe, _ = hands_pipeline(tmp_path, FakePalm(), FakePose())

    def boom():
        raise RuntimeError("radio state")

    pipe.hands_wanted = boom
    run_frames(pipe, 0.5)
    assert "hand check step failed: RuntimeError" in caplog.text
    assert pipe.stats()["hands"]["checks"] == 0


def test_the_report_block_holds_counts_only(tmp_path):
    pipe, _ = hands_pipeline(tmp_path, FakePalm(), FakePose(FIST_HAND))
    pipe.hands_wanted = lambda: True
    run_frames(pipe, 2.0)
    stats = pipe.stats()
    block = stats["hands"]
    assert block["state"] == "watching" and block["events"] == 1
    assert block["checks"] == 4 and block["pose_runs"] == 4
    assert block["last_result_age_s"] == pytest.approx(2.0 - 1 / 30 - 1.5)
    assert all(isinstance(v, (int, float, str)) or v is None for v in block.values())
    assert {HAND_PALM, HAND_POSE} <= set(stats["stage_timing"])
    text = json.dumps(stats)
    assert "landmark" not in text and "embedding" not in text
    pipe.reset()
    assert pipe.stats()["hands"]["checks"] == 0


def test_the_hand_check_is_off_by_default(tmp_path):
    _, _, pipe, _, _ = rig(tmp_path, lambda i: [])
    pipe.hands_wanted = lambda: True
    run_frames(pipe, 1.0)
    assert pipe.stats()["hands"]["state"] == "off"


# ------------------------------------------------------------------------ config


def test_hand_defaults():
    cfg = RoboEmotConfig().vision.hands
    assert cfg.enabled is True  # the radio's fist stop ships on (owner decision 2026-11-19)
    assert cfg.threads == 1
    assert (cfg.rate_hz, cfg.fist_hold_s, cfg.pose_model) == (2.0, 1.0, "fp32")
    assert cfg.fist_ratio_max < cfg.open_ratio_min


@pytest.mark.parametrize(
    "field,value",
    [
        ("rate_hz", 0.1),
        ("rate_hz", 11),
        ("threads", 0),
        ("threads", 5),
        ("palm_score_min", 1.1),
        ("presence_min", -0.1),
        ("fist_ratio_max", 0.2),
        ("open_ratio_min", 2.6),
        ("fist_hold_s", 0.1),
        ("fist_hold_s", 11),
        ("pose_model", "fp16"),
    ],
)
def test_hand_ranges(field, value):
    with pytest.raises(ValidationError):
        HandsConfig(**{field: value})


def test_open_must_be_above_fist():
    with pytest.raises(ValidationError, match="open_ratio_min"):
        HandsConfig(fist_ratio_max=1.2, open_ratio_min=1.2)


# ------------------------------------------------------------------------ loading


def _hands_rig(monkeypatch, tmp_path, missing=None):
    from roboemot.vision import onnx

    built: list[tuple] = []

    def file(root, name, index=0):
        if name == missing:
            raise ModelError(f"{name}: not verified in {root}; run `roboemot models fetch`")
        return tmp_path / f"{name}-{index}"

    monkeypatch.setattr(onnx, "configure", lambda threads: None)
    monkeypatch.setattr(onnx, "_file", file)
    monkeypatch.setattr(onnx, "YuNetDetector", lambda *a: "yunet")
    monkeypatch.setattr(onnx, "SFaceEmbedder", lambda *a: "sface")
    for name in ("MediaPipePalmDetector", "MediaPipeHandPose"):
        monkeypatch.setattr(
            onnx, name, lambda path, *a, n=name, **kw: built.append((n, path.name, a, kw)) or n
        )
    return onnx, built


def test_the_hand_models_load_only_with_the_switch_on(monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    onnx, built = _hands_rig(monkeypatch, tmp_path)
    off = HandsConfig(enabled=False)
    cfg = RoboEmotConfig().vision.model_copy(update={"objects_enabled": False, "hands": off})
    stages = onnx.load_stages(tmp_path, cfg, appearance=False)
    assert (stages.palm_detector, stages.hand_pose, stages.hands_unavailable) == (None, None, None)
    assert built == []
    on = cfg.model_copy(update={"hands": HANDS})
    stages = onnx.load_stages(tmp_path, on, appearance=False)
    loaded = (stages.palm_detector, stages.hand_pose)
    assert loaded == ("MediaPipePalmDetector", "MediaPipeHandPose")
    spin = {"spinning": HANDS.spinning}
    assert built == [
        ("MediaPipePalmDetector", "mediapipe-palm-0", (HANDS.threads,), spin),
        ("MediaPipeHandPose", "mediapipe-handpose-0", (HANDS.threads,), spin),
    ]
    built.clear()
    three = on.model_copy(update={"hands": HANDS.model_copy(update={"threads": 3})})
    onnx.load_stages(tmp_path, three, appearance=False)
    assert [a for _, _, a, _ in built] == [(3,), (3,)]  # their own field, not vision.threads
    built.clear()
    int8 = on.model_copy(update={"hands": HANDS.model_copy(update={"pose_model": "int8"})})
    onnx.load_stages(tmp_path, int8, appearance=False)
    assert built[1][1] == "mediapipe-handpose-1"
    built.clear()
    assert onnx.load_stages(tmp_path, on, appearance=False, hands=False).hand_pose is None
    assert built == []


def test_a_missing_hand_model_keeps_vision_and_says_why(monkeypatch, tmp_path, caplog):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    onnx, _ = _hands_rig(monkeypatch, tmp_path, missing="mediapipe-handpose")
    cfg = RoboEmotConfig().vision.model_copy(update={"objects_enabled": False, "hands": HANDS})
    stages = onnx.load_stages(tmp_path, cfg, appearance=False)
    assert stages.face_detector == "yunet"
    assert stages.palm_detector is None and stages.hand_pose is None
    assert "mediapipe-handpose" in stages.hands_unavailable
    assert "hand check unavailable" in caplog.text


def test_the_hand_threads_reach_both_session_options(monkeypatch, tmp_path):
    """SPEC 5, 13: vision.hands.threads is each hand session's intra-op pool, not
    vision.threads."""
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from types import SimpleNamespace

    from roboemot.vision import onnx

    seen: list[tuple[str, int, int]] = []
    anchors = len(onnx.palm_anchors())
    outputs = {
        "palm": [
            SimpleNamespace(name="boxes", shape=[1, anchors, onnx.PALM_COLUMNS]),
            SimpleNamespace(name="scores", shape=[1, anchors, 1]),
        ],
        "handpose": [
            SimpleNamespace(name=onnx.HANDPOSE_PRESENCE, shape=[1, 1]),
            SimpleNamespace(name=onnx.HANDPOSE_POINTS, shape=[1, onnx.HANDPOSE_POINT_VALUES]),
        ],
    }

    def session(model, options, providers):
        seen.append((Path(model).name, options.intra_op_num_threads, options.inter_op_num_threads))
        kind = "palm" if "palm" in Path(model).name else "handpose"
        return SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input")],
            get_outputs=lambda: outputs[kind],
        )

    def side(sess, name):
        return onnx.PALM_SIZE if "palm" in name else onnx.HANDPOSE_SIZE

    monkeypatch.setattr(onnx.ort, "InferenceSession", session)
    monkeypatch.setattr(onnx, "_nhwc_side", side)
    monkeypatch.setattr(onnx, "_file", lambda root, name, index=0: tmp_path / f"{name}-{index}")
    vision = RoboEmotConfig().vision
    cfg = vision.hands.model_copy(update={"threads": 3})
    onnx.load_hands(tmp_path, cfg, cfg.threads)
    assert seen == [("mediapipe-palm-0", 3, 1), ("mediapipe-handpose-0", 3, 1)]
    assert vision.threads != cfg.threads


def test_the_installer_and_fetch_all_get_the_hand_models():
    """SPEC 4A Installer, 5: with vision the installer fetches every vision model, the hand
    models included (``models fetch --all``, or the names without the TTS voices)."""
    from roboemot.install import fetch_args

    assert {"mediapipe-palm", "mediapipe-handpose"} <= set(models.load_manifest())
    assert fetch_args(vision=True, tts=True) == "models fetch --all"
    names = fetch_args(vision=True, tts=False).split()
    assert "mediapipe-palm" in names and "mediapipe-handpose" in names
    assert "mediapipe-palm" not in fetch_args(vision=False, tts=False).split()


def test_the_hand_models_are_pinned_to_the_zoo_commit():
    specs = models.load_manifest()
    palm, pose = specs["mediapipe-palm"], specs["mediapipe-handpose"]
    assert (palm.role, pose.role) == ("hand_palm_detector", "hand_landmarker")
    assert palm.licence == pose.licence == "Apache-2.0"
    files = [*palm.files, *pose.files]
    assert [f.path.rsplit("/", 1)[1] for f in files] == [
        "palm_detection_mediapipe_2023feb_int8.onnx",
        "handpose_estimation_mediapipe_2023feb.onnx",
        "handpose_estimation_mediapipe_2023feb_int8.onnx",
    ]
    assert all("/raw/f88e9b2bafd21f1cad242fb5af6d78f2bcba16a3/" in f.url for f in files)
    assert [f.size_bytes for f in files] == [1157004, 4099621, 1167628]


def test_the_palm_anchors_and_decoding():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    anchors = onnx.palm_anchors()
    assert anchors.shape == (2016, 2)
    assert anchors[0] == pytest.approx([0.5 / 24, 0.5 / 24])
    assert anchors[1] == pytest.approx(anchors[0])  # two anchors per cell on the 24x24 grid
    assert anchors[1152] == pytest.approx([0.5 / 12, 0.5 / 12])
    assert np.all(anchors[1152:1158] == anchors[1152])  # six per cell on the 12x12 grid
    raw = np.zeros((2016, 18), np.float32)
    scores = np.full(2016, -10.0, np.float32)
    scores[1152] = 0.0
    raw[1152, :4] = [2.0, -2.0, 40.0, 20.0]
    raw[1152, 4:6] = [1.0, 1.0]
    box, keypoints, score = onnx.decode_palm(raw, scores, anchors)
    centre = anchors[1152] * 192
    assert score == pytest.approx(0.5)
    assert box == pytest.approx([centre[0] - 18, centre[1] - 12, centre[0] + 22, centre[1] + 8])
    assert keypoints[0] == pytest.approx(centre + 1) and keypoints[1] == pytest.approx(centre)


def test_the_hand_crop_turns_the_hand_upright_and_frames_it():
    pytest.importorskip("cv2")
    from roboemot.vision import onnx

    # a palm lying on its side: the fingers (middle base) point right of the wrist
    keypoints = np.array([[100, 200], [180, 170], [190, 200], [185, 230], [175, 255], [110, 160]])
    keypoints = np.vstack([keypoints, [[120, 150]]]).astype(np.float32)
    palm = Palm((100.0, 150.0, 200.0, 260.0), 0.9, keypoints)
    m = onnx.hand_crop(palm)
    mapped = keypoints @ m[:, :2].T + m[:, 2]
    wrist, middle = mapped[0], mapped[2]
    assert abs(middle[0] - wrist[0]) < 1e-3 and middle[1] < wrist[1]  # fingers up
    assert np.all((mapped > 0) & (mapped < 224))


# ------------------------------------------------------------------------ real models


@pytest.mark.models
@pytest.mark.parametrize("variant", ["fp32", "int8"])
def test_the_real_hand_models_load_and_run(variant):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    specs = models.load_manifest()
    if any(models.status(specs[n], CACHE) != "ok" for n in (onnx.PALM_NAME, onnx.HANDPOSE_NAME)):
        pytest.skip(f"hand models missing in {CACHE} (roboemot models fetch --all)")
    cfg = HANDS.model_copy(update={"pose_model": variant})
    palm_detector, pose = onnx.load_hands(CACHE, cfg, cfg.threads)
    grey = np.full((720, 1280, 3), 128, np.uint8)
    palm = palm_detector.best(grey)
    assert palm is not None and palm.score < cfg.palm_score_min  # nothing in a grey frame
    assert palm.keypoints.shape == (7, 2) and np.all(np.isfinite(palm.keypoints))
    noise = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    centred = Palm((540.0, 260.0, 740.0, 460.0), 1.0, palm.keypoints * 0 + [640.0, 360.0])
    centred.keypoints[2] = [640.0, 300.0]  # middle finger base above the wrist
    hand = pose.estimate(noise, centred)
    assert hand.points.shape == (21, 3) and np.all(np.isfinite(hand.points))
    assert 0.0 <= hand.presence <= 1.0
    assert classify(hand.points, hand.presence, cfg) in (FIST, OPEN, UNKNOWN)


# ------------------------------------------------------------------------ review follow-ups


@pytest.mark.parametrize("rate_hz", [0.2, 0.5, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("hold_s", [0.2, 1.0, 10.0])
@pytest.mark.parametrize("dt", [1 / 30, 0.3])  # a fast camera, and frames that delay each check
def test_a_held_fist_fires_once_at_every_rate_and_hold(rate_hz, hold_s, dt):
    """Regression: the gap limit once was fist_hold_s, so a check period at or above the hold
    (0.5 s checks with a 0.2 s hold, rate_hz 0.5, a check delayed by a frame) never fired."""
    cfg = HANDS.model_copy(update={"rate_hz": rate_hz, "fist_hold_s": hold_s})
    watch = HandWatch(cfg, FakePalm(), FakePose(FIST_HAND))
    period = max(1 / rate_hz, dt)
    events = feed(watch, hold_s + 3 * period + 3 * dt, dt=dt)
    assert len(events) == 1
    ((stamp, event),) = events
    assert hold_s - 1e-6 <= event.held_s <= hold_s + period + dt + 1e-6
    assert stamp == pytest.approx(event.held_s)  # the hold started at the first check, 0


def test_slow_checks_on_a_busy_cpu_still_make_one_hold():
    """Checks that take three periods each (the thread busy, frames skipped) stay one hold
    within one period plus vision.hands.max_gap_s."""
    watch = HandWatch(HANDS, FakePalm(), FakePose(FIST_HAND), threaded=True)
    events, dt = [], 0.25
    for i in range(int(4.0 / dt)):
        events += watch.step(i * dt, image(), True)
        if i % 6 == 0:  # the thread finishes a check only every 1.5 s
            watch.run_one()
    events += watch.step(4.0, image(), True)
    # checks at 0, 0.5 and 1.75: 1.25 s apart, more than the 1 s hold, still one hold
    assert [(s, e.held_s) for s, e in events] == [(1.75, pytest.approx(1.75))]
    assert watch.snapshot(4.0)["busy"] > 0


def test_a_gap_beyond_the_period_plus_max_gap_s_breaks_the_hold():
    hold, gap = FistHold(), 1 / HANDS.rate_hz + HANDS.max_gap_s
    assert hold.observe(0.0, True, 1.0, gap) is None
    assert hold.observe(gap + 0.01, True, 1.0, gap) is None  # too late: a new run
    assert hold.observe(gap + 0.5, True, 1.0, gap) is None
    assert hold.observe(gap + 1.01, True, 1.0, gap) == pytest.approx(1.0)
    hold.clear()
    assert hold.observe(0.0, True, 1.0, gap) is None
    assert hold.observe(gap, True, 1.0, gap) == pytest.approx(gap)  # just inside: one hold


def test_a_hot_reload_of_the_hold_and_the_rate_applies_to_the_run_in_progress():
    watch = HandWatch(HANDS, FakePalm(), FakePose(FIST_HAND))
    assert watch.step(0.0, image(), True) == []
    watch.cfg = HANDS.model_copy(update={"fist_hold_s": 0.5})
    assert watch.step(0.5, image(), True) == [(0.5, HandGesture(FIST, pytest.approx(0.5)))]
    watch.step(0.9, image(), False)
    watch.cfg = HANDS
    assert watch.step(1.0, image(), True) == []
    watch.cfg = HANDS.model_copy(update={"rate_hz": 0.5})  # next check 2 s later
    assert watch.step(1.5, image(), True) == []  # not due at the new rate
    assert watch.step(3.0, image(), True) == [(3.0, HandGesture(FIST, pytest.approx(2.0)))]


def test_stop_during_a_running_check_is_bounded_and_a_late_step_queues_nothing(caplog):
    gate, entered = threading.Event(), threading.Event()

    class SlowPalm(FakePalm):
        def best(self, image):
            entered.set()
            gate.wait(5.0)
            return super().best(image)

    watch = HandWatch(HANDS, SlowPalm(), FakePose(FIST_HAND), threaded=True, join_timeout_s=0.1)
    watch.start()
    try:
        watch.step(0.0, image(), True)
        assert entered.wait(5.0)
        t0 = time.monotonic()
        watch.stop()
        assert time.monotonic() - t0 < 2.0
        assert "did not stop" in caplog.text
        watch.step(0.5, image(), True)
        assert watch._job is None  # nothing queued after stop
    finally:
        gate.set()
    assert watch.wait_idle(5.0)
    thread = watch._thread
    assert thread is not None
    thread.join(5.0)
    assert not thread.is_alive()


def test_the_vision_service_starts_and_joins_the_hand_thread():
    from roboemot.bus import Bus
    from roboemot.vision.service import open_perception
    from test_perception import Source, fake_stages

    def stages(cfg, profile):
        s = fake_stages(cfg, profile)
        s.palm_detector, s.hand_pose = FakePalm(score=None), FakePose()
        return s

    base = RoboEmotConfig()
    cfg = base.model_copy(update={"vision": base.vision.model_copy(update={"hands": HANDS})})
    p = open_perception(cfg, Bus(), Source(), load_stages=stages)
    assert p.service is not None and p.service.hands is p.pipeline.hands
    p.start()
    try:
        assert any(t.name == "roboemot-hands" and t.is_alive() for t in threading.enumerate())
    finally:
        p.stop()
    assert not any(t.name == "roboemot-hands" and t.is_alive() for t in threading.enumerate())


def test_no_image_or_box_in_a_repr():
    job_repr = repr(hands_module._Job(np.full((2, 2, 3), 7, np.uint8), 0.0, 1))
    assert "array" not in job_repr and "image" not in job_repr
    palm_repr = repr(Palm((11.0, 12.0, 13.0, 14.0), 0.9, np.zeros((7, 2))))
    assert "11.0" not in palm_repr and "keypoints" not in palm_repr


def test_a_huge_palm_logit_does_not_overflow():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    anchors = onnx.palm_anchors()
    raw = np.zeros((2016, 18), np.float32)
    for logit, expected in ((-1e6, 0.0), (1e6, 1.0)):
        scores = np.full(2016, logit, np.float32)
        _, _, score = onnx.decode_palm(raw, scores, anchors)
        assert score == pytest.approx(expected, abs=1e-12)


def test_the_hand_models_load_only_for_the_radio_fist_stop(monkeypatch, tmp_path, caplog):
    """SPEC 5, Hand check: the models load only with vision.hands.enabled, tools.radio.enabled
    and tools.radio.fist_stop all on; otherwise no session, no thread, no warning, state off."""
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision.service import hands_load_wanted

    onnx, built = _hands_rig(monkeypatch, tmp_path)
    radio_on = {"tools": {"radio": {"enabled": True}}}
    assert hands_load_wanted(RoboEmotConfig.model_validate(radio_on)) is True
    for off in (
        {},  # the default: the radio is an admin opt-in
        {"tools": {"radio": {"enabled": True, "fist_stop": False}}},
        {**radio_on, "vision": {"hands": {"enabled": False}}},
    ):
        cfg = RoboEmotConfig.model_validate(off)
        assert hands_load_wanted(cfg) is False
        stages = onnx.load_stages(
            tmp_path, cfg.vision, appearance=False, objects=False, hands=hands_load_wanted(cfg)
        )
        assert built == [] and stages.hands_unavailable is None
        watch = HandWatch(cfg.vision.hands, stages.palm_detector, stages.hand_pose, threaded=True)
        watch.start()
        assert watch.snapshot(None)["state"] == "off"
        assert not any(t.name == "roboemot-hands" for t in threading.enumerate())
    assert "hand check" not in caplog.text


def test_missing_hand_models_keep_vision_up_and_never_ask_the_seam(monkeypatch, tmp_path):
    """SPEC 5, Hand check: the default config with the radio on and no hand model files: the
    pipeline runs, the report says unavailable, and hands_wanted is never consulted."""
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision.service import hands_load_wanted

    onnx, _ = _hands_rig(monkeypatch, tmp_path, missing="mediapipe-palm")
    cfg = RoboEmotConfig.model_validate({"tools": {"radio": {"enabled": True}}})
    vision = cfg.vision.model_copy(update={"objects_enabled": False})
    stages = onnx.load_stages(tmp_path, vision, appearance=False, hands=hands_load_wanted(cfg))
    assert "mediapipe-palm" in stages.hands_unavailable
    _, bus, pipe, _, _ = rig(
        tmp_path,
        lambda i: [],
        vision=vision,
        palm_detector=stages.palm_detector,
        hand_pose=stages.hand_pose,
        hands_unavailable=stages.hands_unavailable,
    )

    def never() -> bool:
        raise AssertionError("hands_wanted consulted without the hand models")

    pipe.hands_wanted = never
    got = []
    bus.subscribe("percept.hand", got.append)
    run_frames(pipe, 2.0)
    assert got == []
    assert pipe.hands.snapshot(2.0)["state"] == "unavailable"
