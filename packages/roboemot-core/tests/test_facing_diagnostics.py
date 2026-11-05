"""Facing check diagnostics (SPEC 8 stage 2, 12A): which face the check used and the numbers
behind its yaw, per-face yaw in the vision section, the frame facts, and the preview's landmark
dots (SPEC 11A). Recording never changes a decision; no image or embedding in any snapshot."""

from __future__ import annotations

import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.attention.gaze import GazeMap
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, VoiceDirectionConfig, VoiceFacingConfig
from roboemot.motion.state import MotionSample
from roboemot.registry import FileStore, Matcher, Registry
from roboemot.vision.facing import eye_span
from roboemot.vision.geometry import letterbox_scale
from roboemot.vision.pipeline import FACE_REPORT_MAX, VisionPipeline
from roboemot.vision.types import Detection, Frame, FrameReport
from roboemot.voice.direction import DirectionSample, FaceRef, VoiceDirection, matched_yaw
from roboemot.voice.facing import CpuLoad, FacingCheck
from roboemot.voice.pipeline import face_ref, measured_yaw
from roboemot.voice.wake import AT_FACE, NOT_FACING
from roboemot.world import Entry, Observation, WorldModel
from roboemot.world.frame import wrap_angle

OWNER_AZ, PHOTO_AZ = math.radians(10.0), math.radians(20.0)
OWNER_YAW, PHOTO_YAW = 0.0, math.radians(70.0)
HEAD, BODY = math.radians(12.0), math.radians(5.0)
OWNER = FaceRef("e-owner", "person", 0.1, 20.0)
PHOTO = FaceRef("e-photo", "person", 0.2, 6.0)


def idle_cpu():
    """(busy, total) seconds that grow at 10 % per call: never cpu_high."""
    state = {"busy": 0.0, "total": 0.0}

    def read():
        state["busy"] += 0.1
        state["total"] += 1.0
        return state["busy"], state["total"]

    return read


def facing_direction(samples, *, doa=True, **facing_cfg):
    """A VoiceDirection with the facing check on, over ``samples`` (one per call)."""
    feed = iter(samples)
    facing = FacingCheck(
        VoiceFacingConfig(enabled=True, **facing_cfg), CpuLoad(reader=idle_cpu(), clock=lambda: 0)
    )
    d = VoiceDirection(
        VoiceDirectionConfig(), lambda now: next(feed), doa_available=doa, facing=facing
    )
    return d


def utterance(d, n):
    d.begin()
    for k in range(n):
        d.sample(float(k))
    return d.check()


def scene(bearing_deg):
    """The owner at 10 deg looking at the robot, a photo 10 deg away turned 70 deg."""
    return DirectionSample(
        math.radians(bearing_deg),
        (OWNER_AZ, PHOTO_AZ),
        False,
        faces_yaw_rad=(OWNER_YAW, PHOTO_YAW),
        faces_ref=(OWNER, PHOTO),
        head_yaw_rad=HEAD,
        body_yaw_rad=BODY,
    )


def test_the_record_names_the_owner_when_the_voice_is_nearest_the_owner():
    d = facing_direction([scene(11.0)] * 4)
    assert utterance(d, 4) == AT_FACE
    (rec,) = d.snapshot()["facing"]["recent"]
    assert rec["face_source"] == "matched" and rec["path"] == "heard"
    assert rec["outcome"] == "passed" and rec["verdict"] == AT_FACE
    assert [f["entry_id"] for f in rec["faces"]] == ["e-owner"]
    face = rec["faces"][0]
    assert face["kind"] == "person" and face["samples"] == 4
    assert face["azimuth_deg"] == pytest.approx(10.0) and face["yaw_deg"] == pytest.approx(0.0)
    assert face["age_s"] == pytest.approx(0.1) and face["eye_px"] == pytest.approx(20.0)
    assert rec["yaw_deg"] == pytest.approx(0.0) and rec["yaw_samples"] == 4
    assert "eye_px" not in rec  # per face only (faces[].eye_px)
    assert rec["bearing_deg"] == pytest.approx(11.0) and rec["offset_deg"] == pytest.approx(1.0)
    assert rec["head_yaw_deg"] == pytest.approx(12.0)
    assert rec["body_yaw_deg"] == pytest.approx(5.0)
    assert rec["bearing_head_deg"] == pytest.approx(-1.0)


def test_a_photo_nearer_the_sound_bearing_is_the_face_matched():
    # Documents the suspected bug (SPEC 8 stage 2, owner 2026-11-05): matched_yaw takes the face
    # nearest the sound bearing within the tolerance, whatever the other faces look like, so a
    # photo 2 deg from the bearing wins over the owner 8 deg away and the owner, who looks at the
    # robot, is judged not facing. This asserts the current behaviour, not the wanted one.
    d = facing_direction([scene(18.0)] * 4)
    assert utterance(d, 4) == NOT_FACING
    (rec,) = d.snapshot()["facing"]["recent"]
    assert [f["entry_id"] for f in rec["faces"]] == ["e-photo"]
    assert rec["yaw_deg"] == pytest.approx(70.0) and rec["outcome"] == "not_facing"
    assert rec["offset_deg"] == pytest.approx(2.0) and rec["faces"][0]["eye_px"] == pytest.approx(
        6.0
    )


def test_a_voice_behind_the_head_reads_near_180_against_the_head_yaw():
    sample = DirectionSample(
        math.radians(-170.0),
        (math.radians(-165.0),),
        False,
        faces_yaw_rad=(0.0,),
        faces_ref=(OWNER,),
        head_yaw_rad=math.radians(15.0),
    )
    d = facing_direction([sample] * 2)
    utterance(d, 2)
    (rec,) = d.snapshot()["facing"]["recent"]
    assert abs(rec["bearing_head_deg"]) == pytest.approx(175.0)
    assert rec["body_yaw_deg"] is None


def test_a_degraded_check_records_the_engaged_face():
    sample = DirectionSample(
        None,
        (OWNER_AZ,),
        True,
        faces_yaw_rad=(OWNER_YAW,),
        engaged_yaw_rad=math.radians(-8.0),
        faces_ref=(OWNER,),
        engaged_ref=OWNER,
        engaged_rad=OWNER_AZ,
    )
    d = facing_direction([sample] * 3, doa=False)
    assert utterance(d, 3) == AT_FACE
    (rec,) = d.snapshot()["facing"]["recent"]
    assert rec["face_source"] == "engaged" and rec["path"] == "no_source"
    assert rec["faces"][0]["entry_id"] == "e-owner"
    assert rec["yaw_deg"] == pytest.approx(-8.0) and rec["bearing_deg"] is None


def test_records_are_bounded_and_follow_a_reload():
    d = facing_direction([scene(11.0)] * 50, report_checks=3)
    for _ in range(5):
        utterance(d, 2)
    assert len(d.snapshot()["facing"]["recent"]) == 3
    d.apply_config(VoiceDirectionConfig(), VoiceFacingConfig(enabled=True, report_checks=2))
    assert len(d.snapshot()["facing"]["recent"]) == 2
    with pytest.raises(ValueError):
        VoiceFacingConfig(report_checks=65)


def test_recording_changes_no_verdict(monkeypatch):
    from roboemot.voice import direction

    def boom(*a, **k):
        raise RuntimeError

    monkeypatch.setattr(direction, "facing_record", boom)
    d = facing_direction([scene(18.0)] * 4)
    assert utterance(d, 4) == NOT_FACING
    assert d.snapshot()["facing"]["recent"] == []


def test_face_ref_and_measured_yaw_read_the_entry_and_the_gaze_dofs():
    e = Entry(
        entry_id="e7", kind="person", category="person", identity="", azimuth_rad=0.0,
        elevation_rad=0.0, distance_m=None, uncertainty_rad=0.1, confidence=0.5,
        first_seen=0.0, last_seen=9.0, facing_eye_px=12.5,
    )  # fmt: skip
    assert face_ref(e, 10.0) == FaceRef("e7", "person", 1.0, 12.5)
    pose = {"h": 0.2, "b": -0.1}
    motion = SimpleNamespace(
        motion_sample=lambda: MotionSample(0.0, {}, (None, None, 1.0, pose), False, False)
    )
    assert measured_yaw(motion, GazeMap("h", "p", "b")) == (0.2, -0.1)
    assert measured_yaw(motion, GazeMap("h", "p")) == (0.2, None)
    assert measured_yaw(None, GazeMap("h", "p")) == (None, None)
    broken = SimpleNamespace(motion_sample=lambda: 1 / 0)
    assert measured_yaw(broken, GazeMap("h", "p", "b")) == (None, None)
    assert measured_yaw(SimpleNamespace(motion_sample=lambda: None), GazeMap("h", "p")) == (
        None,
        None,
    )
    unmeasured = SimpleNamespace(motion_sample=lambda: MotionSample(0.0, {}, None, False, False))
    assert measured_yaw(unmeasured, GazeMap("h", "p", "b")) == (None, None)


def old_matched_yaw(sample, tolerance_rad):
    """``matched_yaw`` as it was before ``matched_index`` was split out (origin/main d93f4df)."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    offsets = [abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad]
    nearest = min(range(len(offsets)), key=offsets.__getitem__)
    if offsets[nearest] > tolerance_rad or nearest >= len(sample.faces_yaw_rad):
        return None
    return sample.faces_yaw_rad[nearest]


@pytest.mark.parametrize("seed", range(20))
def test_matched_yaw_is_the_old_formula_on_random_samples(seed):
    rng = random.Random(seed)
    for _ in range(200):
        n = rng.randint(0, 4)
        faces = tuple(rng.uniform(-math.pi, math.pi) for _ in range(n))
        yaws = tuple(
            None if rng.random() < 0.2 else rng.uniform(-1.5, 1.5)
            for _ in range(rng.randint(0, n))  # also fewer yaws than faces
        )
        bearing = None if rng.random() < 0.1 else rng.uniform(-math.pi, math.pi)
        sample = DirectionSample(bearing, faces, False, faces_yaw_rad=yaws)
        tol = rng.uniform(0.0, 1.0)
        assert matched_yaw(sample, tol) == old_matched_yaw(sample, tol)


# ---- vision: eye span through the letterbox, per-face report, frame facts, preview marks


def frontal(cx: float, cy: float, span: float) -> np.ndarray:
    """5 landmarks of a frontal face (eye span ``span`` px) centred on ``(cx, cy)``."""
    h = span / 2
    return np.array(
        [[cx - h, cy], [cx + h, cy], [cx, cy + 0.5 * span], [cx - 0.6 * h, cy + span],
         [cx + 0.6 * h, cy + span]]
    )  # fmt: skip


class Scripted:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return list(self.dets)


class Embedder:
    model_id = "fixed"

    def embed(self, img, det):
        v = np.zeros(128, dtype=np.float32)
        v[0] = 1.0
        return v


def pipeline(tmp_path, dets):
    cfg = RoboEmotConfig()
    bus = Bus()
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), cfg.identity)
    pipe = VisionPipeline(
        cfg.vision,
        cfg.identity,
        bus,
        Matcher(reg, cfg.identity),
        face_detector=Scripted(dets),
        face_embedder=Embedder(),
    )
    seen: list[Observation] = []
    bus.subscribe("percept.entity", lambda m: seen.append(m.payload))
    return pipe, seen


def run_frames(pipe, n=6, width=1280, height=720, start=0):
    report = None
    for i in range(start, start + n):
        report = pipe.process(Frame(np.zeros((height, width, 3), np.uint8), i * 0.125))
    return report


def face(cx=640.0, cy=300.0, span=80.0, size=160.0):
    box = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
    return Detection(box, 0.9, landmarks=frontal(cx, cy, span))


def test_letterbox_scale_matches_the_detector():
    assert letterbox_scale(1280, 720, 320, 240) == pytest.approx(0.25)
    assert letterbox_scale(640, 480, 320, 240) == pytest.approx(0.5)
    assert eye_span(frontal(0.0, 0.0, 80.0)) == pytest.approx(80.0)
    assert eye_span(None) is None and eye_span(np.full((5, 2), np.nan)) is None


def test_the_eye_span_in_detector_pixels_goes_through_the_letterbox(tmp_path):
    pipe, seen = pipeline(tmp_path, [face(span=80.0)])
    run_frames(pipe)
    assert seen and seen[-1].facing_eye_px == pytest.approx(20.0)  # 80 camera px x 0.25
    stats = pipe.stats()
    (row,) = stats["faces"]
    assert row["eye_px"] == pytest.approx(20.0)
    assert row["box_px"] == [160.0, 160.0] and row["score"] == pytest.approx(0.9)
    assert row["yaw_deg"] == pytest.approx(0.0, abs=0.5) and row["known"] is False
    assert row["camera_azimuth_deg"] == pytest.approx(0.0, abs=0.5)
    assert row["base_azimuth_deg"] is None  # no frame spec in this pipeline
    assert stats["frame"] == {
        "width_px": 1280,
        "height_px": 720,
        "channel_order": "BGR",
        "detector_input_px": [320, 240],
        "letterbox_scale": 0.25,
    }


def test_a_face_too_small_for_a_yaw_still_reports_its_eye_span(tmp_path):
    pipe, seen = pipeline(tmp_path, [face(span=12.0)])  # under facing_min_eye_px (16 camera px)
    run_frames(pipe)
    (row,) = pipe.stats()["faces"]
    assert row["yaw_deg"] is None and row["eye_px"] == pytest.approx(3.0)
    assert seen[-1].facing_yaw_rad is None and seen[-1].facing_eye_px == pytest.approx(3.0)


def test_the_world_keeps_the_latest_eye_span():
    world = WorldModel(RoboEmotConfig().world)
    person = dict(kind="person", source="vision", azimuth_rad=0.1, elevation_rad=0.0)
    world.observe(Observation(stamp=1.0, facing_eye_px=20.0, **person))
    world.observe(Observation(stamp=2.0, **person))  # no landmarks: the last span stays
    (e,) = world.entries
    assert e.facing_eye_px == pytest.approx(20.0)


def test_the_face_report_is_bounded_and_holds_numbers_only(tmp_path):
    dets = [face(cx=80.0 + 120.0 * k, span=40.0, size=100.0) for k in range(10)]
    pipe, _ = pipeline(tmp_path, dets)
    run_frames(pipe)
    stats = pipe.stats()
    assert len(stats["faces"]) == FACE_REPORT_MAX
    text = json.dumps(stats)  # plain JSON: no array, image or embedding
    for row in stats["faces"]:
        assert all(not isinstance(v, (list, dict)) or k == "box_px" for k, v in row.items())
    assert "landmark" not in text and "embedding" not in text


def test_the_preview_has_landmarks_only_when_the_flag_is_on(tmp_path):
    pipe, _ = pipeline(tmp_path, [face(span=80.0)])
    report = run_frames(pipe)
    assert report.faces_detected
    off = pipe.preview_overlay(report)
    assert off.faces and off.marks == ()
    pipe.landmarks_wanted = lambda: True
    on = pipe.preview_overlay(report)
    (mark,) = on.marks
    assert len(mark.points) == 5 and mark.points[0] == pytest.approx((600.0, 300.0))
    assert mark.yaw_deg == pytest.approx(0.0, abs=0.5)
    assert pipe.preview_overlay(FrameReport(0.0)).marks == ()  # no face run on that frame


def test_a_frame_without_landmarks_clears_the_reported_span_and_the_world_keeps_its_own(
    tmp_path,
):
    dets = [face(span=80.0)]
    pipe, seen = pipeline(tmp_path, dets)
    run_frames(pipe, n=4)
    dets[0] = Detection(dets[0].box, 0.9, landmarks=None)  # matched, but no landmarks
    run_frames(pipe, n=2, start=4)
    (row,) = pipe.stats()["faces"]
    assert row["eye_px"] is None and row["yaw_deg"] is None
    assert seen[-1].facing_eye_px is None
    world = WorldModel(RoboEmotConfig().world)
    for obs in seen:
        world.observe(obs)
    (e,) = world.entries
    assert e.facing_eye_px == pytest.approx(20.0)  # the last span the landmarks gave


def test_a_track_without_a_detection_has_no_eye_span(tmp_path):
    pipe, _ = pipeline(tmp_path, [face(span=80.0)])
    run_frames(pipe, n=4)
    track = pipe._visible[0]
    state = pipe._faces[track.track_id]
    assert state.eye_px == pytest.approx(20.0)
    coasting = SimpleNamespace(detection=None)
    pipe._track_yaw(coasting, state, 10.0, 0.25)
    assert state.eye_px is None and state.yaw is None


def test_the_landmark_wiring_reads_the_effective_config_and_needs_the_preview():
    from roboemot.config.stack import ConfigStack
    from roboemot.runtime.assembly import preview_landmarks

    assert RoboEmotConfig().webui.live.video_landmarks is False
    plain = preview_landmarks(ConfigStack())
    assert plain() is False
    no_preview = ConfigStack()
    no_preview.set("webui.live.video_landmarks", True)
    assert preview_landmarks(no_preview)() is False  # no effect without the preview
    stack = ConfigStack(app={"webui": {"live": {"enabled": True, "video": True}}})
    wanted = preview_landmarks(stack)
    assert wanted() is False
    stack.set("webui.live.video_landmarks", True)
    assert wanted() is True
    stack.set("webui.live.video_landmarks", False)
    assert wanted() is False
