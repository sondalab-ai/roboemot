"""Head-yaw proxy from the 5 face landmarks (SPEC 8, facing check)."""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from pathlib import Path

import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, WorldConfig
from roboemot.registry import FileStore, Matcher, Registry
from roboemot.vision.facing import head_yaw, smoothed
from roboemot.vision.pipeline import VisionPipeline
from roboemot.vision.types import Detection, Frame
from roboemot.voice.pipeline import yaw_at
from roboemot.world import Entry, Observation, WorldModel

VISION = RoboEmotConfig().vision
RATIO = VISION.facing_nose_depth_ratio
# The ArcFace alignment template (112 px): a frontal face, the person's right eye on the left.
FRONTAL = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]])
EYE_SPAN_MM, NOSE_DEPTH_MM = 63.0, RATIO * 63.0
REAL = Path(__file__).parent / "data" / "facing" / "astronaut_yunet_landmarks.json"


def yaw(lm, min_eye_px=1.0, ratio=RATIO):
    return head_yaw(lm, ratio, min_eye_px)


def turned(yaw_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Landmarks of a rigid head turned by ``yaw_deg`` (positive toward the robot's left, i.e.
    the image left) and tilted by ``roll_deg``, orthographic, in pixels (1 px per mm)."""
    yaw = math.radians(yaw_deg)
    # 3D points (x image-right, depth toward the camera): eyes, nose tip, mouth corners
    pts3 = [
        (-EYE_SPAN_MM / 2, 0.0, 0.0),
        (EYE_SPAN_MM / 2, 0.0, 0.0),
        (0.0, 20.0, NOSE_DEPTH_MM),
        (-25.0, 40.0, 0.0),
        (25.0, 40.0, 0.0),
    ]
    xy = [(x * math.cos(yaw) - z * math.sin(yaw), y) for x, y, z in pts3]
    roll = math.radians(roll_deg)
    c, s = math.cos(roll), math.sin(roll)
    return np.array([(100 + c * x - s * y, 100 + s * x + c * y) for x, y in xy])


def test_a_frontal_face_reads_about_zero():
    assert abs(math.degrees(yaw(FRONTAL))) < 2.0
    assert yaw(turned(0.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("deg", [15.0, 30.0, 50.0, -15.0, -40.0])
def test_a_turned_face_reads_its_angle_with_the_sign(deg):
    assert math.degrees(yaw(turned(deg))) == pytest.approx(deg, abs=0.5)


def test_turning_to_the_image_left_is_positive():
    lm = FRONTAL.copy()
    lm[2, 0] -= 8.0  # nose toward the image left: the face turned toward the robot's left
    assert yaw(lm) > math.radians(10.0)


def test_a_tilted_head_reads_the_same_yaw():
    assert math.degrees(yaw(turned(30.0, roll_deg=25.0))) == pytest.approx(30.0, abs=0.5)


@pytest.mark.parametrize(
    "lm",
    [
        None,
        np.zeros((5, 2)),  # every point on top of each other
        np.array([[50, 50], [50.2, 50], [50, 60], [45, 70], [55, 70]], dtype=float),  # eyes 0.2 px
        np.full((5, 2), np.nan),
        np.zeros((4, 2)),
    ],
)
def test_degenerate_landmarks_give_no_estimate(lm):
    assert yaw(lm) is None


def test_a_face_whose_eyes_are_closer_than_the_minimum_gives_no_estimate():
    small = turned(20.0) * 0.2 + 50  # eyes about 12 px apart
    assert yaw(small, min_eye_px=VISION.facing_min_eye_px) is None
    assert yaw(small, min_eye_px=10.0) is not None


def test_the_nose_depth_ratio_scales_the_yaw():
    assert math.degrees(yaw(turned(30.0), ratio=RATIO / 2)) > 45.0


def test_real_yunet_landmarks_of_a_near_frontal_portrait_read_within_the_default_limit():
    """Real detector output (numbers only, see the fixture's ``source``): a near-frontal face,
    mirrored, rolled by up to 20 deg and down to a 10 px eye span. Its true yaw is not
    measured, so this bounds the size, not the exact value: the landmark noise seen here
    (-15 to +9 deg) stays inside voice.facing.max_yaw_deg, so a face looking at the robot
    passes."""
    views = json.loads(REAL.read_text())["views"]
    limit = RoboEmotConfig().voice.facing.max_yaw_deg
    yaws = {v["view"]: yaw(np.array(v["landmarks"])) for v in views}
    assert len(yaws) == 14 and all(y is not None for y in yaws.values())
    assert all(abs(math.degrees(y)) < limit for y in yaws.values()), yaws
    eye_px = [np.hypot(*np.subtract(v["landmarks"][1], v["landmarks"][0])) for v in views]
    big = [y for y, px in zip(yaws.values(), eye_px, strict=True) if px >= 16.0]
    assert statistics.median(abs(math.degrees(y)) for y in big) < 8.0


def test_smoothed_is_the_median_over_the_window_and_drops_older_values():
    h = deque([(0.0, 1.0), (0.4, 0.1), (0.6, 0.2), (0.8, 0.9)])
    assert smoothed(h, 0.9, 0.5) == pytest.approx(0.2)  # (0.4, 0.6, 0.8) kept
    assert len(h) == 3
    assert smoothed(h, 5.0, 0.5) is None and not h


# ---- through the pipeline and the world model


def face_with(landmarks: np.ndarray, x: float = 100.0, size: float = 60.0) -> Detection:
    return Detection((x, 80.0, x + size, 80.0 + size), 0.9, landmarks=landmarks)


class Scripted:
    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, img):
        out = self.script(self.i)
        self.i += 1
        return out


class Embedder:
    model_id = "fixed"

    def detect(self, img):  # pragma: no cover - unused
        return []

    def embed(self, img, det):
        v = np.zeros(128, dtype=np.float32)
        v[0] = 1.0
        return v


def pipeline_entities(tmp_path, script, frames=8, dt=0.125):
    cfg = RoboEmotConfig()
    bus = Bus()
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), cfg.identity)
    pipe = VisionPipeline(
        cfg.vision,
        cfg.identity,
        bus,
        Matcher(reg, cfg.identity),
        face_detector=Scripted(script),
        face_embedder=Embedder(),
    )
    seen: list[Observation] = []
    bus.subscribe("percept.entity", lambda m: seen.append(m.payload))
    for i in range(frames):
        pipe.process(Frame(np.zeros((240, 320, 3), np.uint8), i * dt))
    return seen


def test_the_pipeline_publishes_the_smoothed_yaw_of_each_face(tmp_path):
    def script(i):  # one noisy frame among frames turned by 30 deg
        return [face_with(turned(-60.0 if i == 4 else 30.0))]

    seen = pipeline_entities(tmp_path, script)
    assert seen and all(o.facing_yaw_rad is not None for o in seen)
    assert math.degrees(seen[-1].facing_yaw_rad) == pytest.approx(30.0, abs=0.5)


def test_degenerate_landmarks_publish_no_yaw(tmp_path):
    seen = pipeline_entities(tmp_path, lambda i: [face_with(np.zeros((5, 2)))])
    assert seen and all(o.facing_yaw_rad is None for o in seen)


def test_the_world_keeps_a_bounded_yaw_history_and_a_bearing_without_one_adds_nothing():
    world = WorldModel(WorldConfig(facing_history_s=5.0))
    person = dict(kind="person", source="vision", azimuth_rad=0.1, elevation_rad=0.0)
    world.observe(Observation(stamp=1.0, facing_yaw_rad=0.3, **person))
    world.observe(Observation(stamp=2.0, **person))  # a frame with no usable landmarks
    world.observe(Observation(stamp=3.0, facing_yaw_rad=-0.1, **person))
    world.observe(Observation(stamp=2.5, facing_yaw_rad=0.9, **person))  # out of order: dropped
    (e,) = world.entries
    assert e.facing == ((1.0, 0.3), (3.0, -0.1))
    world.observe(Observation(stamp=7.0, facing_yaw_rad=0.0, **person))
    (e,) = world.entries
    assert e.facing == ((3.0, -0.1), (7.0, 0.0))  # 1.0 is older than 5 s


HISTORY = ((5.0, 0.5), (9.6, 0.2), (10.7, -0.3), (30.0, 0.4))


@pytest.mark.parametrize(
    "now,expected",
    [
        (10.0, 0.2),  # nearest, 0.4 s before
        (10.4, -0.3),  # nearest, 0.3 s after
        (12.0, None),  # 1.3 s after the last before it, 18 s before the next
        (11.0, -0.3),
        (20.0, None),  # a yaw 10 s later is not this sample's
        (4.5, 0.5),
        (40.0, None),
    ],
)
def test_yaw_at_takes_the_nearest_reading_within_the_age_on_both_sides(now, expected):
    e = Entry("person-1", "person", "person", "unknown", 0.0, 0.0, None, 0.1, 1.0, 0.0, 10.0)
    e.facing = HISTORY
    assert yaw_at(e, now, 1.0) == expected


def test_yaw_at_without_a_history_is_none():
    e = Entry("person-1", "person", "person", "unknown", 0.0, 0.0, None, 0.1, 1.0, 0.0, 10.0)
    assert yaw_at(e, 10.0, 1.0) is None
