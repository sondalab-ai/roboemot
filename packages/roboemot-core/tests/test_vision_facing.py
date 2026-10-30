"""Head-yaw proxy from the 5 face landmarks (SPEC 8, facing check)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, WorldConfig
from roboemot.registry import FileStore, Matcher, Registry
from roboemot.vision.facing import NOSE_DEPTH_PER_EYE_SPAN, head_yaw, smoothed
from roboemot.vision.pipeline import VisionPipeline
from roboemot.vision.types import Detection, Frame
from roboemot.voice.pipeline import fresh_yaw
from roboemot.world import Entry, Observation, WorldModel

# The ArcFace alignment template (112 px): a frontal face, the person's right eye on the left.
FRONTAL = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]])
EYE_SPAN_MM, NOSE_DEPTH_MM = 63.0, NOSE_DEPTH_PER_EYE_SPAN * 63.0


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
    assert abs(math.degrees(head_yaw(FRONTAL))) < 2.0
    assert head_yaw(turned(0.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("deg", [15.0, 30.0, 50.0, -15.0, -40.0])
def test_a_turned_face_reads_its_angle_with_the_sign(deg):
    assert math.degrees(head_yaw(turned(deg))) == pytest.approx(deg, abs=0.5)


def test_turning_to_the_image_left_is_positive():
    lm = FRONTAL.copy()
    lm[2, 0] -= 8.0  # nose toward the image left: the face turned toward the robot's left
    assert head_yaw(lm) > math.radians(10.0)


def test_a_tilted_head_reads_the_same_yaw():
    assert math.degrees(head_yaw(turned(30.0, roll_deg=25.0))) == pytest.approx(30.0, abs=0.5)


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
    assert head_yaw(lm) is None


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


def test_the_world_keeps_the_last_yaw_and_a_bearing_without_one_leaves_it():
    world = WorldModel(WorldConfig())
    person = dict(kind="person", source="vision", azimuth_rad=0.1, elevation_rad=0.0)
    world.observe(Observation(stamp=1.0, facing_yaw_rad=0.3, **person))
    world.observe(Observation(stamp=2.0, **person))  # a frame with no usable landmarks
    (e,) = world.entries
    assert e.facing_yaw_rad == 0.3 and e.facing_stamp == 1.0
    world.observe(Observation(stamp=3.0, facing_yaw_rad=-0.1, **person))
    (e,) = world.entries
    assert e.facing_yaw_rad == -0.1 and e.facing_stamp == 3.0


@pytest.mark.parametrize("stamp,expected", [(None, None), (9.5, 0.2), (10.8, 0.2), (8.0, None)])
def test_fresh_yaw_is_measured_close_enough_on_either_side(stamp, expected):
    e = Entry("person-1", "person", "person", "unknown", 0.0, 0.0, None, 0.1, 1.0, 0.0, 10.0)
    e.facing_yaw_rad, e.facing_stamp = 0.2, stamp
    assert fresh_yaw(e, 10.0, 1.0) == expected
