"""SPEC 5, 7A: the day nap's motion watch. While a nap waits for it, captured frames are
compared at vision.scene_motion.interval_s with the last still view (the object gate's scene-change
test); motion counts only while the head is still, after the gate's settle time from the nap's
start, and when confirm_samples samples in a row differ. It works with the object stage off."""

from __future__ import annotations

import math

import numpy as np
import pytest
from roboemot.bus import TOPICS
from roboemot.config.schema import SceneMotionConfig
from roboemot.vision import CIRCADIAN, Frame
from roboemot.vision.scene_motion import SceneMotionWatch, ScenePercept
from roboemot.world import FrameSpec
from test_vision import CFG, H, W, rig

GATE = CFG.vision.object_gate
MOTION = CFG.vision.scene_motion
SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))
DT = 1 / 30  # captured frames


def scene(block_x: int | None = None, shift: int = 0) -> np.ndarray:
    """A textured still scene; a bright block at ``block_x`` (a hand); ``shift`` adds to every
    pixel (slow daylight drift)."""
    rng = np.random.default_rng(3)
    img = rng.integers(60, 120, (H, W, 3), dtype=np.uint8) + np.uint8(shift)
    if block_x is not None:
        img[60:160, block_x : block_x + 80] = 250
    return img


def pose(yaw_deg: float = 0.0) -> tuple[float, float]:
    return math.radians(yaw_deg), 0.0


def feed(watch, frames, active=True):
    """Feed ``(stamp, image, pose)`` frames; the stamps that reported motion."""
    return [s for s, img, p in frames if watch.offer(s, img, p, active) is not None]


def still(start: float, seconds: float, img=None, yaw_deg: float = 0.0):
    img = scene() if img is None else img
    n = round(seconds / DT)
    return [(start + i * DT, img, pose(yaw_deg)) for i in range(n)]


def settled(watch, start: float = 0.0) -> float:
    """Arm the watch and wait past the settle time with a still scene; the next stamp."""
    span = GATE.settle_s + 2 * MOTION.interval_s
    assert feed(watch, still(start, span)) == []
    return start + span


def test_defaults_and_topic():
    assert MOTION.interval_s == 0.5 and MOTION.confirm_samples == 2
    assert "percept.scene_motion" in TOPICS


def test_a_hand_in_front_of_a_still_head_counts_after_two_samples():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    hits = feed(watch, still(t, 3 * MOTION.interval_s, scene(block_x=100)))
    assert len(hits) == 1
    assert hits[0] >= t + MOTION.interval_s - DT  # the second changed sample confirms
    snap = watch.snapshot()
    assert snap["confirmed"] == 1 and snap["changed"] >= MOTION.confirm_samples


def test_the_payload_is_a_scalar_share():
    watch = SceneMotionWatch(MOTION.model_copy(update={"confirm_samples": 1}), GATE)
    t = settled(watch)
    share = watch.offer(t, scene(block_x=100), pose(), True)
    assert isinstance(share, float) and GATE.changed_share < share <= 1.0
    assert ScenePercept(share).share == share


def test_one_odd_frame_is_not_motion():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    frames = [(t, scene(block_x=100), pose())] + still(t + MOTION.interval_s, 3.0)
    assert feed(watch, frames) == []


def test_nothing_counts_within_the_settle_time_of_the_nap_start():
    watch = SceneMotionWatch(MOTION, GATE)
    frames = [(i * DT, scene(block_x=100 if i % 2 else None), pose()) for i in range(30)]
    assert GATE.settle_s >= 1.0
    assert feed(watch, frames) == []  # 1 s of a changing scene right after it armed
    assert watch.snapshot()["head_moving"] > 0


def test_a_moving_head_never_counts_and_settles_afterwards():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    step = 2 * GATE.head_still_deg  # every sample a move past head_still_deg
    moving = [(t + i * DT, scene(block_x=(i * 7) % 200), pose(step * (i // 5))) for i in range(150)]
    assert feed(watch, moving) == []
    end = moving[-1][0] + DT
    yaw = step * (149 // 5)
    assert feed(watch, still(end, GATE.settle_s + 2 * MOTION.interval_s, yaw_deg=yaw)) == []
    t2 = end + GATE.settle_s + 2 * MOTION.interval_s
    assert feed(watch, still(t2, 3 * MOTION.interval_s, scene(block_x=100), yaw_deg=yaw))


def test_without_a_head_pose_nothing_counts():
    watch = SceneMotionWatch(MOTION, GATE)
    frames = [(i * DT, scene(block_x=100 if (i // 15) % 2 else None), None) for i in range(300)]
    assert feed(watch, frames) == []
    assert watch.snapshot()["no_pose"] > 0


def test_slow_daylight_drift_is_not_motion():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    n = 120
    frames = [(t + i * MOTION.interval_s, scene(shift=i // 4), pose()) for i in range(n)]
    assert feed(watch, frames) == []


def test_it_samples_at_interval_s_not_every_frame():
    watch = SceneMotionWatch(MOTION, GATE)
    seconds = 10.0
    feed(watch, still(0.0, seconds))
    samples = watch.snapshot()["samples"]
    assert samples == pytest.approx(seconds / MOTION.interval_s, abs=1)


def test_inactive_it_does_nothing_and_rearms_on_the_next_nap():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    before = watch.snapshot()["samples"]
    assert feed(watch, still(t, 3.0, scene(block_x=100)), active=False) == []
    assert watch.snapshot()["samples"] == before
    t2 = t + 3.0
    # armed again: the settle time counts from the new start, and the old view is gone
    changing = [
        (t2 + i * DT, scene(block_x=100 if (i // 5) % 2 else None), pose())
        for i in range(round((GATE.settle_s - DT) / DT))
    ]
    assert feed(watch, changing) == []


def test_after_a_report_it_waits_for_a_new_still_view():
    watch = SceneMotionWatch(MOTION, GATE)
    t = settled(watch)
    hits = feed(watch, still(t, GATE.settle_s + 4 * MOTION.interval_s, scene(block_x=100)))
    assert len(hits) == 1  # the hand stays: one report, not one per sample


def config_with(**update) -> SceneMotionConfig:
    return MOTION.model_copy(update=update)


@pytest.mark.parametrize("bad", [{"interval_s": 0.0}, {"confirm_samples": 0}])
def test_config_ranges(bad):
    with pytest.raises(ValueError):
        SceneMotionConfig(**{**MOTION.model_dump(), **bad})


# ------------------------------------------------------------------ the pipeline


def joint(yaw_deg: float = 0.0) -> dict[str, float]:
    return {"head.yaw": math.radians(yaw_deg), "head.pitch": 0.0}


def watched(tmp_path, wanted):
    """A pipeline with no object detector (the object stage off, as on the robot)."""
    _, bus, pipe, _, _ = rig(tmp_path, lambda i: [], frame_spec=SPEC)
    got = []
    bus.subscribe("percept.scene_motion", got.append)
    pipe.motion_wanted = wanted
    pipe.governor.set_scale(CIRCADIAN, 0.05)  # resting vision
    return pipe, got


def frames(start: float, seconds: float, block_x: int | None = None):
    return [Frame(scene(block_x), start + i * DT, joint()) for i in range(round(seconds / DT))]


def test_the_pipeline_publishes_motion_with_the_object_stage_off(tmp_path):
    pipe, got = watched(tmp_path, lambda: True)
    assert pipe.object_detector is None
    span = GATE.settle_s + 2 * MOTION.interval_s
    for f in frames(0.0, span) + frames(span, 3 * MOTION.interval_s, block_x=100):
        pipe.process(f)
    assert len(got) == 1
    assert isinstance(got[0].payload, ScenePercept)
    assert pipe.stats()["scene_motion"]["confirmed"] == 1


def test_the_pipeline_does_not_watch_unless_a_nap_wants_it(tmp_path):
    pipe, got = watched(tmp_path, lambda: False)
    span = GATE.settle_s + 2 * MOTION.interval_s
    for f in frames(0.0, span) + frames(span, 3.0, block_x=100):
        pipe.process(f)
    assert got == [] and pipe.stats()["scene_motion"]["samples"] == 0


def test_a_hot_reload_reaches_the_watch(tmp_path):
    pipe, got = watched(tmp_path, lambda: True)
    vision = CFG.vision.model_copy(update={"scene_motion": config_with(confirm_samples=1)})
    pipe.apply_config(vision, CFG.identity)
    span = GATE.settle_s + 2 * MOTION.interval_s
    for f in frames(0.0, span) + frames(span, DT, block_x=100):
        pipe.process(f)
    assert len(got) == 1
