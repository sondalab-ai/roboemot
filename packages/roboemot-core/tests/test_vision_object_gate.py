"""Scene-change gate of the object stage (SPEC 5, 13): object detection runs only when the
image changed while the head was still, after the head settles, at a floor interval, and on the
fixed schedule without a head pose or while an animal, tentative or recently moved track is
live. Mock stages only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.config.schema import ObjectGateConfig, RoboEmotConfig
from roboemot.vision import CIRCADIAN, Detection, Frame
from roboemot.vision.object_gate import RUN_REASONS, SKIP_REASONS, ObjectGate, thumbnail
from roboemot.world import FrameSpec
from test_vision import CFG, H, ObjectMock, W, rig

SPEC = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))
DT = 0.125  # 8 Hz frames; objects_hz is 1, so every 8th frame is an object tick
GATE = CFG.vision.object_gate
PERIOD = 1 / CFG.vision.rates.objects_hz


def scene(block_x: int | None = None) -> np.ndarray:
    """A textured still scene, with a bright block at ``block_x`` (a thing that appeared)."""
    rng = np.random.default_rng(3)
    img = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    if block_x is not None:
        img[60:160, block_x : block_x + 80] = 250
    return img


def joint(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> dict[str, float]:
    return {"head.yaw": math.radians(yaw_deg), "head.pitch": math.radians(pitch_deg)}


def gated(tmp_path, dets=(), vision=None, frame_spec=SPEC):
    """A pipeline with the head spec and a scripted object detector."""
    objs = ObjectMock(list(dets))
    _, _, pipe, _, got = rig(
        tmp_path, lambda i: [], vision=vision, object_detector=objs, frame_spec=frame_spec
    )
    return pipe, objs, got


def drive(pipe, frames):
    """Feed ``(stamp, image, joint_state)`` frames; the stamps of the frames objects ran on."""
    return [s for s, img, j in frames if pipe.process(Frame(img, s, j)).objects_detected]


def gate_cfg(**update) -> object:
    """The default vision config with ``vision.object_gate`` fields replaced."""
    return CFG.vision.model_copy(update={"object_gate": GATE.model_copy(update=update)})


def counts(pipe) -> tuple[dict[str, int], dict[str, int]]:
    gate = pipe.stats()["object_gate"]
    return gate["skipped"], gate["triggered"]


# ------------------------------------------------------------------------- config


def test_defaults_and_units():
    g = ObjectGateConfig()
    assert g.enabled is True
    assert (g.thumb_width_px, g.thumb_height_px) == (80, 60)
    assert g.pixel_diff == 25.0 and g.changed_share == 0.02
    assert g.head_still_deg == 3.0 and g.settle_s == 1.0 and g.max_interval_s == 10.0
    assert g.moved_share == 0.05 and g.moved_hold_s == 3.0
    fields = ObjectGateConfig.model_fields
    for name, unit in {
        "thumb_width_px": "px",
        "thumb_height_px": "px",
        "head_still_deg": "deg",
        "settle_s": "s",
        "max_interval_s": "s",
        "moved_hold_s": "s",
    }.items():
        assert fields[name].json_schema_extra["unit"] == unit
    assert RoboEmotConfig().vision.object_gate == g


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("thumb_width_px", 7),
        ("thumb_width_px", 321),
        ("thumb_height_px", 5),
        ("thumb_height_px", 241),
        ("pixel_diff", 0.5),
        ("pixel_diff", 256),
        ("changed_share", 0),
        ("changed_share", 1.01),
        ("head_still_deg", 0.0),
        ("head_still_deg", 31),
        ("settle_s", -0.1),
        ("settle_s", 31),
        ("max_interval_s", 0.5),
        ("max_interval_s", 601),
        ("moved_share", 0),
        ("moved_share", 1.01),
        ("moved_hold_s", -0.1),
        ("moved_hold_s", 61),
        ("enabled", "maybe"),
    ],
)
def test_every_field_is_range_checked(field, bad):
    with pytest.raises(ValidationError):
        RoboEmotConfig.model_validate({"vision": {"object_gate": {field: bad}}})


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        RoboEmotConfig.model_validate({"vision": {"object_gate": {"thumb_px": 80}}})


# ------------------------------------------------------------------------ motion test


def test_thumbnail_is_small_grayscale_and_area_pooled():
    img = np.zeros((H, W, 3), np.uint8)
    img[:, : W // 2] = 200
    small = thumbnail(img, 80, 60)
    assert small.shape == (60, 80) and small.dtype == np.float32
    assert small[:, :40] == pytest.approx(200.0, abs=0.5)
    assert small[:, 40:] == pytest.approx(0.0)
    assert thumbnail(np.zeros((4, 4, 3), np.uint8), 80, 60).shape == (60, 80)


def test_unsigned_difference_does_not_wrap():
    gate = ObjectGate(GATE)
    dark, bright = np.full((H, W, 3), 10, np.uint8), np.full((H, W, 3), 12, np.uint8)
    assert gate.decide(0.0, dark, (0.0, 0.0), holds=())  # floor: never ran
    assert not gate.decide(1.0, bright, (0.0, 0.0), holds=())  # 2 levels, not 254


# ---------------------------------------------------------------------- the pipeline


def test_still_head_still_scene_is_skipped_until_the_floor(tmp_path):
    pipe, _, _ = gated(tmp_path)
    n = int(25 / DT)
    ran = drive(pipe, [(i * DT, scene(), joint()) for i in range(n)])
    floor = GATE.max_interval_s
    assert ran == [0.0, floor, 2 * floor]
    skipped, triggered = counts(pipe)
    assert triggered["floor"] == 3 and sum(triggered.values()) == len(ran)
    assert skipped["still_scene"] > 0 and skipped["head_moving"] == 0


def test_still_head_changed_scene_runs(tmp_path):
    pipe, _, _ = gated(tmp_path)
    frames = [(i * DT, scene(None if i < 24 else 100), joint()) for i in range(40)]
    ran = drive(pipe, frames)
    assert ran == [0.0, 3.0]  # first tick that sees the block (frame 24, 3.0 s)
    assert counts(pipe)[1]["motion"] == 1


def test_head_moving_skips_then_one_run_after_settling(tmp_path):
    pipe, _, _ = gated(tmp_path)
    frames = []
    for i in range(80):
        stamp = i * DT
        if stamp < 2.0:
            yaw = 0.0
        elif stamp < 4.0:  # a 20 deg turn over 2 s; the image changes all along
            yaw = 10.0 * (stamp - 2.0)
        else:
            yaw = 20.0
        frames.append((stamp, scene(int(yaw * 4)), joint(yaw)))
    ran = drive(pipe, frames)
    last_move = max(s for s, _, j in frames if 2.0 <= s < 4.0)
    settled = [s for s in ran if s > 2.0]
    assert ran[0] == 0.0
    assert len(settled) == 1 and settled[0] >= last_move + GATE.settle_s - 1e-9
    assert settled[0] < last_move + GATE.settle_s + PERIOD + 1e-9
    skipped, triggered = counts(pipe)
    assert skipped["head_moving"] >= 1 and triggered["settled"] == 1
    assert triggered["motion"] == 0


def test_idle_breathing_is_not_a_head_move(tmp_path):
    pipe, _, _ = gated(tmp_path)
    amp = math.degrees(CFG.motion.idle.breathing_amplitude)
    frames = [
        (i * DT, scene(), joint(pitch_deg=amp * math.sin(i * DT * 2 * math.pi / 4)))
        for i in range(80)
    ]
    drive(pipe, frames)
    skipped, triggered = counts(pipe)
    assert skipped["head_moving"] == 0 and triggered["settled"] == 0


def test_no_joint_state_keeps_the_fixed_schedule(tmp_path):
    pipe, _, _ = gated(tmp_path)
    ran = drive(pipe, [(i * DT, scene(), None) for i in range(40)])
    assert ran == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert counts(pipe)[1]["no_pose"] == 5


def test_no_frame_spec_keeps_the_fixed_schedule(tmp_path):
    pipe, _, _ = gated(tmp_path, frame_spec=None)
    ran = drive(pipe, [(i * DT, scene(), joint()) for i in range(40)])
    assert ran == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_live_animal_track_keeps_the_fixed_schedule(tmp_path):
    cat = Detection((20, 100, 80, 160), 0.8, "cat")
    pipe, _, got = gated(tmp_path, dets=[cat])
    ran = drive(pipe, [(i * DT, scene(), joint()) for i in range(40)])
    assert ran == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert counts(pipe)[1]["live_track"] == 4
    assert pipe.stats()["object_gate"]["live_track_by"]["animal"] == 4
    assert got["percept.entity"]  # confirmed on the second object frame


def test_a_still_animal_that_appears_is_confirmed(tmp_path):
    pipe, objs, got = gated(tmp_path)
    cat = Detection((100, 60, 180, 160), 0.8, "cat")
    frames = [(i * DT, scene(None if i < 20 else 100), joint()) for i in range(60)]
    for stamp, img, j in frames:
        if stamp >= 2.5:
            objs.dets = [cat]
        pipe.process(Frame(img, stamp, j))
    entities = [m.payload for m in got["percept.entity"]]
    assert entities and min(e.stamp for e in entities) <= 3.0 + 2 * PERIOD


def test_never_more_runs_than_the_governor_allows(tmp_path):
    pipe, _, _ = gated(tmp_path)
    pipe.governor.set_scale(CIRCADIAN, 0.5)
    rng = np.random.default_rng(1)
    frames = [(i * DT, rng.integers(0, 255, (H, W, 3), dtype=np.uint8), joint()) for i in range(80)]
    ran = drive(pipe, frames)
    assert np.all(np.diff(ran) >= 1 / (CFG.vision.rates.objects_hz * 0.5) - 1e-9)
    assert len(ran) == 5  # every allowed tick sees a new image


def test_skipped_and_triggered_add_up_to_the_allowed_ticks(tmp_path):
    pipe, _, _ = gated(tmp_path)
    frames = [
        (i * DT, scene(None if i < 30 else 100), joint(0 if i < 50 else 15)) for i in range(160)
    ]
    ran = drive(pipe, frames)
    skipped, triggered = counts(pipe)
    assert sum(triggered.values()) == len(ran)
    ticks = int(frames[-1][0] / PERIOD) + 1
    assert sum(triggered.values()) + sum(skipped.values()) == ticks
    assert set(skipped) == set(SKIP_REASONS) and set(triggered) == set(RUN_REASONS)
    json.dumps(pipe.stats())


def test_replay_gives_the_same_decisions(tmp_path):
    frames = []
    for i in range(120):
        stamp = i * DT
        yaw = 0.0 if stamp < 5 else min(15.0, 10 * (stamp - 5))
        frames.append((stamp, scene(None if i < 20 else 100 + int(yaw)), joint(yaw)))
    first, _, _ = gated(tmp_path / "a")
    a = drive(first, frames)
    a_counts = counts(first)
    first.reset()
    again = drive(first, frames)
    second, _, _ = gated(tmp_path / "b")
    b = drive(second, frames)
    assert a == again == b and counts(second) == a_counts


def test_disabled_gate_keeps_the_fixed_schedule(tmp_path):
    off = CFG.vision.model_copy(update={"object_gate": ObjectGateConfig(enabled=False)})
    pipe, _, _ = gated(tmp_path, vision=off)
    ran = drive(pipe, [(i * DT, scene(), joint()) for i in range(40)])
    assert ran == [0.0, 1.0, 2.0, 3.0, 4.0]
    skipped, triggered = counts(pipe)
    assert sum(skipped.values()) == sum(triggered.values()) == 0


def test_hot_reload_of_the_thumbnail_size_never_raises(tmp_path):
    pipe, _, _ = gated(tmp_path)
    drive(pipe, [(i * DT, scene(), joint()) for i in range(10)])
    bigger = CFG.vision.model_copy(
        update={"object_gate": ObjectGateConfig(thumb_width_px=40, thumb_height_px=30)}
    )
    pipe.apply_config(bigger, CFG.identity)
    ran = drive(pipe, [(i * DT, scene(), joint()) for i in range(10, 30)])
    assert ran == [2.0]  # cannot compare the old copy: treated as changed, then still


def test_reset_clears_the_gate(tmp_path):
    pipe, _, _ = gated(tmp_path)
    drive(pipe, [(i * DT, scene(), joint()) for i in range(20)])
    pipe.reset()
    assert counts(pipe) == (dict.fromkeys(SKIP_REASONS, 0), dict.fromkeys(RUN_REASONS, 0))


# --------------------------------------------------------------- fix round (review)


def track_ids(pipe) -> set[int]:
    return {t.track_id for t in pipe.object_tracker._tracks}


def test_a_still_confirmed_object_does_not_hold_and_keeps_one_track_id(tmp_path):
    chair = Detection((100, 60, 220, 200), 0.8, "chair")
    pipe, _, _ = gated(tmp_path, dets=[chair])
    ran, ids = [], set()
    for i in range(int(30 / DT)):
        stamp = i * DT
        if pipe.process(Frame(scene(), stamp, joint())).objects_detected:
            ran.append(stamp)
        ids |= track_ids(pipe)
    floor = GATE.max_interval_s
    assert ran == [0.0, 1.0, 1.0 + floor, 1.0 + 2 * floor]  # tentative holds once, then floor
    assert len(ids) == 1
    by = pipe.stats()["object_gate"]["live_track_by"]
    assert by == {"animal": 0, "tentative": 1, "moving": 0}


def test_object_lifetime_covers_the_floor_only_while_the_gate_is_on(tmp_path):
    pipe, _, _ = gated(tmp_path)
    periods = CFG.vision.object_lost_periods * PERIOD
    assert pipe._object_lost_s() == GATE.max_interval_s + periods
    pipe.apply_config(gate_cfg(enabled=False), CFG.identity)
    assert pipe._object_lost_s() == max(CFG.vision.object_lost_s, periods)


def test_a_moving_box_holds_the_fixed_schedule_until_the_hold_time(tmp_path):
    pipe, objs, _ = gated(tmp_path)
    step = GATE.moved_share * W + 4  # just over the moved share per object frame
    ran = []
    for i in range(int(12 / DT)):
        stamp = i * DT
        x = 20 + step * math.floor(stamp)
        objs.dets = [Detection((x, 60, x + 80, 140), 0.8, "cup")] if stamp <= 4.0 else []
        if pipe.process(Frame(scene(), stamp, joint())).objects_detected:
            ran.append(stamp)
    last_hit = 4.0
    held = [float(t) for t in range(int(last_hit + GATE.moved_hold_s) + 1)]
    assert ran == held  # every tick while moving, then until the last hit is moved_hold_s old
    assert pipe.stats()["object_gate"]["live_track_by"]["moving"] >= 1
    assert counts(pipe)[0]["still_scene"] > 0


def test_negative_and_large_yaw_and_a_move_across_the_anchor():
    gate, img, r = ObjectGate(GATE), scene(), math.radians

    def tick(stamp, yaw_deg):
        gate.observe(stamp, (r(yaw_deg), 0.0))
        return gate.decide(stamp, img, (r(yaw_deg), 0.0), ())

    assert tick(0.0, -170)  # floor
    assert not tick(1.0, -172)  # 2 deg: still
    gate.observe(1.5, (r(-174), 0.0))  # 4 deg: a move
    assert not tick(2.0, -174)  # head moving
    assert tick(3.0, -174)  # settled; anchored at -174
    gate.observe(3.5, (r(-176.5), 0.0))  # 2.5 deg one side
    gate.observe(3.75, (r(-172), 0.0))  # 2 deg the other side
    assert not tick(4.0, -172)  # still: both within 3 deg of the anchor
    gate.observe(4.5, (r(-170.5), 0.0))  # 3.5 deg across the anchor: a move
    assert not tick(5.0, -170.5)
    snap = gate.snapshot()
    assert snap["skipped"]["head_moving"] == 2 and snap["triggered"]["settled"] == 1


def test_joint_state_missing_on_alternate_frames(tmp_path):
    pipe, _, _ = gated(tmp_path)  # pose on even frames, which carry every object tick
    frames = [(i * DT, scene(), joint() if i % 2 == 0 else None) for i in range(int(25 / DT))]
    assert drive(pipe, frames) == [0.0, GATE.max_interval_s, 2 * GATE.max_interval_s]
    assert counts(pipe)[1]["no_pose"] == 0
    odd, _, _ = gated(tmp_path / "odd")  # pose only between ticks: the fixed schedule
    frames = [(i * DT, scene(), None if i % 2 == 0 else joint()) for i in range(40)]
    assert drive(odd, frames) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_the_floor_fires_while_the_head_moves_continuously(tmp_path):
    pipe, _, _ = gated(tmp_path)
    frames = [
        (i * DT, scene(int(i * DT * 8) % 200), joint(10.0 * i * DT)) for i in range(int(25 / DT))
    ]
    floor = GATE.max_interval_s
    assert drive(pipe, frames) == [0.0, floor, 2 * floor]
    skipped, triggered = counts(pipe)
    assert triggered["floor"] == 3 and skipped["head_moving"] == 22 and triggered["motion"] == 0


def test_a_gate_skipped_tick_uses_the_governor_slot(tmp_path):
    pipe, _, _ = gated(tmp_path)
    frames = [(i * DT, scene(None if i < 9 else 100), joint()) for i in range(24)]
    assert drive(pipe, frames) == [0.0, 2.0]  # the change at 1.125 s waits for the 2.0 s tick


def test_thresholds_are_strict():
    gate, pose = ObjectGate(GATE), (0.0, 0.0)
    h, w = GATE.thumb_height_px, GATE.thumb_width_px
    base = np.full((h, w), 100, np.uint8)
    assert gate.decide(0.0, base, pose, ())
    assert not gate.decide(1.0, base + int(GATE.pixel_diff), pose, ())  # exactly pixel_diff
    assert gate.decide(2.0, base + int(GATE.pixel_diff) + 1, pose, ())
    img = base + int(GATE.pixel_diff) + 1
    n = round(h * w * GATE.changed_share)
    img.flat[:n] = 250
    assert not gate.decide(3.0, img, pose, ())  # exactly changed_share
    img.flat[: n + 1] = 250
    assert gate.decide(4.0, img, pose, ())


@pytest.mark.parametrize("shape", [(479, 641, 3), (479, 641), (479, 641, 4)])
def test_odd_frame_sizes_gray_and_bgra(shape):
    small = thumbnail(np.full(shape, 90, np.uint8), 80, 60)
    assert small.shape == (60, 80) and small.dtype == np.float32
    assert small == pytest.approx(90.0, abs=0.01)


def test_an_unsupported_frame_shape_runs_instead_of_raising():
    assert thumbnail(np.zeros((10, 10, 2), np.uint8), 80, 60) is None
    gate = ObjectGate(GATE)
    odd = np.zeros((10, 10, 2), np.uint8)
    assert gate.decide(0.0, odd, (0.0, 0.0), ())  # floor
    assert gate.decide(1.0, odd, (0.0, 0.0), ())  # cannot compare: counts as changed
    assert gate.snapshot()["triggered"]["motion"] == 1


def test_disabled_then_reenabled_by_hot_reload(tmp_path):
    pipe, _, _ = gated(tmp_path, vision=gate_cfg(enabled=False))
    still = [(i * DT, scene(), joint()) for i in range(160)]
    assert drive(pipe, still[:40]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    pipe.apply_config(gate_cfg(enabled=True), CFG.identity)
    assert drive(pipe, still[40:120]) == [5.0]  # never ran while on: floor, then still
    pipe.apply_config(gate_cfg(enabled=False), CFG.identity)
    assert drive(pipe, still[120:]) == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_stamps_that_go_back_never_raise_and_reset_replays_the_same(tmp_path):
    frames = [(i * DT, scene(None if i < 20 else 100), joint()) for i in range(100)]
    pipe, _, _ = gated(tmp_path)
    first = drive(pipe, frames)
    assert drive(pipe, frames[:10]) == []  # stamps went back without a reset: no run, no raise
    pipe.reset()
    assert drive(pipe, frames) == first
