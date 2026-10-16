"""SPEC 5, 10: attention selection, the attention target on the bus, and the gaze conversion."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.attention import (
    SOURCE,
    AttentionSelector,
    GazeController,
    GazeMap,
    SpeakerHint,
    look_at,
)
from roboemot.bus import Bus
from roboemot.config.schema import AttentionConfig
from roboemot.manifest import load_manifest
from roboemot.motion import MotionCore, Priority
from roboemot.world import Entry, FrameSpec, to_base_frame

REACHY = Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
CFG = AttentionConfig()
DEG = math.radians(1)
GAZE = GazeMap("head.yaw", "head.pitch", "body_yaw")
FRAME = FrameSpec(yaw_dofs=("head.yaw",), pitch_dofs=("head.pitch",))
COMFORT = math.radians(CFG.head_yaw_comfort_deg)


@pytest.fixture(scope="module")
def reachy():
    return load_manifest(REACHY, allow_unmeasured_dynamics=True)


def entry(eid, kind="person", az=0.0, el=0.0, seen=0.0, identity="unknown", **kw) -> Entry:
    kw.setdefault("category", "face" if kind == "person" else "thing")
    kw.setdefault("distance_m", None)
    kw.setdefault("confidence", 0.8)
    return Entry(
        entry_id=eid,
        kind=kind,
        identity=identity,
        azimuth_rad=az * DEG,
        elevation_rad=el * DEG,
        uncertainty_rad=0.1,
        first_seen=0.0,
        last_seen=seen,
        **kw,
    )


# ---- selection


def test_default_order_speaker_enrolled_nearest_motion():
    sel = AttentionSelector(CFG)
    near = entry("p1", distance_m=1.0)
    far = entry("p2", distance_m=3.0)
    alice = entry("p3", identity="alice", distance_m=2.0)
    assert sel.update(0.0, [far, near]).entry_id == "p1"
    assert sel.update(0.1, [far, near]).priority == "nearest_face"
    assert sel.update(0.2, [far, near, alice]).entry_id == "p1"  # held by hysteresis
    sel2 = AttentionSelector(CFG)
    assert sel2.update(0.0, [far, near, alice]).entry_id == "p3"
    assert sel2.update(0.0, [far, near, alice]).priority == "enrolled_person"
    sel2.hint_speaker(SpeakerHint(stamp=CFG.switch_hysteresis_s + 1.0, entry_id="p2"))
    t = sel2.update(
        CFG.switch_hysteresis_s + 1.0,
        [dataclasses.replace(e, last_seen=2.5) for e in (far, near, alice)],
    )
    assert (t.entry_id, t.priority) == ("p2", "active_speaker")


def test_each_category_alone_and_custom_order():
    cup = entry("o1", kind="object")
    sel = AttentionSelector(CFG)
    assert sel.update(0.0, [cup]) is None  # not moving: nothing to look at
    moved = dataclasses.replace(cup, azimuth_rad=cup.azimuth_rad + 20 * DEG, last_seen=0.6)
    t = sel.update(0.6, [moved])
    assert (t.entry_id, t.priority) == ("o1", "salient_motion")

    order = ["nearest_face", "enrolled_person", "active_speaker", "salient_motion"]
    sel = AttentionSelector(AttentionConfig(priority_order=order))
    near = entry("p1", distance_m=1.0)
    alice = entry("p2", identity="alice", distance_m=3.0)
    assert sel.update(0.0, [alice, near]).entry_id == "p1"


def test_nearest_face_without_distance_ranks_last_then_confidence():
    sel = AttentionSelector(CFG)
    a = entry("p1", confidence=0.9)
    b = entry("p2", distance_m=4.0, confidence=0.3)
    c = entry("p3", confidence=0.5)
    assert sel.update(0.0, [a, c, b]).entry_id == "p2"
    sel = AttentionSelector(CFG)
    assert sel.update(0.0, [c, a]).entry_id == "p1"


def test_speaker_bearing_matches_an_entry_or_stands_alone():
    sel = AttentionSelector(CFG)
    p = entry("p1", az=30.0)
    sel.hint_speaker(SpeakerHint(stamp=0.0, azimuth_rad=35 * DEG))
    t = sel.update(0.0, [p])
    assert (t.entry_id, t.priority) == ("p1", "active_speaker")

    sel = AttentionSelector(CFG)
    sel.hint_speaker(SpeakerHint(stamp=0.0, azimuth_rad=-90 * DEG, elevation_rad=5 * DEG))
    t = sel.update(0.0, [p])
    assert t.entry_id is None and t.priority == "active_speaker"
    assert t.azimuth_rad == pytest.approx(-90 * DEG)
    assert (
        sel.update(CFG.speaker_hold_s + 1.0, [dataclasses.replace(p, last_seen=1.6)]).entry_id
        == "p1"
    )


def test_speaker_hint_needs_something_to_point_at():
    with pytest.raises(ValueError):
        SpeakerHint(stamp=0.0)


def test_hysteresis_holds_then_switches():
    sel = AttentionSelector(CFG)
    h = CFG.switch_hysteresis_s
    far = entry("p1", distance_m=3.0)
    assert sel.update(0.0, [far]).reason == "new"
    alice = entry("p2", identity="alice")
    t = sel.update(
        h / 2,
        [dataclasses.replace(far, last_seen=h / 2), dataclasses.replace(alice, last_seen=h / 2)],
    )
    assert (t.entry_id, t.reason) == ("p1", "held")
    t = sel.update(
        h, [dataclasses.replace(far, last_seen=h), dataclasses.replace(alice, last_seen=h)]
    )
    assert (t.entry_id, t.reason, t.priority) == ("p2", "switched", "enrolled_person")


def test_target_gone_switches_at_once_despite_hysteresis():
    sel = AttentionSelector(CFG)
    a, b = entry("p1", distance_m=1.0), entry("p2", distance_m=2.0)
    assert sel.update(0.0, [a, b]).entry_id == "p1"
    t = sel.update(0.1, [dataclasses.replace(b, last_seen=0.1)])
    assert (t.entry_id, t.reason) == ("p2", "switched")


def test_stale_entry_is_no_candidate():
    sel = AttentionSelector(CFG)
    p = entry("p1")
    assert sel.update(0.0, [p]) is not None
    assert sel.update(CFG.stale_s + 0.1, [p]) is None


def test_replay_is_deterministic():
    entries = [entry("p1", distance_m=2.0), entry("p2", identity="bob")]
    run = lambda: [AttentionSelector(CFG).update(t, entries) for t in (0.0, 0.0)]  # noqa: E731
    assert run() == run()


def test_priority_order_validated():
    with pytest.raises(ValidationError):
        AttentionConfig(priority_order=["nearest_face", "loudest"])
    with pytest.raises(ValidationError):
        AttentionConfig(priority_order=["nearest_face", "nearest_face"])


# ---- bus


def test_publishes_on_change_and_at_rate_without_embeddings():
    bus = Bus()
    got: list = []
    bus.subscribe("attention.target", lambda m: got.append((m.stamp, m.payload)))
    sel = AttentionSelector(CFG, bus)
    p = dataclasses.replace(entry("p1", az=10.0), embedding=np.ones(4))
    period = 1.0 / CFG.publish_hz
    sel.update(0.0, [p])
    sel.update(period / 4, [dataclasses.replace(p, last_seen=period / 4)])
    assert len(got) == 1  # unchanged and not yet due
    t = period * 1.1
    sel.update(t, [dataclasses.replace(p, last_seen=t)])
    assert len(got) == 2
    moved = dataclasses.replace(
        p, azimuth_rad=p.azimuth_rad + 2 * CFG.publish_move_deg * DEG, last_seen=t
    )
    sel.update(t + 0.001, [moved])
    assert len(got) == 3  # bearing moved: published at once
    payload = got[0][1]
    assert payload.entry_id == "p1" and payload.priority == "nearest_face"
    assert payload.azimuth_rad == pytest.approx(10 * DEG)
    fields = {f.name for f in dataclasses.fields(payload)}
    assert not any("embedding" in f for f in fields)
    assert "embedding" not in repr(payload)


def test_publishes_none_once_when_lost():
    bus = Bus()
    got: list = []
    bus.subscribe("attention.target", lambda m: got.append(m.payload))
    sel = AttentionSelector(CFG, bus)
    sel.update(0.0, [entry("p1")])
    sel.update(0.1, [])
    sel.update(0.2, [])
    assert got[1:] == [None]


# ---- look_at


def pose(reachy, az_deg, el_deg, body_now=0.0):
    return look_at(
        az_deg * DEG,
        el_deg * DEG,
        GAZE,
        reachy.dofs,
        reachy.constraints,
        comfort_rad=COMFORT,
        body_now=body_now,
    )


def test_pitch_sign_target_above_is_nose_up(reachy):
    assert pose(reachy, 0, 10)["head.pitch"] == pytest.approx(-10 * DEG)
    assert pose(reachy, 0, -10)["head.pitch"] == pytest.approx(10 * DEG)


def test_target_to_the_left_is_positive_yaw(reachy):
    assert pose(reachy, 30, 0)["head.yaw"] == pytest.approx(30 * DEG)


@pytest.mark.parametrize("az,el", [(30, 10), (-20, -15), (0, 0), (120, 5)])
def test_round_trip_through_the_frame_transform(reachy, az, el):
    p = pose(reachy, az, el)
    back = to_base_frame(0.0, 0.0, p, FRAME)
    assert back == pytest.approx((az * DEG, el * DEG), abs=1e-9)


def test_body_holds_inside_comfort_and_turns_by_the_excess(reachy):
    inside = pose(reachy, CFG.head_yaw_comfort_deg - 5, 0)
    assert inside["body_yaw"] == pytest.approx(0.0)
    beyond = pose(reachy, 100, 0)
    assert beyond["head.yaw"] == pytest.approx(100 * DEG)
    assert beyond["body_yaw"] == pytest.approx((100 - CFG.head_yaw_comfort_deg) * DEG)
    left_then_right = pose(reachy, -100, 0, body_now=beyond["body_yaw"])
    assert left_then_right["body_yaw"] == pytest.approx((-100 + CFG.head_yaw_comfort_deg) * DEG)


def test_limits_and_yaw_delta_respected_for_any_bearing(reachy):
    delta = math.radians(reachy.constraints[0].max_deg)
    body = reachy.dofs["body_yaw"]
    for az in range(-180, 181, 15):
        for el in (-90, -45, 0, 45, 90):
            p = pose(reachy, az, el)
            for name, v in p.items():
                assert reachy.dofs[name].min - 1e-9 <= v <= reachy.dofs[name].max + 1e-9
            assert abs(p["head.yaw"] - p["body_yaw"]) <= delta + 1e-9
            assert body.min - 1e-9 <= p["body_yaw"] <= body.max + 1e-9


def test_behind_the_robot_is_clamped_to_reach(reachy):
    p = pose(reachy, 180, 0)
    assert p["head.yaw"] == pytest.approx(math.pi)
    assert p["body_yaw"] == pytest.approx(math.pi - COMFORT)


def test_margin_shrinks_the_head_body_offset(reachy):
    delta = math.radians(reachy.constraints[0].max_deg)
    p = look_at(
        170 * DEG,
        0.0,
        GAZE,
        reachy.dofs,
        reachy.constraints,
        comfort_rad=math.pi,
        margin_rad=10 * DEG,
    )
    assert p["head.yaw"] - p["body_yaw"] == pytest.approx(delta - 10 * DEG)


def test_gaze_without_body_dof(reachy):
    p = look_at(
        90 * DEG,
        0.0,
        GazeMap("head.yaw", "head.pitch"),
        reachy.dofs,
        reachy.constraints,
        comfort_rad=COMFORT,
    )
    assert set(p) == {"head.yaw", "head.pitch"}


def test_gaze_map_must_name_manifest_dofs(reachy):
    with pytest.raises(KeyError):
        GazeController(CFG, GazeMap("neck.yaw", "head.pitch"), reachy.dofs, reachy.constraints)


# ---- controller: smoothing and release


def target(az=30.0, el=10.0, now=0.0):
    from roboemot.attention import AttentionTarget

    return AttentionTarget("p1", "person", az * DEG, el * DEG, "nearest_face", "new", now)


def controller(reachy):
    return GazeController(CFG, GAZE, reachy.dofs, reachy.constraints)


def test_gaze_starts_from_the_measured_pose_and_never_exceeds_the_rate(reachy):
    g = controller(reachy)
    t, dt = 0.0, 0.05
    first = g.step(t, target(), {"head.yaw": -0.3, "head.pitch": 0.1, "body_yaw": 0.0})
    assert first.targets["head.yaw"] == pytest.approx(-0.3)
    prev = first.targets
    cap = math.radians(CFG.max_rate_dps) * dt
    for _ in range(100):
        t += dt
        cmd = g.step(t, target(now=t))
        for k, v in cmd.targets.items():
            assert abs(v - prev[k]) <= cap + 1e-12
        prev = cmd.targets
    assert prev["head.yaw"] == pytest.approx(30 * DEG, abs=0.01)
    assert prev["head.pitch"] == pytest.approx(-10 * DEG, abs=0.01)
    assert cmd.source == SOURCE and cmd.priority == Priority.IDLE
    assert cmd.expires == pytest.approx(t + CFG.command_ttl_s)


def test_release_after_release_s_without_target(reachy):
    g = controller(reachy)
    g.step(0.0, target())
    assert g.active
    lost = 1.0
    assert g.step(lost, None) is not None  # holds the pose during the grace period
    assert g.active
    assert g.step(lost + CFG.release_s - 0.01, None) is not None
    assert g.step(lost + CFG.release_s, None) is None
    assert not g.active
    assert g.step(lost + CFG.release_s + 1.0, None) is None


def test_target_back_within_grace_cancels_release(reachy):
    g = controller(reachy)
    g.step(0.0, target())
    g.step(1.0, None)
    g.step(2.0, target(now=2.0))
    assert g.step(1.0 + CFG.release_s + 0.5, None) is not None


def test_idle_wins_again_once_the_attention_command_is_cleared(reachy):
    from roboemot.motion import Command

    core = MotionCore(
        reachy.dofs,
        reachy.constraints,
        crossfade_s=0.0,
        watchdog_s=10.0,
        minor_factor=1.0,
        watchdog_return_factor=1.0,
    )
    g = controller(reachy)
    t = 0.0
    for _ in range(200):
        t += 0.02
        core.feed(g.step(t, target(now=t)))
        core.feed(Command("idle", Priority.IDLE, {"head.pitch": 0.0}, t - 0.01))
        core.tick(t, 0.02)
    assert core.arbiter.last["head.yaw"] == pytest.approx(30 * DEG, abs=0.02)
    assert core.arbiter.last["head.pitch"] == pytest.approx(-10 * DEG, abs=0.02)
    core.arbiter.clear(SOURCE)
    core.tick(t + 0.02, 0.02)
    assert core.arbiter.last["head.yaw"] == pytest.approx(reachy.dofs["head.yaw"].home)
    assert core.arbiter.last["head.pitch"] == pytest.approx(0.0)


# ---- review round: speaker gate, bypass, margin, rear side, recentre, replay


def at(e: Entry, t: float, **kw) -> Entry:
    return dataclasses.replace(e, last_seen=t, **kw)


def test_speaker_bearing_gate_ignores_non_person_entries():
    sel = AttentionSelector(CFG)
    cup = entry("o1", kind="object", az=30.0)
    sel.hint_speaker(SpeakerHint(stamp=0.0, azimuth_rad=32 * DEG))
    t = sel.update(0.0, [cup])
    assert t.entry_id is None and t.priority == "active_speaker"


def test_speaker_hint_expires():
    sel = AttentionSelector(CFG)
    p = entry("p1", az=30.0, distance_m=2.0)
    sel.hint_speaker(SpeakerHint(stamp=0.0, entry_id="p2", azimuth_rad=-60 * DEG))
    assert sel.update(0.0, [p]).entry_id is None
    late = CFG.speaker_hold_s + 0.1
    t = sel.update(late, [at(p, late)])
    assert (t.entry_id, t.priority) == ("p1", "nearest_face")


def test_stale_entry_id_hint_falls_back_to_the_bearing():
    sel = AttentionSelector(CFG)
    p = entry("p1", az=30.0)
    sel.hint_speaker(SpeakerHint(stamp=0.0, entry_id="gone", azimuth_rad=31 * DEG))
    assert sel.update(0.0, [p]).entry_id == "p1"
    sel = AttentionSelector(CFG)
    sel.hint_speaker(SpeakerHint(stamp=0.0, entry_id="gone"))
    assert sel.update(0.0, [p]).priority == "nearest_face"


def test_new_speaker_bypasses_hysteresis():
    sel = AttentionSelector(CFG)
    a, b = entry("p1", distance_m=1.0), entry("p2", distance_m=3.0)
    assert sel.update(0.0, [a, b]).entry_id == "p1"
    t = CFG.switch_hysteresis_s / 4
    sel.hint_speaker(SpeakerHint(stamp=t, entry_id="p2"))
    got = sel.update(t, [at(a, t), at(b, t)])
    assert (got.entry_id, got.priority, got.reason) == ("p2", "active_speaker", "switched")


def test_same_category_switch_needs_a_score_margin():
    sel = AttentionSelector(CFG)
    h = CFG.switch_hysteresis_s
    near = 2.0 * (1 - CFG.switch_margin / 2)  # barely nearer than 2.0
    a = entry("p1", distance_m=2.0)
    b = entry("p2", distance_m=near)
    assert sel.update(0.0, [a, b]).entry_id == "p2"
    t = 0.0
    for i in range(1, 6):  # noisy, similar distances flip which one is nearer
        t = h * i
        d1, d2 = (near, 2.0) if i % 2 else (2.0, near)
        got = sel.update(t, [at(a, t, distance_m=d1), at(b, t, distance_m=d2)])
        assert got.entry_id == "p2", i
    far = 2.0 / (1 + 2 * CFG.switch_margin)
    t += h
    got = sel.update(t, [at(a, t, distance_m=far), at(b, t, distance_m=2.0)])
    assert (got.entry_id, got.reason) == ("p1", "switched")


def test_category_names_are_one_constant():
    from roboemot.config.schema import ATTENTION_CATEGORIES

    assert set(CFG.priority_order) == set(ATTENTION_CATEGORIES)
    with pytest.raises(ValidationError):
        AttentionConfig(priority_order=[])
    assert AttentionConfig(priority_order=["nearest_face"]).priority_order == ["nearest_face"]
    sel = AttentionSelector(AttentionConfig(priority_order=["salient_motion"]))
    assert sel.update(0.0, [entry("p1")]) is None


def test_replay_runs_without_a_wall_clock(monkeypatch):
    import time

    def boom(*a, **k):
        raise AssertionError("wall clock used")

    series = []
    for i in range(60):
        t = i * 0.1
        series.append(
            (
                t,
                [
                    entry("p1", az=10.0 + i, seen=t, distance_m=2.0),
                    entry("p2", identity="bob" if i > 20 else "unknown", az=-40.0, seen=t),
                    entry("o1", kind="object", az=3.0 * i, seen=t),
                ],
            )
        )
    monkeypatch.setattr(time, "time", boom)
    monkeypatch.setattr(time, "monotonic", boom)

    def run():
        sel = AttentionSelector(CFG, Bus())
        return [sel.update(t, es) for t, es in series]

    first, second = run(), run()
    assert first == second
    assert {x.priority for x in first if x} >= {"nearest_face", "enrolled_person"}


def test_rear_target_does_not_flip_sides_on_jitter(reachy):
    g = controller(reachy)
    t, dt = 0.0, 0.05
    g.step(t, target(179.0, 0.0), {"head.yaw": 170 * DEG, "body_yaw": 120 * DEG})
    prev = None
    for i in range(80):
        t += dt
        sign = 1 if i % 2 else -1
        cmd = g.step(t, target(sign * 179.0, 0.0, now=t))
        if prev is not None:
            for k in ("head.yaw", "body_yaw"):
                assert abs(cmd.targets[k] - prev[k]) <= math.radians(CFG.max_rate_dps) * dt + 1e-9
        prev = cmd.targets
        assert cmd.targets["head.yaw"] > 0 and cmd.targets["body_yaw"] > 0


def test_rear_side_is_released_once_the_target_leaves_the_band(reachy):
    g = controller(reachy)
    g.step(0.0, target(179.0, 0.0), {"head.yaw": 170 * DEG, "body_yaw": 120 * DEG})
    t = 0.0
    for _ in range(200):
        t += 0.05
        g.step(t, target(-100.0, 0.0, now=t))
    assert g._pos["head.yaw"] == pytest.approx(-100 * DEG, abs=0.02)
    g.step(t + 0.05, target(-179.0, 0.0))
    assert g._aim["head.yaw"] < 0  # the side follows the current head again


def test_body_recentres_slowly_while_head_is_inside_comfort(reachy):
    g = controller(reachy)
    t, dt = 0.0, 0.05
    start = {"head.yaw": 60 * DEG, "head.pitch": 0.0, "body_yaw": 50 * DEG}
    g.step(t, target(60.0, 0.0), start)
    body = [50 * DEG]
    for _ in range(40):
        t += dt
        body.append(g.step(t, target(60.0, 0.0, now=t)).targets["body_yaw"])
    cap = math.radians(CFG.body_recenter_dps) * dt
    steps = [a - b for a, b in zip(body, body[1:], strict=False)]
    assert all(0 <= s <= cap + 1e-9 for s in steps) and sum(steps) > 0
    assert 60 * DEG - body[-1] <= COMFORT + 1e-9  # head stays inside the comfort band


def test_look_at_recentre_target_stays_in_band(reachy):
    p = pose(reachy, 60, 0, body_now=50 * DEG)
    assert p["body_yaw"] == pytest.approx(60 * DEG - COMFORT)  # nearest to home inside band


def test_gaze_dt_is_clamped_to_the_command_ttl(reachy):
    g = controller(reachy)
    g.step(0.0, target(90.0, 0.0), {"head.yaw": 0.0, "head.pitch": 0.0, "body_yaw": 0.0})
    cmd = g.step(100.0, target(90.0, 0.0, now=100.0))
    assert cmd.targets["head.yaw"] <= math.radians(CFG.max_rate_dps) * CFG.command_ttl_s + 1e-9


def test_minor_present_shrinks_attention_motion(reachy):
    def settle(minor: bool) -> float:
        core = MotionCore(
            reachy.dofs,
            reachy.constraints,
            crossfade_s=0.0,
            watchdog_s=10.0,
            minor_factor=0.5,
            watchdog_return_factor=1.0,
        )
        core.minor_present = minor
        g = controller(reachy)
        t = 0.0
        for _ in range(300):
            t += 0.02
            core.feed(g.step(t, target(60.0, 0.0, now=t)))
            core.tick(t, 0.02)
        return core.limiter.pos["head.yaw"]

    assert settle(False) == pytest.approx(60 * DEG, abs=0.02)
    assert settle(True) == pytest.approx(60 * DEG * 0.5, abs=0.02)  # amplitude around home


def test_released_attention_returns_the_head_towards_home(reachy):
    core = MotionCore(
        reachy.dofs,
        reachy.constraints,
        crossfade_s=0.0,
        watchdog_s=10.0,
        minor_factor=1.0,
        watchdog_return_factor=1.0,
    )
    g = controller(reachy)
    t = 0.0
    for _ in range(200):
        t += 0.02
        core.feed(g.step(t, target(30.0, 0.0, now=t)))
        core.tick(t, 0.02)
    peak = core.limiter.pos["head.yaw"]
    for _ in range(round((CFG.release_s + 8.0) / 0.02)):
        t += 0.02
        cmd = g.step(t, None)
        if cmd is None:
            core.arbiter.clear(SOURCE)
        else:
            core.feed(cmd)
        core.tick(t, 0.02)
    assert not g.active
    assert abs(core.limiter.pos["head.yaw"]) < 0.1 * abs(peak)


def test_selector_engaged_only_for_a_person_or_the_speaker():
    sel = AttentionSelector(CFG)
    assert not sel.engaged
    sel.update(0.0, [entry("p", "person", seen=0.0, distance_m=1.0)])
    assert sel.engaged
    sel = AttentionSelector(CFG)
    moving = [entry("o", "object", az=0.0, seen=0.0)]
    sel.update(0.0, moving)
    moving = [entry("o", "object", az=40.0, seen=1.0)]
    target = sel.update(1.0, moving)
    assert target is not None and target.kind == "object"
    assert not sel.engaged  # a held object target does not engage
    sel = AttentionSelector(CFG)
    sel.hint_speaker(SpeakerHint(0.0, azimuth_rad=1.0))
    sel.update(0.0, [])
    assert sel.engaged  # a bare speaker bearing counts
    sel.update(30.0, [])
    assert not sel.engaged  # the hint expired
