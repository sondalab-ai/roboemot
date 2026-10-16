"""SPEC 5A world model: matching, fading, events and the frame transform."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import RoboEmotConfig, WorldConfig
from roboemot.world import FrameSpec, Observation, WorldModel, to_base_frame

CFG = RoboEmotConfig().world
FRAME = FrameSpec(
    yaw_dofs=("body_yaw", "head_yaw"), pitch_dofs=("head_pitch",), roll_dofs=("head_roll",)
)
DEG = math.radians(1)
EVENT_TOPICS = (
    "world.entity_appeared",
    "world.entity_moved",
    "world.entity_gone",
    "world.snapshot",
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def obs(clock: Clock, az_deg: float = 0.0, **kw) -> Observation:
    kw.setdefault("kind", "person")
    kw.setdefault("source", "vision")
    el = kw.pop("el_deg", 0.0) * DEG
    return Observation(stamp=clock.t, azimuth_rad=az_deg * DEG, elevation_rad=el, **kw)


def setup() -> tuple[WorldModel, Bus, Clock, dict[str, list]]:
    bus = Bus()
    events: dict[str, list] = {}
    for topic in EVENT_TOPICS:
        events[topic] = []
        bus.subscribe(topic, lambda m, t=topic: events[t].append(m.payload))
    return WorldModel(CFG, FRAME, bus), bus, Clock(), events


def always_in_view(az: float, el: float) -> bool:
    return True


def never_in_view(az: float, el: float) -> bool:
    return False


def conf(w: WorldModel, entry_id: str) -> float:
    e = w.get(entry_id)
    assert e is not None
    return e.confidence


def seen(w: WorldModel, o: Observation, joint_state=None) -> str:
    r = w.observe(o, joint_state)
    assert r.entry is not None
    return r.entry.entry_id


def test_topics_registered():
    for t in ("percept.entity", *EVENT_TOPICS):
        assert t in TOPICS


# ---------------------------------------------------------------- matching


def test_new_entry_and_appeared_event():
    w, _, clock, ev = setup()
    r = w.observe(obs(clock, 10, category="person", confidence=0.8))
    assert r.created and r.entry is not None
    assert r.entry.identity == "unknown"
    assert r.entry.confidence == pytest.approx(CFG.initial_confidence * 0.8)
    assert len(ev["world.entity_appeared"]) == 1


def test_identity_match_regardless_of_position():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 0, identity="p1", category="face"))
    clock.advance(1)
    b = w.observe(obs(clock, 120, identity="p1", category="face"))
    assert not b.created and b.entry is not None and b.entry.entry_id == a
    assert len(w.entries) == 1


def test_conflicting_known_identity_does_not_match_by_gate():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, identity="p1", category="face"))
    assert w.observe(obs(clock, 1, identity="p2", category="face")).created
    assert len(w.entries) == 2


def test_identityless_observation_matches_known_entry():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 0, identity="p1", category="face"))
    clock.advance(1)
    r = w.observe(obs(clock, 2, category="face"))
    assert not r.created and r.entry is not None and r.entry.entry_id == a
    assert r.entry.identity == "p1"
    assert len(w.entries) == 1


def test_identity_observation_promotes_unknown_entry():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 0, category="face"))
    clock.advance(1)
    r = w.observe(obs(clock, 2, identity="p1", category="face"))
    assert not r.created and r.entry is not None and r.entry.entry_id == a
    assert r.entry.identity == "p1"


def test_gate_match_and_miss():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, kind="animal", category="cat"))
    assert not w.observe(obs(clock, CFG.match_gate_deg - 1, kind="animal", category="cat")).created
    assert w.observe(obs(clock, 90, kind="animal", category="cat")).created
    assert w.observe(obs(clock, 0, kind="animal", category="dog")).created


def test_gate_wraps_around_pi():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 179, kind="animal", category="cat"))
    r = w.observe(obs(clock, -179, kind="animal", category="cat"))
    assert not r.created and r.entry is not None and r.entry.entry_id == a


def test_gate_includes_elevation():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, kind="object", category="cup"))
    assert w.observe(
        obs(clock, 0, el_deg=CFG.match_gate_deg * 2, kind="object", category="cup")
    ).created
    assert not w.observe(
        obs(clock, 0, el_deg=CFG.match_gate_deg / 2, kind="object", category="cup")
    ).created


def test_range_gate_only_when_both_have_distance():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, kind="object", category="cup", distance_m=1.0))
    far = 1.0 + CFG.range_gate_m + 0.5
    assert w.observe(obs(clock, 0, kind="object", category="cup", distance_m=far)).created
    assert not w.observe(obs(clock, 0, kind="object", category="cup")).created
    near = 1.0 + CFG.range_gate_m / 2
    assert not w.observe(obs(clock, 0, kind="object", category="cup", distance_m=near)).created


def test_embedding_breaks_near_ties_but_angle_is_primary():
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    gate, tol = CFG.match_gate_deg, CFG.tie_tolerance_deg
    near_tie = gate * 0.6 - tol / 4  # within tol of the tie point between a and b
    clear = 1.0  # much nearer to a than to b

    def fresh():
        w, _, clock, _ = setup()
        a = seen(w, obs(clock, 0, kind="animal", category="cat", embedding=e1))
        b = seen(w, obs(clock, gate * 1.2, kind="animal", category="cat", embedding=e2))
        assert a != b
        return w, clock, a, b

    def probe(az, **kw):
        w, clock, a, b = fresh()
        r = w.observe(obs(clock, az, kind="animal", category="cat", **kw))
        assert r.entry is not None
        return r.entry.entry_id, a, b

    got, a, b = probe(near_tie, embedding=e2)
    assert got == b
    got, a, b = probe(clear, embedding=e2)
    assert got == a
    got, a, b = probe(near_tie)
    assert got == a


def test_embedding_dimension_mismatch_is_neutral():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 0, kind="animal", category="cat", embedding=np.ones(3)))
    r = w.observe(obs(clock, 1, kind="animal", category="cat", embedding=np.ones(5)))
    assert r.entry is not None and r.entry.entry_id == a


def test_update_copies_embedding():
    w, _, clock, _ = setup()
    emb = np.ones(3)
    seen(w, obs(clock, 0, identity="p1", category="face", embedding=np.ones(3)))
    w.observe(obs(clock, 0, identity="p1", category="face", embedding=emb))
    emb[:] = 99.0
    assert w._entries[next(iter(w._entries))].embedding.max() < 99.0


def test_bearing_only_cannot_create_but_confirms():
    w, _, clock, ev = setup()
    r = w.observe(obs(clock, 30, kind="unknown", source="doa"))
    assert r.entry is None and not r.created
    assert r.unexplained_bearing == pytest.approx((30 * DEG, 0.0))
    assert w.entries == [] and not ev["world.entity_appeared"]
    e = seen(w, obs(clock, 30, category="person"))
    before = conf(w, e)
    clock.advance(1)
    c = w.observe(obs(clock, 31, kind="unknown", source="doa"))
    assert c.entry is not None and c.entry.entry_id == e and c.unexplained_bearing is None
    assert conf(w, e) > before
    assert w.get(e).sources == {"vision", "doa"}


def test_low_confidence_observation_does_not_create():
    w, _, clock, ev = setup()
    r = w.observe(obs(clock, 0, category="person", confidence=CFG.forget_confidence))
    assert r.rejected and r.entry is None and not r.created
    assert w.entries == [] and ev["world.entity_appeared"] == []


def test_observation_validation():
    with pytest.raises(ValueError):
        Observation(kind="robot", stamp=0, source="s", azimuth_rad=0, elevation_rad=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Observation(
            kind="person", stamp=0, source="s", azimuth_rad=0, elevation_rad=0, confidence=1.5
        )


# ---------------------------------------------------------------- confidence and decay


def test_confidence_rises_with_matches():
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    prev = conf(w, e)
    for _ in range(3):
        clock.advance(0.01)
        w.observe(obs(clock, 0, category="person"))
        assert conf(w, e) > prev
        prev = conf(w, e)
    assert prev <= 1.0


def test_decay_in_view_vs_out_of_view():
    for in_view, rate in ((always_in_view, 1.0), (never_in_view, CFG.out_of_view_decay_factor)):
        w, _, clock, _ = setup()
        e = seen(w, obs(clock, 0, category="person"))
        c0 = conf(w, e)
        w.tick(clock.advance(CFG.half_life_s.person), in_view)
        assert conf(w, e) == pytest.approx(c0 * 0.5**rate)


def test_half_life_per_kind():
    w, _, clock, _ = setup()
    person = seen(w, obs(clock, 0, kind="person", category="p"))
    thing = seen(w, obs(clock, 90, kind="object", category="cup"))
    w.tick(clock.advance(CFG.half_life_s.person), always_in_view)
    assert conf(w, person) / CFG.initial_confidence == pytest.approx(0.5)
    assert conf(w, thing) > conf(w, person)


def test_two_half_ticks_equal_one_tick():
    h = CFG.half_life_s.person
    w1, _, c1, _ = setup()
    w2, _, c2, _ = setup()
    e1 = seen(w1, obs(c1, 0, category="person"))
    e2 = seen(w2, obs(c2, 0, category="person"))
    w1.tick(c1.advance(h / 2), always_in_view)
    w1.tick(c1.advance(h / 2), always_in_view)
    w2.tick(c2.advance(h), always_in_view)
    assert conf(w1, e1) == pytest.approx(conf(w2, e2))


def test_observe_after_gap_decays_first():
    h = CFG.half_life_s.person
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    c0 = conf(w, e)
    clock.advance(h)
    w.observe(obs(clock, 0, category="person"))
    decayed = c0 * 0.5
    expected = decayed + CFG.confidence_gain * (1 - decayed)
    assert conf(w, e) == pytest.approx(expected)


def test_bearing_only_confirmation_decays_first():
    h = CFG.half_life_s.person
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    c0 = conf(w, e)
    clock.advance(h)
    w.observe(obs(clock, 0, kind="unknown", source="doa"))
    decayed = c0 * 0.5
    assert conf(w, e) == pytest.approx(decayed + CFG.confidence_gain * (1 - decayed))


def test_tick_in_the_past_is_noop_and_never_moves_time_back():
    h = CFG.half_life_s.person
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    w.tick(clock.advance(h), always_in_view)
    c1 = conf(w, e)
    assert w.tick(h / 2, always_in_view) == []
    assert conf(w, e) == c1
    w.tick(clock.advance(h), always_in_view)  # decays only the one new half-life
    assert conf(w, e) == pytest.approx(c1 * 0.5)


def test_out_of_view_to_in_view_transition():
    h, f = CFG.half_life_s.person, CFG.out_of_view_decay_factor
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    c0 = conf(w, e)
    w.tick(clock.advance(h), never_in_view)
    w.tick(clock.advance(h), always_in_view)
    assert conf(w, e) == pytest.approx(c0 * 0.5**f * 0.5)


def test_observe_uses_last_view_state_for_gap_then_marks_in_view():
    h, f = CFG.half_life_s.person, CFG.out_of_view_decay_factor
    w, _, clock, _ = setup()
    e = seen(w, obs(clock, 0, category="person"))
    c0 = conf(w, e)
    w.tick(clock.advance(h / 2), never_in_view)
    clock.advance(h / 2)
    w.observe(obs(clock, 0, category="person"))
    decayed = c0 * 0.5 ** (f * 0.5) * 0.5 ** (f * 0.5)
    assert conf(w, e) == pytest.approx(decayed + CFG.confidence_gain * (1 - decayed))
    c1 = conf(w, e)
    w.tick(clock.advance(h), always_in_view)
    assert conf(w, e) == pytest.approx(c1 * 0.5)


def test_forgetting_emits_gone():
    w, _, clock, ev = setup()
    e = seen(w, obs(clock, 0, category="person"))
    gone = w.tick(clock.advance(CFG.half_life_s.person * 10), always_in_view)
    assert [g.entry_id for g in gone] == [e]
    assert w.entries == []
    assert [g.entry_id for g in ev["world.entity_gone"]] == [e]
    assert w.tick(clock.advance(1), always_in_view) == []


# ---------------------------------------------------------------- moved events


def test_moved_threshold():
    w, _, clock, ev = setup()
    thr, blend = CFG.moved_threshold_deg, CFG.position_blend
    w.observe(obs(clock, 0, identity="p1", category="person"))
    clock.advance(1)
    w.observe(obs(clock, 0.5 * thr / blend, identity="p1", category="person"))
    assert ev["world.entity_moved"] == []
    clock.advance(1)
    w.observe(obs(clock, 3 * thr / blend, identity="p1", category="person"))
    assert len(ev["world.entity_moved"]) == 1


def test_moved_event_on_gate_path_and_anchor_resets():
    w, _, clock, ev = setup()
    thr = CFG.moved_threshold_deg
    step = CFG.match_gate_deg * 0.5
    e = seen(w, obs(clock, 0, kind="animal", category="cat"))
    for i in range(1, 4):
        clock.advance(1)
        r = w.observe(obs(clock, step * i, kind="animal", category="cat"))
        assert r.entry is not None and r.entry.entry_id == e
    assert len(ev["world.entity_moved"]) >= 1
    n = len(ev["world.entity_moved"])
    clock.advance(1)
    pos = w.get(e).azimuth_rad / DEG
    w.observe(obs(clock, pos, kind="animal", category="cat"))
    assert len(ev["world.entity_moved"]) == n  # re-anchored: no repeat without new motion
    assert thr > 0


def test_slow_drift_eventually_fires():
    w, _, clock, ev = setup()
    thr = CFG.moved_threshold_deg
    w.observe(obs(clock, 0, identity="p1", category="person"))
    az = 0.0
    for _ in range(40):
        clock.advance(1)
        az += thr / 8
        w.observe(obs(clock, az, identity="p1", category="person"))
    assert len(ev["world.entity_moved"]) >= 1


# ---------------------------------------------------------------- frame


def test_frame_yaw_and_wrap():
    az, el = to_base_frame(0.0, 0.0, {"head_yaw": math.radians(40)}, FRAME)
    assert az == pytest.approx(math.radians(40)) and el == pytest.approx(0.0)
    az, _ = to_base_frame(math.radians(10), 0.0, {"body_yaw": 0.5, "head_yaw": 0.25}, FRAME)
    assert az == pytest.approx(math.radians(10) + 0.75)
    az, _ = to_base_frame(math.radians(170), 0.0, {"head_yaw": math.radians(30)}, FRAME)
    assert az == pytest.approx(math.radians(-160))
    assert to_base_frame(0.2, 0.1, None, FRAME) == pytest.approx((0.2, 0.1))
    assert to_base_frame(0.2, 0.1, {"head_yaw": 1.0}, FrameSpec()) == pytest.approx((0.2, 0.1))


def test_positive_pitch_is_nose_down():
    az, el = to_base_frame(0.0, 0.0, {"head_pitch": 0.3}, FRAME)
    assert az == pytest.approx(0.0, abs=1e-9) and el == pytest.approx(-0.3)


def test_roll_tilts_off_axis_bearing():
    az, el = to_base_frame(0.0, 0.0, {"head_roll": 0.3}, FRAME)
    assert (az, el) == pytest.approx((0.0, 0.0))
    az, el = to_base_frame(math.pi / 2, 0.0, {"head_roll": 0.3}, FRAME)
    assert el == pytest.approx(0.3) and az == pytest.approx(math.pi / 2)


def test_pitch_then_yaw_composition_order():
    az0, el0 = 0.4, 0.2
    pitch, yaw = 0.5, 0.7
    az, el = to_base_frame(az0, el0, {"head_pitch": pitch, "head_yaw": yaw}, FRAME)
    x, y, z = (
        math.cos(el0) * math.cos(az0),
        math.cos(el0) * math.sin(az0),
        math.sin(el0),
    )
    x, z = x * math.cos(pitch) + z * math.sin(pitch), -x * math.sin(pitch) + z * math.cos(pitch)
    x, y = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    assert (az, el) == pytest.approx((math.atan2(y, x), math.asin(z)))
    # yaw first, then pitch gives a different answer, so the order is pinned
    x, y, z = (
        math.cos(el0) * math.cos(az0),
        math.cos(el0) * math.sin(az0),
        math.sin(el0),
    )
    x, y = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    x, z = x * math.cos(pitch) + z * math.sin(pitch), -x * math.sin(pitch) + z * math.cos(pitch)
    assert abs(math.asin(z) - el) > 0.01


def test_observations_from_turned_head_match_the_same_entry():
    w, _, clock, _ = setup()
    a = seen(w, obs(clock, 40, category="person"))
    clock.advance(1)
    r = w.observe(obs(clock, 0, category="person"), {"head_yaw": math.radians(40)})
    assert not r.created and r.entry is not None and r.entry.entry_id == a


# ---------------------------------------------------------------- events, privacy, threads


def test_snapshot_publishes_without_embeddings():
    w, _, clock, ev = setup()
    w.observe(obs(clock, 0, category="person", embedding=np.ones(4)))
    w.observe(obs(clock, 90, kind="object", category="cup"))
    snap = w.snapshot(clock.t)
    assert len(snap) == 2
    assert ev["world.snapshot"][0] == snap
    assert all(e.embedding is None for e in snap)
    assert "embedding" not in repr(w.entries[0])


def test_no_embeddings_in_any_payload_or_result():
    w, _, clock, ev = setup()
    thr, blend = CFG.moved_threshold_deg, CFG.position_blend
    emb = np.ones(4)
    r = w.observe(obs(clock, 0, identity="p1", category="person", embedding=emb))
    assert r.entry is not None and r.entry.embedding is None
    clock.advance(1)
    r = w.observe(obs(clock, 3 * thr / blend, identity="p1", category="person", embedding=emb))
    assert r.entry is not None and r.entry.embedding is None
    gone = w.tick(clock.advance(CFG.half_life_s.person * 10), always_in_view)
    assert gone and gone[0].embedding is None
    for topic in ("world.entity_appeared", "world.entity_moved", "world.entity_gone"):
        assert ev[topic], topic
        assert all(p.embedding is None for p in ev[topic])


def test_raising_subscriber_does_not_abort_tick_or_others():
    w, bus, clock, ev = setup()

    def boom(_msg):
        raise RuntimeError("subscriber bug")

    bus.subscribe("world.entity_gone", boom)
    a = seen(w, obs(clock, 0, category="person"))
    b = seen(w, obs(clock, 90, category="person"))
    gone = w.tick(clock.advance(CFG.half_life_s.person * 10), always_in_view)
    assert {g.entry_id for g in gone} == {a, b}
    assert w.entries == []
    assert len(ev["world.entity_gone"]) == 2


def test_events_published_outside_the_lock():
    w, bus, clock, _ = setup()
    got: list[bool] = []

    def probe(_msg):
        t = threading.Thread(target=lambda: got.append(w._lock.acquire(timeout=1)))
        t.start()
        t.join()
        if got and got[-1]:
            w._lock.release()

    bus.subscribe("world.entity_appeared", probe)
    w.observe(obs(clock, 0, category="person"))
    assert got == [True]


def test_concurrent_observe_and_tick():
    w, _, _, _ = setup()
    errors: list[Exception] = []
    ids: list[str] = []

    def producer(n: int) -> None:
        try:
            for i in range(200):
                o = Observation(
                    kind="object",
                    stamp=float(i),
                    source=f"s{n}",
                    azimuth_rad=(n * 40 + i % 3) * DEG,
                    elevation_rad=0.0,
                    category=f"c{n}-{i % 5}",
                )
                r = w.observe(o)
                if r.created and r.entry:
                    ids.append(r.entry.entry_id)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def ticker() -> None:
        try:
            for i in range(200):
                w.tick(float(i), always_in_view)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=producer, args=(n,)) for n in range(4)]
    threads.append(threading.Thread(target=ticker))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------- config


def test_world_config_validation():
    with pytest.raises(ValueError):
        WorldConfig(initial_uncertainty_deg=1.0, min_uncertainty_deg=2.0)
    with pytest.raises(ValueError):
        WorldConfig(forget_confidence=0.0)
    with pytest.raises(ValueError):
        WorldConfig(out_of_view_decay_factor=0.0)
