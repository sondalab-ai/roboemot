"""SPEC 5A world model: matching, fading, events and the frame transform."""

from __future__ import annotations

import math

import numpy as np
import pytest
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.world import FrameSpec, Observation, WorldModel, to_base_frame

CFG = RoboEmotConfig().world
FRAME = FrameSpec(yaw_dofs=("body_yaw", "head_yaw"), pitch_dofs=("head_pitch",))
DEG = math.radians(1)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def obs(clock: Clock, az_deg: float = 0.0, **kw) -> Observation:
    kw.setdefault("kind", "person")
    kw.setdefault("source", "vision")
    return Observation(
        stamp=clock.t, azimuth_rad=az_deg * DEG, elevation_rad=kw.pop("el", 0.0), **kw
    )


def setup() -> tuple[WorldModel, Bus, Clock, dict[str, list]]:
    bus = Bus()
    events: dict[str, list] = {}
    for topic in (
        "world.entity_appeared",
        "world.entity_moved",
        "world.entity_gone",
        "world.snapshot",
    ):
        events[topic] = []
        bus.subscribe(topic, lambda m, t=topic: events[t].append(m.payload))
    return WorldModel(CFG, FRAME, bus), bus, Clock(), events


def always_in_view(az: float, el: float) -> bool:
    return True


def never_in_view(az: float, el: float) -> bool:
    return False


def test_topics_registered():
    for t in ("percept.entity", "world.entity_appeared", "world.entity_moved", "world.entity_gone"):
        assert t in TOPICS
    assert "world.snapshot" in TOPICS


def test_new_entry_and_appeared_event():
    w, _, clock, ev = setup()
    r = w.observe(obs(clock, 10, category="person", confidence=0.8))
    assert r.created and r.entry is not None
    assert r.entry.identity == "unknown"
    assert r.entry.confidence == pytest.approx(CFG.initial_confidence * 0.8)
    assert len(ev["world.entity_appeared"]) == 1
    assert ev["world.entity_appeared"][0].embedding is None


def test_identity_match_regardless_of_position():
    w, _, clock, _ = setup()
    a = w.observe(obs(clock, 0, identity="p1", category="face")).entry
    clock.advance(1)
    b = w.observe(obs(clock, 120, identity="p1", category="face"))
    assert not b.created and b.entry is a
    assert len(w.entries) == 1


def test_different_known_identity_does_not_match_by_gate():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, identity="p1", category="face"))
    r = w.observe(obs(clock, 1, identity="p2", category="face"))
    assert r.created
    assert len(w.entries) == 2


def test_gate_match_and_miss():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, kind="animal", category="cat"))
    near = w.observe(obs(clock, CFG.match_gate_deg - 1, kind="animal", category="cat"))
    assert not near.created
    far = w.observe(obs(clock, 90, kind="animal", category="cat"))
    assert far.created
    other = w.observe(obs(clock, 0, kind="animal", category="dog"))
    assert other.created


def test_range_gate_only_when_both_have_distance():
    w, _, clock, _ = setup()
    w.observe(obs(clock, 0, kind="object", category="cup", distance_m=1.0))
    far = w.observe(
        obs(clock, 0, kind="object", category="cup", distance_m=1.0 + CFG.range_gate_m + 0.5)
    )
    assert far.created
    no_dist = w.observe(obs(clock, 0, kind="object", category="cup"))
    assert not no_dist.created
    close = w.observe(
        obs(clock, 0, kind="object", category="cup", distance_m=1.0 + CFG.range_gate_m / 2)
    )
    assert not close.created


def test_embedding_breaks_ties():
    w, _, clock, _ = setup()
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    gate = CFG.match_gate_deg
    a = w.observe(obs(clock, 0, kind="animal", category="cat", embedding=e1)).entry
    b = w.observe(obs(clock, gate * 1.2, kind="animal", category="cat", embedding=e2)).entry
    assert a is not b
    probe = gate * 0.6  # inside both gates, angularly nearer to a
    assert w.observe(obs(clock, probe, kind="animal", category="cat")).entry is a
    assert w.observe(obs(clock, probe, kind="animal", category="cat", embedding=e2)).entry is b


def test_bearing_only_cannot_create_but_confirms():
    w, _, clock, ev = setup()
    r = w.observe(obs(clock, 30, kind="unknown", source="doa"))
    assert r.entry is None and not r.created
    assert r.unexplained_bearing == pytest.approx((30 * DEG, 0.0))
    assert w.entries == [] and not ev["world.entity_appeared"]
    e = w.observe(obs(clock, 30, category="person")).entry
    assert e is not None
    before = e.confidence
    clock.advance(1)
    c = w.observe(obs(clock, 31, kind="unknown", source="doa"))
    assert c.entry is e and c.unexplained_bearing is None
    assert e.confidence > before
    assert e.sources == {"vision", "doa"}


def test_confidence_rises_with_matches():
    w, _, clock, _ = setup()
    e = w.observe(obs(clock, 0, category="person")).entry
    assert e is not None
    prev = e.confidence
    for _ in range(3):
        clock.advance(0.1)
        w.observe(obs(clock, 0, category="person"))
        assert e.confidence > prev
        prev = e.confidence
    assert prev <= 1.0


def test_decay_in_view_vs_out_of_view():
    for in_view, rate in ((always_in_view, 1.0), (never_in_view, CFG.out_of_view_decay_factor)):
        w, _, clock, _ = setup()
        e = w.observe(obs(clock, 0, category="person")).entry
        assert e is not None
        c0 = e.confidence
        w.tick(clock.advance(CFG.half_life_s.person), in_view)
        assert e.confidence == pytest.approx(c0 * 0.5**rate)


def test_half_life_per_kind():
    w, _, clock, _ = setup()
    person = w.observe(obs(clock, 0, kind="person", category="p")).entry
    thing = w.observe(obs(clock, 90, kind="object", category="cup")).entry
    assert person is not None and thing is not None
    w.tick(clock.advance(CFG.half_life_s.person), always_in_view)
    assert person.confidence / CFG.initial_confidence == pytest.approx(0.5)
    assert thing.confidence > person.confidence


def test_forgetting_emits_gone():
    w, _, clock, ev = setup()
    e = w.observe(obs(clock, 0, category="person")).entry
    assert e is not None
    gone = w.tick(clock.advance(CFG.half_life_s.person * 10), always_in_view)
    assert gone == [e]
    assert w.entries == []
    assert [g.entry_id for g in ev["world.entity_gone"]] == [e.entry_id]
    assert w.tick(clock.advance(1), always_in_view) == []


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


def test_frame_rotation_with_turned_head():
    az, el = to_base_frame(0.0, 0.0, {"head_yaw": math.radians(40)}, FRAME)
    assert az == pytest.approx(math.radians(40)) and el == pytest.approx(0.0)
    az, el = to_base_frame(math.radians(10), 0.0, {"body_yaw": 0.5, "head_yaw": 0.25}, FRAME)
    assert az == pytest.approx(math.radians(10) + 0.75)
    az, el = to_base_frame(0.0, 0.0, {"head_pitch": 0.3}, FRAME)
    assert az == pytest.approx(0.0, abs=1e-9) and el == pytest.approx(0.3)
    az, el = to_base_frame(math.radians(170), 0.0, {"head_yaw": math.radians(30)}, FRAME)
    assert az == pytest.approx(math.radians(-160))
    assert to_base_frame(0.2, 0.1, None, FRAME) == pytest.approx((0.2, 0.1))
    assert to_base_frame(0.2, 0.1, {"head_yaw": 1.0}, FrameSpec()) == pytest.approx((0.2, 0.1))


def test_observations_from_turned_head_match_the_same_entry():
    w, _, clock, _ = setup()
    a = w.observe(obs(clock, 40, category="person")).entry
    clock.advance(1)
    r = w.observe(obs(clock, 0, category="person"), {"head_yaw": math.radians(40)})
    assert not r.created and r.entry is a


def test_snapshot_publishes_without_embeddings():
    w, _, clock, ev = setup()
    w.observe(obs(clock, 0, category="person", embedding=np.ones(4)))
    w.observe(obs(clock, 90, kind="object", category="cup"))
    snap = w.snapshot(clock.t)
    assert len(snap) == 2
    assert ev["world.snapshot"][0] == snap
    assert all(e.embedding is None for e in snap)
    assert "embedding" not in repr(w.entries[0])
