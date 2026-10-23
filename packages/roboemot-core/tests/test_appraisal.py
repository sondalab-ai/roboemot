"""SPEC 7: bus to appraisal events, `when` conditions, and the affect runner."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboemot.affect import AffectEngine
from roboemot.affect.appraisal import AppraisalAdapter
from roboemot.affect.runner import AffectRunner
from roboemot.bus import Bus
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import AppraisalConfig, RoboEmotConfig
from roboemot.vision.types import FacePercept, PersonRecognized
from roboemot.world.types import Entry

CFG = RoboEmotConfig()
AP = CFG.appraisal
PRESETS = Path(__file__).resolve().parents[3] / "configs" / "presets"


def face(track=1, stamp=0.0, owner=None, new=True, age="unknown") -> FacePercept:
    return FacePercept(
        track_id=track, stamp=stamp, box=(0, 0, 1, 1), score=0.9, azimuth_rad=0, elevation_rad=0,
        owner_id=owner, anonymous_id=None if owner else "anon", age_group=age, age_source="x",
        gender="unknown", gender_source="x", attribute_confidence=0.0, new_track=new,
    )  # fmt: skip


def recognized(age="adult", track=1, stamp=0.0) -> PersonRecognized:
    return PersonRecognized(track, "p-1", stamp, 0.0, 0.0, age)


def entry(kind="animal", category="cat") -> Entry:
    return Entry("e-1", kind, category, "", 0, 0, None, 0, 1.0, 0.0, 0.0)


@pytest.fixture()
def rig():
    bus, events = Bus(), []
    adapter = AppraisalAdapter(bus, AP, lambda n, t, p: events.append((n, t, p)))
    return bus, adapter, events


def names(events) -> list[str]:
    return [e[0] for e in events]


@pytest.mark.parametrize("age", ["adult", "child", "teen", "senior", "unknown"])
def test_person_recognized_is_one_event_whatever_the_age(rig, age):
    """Owner decision 2026-10-23: no age groups; the payload carries the id alone."""
    bus, _, events = rig
    bus.publish("person.recognized", recognized(age), 1.0)
    assert names(events) == ["person.known"]
    assert events[0][2] == {"person_id": "p-1"}


def test_animal_appearing_maps_and_other_kinds_do_not(rig):
    bus, _, events = rig
    bus.publish("world.entity_appeared", entry("person"), 1.0)
    bus.publish("world.entity_appeared", entry("object", "cup"), 1.0)
    assert not events
    bus.publish("world.entity_appeared", entry("animal", "cat"), 2.0)
    assert events == [("animal.appears", 2.0, {"entity_id": "e-1", "category": "cat"})]


def test_unknown_face_fires_once_after_the_decision_window(rig):
    bus, adapter, events = rig
    w = AP.unknown_face_after_s
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    adapter.tick(0.1)
    assert not events
    bus.publish("percept.face", face(stamp=w - 0.5, new=False), w - 0.5)
    bus.publish("percept.face", face(stamp=w, new=False), w)
    adapter.tick(w)
    assert names(events) == ["person.unknown_appears"]
    assert events[0][2]["track_id"] == 1
    bus.publish("percept.face", face(stamp=w + 1, new=False), w + 1)
    adapter.tick(w + 1)
    assert names(events) == ["person.unknown_appears"]  # once per track


def test_a_face_that_gets_recognized_never_fires_unknown(rig):
    bus, adapter, events = rig
    w = AP.unknown_face_after_s
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    bus.publish("person.recognized", recognized(stamp=1.0), 1.0)
    bus.publish("percept.face", face(stamp=2.0, owner="p-1", new=False), 2.0)
    adapter.tick(w + 1)
    assert names(events) == ["person.known"]


def test_an_owner_on_the_face_percept_alone_cancels_the_pending_track(rig):
    bus, adapter, events = rig
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    bus.publish("percept.face", face(stamp=1.0, owner="p-1", new=False), 1.0)
    adapter.tick(AP.unknown_face_after_s + 1)
    assert not events


def test_a_face_gone_inside_the_window_fires_nothing(rig):
    bus, adapter, events = rig
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    adapter.tick(AP.track_lost_s + 0.5)  # last seen long ago, window not reached
    adapter.tick(AP.unknown_face_after_s + 1)
    assert not events


def test_idle_long_fires_once_and_rearms_on_interaction(rig):
    bus, adapter, events = rig
    idle = AP.idle_after_s
    adapter.tick(idle - 1)
    assert not events
    adapter.tick(idle)
    adapter.tick(idle + 100)
    assert names(events) == ["idle.long"]
    bus.publish("attention.target", object(), idle + 200)
    adapter.tick(idle + 201)
    assert names(events) == ["idle.long"]
    adapter.tick(2 * idle + 200)
    assert names(events) == ["idle.long", "idle.long"]


@pytest.mark.parametrize(
    ("topic", "payload"),
    [
        ("percept.face", face(stamp=500.0, owner="p-1", new=False)),
        ("person.recognized", recognized()),
        ("percept.utterance", SimpleNamespace(addressed=True)),
    ],
)
def test_faces_and_recognition_count_as_interaction(rig, topic, payload):
    bus, adapter, events = rig
    bus.publish(topic, payload, AP.idle_after_s - 1)
    adapter.tick(AP.idle_after_s + 100)
    assert "idle.long" not in names(events)


def test_attention_cleared_is_not_interaction(rig):
    bus, adapter, events = rig
    bus.publish("attention.target", None, AP.idle_after_s - 1)
    adapter.tick(AP.idle_after_s + 1)
    assert names(events) == ["idle.long"]


def test_unknown_face_still_fires_at_the_resting_vision_rate():
    bus, events = Bus(), []
    period = 20.0  # night rate: one detection every 20 s
    adapter = AppraisalAdapter(
        bus, AP, lambda n, t, p: events.append(n), detection_period=lambda: period
    )
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    adapter.tick(AP.unknown_face_after_s + AP.track_lost_s + 1)  # past track_lost_s, one period
    assert events == ["person.unknown_appears"]
    bus.publish("percept.face", face(track=2, stamp=100.0), 100.0)
    adapter.tick(100.0 + AP.track_lost_periods * period + 1)  # gone for good: no event
    assert events == ["person.unknown_appears"]


def test_announced_tracks_are_pruned_once_lost(rig):
    bus, adapter, events = rig
    bus.publish("percept.face", face(stamp=0.0), 0.0)
    bus.publish("percept.face", face(stamp=AP.unknown_face_after_s, new=False), 3.0)
    adapter.tick(AP.unknown_face_after_s)
    assert adapter._announced
    adapter.tick(AP.unknown_face_after_s + AP.track_lost_s + 1)
    assert not adapter._announced


def test_no_event_names_an_age_group():
    from roboemot.config.schema import APPRAISAL_EVENTS

    assert not [e for e in APPRAISAL_EVENTS if e.startswith("person.known_")]


def test_close_detaches_from_the_bus(rig):
    bus, adapter, events = rig
    adapter.close()
    bus.publish("person.recognized", recognized(), 1.0)
    assert not events


# ----------------------------------------------------------------------------- when / rules


def cfg_with(*rules: dict) -> AppraisalConfig:
    plugin_events = ["x", "y"]
    return AppraisalConfig.model_validate({"rules": list(rules), "extra_events": plugin_events})


def test_when_requires_every_field_to_equal():
    e = AffectEngine(
        CFG.affect,
        cfg_with({"event": "x", "valence": 0.5, "when": {"kind": "cat", "near": True}}),
    )
    assert not e.on_event("x", 0.0)
    assert not e.on_event("x", 0.0, {"kind": "cat"})  # missing field
    assert not e.on_event("x", 0.0, {"kind": "dog", "near": True})
    assert e.on_event("x", 0.0, {"kind": "cat", "near": True, "extra": 1})
    assert e.valence == pytest.approx(0.5)


def test_rules_for_one_event_with_different_conditions_are_independent():
    e = AffectEngine(
        CFG.affect,
        cfg_with(
            {"event": "x", "valence": 0.5, "when": {"k": "a"}},
            {"event": "x", "valence": -0.5, "when": {"k": "b"}},
            {"event": "x", "arousal": 0.2},
        ),
    )
    e.on_event("x", 0.0, {"k": "b"})
    assert e.valence == pytest.approx(-0.5) and e.arousal == pytest.approx(0.2)


def test_cooldown_is_per_rule_and_a_non_matching_rule_does_not_burn_it():
    e = AffectEngine(
        CFG.affect, cfg_with({"event": "x", "valence": 0.1, "when": {"k": 1.0}, "cooldown_s": 10})
    )
    assert not e.on_event("x", 0.0, {"k": 2.0})
    assert e.on_event("x", 1.0, {"k": 1.0})
    assert not e.on_event("x", 5.0, {"k": 1.0})
    assert e.on_event("x", 11.0, {"k": 1.0})


def test_when_accepts_bool_string_and_float_and_existing_rules_stay_valid():
    parsed = cfg_with({"event": "x", "when": {"a": True, "b": "s", "c": 1.5}}).rules[0]
    assert parsed.when == {"a": True, "b": "s", "c": 1.5}
    assert all(r.when == {} for r in AP.rules)


def test_keyed_merge_keeps_conditional_rules_apart(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text(
        "schema_version: 1\nappraisal:\n  rules:\n"
        "    - {event: person.known, when: {person_id: p-1}, valence: 0.9}\n"
        "    - {event: person.known, valence: 0.1}\n"
    )
    rules = ConfigStack.from_files(app=app).effective().appraisal.rules
    known = [r for r in rules if r.event == "person.known"]
    assert [(r.when, r.valence) for r in known] == [({}, 0.1), ({"person_id": "p-1"}, 0.9)]


# ----------------------------------------------------------------------------- defaults


def react(*events: str):
    e = AffectEngine(CFG.affect, AP)
    for ev in events:
        e.on_event(ev, 0.0)
    return e.state(0.0)


def test_default_rules_separate_a_known_person_and_a_stranger():
    known = react("person.known")
    stranger = react("person.unknown_appears")
    assert known.valence == pytest.approx(0.5) and known.arousal == pytest.approx(0.2)
    assert known.label == "calm"
    assert stranger.label == "curiosity" and 0 < stranger.valence < 0.1


def test_curious_preset_stranger_rule_fires():
    cfg = ConfigStack.from_files(preset="curious", preset_dirs=[PRESETS]).effective()
    e = AffectEngine(cfg.affect, cfg.appraisal)
    assert e.on_event("person.unknown_appears", 0.0)
    assert e.valence == pytest.approx(0.2) and e.state(0.0).label == "curiosity"
    assert len([r for r in cfg.appraisal.rules if r.event == "person.unknown_appears"]) == 1


# ----------------------------------------------------------------------------- runner


def test_runner_publishes_at_update_hz_and_not_faster():
    bus, got = Bus(), []
    bus.subscribe("affect.state", lambda m: got.append(m.stamp))
    runner = AffectRunner(CFG.affect, AP, bus)
    period = 1 / CFG.affect.update_hz
    for i in range(1000):
        runner.tick(i * period / 5)  # the app loop runs five times faster than the affect rate
    assert len(got) == pytest.approx(1000 / 5, abs=2)
    gaps = [b - a for a, b in zip(got, got[1:], strict=False)]
    assert min(gaps) >= period * 4 / 5 - 1e-9  # on the grid, at most one loop tick early


def test_runner_with_a_slow_loop_publishes_every_tick():
    bus, got = Bus(), []
    bus.subscribe("affect.state", lambda m: got.append(m.stamp))
    runner = AffectRunner(CFG.affect, AP, bus)
    for i in range(5):
        runner.tick(i * 0.06)  # slower than 20 Hz
    assert len(got) == 5


def test_runner_routes_bus_events_into_the_one_engine_and_publishes_the_state():
    bus, got = Bus(), []
    bus.subscribe("affect.state", lambda m: got.append(m.payload))
    runner = AffectRunner(CFG.affect, AP, bus)
    bus.publish("person.recognized", recognized("child"), 1.0)
    runner.tick(1.0)
    assert got[-1].valence > 0.4 and got[-1].label == "calm"
    assert got[-1] is not runner.engine


def test_runner_set_circadian_moves_the_baseline():
    runner = AffectRunner(CFG.affect, AP, Bus())
    resting = CFG.circadian.states["resting"]
    runner.set_circadian(resting)
    assert runner.engine.baseline_arousal == resting.arousal_baseline
    for i in range(1, 300):
        runner.tick(float(i))
    assert runner.engine.arousal == pytest.approx(resting.arousal_baseline, abs=1e-3)


def test_runner_apply_config_swaps_rules_and_keeps_state():
    runner = AffectRunner(CFG.affect, AP, Bus())
    runner.engine.impulse(0.0, valence=0.3)
    runner.apply_config(CFG.affect, cfg_with({"event": "y", "arousal": 0.5}))
    runner.on_event("y", 1.0)
    assert runner.engine.valence > 0 and runner.engine.arousal > 0
    assert not runner.engine.on_event("person.known", 2.0)  # the old rule is gone


def test_the_engine_is_safe_under_concurrent_events_and_ticks():
    rules = cfg_with({"event": "x", "valence": 0.001, "cooldown_s": 0})
    runner = AffectRunner(CFG.affect, rules, Bus())
    errors = []

    def spam():
        try:
            for i in range(500):
                runner.on_event("x", i * 0.001)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=spam) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(200):
        runner.tick(i * 0.001)
    for t in threads:
        t.join()
    assert not errors


# ----------------------------------------------------------------------------- fix round


def test_a_stale_stamp_neither_decays_twice_nor_shortens_the_label_hold():
    e = AffectEngine(CFG.affect, cfg_with({"event": "x", "valence": 0.4, "label": "joy"}))
    e.impulse(100.0, valence=0.8)
    e.update(110.0)
    v = e.valence
    e.update(105.0)  # an older capture stamp
    assert e.valence == pytest.approx(v)
    e.update(110.0)
    assert e.valence == pytest.approx(v)  # no second decay of 105..110
    e.on_event("x", 104.0)  # stale stamp: the hold still starts at the engine's time
    assert e.state(110.0 + CFG.affect.event_label_hold_s - 0.1).label == "joy"


def test_cooldowns_follow_the_rule_through_a_reordered_reload():
    a = {"event": "x", "valence": 0.1, "cooldown_s": 100}
    b = {"event": "y", "arousal": 0.1, "cooldown_s": 100}
    e = AffectEngine(CFG.affect, cfg_with(a, b))
    assert e.on_event("x", 0.0) and e.on_event("y", 0.0)
    e.set_config(CFG.affect, cfg_with(b, a))
    assert not e.on_event("x", 1.0) and not e.on_event("y", 1.0)
    e.set_config(CFG.affect, cfg_with(b))  # x gone: its cooldown is dropped
    e.set_config(CFG.affect, cfg_with(b, a))
    assert e.on_event("x", 2.0) and not e.on_event("y", 2.0)


def test_cooldown_is_per_subject():
    e = AffectEngine(CFG.affect, AP)
    assert e.on_event("person.known", 0.0, {"person_id": "p-1"})
    assert e.on_event("person.known", 1.0, {"person_id": "p-2"})  # another person reacts
    assert not e.on_event("person.known", 2.0, {"person_id": "p-1"})  # the same one does not
    assert e.on_event("person.known", 61.0, {"person_id": "p-1"})


def test_rule_identity_normalizes_numbers_for_cooldown_and_merge():
    from roboemot.config.schema import rule_identity

    one = rule_identity("x", {"k": 1})
    assert one == rule_identity("x", {"k": 1.0}) == rule_identity("x", {"k": True})
    assert rule_identity("x", {"a": 1, "b": "s"}) == rule_identity("x", {"b": "s", "a": 1.0})
    assert rule_identity("x", {"k": "1"}) != one


def test_unknown_event_names_and_labels_are_rejected_at_config_load(tmp_path):
    head = "schema_version: 1\nappraisal:\n"
    bad_event = tmp_path / "event.yaml"
    bad_event.write_text(head + "  rules:\n    - {event: person.recognized, valence: 0.1}\n")
    with pytest.raises(ConfigError, match=r"person\.recognized"):
        ConfigStack.from_files(app=bad_event).effective()
    bad_label = tmp_path / "label.yaml"
    bad_label.write_text(head + "  rules:\n    - {event: idle.long, label: nostalgia}\n")
    with pytest.raises(ConfigError, match="nostalgia"):
        ConfigStack.from_files(app=bad_label).effective()
    plugin = tmp_path / "plugin.yaml"
    plugin.write_text(
        head + "  extra_events: [door.knock]\n  rules:\n    - {event: door.knock, valence: 0.1}\n"
        "    - {event: link.cloud.lost}\n"
    )
    assert ConfigStack.from_files(app=plugin).effective().appraisal.rules


def test_duplicate_rules_are_rejected():
    with pytest.raises(ValueError, match="same"):
        cfg_with({"event": "x"}, {"event": "x"})


def test_unaddressed_speech_is_not_interaction(rig):
    bus, adapter, events = rig
    bus.publish("percept.utterance", SimpleNamespace(addressed=False), AP.idle_after_s - 1)
    adapter.tick(AP.idle_after_s + 100)
    assert "idle.long" in names(events)
