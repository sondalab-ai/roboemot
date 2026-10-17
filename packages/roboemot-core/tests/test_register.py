"""SPEC 7, 11A, 12: register selection, the most restrictive rule and persona shaping."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from roboemot.affect.register import (
    TOPIC,
    ActiveRegister,
    Person,
    RegisterTracker,
    apply_overrides,
    most_restrictive,
    resolve_person,
    resolve_register,
)
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import PersonaConfig, RegisterParams, RoboEmotConfig

CFG = RoboEmotConfig()
REG = CFG.register
NEUTRAL = CFG.persona


def person(group="adult", source="declared", **overrides) -> Person:
    return Person(group, source, overrides)


def resolve(*people: Person, persona=NEUTRAL, minor_present=False) -> ActiveRegister:
    return resolve_register(list(people), REG, persona, minor_present=minor_present)


# ------------------------------------------------------------------ per person


def test_overrides_declared_estimated_unknown_in_that_order():
    declared = resolve_person(person("child", "declared"), REG)
    assert declared == ("child", REG.child)
    estimated = resolve_person(person("adult", "estimated"), REG)
    assert estimated == ("adult", REG.adult)
    for source in ("none", "garbage"):
        assert resolve_person(person("child", source), REG) == ("unknown", REG.unknown)
    assert resolve_person(person("martian", "declared"), REG) == ("unknown", REG.unknown)
    # the record's overrides come first: they win field by field over the group's register
    over = resolve_person(
        person("child", "declared", motion_amplitude=0.4, vocabulary="formal"), REG
    )
    assert over[0] == "child"
    assert over[1].motion_amplitude == 0.4 and over[1].vocabulary == "formal"
    assert over[1].speech_rate == REG.child.speech_rate  # the rest is the group's


def test_teen_uses_adult_defaults_with_the_minor_policy_and_senior_is_adult():
    assert REG.teen.model_copy(update={"minor_content_policy": False}) == REG.adult
    assert REG.teen.minor_content_policy and not REG.adult.minor_content_policy
    assert REG.senior == REG.adult
    assert resolve(person("teen")).params.minor_content_policy
    assert resolve(person("senior")).params == REG.adult


def test_overrides_never_loosen_a_guarded_field():
    loose = {"approach_allowed": True, "minor_content_policy": False}
    out = apply_overrides(REG.child, loose)
    assert not out.approach_allowed and out.minor_content_policy
    out = apply_overrides(REG.unknown, loose)
    assert not out.approach_allowed and out.minor_content_policy
    tight = apply_overrides(REG.adult, {"approach_allowed": False, "minor_content_policy": True})
    assert not tight.approach_allowed and tight.minor_content_policy
    free = apply_overrides(REG.adult, {"approach_allowed": True, "speech_rate": 0.8})
    assert free.approach_allowed and free.speech_rate == 0.8


def test_an_invalid_override_is_ignored_and_logged_without_values(caplog):
    with caplog.at_level(logging.WARNING):
        out = apply_overrides(REG.adult, {"motion_amplitude": 99, "secret": "Mia"})
    assert out == REG.adult
    assert "Mia" not in caplog.text and "99" not in caplog.text and "ignored" in caplog.text


# ------------------------------------------------------------------ several people


def test_most_restrictive_is_field_wise_on_the_four_criteria():
    child_adult = resolve(person("child"), person("adult"))
    assert child_adult.group == "child"
    assert child_adult.params.minor_content_policy and not child_adult.params.approach_allowed
    assert child_adult.params.motion_amplitude == 1.0  # the adult's, lower than the child's 1.2
    assert child_adult.params.reaction_intensity == "medium"  # lower than the child's high
    assert child_adult.params.vocabulary == "simple"  # unordered fields: the lead person's

    unknown_child = resolve(person("child"), person(source="none"))
    assert unknown_child.group == "unknown"
    assert unknown_child.params.motion_amplitude == 0.7
    assert unknown_child.params.reaction_intensity == "low"

    teen_adult = resolve(person("teen"), person("adult"))
    assert teen_adult.group == "teen" and teen_adult.params.minor_content_policy
    assert teen_adult.params.approach_allowed  # both allow it
    assert resolve(person("adult"), person("senior")).group == "adult"
    assert resolve(person("adult"), person("adult")).people == 2


def test_most_restrictive_does_not_depend_on_the_order_of_the_people():
    pairs = [("child", REG.child), ("adult", REG.adult), ("unknown", REG.unknown)]
    a = most_restrictive(pairs)
    assert a == most_restrictive(pairs[::-1]) == most_restrictive([pairs[1], pairs[2], pairs[0]])


def test_nobody_in_view_is_unknown_and_a_minor_hold_applies_the_policy():
    none = resolve()
    assert none.group == "unknown" and none.people == 0 and none.params == REG.unknown
    held = resolve(person("adult"), minor_present=True)
    assert held.group == "adult" and held.params.minor_content_policy


# ------------------------------------------------------------------ persona


def test_persona_shapes_the_register_once_whoever_is_present():
    lively = PersonaConfig(traits={"energy": 1.0})
    x = 1.2  # trait_effects.energy.motion_amplitude_x at energy 1
    assert resolve(person("adult"), persona=lively).params.motion_amplitude == pytest.approx(x)
    both = resolve(person("adult"), person("adult"), persona=lively)
    assert both.params.motion_amplitude == pytest.approx(x)  # not x squared
    mixed = resolve(person("child"), person("adult"), persona=lively)
    assert mixed.params.motion_amplitude == pytest.approx(1.0 * x)
    assert resolve(person("adult"), persona=NEUTRAL).params == REG.adult  # neutral: no change


# ------------------------------------------------------------------ tracker and bus


def face(track, group, source, owner=None, stamp=0.0):
    return SimpleNamespace(
        track_id=track, age_group=group, age_source=source, owner_id=owner, stamp=stamp
    )


def test_tracker_publishes_only_on_change_and_never_an_id():
    bus, seen = Bus(), []
    bus.subscribe(TOPIC, lambda m: seen.append(m.payload))
    lookups = []

    def overrides_for(pid):
        lookups.append(pid)
        return {"speech_rate": 0.8}

    tracker = RegisterTracker(bus, CFG.expression, overrides_for=overrides_for)
    assert TOPIC in TOPICS
    first = tracker.tick(0.0, REG, NEUTRAL)
    assert first.group == "unknown" and len(seen) == 1
    tracker.tick(0.1, REG, NEUTRAL)
    assert len(seen) == 1  # unchanged: not republished
    bus.publish("percept.face", face(1, "child", "declared", owner="p-secret"), 1.0)
    now = tracker.tick(1.1, REG, NEUTRAL)
    assert now.group == "child" and now.params.speech_rate == 0.8 and lookups == ["p-secret"]
    assert len(seen) == 2
    assert "p-secret" not in repr(seen) and "p-secret" not in str(now.summary())
    assert "persona_fragment" not in now.summary()
    tracker.close()


def test_a_track_stays_for_the_hold_and_the_detection_periods_then_expires():
    bus, period = Bus(), [float("inf")]
    tracker = RegisterTracker(bus, CFG.expression, detection_period=lambda: period[0])
    hold, periods = CFG.expression.register_hold_s, CFG.expression.register_hold_periods
    bus.publish("percept.face", face(1, "adult", "declared"), 10.0)
    assert tracker.tick(10.0 + hold - 0.1, REG, NEUTRAL).group == "adult"
    assert tracker.tick(10.0 + hold + 0.1, REG, NEUTRAL).group == "unknown"
    bus.publish("percept.face", face(2, "adult", "declared"), 20.0)
    period[0] = 5.0  # resting vision rate: three periods outlive the hold
    assert tracker.tick(20.0 + hold + 1.0, REG, NEUTRAL).group == "adult"
    assert tracker.tick(20.0 + periods * 5.0 + 0.1, REG, NEUTRAL).group == "unknown"


def test_a_failing_registry_lookup_falls_back_to_no_overrides():
    def broken(_pid):
        raise OSError("disk")

    bus = Bus()
    tracker = RegisterTracker(bus, CFG.expression, overrides_for=broken)
    bus.publish("percept.face", face(1, "adult", "declared", owner="p"), 0.0)
    assert tracker.tick(0.1, REG, NEUTRAL).params == REG.adult


def test_resolved_params_are_register_params():
    assert isinstance(resolve(person()).params, RegisterParams)
