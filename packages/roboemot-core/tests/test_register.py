"""SPEC 7, 11A, 12: the one register, a person's overrides, the most restrictive rule and
persona shaping. No age groups (owner decision 2026-10-23)."""

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
    resolve_register,
)
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import PersonaConfig, RegisterParams, RoboEmotConfig

CFG = RoboEmotConfig()
REG = CFG.register
BASE = REG.params
NEUTRAL = CFG.persona


def person(**overrides) -> Person:
    return Person(overrides)


def resolve(*people: Person, persona=NEUTRAL) -> ActiveRegister:
    return resolve_register(list(people), REG, persona)


# ------------------------------------------------------------------ one register


def test_everyone_gets_the_one_register_and_nobody_in_view_too():
    assert resolve().params == BASE and resolve().people == 0
    assert resolve(person()).params == BASE
    assert resolve(person(), person()).people == 2
    assert BASE.approach_allowed and BASE.motion_amplitude == 1.0


def test_the_register_is_flat_and_has_no_age_group():
    assert set(RegisterParams.model_fields) <= set(type(REG).model_fields)
    assert not {"child", "teen", "adult", "senior", "unknown"} & set(type(REG).model_fields)
    assert "minor_content_policy" not in RegisterParams.model_fields
    assert "group" not in resolve().summary()


def test_a_persons_overrides_win_field_by_field():
    over = resolve(person(motion_amplitude=0.4, vocabulary="formal")).params
    assert over.motion_amplitude == 0.4 and over.vocabulary == "formal"
    assert over.speech_rate == BASE.speech_rate  # the rest is the register's


def test_overrides_never_loosen_a_guarded_field():
    tight = BASE.model_copy(update={"approach_allowed": False})
    assert not apply_overrides(tight, {"approach_allowed": True}).approach_allowed
    assert not apply_overrides(BASE, {"approach_allowed": False}).approach_allowed
    free = apply_overrides(BASE, {"approach_allowed": True, "speech_rate": 0.8})
    assert free.approach_allowed and free.speech_rate == 0.8


def test_an_old_record_with_the_removed_minor_policy_keeps_its_other_overrides():
    """A record written before 2026-10-23 may hold ``minor_content_policy``: it is ignored, not
    the reason the person's whole override is thrown away."""
    out = apply_overrides(BASE, {"minor_content_policy": True, "speech_rate": 0.8})
    assert out.speech_rate == 0.8 and "minor_content_policy" not in out.model_dump()


def test_an_invalid_override_is_ignored_and_logged_without_values(caplog):
    with caplog.at_level(logging.WARNING):
        out = apply_overrides(BASE, {"motion_amplitude": 99, "secret": "Mia"})
    assert out == BASE
    assert "Mia" not in caplog.text and "99" not in caplog.text and "ignored" in caplog.text


def test_a_restriction_survives_an_invalid_field_in_the_same_override():
    out = apply_overrides(BASE, {"approach_allowed": False, "motion_amplitude": 99})
    assert not out.approach_allowed and out.motion_amplitude == BASE.motion_amplitude


# ------------------------------------------------------------------ several people


def test_most_restrictive_is_field_wise_on_the_three_criteria():
    calm = person(motion_amplitude=0.6, reaction_intensity="high", vocabulary="simple")
    still = person(approach_allowed=False, reaction_intensity="low", max_reply_sentences=2)
    both = resolve(calm, still).params
    assert not both.approach_allowed  # only if everyone allows it
    assert both.motion_amplitude == 0.6  # the lowest
    assert both.reaction_intensity == "low"  # the lowest
    assert both.max_reply_sentences == 2  # the shortest that is set
    assert both.vocabulary == BASE.vocabulary  # unordered fields: the lead person's (no approach)


def test_most_restrictive_does_not_depend_on_the_order_of_the_people():
    regs = [
        BASE,
        BASE.model_copy(update={"motion_amplitude": 0.5}),
        BASE.model_copy(update={"approach_allowed": False}),
    ]
    a = most_restrictive(regs)
    assert a == most_restrictive(regs[::-1]) == most_restrictive([regs[1], regs[2], regs[0]])


# ------------------------------------------------------------------ persona


def test_persona_shapes_the_register_once_whoever_is_present():
    lively = PersonaConfig(traits={"energy": 1.0})
    x = 1.2  # trait_effects.energy.motion_amplitude_x at energy 1
    assert resolve(person(), persona=lively).params.motion_amplitude == pytest.approx(x)
    both = resolve(person(), person(), persona=lively)
    assert both.params.motion_amplitude == pytest.approx(x)  # not x squared
    assert resolve(person(), persona=NEUTRAL).params == BASE  # neutral: no change


# ------------------------------------------------------------------ tracker and bus


def face(track, owner=None, stamp=0.0):
    return SimpleNamespace(track_id=track, owner_id=owner, stamp=stamp)


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
    assert first.people == 0 and len(seen) == 1
    tracker.tick(0.1, REG, NEUTRAL)
    assert len(seen) == 1  # unchanged: not republished
    bus.publish("percept.face", face(1, owner="p-secret"), 1.0)
    now = tracker.tick(1.1, REG, NEUTRAL)
    assert now.people == 1 and now.params.speech_rate == 0.8 and lookups == ["p-secret"]
    assert len(seen) == 2
    assert "p-secret" not in repr(seen) and "p-secret" not in str(now.summary())
    assert "persona_fragment" not in now.summary()
    tracker.close()


def test_a_track_stays_for_the_hold_and_the_detection_periods_then_expires():
    bus, period = Bus(), [float("inf")]
    tracker = RegisterTracker(bus, CFG.expression, detection_period=lambda: period[0])
    hold, periods = CFG.expression.register_hold_s, CFG.expression.register_hold_periods
    bus.publish("percept.face", face(1), 10.0)
    assert tracker.tick(10.0 + hold - 0.1, REG, NEUTRAL).people == 1
    assert tracker.tick(10.0 + hold + 0.1, REG, NEUTRAL).people == 0
    bus.publish("percept.face", face(2), 20.0)
    period[0] = 5.0  # resting vision rate: three periods outlive the hold
    assert tracker.tick(20.0 + hold + 1.0, REG, NEUTRAL).people == 1
    assert tracker.tick(20.0 + periods * 5.0 + 0.1, REG, NEUTRAL).people == 0


def test_a_failing_registry_lookup_gives_the_register_without_overrides():
    def broken(_pid):
        raise OSError("disk")

    bus = Bus()
    tracker = RegisterTracker(bus, CFG.expression, overrides_for=broken)
    bus.publish("percept.face", face(1, owner="p"), 0.0)
    got = tracker.tick(0.1, REG, NEUTRAL)
    assert got.people == 1 and got.params == BASE


def test_register_overrides_are_validated_when_a_record_is_written():
    from roboemot.registry.records import check_register_overrides

    assert check_register_overrides({"vocabulary": "simple"}) == {"vocabulary": "simple"}
    assert check_register_overrides(None) == {}
    for bad in ({"motion_amplitude": 99}, {"nope": 1}, {"approach_allowed": "maybe"}):
        with pytest.raises(ValueError):
            check_register_overrides(bad)


def test_resolved_params_are_register_params():
    assert type(resolve(person()).params) is RegisterParams
