"""SPEC 8 (Robot expression), 11A: the ``expression.reactions`` settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import (
    SITUATIONS,
    ExpressionConfig,
    ExpressionReactionsConfig,
    RoboEmotConfig,
    SituationConfig,
)


def test_defaults_match_the_design():
    r = RoboEmotConfig().expression.reactions
    assert r.enabled and r.model_asks
    assert r.didnt_catch.max_age_s == 10 and r.failed_action.max_age_s == 30
    assert r.returning_person.max_age_s == 30 and r.greet_absence_s == 1800
    assert r.failed_action.gesture == "puzzled" and r.shown_object.gesture == ""
    assert r.shown_object.max_age_s == 60 and r.shown_object.cooldown_s == 120
    assert r.didnt_catch.cooldown_s == 15 and r.failed_action.cooldown_s == 120
    assert r.returning_person.cooldown_s == 0 and r.turn_timeout_s == 5.0
    assert all(r.situation(name).enabled for name in SITUATIONS)


def test_situation_names_each_situation():
    r = ExpressionReactionsConfig()
    assert SITUATIONS == ("didnt_catch", "failed_action", "shown_object", "returning_person")
    assert r.situation("failed_action") is r.failed_action


def test_ranges_refuse_out_of_bounds():
    with pytest.raises(ValidationError):
        SituationConfig(max_age_s=0, cooldown_s=1)
    with pytest.raises(ValidationError):
        ExpressionReactionsConfig(greet_absence_s=59)
    with pytest.raises(ValidationError):
        ExpressionReactionsConfig(turn_timeout_s=31)


def test_existing_expression_fields_untouched():
    assert ExpressionConfig().max_amplitude == 2.0
