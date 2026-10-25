"""SPEC 7 (Speaking motion), 11A: the speaking motion config and the manifest map validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RoboEmotConfig, SpeechMotionConfig, SpeechMotionMap
from roboemot.manifest import load_manifest

MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)
SHAPE = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST, allow_unmeasured_dynamics=True)


def stack_with(manifest, smap: dict) -> ConfigStack:
    behaviour = {**manifest.model.behaviour, "speech_motion_map": smap}
    stack = ConfigStack()
    stack.set_manifest_behaviour(behaviour, manifest.channels, body_yaw="body_yaw")
    return stack


def test_the_presets_ship_the_design_table_and_lively_is_the_default():
    sm = SpeechMotionConfig()
    assert sm.enabled and sm.liveliness == "lively"
    assert sm.level() == sm.levels.lively
    table = {
        "beat_deg": (1.0, 1.5, 2.5),
        "phrase_nod_deg": (2.0, 3.0, 5.0),
        "nod_share": (0.3, 0.5, 0.7),
        "question_tilt_deg": (3.0, 5.0, 7.0),
        "accent_deg": (0.0, 6.0, 10.0),
        "emphasis_min_interval_s": (2.0, 1.2, 0.8),
        "breath_deg": (1.0, 1.5, 2.0),
        "sway_deg": (0.0, 0.0, 1.5),
        "arousal_gain": (0.3, 0.5, 0.8),
    }
    for field, values in table.items():
        got = tuple(getattr(getattr(sm.levels, p), field) for p in ("subtle", "natural", "lively"))
        assert got == values, field


def test_every_new_field_has_a_range_and_the_degrees_a_unit():
    for model in (SpeechMotionConfig, type(SpeechMotionConfig().levels.natural)):
        for name, f in model.model_fields.items():
            extra = f.json_schema_extra or {}
            if name.endswith(("_deg", "_ms", "_s")):
                assert extra.get("unit"), name


def test_mood_gain_labels_must_be_affect_labels_and_fade_fits_the_crossfade():
    with pytest.raises(ValidationError, match="mood_gain"):
        RoboEmotConfig.model_validate({"speech_motion": {"mood_gain": {"glee": 1.0}}})
    with pytest.raises(ValidationError, match="mood_gain"):
        RoboEmotConfig.model_validate({"speech_motion": {"mood_gain": {"joy": 9.0}}})
    with pytest.raises(ValidationError, match="fade_ms"):
        RoboEmotConfig.model_validate(
            {"speech_motion": {"fade_ms": 200}, "motion": {"arbiter": {"crossfade_ms": 150}}}
        )


def test_the_tilt_hold_must_lie_inside_its_keyframes():
    tilt = {"curve": "minjerk", "channels": {"a": SHAPE}}
    with pytest.raises(ValidationError, match="hold_at_s"):
        SpeechMotionMap.model_validate({"question_tilt": tilt})
    with pytest.raises(ValidationError, match="hold_at_s"):
        SpeechMotionMap.model_validate({"question_tilt": {**tilt, "hold_at_s": 1.0}})
    ok = SpeechMotionMap.model_validate({"question_tilt": {**tilt, "hold_at_s": 0.5}})
    assert ok.question_tilt.hold_at_s == 0.5


def test_the_map_signs_are_plus_or_minus_one():
    with pytest.raises(ValidationError):
        SpeechMotionMap.model_validate({"beat": {"a": 2}})


def test_map_validation_rejects_unknown_channels_body_yaw_and_translations(manifest):
    bad = {
        "unknown": {"beat": {"no.such.dof": 1}},
        "body": {"sway": {"body_yaw": 1}},
        "body gesture": {"phrase_nod": {"channels": {"body_yaw": SHAPE}}},
        "translation": {"breath": {"channels": {"head.z": SHAPE}}},
    }
    for case, smap in bad.items():
        with pytest.raises(ConfigError, match="speech_motion_map"):
            stack_with(manifest, smap)
        assert case


def test_a_runtime_override_naming_body_yaw_is_refused(manifest):
    stack = stack_with(manifest, {})
    with pytest.raises(ConfigError, match="turns the body"):
        stack.set("speech_motion_map.sway", {"body_yaw": 1})
