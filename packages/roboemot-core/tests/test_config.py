"""SPEC 11A: layers, validation, guardrails, TTL, audit, hot reload, presets."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from roboemot.config import ConfigError, ConfigStack, GuardrailError, field_meta, resolve_preset
from roboemot.config.schema import RoboEmotConfig

REPO = Path(__file__).resolve().parents[3]
PRESETS = REPO / "configs" / "presets"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_schema_defaults_validate_and_export_json_schema():
    cfg = RoboEmotConfig()
    assert cfg.circadian.quiet_hours == ("22:30", "07:00")
    schema = RoboEmotConfig.model_json_schema()
    assert "affect" in schema["properties"]


def test_layer_precedence():
    stack = ConfigStack(
        defaults={"affect": {"half_life_s": 5}},
        preset={"affect": {"half_life_s": 6}},
        app={"affect": {"half_life_s": 7}},
    )
    assert stack.get("affect.half_life_s") == 7
    person = {"voice": {"expressivity": 0.9}}  # person layer: register, voice, conversation
    assert stack.get("voice.expressivity", person=person) == 0.9
    stack.set("voice.expressivity", 0.3)
    assert stack.get("voice.expressivity", person=person) == 0.3  # runtime wins


def test_unknown_keys_and_out_of_range_are_rejected():
    with pytest.raises(ConfigError, match="extra|Extra"):
        ConfigStack(app={"affect": {"bogus": 1}})
    stack = ConfigStack()
    with pytest.raises(ConfigError):
        stack.set("affect.half_life_s", -3)
    assert stack.get("affect.half_life_s") == 8.0  # rolled back
    with pytest.raises(ConfigError, match="unknown parameter"):
        stack.set("affect.nope", 1)


def test_safety_only_tightens_at_runtime():
    stack = ConfigStack()
    stack.set("safety.minor_motion_factor", 0.4)  # more conservative: allowed
    with pytest.raises(GuardrailError):
        stack.set("safety.minor_motion_factor", 0.9)  # looser: refused
    with pytest.raises(GuardrailError):
        stack.set("safety.minors_allowed_llm", True)
    # loosening is possible only by editing the app layer with admin confirmation
    with pytest.raises(GuardrailError):
        stack.set("safety.minor_motion_factor", 0.9, layer="app")
    stack.clear("safety.minor_motion_factor")
    stack.set("safety.minor_motion_factor", 0.9, layer="app", admin=True)
    assert stack.get("safety.minor_motion_factor") == 0.9


def test_privacy_opt_in_needs_admin():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("privacy.multimodal_frames", True)
    stack.set("privacy.multimodal_frames", True, admin=True)
    assert stack.get("privacy.multimodal_frames") is True


def test_runtime_ttl_expires_and_is_audited(tmp_path):
    clock = Clock()
    audit = tmp_path / "audit.jsonl"
    stack = ConfigStack(clock=clock, audit_path=audit)
    stack.set("voice.expressivity", 0.2, ttl_s=60, source="webui")
    assert stack.get("voice.expressivity") == 0.2
    clock.t += 61
    assert stack.get("voice.expressivity") == 0.7  # expired values are ignored at once
    stack.poll()  # and removed, audited and announced by poll()
    sources = [e.source for e in stack.audit]
    assert sources == ["webui", "ttl"]
    line = json.loads(audit.read_text().splitlines()[0])
    assert line["key"] == "voice.expressivity" and line["new"] == 0.2


def test_change_notification_and_restart_flags():
    stack = ConfigStack()
    seen = []
    stack.subscribe(seen.append)
    stack.set("voice.voices", {"it": "it_IT-riccardo-x_low", "en": "en_US-amy-medium"})
    stack.set("voice.expressivity", 0.5)
    assert seen[0].restart_required == ["voice.voices.it"]
    assert seen[1].changed_keys == ["voice.expressivity"] and not seen[1].restart_required


def test_dict_entries_are_addressable():
    assert field_meta("voice.moods.joy.rate") is not None
    stack = ConfigStack()
    stack.set("voice.moods.joy.pitch_st", 3)
    stack.set("voice.moods.anger", {"rate": 0.1, "instructions": "tense"})
    assert stack.get("voice.moods.anger.instructions") == "tense"


def test_presets_extend_and_merge_rules_by_event():
    curious = resolve_preset("curious", [PRESETS])
    assert curious["personality"] == "curious"
    stack = ConfigStack(preset=curious, preset_dirs=[PRESETS])
    events = [r.event for r in stack.effective().appraisal.rules]
    assert "person.known" in events and events.count("person.unknown_appears") == 1
    assert stack.get("affect.half_life_s") == 6
    stack.apply_preset("calm")
    assert stack.get("voice.expressivity") == 0.4


def test_preset_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b\n")
    (tmp_path / "b.yaml").write_text("extends: a\n")
    with pytest.raises(ConfigError, match="cycle"):
        resolve_preset("a", [tmp_path])


def test_hot_reload_from_file(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("affect: {half_life_s: 7}\n")
    stack = ConfigStack.from_files(app=app, preset="default", preset_dirs=[PRESETS])
    assert stack.get("affect.half_life_s") == 7
    time.sleep(0.01)
    app.write_text("affect: {half_life_s: 11}\n")
    os.utime(app, (time.time() + 5, time.time() + 5))
    change = stack.poll()
    assert change and change.changed_keys == ["affect.half_life_s"]
    assert stack.get("affect.half_life_s") == 11


def test_example_app_config_is_valid():
    stack = ConfigStack.from_files(
        app=REPO / "configs" / "companion.yaml", preset="curious", preset_dirs=[PRESETS]
    )
    routes = stack.effective().llm_routes
    assert [r.name for r in routes] == ["lan", "cloud"]


def test_schema_version_mismatch():
    with pytest.raises(ConfigError, match="schema_version"):
        ConfigStack(app={"schema_version": 99})


# ----------------------------------------------------------------------------- review fixes


def test_section_level_override_cannot_bypass_guardrails():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("safety", {"minor_motion_factor": 1.0, "watchdog_ms": 2000})
    with pytest.raises(GuardrailError):
        stack.set("privacy", {"debug_recording": True, "multimodal_frames": True})
    with pytest.raises(GuardrailError):
        stack.set(
            "llm_routes",
            [{"name": "x", "base_url": "http://a/v1", "model": "m", "allowed_for_minors": True}],
        )
    assert stack.get("safety.watchdog_ms") == 200


def test_section_override_keeps_other_values():
    stack = ConfigStack(app={"affect": {"half_life_s": 9}})
    stack.set("affect", {"update_hz": 30})
    assert stack.get("affect.half_life_s") == 9 and stack.get("affect.update_hz") == 30


def test_privacy_opt_in_with_integer_one_still_needs_admin():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("privacy.debug_recording", 1)


def test_minors_policy_is_guarded():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("register.child.minor_content_policy", False)
    with pytest.raises(GuardrailError):
        stack.set("register.unknown.approach_allowed", True)


def test_loosening_preset_needs_admin(tmp_path):
    (tmp_path / "default.yaml").write_text("schema_version: 1\n")
    (tmp_path / "evil.yaml").write_text(
        "extends: default\nsafety: {minor_motion_factor: 1.0}\nprivacy: {debug_recording: true}\n"
    )
    stack = ConfigStack(preset_dirs=[tmp_path])
    with pytest.raises(GuardrailError):
        stack.apply_preset("evil")
    assert stack.get("safety.minor_motion_factor") == 0.6
    stack.apply_preset("evil", admin=True)
    assert stack.get("safety.minor_motion_factor") == 1.0


def test_invalid_preset_is_rolled_back(tmp_path):
    (tmp_path / "broken.yaml").write_text("affect: {half_life_s: -5}\n")
    stack = ConfigStack(preset_dirs=[tmp_path])
    with pytest.raises(ConfigError):
        stack.apply_preset("broken")
    assert stack.get("affect.half_life_s") == 8.0  # still usable


def test_person_layer_cannot_touch_safety_or_minors():
    stack = ConfigStack()
    person = {
        "safety": {"minor_motion_factor": 1.0},
        "register": {"child": {"minor_content_policy": False, "speech_rate": 1.1}},
    }
    cfg = stack.effective(person)
    assert cfg.safety.minor_motion_factor == 0.6
    assert cfg.register.child.minor_content_policy is True
    assert cfg.register.child.speech_rate == 1.1  # allowed part kept
    assert any(e.source == "person" for e in stack.audit)


def test_hot_reload_refuses_loosening_and_invalid_files(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("affect: {half_life_s: 7}\n")
    stack = ConfigStack.from_files(app=app)

    def write(text: str, bump: int) -> None:
        app.write_text(text)
        os.utime(app, (time.time() + bump, time.time() + bump))

    write("safety: {minor_motion_factor: 1.0}\n", 5)
    assert stack.poll() is None and "loosen" in stack.last_error
    assert stack.get("safety.minor_motion_factor") == 0.6
    write("affect: {half_life_s: -1}\n", 10)
    assert stack.poll() is None and stack.last_error
    assert stack.get("affect.half_life_s") == 7  # last valid config kept
    write("affect: {half_life_s: 12}\n", 15)
    assert stack.poll().changed_keys == ["affect.half_life_s"] and stack.last_error is None


def test_admin_app_edit_is_persisted(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("schema_version: 1\n")
    stack = ConfigStack.from_files(app=app)
    stack.set("safety.minor_motion_factor", 0.9, layer="app", admin=True)
    assert "minor_motion_factor: 0.9" in app.read_text()
    assert stack.poll() is None  # own write is not a reload


def test_ttl_expiry_emits_change_event():
    clock = Clock()
    stack = ConfigStack(clock=clock)
    seen = []
    stack.subscribe(seen.append)
    stack.set("voice.expressivity", 0.2, ttl_s=10)
    clock.t += 11
    change = stack.poll()
    assert change and change.changed_keys == ["voice.expressivity"] and len(seen) == 2


# ----------------------------------------------------------------------------- second review


def test_baseline_never_absorbs_runtime_values():
    """N1: a runtime tightening + an admin approval must not lock the stack afterwards."""
    clock = Clock()
    stack = ConfigStack(clock=clock)
    stack.set("safety.minor_motion_factor", 0.3, ttl_s=60)
    stack.set("privacy.multimodal_frames", True, admin=True)
    clock.t += 61
    stack.poll()  # tightening expired
    stack.set("safety.watchdog_ms", 150)  # a further tightening must still be accepted
    assert stack.get("safety.watchdog_ms") == 150


def test_runtime_privacy_approval_lapses_with_the_override():
    """N2: an admin opt-in with TTL does not become a permanent approval."""
    clock = Clock()
    stack = ConfigStack(clock=clock)
    stack.set("privacy.debug_recording", True, admin=True, ttl_s=60)
    stack.set("voice.expressivity", 0.5)  # unrelated changes still work meanwhile
    clock.t += 61
    stack.poll()
    assert stack.get("privacy.debug_recording") is False
    with pytest.raises(GuardrailError):
        stack.set("privacy.debug_recording", True)


def test_route_names_validated_and_unique():
    """N4: names with dots or duplicates could bypass the minors guardrail."""
    with pytest.raises(ConfigError):
        ConfigStack(
            app={"llm_routes": [{"name": "cloud.v2", "base_url": "http://a/v1", "model": "m"}]}
        )
    with pytest.raises(ConfigError):
        ConfigStack(
            app={
                "llm_routes": [
                    {"name": "r", "base_url": "http://a/v1", "model": "m"},
                    {"name": "r", "base_url": "http://b/v1", "model": "m"},
                ]
            }
        )
    with pytest.raises(ConfigError):
        ConfigStack(app={"llm_routes": [{"name": "r", "base_url": "", "model": "m"}]})


def test_single_route_field_addressable_at_runtime():
    """N7: change one route without wiping the others."""
    stack = ConfigStack.from_files(app=REPO / "configs" / "companion.yaml")
    stack.set("llm_routes.cloud.max_tokens", 100)
    routes = {r.name: r for r in stack.effective().llm_routes}
    assert routes["cloud"].max_tokens == 100 and routes["cloud"].base_url
    assert "lan" in routes
    stack.set("llm_routes", [{"name": "cloud", "max_tokens": 120}])  # merged by name
    routes = {r.name: r for r in stack.effective().llm_routes}
    assert set(routes) == {"lan", "cloud"} and routes["cloud"].max_tokens == 120
    with pytest.raises(GuardrailError):
        stack.set("llm_routes.cloud.allowed_for_minors", True)
