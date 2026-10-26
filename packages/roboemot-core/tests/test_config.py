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


# ------------------------------------------------------------------ batch, preview, clear


def rules_stack():
    rules = [{"event": "zz.a", "valence": 0.1}, {"event": "zz.b", "valence": 0.2}]
    return ConfigStack(
        app={"appraisal": {"rules": rules, "extra_events": ["zz.a", "zz.b"]}},
    )


def test_set_many_is_all_or_nothing_with_one_event():
    stack = ConfigStack()
    events = []
    stack.subscribe(events.append)
    stack.set_many({"affect.update_hz": 30.0, "personality": "calm"}, source="t")
    assert len(events) == 1 and set(events[0].changed_keys) == {"affect.update_hz", "personality"}
    before = dict(stack._runtime)
    with pytest.raises(ConfigError):
        stack.set_many({"personality": "x", "affect.update_hz": 9999.0}, source="t")
    with pytest.raises(GuardrailError):
        stack.set_many({"personality": "y", "webui.name_max_chars": 200}, source="t")
    assert stack._runtime == before and len(events) == 1
    assert stack.effective().personality == "calm"


def test_set_many_gives_each_approval_to_the_override_that_covers_it():
    stack = ConfigStack()
    stack.set_many(
        {"privacy.multimodal_frames": True, "affect.update_hz": 30.0}, source="t", admin=True
    )
    assert "privacy.multimodal_frames" in stack._runtime["privacy.multimodal_frames"].approved
    assert stack._runtime["affect.update_hz"].approved == {}


def test_preview_changes_nothing_even_on_error():
    stack = ConfigStack()
    stack.set("personality", "calm", source="t")
    before = {k: v.value for k, v in stack._runtime.items()}
    audit, events = len(stack.audit), []
    stack.subscribe(events.append)
    assert stack.preview({"affect.update_hz": 33.0}).affect.update_hz == 33.0
    with pytest.raises(ConfigError):
        stack.preview({"affect.update_hz": 9999.0})
    assert {k: v.value for k, v in stack._runtime.items()} == before
    assert len(stack.audit) == audit and events == []
    assert stack.effective().affect.update_hz == 20.0


def test_clear_subtree_rolls_back_when_the_result_is_invalid():
    stack = ConfigStack()
    stack.set("voice.moods.tender", {"rate": 0.1}, source="t")
    stack.set("voice.moods.tender.pitch_st", 1.0, source="t")
    real, calls = stack.effective, {"n": 0}

    def flaky(person=None):
        calls["n"] += 1
        if calls["n"] == 1:  # the commit's validation
            raise ConfigError("boom")
        return real(person)

    stack.effective = flaky
    with pytest.raises(ConfigError):
        stack.clear("voice.moods.tender", subtree=True)
    stack.effective = real
    assert set(stack._runtime) == {"voice.moods.tender", "voice.moods.tender.pitch_st"}
    assert stack.clear("voice.moods.tender", subtree=True)
    assert stack._runtime == {}


def test_clear_of_an_index_addressed_partial_finds_it_by_identity():
    stack = rules_stack()
    at = next(n for n, r in enumerate(stack.effective().appraisal.rules) if r.event == "zz.b")
    stack.set(f"appraisal.rules.{at}.valence", 0.9, source="t")
    stack.app["appraisal"]["rules"].reverse()
    moved = next(n for n, r in enumerate(stack.effective().appraisal.rules) if r.event == "zz.b")
    assert stack.clear(f"appraisal.rules.{moved}.valence")
    assert stack._runtime == {}
    assert stack.effective().appraisal.rules[moved].valence == 0.2


def test_a_renamed_route_drops_its_override_on_restore(tmp_path):
    from roboemot.webui.settings import SettingsEditor

    route = {"name": "lan", "base_url": "https://api.example.com/v1", "model": "m"}
    stack = ConfigStack(app={"llm_routes": [route]})
    store = tmp_path / "webui_settings.json"
    SettingsEditor(stack, store=store).change("llm_routes.lan.model", "m2")
    renamed = ConfigStack(app={"llm_routes": [{**route, "name": "home"}]})
    SettingsEditor(renamed, store=store).restore()
    assert renamed.effective().llm_routes[0].model == "m"
    assert json.loads(store.read_text())["overrides"] == {}


def test_reset_with_a_stale_file_key_cleans_the_file(tmp_path):
    from roboemot.webui.settings import SettingsEditor

    store = tmp_path / "webui_settings.json"
    store.write_text(json.dumps({"v": 1, "overrides": {"affect.update_hz": 33.0}}))
    store.chmod(0o600)
    stack = ConfigStack()
    editor = SettingsEditor(stack, store=store)
    editor.restore()
    stack.clear("affect.update_hz")  # the live value goes, the file key is stale
    assert editor.reset("affect.update_hz")["changed"] is False
    assert json.loads(store.read_text())["overrides"] == {}


def test_clear_and_set_validate_the_configuration_no_more_than_before(monkeypatch):
    """Barge-in latency: the persona traits worker clears and sets a derived value on every run
    and holds the GIL meanwhile. The old clear validated twice (check and notify), the old set
    three times; guarded commits must not add validations."""
    stack = ConfigStack()
    traits = stack.effective().persona.traits.model_dump()
    stack.set("persona.traits", traits, source="t", layer="derived")
    real, calls = RoboEmotConfig.model_validate.__func__, {"n": 0}

    def counting(cls, *args, **kwargs):
        calls["n"] += 1
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(RoboEmotConfig, "model_validate", classmethod(counting))
    assert stack.clear("persona.traits", source="t", layer="derived")
    assert calls["n"] <= 2
    calls["n"] = 0
    stack.set("persona.traits", traits, source="t", layer="derived")
    assert calls["n"] <= 3


def test_soothe_defaults_validate():
    cfg = RoboEmotConfig()
    assert cfg.soothe.enabled and (cfg.soothe.interval_s, cfg.soothe.jitter) == (900, 0.5)
    pairs = [(v.label, v.sound) for v in cfg.soothe.variants]
    assert pairs == [
        ("calm", "hum"),
        ("joy", "hum"),
        ("joy", "blirp"),
        ("curiosity", "blirp"),
        ("sleepy", "hum"),
    ]
    assert cfg.soothe.variants[-1].states == ["drowsy"]
    assert all("resting" not in v.states for v in cfg.soothe.variants)


@pytest.mark.parametrize(
    "variant",
    [
        {"label": "calm", "sound": "hum", "states": ["awake", "resting"]},
        {"label": "calm", "sound": "hum", "states": []},
        {"label": "calm", "sound": "whistle"},
        {"label": "calm", "sound": "hum", "weight": 0},
        {"label": "calm", "sound": "hum", "weight": -1},
        {"label": "calm", "sound": "hum", "weight": float("nan")},
        {"label": "calm", "sound": "hum", "shape": {"gain_db": 3}},
        {"label": "calm", "sound": "hum", "shape": {"tempo_bpm": [90, 60]}},
        {"label": "calm", "sound": "hum", "shape": {"pitch_hz": [10, 300]}},
    ],
)
def test_soothe_rejects_bad_variants(variant):
    with pytest.raises(ValueError):
        RoboEmotConfig(soothe={"variants": [variant]})


def test_soothe_rejects_an_unknown_label_and_empty_variants():
    with pytest.raises(ValueError, match="affect.labels"):
        RoboEmotConfig(soothe={"variants": [{"label": "euphoria", "sound": "hum"}]})
    with pytest.raises(ValueError, match="at least one"):
        RoboEmotConfig(soothe={"variants": []})


def test_a_layer_replaces_soothe_variants_whole():
    from roboemot.config.stack import deep_merge

    base = RoboEmotConfig().model_dump(mode="json")
    over = {"soothe": {"variants": [{"label": "joy", "sound": "blirp"}]}}
    merged = RoboEmotConfig(**deep_merge(base, over))
    assert [(v.label, v.sound) for v in merged.soothe.variants] == [("joy", "blirp")]
