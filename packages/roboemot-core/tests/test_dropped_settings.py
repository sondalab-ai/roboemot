"""Settings removed without a replacement still load (SPEC 11A, Renamed and removed settings).

The owner decision of 2026-10-22 (SPEC 8, 12) removed ``safety.minors_allowed_llm`` and
``llm_routes.*.allowed_for_minors``. A robot's app config and web UI overrides file written
before it must still start: the keys are ignored, logged by path, never by value.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from roboemot.config import stack as stack_module
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import DROPPED_SETTINGS, ConfigError, ConfigStack, _inner_model
from roboemot.webui import settings as settings_module
from roboemot.webui.settings import SettingsEditor

FILE = "webui_settings.json"
ROUTE_YAML = """\
llm_routes:
  - name: lan
    base_url: https://lan.example/v1
    model: m
"""
APP_YAML = "safety:\n  minors_allowed_llm: true\n" + ROUTE_YAML.replace(
    "    model: m\n", "    model: m\n    allowed_for_minors: true\n"
)
ADDED = 'llm_routes[{"name":"OpenRouterMiniMax"}]'
PATCHED = 'llm_routes[{"name":"lan"}].allowed_for_minors'


@pytest.fixture(autouse=True)
def fresh_log_once(monkeypatch):
    """Each test sees its own log-once state, whatever ran before it in the process."""
    monkeypatch.setattr(stack_module, "_dropped_logged", set())


def write_overrides(path, overrides):
    path.write_text(json.dumps({"v": 1, "overrides": overrides}))
    path.chmod(0o600)


def has_no_removed_fields(cfg):
    assert "minors_allowed_llm" not in cfg.safety.model_dump()
    assert all("allowed_for_minors" not in r.model_dump() for r in cfg.llm_routes)


def schema_has(pattern: str) -> bool:
    """Whether ``pattern`` (``*`` for any entry) names a field of the configuration schema."""
    model = RoboEmotConfig
    for part in pattern.split("."):
        if part == "*":
            continue
        if model is None or part not in model.model_fields:
            return False
        model = _inner_model(model.model_fields[part].annotation)
    return True


def test_no_dropped_setting_names_a_live_field():
    """A live path in ``dropped`` would be discarded on every load without an error."""
    assert schema_has("safety.watchdog_ms") and schema_has("llm_routes.*.model")
    assert DROPPED_SETTINGS
    assert [p for p in DROPPED_SETTINGS if schema_has(p)] == []


def test_an_app_config_with_the_removed_keys_loads_and_logs_the_paths(tmp_path, caplog):
    app = tmp_path / "app.yaml"
    app.write_text(APP_YAML)
    with caplog.at_level("WARNING"):
        stack = ConfigStack.from_files(app=app)
    cfg = stack.effective()
    has_no_removed_fields(cfg)
    assert [r.name for r in cfg.llm_routes] == ["lan"]
    assert "safety.minors_allowed_llm" in caplog.text
    assert "llm_routes.0.allowed_for_minors" in caplog.text
    assert "true" not in caplog.text.lower()


@pytest.mark.parametrize(
    "text",
    [
        "safety:\n  minors_allowed_lm: true\n",
        ROUTE_YAML.replace("    model: m\n", "    model: m\n    allowed_for_minorsX: true\n"),
    ],
    ids=["safety_typo", "route_typo"],
)
def test_a_near_miss_of_a_dropped_key_is_still_rejected(tmp_path, text):
    app = tmp_path / "app.yaml"
    app.write_text(text)
    with pytest.raises(ConfigError):
        ConfigStack.from_files(app=app)


def test_a_hot_reload_of_the_app_config_ignores_the_removed_keys(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("affect: {half_life_s: 7}\n")
    stack = ConfigStack.from_files(app=app)
    time.sleep(0.01)
    app.write_text(APP_YAML + "affect: {half_life_s: 11}\n")
    os.utime(app, (time.time() + 5, time.time() + 5))
    assert stack.poll() is not None
    assert stack.last_error is None
    assert stack.get("affect.half_life_s") == 11
    has_no_removed_fields(stack.effective())


def test_the_removed_keys_are_logged_once_per_file(tmp_path, caplog):
    app = tmp_path / "app.yaml"
    app.write_text(APP_YAML)
    with caplog.at_level("WARNING"):
        ConfigStack.from_files(app=app)
        ConfigStack.from_files(app=app)
    assert caplog.text.count("safety.minors_allowed_llm") == 1


def test_a_web_ui_overrides_file_with_the_removed_keys_keeps_its_route(tmp_path, caplog):
    app = tmp_path / "app.yaml"
    app.write_text(ROUTE_YAML)  # clean: every dropped-key line below comes from the overrides file
    store = tmp_path / FILE
    added = {
        "allowed_for_minors": False,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "minimax",
        "name": "OpenRouterMiniMax",
    }
    write_overrides(store, {ADDED: added, PATCHED: False, "safety.minors_allowed_llm": True})
    stack = ConfigStack.from_files(app=app)
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=store).restore()
    cfg = stack.effective()
    has_no_removed_fields(cfg)
    assert [r.name for r in cfg.llm_routes] == ["lan", "OpenRouterMiniMax"]
    stored = json.loads(store.read_text())["overrides"]
    assert stored == {ADDED: {k: v for k, v in added.items() if k != "allowed_for_minors"}}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(str(store))]
    assert len(lines) == 2  # the two routes share one pattern path: logged once per file
    assert any("llm_routes.*.allowed_for_minors" in line for line in lines)
    assert any("safety.minors_allowed_llm" in line for line in lines)
    assert "true" not in caplog.text.lower() and "false" not in caplog.text.lower()


def test_a_near_miss_in_the_web_ui_file_is_dropped_as_no_longer_acceptable(tmp_path, caplog):
    store = tmp_path / FILE
    write_overrides(store, {"safety.minors_allowed_lm": True, "affect.update_hz": 33.0})
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=store).restore()
    assert set(json.loads(store.read_text())["overrides"]) == {"affect.update_hz"}
    assert "safety.minors_allowed_lm dropped: no longer acceptable" in caplog.text
    assert "removed from the configuration" not in caplog.text


@pytest.mark.parametrize(
    ("key", "value", "kept"),
    [
        ("llm_routes.0.allowed_for_minors", False, None),
        ("safety", {"minors_allowed_llm": True, "watchdog_ms": 150.0}, {"watchdog_ms": 150.0}),
        (
            "llm_routes",
            [{"name": "x", "model": "m", "allowed_for_minors": True}],
            [{"name": "x", "model": "m"}],
        ),
    ],
    ids=["index_form", "safety_section", "routes_list"],
)
def test_index_form_and_section_keys_lose_only_the_dropped_field(key, value, kept, caplog):
    with caplog.at_level("WARNING"):
        out, stripped = settings_module._migrated({key: value}, "overrides-file")
    assert out == ({} if kept is None else {key: kept})
    assert stripped is (kept is not None)
    assert "overrides-file" in caplog.text and "true" not in caplog.text.lower()


# ------------------------------------------------------------------ owner decision 2026-10-23

# Every setting of the adult/minor split, as an app.yaml written before 2026-10-23 holds them.
NO_MINOR_YAML = """\
safety:
  minor_motion_factor: 0.5
  minor_hold_s: 9
  minor_hold_periods: 4
  minor_output_blocklist: {en: [zork*]}
  minor_lines:
    en: {greeting: Hi, robot_name: "I am {name}.", ask_grown_up: Ask, filtered: No}
register:
  child: {motion_amplitude: 1.1, minor_content_policy: true}
  teen: {minor_content_policy: true}
  senior: {}
  unknown: {approach_allowed: false}
  adult: {minor_content_policy: false}
reflexes:
  rules:
    - {name: startle-loud, trigger: loud_onset, primitive: startle, threshold: 6.0,
       amplitude_deg: 12, cooldown_s: 5, duration_ms: 400, suppress_with_minor: true}
conversation:
  transcript_settle_ms: 300
appraisal:
  rules:
    - {event: person.known_child, valence: 0.3, label: joy}
    - {event: person.known_adult, valence: 0.1}
    - {event: person.known, valence: 0.45}
"""
NO_MINOR_PATHS = (
    "safety.minor_motion_factor", "safety.minor_hold_s", "safety.minor_hold_periods",
    "safety.minor_output_blocklist", "safety.minor_lines", "register.child", "register.teen",
    "register.senior", "register.unknown", "register.adult.minor_content_policy",
    "reflexes.rules.0.suppress_with_minor", "conversation.transcript_settle_ms",
    "appraisal.rules[person.known_child]", "appraisal.rules[person.known_adult]",
)  # fmt: skip


def test_an_old_app_config_with_the_minor_settings_loads_and_logs_each_path(tmp_path, caplog):
    app = tmp_path / "app.yaml"
    app.write_text(NO_MINOR_YAML)
    with caplog.at_level("WARNING"):
        cfg = ConfigStack.from_files(app=app).effective()
    for path in NO_MINOR_PATHS:
        assert caplog.text.count(f"setting {path} ignored") == 1, path
    assert "zork" not in caplog.text and "Ask" not in caplog.text  # paths, never values
    assert cfg.appraisal.rules[0].event == "person.known" and cfg.appraisal.rules[0].valence == 0.45
    assert not [r for r in cfg.appraisal.rules if r.event.startswith("person.known_")]
    assert cfg.reflexes.rules[0].name == "startle-loud"
    assert cfg.register.params == RoboEmotConfig().register.params  # adult's only key was dropped


def test_an_old_web_ui_file_with_the_minor_settings_restores_the_rest(tmp_path, caplog):
    store = tmp_path / FILE
    write_overrides(
        store,
        {
            "safety.minor_motion_factor": 0.4,
            "safety.minor_lines.en.greeting": "Hey",
            "safety.minor_output_blocklist.en": ["zork*"],
            "register.child.speech_rate": 1.1,
            "register.unknown.approach_allowed": False,
            "reflexes.rules.turn-to-voice.suppress_with_minor": True,
            "conversation.transcript_settle_ms": 300,
            'appraisal.rules[{"event":"person.known_child"}].valence': 0.2,
            "register.adult.speech_rate": 0.9,  # renamed: the one register keeps the tweak
            "affect.update_hz": 33.0,
        },
    )
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=store).restore()
    cfg = stack.effective()
    assert cfg.affect.update_hz == 33.0 and cfg.register.speech_rate == 0.9
    assert set(json.loads(store.read_text())["overrides"]) == {
        "affect.update_hz",
        "register.speech_rate",
    }
    for path in (
        "safety.minor_motion_factor",
        "register.child",
        "appraisal.rules[person.known_child]",
    ):
        assert path in caplog.text, path
    assert "setting register.adult.speech_rate moved to register.speech_rate" in caplog.text
    assert "zork" not in caplog.text and "Hey" not in caplog.text


def test_an_old_app_config_with_adult_register_values_moves_them_to_the_one_register(
    tmp_path, caplog
):
    """An admin-confirmed adult value (a raised reply cap is a loosening) may sit in app.yaml: it
    moves to the one register instead of failing the start."""
    app = tmp_path / "app.yaml"
    app.write_text(
        "register:\n"
        "  adult: {speech_rate: 1.1, max_reply_sentences: 2, minor_content_policy: false}\n"
        "  child: {speech_rate: 0.7}\n"
    )
    with caplog.at_level("WARNING"):
        cfg = ConfigStack.from_files(app=app).effective()
    assert cfg.register.speech_rate == 1.1 and cfg.register.max_reply_sentences == 2
    assert "setting register.adult.speech_rate moved to register.speech_rate" in caplog.text
    assert "setting register.adult.minor_content_policy ignored" in caplog.text
    assert "setting register.child ignored" in caplog.text
    assert "1.1" not in caplog.text and "0.7" not in caplog.text


AGE_WHEN_YAML = """\
appraisal:
  rules:
    - {event: person.unknown_appears, when: {possible_minor: false}, valence: 0.2}
    - {event: person.known, when: {age_group: child}, valence: 0.9}
    - {event: person.known, when: {person_id: p-1}, valence: 0.7}
"""


def test_old_appraisal_rules_on_an_age_condition_are_dropped_and_logged(tmp_path, caplog):
    """Review ZM S2: a rule whose `when` names age_group or possible_minor could never match
    again (the payloads no longer carry them): it is dropped, by event, not left silently dead."""
    app = tmp_path / "app.yaml"
    app.write_text(AGE_WHEN_YAML)
    with caplog.at_level("WARNING"):
        cfg = ConfigStack.from_files(app=app).effective()
    whens = [r.when for r in cfg.appraisal.rules]
    assert {"person_id": "p-1"} in whens
    assert not [w for w in whens if {"age_group", "possible_minor"} & set(w)]
    assert "setting appraisal.rules[person.unknown_appears] ignored" in caplog.text
    assert "setting appraisal.rules[person.known] ignored" in caplog.text
    store = tmp_path / FILE
    key = 'appraisal.rules[{"event":"person.known","when":{"age_group":"child"}}].valence'
    write_overrides(store, {key: 0.3, "affect.update_hz": 33.0})
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=store).restore()
    assert set(json.loads(store.read_text())["overrides"]) == {"affect.update_hz"}
    assert f"{store}: setting appraisal.rules[person.known] ignored" in caplog.text


def test_a_value_already_at_the_new_register_path_wins(tmp_path, caplog):
    """Review ZM S5: an old adult value next to a value of the one register is dropped, and
    logged as such, in app.yaml and in the web UI file."""
    app = tmp_path / "app.yaml"
    app.write_text("register:\n  speech_rate: 0.8\n  adult: {speech_rate: 1.1}\n")
    with caplog.at_level("WARNING"):
        cfg = ConfigStack.from_files(app=app).effective()
    assert cfg.register.speech_rate == 0.8
    assert "register.adult.speech_rate ignored: register.speech_rate already set" in caplog.text
    assert "moved to" not in caplog.text
    store = tmp_path / FILE
    write_overrides(store, {"register.adult.speech_rate": 1.1, "register.speech_rate": 0.8})
    stack = ConfigStack()
    caplog.clear()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=store).restore()
    assert stack.effective().register.speech_rate == 0.8
    assert set(json.loads(store.read_text())["overrides"]) == {"register.speech_rate"}
    assert f"{store}: setting register.adult.speech_rate ignored: register.speech_rate" in (
        caplog.text
    )


def test_a_preset_with_the_old_settings_loads_through_extends(tmp_path, caplog):
    """Review ZM S5: presets go through the same migration and dropping, also a parent one."""
    (tmp_path / "base.yaml").write_text(
        "safety: {minor_motion_factor: 0.5}\n"
        "appraisal:\n  rules:\n    - {event: person.known_child, valence: 0.3, label: joy}\n"
    )
    (tmp_path / "old.yaml").write_text(
        "extends: base\nregister:\n  adult: {speech_rate: 1.1}\n  child: {pitch_st: 2}\n"
    )
    with caplog.at_level("WARNING"):
        cfg = ConfigStack.from_files(preset="old", preset_dirs=[tmp_path]).effective()
    assert cfg.register.speech_rate == 1.1
    assert not [r for r in cfg.appraisal.rules if r.event.startswith("person.known_")]
    for line in (
        "old.yaml: setting register.adult.speech_rate moved to register.speech_rate",
        "old.yaml: setting register.child ignored",
        "base.yaml: setting safety.minor_motion_factor ignored",
        "base.yaml: setting appraisal.rules[person.known_child] ignored",
    ):
        assert line in caplog.text, line
