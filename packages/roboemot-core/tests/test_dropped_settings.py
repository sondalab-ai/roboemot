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
    assert schema_has("safety.minor_hold_s") and schema_has("llm_routes.*.model")
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
        ("safety", {"minors_allowed_llm": True, "minor_hold_s": 6.0}, {"minor_hold_s": 6.0}),
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
