"""Settings removed without a replacement still load (SPEC 11A, Renamed and removed settings).

The owner decision of 2026-10-22 (SPEC 8, 12) removed ``safety.minors_allowed_llm`` and
``llm_routes.*.allowed_for_minors``. A robot's app config and web UI overrides file written
before it must still start: the keys are ignored, logged by path, never by value.
"""

from __future__ import annotations

import json
import os
import time

from roboemot.config.stack import ConfigStack
from roboemot.webui.settings import SettingsEditor

FILE = "webui_settings.json"
APP_YAML = """\
safety:
  minors_allowed_llm: true
llm_routes:
  - name: lan
    base_url: https://lan.example/v1
    model: m
    allowed_for_minors: true
"""
ADDED = 'llm_routes[{"name":"OpenRouterMiniMax"}]'
PATCHED = 'llm_routes[{"name":"lan"}].allowed_for_minors'


def write_overrides(path, overrides):
    path.write_text(json.dumps({"v": 1, "overrides": overrides}))
    path.chmod(0o600)


def has_no_removed_fields(cfg):
    assert "minors_allowed_llm" not in cfg.safety.model_dump()
    assert all("allowed_for_minors" not in r.model_dump() for r in cfg.llm_routes)


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


def test_a_hot_reload_of_the_app_config_ignores_the_removed_keys(tmp_path, caplog):
    app = tmp_path / "app.yaml"
    app.write_text("affect: {half_life_s: 7}\n")
    stack = ConfigStack.from_files(app=app)
    time.sleep(0.01)
    app.write_text(APP_YAML + "affect: {half_life_s: 11}\n")
    os.utime(app, (time.time() + 5, time.time() + 5))
    with caplog.at_level("WARNING"):
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
    app.write_text(APP_YAML)
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
    assert "allowed_for_minors" in caplog.text and "safety.minors_allowed_llm" in caplog.text
    assert "true" not in caplog.text.lower() and "false" not in caplog.text.lower()
