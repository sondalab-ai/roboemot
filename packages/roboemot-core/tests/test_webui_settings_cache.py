"""The settings catalogue is built once per configuration version (SPEC 11A, 12).

A GET of ``/api/settings`` and the ``_rows()`` of a write share one build of the catalogue while
the effective configuration is the same object. Every change that can alter a row (a write, a
reset, a hot reload, a lapsed runtime TTL, a language install, a reveal) gives fresh rows, and
private values never reach a response without the PIN.
"""

from __future__ import annotations

import copy
import json
import os

import pytest
from roboemot.config.catalogue import build_catalogue, descriptor
from roboemot.config.stack import ConfigStack
from roboemot.webui import settings as settings_module
from roboemot.webui.settings import SettingsEditor, _writable
from test_webui_robot_section import BASE, FakePlugin, manifest
from test_webui_settings import PIN, put, serve
from test_webui_settings_write import ROUTE

HALF_LIFE = "affect.half_life_s"
LANGUAGE = "conversation.language"
LOCATION = "circadian.location"
TEMPLATES = "privacy.pseudonym_templates"
CANARY = "CANARY-{label}"
LAT = 45.4642


class Builds:
    """Counts calls of ``build_catalogue`` made by the settings editor."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return build_catalogue(*args, **kwargs)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def builds(monkeypatch):
    counter = Builds()
    monkeypatch.setattr(settings_module, "build_catalogue", counter)
    return counter


@pytest.fixture
def stack(tmp_path):
    return ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3}},
        audit_path=tmp_path / "audit.jsonl",
    )


def rows(editor):
    return {f["path"]: f for f in editor.fields()}


def with_private(stack):
    stack.set(LOCATION, {"lat": LAT, "lon": 9.19}, source="t", admin=True)
    stack.set(TEMPLATES, {"person": CANARY}, source="t", admin=True)
    return stack


def test_a_second_read_does_not_build_the_catalogue_again(stack, builds):
    editor = SettingsEditor(stack)
    first = editor.fields()
    assert editor.fields() is first
    editor.reveal()
    assert builds.count == 1


def test_a_write_reuses_the_rows_of_the_read_before_it(stack, builds):
    editor = SettingsEditor(stack)
    editor.fields()
    editor.change(HALF_LIFE, 12.0)
    assert builds.count == 1, "the write's _rows() came from the cached build"
    assert rows(editor)[HALF_LIFE]["value"] == 12.0
    assert builds.count == 2


def test_a_write_and_a_reset_give_fresh_rows(stack, builds):
    editor = SettingsEditor(stack)
    before = rows(editor)[HALF_LIFE]
    editor.change(HALF_LIFE, before["value"] + 1.0)
    changed = rows(editor)[HALF_LIFE]
    assert changed["value"] == before["value"] + 1.0 and changed["source"] == "runtime"
    editor.reset(HALF_LIFE)
    back = rows(editor)[HALF_LIFE]
    assert back["value"] == before["value"] and back["source"] == before["source"]


def test_an_added_and_a_removed_entry_give_fresh_rows(stack):
    editor = SettingsEditor(stack)
    assert "extra" not in rows(editor)["llm_routes"]["value"]
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "extra"})
    assert "extra" in rows(editor)["llm_routes"]["value"]
    editor.remove_entry("llm_routes.extra")
    assert "extra" not in rows(editor)["llm_routes"]["value"]


def test_a_hot_reload_of_a_layer_gives_fresh_rows(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("affect:\n  half_life_s: 11.0\n")
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack)
    assert rows(editor)[HALF_LIFE]["value"] == 11.0
    app.write_text("affect:\n  half_life_s: 13.0\n")
    later = os.stat(app).st_mtime + 5
    os.utime(app, (later, later))
    assert stack.poll() is not None
    row = rows(editor)[HALF_LIFE]
    assert row["value"] == 13.0 and row["default"] == 13.0 and row["source"] == "app"


def test_a_runtime_value_whose_ttl_lapses_gives_fresh_rows(tmp_path):
    clock = Clock()
    stack = ConfigStack(clock=clock, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack)
    lower = rows(editor)[HALF_LIFE]
    stack.set(HALF_LIFE, lower["value"] + 2.0, ttl_s=5.0, source="t")
    held = rows(editor)[HALF_LIFE]
    assert held["value"] == lower["value"] + 2.0 and held["source"] == "runtime"
    clock.now += 10.0  # no poll(): the lapse alone must show
    lapsed = rows(editor)[HALF_LIFE]
    assert lapsed["value"] == lower["value"] and lapsed["source"] == lower["source"]


def test_a_language_install_gives_fresh_choices_without_a_rebuild(stack, builds):
    current = stack.effective().conversation.language
    other = next(lang for lang in ("en", "it") if lang != current)
    installed: set[str] = set()
    editor = SettingsEditor(stack, installed=lambda cfg: set(installed))
    assert rows(editor)[LANGUAGE]["choices"] == [current]
    installed.add(other)
    assert rows(editor)[LANGUAGE]["choices"] == sorted({current, other})
    assert editor._rows()[LANGUAGE]["choices"] == sorted({current, other}), "write path too"
    installed.discard(other)
    assert rows(editor)[LANGUAGE]["choices"] == [current]
    assert builds.count == 1


def test_private_values_never_follow_a_reveal_into_a_plain_read(stack):
    editor = SettingsEditor(with_private(stack))
    shown = {f["path"]: f for f in editor.reveal()}
    assert shown[LOCATION]["value"] == {"lat": LAT, "lon": 9.19}
    plain = json.dumps(editor.fields())
    assert CANARY not in plain and str(LAT) not in plain
    assert rows(editor)[LOCATION]["value"] == "***"
    assert {f["path"]: f for f in editor.reveal()}[TEMPLATES]["value"]["person"] == CANARY


def test_the_plain_rows_equal_a_build_without_reveal(stack):
    stack = with_private(stack)
    stack.set("voice.keywords.en.wake_phrases", ["hey you"], source="t")
    editor = SettingsEditor(stack, installed=lambda cfg: {"en", "it"})
    expected = editor_rows_without_cache(stack, ["en", "it"])
    assert editor.fields() == expected


def editor_rows_without_cache(stack, offered):
    """What ``fields()`` returned before the cache: a build without reveal, language patched."""
    out = []
    for row in build_catalogue(stack, _writable, reveal_private=False):
        if row["path"] == LANGUAGE:
            row = {**row, "kind": "enum", "choices": offered}
        out.append(row)
    return out


def test_callers_do_not_change_the_cached_rows(tmp_path, stack):
    editor = SettingsEditor(with_private(stack), store=tmp_path / "s.json")
    editor.fields()
    plain, revealed = editor.fields(), editor._rows()
    snapshot = copy.deepcopy((plain, revealed))
    srv, client = serve(tmp_path, editor, stack)
    try:
        assert client.request("GET", "/api/settings").status == 200
        assert client.request("POST", "/api/settings/reveal", {"pin": PIN}).status == 200
        editor.robot_split(editor.fields())
        editor.restore()
    finally:
        srv.stop()
    assert (plain, revealed) == snapshot


def test_a_get_after_a_write_through_the_api_shows_the_new_value(tmp_path, stack, builds):
    editor = SettingsEditor(stack, store=tmp_path / "s.json")
    srv, client = serve(tmp_path, editor, stack)
    try:
        first = client.request("GET", "/api/settings").json
        again = client.request("GET", "/api/settings").json
        assert first == again and builds.count == 1
        assert put(client, HALF_LIFE, 12.5).status == 200
        assert builds.count == 1
        fields = {f["path"]: f for f in client.request("GET", "/api/settings").json["fields"]}
        assert fields[HALF_LIFE]["value"] == 12.5 and builds.count == 2
    finally:
        srv.stop()


class CountingPlugin(FakePlugin):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    def section(self, stack, manifest):
        self.calls += 1
        return super().section(stack, manifest)


def test_the_robot_plugin_is_asked_on_every_read(tmp_path, stack):
    plugin = CountingPlugin()
    editor = SettingsEditor(stack, manifest=manifest(), plugin=plugin)
    srv, client = serve(tmp_path, editor, stack)
    try:
        first = client.request("GET", "/api/settings").json
        assert HALF_LIFE not in {f["path"] for f in first["fields"]}
        plugin.rows = [descriptor(f"{BASE}.arm.max_deg", "number", 45.0, source="manifest")]
        second = client.request("GET", "/api/settings").json
    finally:
        srv.stop()
    assert plugin.calls == 2
    assert second["robot"]["fields"][0]["value"] == 45.0, "plugin rows are never cached"
    assert HALF_LIFE in {f["path"] for f in second["robot"]["fields"]}


def test_changed_claims_split_the_rows_again(stack):
    plugin = CountingPlugin()
    editor = SettingsEditor(stack, manifest=manifest(), plugin=plugin)
    rest, _ = editor.robot_split(editor.fields())
    assert HALF_LIFE not in {f["path"] for f in rest}
    plugin.claims = ("affect.event_label_hold_s",)
    rest, block = editor.robot_split(editor.fields())
    assert HALF_LIFE in {f["path"] for f in rest}
    assert [f["path"] for f in block["fields"]][1:] == ["affect.event_label_hold_s"]


class VectorPort:
    """A settings port whose catalogue holds an embedding-sized list of numbers."""

    def __init__(self) -> None:
        self.rows = [descriptor("x.vector", "json", [0.5] * 64)]

    def fields(self):
        return self.rows

    def robot_split(self, fields):
        return fields, None


class VectorRobotPort(VectorPort):
    """Clean catalogue rows, but the per-request robot block holds a vector."""

    def __init__(self) -> None:
        self.rows = [descriptor("x.small", "number", 1.0)]

    def robot_split(self, fields):
        row = descriptor(f"{BASE}.vector", "json", [0.5] * 64)
        block = {"id": "fake-bot", "title": "t", "actions": [], "restart_effects": []}
        return fields, {**block, "fields": [row]}


@pytest.mark.parametrize("port", [VectorPort, VectorRobotPort])
def test_a_vector_in_the_settings_payload_is_blocked_on_every_request(tmp_path, stack, port):
    srv, client = serve(tmp_path, port(), stack)
    try:
        for _ in range(2):
            resp = client.request("GET", "/api/settings")
            assert resp.status == 500 and resp.json == {"error": "vector_leak"}
    finally:
        srv.stop()


def test_the_cached_rows_pass_the_vector_guard_once(tmp_path, stack, monkeypatch):
    from roboemot.webui import api as api_module

    real, calls = api_module.holds_vector, [0]

    def counted(value, vetted=()):
        calls[0] += 1
        return real(value, vetted)

    monkeypatch.setattr(api_module, "holds_vector", counted)
    srv, client = serve(tmp_path, SettingsEditor(stack), stack)
    try:
        assert client.request("GET", "/api/settings").status == 200
        first, calls[0] = calls[0], 0
        assert client.request("GET", "/api/settings").status == 200
    finally:
        srv.stop()
    assert first > 1000 and calls[0] < 10, (first, calls[0])
