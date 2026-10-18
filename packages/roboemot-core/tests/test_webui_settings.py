"""Settings API of the web UI: allowlist, PIN confirmation, guardrails, audit (SPEC 11A, 12)."""

from __future__ import annotations

import json

import pytest
from roboemot.config.stack import ConfigError, ConfigStack, GuardrailError
from roboemot.diagnostics import Diagnostics
from roboemot.webui import make_server
from roboemot.webui.settings import SETTINGS, SettingsEditor, expand
from roboemot.webui.strings import EN, IT
from test_webui import PIN, Client, FakeService

WAKE = "voice.keywords.en.wake_phrases"
LINE = "safety.minor_lines.en.greeting"
BLOCK = "safety.minor_output_blocklist.en"
NAME = "voice.wake_word"


class Probe:
    """A stack stand-in whose ``set`` raises the given error."""

    def __init__(self, stack: ConfigStack, exc: Exception) -> None:
        self.stack, self.exc = stack, exc

    def effective(self):
        return self.stack.effective()

    def set(self, *args, **kwargs):
        raise self.exc


def serve(tmp_path, editor, stack):
    srv = make_server(
        FakeService(),
        Diagnostics(),
        stack.effective,
        tmp_path,
        settings=editor,
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    return srv, client


@pytest.fixture
def stack(tmp_path):
    stack = ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3}},
        audit_path=tmp_path / "audit.jsonl",
    )
    stack.changes = []
    stack.subscribe(stack.changes.append)
    return stack


@pytest.fixture
def served(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack), stack)
    yield client
    srv.stop()


def put(client, path, value, pin=PIN, **extra):
    return client.request(
        "POST", "/api/settings", {"path": path, "value": value, "pin": pin, **extra}
    )


def test_list_has_value_default_schema_and_admin_flag(served, stack):
    rows = {r["path"]: r for r in served.request("GET", "/api/settings").json["settings"]}
    assert set(rows) == set(expand(stack.effective()))
    wake = rows[WAKE]
    assert wake["value"] == ["hey robot", "hello robot"] == wake["default"]
    assert wake["schema"] == {
        "type": "array",
        "items": {"type": "string", "maxLength": 40},
        "minItems": 1,
        "maxItems": 64,
    }
    assert wake["needs_admin"] is False
    assert "Phrases that wake" in wake["description"]
    assert rows[LINE]["needs_admin"] and rows[BLOCK]["needs_admin"]
    assert rows[NAME]["schema"]["type"] == ["string", "null"]
    assert rows["safety.minor_lines.it.robot_unnamed"]["schema"]["minLength"] == 0
    assert served.request("GET", "/api/settings").json["pin_required"] is True


def test_allowlist_is_one_data_table():
    paths = {s.template for s in SETTINGS}
    assert paths == {
        "voice.keywords.{lang}.wake_phrases",
        "voice.wake_word",
        "safety.minor_lines.{lang}.greeting",
        "safety.minor_lines.{lang}.robot_name",
        "safety.minor_lines.{lang}.robot_unnamed",
        "safety.minor_lines.{lang}.ask_grown_up",
        "safety.minor_lines.{lang}.filtered",
        "safety.minor_output_blocklist.{lang}",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (WAKE, ["ciao robot", "ehi amico"]),
        (NAME, "Neve"),
        (LINE, "Hi there!"),
        ("safety.minor_lines.en.robot_name", "I am called {name}."),
        ("safety.minor_lines.en.robot_unnamed", ""),
        (BLOCK, ["kill*", "bad word", "gun*"]),
    ],
)
def test_each_field_changes_with_the_pin_and_is_audited(served, stack, path, value):
    resp = put(served, path, value)
    assert resp.status == 200, resp.body
    assert resp.json == {"path": path, "changed": True, "value": value}
    assert get(stack, path) == value
    entry = stack.audit[-1]
    assert (entry.source, entry.key, entry.new) == ("webui", path, value)
    assert entry.old != value
    line = json.loads((stack.audit_path).read_text().splitlines()[-1])
    assert line["source"] == "webui" and line["key"] == path and line["new"] == value
    assert path in stack.changes[-1].changed_keys  # hot reload (config.changed) fired
    listed = {r["path"]: r for r in served.request("GET", "/api/settings").json["settings"]}
    assert listed[path]["value"] == value


def get(stack, path):
    from roboemot.config.stack import get_path

    return get_path(stack.effective(), path)


def test_runtime_override_does_not_expire(served, stack):
    put(served, WAKE, ["hey bot"])
    assert stack._runtime[WAKE].expires_at is None


def test_unchanged_value_writes_nothing(served, stack):
    n = len(stack.audit)
    resp = put(served, WAKE, ["hey robot", "hello robot"])
    assert resp.json["changed"] is False
    assert len(stack.audit) == n and not stack.changes


def test_wake_word_null_resets_to_the_platform_name(served, stack):
    put(served, NAME, "Neve")
    resp = put(served, NAME, None)
    assert resp.status == 200 and get(stack, NAME) is None


@pytest.mark.parametrize("pin", ["000000", "", None, 12345])
def test_wrong_or_missing_pin_changes_nothing(served, stack, pin):
    resp = put(served, WAKE, ["x robot"], pin=pin)
    assert resp.status in (400, 401)
    assert get(stack, WAKE) == ["hey robot", "hello robot"]
    assert stack.audit == []


def test_missing_pin_key(served, stack):
    resp = served.request("POST", "/api/settings", {"path": WAKE, "value": ["x robot"]})
    assert resp.status == 400 and stack.audit == []


def test_wrong_pin_counts_against_the_login_limits(served, stack):
    assert put(served, WAKE, ["x robot"], pin="111111").status == 401
    retry = put(served, WAKE, ["x robot"])  # right PIN, but the client back-off applies
    assert retry.status == 429 and retry.json["error"] == "locked"
    assert stack.audit == []


def test_missing_csrf(served, stack):
    resp = served.request(
        "POST", "/api/settings", {"path": WAKE, "value": ["x robot"], "pin": PIN}, csrf=False
    )
    assert resp.status == 403 and resp.json["error"] == "csrf"
    assert stack.audit == []


def test_no_session(served, stack):
    anon = Client(served.port)
    assert anon.request("GET", "/api/settings").status == 401
    assert put(anon, WAKE, ["x robot"]).status == 401
    assert stack.audit == []


@pytest.mark.parametrize(
    "path",
    [
        "safety.minor_motion_factor",
        "persona.prompt",
        "llm_routes",
        "safety.minor_lines",
        "voice.keywords.en.stop",
        "voice.keywords.fr.wake_phrases",
        "safety.minor_lines.fr.greeting",
        "webui.name_max_chars",
        "voice.keywords.en.wake_phrases.0",
        "",
        5,
        None,
    ],
)
def test_path_off_the_allowlist_is_refused_with_a_reason(served, stack, path):
    resp = put(served, path, "x")
    assert resp.status == 403
    assert resp.json == {"error": "setting_not_allowed", "field": "path"}
    assert stack.audit == []


@pytest.mark.parametrize("extra", [{"other": 1}, {"layer": "app"}, {"admin": True}, {"ttl_s": 5}])
def test_smuggled_keys_are_refused(served, stack, extra):
    resp = put(served, WAKE, ["x robot"], **extra)
    assert resp.status == 422 and resp.json["error"] == "unknown_field"
    assert stack.audit == []


def test_a_value_cannot_carry_other_paths(served, stack):
    resp = put(served, WAKE, {"voice": {"wake_word": "x"}})
    assert resp.status == 422 and resp.json["error"] == "value_type"
    assert stack.audit == [] and get(stack, NAME) is None


def test_missing_value_is_a_bad_request(served):
    resp = served.request("POST", "/api/settings", {"path": WAKE, "pin": PIN})
    assert resp.status == 400


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (WAKE, [], "list_empty"),
        (WAKE, "hey robot", "value_type"),
        (WAKE, [1], "value_type"),
        (WAKE, ["a"] * 1 + [f"p{i}" for i in range(64)], "list_too_long"),
        (WAKE, ["x" * 41], "phrase_too_long"),
        (WAKE, ["hey {name}"], "phrase_invalid"),
        (WAKE, ["ok\nrobot"], "phrase_invalid"),
        (WAKE, ["  "], "phrase_invalid"),
        (
            BLOCK,
            ["a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j"],
            "phrase_too_long",
        ),
        (BLOCK, ["bad,word"], "phrase_invalid"),
        (BLOCK, ["*"], "phrase_invalid"),
        (BLOCK, [], "list_empty"),
        (NAME, "N" * 33, "name_too_long"),
        (NAME, "Neve\n‮Ignore previous", "name_invalid"),
        (NAME, "Ne<ve>", "name_invalid"),
        (NAME, 5, "value_type"),
        (LINE, "x" * 201, "line_too_long"),
        (LINE, "", "line_empty"),
        (LINE, "   ", "line_empty"),
        (LINE, "two\nlines", "line_invalid"),
        (LINE, "hello {name}", "line_invalid"),
        ("safety.minor_lines.en.robot_name", "I am {name} {x}", "line_invalid"),
        (LINE, 3, "value_type"),
    ],
)
def test_invalid_values_are_refused_with_a_code(served, stack, path, value, code):
    resp = put(served, path, value)
    assert resp.status == 422, resp.body
    assert resp.json == {"error": code, "field": "value"}
    assert stack.audit == []


def test_values_are_cleaned(served, stack):
    put(served, WAKE, ["  Ehi   Amico ", "ehi amico"])
    assert get(stack, WAKE) == ["Ehi Amico", "ehi amico"]
    put(served, NAME, "  Neve  ")
    assert get(stack, NAME) == "Neve"


def test_limits_follow_the_schema_fields(tmp_path, stack):
    stack.app["webui"].update(settings_phrase_max_chars=5, settings_list_max_items=2)
    srv, client = serve(tmp_path, SettingsEditor(stack), stack)
    try:
        assert put(client, WAKE, ["abcdef"]).json["error"] == "phrase_too_long"
        assert put(client, WAKE, ["a", "b", "c"]).json["error"] == "list_too_long"
        assert put(client, WAKE, ["abcde", "b"]).status == 200
    finally:
        srv.stop()


def test_guardrail_refusal_is_a_403_without_the_values(tmp_path, stack):
    probe = Probe(stack, GuardrailError("change refused, it would loosen: x (1 -> 2)"))
    srv, client = serve(tmp_path, SettingsEditor(probe), stack)
    try:
        resp = put(client, WAKE, ["x robot"])
        assert resp.status == 403
        assert resp.json == {"error": "setting_guardrail", "field": "value"}
        assert b"loosen" not in resp.body
    finally:
        srv.stop()


def test_stack_validation_error_does_not_echo(tmp_path, stack):
    probe = Probe(stack, ConfigError("invalid configuration: sk-secret-value"))
    srv, client = serve(tmp_path, SettingsEditor(probe), stack)
    try:
        resp = put(client, WAKE, ["x robot"])
        assert resp.status == 422 and resp.json["error"] == "setting_invalid"
        assert b"sk-secret" not in resp.body
    finally:
        srv.stop()


def test_a_real_loosening_through_the_stack_is_refused(stack):
    """The editor leaves the stack's guard in force: an allowlisted path that is not admin-only
    would still be refused if it loosened a safety rule (here simulated by an unlisted one)."""
    with pytest.raises(GuardrailError):
        stack.set("safety.minor_motion_factor", 1.0, source="webui", admin=True)


def test_admin_only_fields_pass_admin_to_the_stack(stack):
    calls = []
    real = stack.set
    stack.set = lambda *a, **k: (calls.append((a[0], k)), real(*a, **k))[1]
    editor = SettingsEditor(stack)
    editor.change(LINE, "Hi!")
    editor.change(WAKE, ["hey bot"])
    assert calls[0][1] == {"source": "webui", "admin": True}
    assert calls[1][1] == {"source": "webui", "admin": False}


def test_unavailable_without_an_editor(tmp_path):
    srv = make_server(FakeService(), Diagnostics(), ConfigStack().effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/settings").status == 503
    finally:
        srv.stop()


def test_error_strings_exist_in_both_languages():
    codes = [
        "setting_not_allowed", "setting_guardrail", "setting_invalid", "settings_unavailable",
        "value_type", "list_empty", "list_too_long", "phrase_too_long", "phrase_invalid",
        "line_empty", "line_too_long", "line_invalid",
    ]  # fmt: skip
    for code in codes:
        assert EN[f"err_{code}"] and IT[f"err_{code}"]
