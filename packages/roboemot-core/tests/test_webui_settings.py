"""Settings API of the web UI: allowlist, PIN confirmation, guardrails, audit (SPEC 11A, 12)."""

from __future__ import annotations

import json
import stat

import pytest
from roboemot.config.catalogue import CONTAINERS
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigError, ConfigStack, GuardrailError
from roboemot.diagnostics import Diagnostics
from roboemot.webui import make_server
from roboemot.webui.setting_clean import HOOKS, hook_for
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.strings import EN, IT
from test_webui import PIN, Client, FakeService

WAKE = "voice.keywords.en.wake_phrases"
LINE = "safety.minor_lines.en.greeting"
BLOCK = "safety.minor_output_blocklist.en"
NAME = "voice.wake_word"
FILE = "webui_settings.json"


class Probe:
    """A stack stand-in whose ``set`` raises the given error."""

    def __init__(self, stack: ConfigStack, exc: Exception) -> None:
        self.stack, self.exc = stack, exc

    def __getattr__(self, name):
        return getattr(self.stack, name)

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
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    yield client
    srv.stop()


def put(client, path, value, pin=PIN, **extra):
    return client.request(
        "POST", "/api/settings", {"path": path, "value": value, "pin": pin, **extra}
    )


def test_list_also_carries_the_whole_catalogue_with_the_allowlist_writable(served, stack):
    data = served.request("GET", "/api/settings").json
    fields = {f["path"]: f for f in data["fields"]}
    assert len(fields) > len(data["settings"])
    writable = {p for p, f in fields.items() if f["writable"]}
    assert writable == {
        p for p, f in fields.items() if f["kind"] not in CONTAINERS and not f["secret"]
    } - {"schema_version"}
    assert {WAKE, NAME, LINE, BLOCK, "persona.prompt", "voice.base.volume_db"} <= writable
    assert fields[WAKE]["kind"] == "string_list" and fields[WAKE]["source"] == "schema"
    assert fields[NAME]["nullable"] and fields[LINE]["admin_only"]


def test_list_has_value_default_schema_and_admin_flag(served, stack):
    rows = {r["path"]: r for r in served.request("GET", "/api/settings").json["settings"]}
    assert set(rows) == {
        f["path"] for f in SettingsEditor(stack).fields() if hook_for(f["path"]) and f["writable"]
    }
    assert WAKE in rows and "persona.prompt" not in rows  # the legacy page lists the hooked ones
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


def test_hooks_are_one_data_table():
    assert {h.pattern for h in HOOKS} == {
        "voice.keywords.*.wake_phrases",
        "voice.wake_word",
        "safety.minor_lines.*.*",
        "safety.minor_output_blocklist.*",
    }
    assert hook_for("safety.minor_lines.en.greeting") and not hook_for("safety.minor_lines.en")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (WAKE, ["ciao robot", "ehi amico"]),
        (NAME, "Neve"),
        (LINE, "Hi there!"),
        ("safety.minor_lines.en.robot_name", "I am called {name}."),
        ("safety.minor_lines.en.robot_unnamed", ""),
        (BLOCK, [*RoboEmotConfig().safety.minor_output_blocklist["en"], "bad word"]),
    ],
)
def test_each_field_changes_with_the_pin_and_is_audited(served, stack, path, value):
    resp = put(served, path, value)
    assert resp.status == 200, resp.body
    assert resp.json == {
        "path": path,
        "changed": True,
        "value": value,
        "persisted": True,
        "layer": "runtime",
        "restart_required": False,
    }
    assert get(stack, path) == value
    entry = stack.audit[-1]
    assert (entry.source, entry.key, entry.new) == ("webui:127.0.0.1", path, value)
    assert entry.old != value
    line = json.loads((stack.audit_path).read_text().splitlines()[-1])
    assert line["source"] == "webui:127.0.0.1" and line["key"] == path and line["new"] == value
    assert path in stack.changes[-1].changed_keys  # hot reload (config.changed) fired
    shown = served.request("GET", "/api/settings").json["settings"]
    if "minor" in path:
        assert {r["path"]: r for r in shown}[path]["value"] == "***"
        shown = served.request("POST", "/api/settings/reveal", {"pin": PIN}).json["fields"]
    assert {r["path"]: r for r in shown}[path]["value"] == value


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
        "llm_routes",
        "safety.minor_lines",
        "voice.keywords",
        "voice.keywords.fr.wake_phrases",
        "safety.minor_lines.fr.greeting",
        "voice.keywords.en.wake_phrases.0",
        "schema_version",
        "no.such.path",
        "persona.prompt.x",
        "",
        5,
        None,
    ],
)
def test_a_path_that_is_not_a_writable_leaf_is_refused_with_a_reason(served, stack, path):
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
        (WAKE, ["hey, robot"], "phrase_invalid"),
        (WAKE, ["it's robot"], "phrase_invalid"),
        (LINE, "one\u2028two", "line_invalid"),
        (LINE, "one\u2029two", "line_invalid"),
        (LINE, "what a fuck", "line_blocked"),
        (NAME, "Cazzo", "name_blocked"),
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
    expected = {
        "source": "webui",
        "admin": True,  # one PIN covers operator and admin
        "must_exist": True,
        "refuse": frozenset({"reflex", "locality", "secret"}),
    }
    assert [c[1] for c in calls] == [expected, expected]


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


# ------------------------------------------------------------------ fix round: guardrails


def test_blocklist_additions_pass_and_removals_are_refused_even_with_the_pin(served, stack):
    current = get(stack, BLOCK)
    assert put(served, BLOCK, [*current, "newword*"]).status == 200
    for smaller in (current[:-1], [*current[1:], "newword*"], ["gun*"]):
        resp = put(served, BLOCK, smaller)
        assert resp.status == 403 and resp.json["error"] == "setting_guardrail"
    assert get(stack, BLOCK) == [*current, "newword*"]


def test_blocklist_swap_for_a_lookalike_counts_as_a_removal(served, stack):
    current = get(stack, BLOCK)
    swapped = ["kіll*" if e == "kill*" else e for e in current]  # Cyrillic i
    assert put(served, BLOCK, swapped).status == 403


def test_blocklist_case_variants_are_not_removals(served, stack):
    current = get(stack, BLOCK)
    assert put(served, BLOCK, [*[e.upper() for e in current], "extra"]).status == 200


def test_stack_refuses_runtime_blocklist_removal_with_admin_true(stack):
    current = get(stack, BLOCK)
    with pytest.raises(GuardrailError):
        stack.set(BLOCK, current[1:], source="t", admin=True)
    with pytest.raises(GuardrailError):
        stack.set(BLOCK, ["only*"], source="t", admin=True)
    stack.set(BLOCK, [*current, "extra*"], source="t", admin=True)


def test_blocklist_removal_in_the_app_layer_with_admin_still_works(stack):
    current = get(stack, BLOCK)
    stack.set(BLOCK, current[1:], source="t", admin=True, layer="app")
    assert get(stack, BLOCK) == current[1:]


def test_a_line_may_not_name_a_blocked_word_through_the_name_placeholder(served, stack):
    stack.set("voice.wake_word", "Neve", source="t", admin=True)
    stack.set(BLOCK, [*get(stack, BLOCK), "neve"], source="t", admin=True)
    resp = put(served, "safety.minor_lines.en.robot_name", "My name is {name}!")
    assert resp.status == 422 and resp.json["error"] == "line_blocked"


def test_wake_word_is_admin_only():
    from roboemot.config.stack import field_meta

    assert field_meta(NAME).get("admin_only") is True
    assert not field_meta(WAKE).get("admin_only")


# ------------------------------------------------------------------ fix round: persistence


def test_edits_are_persisted_private_and_replayed_at_startup(served, tmp_path):
    put(served, WAKE, ["hey bot"])
    put(served, LINE, "Hi!")
    path = tmp_path / FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["overrides"] == {WAKE: ["hey bot"], LINE: "Hi!"}
    other = ConfigStack()
    SettingsEditor(other, store=path).restore()
    assert get(other, WAKE) == ["hey bot"] and get(other, LINE) == "Hi!"
    assert other.audit[-1].source == "webui:restore"


def test_an_unchanged_value_is_not_persisted(served, tmp_path):
    put(served, WAKE, ["hey robot", "hello robot"])
    assert not (tmp_path / FILE).exists()


def test_the_app_yaml_is_never_written(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text("schema_version: 1\n")
    stack = ConfigStack.from_files(app=app)
    SettingsEditor(stack, store=tmp_path / FILE).change(WAKE, ["hey bot"])
    assert app.read_text() == "schema_version: 1\n" and stack.app == {"schema_version": 1}


def write_overrides(path, overrides, mode=0o600):
    path.write_text(json.dumps({"v": 1, "overrides": overrides}))
    path.chmod(mode)


def test_restore_drops_entries_that_no_longer_pass_and_names_only_the_path(tmp_path, caplog):
    path = tmp_path / FILE
    write_overrides(
        path,
        {
            WAKE: ["good bot"],
            "voice.keywords.fr.wake_phrases": ["salut robot"],
            "safety.minor_lines.fr.greeting": "Salut",
            "persona.prompt": "${secret:leak-me}",
            "webui.name_max_chars": 200,
            "no.such.path": 1,
            BLOCK: ["only*"],
            LINE: "x" * 500,
        },
    )
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=path).restore()
    assert get(stack, WAKE) == ["good bot"]
    assert "fr" not in stack.effective().voice.keywords
    assert "fr" not in stack.effective().safety.minor_lines
    assert get(stack, BLOCK) != ["only*"]  # a removal is a guardrail refusal
    assert get(stack, LINE) == "Hello! Nice to see you."
    assert "leak-me" not in caplog.text and "only*" not in caplog.text
    assert stack.effective().webui.name_max_chars == 64  # a loosening is not replayed
    assert "persona.prompt" in caplog.text
    assert set(json.loads(path.read_text())["overrides"]) == {WAKE}


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"v": 2, "overrides": {}}', '{"v": 1, "overrides": []}', "x" * 2_000_000],
)
def test_malformed_file_is_ignored_with_a_warning(tmp_path, caplog, content):
    path = tmp_path / FILE
    path.write_text(content)
    path.chmod(0o600)
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=path).restore()
    assert get(stack, WAKE) == ["hey robot", "hello robot"]
    assert "ignored" in caplog.text


def test_unsafe_modes_and_links_are_ignored(tmp_path):
    path = tmp_path / FILE
    write_overrides(path, {WAKE: ["hey bot"]}, mode=0o644)
    stack = ConfigStack()
    SettingsEditor(stack, store=path).restore()
    assert get(stack, WAKE) == ["hey robot", "hello robot"]
    real = tmp_path / "real.json"
    write_overrides(real, {WAKE: ["hey bot"]})
    link = tmp_path / "link.json"
    link.symlink_to(real)
    SettingsEditor(stack, store=link).restore()
    assert get(stack, WAKE) == ["hey robot", "hello robot"]


def test_no_file_is_fine(tmp_path):
    SettingsEditor(ConfigStack(), store=tmp_path / FILE).restore()


def test_a_language_removed_between_expand_and_set_is_not_recreated():
    stack = ConfigStack(app={"voice": {"keywords": {"zz": {"wake_phrases": ["hey zz"]}}}})
    editor = SettingsEditor(stack)
    path = "voice.keywords.zz.wake_phrases"
    assert path in {f["path"] for f in editor.fields()}
    original = stack.set

    def set_after_removal(key, value, **kw):
        stack.app = {}  # the language disappears after the allowlist was expanded
        return original(key, value, **kw)

    stack.set = set_after_removal
    with pytest.raises(ConfigError):
        editor.change(path, ["hey zzz"])
    assert "zz" not in stack.effective().voice.keywords


PRIVATE_SAMPLES = {
    "circadian.location": {"lat": 45.4642, "lon": 9.19},
    "privacy.multimodal_frames": True,
    "privacy.pseudonym_templates": {"person": "Zed-{label}"},
    "safety.minor_lines.en.greeting": "Hello CANARY little one",
    "safety.minor_output_blocklist.en": [
        *RoboEmotConfig().safety.minor_output_blocklist["en"],
        "canaryword",
    ],
    "register.child.minor_content_policy": True,
    "safety.minor_hold_s": 123.0,
}


@pytest.fixture
def private_served(tmp_path, stack):
    for path, value in PRIVATE_SAMPLES.items():
        stack.set(path, value, source="t", admin=True)
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    yield client
    srv.stop()


def test_private_values_are_hidden_without_the_pin(private_served):
    data = private_served.request("GET", "/api/settings").json
    fields = {f["path"]: f for f in data["fields"]}
    for path in PRIVATE_SAMPLES:
        row = fields[path]
        assert row["private"] and row["admin_only"], path
        assert row["value"] == "***" and row["default"] in ("***", None), path
    for blob in (json.dumps(data), json.dumps(data["settings"])):
        for canary in ("CANARY", "canaryword", "45.4642", "123.0"):
            assert canary not in blob
    for container in ("safety.minor_lines", "safety.minor_output_blocklist", "privacy"):
        row = fields.get(container)
        assert row is None or (row["value"] == [] and row["entry"] is None)
    assert fields["reflexes.rules.startle-loud.suppress_with_minor"]["value"] == "***"
    legacy = {s["path"]: s for s in data["settings"]}
    assert legacy[LINE]["private"] and legacy[LINE]["value"] == "***"
    assert not legacy[WAKE]["private"]


def test_reveal_with_the_pin_returns_only_private_fields_with_real_values(private_served):
    resp = private_served.request("POST", "/api/settings/reveal", {"pin": PIN})
    assert resp.status == 200
    fields = {f["path"]: f for f in resp.json["fields"]}
    assert all(f["private"] for f in fields.values())
    assert fields["circadian.location"]["value"] == {"lat": 45.4642, "lon": 9.19}
    assert fields["safety.minor_lines.en.greeting"]["value"] == "Hello CANARY little one"
    assert "canaryword" in fields["safety.minor_output_blocklist.en"]["value"]
    assert fields["register.child.minor_content_policy"]["value"] is True
    assert "safety.minor_lines" in fields and "en" in fields["safety.minor_lines"]["value"]
    assert WAKE not in fields and "voice.wake_word" not in fields
    assert "no-store" in resp.headers.get("Cache-Control", "no-store")


def test_reveal_keeps_secrets_redacted(tmp_path, stack):
    stack.set(LINE, "${secret:x}", source="t", admin=True)
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        fields = {
            f["path"]: f
            for f in client.request("POST", "/api/settings/reveal", {"pin": PIN}).json["fields"]
        }
        row = fields[LINE]
        assert row["secret"] and row["value"] == "***"
    finally:
        srv.stop()


def test_reveal_needs_the_pin_and_wrong_pins_count_toward_the_lockout(private_served):
    assert private_served.request("POST", "/api/settings/reveal", {}).status == 400
    assert private_served.request("POST", "/api/settings/reveal", {"pin": "111111"}).status == 401
    locked = private_served.request("POST", "/api/settings/reveal", {"pin": PIN})
    assert locked.status == 429 and locked.json["error"] == "locked"
    assert "CANARY" not in json.dumps(locked.json)


def test_writing_a_private_path_needs_the_pin(private_served, stack):
    before = get(stack, LINE)
    resp = private_served.request("POST", "/api/settings", {"path": LINE, "value": "Hi there"})
    assert resp.status == 400 and get(stack, LINE) == before
    wrong = put(private_served, LINE, "Hi there", pin="111111")
    assert wrong.status == 401 and get(stack, LINE) == before
