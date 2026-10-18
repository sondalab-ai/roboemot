"""Generic settings write path: kinds, collections, layers, reset, batch restore (SPEC 11A, 12)."""

from __future__ import annotations

import copy
import json

import pytest
import yaml
from roboemot.config.stack import (
    ConfigError,
    ConfigStack,
    GuardrailError,
    Violation,
    get_path,
    keyed_ref,
)
from roboemot.webui.settings import SettingNotAllowed, SettingsEditor, loosening_only, okey
from roboemot.webui.validation import ValidationError
from test_webui import PIN
from test_webui_settings import FILE, serve

ROUTE = {
    "name": "lan",
    "base_url": "https://api.example.com/v1",
    "model": "m",
    "first_token_timeout_s": 5,
}
RULE_A = {"event": "zz.first", "valence": 0.1}
RULE_B = {"event": "zz.second", "valence": 0.2}


def appraisal(rules):
    return {"rules": rules, "extra_events": ["zz.first", "zz.second", "zz.third", "custom.thing"]}


def make_stack(tmp_path, **app):
    cfg = copy.deepcopy({"webui": {"bind": "127.0.0.1", "port": 0}, **app})
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def stack(tmp_path):
    return make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))


@pytest.fixture
def editor(tmp_path, stack):
    return SettingsEditor(stack, store=tmp_path / FILE)


def rule_index(stack, event):
    rules = get_path(stack.effective(), "appraisal.rules")
    return next(n for n, r in enumerate(rules) if r.event == event)


def rule(stack, event):
    return get_path(stack.effective(), "appraisal.rules")[rule_index(stack, event)]


# ------------------------------------------------------------------ scalar kinds


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("affect.update_hz", 25.5),  # number
        ("affect.update_hz", 30),  # int accepted for a number
        ("register.child.max_reply_sentences", 1),  # int
        ("register.teen.approach_allowed", False),  # bool
        ("register.child.vocabulary", "normal"),  # enum
        ("personality", "calm"),  # text
        ("circadian.quiet_hours", ["23:00", "06:30"]),  # tuple
        ("vision.animal_categories", ["cat", "dog"]),  # string_list
        ("world.autoenroll.kinds", ["animal"]),  # enum_list
        ("persona.trait_effects.energy.motion_amplitude_x", [[0.0, 0.7], [1.0, 1.1]]),  # tuple_list
        ("circadian.location", {"lat": 10.5, "lon": 20.25}),  # number_dict (nullable, private)
    ],
)
def test_each_kind_is_written(editor, stack, path, value):
    result = editor.change(path, value)
    assert result["changed"] and result["layer"] == "runtime"
    now = get_path(stack.effective(), path)
    assert json.loads(json.dumps(now, default=lambda o: o.model_dump(mode="json"))) == value


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        ("affect.update_hz", 0.5, "value_range"),
        ("affect.update_hz", 201, "value_range"),
        ("affect.update_hz", float("nan"), "value_type"),
        ("affect.update_hz", float("inf"), "value_type"),
        ("affect.update_hz", True, "value_type"),
        ("affect.update_hz", "20", "value_type"),
        ("affect.update_hz", None, "value_type"),
        ("register.child.max_reply_sentences", 2.5, "value_type"),
        ("register.child.max_reply_sentences", 21, "value_range"),
        ("register.teen.approach_allowed", 1, "value_type"),
        ("register.child.vocabulary", "loud", "value_choice"),
        ("register.child.vocabulary", True, "value_choice"),
        ("personality", 5, "value_type"),
        ("personality", "a\x00b", "text_invalid"),
        ("personality", "x" * 4001, "text_too_long"),
        ("circadian.quiet_hours", ["23:00"], "value_type"),
        ("circadian.quiet_hours", "23:00", "value_type"),
        ("vision.animal_categories", "cat", "value_type"),
        ("vision.animal_categories", ["x" * 201], "phrase_too_long"),
        ("vision.animal_categories", [f"e{i}" for i in range(65)], "list_too_long"),
        ("world.autoenroll.kinds", ["person"], "value_choice"),
        ("vision.object_display_names", {"cup": 5}, "value_type"),
        ("circadian.location", {"lat": "x"}, "value_type"),
    ],
)
def test_limits_and_types_are_refused_with_a_code(editor, stack, path, value, code):
    with pytest.raises(ValidationError) as caught:
        editor.change(path, value)
    assert caught.value.code == code
    assert stack.audit == []


def test_a_dict_can_gain_a_key_but_not_lose_one(editor, stack):
    path = "vision.object_display_names"
    current = dict(get_path(stack.effective(), path))
    assert editor.change(path, {**current, "cup": "mug"})["changed"]
    assert get_path(stack.effective(), path)["cup"] == "mug"
    smaller = {k: v for k, v in current.items() if k != "teddy_bear"}
    with pytest.raises(ValidationError) as caught:
        editor.change(path, smaller)
    assert caught.value.code == "value_merged"
    assert "teddy_bear" in get_path(stack.effective(), path)
    assert get_path(stack.effective(), path)["cup"] == "mug"  # the earlier edit is untouched


def test_the_final_check_is_pydantic_through_the_stack(editor, stack):
    with pytest.raises(ConfigError):
        editor.change("circadian.quiet_hours", ["25:99", "07:00"])
    assert get_path(stack.effective(), "circadian.quiet_hours") != ("25:99", "07:00")


def test_restart_required_comes_from_the_field_meta(editor):
    assert editor.change("reflexes.audio.poll_hz", 40.0)["restart_required"] is True
    assert editor.change("affect.update_hz", 25.0)["restart_required"] is False


def test_identity_leaves_of_a_keyed_list_cannot_be_renamed(editor):
    for path in ("llm_routes.lan.name", f"appraisal.rules.{0}.event"):
        with pytest.raises(SettingNotAllowed):
            editor.change(path, "other")


# ------------------------------------------------------------------ private fields


def test_private_paths_are_writable_and_shown_only_with_the_pin(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        path = "safety.minor_hold_s"
        before = client.request("POST", "/api/settings", {"path": path, "value": 99.0})
        assert before.status == 400  # no PIN
        wrong = client.request(
            "POST", "/api/settings", {"path": path, "value": 99.0, "pin": "111111"}
        )
        assert wrong.status == 401
        assert get_path(stack.effective(), path) != 99.0
    finally:
        srv.stop()


def test_a_private_value_can_be_written_with_the_pin(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        resp = client.request(
            "POST",
            "/api/settings",
            {"path": "circadian.location", "value": {"lat": 1.5, "lon": 2.5}, "pin": PIN},
        )
        assert resp.status == 200 and resp.json["layer"] == "runtime"
        assert get_path(stack.effective(), "circadian.location")["lat"] == 1.5
    finally:
        srv.stop()


# ------------------------------------------------------------------ secrets


def test_a_secret_field_is_refused_whatever_the_value(editor, stack):
    for path in ("llm_routes.lan.api_key",):
        with pytest.raises(SettingNotAllowed):
            editor.change(path, "plain")
    assert stack.audit == []


@pytest.mark.parametrize(
    "value",
    ["${secret:k}", "https://u:p@host/v1", "https://h/v1?api_key=abc", ["x", "${secret:k}"]],
)
def test_a_secret_shaped_value_is_refused_on_every_write_path(editor, stack, value):
    with pytest.raises(ValidationError):
        editor.change("personality", value)
    with pytest.raises(ValidationError):
        editor.change("vision.animal_categories", value if isinstance(value, list) else [value])
    with pytest.raises(ValidationError):
        editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2", "model": value})
    with pytest.raises(ValidationError):
        editor.add_entry("voice.moods", "tender", {"instructions": value})
    assert stack.audit == []


def test_a_secret_nested_in_an_added_entry_is_refused(editor, stack):
    nested = {**ROUTE, "name": "n2", "api_key": "${secret:k}"}
    with pytest.raises(ValidationError):
        editor.add_entry("llm_routes", None, nested)
    literal = {**ROUTE, "name": "n3", "api_key": "sk-plain-key"}
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("llm_routes", None, literal)
    assert caught.value.code == "entry_invalid"
    assert [r.name for r in stack.effective().llm_routes] == ["lan"]


# ------------------------------------------------------------------ dict of model


def test_dict_of_model_entry_is_written_by_key(editor, stack):
    editor.change("voice.moods.joy.rate", 0.3)
    assert stack.effective().voice.moods["joy"].rate == 0.3


def test_add_and_remove_a_dict_entry(editor, stack, tmp_path):
    out = editor.add_entry("voice.moods", "tender", {"rate": -0.1, "instructions": "soft"})
    assert out["layer"] == "runtime" and out["persisted"]
    assert stack.effective().voice.moods["tender"].rate == -0.1
    saved = json.loads((tmp_path / FILE).read_text())["overrides"]
    assert saved["voice.moods.tender"] == {"rate": -0.1, "instructions": "soft"}
    editor.change("voice.moods.tender.rate", -0.2)  # joins the added entry in the file
    assert (
        json.loads((tmp_path / FILE).read_text())["overrides"]["voice.moods.tender"]["rate"] == -0.2
    )
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("voice.moods", "tender", {})
    assert caught.value.code == "entry_exists"
    editor.remove_entry("voice.moods.tender")
    assert "tender" not in stack.effective().voice.moods
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {}


def test_a_key_must_be_plain(editor):
    for key in ("a.b", "", "x y", None, 5, "k" * 100):
        with pytest.raises(ValidationError) as caught:
            editor.add_entry("voice.moods", key, {})
        assert caught.value.code == "key_invalid"
    with pytest.raises(ValidationError):
        editor.add_entry("voice.moods", "${secret:k}", {})


def test_an_unknown_field_in_an_entry_is_refused(editor):
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("voice.moods", "tender", {"loudness": 3})
    assert caught.value.code == "entry_invalid"


def test_no_new_language_can_be_added(editor, stack):
    for path, key in (
        ("voice.keywords", "fr"),
        ("safety.minor_lines", "fr"),
        ("conversation.confirm_yes", "fr"),
    ):
        with pytest.raises(SettingNotAllowed):
            editor.add_entry(path, key, {})
    for path in (
        "voice.keywords.fr.wake_phrases",
        "safety.minor_lines.fr.greeting",
        "safety.minor_output_blocklist.fr",
        "conversation.confirm_yes.fr",
    ):
        with pytest.raises(SettingNotAllowed):
            editor.change(path, ["x"])
    assert "fr" not in stack.effective().voice.keywords


def test_a_lower_layer_dict_entry_cannot_be_removed(editor, stack):
    with pytest.raises(ValidationError) as caught:
        editor.remove_entry("voice.moods.joy")
    assert caught.value.code == "entry_in_lower_layer"
    assert "joy" in stack.effective().voice.moods


def test_removing_an_unknown_entry_is_not_allowed(editor):
    for path in ("voice.moods.nope", "voice.moods", "personality", 5, None):
        with pytest.raises(SettingNotAllowed):
            editor.remove_entry(path)


# ------------------------------------------------------------------ keyed lists


def test_keyed_list_by_name_is_written(editor, stack):
    editor.change("llm_routes.lan.model", "m2")
    assert stack.effective().llm_routes[0].model == "m2"


def test_index_addressed_entry_stays_the_same_entry_when_the_layer_reorders(
    editor, stack, tmp_path
):
    at = rule_index(stack, "zz.first")
    assert at != rule_index(stack, "zz.second") and keyed_ref(
        stack.merged_tree(), f"appraisal.rules.{at}.valence"
    )
    editor.change(f"appraisal.rules.{at}.valence", 0.9)
    assert rule(stack, "zz.first").valence == 0.9 and rule(stack, "zz.second").valence == 0.2
    saved = json.loads((tmp_path / FILE).read_text())["overrides"]
    assert list(saved) == ['appraisal.rules[{"event":"zz.first"}].valence']
    stack.app["appraisal"]["rules"].reverse()  # a hot reload that reorders the app layer
    assert rule(stack, "zz.first").valence == 0.9 and rule(stack, "zz.second").valence == 0.2
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_A, RULE_B]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()  # a restart after the reorder
    assert rule(fresh, "zz.first").valence == 0.9 and rule(fresh, "zz.second").valence == 0.2


def test_catalogue_source_follows_the_entry(editor, stack):
    at = rule_index(stack, "zz.first")
    editor.change(f"appraisal.rules.{at}.valence", 0.9)
    stack.app["appraisal"]["rules"].reverse()
    rows = {f["path"]: f for f in editor.fields()}
    moved = rule_index(stack, "zz.first")
    assert rows[f"appraisal.rules.{moved}.valence"]["source"] == "runtime"
    assert rows[f"appraisal.rules.{rule_index(stack, 'zz.second')}.valence"]["source"] == "app"


def test_app_layer_write_finds_the_entry_by_identity(stack):
    at = rule_index(stack, "zz.first")  # effective index differs from the app layer's
    assert stack.app["appraisal"]["rules"][1]["event"] == "zz.first" and at != 1
    stack.set(f"appraisal.rules.{at}.valence", 0.7, layer="app", admin=True)
    assert stack.app["appraisal"]["rules"][1]["valence"] == 0.7
    assert stack.app["appraisal"]["rules"][0]["valence"] == 0.2
    assert len(stack.app["appraisal"]["rules"]) == 2


def test_app_layer_write_into_a_list_the_layer_lacks_adds_an_identity_entry(tmp_path):
    stack = make_stack(tmp_path)
    stack.set("appraisal.rules.0.valence", 0.5, layer="app", admin=True)
    assert stack.app["appraisal"]["rules"] == [{"event": "person.known", "valence": 0.5}]


def test_add_and_remove_a_keyed_list_entry(editor, stack, tmp_path):
    out = editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2"})
    assert out["layer"] == "runtime"
    assert [r.name for r in stack.effective().llm_routes] == ["lan", "n2"]
    editor.change("llm_routes.n2.model", "m9")
    saved = json.loads((tmp_path / FILE).read_text())["overrides"]
    assert (
        list(saved) == ['llm_routes[{"name":"n2"}]']
        and saved['llm_routes[{"name":"n2"}]']["model"] == "m9"
    )
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2"})
    assert caught.value.code == "entry_exists"
    editor.remove_entry("llm_routes.n2")
    assert [r.name for r in stack.effective().llm_routes] == ["lan"]
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {}


def test_an_added_rule_with_dotted_event_round_trips_through_restore(editor, stack, tmp_path):
    editor.add_entry("appraisal.rules", None, {"event": "custom.thing", "valence": 0.4})
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert rule(fresh, "custom.thing").valence == 0.4
    editor.remove_entry(f"appraisal.rules.{rule_index(stack, 'custom.thing')}")
    assert all(r.event != "custom.thing" for r in stack.effective().appraisal.rules)


def test_an_entry_of_a_lower_layer_cannot_be_removed(editor, stack):
    with pytest.raises(ValidationError) as caught:
        editor.remove_entry("llm_routes.lan")
    assert caught.value.code == "entry_in_lower_layer"
    with pytest.raises(ValidationError) as caught:
        editor.remove_entry(f"appraisal.rules.{rule_index(stack, 'zz.first')}")
    assert caught.value.code == "entry_in_lower_layer"
    assert len(stack.effective().llm_routes) == 1


def test_a_new_reflex_rule_is_refused_whole(editor):
    rules = editor.stack.effective().reflexes.rules
    entry = rules[0].model_dump(mode="json") | {"name": "extra"}
    with pytest.raises(GuardrailError):
        editor.add_entry("reflexes.rules", None, entry)


# ------------------------------------------------------------------ layers


def test_a_safety_tighten_lands_in_runtime(editor, stack):
    out = editor.change("webui.name_max_chars", 32)
    assert out["layer"] == "runtime" and stack.effective().webui.name_max_chars == 32


def test_a_safety_loosen_lands_in_the_app_yaml_with_an_audit_entry(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text(yaml.safe_dump({"webui": {"bind": "127.0.0.1", "port": 0}}))
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("webui.name_max_chars", 32)  # a runtime value that would shadow the app one
    out = editor.change("webui.name_max_chars", 100)
    assert out["layer"] == "app" and out["restart_required"] is False
    assert stack.effective().webui.name_max_chars == 100
    assert yaml.safe_load(app.read_text())["webui"]["name_max_chars"] == 100
    entry = next(a for a in stack.audit if a.key == "webui.name_max_chars" and a.new == 100)
    assert entry.source == "webui" and entry.old == 32
    assert "webui.name_max_chars" not in stack._runtime  # nothing shadows the app value
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {}
    assert (
        "audit" in (tmp_path / "audit.jsonl").name
        and "100" in (tmp_path / "audit.jsonl").read_text()
    )


def test_loosening_with_another_violation_is_refused_not_retried(editor, stack):
    path = "reflexes.rules.withdraw-close.amplitude_deg"
    new = get_path(stack.effective(), path) + 5
    with pytest.raises(GuardrailError) as caught:
        editor.change(path, new)
    assert not loosening_only(caught.value)
    assert {v.check for v in caught.value.violations} >= {"reflex"}
    assert "amplitude_deg" not in json.dumps(stack.app)  # the app layer was not touched
    assert get_path(stack.effective(), path) == new - 5


def test_a_non_protective_reflex_field_loosening_goes_to_the_app_layer_by_name(editor, stack):
    path = "reflexes.rules.startle-loud.amplitude_deg"
    new = get_path(stack.effective(), path) + 1
    assert editor.change(path, new)["layer"] == "app"
    rules = {r["name"]: r for r in stack.app["reflexes"]["rules"]}
    assert rules["startle-loud"]["amplitude_deg"] == new and len(rules) == 3


def test_locality_stays_refused_on_both_layers(editor, stack):
    with pytest.raises(GuardrailError) as caught:
        editor.change("llm_routes.lan.base_url", "http://nas.local:8080/v1")
    assert not loosening_only(caught.value)
    assert {v.check for v in caught.value.violations} == {"locality"}
    assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]
    assert "llm_routes" in stack.app and "nas.local" not in json.dumps(stack.app)


def test_a_secret_route_cannot_be_pointed_at_another_host(tmp_path):
    route = {**ROUTE, "api_key": "${secret:k}"}
    stack = make_stack(tmp_path, llm_routes=[route])
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    with pytest.raises(GuardrailError) as caught:
        editor.change("llm_routes.lan.base_url", "https://evil.example.com/v1")
    assert {v.check for v in caught.value.violations} == {"secret"}
    assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]


def test_only_field_direction_safety_violations_are_retried():
    field = Violation("safety", "a", 1, 2)
    assert loosening_only(GuardrailError("x", [field]))
    assert not loosening_only(
        GuardrailError("x", [field, Violation("safety", "b", 1, 2, "reflex")])
    )
    assert not loosening_only(GuardrailError("x", [Violation("privacy", "p", False, True)]))
    assert not loosening_only(GuardrailError("x", [field, Violation("admin", "a", 1, 2)]))
    assert not loosening_only(GuardrailError("x"))


def test_the_layer_is_decided_by_the_server(editor):
    assert editor.change("webui.name_max_chars", 32)["layer"] == "runtime"


# ------------------------------------------------------------------ reset


def test_reset_clears_the_runtime_value_and_the_file_entry(editor, stack, tmp_path):
    editor.change("affect.update_hz", 33.0)
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {"affect.update_hz": 33.0}
    out = editor.reset("affect.update_hz")
    assert out["changed"] and out["layer"] == "runtime" and out["value"] == 20.0
    assert stack.effective().affect.update_hz == 20.0
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {}
    assert editor.reset("affect.update_hz")["changed"] is False


def test_reset_of_an_index_addressed_entry_uses_identity(editor, stack):
    at = rule_index(stack, "zz.first")
    editor.change(f"appraisal.rules.{at}.valence", 0.9)
    stack.app["appraisal"]["rules"].reverse()
    out = editor.reset(f"appraisal.rules.{rule_index(stack, 'zz.first')}.valence")
    assert out["changed"] and rule(stack, "zz.first").valence == 0.1
    assert stack._runtime == {}


def test_reset_leaves_the_app_value_in_place(tmp_path):
    stack = make_stack(tmp_path)
    stack.app["affect"] = {"update_hz": 40.0}
    editor = SettingsEditor(stack)
    editor.change("affect.update_hz", 50.0)
    assert editor.reset("affect.update_hz")["value"] == 40.0


def test_reset_only_takes_writable_leaves(editor):
    for path in ("llm_routes", "llm_routes.lan.api_key", "schema_version", "nope", 5):
        with pytest.raises(SettingNotAllowed):
            editor.reset(path)


def test_a_field_of_an_added_entry_cannot_be_reset_alone(editor):
    editor.add_entry("voice.moods", "tender", {})
    with pytest.raises(ValidationError) as caught:
        editor.reset("voice.moods.tender.rate")
    assert caught.value.code == "not_resettable"


def test_clear_that_would_invalidate_the_config_is_rolled_back(tmp_path):
    stack = make_stack(tmp_path)
    stack.set("voice.moods.tender", {"rate": 0.1}, source="t")
    saved = dict(stack._runtime)
    real = stack.effective
    calls = {"n": 0}

    def flaky(person=None):
        calls["n"] += 1
        if calls["n"] == 2:  # the first call is the validity probe
            raise ConfigError("boom")
        return real(person)

    stack.effective = flaky
    with pytest.raises(ConfigError):
        stack.clear("voice.moods.tender")
    stack.effective = real
    assert set(stack._runtime) == set(saved)


# ------------------------------------------------------------------ batch restore


def test_restore_is_one_batch_so_cross_field_rules_do_not_depend_on_order(tmp_path):
    """A line that names the robot is blocked by a blocklist entry stored later in the file; the
    batch cleans the line against the whole file, so it is refused whatever the order."""
    first = {
        "safety.minor_output_blocklist.en": [
            *ConfigStack().effective().safety.minor_output_blocklist["en"],
            "zork",
        ],
        "safety.minor_lines.en.greeting": "Hello zork",
    }
    for order in (first, dict(reversed(list(first.items())))):
        path = tmp_path / FILE
        path.write_text(json.dumps({"v": 1, "overrides": order}))
        path.chmod(0o600)
        stack = ConfigStack()
        SettingsEditor(stack, store=path).restore()
        assert "zork" in stack.effective().safety.minor_output_blocklist["en"]
        assert stack.effective().safety.minor_lines["en"].greeting == "Hello! Nice to see you."
        assert set(json.loads(path.read_text())["overrides"]) == {
            "safety.minor_output_blocklist.en"
        }


def test_restore_applies_the_whole_file_with_one_notification(tmp_path):
    path = tmp_path / FILE
    path.write_text(
        json.dumps({"v": 1, "overrides": {"affect.update_hz": 33.0, "personality": "calm"}})
    )
    path.chmod(0o600)
    stack = ConfigStack()
    events = []
    stack.subscribe(events.append)
    SettingsEditor(stack, store=path).restore()
    assert len(events) == 1 and set(events[0].changed_keys) == {"affect.update_hz", "personality"}


def test_restore_skips_a_bad_entry_and_keeps_the_rest(tmp_path, caplog):
    path = tmp_path / FILE
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "overrides": {
                    "affect.update_hz": 33.0,
                    "register.child.max_reply_sentences": 999,
                    "circadian.quiet_hours": ["25:99", "07:00"],
                    "llm_routes.gone.model": "x",
                    'appraisal.rules[{"event":"vanished"}].valence': 0.5,
                },
            }
        )
    )
    path.chmod(0o600)
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=path).restore()
    assert stack.effective().affect.update_hz == 33.0
    assert set(json.loads(path.read_text())["overrides"]) == {"affect.update_hz"}
    assert "vanished" not in caplog.text and "25:99" not in caplog.text
    assert "circadian.quiet_hours" in caplog.text


def test_restore_never_applies_a_secret_or_a_reflex_loosening(tmp_path):
    path = tmp_path / FILE
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "overrides": {
                    "personality": "${secret:k}",
                    "reflexes.rules.withdraw-close.amplitude_deg": 80.0,
                    "affect.update_hz": 33.0,
                },
            }
        )
    )
    path.chmod(0o600)
    stack = ConfigStack()
    SettingsEditor(stack, store=path).restore()
    assert stack.effective().personality == "default"
    assert stack.effective().reflexes.rules[1].amplitude_deg != 80.0
    assert set(json.loads(path.read_text())["overrides"]) == {"affect.update_hz"}


def test_okey_round_trip(stack):
    tree = stack.merged_tree()
    at = rule_index(stack, "zz.first")
    key = okey(tree, f"appraisal.rules.{at}.valence")
    assert key == 'appraisal.rules[{"event":"zz.first"}].valence'
    assert okey(tree, "llm_routes.lan.model") == 'llm_routes[{"name":"lan"}].model'
    assert okey(tree, "affect.update_hz") == "affect.update_hz"


# ------------------------------------------------------------------ HTTP endpoints


@pytest.fixture
def web(tmp_path, stack):
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    yield client
    srv.stop()


def post(client, url, pin=PIN, **body):
    return client.request("POST", url, {**body, "pin": pin} if pin else body)


def test_endpoints_write_entries_and_reset_with_the_pin(web, stack):
    added = post(web, "/api/settings/entry", path="voice.moods", key="tender", value={"rate": -0.1})
    assert added.status == 200
    assert added.json == {
        "path": "voice.moods",
        "changed": True,
        "layer": "runtime",
        "restart_required": False,
        "persisted": True,
    }
    assert stack.effective().voice.moods["tender"].rate == -0.1
    assert post(web, "/api/settings/entry/remove", path="voice.moods.tender").status == 200
    assert "tender" not in stack.effective().voice.moods
    assert (
        post(web, "/api/settings", path="affect.update_hz", value=33.0).json["layer"] == "runtime"
    )
    reset = post(web, "/api/settings/reset", path="affect.update_hz")
    assert reset.status == 200 and reset.json["changed"] and reset.json["value"] == 20.0


def test_restart_flag_is_in_the_response(web):
    out = post(web, "/api/settings", path="reflexes.audio.poll_hz", value=40.0)
    assert out.json["restart_required"] is True


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("/api/settings/reset", {"path": "affect.update_hz"}),
        ("/api/settings/entry", {"path": "voice.moods", "key": "t", "value": {}}),
        ("/api/settings/entry/remove", {"path": "voice.moods.joy"}),
    ],
)
def test_write_endpoints_need_the_pin(web, stack, url, body):
    n = len(stack.audit)
    assert post(web, url, pin=None, **body).status == 400
    assert post(web, url, pin="111111", **body).status == 401
    assert len(stack.audit) == n


def test_endpoint_errors_are_bare_codes(web, stack):
    gone = post(web, "/api/settings/entry/remove", path="voice.moods.joy")
    assert gone.status == 422 and gone.json == {"error": "entry_in_lower_layer", "field": "path"}
    odd = post(web, "/api/settings/entry", path="voice.keywords", key="fr", value={})
    assert odd.status == 403 and odd.json["error"] == "setting_not_allowed"
    smuggled = post(web, "/api/settings/reset", path="affect.update_hz", layer="app")
    assert smuggled.status == 422 and smuggled.json["error"] == "unknown_field"
    assert post(web, "/api/settings/reset").status == 400


def test_new_error_codes_exist_in_both_languages():
    from roboemot.webui.strings import EN, IT

    for code in (
        "value_range", "value_choice", "value_merged", "text_invalid", "text_too_long",
        "key_invalid", "entry_exists", "entry_invalid", "entry_in_lower_layer", "not_resettable",
    ):  # fmt: skip
        assert EN[f"err_{code}"] and IT[f"err_{code}"]


def test_a_loosen_over_http_names_the_app_layer(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text(yaml.safe_dump({"webui": {"bind": "127.0.0.1", "port": 0}}))
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        out = post(client, "/api/settings", path="webui.name_max_chars", value=100)
        assert out.status == 200 and out.json["layer"] == "app"
        assert yaml.safe_load(app.read_text())["webui"]["name_max_chars"] == 100
    finally:
        srv.stop()


# ------------------------------------------------------------------ fix round: security


def lan_stack(tmp_path, **route):
    """One cloud route with locality auto, so a LAN host would turn it local."""
    return make_stack(tmp_path, llm_routes=[{**ROUTE, **route}])


def test_reset_cannot_reopen_the_locality_hole(tmp_path):
    """cloud, then LAN host, then reset locality: the reset would make the route local with a
    LAN base_url nobody approved. It is refused like the write would be."""
    stack = lan_stack(tmp_path)
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("llm_routes.lan.locality", "cloud")
    editor.change("llm_routes.lan.base_url", "http://nas.local:8080/v1")
    with pytest.raises(GuardrailError) as caught:
        editor.reset("llm_routes.lan.locality")
    assert {v.check for v in caught.value.violations} == {"locality"}
    assert stack.effective().llm_routes[0].locality == "cloud"
    with pytest.raises(GuardrailError):
        stack.clear("llm_routes.lan.locality")
    assert stack.effective().llm_routes[0].locality == "cloud"
    fresh = lan_stack(tmp_path)
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert fresh.effective().llm_routes[0].locality == "cloud"


def test_a_guarded_clear_still_removes_a_harmless_override(editor, stack):
    editor.change("webui.name_max_chars", 32)
    assert editor.reset("webui.name_max_chars")["changed"]
    assert stack.effective().webui.name_max_chars == 64


def test_audit_never_holds_a_key_or_a_reference_from_a_whole_list_write(tmp_path):
    plain_key, ref = "sk-PLAIN-CANARY-1", "${secret:ref-canary}"
    routes = [
        {**ROUTE, "name": "a", "api_key": plain_key},
        {**ROUTE, "name": "b", "api_key": ref},
    ]
    stack = make_stack(tmp_path, llm_routes=routes)
    stack.set(
        "llm_routes", [{**routes[0], "model": "m2"}, {**routes[1], "model": "m3"}], admin=True
    )
    stack.set_many({"llm_routes": [{**routes[0], "model": "m4"}]}, admin=True)
    stack.set("llm_routes", [{**routes[0], "model": "m5"}], admin=True, layer="app")
    blob = json.dumps([a.__dict__ for a in stack.audit], default=str)
    blob += (tmp_path / "audit.jsonl").read_text()
    for canary in ("sk-PLAIN-CANARY-1", "ref-canary", "${secret"):
        assert canary not in blob


def test_restore_audit_holds_no_key(tmp_path):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": "sk-PLAIN-CANARY-2"}])
    path = tmp_path / FILE
    path.write_text(json.dumps({"v": 1, "overrides": {"llm_routes.lan.model": "m9"}}))
    path.chmod(0o600)
    SettingsEditor(stack, store=path).restore()
    assert stack.effective().llm_routes[0].model == "m9"
    assert "sk-PLAIN-CANARY-2" not in (tmp_path / "audit.jsonl").read_text()
    assert "sk-PLAIN-CANARY-2" not in json.dumps([a.__dict__ for a in stack.audit], default=str)


def test_the_audit_file_is_private(tmp_path):
    import stat

    stack = make_stack(tmp_path)
    stack.set("affect.update_hz", 30.0)
    assert stat.S_IMODE((tmp_path / "audit.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize("api_key", ["sk-literal-key", "${secret:k}"])
@pytest.mark.parametrize("url", ["https://evil.example.com/v1", "http://api.example.com/v1"])
def test_a_keyed_route_cannot_change_endpoint_from_the_web(tmp_path, api_key, url):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": api_key}])
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    with pytest.raises(GuardrailError) as caught:  # a different host, or an http downgrade
        editor.change("llm_routes.lan.base_url", url)
    assert {v.check for v in caught.value.violations} == {"secret"}
    assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]
    assert api_key not in str(caught.value)
    editor.change("llm_routes.lan.base_url", "https://api.example.com/v2")  # same endpoint
    assert stack.effective().llm_routes[0].base_url.endswith("/v2")


def test_a_route_without_a_key_may_move(tmp_path):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": "none"}])
    SettingsEditor(stack).change("llm_routes.lan.base_url", "https://other.example.com/v1")
    assert stack.effective().llm_routes[0].base_url.startswith("https://other")


def test_violation_values_stay_exact_and_only_the_text_is_redacted():
    from roboemot.config.stack import locality_violations, secret_violations

    def cfg(**route):
        return ConfigStack(app={"llm_routes": [{**ROUTE, **route}]}).effective()

    loc = locality_violations(cfg(), cfg(base_url="http://nas.local:1/v1"))[0]
    other_path = locality_violations(cfg(), cfg(base_url="http://nas.local:1/v2"))[0]
    other_scheme = locality_violations(cfg(), cfg(base_url="https://nas.local:1/v1"))[0]
    assert loc.candidate == "local http://nas.local:1/v1 (auto)"
    assert len({loc.candidate, other_path.candidate, other_scheme.candidate}) == 3
    one = secret_violations(cfg(api_key="${secret:k1}"), cfg(api_key="${secret:k2}"))[0]
    two = secret_violations(cfg(api_key="${secret:k1}"), cfg(api_key="${secret:k3}"))[0]
    assert "${secret:k2}" in one.candidate and one.candidate != two.candidate
    assert "k2" not in str(one) and "k1" not in str(one)


def test_an_admin_approval_matches_the_exact_value(tmp_path):
    stack = lan_stack(tmp_path)
    stack.set("llm_routes.lan.base_url", "http://nas.local:1/v1", admin=True)
    with pytest.raises(GuardrailError):  # another path under the same host is not approved
        stack.set("llm_routes.lan.base_url", "http://nas.local:1/v2")
    stack.set("llm_routes.lan.base_url", "http://nas.local:1/v1")  # the approved value again


def test_two_added_entries_both_stay(editor, stack, tmp_path):
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2"})
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "n3"})
    names = [r.name for r in stack.effective().llm_routes]
    assert names == ["lan", "n2", "n3"]
    editor.add_entry("appraisal.rules", None, {"event": "custom.thing", "valence": 0.4})
    editor.add_entry("appraisal.rules", None, {"event": "zz.third", "valence": 0.5})
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert [r.name for r in fresh.effective().llm_routes] == ["lan", "n2", "n3"]
    assert rule(fresh, "custom.thing").valence == 0.4 and rule(fresh, "zz.third").valence == 0.5


def test_reset_of_the_blocklist_is_refused_other_tightenings_reset(editor, stack):
    path = "safety.minor_output_blocklist.en"
    editor.change(path, [*get_path(stack.effective(), path), "extra*"])
    with pytest.raises(ValidationError) as caught:
        editor.reset(path)
    assert caught.value.code == "not_resettable"
    assert "extra*" in get_path(stack.effective(), path)
    editor.change("webui.name_max_chars", 32)
    assert editor.reset("webui.name_max_chars")["changed"]
    assert stack.effective().webui.name_max_chars == 64


def test_restore_drops_an_added_entry_that_does_not_fit(tmp_path):
    path = tmp_path / FILE
    good = {**ROUTE, "name": "ok"}
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "overrides": {
                    'llm_routes[{"name":"ok"}]': good,
                    'llm_routes[{"name":"liar"}]': {**ROUTE, "name": "other"},
                    'llm_routes[{"name":"lan"}]': {**ROUTE, "model": "x"},  # a lower layer has it
                    "voice.moods.joy": {"rate": 0.9},  # a lower layer has it
                },
            }
        )
    )
    path.chmod(0o600)
    stack = make_stack(tmp_path, llm_routes=[ROUTE])
    SettingsEditor(stack, store=path).restore()
    assert [r.name for r in stack.effective().llm_routes] == ["lan", "ok"]
    assert stack.effective().voice.moods["joy"].rate != 0.9
    assert list(json.loads(path.read_text())["overrides"]) == ['llm_routes[{"name":"ok"}]']


# ------------------------------------------------------------------ fix round: correctness


def test_a_patch_whose_entry_vanishes_does_not_become_an_entry(editor, stack, tmp_path):
    at = rule_index(stack, "zz.first")
    editor.change(f"appraisal.rules.{at}.valence", 0.9)
    editor.change("llm_routes.lan.model", "m2")
    stack.app["appraisal"]["rules"] = [RULE_B]  # a reload that drops zz.first (still declared)
    stack.app["llm_routes"] = []
    assert all(r.event != "zz.first" for r in stack.effective().appraisal.rules)
    assert stack.effective().llm_routes == []
    assert stack.effective() is not None  # the stack is still valid
    fresh = make_stack(tmp_path, appraisal=appraisal([RULE_B]), llm_routes=[])
    SettingsEditor(fresh, store=tmp_path / FILE).restore()  # live equals after a restart
    live = stack.effective().model_dump(mode="json")
    assert (
        live["appraisal"]["rules"]
        == fresh.effective().model_dump(mode="json")["appraisal"]["rules"]
    )
    assert live["llm_routes"] == fresh.effective().model_dump(mode="json")["llm_routes"]
    assert not fresh._runtime


def test_a_persist_failure_changes_nothing(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text(yaml.safe_dump({"webui": {"bind": "127.0.0.1", "port": 0}}))
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("webui.name_max_chars", 32)
    stack.files["app"] = tmp_path  # a directory: writing it raises OSError
    events, audit, approved = [], len(stack.audit), stack._approved
    stack.subscribe(events.append)
    saved_app = json.dumps(stack.app)
    with pytest.raises(OSError):
        editor.change("webui.name_max_chars", 100)
    assert json.dumps(stack.app) == saved_app and stack._approved is approved
    assert events == [] and len(stack.audit) == audit
    assert stack.effective().webui.name_max_chars == 32  # the runtime value is untouched


def test_persist_failure_is_a_bare_code_over_http(tmp_path):
    app = tmp_path / "app.yaml"
    app.write_text(yaml.safe_dump({"webui": {"bind": "127.0.0.1", "port": 0}}))
    stack = ConfigStack.from_files(app=app)
    stack.files["app"] = tmp_path
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        out = post(client, "/api/settings", path="webui.name_max_chars", value=100)
        assert out.status == 503 and out.json == {"error": "persist_failed"}
    finally:
        srv.stop()


def test_no_app_retry_for_a_field_inside_an_entry_added_here(editor, stack):
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2"})
    with pytest.raises(GuardrailError):
        editor.change("llm_routes.n2.allowed_for_minors", True)
    assert "llm_routes" in stack.app and [r["name"] for r in stack.app["llm_routes"]] == ["lan"]
    assert editor.change("llm_routes.lan.allowed_for_minors", True)["layer"] == "app"


def test_reset_with_nothing_to_clear_keeps_file_and_memory_consistent(editor, stack, tmp_path):
    editor.change("affect.update_hz", 33.0)
    stack.clear("affect.update_hz")  # the runtime value goes, the file entry stays: stale
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {"affect.update_hz": 33.0}
    out = editor.reset("affect.update_hz")
    assert out["changed"] is False and out["persisted"] is True
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {}
    assert editor._overrides == {}


def test_a_keyed_entry_cannot_expire_and_natural_keys_use_the_patch_form(stack):
    with pytest.raises(ConfigError):
        stack.set("llm_routes.lan.model", "m2", ttl_s=5)
    with pytest.raises(ConfigError):
        stack.set(f"appraisal.rules.{rule_index(stack, 'zz.first')}.valence", 0.5, ttl_s=5)
    stack.set("llm_routes.lan.model", "m2")
    stack.set(f"appraisal.rules.{rule_index(stack, 'zz.first')}.valence", 0.5)
    assert set(stack._runtime) == {"llm_routes#patch", "appraisal.rules#patch"}
    assert stack.clear("llm_routes.lan.model") and stack.clear(
        f"appraisal.rules.{rule_index(stack, 'zz.first')}.valence"
    )
    assert stack._runtime == {}  # dead partials are dropped


def test_pop_leaf_prunes_empty_parents():
    from roboemot.config.stack import _pop_leaf

    entry = {"event": "e", "when": {"a": {"b": 1}}}
    assert _pop_leaf(entry, "when.a.b") == (True, 1)
    assert entry == {"event": "e"}
    assert _pop_leaf(entry, "when.a.b")[0] is False


def test_a_clear_that_does_not_make_an_invalid_state_worse_is_allowed(tmp_path):
    from roboemot.config.stack import _RuntimeValue

    stack = make_stack(tmp_path)
    stack._runtime["affect.update_hz"] = _RuntimeValue(9999.0, None)  # out of range
    with pytest.raises(ConfigError):
        stack.effective()
    assert stack.clear("affect.update_hz") is True
    assert stack.effective().affect.update_hz == 20.0


def test_duplicate_identities_resolve_to_the_first_entry():
    from roboemot.config.stack import _merge_keyed, keyed_index

    base = [{"name": "a", "x": 1}, {"name": "a", "x": 2}]
    assert _merge_keyed(base, [{"name": "a", "y": 9}], "name") == [
        {"name": "a", "x": 1, "y": 9},
        {"name": "a", "x": 2},
    ]
    assert (
        keyed_index(
            {"appraisal": {"rules": [{"event": "e"}, {"event": "e"}]}},
            "appraisal.rules",
            {"event": "e"},
        )
        == 0
    )
