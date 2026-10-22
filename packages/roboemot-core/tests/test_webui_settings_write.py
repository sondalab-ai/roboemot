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


def test_the_identity_of_an_entry_with_a_dotted_event_cannot_be_renamed(editor, stack):
    editor.add_entry(
        "appraisal.rules",
        None,
        {"event": "link.up", "valence": 0.1, "arousal": 0.1, "label": "joy"},
    )
    index = rule_index(stack, "link.up")
    for leaf in ("event", "when"):
        with pytest.raises(SettingNotAllowed):
            editor.change(f"appraisal.rules.{index}.{leaf}", "link.down" if leaf == "event" else {})
    assert rule(stack, "link.up").event == "link.up"


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


LINES = {
    "greeting": "Hallo!",
    "robot_name": "Ich heisse {name}.",
    "ask_grown_up": "Frag einen Erwachsenen.",
    "filtered": "Das sage ich nicht.",
}


@pytest.fixture
def german(monkeypatch):
    monkeypatch.setattr("roboemot.webui.settings.supported_languages", lambda: {"it", "en", "de"})


def test_the_supported_languages_come_from_the_voice_models():
    from roboemot.voice.models import supported_languages

    assert supported_languages() == {"it", "en"}


def test_a_language_the_voice_stack_lacks_cannot_be_added(editor, stack):
    for path in ("voice.keywords", "safety.minor_lines"):
        with pytest.raises(ValidationError) as caught:
            editor.add_entry(path, "fr", {})
        assert caught.value.code == "language_unsupported"
    with pytest.raises(SettingNotAllowed):
        editor.add_entry("conversation.confirm_yes", "fr", {})
    for path in (
        "voice.keywords.fr.wake_phrases",
        "safety.minor_lines.fr.greeting",
        "conversation.confirm_yes.fr",
    ):
        with pytest.raises(SettingNotAllowed):
            editor.change(path, ["x"])
    assert "fr" not in stack.effective().voice.keywords


def test_a_supported_language_can_be_added_removed_and_restored(german, editor, stack, tmp_path):
    editor.add_entry("safety.minor_lines", "de", LINES)
    editor.add_entry("voice.keywords", "de", {"wake_phrases": ["hallo robo"]})
    assert stack.effective().safety.minor_lines["de"].greeting == "Hallo!"
    rows = {r["path"]: r for r in editor.fields()}
    assert rows["voice.keywords"]["removable"] == ["de"]
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert "de" in fresh.effective().voice.keywords and "de" in fresh.effective().safety.minor_lines
    assert editor.remove_entry("voice.keywords.de")["changed"]
    assert "de" not in stack.effective().voice.keywords


def test_a_new_language_gets_the_same_line_checks_as_an_existing_one(german, editor):
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("safety.minor_lines", "de", {**LINES, "greeting": "Hi {oops}"})
    assert caught.value.code == "line_invalid"
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("voice.keywords", "de", {"wake_phrases": ["bad!!"]})
    assert caught.value.code == "phrase_invalid"


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
        if calls["n"] == 1:  # the commit's validation
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


def test_restore_carries_a_renamed_tightening_over_and_logs_removed_keys(tmp_path, caplog):
    path = tmp_path / FILE
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "overrides": {
                    "reflexes.antenna_ratio": 0.5,
                    "connectivity.gestures.antenna_rad": 0.2,
                },
            }
        )
    )
    path.chmod(0o600)
    stack = ConfigStack()
    with caplog.at_level("WARNING"):
        SettingsEditor(stack, store=path).restore()
    assert stack.effective().reflexes.startle_lift_ratio == 0.5
    assert set(json.loads(path.read_text())["overrides"]) == {"reflexes.startle_lift_ratio"}
    assert "connectivity.gestures.antenna_rad" in caplog.text and "removed" in caplog.text


def test_restore_prefers_the_new_key_when_both_are_stored(tmp_path):
    path = tmp_path / FILE
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "overrides": {"reflexes.antenna_ratio": 0.5, "reflexes.startle_lift_ratio": 1.0},
            }
        )
    )
    path.chmod(0o600)
    stack = ConfigStack()
    SettingsEditor(stack, store=path).restore()
    assert stack.effective().reflexes.startle_lift_ratio == 1.0


@pytest.mark.parametrize(
    ("app", "named", "hint"),
    [
        ({"reflexes": {"antenna_ratio": 1.0}}, "reflexes.antenna_ratio", "reflexes.startle_lift"),
        ({"connectivity": {"gestures": {"duration_s": 1}}}, "connectivity.gestures", "gesture_map"),
    ],
)
def test_a_removed_key_in_app_yaml_names_the_key_and_its_replacement(app, named, hint):
    with pytest.raises(ConfigError) as caught:
        ConfigStack(app=app)
    assert named in str(caught.value) and hint in str(caught.value)


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
    assert odd.status == 422 and odd.json == {"error": "language_unsupported", "field": "key"}
    smuggled = post(web, "/api/settings/reset", path="affect.update_hz", layer="app")
    assert smuggled.status == 422 and smuggled.json["error"] == "unknown_field"
    assert post(web, "/api/settings/reset").status == 400


def test_new_error_codes_exist_in_both_languages():
    from roboemot.webui.strings import EN, IT

    for code in (
        "value_range", "value_choice", "value_merged", "text_invalid", "text_too_long",
        "key_invalid", "entry_exists", "entry_invalid", "entry_in_lower_layer", "not_resettable",
        "language_unsupported", "endpoint_ack_required", "url_credentials",
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
@pytest.mark.parametrize(
    ("url", "target"),
    [
        ("https://evil.example.com/v1", "https://evil.example.com"),
        ("http://api.example.com/v1", "http://api.example.com"),
        ("https://api.example.com:8443/v1", "https://api.example.com:8443"),
    ],
)
def test_a_keyed_route_changes_endpoint_only_with_an_ack_naming_the_new_one(
    tmp_path, api_key, url, target
):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": api_key}])
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    path = "llm_routes.lan.base_url"
    with pytest.raises(ValidationError) as caught:
        editor.change(path, url)
    assert caught.value.code == "endpoint_ack_required" and caught.value.extra == {
        "endpoint": target
    }
    for wrong in (None, "", "https://api.example.com", target + "/", "evil.example.com"):
        with pytest.raises(ValidationError):
            editor.change(path, url, ack=wrong)
    assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]
    assert api_key not in str(caught.value) and api_key not in caught.value.extra["endpoint"]
    assert editor.change(path, url, ack=target)["changed"]
    assert stack.effective().llm_routes[0].base_url == url
    note = next(a for a in stack.audit if a.key == "llm_routes.lan.endpoint")
    assert note.old == "https://api.example.com" and note.new == target
    assert api_key not in (tmp_path / "audit.jsonl").read_text()
    fresh = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": api_key}])
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert fresh.effective().llm_routes[0].base_url == url  # the confirmed move survives a restart


def test_the_same_endpoint_needs_no_ack_and_the_audit_names_hosts_only(tmp_path):
    userinfo = "https://alice:pw-CANARY-9@api.example.com/v1"
    stack = make_stack(
        tmp_path, llm_routes=[{**ROUTE, "api_key": "${secret:k}", "base_url": userinfo}]
    )
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("llm_routes.lan.base_url", "https://api.example.com/v2")  # same endpoint
    assert not any(a.key.endswith(".endpoint") for a in stack.audit)
    editor.change(
        "llm_routes.lan.base_url", "https://other.example.com/v1", ack="https://other.example.com"
    )
    blob = (tmp_path / "audit.jsonl").read_text()
    assert "pw-CANARY-9" not in blob and "alice" not in blob and "${secret" not in blob
    note = next(a for a in stack.audit if a.key == "llm_routes.lan.endpoint")
    assert (note.old, note.new) == ("https://api.example.com", "https://other.example.com")


def test_an_ack_does_not_unlock_the_other_secret_rules(tmp_path):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": "${secret:k}"}])
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    with pytest.raises(GuardrailError) as caught:  # a LAN host would also make the route local
        editor.change(
            "llm_routes.lan.base_url", "http://nas.local:8080/v1", ack="http://nas.local:8080"
        )
    assert {v.check for v in caught.value.violations} == {"locality"}
    assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]
    with pytest.raises(SettingNotAllowed):
        editor.change("llm_routes.lan.api_key", "x")


def test_the_ack_travels_in_the_request_and_the_error_names_the_endpoint(tmp_path):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": "${secret:k}"}])
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        body = {
            "path": "llm_routes.lan.base_url",
            "value": "https://evil.example.com/v1",
            "pin": PIN,
        }
        first = client.request("POST", "/api/settings", body)
        assert first.status == 422
        assert first.json == {
            "error": "endpoint_ack_required",
            "field": "value",
            "endpoint": "https://evil.example.com",
        }
        done = client.request(
            "POST", "/api/settings", {**body, "ack_endpoint": first.json["endpoint"]}
        )
        assert done.status == 200 and done.json["layer"] == "runtime"
    finally:
        srv.stop()


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


def test_the_blocklist_and_other_tightenings_reset(editor, stack):
    path = "safety.minor_output_blocklist.en"
    base = get_path(stack.effective(), path)
    editor.change(path, [*base, "extra*"])
    assert editor.reset(path)["changed"]
    assert get_path(stack.effective(), path) == base
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
    stack._invalidate()  # an in-place edit of the store is invisible to the effective() cache
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


# ------------------------------------------------------------------ fix round 2


def test_an_existing_loose_audit_file_becomes_private(tmp_path):
    import stat

    audit = tmp_path / "audit.jsonl"
    audit.write_text("")
    audit.chmod(0o644)
    make_stack(tmp_path).set("affect.update_hz", 30.0)
    assert stat.S_IMODE(audit.stat().st_mode) == 0o600
    assert audit.read_text()


def test_a_clear_from_an_invalid_state_still_runs_the_guardrails(tmp_path):
    from roboemot.config.stack import _RuntimeValue

    stack = lan_stack(tmp_path)
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("llm_routes.lan.locality", "cloud")
    editor.change("llm_routes.lan.base_url", "http://nas.local:8080/v1")
    stack._runtime["affect.update_hz"] = _RuntimeValue(9999.0, None)  # e.g. a bad reload
    with pytest.raises(ConfigError):
        stack.effective()
    before = json.dumps({k: v.value for k, v in stack._runtime.items()}, sort_keys=True)
    with pytest.raises(GuardrailError) as caught:
        stack.clear("llm_routes.lan.locality", admin=True, refuse=frozenset({"locality"}))
    assert {v.check for v in caught.value.violations} == {"locality"}
    with pytest.raises(GuardrailError):  # without refuse the unapproved loosening is still blocked
        stack.clear("llm_routes.lan.locality")
    assert json.dumps({k: v.value for k, v in stack._runtime.items()}, sort_keys=True) == before
    assert stack.clear("affect.update_hz") is True  # an invalid override can still go
    assert stack.effective().llm_routes[0].locality == "cloud"


def test_a_refused_app_retry_leaves_the_yaml_and_the_runtime_shadow_alone(tmp_path, monkeypatch):
    import roboemot.config.stack as stack_mod

    app = tmp_path / "app.yaml"
    app.write_text(yaml.safe_dump({"webui": {"bind": "127.0.0.1", "port": 0}}))
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / FILE)
    editor.change("webui.name_max_chars", 32)
    yaml_before, runtime_before = app.read_text(), json.dumps(stack.app)
    real = stack_mod.reflex_violations

    def refuse_when_loosened(approved, candidate):
        found = real(approved, candidate)
        if candidate.webui.name_max_chars == 100:
            found.append(Violation("safety", "reflexes.x", 1, 2, "reflex"))
        return found

    monkeypatch.setattr(stack_mod, "reflex_violations", refuse_when_loosened)
    with pytest.raises(GuardrailError):
        editor.change("webui.name_max_chars", 100)
    assert app.read_text() == yaml_before and json.dumps(stack.app) == runtime_before
    assert stack.effective().webui.name_max_chars == 32  # the runtime value is still there
    assert json.loads((tmp_path / FILE).read_text())["overrides"] == {"webui.name_max_chars": 32}


def test_an_edit_of_an_added_entry_survives_a_later_add(editor, stack, tmp_path):
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "a", "max_tokens": 300})
    editor.change("llm_routes.a.max_tokens", 50)
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "b"})
    editor.add_entry("appraisal.rules", None, {"event": "custom.thing", "valence": 0.4})
    editor.change(f"appraisal.rules.{rule_index(stack, 'custom.thing')}.valence", 0.1)
    editor.add_entry("appraisal.rules", None, {"event": "zz.third", "valence": 0.5})

    def check(s):
        routes = {r.name: r for r in s.effective().llm_routes}
        assert routes["a"].max_tokens == 50 and set(routes) == {"lan", "a", "b"}
        assert rule(s, "custom.thing").valence == 0.1 and rule(s, "zz.third").valence == 0.5

    check(stack)
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    check(fresh)
    assert "llm_routes#patch" not in stack._runtime  # the patch slot holds lower-layer edits only


def test_remove_is_accepted_exactly_for_the_removable_entries(editor, stack):
    editor.add_entry("voice.moods", "tender", {})
    editor.add_entry("llm_routes", None, {**ROUTE, "name": "n2"})
    rows = {r["path"]: r for r in editor.fields()}
    assert rows["voice.moods"]["removable"] == ["tender"]
    assert rows["llm_routes"]["removable"] == ["n2"]
    for container in ("voice.moods", "llm_routes", "voice.keywords", "appraisal.rules"):
        for key in rows[container]["value"]:
            if key in rows[container]["removable"]:
                continue
            with pytest.raises(ValidationError) as caught:
                editor.remove_entry(f"{container}.{key}")
            assert caught.value.code == "entry_in_lower_layer"
    assert editor.remove_entry("voice.moods.tender")["changed"]
    assert editor.remove_entry("llm_routes.n2")["changed"]


def two_routes(tmp_path):
    routes = [
        {**ROUTE, "name": "a", "api_key": "${secret:k}"},
        {**ROUTE, "name": "b", "api_key": "${secret:k}"},
    ]
    return make_stack(tmp_path, llm_routes=routes)


def test_restore_replays_only_the_acknowledged_route_move(tmp_path):
    editor = SettingsEditor(two_routes(tmp_path), store=tmp_path / FILE)
    editor.change(
        "llm_routes.a.base_url", "https://new-a.example.com/v1", ack="https://new-a.example.com"
    )
    data = json.loads((tmp_path / FILE).read_text())
    assert [a["origin"] for a in data["acks"].values()] == ["https://new-a.example.com"]
    assert "secret:k" not in json.dumps(data["acks"])  # a fingerprint, never the key or its name
    key_b = next(iter(data["acks"])).replace('"a"', '"b"')
    data["overrides"][key_b] = "https://new-b.example.com/v1"  # edited into the file: never acked
    (tmp_path / FILE).write_text(json.dumps(data))
    fresh = two_routes(tmp_path)
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    urls = {r.name: r.base_url for r in fresh.effective().llm_routes}
    assert urls == {"a": "https://new-a.example.com/v1", "b": ROUTE["base_url"]}
    again = json.loads((tmp_path / FILE).read_text())
    assert key_b not in again["overrides"] and len(again["acks"]) == 1


def test_restore_drops_a_move_whose_origin_no_longer_matches_the_ack(tmp_path):
    editor = SettingsEditor(two_routes(tmp_path), store=tmp_path / FILE)
    editor.change(
        "llm_routes.a.base_url", "https://new-a.example.com/v1", ack="https://new-a.example.com"
    )
    data = json.loads((tmp_path / FILE).read_text())
    (key,) = data["acks"]
    data["overrides"][key] = "https://evil.example.com/v1"
    (tmp_path / FILE).write_text(json.dumps(data))
    fresh = two_routes(tmp_path)
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert {r.base_url for r in fresh.effective().llm_routes} == {ROUTE["base_url"]}


def test_a_later_edit_at_another_origin_forgets_the_ack_and_reset_drops_it(tmp_path):
    editor = SettingsEditor(two_routes(tmp_path), store=tmp_path / FILE)
    path = "llm_routes.a.base_url"
    editor.change(path, "https://new-a.example.com/v1", ack="https://new-a.example.com")
    editor.change(path, "https://new-a.example.com/v2")  # same origin: the ack stays
    assert len(json.loads((tmp_path / FILE).read_text())["acks"]) == 1
    editor.reset(path)
    assert "acks" not in json.loads((tmp_path / FILE).read_text())


def test_a_correct_ack_without_the_right_pin_changes_nothing(tmp_path):
    stack = make_stack(tmp_path, llm_routes=[{**ROUTE, "api_key": "${secret:k}"}])
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / FILE), stack)
    try:
        body = {
            "path": "llm_routes.lan.base_url",
            "value": "https://evil.example.com/v1",
            "ack_endpoint": "https://evil.example.com",
        }
        assert client.request("POST", "/api/settings", {**body, "pin": "000000"}).status in (
            401,
            403,
        )
        assert client.request("POST", "/api/settings", body).status in (400, 401, 403)
        assert stack.effective().llm_routes[0].base_url == ROUTE["base_url"]
        assert not any(a.key.endswith(".endpoint") for a in stack.audit)
    finally:
        srv.stop()


URL_A, ACK_A = "https://new-a.example.com/v1", "https://new-a.example.com"


def test_an_ack_does_not_survive_a_change_of_the_routes_key(tmp_path):
    s = make_stack(tmp_path, llm_routes=[{**ROUTE, "name": "a", "api_key": "${secret:k1}"}])
    SettingsEditor(s, store=tmp_path / FILE).change("llm_routes.a.base_url", URL_A, ack=ACK_A)
    for other in ("${secret:k2}", "sk-literal-key"):
        fresh = make_stack(tmp_path, llm_routes=[{**ROUTE, "name": "a", "api_key": other}])
        SettingsEditor(fresh, store=tmp_path / FILE).restore()
        assert fresh.effective().llm_routes[0].base_url == ROUTE["base_url"]
        data = json.loads((tmp_path / FILE).read_text())
        assert "acks" not in data and not data["overrides"]
        s = make_stack(tmp_path, llm_routes=[{**ROUTE, "name": "a", "api_key": "${secret:k1}"}])
        SettingsEditor(s, store=tmp_path / FILE).change("llm_routes.a.base_url", URL_A, ack=ACK_A)


def test_an_ack_does_not_survive_a_route_replaced_under_the_same_name(tmp_path):
    s = make_stack(tmp_path, llm_routes=[{**ROUTE, "name": "a", "api_key": "${secret:k1}"}])
    SettingsEditor(s, store=tmp_path / FILE).change("llm_routes.a.base_url", URL_A, ack=ACK_A)
    other = {**ROUTE, "name": "a", "base_url": "https://api.other-vendor.com/v1"}
    fresh = make_stack(tmp_path, llm_routes=[{**other, "api_key": "${secret:vendor2}"}])
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert fresh.effective().llm_routes[0].base_url == other["base_url"]


def test_a_literal_key_ack_holds_for_the_same_key_and_stores_no_literal(tmp_path):
    route = {**ROUTE, "name": "a", "api_key": "sk-literal-CANARY"}
    s = make_stack(tmp_path, llm_routes=[route])
    SettingsEditor(s, store=tmp_path / FILE).change("llm_routes.a.base_url", URL_A, ack=ACK_A)
    assert "CANARY" not in (tmp_path / FILE).read_text()
    fresh = make_stack(tmp_path, llm_routes=[route])
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert fresh.effective().llm_routes[0].base_url == URL_A


def test_a_live_path_edit_after_the_key_changed_is_not_covered_by_the_old_ack(tmp_path):
    s = make_stack(tmp_path, llm_routes=[{**ROUTE, "name": "a", "api_key": "${secret:k1}"}])
    editor = SettingsEditor(s, store=tmp_path / FILE)
    editor.change("llm_routes.a.base_url", URL_A, ack=ACK_A)
    s.app["llm_routes"][0]["api_key"] = "${secret:k2}"  # the owner re-keyed the route in the YAML
    with pytest.raises(GuardrailError):
        editor.change("llm_routes.a.base_url", ACK_A + "/v9")


def test_exempting_one_routes_secret_finding_leaves_the_others_refused(tmp_path):
    stack = two_routes(tmp_path)
    changes = {
        "llm_routes.a.base_url": "https://new-a.example.com/v1",
        "llm_routes.b.base_url": "https://new-b.example.com/v1",
    }
    with pytest.raises(GuardrailError) as caught:
        stack.set_many(
            changes,
            admin=True,
            refuse=frozenset({"secret"}),
            exempt=frozenset({"llm_routes.a.api_key"}),
        )
    assert {v.path for v in caught.value.violations} == {"llm_routes.b.api_key"}
    stack.set_many(
        changes,
        admin=True,
        refuse=frozenset({"secret"}),
        exempt=frozenset({"llm_routes.a.api_key", "llm_routes.b.api_key"}),
    )


def test_a_new_blocklist_language_is_added_cleaned_and_restored(german, editor, stack, tmp_path):
    path = "safety.minor_output_blocklist"
    with pytest.raises(ValidationError) as caught:
        editor.add_entry(path, "fr", ["mot"])
    assert caught.value.code == "language_unsupported"
    with pytest.raises(ValidationError) as caught:
        editor.add_entry(path, "de", ["bad!!"])
    assert caught.value.code == "phrase_invalid"
    with pytest.raises(ValidationError) as caught:
        editor.add_entry(path, "en", ["mot"])
    assert caught.value.code == "entry_exists"
    editor.add_entry(path, "de", ["scheisse*", "mist"])
    assert stack.effective().safety.minor_output_blocklist["de"] == ["scheisse*", "mist"]
    fresh = make_stack(tmp_path, llm_routes=[ROUTE], appraisal=appraisal([RULE_B, RULE_A]))
    SettingsEditor(fresh, store=tmp_path / FILE).restore()
    assert fresh.effective().safety.minor_output_blocklist["de"] == ["scheisse*", "mist"]
    assert "en" in fresh.effective().safety.minor_output_blocklist  # existing edits still replay


def test_a_new_minor_lines_language_is_still_filtered_by_every_blocklist(german, editor, stack):
    from roboemot.conversation.output_filter import OutputFilter

    editor.add_entry("safety.minor_lines", "de", LINES)
    lists = stack.effective().safety.minor_output_blocklist
    assert "de" not in lists  # no list of its own
    assert OutputFilter(lists).blocked("I will kill you")  # every language's list applies


def test_an_unreadable_voice_manifest_is_a_clean_refusal(editor, monkeypatch):
    from roboemot.voice.models import ModelError

    def broken():
        raise ModelError("manifest")

    monkeypatch.setattr("roboemot.webui.settings.supported_languages", broken)
    with pytest.raises(ValidationError) as caught:
        editor.add_entry("voice.keywords", "de", {})
    assert caught.value.code == "language_unsupported"


def test_the_supported_languages_are_cached():
    from roboemot.voice.models import supported_languages

    assert supported_languages() is supported_languages()
