"""Field catalogue: descriptor contract, source layers, secret redaction (SPEC 11A, 12)."""

from __future__ import annotations

import json

import pytest
from roboemot.config.catalogue import (
    CONTAINERS,
    DESCRIPTOR_KEYS,
    KINDS,
    PRIVATE_PATHS,
    build_catalogue,
    descriptor,
    is_private,
)
from roboemot.config.stack import ConfigStack, flatten, get_path
from roboemot.diagnostics import jsonable
from roboemot.redact import REDACTED, secret_shaped

CANARY = "hunter2-CANARY"
ROUTE = {
    "name": "cloud",
    "base_url": f"https://alice:{CANARY}@cloud.example/v1",
    "api_key": "${secret:cloud_key}",
    "model": "m",
}


def rows(stack: ConfigStack) -> dict:
    return {r["path"]: r for r in build_catalogue(stack)}


def test_the_descriptor_contract_is_pinned():
    assert set(DESCRIPTOR_KEYS) == {
        "path", "kind", "nullable", "value", "default", "source", "ge", "le", "unit",
        "restart", "admin_only", "opt_in", "conservative", "description", "choices",
        "item_kind", "length", "key", "entry", "secret", "private", "writable", "removable",
    }  # fmt: skip
    assert set(KINDS) == {
        "number", "int", "bool", "text", "enum", "tuple", "string_list", "enum_list",
        "tuple_list", "string_dict", "number_dict", "string_list_dict", "json",
        "dict_of_model", "keyed_list",
    }  # fmt: skip


def test_every_descriptor_has_exactly_the_contract_keys_and_is_json():
    catalogue = build_catalogue(ConfigStack())
    assert catalogue
    for row in catalogue:
        assert set(row) == DESCRIPTOR_KEYS, row["path"]
        assert row["kind"] in KINDS, row["path"]
        json.dumps(row)


def test_every_config_leaf_is_covered_by_a_descriptor():
    stack = ConfigStack()
    by_path = rows(stack)
    cfg = stack.effective()
    for leaf in flatten(cfg.model_dump(mode="json")):
        owners = [p for p in by_path if leaf == p or leaf.startswith(p + ".")]
        assert owners, f"{leaf} has no descriptor"
    for path, row in by_path.items():
        get_path(cfg, path)  # every descriptor path resolves
        if row["kind"] not in CONTAINERS and not row["private"]:
            assert row["value"] == jsonable_free(get_path(cfg, path)), path


def jsonable_free(value):
    return json.loads(json.dumps(value, default=lambda o: o.model_dump(mode="json")))


def test_metadata_comes_from_the_schema():
    r = rows(ConfigStack())
    volume = r["circadian.states.resting.volume_db"]
    assert (volume["kind"], volume["unit"], volume["ge"], volume["le"]) == ("number", "dB", -24, 6)
    assert r["voice.backend"]["kind"] == "enum" and "piper" in r["voice.backend"]["choices"]
    assert r["voice.limits.rate"]["kind"] == "tuple" and r["voice.limits.rate"]["length"] == 2
    assert r["voice.keywords"]["kind"] == "dict_of_model"
    assert r["llm_routes"]["kind"] == "keyed_list" and r["llm_routes"]["key"] == "name"
    assert r["safety.minor_output_blocklist.en"]["kind"] == "string_list"
    assert r["safety.minor_output_blocklist.en"]["conservative"] == "superset"
    assert r["voice.wake_word"]["nullable"] is True
    assert any(row["restart"] for row in r.values())
    assert any(row["admin_only"] for row in r.values())
    assert any(row["opt_in"] for row in r.values())


def test_source_of_names_the_deciding_layer():
    path = "voice.base.volume_db"
    stack = ConfigStack(
        defaults={"voice": {"base": {"volume_db": -1.0}}},
        manifest={"voice": {"base": {"volume_db": -2.0}}},
        app={"voice": {"base": {"rate": 1.1}}},
    )
    assert stack.source_of(path) == "manifest"
    assert stack.source_of("voice.base.rate") == "app"
    assert stack.source_of("voice.base.pitch_st") == "schema"
    stack.derived = {"voice": {"base": {"volume_db": -3.0}}}
    assert stack.source_of(path) == "derived"
    stack.set(path, -4.0, source="t")
    assert stack.source_of(path) == "runtime"
    assert stack.source_of("voice.base") == "runtime"  # an override below the section
    stack.clear(path)
    assert stack.source_of(path) == "derived"


def test_source_of_sees_a_section_override_and_ignores_an_expired_one():
    now = [1000.0]
    stack = ConfigStack(clock=lambda: now[0])
    stack.set("voice.limits", {"rate": [0.9, 1.1]}, source="t")
    assert stack.source_of("voice.limits.rate") == "runtime"
    stack.set("voice.base.volume_db", -5.0, ttl_s=10, source="t")
    assert stack.source_of("voice.base.volume_db") == "runtime"
    now[0] += 11
    assert stack.source_of("voice.base.volume_db") == "schema"


def test_source_of_follows_keyed_list_entries_by_name():
    stack = ConfigStack(app={"llm_routes": [{**ROUTE, "base_url": "http://x/v1"}]})
    assert stack.source_of("llm_routes.cloud.model") == "app"
    assert stack.source_of("llm_routes.other.model") == "schema"


def test_the_catalogue_carries_source_and_a_layered_default():
    stack = ConfigStack(app={"voice": {"base": {"volume_db": -2.0}}})
    stack.set("voice.base.volume_db", -5.0, source="t")
    row = rows(stack)["voice.base.volume_db"]
    assert (row["value"], row["default"], row["source"]) == (-5.0, -2.0, "runtime")
    assert rows(stack)["voice.base.rate"]["source"] == "schema"


def test_a_secret_reference_and_credentials_in_a_url_are_redacted_everywhere():
    stack = ConfigStack(app={"llm_routes": [ROUTE]})
    r = rows(stack)
    key, url = r["llm_routes.cloud.api_key"], r["llm_routes.cloud.base_url"]
    for row in (key, url):
        assert row["secret"] is True and row["writable"] is False
        assert row["value"] == REDACTED and row["default"] in (REDACTED, None)
    blob = json.dumps(build_catalogue(stack))
    assert CANARY not in blob and "alice" not in blob and "cloud_key" not in blob
    assert r["llm_routes.cloud.model"]["secret"] is False


def test_a_secret_shaped_default_or_runtime_value_is_redacted_too():
    stack = ConfigStack(app={"llm_routes": [{**ROUTE, "base_url": "https://cloud.example/v1"}]})
    stack.derived = {"llm_routes": [{"name": "cloud", "base_url": f"https://u:{CANARY}@h/v1"}]}
    row = rows(stack)["llm_routes.cloud.base_url"]
    assert row["secret"] and CANARY not in json.dumps(row)
    assert row["default"] == REDACTED  # one secret-shaped side hides both


def test_only_the_route_key_is_secret_by_name_in_the_default_schema():
    stack = ConfigStack(app={"llm_routes": [{**ROUTE, "base_url": "https://cloud.example/v1"}]})
    names = {p for p, r in rows(stack).items() if r["secret"]}
    assert names == {"llm_routes.cloud.api_key"}
    r = rows(stack)
    assert r["llm_routes.cloud.max_tokens"]["secret"] is False
    assert r["voice.keywords"]["secret"] is False


@pytest.mark.parametrize(
    ("value", "shaped"),
    [
        ("${secret:k}", True),
        ("https://u:p@host/x", True),
        ("https://u@host", True),
        ("https://host/path@x", False),
        ("plain", False),
        (["a", "x://u:p@h"], True),
        ({"a": "${secret:k}"}, True),
        (3, False),
        ("https://h/v1?api_key=abc", True),
        ("https://h/v1?Token=abc&x=1", True),
        ("https://h/v1?key=abc", True),
        ("https://h/cb#access_token=abc", True),
        ("https://h/v1?page=2#top", False),
        ("see https://h/v1?api_key=abc please", True),
        ("ssh://user@host", True),
    ],
)
def test_secret_shape_rules(value, shaped):
    assert secret_shaped(value) is shaped


def test_the_debug_report_redacts_by_value_shape_as_well():
    out = jsonable({"base_url": f"https://u:{CANARY}@h/", "x": ["${secret:k}"], "ok": "fine"})
    assert out == {"base_url": REDACTED, "x": [REDACTED], "ok": "fine"}


def test_a_sibling_of_a_runtime_section_override_is_not_runtime():
    stack = ConfigStack(app={"voice": {"limits": {"pitch_st": [-1.0, 1.0]}}})
    stack.set("voice.limits", {"rate": [0.9, 1.1]}, source="t")
    assert stack.source_of("voice.limits.rate") == "runtime"
    assert stack.source_of("voice.limits.pitch_st") == "app"
    assert stack.source_of("voice.limits.volume_db") == "schema"


def test_source_and_default_follow_rule_identity_not_position():
    stack = ConfigStack(
        app={
            "appraisal": {
                "extra_events": ["brand.new"],
                "rules": [
                    {"event": "brand.new", "valence": 0.5},
                    {"event": "person.known_adult", "valence": 0.9},
                ],
            }
        }
    )
    r = rows(stack)
    events = {r[p]["value"]: p.removesuffix(".event") for p in r if p.endswith(".event")}
    new = events["brand.new"]
    assert (r[new + ".valence"]["source"], r[new + ".valence"]["default"]) == ("app", 0.5)
    adult = events["person.known_adult"]
    assert adult == "appraisal.rules.3"  # the default rule keeps its place
    assert (r[adult + ".valence"]["source"], r[adult + ".valence"]["default"]) == ("app", 0.9)
    assert r[adult + ".arousal"]["source"] == "schema"
    assert r["appraisal.rules.0.valence"]["source"] == "schema"  # not claimed by app index 0
    assert r["appraisal.rules.0.valence"]["default"] == 0.4


def test_keyed_entries_with_numeric_names_get_index_segments_and_plain_ones_their_name():
    routes = [
        {**ROUTE, "name": "7", "base_url": "http://a/v1"},
        {**ROUTE, "name": "plain", "base_url": "http://b/v1"},
    ]
    r = rows(ConfigStack(app={"llm_routes": routes}))
    assert r["llm_routes"]["value"] == ["0", "plain"]
    assert r["llm_routes.0.base_url"]["source"] == "app"
    assert r["llm_routes.plain.model"]["source"] == "app"
    assert r["llm_routes.plain.model"]["default"] == "m"


def test_dotted_rule_events_are_addressed_by_index():
    r = rows(ConfigStack())
    assert all("." not in seg for seg in r["appraisal.rules"]["value"])
    assert "appraisal.rules.0.valence" in r


def test_no_container_key_in_the_default_catalogue_contains_a_dot():
    for row in build_catalogue(ConfigStack()):
        if row["kind"] in CONTAINERS:
            assert all("." not in k for k in row["value"]), row["path"]


def test_dotted_or_secret_shaped_dict_keys_are_not_listed():
    stack = ConfigStack()
    stack.derived = {
        "conversation": {"confirm_yes": {"it": ["si"], "a.b": ["x"], "https://u:p@h": ["y"]}}
    }
    paths = {p for p in rows(stack) if p.startswith("conversation.confirm_yes")}
    assert "conversation.confirm_yes.it" in paths
    assert not any("a.b" in p or "u:p" in p for p in paths)
    assert rows(stack)["conversation.confirm_yes"]["value"] == ["it", "en"]


def test_a_credential_named_inner_key_makes_a_free_form_map_secret():
    stack = ConfigStack()
    stack.derived = {"vision": {"object_display_names": {"api_key": "x", "cup": "tazza"}}}
    row = rows(stack)["vision.object_display_names"]
    assert row["secret"] and row["value"] == REDACTED
    assert not rows(ConfigStack())["vision.object_display_names"]["secret"]


def test_circadian_location_is_private_but_not_secret_and_can_stay_writable():
    assert "circadian.location" in PRIVATE_PATHS
    stack = ConfigStack()
    stack.set("circadian.location", {"lat": 45.46, "lon": 9.19}, source="t")
    rows_ = build_catalogue(stack, lambda p: p == "circadian.location")
    row = {r["path"]: r for r in rows_}["circadian.location"]
    assert row["private"] and row["admin_only"] and not row["secret"] and row["writable"]
    assert row["value"] == REDACTED and "45.46" not in json.dumps(row)
    shown = build_catalogue(stack, reveal_private=True)
    assert {r["path"]: r for r in shown}["circadian.location"]["value"] == {
        "lat": 45.46,
        "lon": 9.19,
    }


def test_private_paths_are_listed_explicitly_and_cover_every_minor_and_privacy_field():
    private = {p for p, r in rows(ConfigStack()).items() if r["private"]}
    assert is_private("privacy.debug_recording") and is_private("safety.minor_lines.it.greeting")
    assert not is_private("privacy_x") and not is_private("voice.wake_word")
    for path in rows(ConfigStack()):
        if path.split(".")[0] in ("safety", "voice", "register", "reflexes") and "minor" in path:
            assert path in private, path
        if path.startswith("privacy."):
            assert path in private, path


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("affect.update_hz", "number"),
        ("schema_version", "int"),
        ("register.child.approach_allowed", "bool"),
        ("personality", "text"),
        ("register.child.vocabulary", "enum"),
        ("circadian.quiet_hours", "tuple"),
        ("appraisal.extra_events", "string_list"),
        ("world.autoenroll.kinds", "enum_list"),
        ("persona.trait_effects.energy.motion_amplitude_x", "tuple_list"),
        ("vision.object_display_names", "string_dict"),
        ("circadian.location", "number_dict"),
        ("conversation.confirm_yes", "string_list_dict"),
        ("appraisal.rules.0.when", "json"),
        ("voice.keywords", "dict_of_model"),
        ("llm_routes", "keyed_list"),
    ],
)
def test_real_schema_paths_have_the_expected_kind(path, kind):
    assert rows(ConfigStack())[path]["kind"] == kind


def test_the_table_covers_every_kind():
    assert len(KINDS) == 15


def test_containers_carry_an_entry_form_and_other_kinds_none():
    r = rows(ConfigStack())
    entry = {e["path"]: e for e in r["llm_routes"]["entry"]}
    assert "base_url" in entry and entry["api_key"]["secret"] is True
    assert entry["base_url"]["kind"] == "text"
    assert {e["path"] for e in r["voice.keywords"]["entry"]} >= {"wake_phrases"}
    assert r["conversation.confirm_yes"]["entry"] is None
    assert all(
        row["entry"] is None
        for row in r.values()
        if row["kind"] not in ("dict_of_model", "keyed_list")
    )
    assert entry["name"]["restart"] is False and entry["first_token_timeout_s"]["ge"] is not None


def test_the_descriptor_factory_fills_every_key_and_redacts():
    row = descriptor("robot.limits.pitch", "number", 0.5, unit="rad")
    assert set(row) == DESCRIPTOR_KEYS and row["unit"] == "rad" and row["source"] == "schema"
    hidden = descriptor("robot.link", "text", "https://u:p@h/")
    assert hidden["secret"] and hidden["value"] == REDACTED
    with pytest.raises(ValueError):
        descriptor("x", "nope")
    with pytest.raises(ValueError):
        descriptor("x", "text", bogus=1)


def test_a_nullable_container_with_no_value_becomes_a_json_leaf():
    row = rows(ConfigStack())["circadian_motion.yawn"]
    assert row["kind"] == "json" and row["nullable"] and row["value"] is None


def test_the_debug_report_shows_a_user_name_without_a_password():
    assert jsonable({"remote": "ssh://user@host"}, bare_user=False) == {"remote": "ssh://user@host"}
    assert jsonable({"r": "see ssh://pollen@reachy.local now"}, bare_user=False) == {
        "r": "see ssh://pollen@reachy.local now"
    }


def test_the_debug_report_still_hides_a_password_and_other_secret_shapes():
    for value in (
        "https://u:p@host/",
        "ssh://user:pw@host",
        "${secret:k}",
        "https://h/v1?api_key=x",
    ):
        assert jsonable({"remote": value}, bare_user=False) == {"remote": REDACTED}


def test_the_catalogue_still_redacts_a_user_name_without_a_password():
    assert jsonable({"remote": "ssh://user@host"}) == {"remote": REDACTED}
    stack = ConfigStack(
        app={"llm_routes": [{**ROUTE, "base_url": "https://alice@cloud.example/v1"}]}
    )
    row = rows(stack)["llm_routes.cloud.base_url"]
    assert row["secret"] is True and row["value"] == REDACTED


def test_the_debug_collector_shows_a_bare_user_and_hides_a_password():
    from roboemot.diagnostics import FnSource, collect_section

    def data():
        return {"a": "ssh://pollen@reachy.local", "b": "ssh://pollen:pw@reachy.local"}

    section = collect_section(FnSource("t", data, "robot"), 5)
    assert section["data"] == {"a": "ssh://pollen@reachy.local", "b": REDACTED}


@pytest.mark.parametrize(
    "url",
    ["https://u@x:p@host", "https://u%3Ap@host", "https://a:b@c@host", "https://u:p@host"],
)
def test_the_debug_report_hides_user_info_that_may_hold_a_password(url):
    assert jsonable({"u": url}, bare_user=False) == {"u": REDACTED}
    assert secret_shaped(url) is True


@pytest.mark.parametrize("url", ["http://[::1]", "https://user@host:8080", "ssh://Pollen@h.local"])
def test_the_debug_report_keeps_plain_hosts_and_user_names(url):
    assert jsonable({"u": url}, bare_user=False) == {"u": url}


@pytest.mark.parametrize(
    "user",
    ["ghp_abc123", "gho_x", "github_pat_11ABC", "glpat-xyz", "xoxb-1", "sk-abc", "a" * 33],
)
def test_the_debug_report_masks_a_user_that_looks_like_a_token_but_keeps_the_host(user):
    out = jsonable({"u": f"https://{user}@git.example/x"}, bare_user=False)
    assert out == {"u": "https://***@git.example/x"}


def test_removable_lists_only_entries_no_lower_layer_holds():
    stack = ConfigStack(app={"llm_routes": [{**ROUTE, "base_url": "http://x/v1"}]})
    stack.set("voice.moods.tender", {}, source="t")
    stack.set(
        "llm_routes",
        [{"name": "added", "base_url": "https://y.example/v1", "model": "m"}],
        source="t",
    )
    r = rows(stack)
    assert r["voice.moods"]["removable"] == ["tender"]
    assert r["llm_routes"]["removable"] == ["added"]
    assert r["voice.keywords"]["removable"] == []
    assert r["affect.update_hz"]["removable"] is None


LINES = dict.fromkeys(("greeting", "robot_name", "ask_grown_up", "filtered"), "hi")


def test_a_private_container_lists_nothing_removable_until_revealed():
    stack = ConfigStack()
    assert rows(stack)["safety.minor_lines"]["removable"] == []
    stack.set("safety.minor_lines.xx", LINES, source="t", admin=True)
    assert rows(stack)["safety.minor_lines"]["removable"] == []
    shown = {r["path"]: r for r in build_catalogue(stack, reveal_private=True)}
    assert shown["safety.minor_lines"]["removable"] == ["xx"]


@pytest.mark.parametrize(
    "text",
    [
        "https://a@b,https://c:d@e",
        "https://host,https://u:p@h2",
        "https://h/x?next=https://u:p@h2",
        "see //u:p@h now",
        "//user@h",
    ],
)
def test_glued_nested_and_scheme_relative_credentials_are_secret_everywhere(text):
    from roboemot.redact import scrub_text

    assert secret_shaped(text) is True
    assert jsonable({"u": text}, bare_user=False) == {"u": REDACTED}
    assert "u:p" not in scrub_text(text) and "c:d" not in scrub_text(text)


def test_scheme_relative_is_not_confused_with_a_path_or_a_full_url():
    assert not secret_shaped("https://h/a//b@c") and not secret_shaped("see https://h/x")


@pytest.mark.parametrize(
    "user",
    [
        "hf_abcdef", "npm_abc", "pypi-AgEI", "sk_live_x", "sk_test_x", "AKIAIOSFODNN7EXAMPLE",
        "ya29.a0Af", "0123456789abcdef0123456789abcdef", "abc123def456ghi789jk", "u" * 21,
    ],
)  # fmt: skip
def test_token_like_users_are_masked_in_the_debug_report(user):
    out = jsonable({"u": f"https://{user}@git.example/x"}, bare_user=False)
    assert out == {"u": "https://***@git.example/x"}


def test_ordinary_user_names_stay_visible_in_the_debug_report():
    for user in ("pollen", "Alice", "deploy.bot", "svc_account-1"):
        assert jsonable({"u": f"ssh://{user}@h"}, bare_user=False) == {"u": f"ssh://{user}@h"}


def test_ipv6_origins_keep_their_brackets_so_port_and_address_differ():
    from roboemot.redact import origin

    assert origin("https://[2001:db8::1]:443/x") != origin("https://[2001:db8::1:443]/x")
    assert origin("https://[2001:db8::1]:8443/x") == "https://[2001:db8::1]:8443"
