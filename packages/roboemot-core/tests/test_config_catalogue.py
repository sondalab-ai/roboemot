"""Field catalogue: descriptor contract, source layers, secret redaction (SPEC 11A, 12)."""

from __future__ import annotations

import json
import time

import pytest
from roboemot.config.catalogue import DESCRIPTOR_KEYS, KINDS, build_catalogue
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigStack, flatten, get_path
from roboemot.diagnostics import REDACTED, jsonable, secret_shaped

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
        "item_kind", "length", "key", "secret", "writable",
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
        if row["kind"] not in ("dict_of_model", "keyed_list", "string_list_dict"):
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
    ],
)
def test_secret_shape_rules(value, shaped):
    assert secret_shaped(value) is shaped


def test_the_debug_report_redacts_by_value_shape_as_well():
    out = jsonable({"base_url": f"https://u:{CANARY}@h/", "x": ["${secret:k}"], "ok": "fine"})
    assert out == {"base_url": REDACTED, "x": [REDACTED], "ok": "fine"}


def test_building_the_catalogue_is_quick_and_validates_against_the_schema():
    start = time.monotonic()
    build_catalogue(ConfigStack())
    assert time.monotonic() - start < 5
    assert RoboEmotConfig().voice.keywords
