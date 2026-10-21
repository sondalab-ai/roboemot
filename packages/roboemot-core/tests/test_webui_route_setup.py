"""Routes set up from the web UI: add, key choice, remove, restart (SPEC 8 Routes and Secrets, 11A,
12). The key is a reference to a stored secret; a literal key is never accepted; a keyed route
needs the acknowledged endpoint; the stack's locality and reflex rules stay refused."""

from __future__ import annotations

import json
import logging

import pytest
from roboemot.config.stack import ConfigStack, GuardrailError
from roboemot.secret_store import SecretStore
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingNotAllowed, SettingsEditor
from roboemot.webui.validation import ValidationError

FILE = "webui_settings.json"
KEY = "sk-ROUTE-CANARY-77aa"
CLOUD = {
    "name": "cloud",
    "base_url": "https://api.cloudco.example/v1",
    "model": "big",
    "locality": "cloud",
}
ORIGIN = "https://api.cloudco.example"
LAN = {"name": "lan", "base_url": "https://api.example.com/v1", "model": "m"}
KNOWN = ["cloud_key", "other_key"]


def make_stack(tmp_path, routes=(LAN,)):
    cfg = {"webui": {"bind": "127.0.0.1", "port": 0}, "llm_routes": [dict(r) for r in routes]}
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


def make_editor(tmp_path, stack, known=KNOWN):
    return SettingsEditor(stack, store=tmp_path / FILE, known_secrets=lambda: list(known))


@pytest.fixture
def stack(tmp_path):
    return make_stack(tmp_path)


@pytest.fixture
def editor(tmp_path, stack):
    return make_editor(tmp_path, stack)


def routes(stack):
    return {r.name: r for r in stack.effective().llm_routes}


def restarted(tmp_path, known=KNOWN):
    """A fresh process: new stack from the same app config, new editor over the same file."""
    stack = make_stack(tmp_path)
    editor = make_editor(tmp_path, stack, known)
    editor.restore()
    return stack, editor


def saved(tmp_path):
    return json.loads((tmp_path / FILE).read_text())


# ------------------------------------------------------------------ add


def test_add_a_route_without_a_key(editor, stack, tmp_path):
    out = editor.add_route(CLOUD, None, None)
    assert out["changed"] and out["persisted"] and out["layer"] == "runtime"
    route = routes(stack)["cloud"]
    assert (route.base_url, route.model, route.api_key, route.locality) == (
        CLOUD["base_url"],
        "big",
        "none",
        "cloud",
    )
    assert "keys" not in saved(tmp_path)


def test_a_keyed_route_needs_the_endpoint_acknowledged(editor, stack):
    with pytest.raises(ValidationError) as caught:
        editor.add_route(CLOUD, "cloud_key", None)
    assert caught.value.code == "endpoint_ack_required" and caught.value.extra == {
        "endpoint": ORIGIN
    }
    with pytest.raises(ValidationError):
        editor.add_route(CLOUD, "cloud_key", "https://evil.example")
    assert "cloud" not in routes(stack)


def test_a_keyed_route_is_added_with_the_acknowledgement(editor, stack, tmp_path):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    assert routes(stack)["cloud"].api_key == "${secret:cloud_key}"
    data = saved(tmp_path)
    assert data["keys"] == {"cloud": "${secret:cloud_key}"}
    assert "api_key" not in json.dumps(data["overrides"])
    assert [a.new for a in stack.audit if a.key == "llm_routes.cloud.endpoint"] == [ORIGIN]


@pytest.mark.parametrize("secret", ["ghost", "sk-LITERAL-1234", "a b", "", 5, ["cloud_key"]])
def test_only_a_stored_secret_name_can_be_chosen(editor, stack, secret):
    with pytest.raises(ValidationError) as caught:
        editor.add_route(CLOUD, secret, ORIGIN)
    assert caught.value.code in {"secret_unknown", "secret_name_invalid"}
    assert "cloud" not in routes(stack)


@pytest.mark.parametrize("literal", [KEY, "${secret:cloud_key}", "none", "Bearer x"])
def test_a_key_typed_into_the_route_is_refused(editor, stack, literal):
    value = {**CLOUD, "api_key": literal}
    with pytest.raises(ValidationError):
        editor.add_route(value, "cloud_key", ORIGIN)
    assert "cloud" not in routes(stack)


def test_the_generic_entry_add_still_cannot_carry_a_key(editor, stack):
    with pytest.raises(ValidationError):
        editor.add_entry("llm_routes", None, {**CLOUD, "api_key": "${secret:cloud_key}"})
    with pytest.raises(ValidationError):
        editor.add_entry("llm_routes", None, {**CLOUD, "api_key": KEY})
    assert "cloud" not in routes(stack)


def test_without_a_secret_store_no_key_can_be_chosen(tmp_path, stack):
    bare = SettingsEditor(stack, store=tmp_path / FILE)
    with pytest.raises(ValidationError) as caught:
        bare.add_route(CLOUD, "cloud_key", ORIGIN)
    assert caught.value.code == "secret_unknown"


def test_a_name_made_of_digits_is_refused(editor, stack):
    with pytest.raises(ValidationError) as caught:
        editor.add_route({**CLOUD, "name": "123"}, None, None)
    assert caught.value.code == "key_invalid"


def test_a_local_route_stays_refused_with_or_without_a_key(editor, stack):
    for value, secret, ack in (
        ({**CLOUD, "base_url": "http://nas.local:8080/v1", "locality": "auto"}, None, None),
        ({**CLOUD, "locality": "local"}, "cloud_key", ORIGIN),
    ):
        with pytest.raises(GuardrailError):
            editor.add_route(value, secret, ack)
    assert "cloud" not in routes(stack)


def test_a_new_route_cannot_be_cleared_for_minors(editor, stack):
    with pytest.raises(GuardrailError):
        editor.add_route({**CLOUD, "allowed_for_minors": True}, None, None)
    assert "cloud" not in routes(stack)


def test_credentials_in_the_url_are_refused(editor, stack):
    with pytest.raises(ValidationError) as caught:
        editor.add_route({**CLOUD, "base_url": "https://u:p@api.cloudco.example/v1"}, None, None)
    assert caught.value.code == "url_credentials"


# ------------------------------------------------------------------ key choice on an existing route


def test_a_route_of_the_config_file_can_be_given_a_key(editor, stack, tmp_path):
    with pytest.raises(ValidationError) as caught:
        editor.set_route_key("lan", "cloud_key", None)
    assert caught.value.code == "endpoint_ack_required"
    assert caught.value.extra == {"endpoint": "https://api.example.com"}
    out = editor.set_route_key("lan", "cloud_key", "https://api.example.com")
    assert out["changed"] and routes(stack)["lan"].api_key == "${secret:cloud_key}"
    assert saved(tmp_path)["keys"] == {"lan": "${secret:cloud_key}"}
    assert editor.set_route_key("lan", "cloud_key", None) == {"name": "lan", "changed": False}


def test_the_key_can_be_switched_and_removed(editor, stack, tmp_path):
    editor.set_route_key("lan", "cloud_key", "https://api.example.com")
    editor.set_route_key("lan", "other_key", "https://api.example.com")
    assert routes(stack)["lan"].api_key == "${secret:other_key}"
    out = editor.set_route_key("lan", None, None)  # no key needs no acknowledgement
    assert out["changed"] and routes(stack)["lan"].api_key == "none"
    assert saved(tmp_path)["keys"] == {"lan": "none"}
    assert "acks" not in saved(tmp_path)


def test_an_unknown_route_is_not_found(editor):
    with pytest.raises(SettingNotAllowed):
        editor.set_route_key("ghost", None, None)


def test_the_audit_names_the_endpoint_not_the_secret(editor, stack):
    editor.set_route_key("lan", "cloud_key", "https://api.example.com")
    keys = [a.key for a in stack.audit]
    assert "llm_routes.lan.endpoint" in keys
    assert "cloud_key" not in json.dumps([a.__dict__ for a in stack.audit], default=str)


# ------------------------------------------------------------------ endpoint moves after a key


def bound(tmp_path, stack, editor):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)


def test_a_move_of_a_keyed_route_still_needs_the_acknowledgement(editor, stack, tmp_path):
    bound(tmp_path, stack, editor)
    path = "llm_routes.cloud.base_url"
    with pytest.raises(ValidationError) as caught:
        editor.change(path, "https://other.example/v1")
    assert caught.value.code == "endpoint_ack_required"
    assert routes(stack)["cloud"].base_url == CLOUD["base_url"]
    editor.change(path, "https://other.example/v1", ack="https://other.example")
    assert routes(stack)["cloud"].base_url == "https://other.example/v1"


def test_a_path_only_edit_of_a_keyed_route_keeps_working(editor, stack, tmp_path):
    bound(tmp_path, stack, editor)
    editor.change("llm_routes.cloud.base_url", "https://api.cloudco.example/v2")
    editor.change("llm_routes.cloud.model", "bigger")
    assert routes(stack)["cloud"].base_url.endswith("/v2")
    assert routes(stack)["cloud"].api_key == "${secret:cloud_key}"


def test_a_keyed_route_of_the_config_file_takes_path_only_edits_too(editor, stack, tmp_path):
    editor.set_route_key("lan", "cloud_key", "https://api.example.com")
    editor.change("llm_routes.lan.base_url", "https://api.example.com/v9")
    editor.change("llm_routes.lan.model", "m2")
    assert routes(stack)["lan"].base_url.endswith("/v9")


# ------------------------------------------------------------------ restart


def test_the_keyed_route_survives_a_restart_twice(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    for _ in range(2):
        stack, editor = restarted(tmp_path)
        assert routes(stack)["cloud"].api_key == "${secret:cloud_key}"
        data = saved(tmp_path)
        assert data["keys"] == {"cloud": "${secret:cloud_key}"}
        assert any(k.endswith(".base_url") for k in data["acks"])


def test_a_path_only_edit_after_a_restart(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    stack, editor = restarted(tmp_path)
    editor.change("llm_routes.cloud.base_url", "https://api.cloudco.example/v2")
    editor.change("llm_routes.cloud.model", "bigger")
    stack, editor = restarted(tmp_path)
    route = routes(stack)["cloud"]
    assert route.base_url.endswith("/v2") and route.model == "bigger"
    assert route.api_key == "${secret:cloud_key}"


def test_an_acknowledged_move_survives_restarts(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    editor.change(
        "llm_routes.cloud.base_url", "https://other.example/v1", ack="https://other.example"
    )
    for _ in range(2):
        stack, editor = restarted(tmp_path)
        route = routes(stack)["cloud"]
        assert route.base_url == "https://other.example/v1"
        assert route.api_key == "${secret:cloud_key}"
        data = saved(tmp_path)
        assert data["keys"] == {"cloud": "${secret:cloud_key}"} and data["acks"]
    editor.change("llm_routes.cloud.base_url", "https://other.example/v3")  # path only
    assert routes(stack)["cloud"].base_url.endswith("/v3")


def test_a_key_of_a_config_file_route_survives_a_restart(tmp_path, editor, stack):
    editor.set_route_key("lan", "cloud_key", "https://api.example.com")
    stack, editor = restarted(tmp_path)
    assert routes(stack)["lan"].api_key == "${secret:cloud_key}"
    editor.change("llm_routes.lan.model", "m3")
    editor.set_route_key("lan", None)
    stack, editor = restarted(tmp_path)
    assert routes(stack)["lan"].api_key == "none"


def test_a_key_whose_acknowledgement_is_gone_is_dropped(tmp_path, editor, stack, caplog):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    data = saved(tmp_path)
    data.pop("acks")
    (tmp_path / FILE).write_text(json.dumps(data))
    (tmp_path / FILE).chmod(0o600)
    caplog.set_level(logging.WARNING)
    stack, editor = restarted(tmp_path)
    assert routes(stack)["cloud"].api_key == "none"
    assert "keys" not in saved(tmp_path)
    assert "cloud_key" not in caplog.text


def test_a_key_stored_for_another_origin_is_dropped(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    data = saved(tmp_path)
    for record in data["acks"].values():
        record["origin"] = "https://elsewhere.example"
    (tmp_path / FILE).write_text(json.dumps(data))
    (tmp_path / FILE).chmod(0o600)
    stack, editor = restarted(tmp_path)
    assert routes(stack)["cloud"].api_key == "none"


def test_a_key_swapped_in_the_file_is_dropped(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    data = saved(tmp_path)
    data["keys"]["cloud"] = "${secret:other_key}"
    (tmp_path / FILE).write_text(json.dumps(data))
    (tmp_path / FILE).chmod(0o600)
    stack, editor = restarted(tmp_path)
    assert routes(stack)["cloud"].api_key == "none"


@pytest.mark.parametrize("bad", ["sk-LITERAL-9", "${secret: x}", 5, None])
def test_a_literal_key_in_the_file_is_never_applied(tmp_path, editor, stack, bad):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    data = saved(tmp_path)
    data["keys"]["cloud"] = bad
    (tmp_path / FILE).write_text(json.dumps(data))
    (tmp_path / FILE).chmod(0o600)
    stack, editor = restarted(tmp_path)
    assert routes(stack)["cloud"].api_key == "none"


# ------------------------------------------------------------------ remove


def test_removing_a_keyed_route_forgets_its_key(tmp_path, editor, stack):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    assert editor.removable_routes() == ["cloud"]
    editor.remove_entry("llm_routes.cloud")
    assert "cloud" not in routes(stack)
    data = saved(tmp_path)
    assert "keys" not in data and "acks" not in data
    editor.add_route(CLOUD, None, None)  # the same name again starts without the old key
    stack, editor = restarted(tmp_path)
    assert routes(stack)["cloud"].api_key == "none"


def test_a_route_of_the_config_file_cannot_be_removed(editor):
    assert editor.removable_routes() == []
    with pytest.raises(ValidationError) as caught:
        editor.remove_entry("llm_routes.lan")
    assert caught.value.code == "entry_in_lower_layer"


# ------------------------------------------------------------------ router and secrets


def test_a_secret_set_from_the_web_is_picked_up_by_the_router(tmp_path):
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.llm.router import Router

    cfg = RoboEmotConfig(
        data_dir=str(tmp_path / "data"),
        llm_routes=[{**CLOUD, "api_key": "${secret:cloud_key}", "first_token_timeout_s": 5}],
    )
    store = SecretStore.for_data_dir(cfg.data_dir)
    admin = SecretsAdmin(lambda: cfg)
    admin.set("cloud_key", "sk-FIRST-1")
    router = Router(cfg.llm_routes, cfg.safety, secrets=store.get)
    assert router.key_problem(cfg.llm_routes[0]) is None and not router.secrets_changed()
    admin.set("cloud_key", "sk-SECOND-2")
    assert router.secrets_changed()


def test_a_key_stored_for_a_route_that_is_gone_is_dropped(tmp_path, editor, stack, caplog):
    editor.add_route(CLOUD, "cloud_key", ORIGIN)
    data = saved(tmp_path)
    data["overrides"] = {}  # the route itself was lost, its key record stayed
    (tmp_path / FILE).write_text(json.dumps(data))
    (tmp_path / FILE).chmod(0o600)
    caplog.set_level(logging.WARNING)
    stack, editor = restarted(tmp_path)
    assert "cloud" not in routes(stack)
    assert "keys" not in saved(tmp_path)
    assert "dropped: no such route" in caplog.text and "cloud_key" not in caplog.text
