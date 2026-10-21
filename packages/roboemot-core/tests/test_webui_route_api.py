"""The route endpoints of the web UI: add, key, remove, list (SPEC 8, 11A, 12). PIN, CSRF and
origin are required; no response and no log line holds a key; a literal key is refused."""

from __future__ import annotations

import logging

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.diagnostics import Diagnostics
from roboemot.secret_store import SecretStore
from roboemot.webui import make_server
from roboemot.webui.routes import RouteChecker
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, Client, FakeService

KEY = "sk-API-CANARY-0e51"
CLOUD = {
    "name": "cloud",
    "base_url": "https://api.cloudco.example/v1",
    "model": "big",
    "locality": "cloud",
}
ORIGIN = "https://api.cloudco.example"
LAN = {"name": "lan", "base_url": "https://api.example.com/v1", "model": "m"}


@pytest.fixture
def stack(tmp_path):
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3},
        "llm_routes": [LAN],
    }
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def store(stack):
    store = SecretStore.for_data_dir(stack.effective().data_dir)
    store.set("cloud_key", KEY)
    return store


@pytest.fixture
def web(tmp_path, stack, store):
    admin = SecretsAdmin(stack.effective)
    editor = SettingsEditor(
        stack, store=tmp_path / "webui_settings.json", known_secrets=admin.names
    )
    srv = make_server(
        FakeService(),
        Diagnostics(),
        stack.effective,
        tmp_path,
        settings=editor,
        secrets=admin,
        routes=RouteChecker(stack.effective),
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    yield client
    srv.stop()


def post(client, path, body, pin=PIN, **kw):
    return client.request("POST", path, {**body, "pin": pin}, **kw)


def names(client):
    return [r["name"] for r in client.request("GET", "/api/routes").json["routes"]]


def test_add_a_route_with_a_key_through_the_acknowledgement(web, stack, caplog):
    caplog.set_level(logging.DEBUG)
    body = {**CLOUD, "secret": "cloud_key"}
    first = post(web, "/api/routes/add", body)
    assert first.status == 422
    assert first.json == {
        "error": "endpoint_ack_required",
        "field": "value",
        "endpoint": ORIGIN,
    }
    done = post(web, "/api/routes/add", {**body, "ack_endpoint": ORIGIN})
    assert done.status == 200 and done.json["changed"] is True
    listing = web.request("GET", "/api/routes").json["routes"]
    cloud = next(r for r in listing if r["name"] == "cloud")
    assert cloud["key"] == "secret" and cloud["secret"] == {"name": "cloud_key", "set": True}
    assert cloud["removable"] is True and cloud["locality"] == "cloud"
    assert cloud["base_url"] == CLOUD["base_url"]
    assert next(r for r in listing if r["name"] == "lan")["removable"] is False
    for text in (first.body, done.body, str(listing).encode(), caplog.text.encode()):
        assert KEY.encode() not in text


def test_a_literal_key_in_any_field_is_refused_and_never_echoed(web, stack, caplog):
    caplog.set_level(logging.DEBUG)
    for body in (
        {**CLOUD, "api_key": KEY},
        {**CLOUD, "secret": KEY, "ack_endpoint": ORIGIN},
        {**CLOUD, "model": "${secret:cloud_key}"},
        {**CLOUD, "secret": "${secret:cloud_key}", "ack_endpoint": ORIGIN},
    ):
        resp = post(web, "/api/routes/add", body)
        assert resp.status in (403, 422), body
        assert KEY.encode() not in resp.body
    assert names(web) == ["lan"]
    assert KEY not in caplog.text


def test_only_an_existing_secret_can_be_chosen(web):
    resp = post(web, "/api/routes/add", {**CLOUD, "secret": "ghost", "ack_endpoint": ORIGIN})
    assert resp.status == 422 and resp.json["error"] == "secret_unknown"
    assert names(web) == ["lan"]


def test_choose_the_key_of_an_existing_route(web, stack):
    body = {"name": "lan", "secret": "cloud_key"}
    need = post(web, "/api/routes/key", body)
    assert need.json["error"] == "endpoint_ack_required"
    assert need.json["endpoint"] == "https://api.example.com"
    ok = post(web, "/api/routes/key", {**body, "ack_endpoint": need.json["endpoint"]})
    assert ok.status == 200
    assert stack.effective().llm_routes[0].api_key == "${secret:cloud_key}"
    assert post(web, "/api/routes/key", {"name": "lan", "secret": None}).status == 200
    assert stack.effective().llm_routes[0].api_key == "none"
    assert post(web, "/api/routes/key", {"name": "ghost", "secret": None}).status == 403


def test_remove_a_route_added_here_and_not_one_of_the_config(web):
    assert post(web, "/api/routes/add", CLOUD).status == 200
    assert post(web, "/api/routes/remove", {"name": "cloud"}).status == 200
    assert names(web) == ["lan"]
    lower = post(web, "/api/routes/remove", {"name": "lan"})
    assert lower.status == 422 and lower.json["error"] == "entry_in_lower_layer"
    assert post(web, "/api/routes/remove", {"name": "bad/name"}).status == 400


def test_the_secret_in_use_cannot_be_removed_but_can_once_the_route_lets_go(web):
    post(web, "/api/routes/add", {**CLOUD, "secret": "cloud_key", "ack_endpoint": ORIGIN})
    assert post(web, "/api/secrets/remove", {"name": "cloud_key"}).json == {
        "error": "secret_in_use"
    }
    assert post(web, "/api/routes/key", {"name": "cloud", "secret": None}).status == 200
    assert post(web, "/api/secrets/remove", {"name": "cloud_key"}).status == 200


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/routes/add", CLOUD),
        ("/api/routes/key", {"name": "lan", "secret": None}),
        ("/api/routes/remove", {"name": "lan"}),
    ],
)
def test_pin_csrf_origin_and_login_are_required(web, stack, path, body):
    wrong = post(web, path, body, pin="000000")
    assert wrong.status == 401 and wrong.json == {"error": "bad_pin"}
    assert post(web, path, body, csrf=False).status == 403
    evil = post(web, path, body, headers={"Origin": "http://evil.example"})
    assert evil.status == 403 and evil.json == {"error": "origin"}
    assert Client(web.port).request("POST", path, body).status == 401
    assert names(web) == ["lan"]
    assert stack.effective().llm_routes[0].api_key == "none"


def test_unknown_fields_are_refused(web):
    assert post(web, "/api/routes/add", {**CLOUD, "daily_token_budget": 5}).status == 422
    assert post(web, "/api/routes/add", {"name": "x"}).status == 400


def test_an_unsafe_secret_store_is_a_bare_code(web, store):
    store.path.chmod(0o644)
    resp = post(web, "/api/routes/add", {**CLOUD, "secret": "cloud_key", "ack_endpoint": ORIGIN})
    assert resp.status == 409 and resp.json == {"error": "secret_store"}
    assert str(store.path).encode() not in resp.body
