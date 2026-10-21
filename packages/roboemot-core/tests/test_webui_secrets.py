"""Write-only secret entry in the web UI (SPEC 8 Secrets, 11A, 12): names listed, values never
returned, logged or echoed; set and remove need the PIN, CSRF and the same origin."""

from __future__ import annotations

import json
import logging
import stat

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.secret_store import MAX_VALUE_CHARS, SecretStore
from roboemot.webui import make_server
from roboemot.webui.secrets_admin import SecretsAdmin
from test_webui import PIN, Client, FakeService

CANARY = "sk-WEB-CANARY-91c4e7a2"
OLD = "sk-OLD-CANARY-5b2d"
ROUTE = {
    "name": "cloud",
    "base_url": "https://api.example.com/v1",
    "model": "m",
    "first_token_timeout_s": 5,
    "api_key": "${secret:cloud_key}",
}


@pytest.fixture
def data(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def store(data):
    return SecretStore.for_data_dir(str(data))


@pytest.fixture
def web(tmp_path, data, store):
    store.set("cloud_key", OLD)
    store.set("spare", "sk-SPARE-1")
    cfg = RoboEmotConfig(
        data_dir=str(data),
        llm_routes=[ROUTE],
        webui={"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3},
    )
    srv = make_server(
        FakeService(), Diagnostics(), cfg, tmp_path, secrets=SecretsAdmin(lambda: cfg)
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    yield client
    srv.stop()


def post(client, path, body, pin=PIN, **kw):
    return client.request("POST", path, {**body, "pin": pin}, **kw)


def test_list_has_names_and_users_and_nothing_else(web):
    resp = web.request("GET", "/api/secrets")
    assert resp.status == 200
    assert resp.json == {
        "secrets": [
            {"name": "cloud_key", "set": True, "used_by": ["cloud"]},
            {"name": "spare", "set": True, "used_by": []},
        ],
        "max_value_chars": MAX_VALUE_CHARS,
    }
    assert OLD.encode() not in resp.body and b"SPARE" not in resp.body


def test_list_shows_a_referenced_name_the_store_lacks(tmp_path, data):
    cfg = RoboEmotConfig(data_dir=str(data), llm_routes=[ROUTE])
    rows = SecretsAdmin(lambda: cfg).listing()
    assert rows == [{"name": "cloud_key", "set": False, "used_by": ["cloud"]}]


def test_set_adds_and_replaces_without_echo(web, store, caplog):
    caplog.set_level(logging.DEBUG)
    resp = post(web, "/api/secrets/set", {"name": "new_key", "value": CANARY})
    assert resp.status == 200 and resp.json == {"name": "new_key", "set": True}
    assert store.get("new_key") == CANARY
    resp = post(web, "/api/secrets/set", {"name": "cloud_key", "value": CANARY})
    assert resp.status == 200 and store.get("cloud_key") == CANARY
    listing = web.request("GET", "/api/secrets")
    for body in (resp.body, listing.body):
        assert CANARY.encode() not in body and OLD.encode() not in body
    assert CANARY not in caplog.text and OLD not in caplog.text


def test_set_strips_a_pasted_newline(web, store):
    assert post(web, "/api/secrets/set", {"name": "k2", "value": f"  {CANARY}\n"}).status == 200
    assert store.get("k2") == CANARY


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "a b",
        "ab\ncd",
        "tab\tx",
        "héllo",
        "x\x00y",
        "z" * (MAX_VALUE_CHARS + 1),
        5,
        None,
        ["a"],
    ],
)
def test_bad_values_are_refused_without_echo(web, store, caplog, value):
    resp = post(web, "/api/secrets/set", {"name": "k3", "value": value})
    assert resp.status == 422 and resp.json == {"error": "secret_value_invalid", "field": "value"}
    assert not store.has("k3")
    assert "zzzz" not in resp.body.decode() and "zzzz" not in caplog.text


@pytest.mark.parametrize("name", ["", "-lead", "a b", "a/b", "x" * 0, 5, None, "../x"])
def test_bad_names_are_refused(web, store, name):
    resp = post(web, "/api/secrets/set", {"name": name, "value": CANARY})
    assert resp.status == 422 and resp.json["error"] == "secret_name_invalid"
    assert CANARY.encode() not in resp.body


def test_a_wrong_pin_stores_nothing_and_echoes_nothing(web, store):
    resp = post(web, "/api/secrets/set", {"name": "k4", "value": CANARY}, pin="000000")
    assert resp.status == 401 and resp.json == {"error": "bad_pin"}
    assert CANARY.encode() not in resp.body and not store.has("k4")


def test_a_wrong_pin_removes_nothing(web, store):
    resp = post(web, "/api/secrets/remove", {"name": "spare"}, pin="000000")
    assert resp.status == 401 and resp.json == {"error": "bad_pin"}
    assert store.has("spare")


def test_pin_attempts_share_the_login_limit(web, store):
    for _ in range(3):
        post(web, "/api/secrets/set", {"name": "k5", "value": CANARY}, pin="000000")
    resp = post(web, "/api/secrets/set", {"name": "k5", "value": CANARY})
    assert resp.status == 429 and not store.has("k5")


def test_csrf_origin_and_login_are_required(web, store, tmp_path):
    body = {"name": "k6", "value": CANARY}
    assert post(web, "/api/secrets/set", body, csrf=False).status == 403
    bad = post(web, "/api/secrets/set", body, headers={"Origin": "http://evil.example"})
    assert bad.status == 403 and bad.json == {"error": "origin"}
    assert not store.has("k6")
    anon = Client(web.port)
    for method, path in (
        ("GET", "/api/secrets"),
        ("POST", "/api/secrets/set"),
        ("POST", "/api/secrets/remove"),
    ):
        assert anon.request(method, path, {} if method == "POST" else None).status == 401


def test_unknown_fields_and_missing_fields_are_refused(web, store):
    assert post(web, "/api/secrets/set", {"name": "k7", "value": CANARY, "x": 1}).status == 422
    assert post(web, "/api/secrets/set", {"name": "k7"}).status == 400
    assert post(web, "/api/secrets/remove", {}).status == 400


def test_remove_deletes_an_unused_secret(web, store):
    resp = post(web, "/api/secrets/remove", {"name": "spare"})
    assert resp.status == 200 and resp.json == {"name": "spare", "set": False}
    assert not store.has("spare")
    assert post(web, "/api/secrets/remove", {"name": "spare"}).status == 404


def test_remove_of_a_secret_a_route_uses_is_refused(web, store):
    resp = post(web, "/api/secrets/remove", {"name": "cloud_key"})
    assert resp.status == 409 and resp.json == {"error": "secret_in_use"}
    assert store.get("cloud_key") == OLD


def test_the_store_file_stays_private(web, store):
    post(web, "/api/secrets/set", {"name": "k8", "value": CANARY})
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_an_unsafe_store_is_reported_without_its_path_or_content(web, store, caplog):
    store.path.chmod(0o644)
    listing = web.request("GET", "/api/secrets")
    assert listing.status == 409 and listing.json == {"error": "secret_store"}
    resp = post(web, "/api/secrets/set", {"name": "k9", "value": CANARY})
    assert resp.status == 409 and resp.json == {"error": "secret_store"}
    assert str(store.path).encode() not in resp.body
    assert CANARY not in caplog.text and OLD not in caplog.text


def test_the_value_is_in_no_other_response(web, store):
    post(web, "/api/secrets/set", {"name": "k10", "value": CANARY})
    for path in ("/api/debug", "/api/routes", "/api/settings", "/api/session", "/api/strings"):
        resp = web.request("GET", path)
        assert CANARY.encode() not in resp.body, path


def test_without_a_secrets_port_the_api_says_so(tmp_path):
    srv = make_server(
        FakeService(),
        Diagnostics(),
        RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0}),
        tmp_path,
    )
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/secrets").json == {"error": "secrets_unavailable"}
        assert json.dumps(post(client, "/api/secrets/set", {"name": "a", "value": "b"}).json)
    finally:
        srv.stop()
