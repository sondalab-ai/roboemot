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


# ------------------------------------------------------------ R3 SF6: where a web route may point

THIS_MACHINE = [
    "http://127.0.0.1:8000/v1",
    "http://127.1/v1",
    "http://0x7f000001/v1",
    "http://2130706433/v1",
    "http://localhost:8000/v1",
    "http://api.localhost/v1",
    "http://[::1]:8000/v1",
    "http://[::ffff:127.0.0.1]/v1",
    "http://0.0.0.0:8000/v1",
    "http://169.254.169.254/v1",
    "http://[fe80::1]/v1",
    "http://0/v1",
    "http://[::]/v1",
    "http://0.0.0.0./v1",
    "http://169.254.169.254./v1",
    "http://[::ffff:169.254.169.254]/v1",
    "http://017700000001/v1",
    "http://LOCALHOST./v1",
    "http://[::127.0.0.1]/v1",  # IPv4-compatible
    "http://[::ffff:0:7f00:1]/v1",  # IPv4-translated
    "http://[64:ff9b::7f00:1]/v1",  # NAT64
    "http://[64:ff9b::a9fe:a9fe]/v1",  # NAT64 of 169.254.169.254
]


@pytest.mark.parametrize("url", THIS_MACHINE)
def test_a_web_route_may_not_point_at_this_machine_or_link_local(web, stack, url):
    """SPEC 8, 12: a PIN holder cannot aim a route (and the check that posts to it) at the robot's
    own services or the link-local metadata range, in any spelling of the address."""
    added = post(web, "/api/routes/add", {**CLOUD, "name": "probe", "base_url": url})
    assert added.status == 422 and added.json["error"] == "url_local_refused"
    moved = post(web, "/api/settings", {"path": "llm_routes.lan.base_url", "value": url})
    assert moved.status == 422 and moved.json["error"] == "url_local_refused"
    assert [r.name for r in stack.effective().llm_routes] == ["lan"]
    assert stack.effective().llm_routes[0].base_url == LAN["base_url"]


def test_a_lan_route_from_the_web_stays_allowed(web, stack):
    added = post(
        web, "/api/routes/add", {**CLOUD, "name": "box", "base_url": "http://192.168.1.20:1234/v1"}
    )
    assert added.status == 200


def test_a_local_route_of_the_config_keeps_its_path_editable(tmp_path):
    """A route on this machine written in the YAML stays usable: a path edit at the same origin
    is not a new target."""
    stack = ConfigStack(
        app={
            "llm_routes": [
                {
                    "name": "here",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "model": "m",
                    "locality": "cloud",
                }
            ]
        }
    )
    editor = SettingsEditor(stack)
    done = editor.change("llm_routes.here.base_url", "http://127.0.0.1:11434/v2")
    assert done["changed"] is True


def test_a_web_route_on_this_machine_from_before_is_kept_with_a_warning(tmp_path, caplog):
    """A route the web added before the address rule is replayed (startup never fails on it) and
    a warning names the route, never the URL."""
    from roboemot.webui.settings import OverridesFile, entry_key

    file = tmp_path / "webui_settings.json"
    old = {
        "name": "old",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "m",
        "locality": "cloud",
    }
    OverridesFile(file).save({entry_key("llm_routes", {"name": "old"}): old}, {}, {})
    stack = ConfigStack(app={"data_dir": str(tmp_path / "data"), "llm_routes": [LAN]})
    caplog.set_level(logging.WARNING)
    SettingsEditor(stack, store=file).restore()
    assert [r.name for r in stack.effective().llm_routes] == ["lan", "old"]
    assert "old" in caplog.text and "this machine" in caplog.text
    assert "127.0.0.1" not in caplog.text


def test_user_info_in_front_of_a_local_host_is_refused_too(web, stack):
    added = post(
        web, "/api/routes/add", {**CLOUD, "name": "probe", "base_url": "http://evil@127.0.0.1/v1"}
    )
    assert added.status == 422 and added.json["error"] in {"url_credentials", "url_local_refused"}
    assert [r.name for r in stack.effective().llm_routes] == ["lan"]


@pytest.mark.parametrize(
    "url", ["https://x.example/v1?x=", "https://x.example/v1#f", "https://x.example/v1?"]
)
def test_a_web_route_has_no_query_or_fragment(web, stack, url):
    """The SDK appends its path to base_url, so a query or fragment would sit inside the request
    path the route check posts to (review ZG). The web UI refuses both, on add and on move."""
    added = post(web, "/api/routes/add", {**CLOUD, "name": "q", "base_url": url})
    assert added.status == 422 and added.json["error"] == "url_query"
    moved = post(web, "/api/settings", {"path": "llm_routes.lan.base_url", "value": url})
    assert moved.status == 422 and moved.json["error"] == "url_query"


def test_moving_a_local_lan_route_to_the_cloud_works_in_the_page_order(tmp_path):
    """Review ZG should-fix 1: the server refuses a new base_url while the route is local, and
    accepts it once the route is cloud. The page sends the locality change first (routes.test.mjs
    pins that order), so one save moves a LAN route to the cloud."""
    from roboemot.config.stack import GuardrailError

    box = {
        "name": "box",
        "base_url": "http://192.168.1.20:1234/v1",
        "model": "m",
        "locality": "local",
    }
    stack = ConfigStack(app={"llm_routes": [box]})
    editor = SettingsEditor(stack)
    with pytest.raises(GuardrailError):
        editor.change("llm_routes.box.base_url", "https://api.cloud.example/v1")
    assert editor.change("llm_routes.box.locality", "cloud")["changed"]
    assert editor.change("llm_routes.box.base_url", "https://api.cloud.example/v1")["changed"]
    route = stack.effective().llm_routes[0]
    assert (route.locality, route.base_url) == ("cloud", "https://api.cloud.example/v1")
