"""Route presets of the Models tab (SPEC 8, 11A, 12): core's Gemini Live preset from data, the
robot platform's own route from its settings plugin, and none whose route already exists, so a
route the installer wrote is never copied over from the web UI. Plus the per-route
``connect_timeout_s`` on add and edit."""

from __future__ import annotations

import logging

import pytest
from roboemot.config.schema import CONNECT_TIMEOUT_MAX_S, CONNECT_TIMEOUT_MIN_S, LLMRoute
from roboemot.config.stack import ConfigStack
from roboemot.diagnostics import Diagnostics
from roboemot.secret_store import SecretStore
from roboemot.webui import make_server
from roboemot.webui import route_presets as rp
from roboemot.webui.robot_section import RoutePreset, RoutePresetted, platform_presets
from roboemot.webui.routes import RouteChecker
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.strings import STRINGS
from test_compat_fixtures import app_stack, platform_route, robot_data, start
from test_webui import PIN, Client, FakeService

KEY = "sk-PRESET-CANARY-5c21"
GEMINI = "gemini_live"
PICTURES = "gemini_pictures"
PICTURES_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_ORIGIN = "wss://generativelanguage.googleapis.com"
TIMEOUT = "connect_timeout_s"


def gemini_fields() -> dict:
    (raw,) = [p for p in rp.core_presets() if p["id"] == GEMINI]
    return dict(raw["fields"])


@pytest.fixture
def stack(tmp_path):
    cfg = {
        "data_dir": str(tmp_path / "data"),
        "webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3},
    }
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def web(tmp_path, stack):
    srv, client = serve(tmp_path, stack)
    yield client
    srv.stop()


def serve(tmp_path, stack, plugin=None):
    """The web UI over ``stack`` with ``gemini_key`` stored and ``plugin`` as the robot's
    settings plugin, logged in. Returns the server (stop it) and the client."""
    SecretStore.for_data_dir(stack.effective().data_dir).set("gemini_key", KEY)
    admin = SecretsAdmin(stack.effective)
    editor = SettingsEditor(
        stack, store=tmp_path / "webui_settings.json", known_secrets=admin.names, plugin=plugin
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
    return srv, client


def post(client, path, body, pin=PIN):
    return client.request("POST", path, {**body, "pin": pin})


def listing(client) -> dict:
    return client.request("GET", "/api/routes").json


# ------------------------------------------------------------------ the data file


def test_every_core_preset_is_a_valid_route_with_its_key_reference():
    for raw in rp.core_presets():
        preset = rp.labelled(raw, "en")
        assert set(preset.fields) <= rp.PRESET_FIELDS
        LLMRoute.model_validate({**preset.fields, "api_key": f"${{secret:{preset.secret}}}"})


def test_every_core_preset_is_labelled_in_every_page_language():
    for raw in rp.core_presets():
        assert set(raw["label"]) == set(STRINGS), raw["id"]


def test_the_gemini_preset_is_the_documented_recipe():
    preset = rp.labelled(next(p for p in rp.core_presets() if p["id"] == GEMINI), "it")
    assert preset.secret == "gemini_key"
    assert preset.fields == {
        "name": "gemini",
        "kind": "realtime",
        "protocol": "gemini_live",
        "base_url": GEMINI_ORIGIN
        + "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        "model": "gemini-3.1-flash-live-preview",
        "locality": "cloud",
        TIMEOUT: 15.0,
    }


def test_the_gemini_pictures_preset_is_a_chat_route_on_the_same_key():
    preset = rp.labelled(next(p for p in rp.core_presets() if p["id"] == PICTURES), "en")
    assert preset.secret == "gemini_key"
    assert preset.fields == {
        "name": "gemini_pictures",
        "kind": "chat",
        "base_url": PICTURES_URL,
        "model": "gemini-3.8-flash",
        "locality": "cloud",
    }
    assert "images" not in preset.fields  # the switch stays the owner's, behind the PIN


# ------------------------------------------------------------------ the list


def test_the_list_returns_the_presets_and_the_timeout_range(web):
    data = listing(web)
    assert [p["id"] for p in data["presets"]] == [GEMINI, PICTURES]
    preset = data["presets"][0]
    assert set(preset) == {"id", "label", "fields", "secret"}
    assert (preset["id"], preset["secret"], preset["fields"]) == (
        GEMINI,
        "gemini_key",
        gemini_fields(),
    )
    assert data[TIMEOUT] == {"min": CONNECT_TIMEOUT_MIN_S, "max": CONNECT_TIMEOUT_MAX_S}


def test_a_preset_is_hidden_when_a_route_of_its_name_exists(tmp_path):
    other = {"name": "gemini", "base_url": "https://lan.example/v1", "model": "m"}
    stack = ConfigStack(app={"llm_routes": [other]}, audit_path=tmp_path / "a.jsonl")
    offered = SettingsEditor(stack, store=tmp_path / "w.json").route_presets()
    assert [p["id"] for p in offered] == [PICTURES]


def test_the_installed_platform_route_hides_its_preset_and_stays_as_installed():
    robot_data("cloud_voice")
    admin, cfg, _ = start(app_stack())
    assert admin is not None
    presets = admin.settings.route_presets("en")
    assert [p["id"] for p in presets] == [GEMINI, PICTURES]
    assert platform_route() in {r.name for r in cfg.llm_routes}


def test_a_robot_without_the_platform_route_is_offered_it_first():
    robot_data("local_voice")
    admin, _, _ = start(app_stack())
    assert admin is not None
    presets = admin.settings.route_presets("en")
    assert [p["fields"]["name"] for p in presets] == [platform_route(), "gemini", PICTURES]


def test_a_robot_with_both_voice_routes_is_offered_only_the_pictures_preset():
    robot_data("gemini_switch")
    admin, _, _ = start(app_stack())
    assert admin is not None
    assert [p["id"] for p in admin.settings.route_presets("en")] == [PICTURES]


# ------------------------------------------------------------------ adding from a preset


def test_an_add_from_the_gemini_preset_passes_the_endpoint_acknowledgement(web, stack, caplog):
    caplog.set_level(logging.DEBUG)
    preset = next(p for p in listing(web)["presets"] if p["id"] == GEMINI)
    body = {**preset["fields"], "secret": preset["secret"]}
    first = post(web, "/api/routes/add", body)
    assert first.status == 422
    assert first.json == {
        "error": "endpoint_ack_required",
        "field": "value",
        "endpoint": GEMINI_ORIGIN,
    }
    done = post(web, "/api/routes/add", {**body, "ack_endpoint": GEMINI_ORIGIN})
    assert done.status == 200 and done.json["changed"] is True
    route = next(r for r in stack.effective().llm_routes if r.name == "gemini")
    assert route.model_dump(include=set(preset["fields"])) == preset["fields"]
    assert route.api_key == "${secret:gemini_key}"
    data = listing(web)
    assert [p["id"] for p in data["presets"]] == [PICTURES]
    row = next(r for r in data["routes"] if r["name"] == "gemini")
    assert row[TIMEOUT] == 15.0 and row["protocol"] == "gemini_live"
    for text in (first.body, done.body, str(data).encode(), caplog.text.encode()):
        assert KEY.encode() not in text


def test_an_add_from_the_pictures_preset_gives_a_chat_route_with_the_switch_off(web, stack):
    preset = next(p for p in listing(web)["presets"] if p["id"] == PICTURES)
    body = {**preset["fields"], "secret": preset["secret"]}
    origin = "https://generativelanguage.googleapis.com"
    done = post(web, "/api/routes/add", {**body, "ack_endpoint": origin})
    assert done.status == 200, done.json
    route = next(r for r in stack.effective().llm_routes if r.name == PICTURES)
    assert (route.kind, route.images, route.api_key) == ("chat", False, "${secret:gemini_key}")
    row = next(r for r in listing(web)["routes"] if r["name"] == PICTURES)
    assert row["images"] is False and row["images_state"] == "off"


# ------------------------------------------------------------------ connect_timeout_s


def test_the_connect_timeout_round_trips_on_add_and_edit(web, stack):
    path = f"llm_routes.gemini.{TIMEOUT}"
    added = post(web, "/api/routes/add", {**gemini_fields(), TIMEOUT: 12.5})
    assert added.status == 200
    assert next(r for r in listing(web)["routes"] if r["name"] == "gemini")[TIMEOUT] == 12.5
    assert post(web, "/api/settings", {"path": path, "value": 20}).status == 200
    assert (
        next(r for r in stack.effective().llm_routes if r.name == "gemini").connect_timeout_s == 20
    )
    assert post(web, "/api/settings", {"path": path, "value": None}).status == 200
    row = next(r for r in listing(web)["routes"] if r["name"] == "gemini")
    assert TIMEOUT not in row  # null: the shared value


@pytest.mark.parametrize("value", [CONNECT_TIMEOUT_MIN_S / 2, CONNECT_TIMEOUT_MAX_S + 1])
def test_an_out_of_range_connect_timeout_is_refused(web, stack, value):
    refused = post(web, "/api/routes/add", {**gemini_fields(), TIMEOUT: value})
    assert refused.status == 422 and refused.json["error"] == "value_range"
    assert "gemini" not in {r.name for r in stack.effective().llm_routes}
    assert post(web, "/api/routes/add", gemini_fields()).status == 200
    edit = post(web, "/api/settings", {"path": f"llm_routes.gemini.{TIMEOUT}", "value": value})
    assert edit.status == 422 and edit.json["error"] == "value_range"


# ------------------------------------------------------------------ the plugin contract


class Offers:
    """A settings plugin whose ``route_presets`` returns ``offered`` (or raises it)."""

    def __init__(self, offered: object) -> None:
        self.offered = offered

    def section(self, stack, manifest):
        raise NotImplementedError

    def route_presets(self, language: str) -> list[RoutePreset]:
        if isinstance(self.offered, Exception):
            raise self.offered
        return self.offered  # type: ignore[return-value]


PLATFORM = RoutePreset(
    "platform",
    "Platform server",
    {"name": "platform", "kind": "realtime", "base_url": "https://alloc.example/s", "model": "s2s"},
    "platform_token",
)


def test_a_plugin_without_the_method_offers_none():
    class Bare:
        def section(self, stack, manifest):
            raise NotImplementedError

    assert not isinstance(Bare(), RoutePresetted)
    assert platform_presets(Bare(), "en") == [] and platform_presets(None, "en") == []


def test_the_platform_preset_comes_first_and_a_broken_plugin_gives_none(caplog):
    offered = rp.route_presets(set(), [PLATFORM], "en")
    assert [p["id"] for p in offered] == ["platform", GEMINI, PICTURES]
    caplog.set_level(logging.WARNING)
    assert platform_presets(Offers(RuntimeError(KEY)), "en") == []
    assert platform_presets(Offers([{"id": "x"}, PLATFORM]), "en") == [PLATFORM]
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        RoutePreset("k", "Keyed", {**PLATFORM.fields, "api_key": KEY}, "platform_token"),
        RoutePreset("u", "Bad URL", {**PLATFORM.fields, "base_url": "ftp://x"}, "platform_token"),
        RoutePreset("s", "Bad secret", dict(PLATFORM.fields), "../etc"),
        RoutePreset(GEMINI, "Same id", {**PLATFORM.fields, "name": "other"}, "platform_token"),
        RoutePreset("n", "List name", {**PLATFORM.fields, "name": ["a"]}, "platform_token"),
        RoutePreset("m", "No name", {"kind": "realtime", "model": "m"}, "platform_token"),
        RoutePreset("d", "Dict value", {**PLATFORM.fields, "model": {"k": KEY}}, "platform_token"),
        RoutePreset(
            "t",
            "Chat timeout",
            {**PLATFORM.fields, "kind": "chat", "connect_timeout_s": 5.0},
            "platform_token",
        ),
        RoutePreset(["unhashable"], "Bad id", dict(PLATFORM.fields), "platform_token"),  # type: ignore[arg-type]
    ],
)
def test_a_preset_with_a_key_an_invalid_route_or_a_taken_id_is_dropped(bad, caplog):
    caplog.set_level(logging.WARNING)
    offered = rp.route_presets(set(), [bad], "en")
    assert [(p["id"], p["fields"]["name"]) for p in offered] == [
        (GEMINI, "gemini"),
        (PICTURES, PICTURES),
    ]
    assert KEY not in caplog.text


def test_a_malformed_plugin_preset_still_lists_the_core_presets(tmp_path, stack, caplog):
    caplog.set_level(logging.WARNING)
    bad = RoutePreset("bad", "Bad", {**PLATFORM.fields, "name": ["a"]}, "platform_token")
    srv, client = serve(tmp_path, stack, plugin=Offers([bad]))
    try:
        reply = client.request("GET", "/api/routes")
    finally:
        srv.stop()
    assert reply.status == 200
    assert [p["id"] for p in reply.json["presets"]] == [GEMINI, PICTURES]


@pytest.mark.parametrize("installed", ["pollen", "gemini"])
def test_an_add_from_a_stale_page_never_shadows_a_route_of_the_config(tmp_path, installed):
    route = {**gemini_fields(), "name": installed}
    stack = ConfigStack(
        app={
            "data_dir": str(tmp_path / "data"),
            "webui": {"bind": "127.0.0.1", "port": 0, "pin_max_attempts": 3},
            "llm_routes": [route],
        },
        audit_path=tmp_path / "audit.jsonl",
    )
    srv, client = serve(tmp_path, stack)
    try:
        refused = post(client, "/api/routes/add", {**gemini_fields(), "name": installed})
    finally:
        srv.stop()
    assert refused.json == {"error": "entry_exists", "field": "value"}, refused.json
    assert refused.status == 422
    (kept,) = [r for r in stack.effective().llm_routes if r.name == installed]
    assert kept.model_dump(include=set(route)) == route
