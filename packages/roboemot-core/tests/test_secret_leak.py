"""A planted API key never reaches logs, the debug report or the web UI (SPEC 8, 12)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics, FnSource
from roboemot.llm import Router
from roboemot.secret_store import SecretStore
from roboemot.webui import make_server
from roboemot.webui.routes import RouteChecker
from test_llm_conformance import Fake
from test_llm_router import collect
from test_webui import PIN, Client, FakeService
from test_webui_routes import finish

KEY = "sk-LEAK-CANARY-7f3a9c41"
REF = "${secret:cloud_key}"


def make_cfg(tmp_path, ref: str = REF) -> RoboEmotConfig:
    return RoboEmotConfig(
        data_dir=str(tmp_path / "data"),
        webui={"bind": "127.0.0.1", "port": 0},
        llm_routes=[
            {
                "name": "cloud",
                "base_url": "https://cloud.example/v1",
                "api_key": ref,
                "model": "m",
                "first_token_timeout_s": 5,
            }
        ],
    )


class Echo:
    """A server that echoes the Authorization header back in a failure body, as a hostile or
    sloppy proxy might; it fails first and answers properly afterwards."""

    def __init__(self) -> None:
        self.fake, self.fail = Fake(), True

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            return httpx.Response(
                401, json={"error": {"message": request.headers["authorization"]}}
            )
        return await self.fake.handler(request)


def serve(tmp_path, cfg, echo, report_extra):
    checker = RouteChecker(
        lambda: cfg,
        tmp_path / "data" / "route_checks.json",
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(echo.handler)),
    )
    diagnostics = Diagnostics().add(
        FnSource("config", lambda: cfg.model_dump(mode="json")),
        FnSource("extra", report_extra),
    )
    srv = make_server(FakeService(), diagnostics, cfg, tmp_path, routes=checker)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    return srv, client


def test_route_list_shows_the_reference_and_whether_it_is_set(tmp_path):
    cfg = make_cfg(tmp_path)
    srv, client = serve(tmp_path, cfg, Echo(), dict)
    try:
        entry = client.request("GET", "/api/routes").json["routes"][0]
        assert entry["secret"] == {"name": "cloud_key", "set": False}
        SecretStore.for_data_dir(cfg.data_dir).set("cloud_key", KEY)
        resp = client.request("GET", "/api/routes")
        assert resp.json["routes"][0]["secret"] == {"name": "cloud_key", "set": True}
        assert KEY.encode() not in resp.body
    finally:
        srv.stop()


def test_a_check_of_a_route_with_a_missing_secret_is_refused_and_sends_nothing(tmp_path):
    echo = Echo()
    srv, client = serve(tmp_path, make_cfg(tmp_path), echo, dict)
    try:
        resp = client.request("POST", "/api/routes/cloud/check", {})
        assert resp.status == 409 and resp.json == {"error": "route_secret"}
        assert echo.fake.bodies == []
    finally:
        srv.stop()


async def test_the_key_is_in_no_log_report_or_response(tmp_path, caplog):
    cfg = make_cfg(tmp_path)
    store = SecretStore.for_data_dir(cfg.data_dir)
    store.set("cloud_key", KEY)
    echo = Echo()
    router = Router(
        cfg.llm_routes,
        cfg.safety,
        secrets=store.get,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(echo.handler)),
    )
    seen: list[bytes] = []
    with caplog.at_level(logging.DEBUG):
        # The route fails with the key echoed in the server's error body.
        with pytest.raises(Exception) as err:
            await collect(router)
        seen.append(f"{err.value!r} {router.failures!r} {router!r}".encode())

        srv, client = serve(
            tmp_path, cfg, echo, lambda: {"router": repr(router), "failures": router.failures}
        )
        try:
            seen.append(client.request("GET", "/api/routes").body)
            assert client.request("POST", "/api/routes/cloud/check", {}).status == 202
            seen.append(json.dumps(finish(client)).encode())  # the failed check
            echo.fail = False
            assert client.request("POST", "/api/routes/cloud/check", {}).status == 202
            entry = finish(client)
            assert entry["last_check"]["ok"] is True  # the key worked: it was sent, not leaked
            seen += [json.dumps(entry).encode(), client.request("GET", "/api/debug").body]
            seen.append(client.request("GET", "/api/routes").body)
        finally:
            srv.stop()
    assert echo.fake.auth and set(echo.fake.auth) == {f"Bearer {KEY}"}
    seen += [caplog.text.encode(), (tmp_path / "data" / "route_checks.json").read_bytes()]
    for blob in seen:
        assert KEY.encode() not in blob


def test_settings_api_never_shows_a_reference_or_url_credentials(tmp_path):
    from roboemot.config.stack import ConfigStack
    from roboemot.webui.settings import SettingsEditor

    password = "p4ss-CANARY-91"
    literal = "sk-CANARY-literal-55"
    route = {
        "name": "cloud",
        "base_url": f"https://alice:{password}@cloud.example/v1",
        "api_key": REF,
        "model": "m",
        "first_token_timeout_s": 5,
    }
    token_route = {
        "name": "tokenised",
        "base_url": "https://t.example/v1?api_key=q-CANARY-1&x=1",
        "api_key": literal,
        "model": "m",
        "first_token_timeout_s": 5,
    }
    fragment_route = {
        "name": "fragment",
        "base_url": "https://f.example/v1#access_token=f-CANARY-2",
        "api_key": REF,
        "model": "m",
        "first_token_timeout_s": 5,
    }
    data_dir = tmp_path / "data"
    stack = ConfigStack(
        app={
            "data_dir": str(data_dir),
            "webui": {"bind": "127.0.0.1", "port": 0},
            "llm_routes": [route, token_route, fragment_route],
        },
        audit_path=tmp_path / "audit.jsonl",
    )
    SecretStore.for_data_dir(str(data_dir)).set("cloud_key", KEY)
    stack.set("llm_routes.cloud.model", "m2", source="t")
    stack.set("voice.wake_word", "${secret:cloud_key}", source="t", admin=True)
    stack.set("circadian.location", {"lat": 45.4642, "lon": 9.1900}, source="t")
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    canaries = [password, "alice", "cloud_key", KEY, literal, "q-CANARY-1", "f-CANARY-2", "45.4642"]
    try:
        assert client.login().status == 200
        resp = client.request("GET", "/api/settings")
        assert resp.status == 200
        by_path = {f["path"]: f for f in resp.json["fields"]}
        for leaf in ("api_key", "base_url"):
            row = by_path[f"llm_routes.cloud.{leaf}"]
            assert row["secret"] is True and row["writable"] is False
            assert row["value"] == "***" and row["default"] == "***"
        for name in ("tokenised", "fragment"):
            assert by_path[f"llm_routes.{name}.base_url"]["value"] == "***"
        assert by_path["llm_routes.tokenised.api_key"]["secret"] is True
        assert by_path["voice.wake_word"]["secret"] and by_path["voice.wake_word"]["value"] == "***"
        loc = by_path["circadian.location"]
        assert loc["private"] and loc["value"] == "***" and not loc["secret"]
        assert by_path["llm_routes.cloud.model"]["value"] == "m2"
        assert by_path["llm_routes"]["value"] == ["cloud", "tokenised", "fragment"]
        paths_and_keys = json.dumps(
            [
                [f["path"], f["value"] if f["kind"] in ("keyed_list", "dict_of_model") else 0]
                for f in resp.json["fields"]
            ]
        )
        legacy = json.dumps(resp.json["settings"])
        for blob in (
            resp.body,
            json.dumps(editor.fields()).encode(),
            paths_and_keys.encode(),
            legacy.encode(),
        ):
            for canary in canaries:
                assert canary.encode() not in blob
        for path in ("llm_routes.cloud.api_key", "llm_routes.cloud.base_url"):
            posted = client.request(
                "POST", "/api/settings", {"path": path, "value": "x", "pin": PIN}
            )
            assert posted.status == 403 and password.encode() not in posted.body
        leaked = client.request(
            "POST", "/api/settings", {"path": "voice.wake_word", "value": REF, "pin": PIN}
        )
        assert leaked.status == 422 or leaked.status == 400
        assert b"cloud_key" not in leaked.body
    finally:
        srv.stop()
