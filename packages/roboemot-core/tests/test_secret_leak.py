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
            assert row["secret"] is True and row["writable"] is (leaf == "base_url")  # write-only
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
        for blob in (
            resp.body,
            json.dumps(editor.fields()).encode(),
            paths_and_keys.encode(),
        ):
            for canary in canaries:
                assert canary.encode() not in blob
        posted = client.request(
            "POST", "/api/settings", {"path": "llm_routes.cloud.api_key", "value": "x", "pin": PIN}
        )
        assert posted.status == 403 and password.encode() not in posted.body
        for bad in (
            f"https://alice:{password}@cloud.example/v2",
            "https://alice@cloud.example/v2",
            "https://cloud.example/v2?api_key=q-CANARY-9",
            REF,
        ):
            posted = client.request(
                "POST",
                "/api/settings",
                {"path": "llm_routes.cloud.base_url", "value": bad, "pin": PIN},
            )
            assert posted.status == 422 and posted.json["error"] == "url_credentials"
            assert b"CANARY" not in posted.body and b"alice" not in posted.body
        entry = {"name": "n2", "base_url": f"https://u:{password}@h.example/v1", "model": "m"}
        added = client.request(
            "POST", "/api/settings/entry", {"path": "llm_routes", "value": entry, "pin": PIN}
        )
        assert added.status == 422 and added.json["error"] == "url_credentials"
        assert password.encode() not in added.body
        replaced = client.request(
            "POST",
            "/api/settings",
            {"path": "llm_routes.cloud.base_url", "value": "https://cloud.example/v2", "pin": PIN},
        )
        assert replaced.status == 200 and replaced.json["changed"] is True
        assert password.encode() not in replaced.body
        again = client.request("GET", "/api/settings")
        row = {f["path"]: f for f in again.json["fields"]}["llm_routes.cloud.base_url"]
        assert (
            row["secret"] is True and row["value"] == "***"
        )  # the lower layer still holds credentials
        for canary in canaries:
            assert canary.encode() not in again.body
        reset = client.request(
            "POST", "/api/settings/reset", {"path": "llm_routes.cloud.base_url", "pin": PIN}
        )
        assert (
            reset.status == 200 and reset.json["value"] == "***"
        )  # the lower layer has credentials
        for canary in canaries:
            assert canary.encode() not in reset.body
        leaked = client.request(
            "POST", "/api/settings", {"path": "voice.wake_word", "value": REF, "pin": PIN}
        )
        assert leaked.status in (400, 403, 422)
        assert b"cloud_key" not in leaked.body
    finally:
        srv.stop()


def test_robot_plugin_rows_never_leak_a_secret_shaped_value(tmp_path):
    from roboemot.config.catalogue import descriptor
    from roboemot.config.stack import ConfigStack
    from roboemot.webui.settings import SettingsEditor
    from test_webui_robot_section import BASE, FakePlugin, manifest

    rows = [
        descriptor(f"{BASE}.token", "text", KEY, default=KEY),
        descriptor(f"{BASE}.url", "text", "https://u:pw-CANARY-8@host/v1"),
        descriptor(f"{BASE}.ref", "text", REF),
    ]
    stack = ConfigStack(app={"webui": {"bind": "127.0.0.1", "port": 0}})
    plugin = FakePlugin(rows=rows, claims=())
    editor = SettingsEditor(stack, manifest=manifest(), plugin=plugin)
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    try:
        assert client.login().status == 200
        resp = client.request("GET", "/api/settings")
        revealed = client.request("POST", "/api/settings/reveal", {"pin": PIN})
        for blob in (resp.body, revealed.body):
            for canary in (KEY, "pw-CANARY-8", "cloud_key"):
                assert canary.encode() not in blob
        robot = {f["path"]: f for f in resp.json["robot"]["fields"]}
        assert all(robot[f"{BASE}.{n}"]["secret"] for n in ("token", "url", "ref"))
        assert all(robot[f"{BASE}.{n}"]["value"] == "***" for n in ("token", "url", "ref"))
    finally:
        srv.stop()


# ------------------------------------------------------------------ write path (SPEC 8, 11A, 12)

CANARY_URL = "http://alice:pw-CANARY-55@nas.local:8080/v1"
CANARIES = ("pw-CANARY-55", "alice", "cloud_key")


def route_stack(tmp_path, **extra):
    from roboemot.config.stack import ConfigStack

    route = {
        "name": "cloud",
        "base_url": "https://cloud.example/v1",
        "api_key": REF,
        "model": "m",
        "first_token_timeout_s": 5,
    }
    return ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0}, "llm_routes": [route], **extra},
        audit_path=tmp_path / "audit.jsonl",
    )


def test_guardrail_messages_hold_no_url_credentials_or_secret_references(tmp_path):
    from roboemot.config.stack import GuardrailError, Violation

    stack = route_stack(tmp_path)
    for value in (CANARY_URL, "https://evil.example/v1?token=pw-CANARY-55"):
        with pytest.raises(GuardrailError) as caught:
            stack.set(
                "llm_routes.cloud.base_url",
                value,
                source="t",
                admin=True,
                refuse=frozenset({"locality", "secret"}),
            )
        text = str(caught.value)
        assert caught.value.violations and "***" not in text.replace("'***'", "")
        for canary in (*CANARIES, "${secret"):
            assert canary not in text
    shown = str(Violation("safety", "llm_routes.x.base_url", CANARY_URL, "https://a:b@h/?k=1"))
    assert "pw-CANARY-55" not in shown and "alice" not in shown and "a:b" not in shown
    key = str(Violation("admin", "llm_routes.x.api_key", REF, "sk-LEAK-CANARY-7f3a9c41"))
    assert "cloud_key" not in key and KEY not in key


def test_audit_entries_are_redacted_but_normal_values_stay_readable(tmp_path):
    stack = route_stack(tmp_path)
    stack.set("llm_routes.cloud.base_url", CANARY_URL, source="t", admin=True, layer="app")
    stack.set("llm_routes.cloud.model", "m2", source="t")
    blob = json.dumps([a.__dict__ for a in stack.audit], default=str)
    blob += (tmp_path / "audit.jsonl").read_text()
    for canary in CANARIES:
        assert canary not in blob
    assert stack.audit[-1].new == "m2" and stack.audit[-1].old == "m"


def test_last_error_and_the_refused_audit_entry_are_redacted(tmp_path):
    import yaml
    from roboemot.config.stack import ConfigStack

    app = tmp_path / "app.yaml"
    route = {"name": "cloud", "base_url": "https://cloud.example/v1", "model": "m"}
    app.write_text(yaml.safe_dump({"llm_routes": [{**route, "first_token_timeout_s": 5}]}))
    stack = ConfigStack.from_files(app=app, audit_path=tmp_path / "audit.jsonl")
    bad = {**route, "base_url": CANARY_URL, "api_key": REF, "first_token_timeout_s": 5}
    app.write_text(yaml.safe_dump({"llm_routes": [bad]}))
    stack.poll()
    assert stack.last_error is not None
    blob = stack.last_error + json.dumps([a.__dict__ for a in stack.audit], default=str)
    for canary in (*CANARIES, "${secret"):
        assert canary not in blob


def test_cli_does_not_print_credentials_from_a_guardrail_error(monkeypatch, capsys):
    from roboemot import cli
    from roboemot.config.stack import GuardrailError

    def boom(args):
        raise GuardrailError(f"refused {CANARY_URL} and {REF}")

    monkeypatch.setattr(cli, "_discover", boom)
    assert cli.main(["discover"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    for canary in (*CANARIES, "${secret"):
        assert canary not in err


def test_write_endpoints_never_echo_a_secret_in_a_response_or_an_error(tmp_path):
    from roboemot.webui.settings import SettingsEditor

    stack = route_stack(tmp_path)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    secretish = [CANARY_URL, REF, "https://h/v1?api_key=pw-CANARY-55"]
    blobs: list[bytes] = []

    def post(url, **body):
        resp = client.request("POST", url, {**body, "pin": PIN})
        blobs.append(resp.body)
        return resp

    try:
        for value in secretish:
            assert post("/api/settings", path="personality", value=value).status == 422
            entry = {"name": "n2", "base_url": "https://ok.example/v1", "model": value}
            assert post("/api/settings/entry", path="llm_routes", value=entry).status == 422
            nested = {"name": "n3", "base_url": "https://ok.example/v1", "api_key": value}
            assert post("/api/settings/entry", path="llm_routes", value=nested).status == 422
            moods = post(
                "/api/settings/entry", path="voice.moods", key="k", value={"instructions": value}
            )
            assert moods.status == 422
            assert (
                post("/api/settings/entry", path="voice.moods", key=value, value={}).status == 422
            )
        for path in ("llm_routes.cloud.api_key", "llm_routes.cloud"):
            assert post("/api/settings", path=path, value="x").status == 403
            assert post("/api/settings/reset", path=path).status == 403
        assert post("/api/settings/entry/remove", path="llm_routes.cloud").status in (403, 422)
        url = "http://nas.local/v1"
        asked = post("/api/settings", path="llm_routes.cloud.base_url", value=url)
        assert asked.status == 422 and asked.json["error"] == "endpoint_ack_required"
        moved = post(
            "/api/settings",
            path="llm_routes.cloud.base_url",
            value=url,
            ack_endpoint=asked.json["endpoint"],
        )
        assert moved.status == 403 and moved.json["error"] == "setting_guardrail"  # locality
        assert post("/api/settings", path="llm_routes.cloud.model", value="m2").status == 200
        listing = client.request("GET", "/api/settings").body  # field texts mention ${secret:name}
        assert not any(c.encode() in listing for c in CANARIES)
        assert [r.name for r in stack.effective().llm_routes] == ["cloud"]
        assert stack.effective().llm_routes[0].base_url == "https://cloud.example/v1"
    finally:
        srv.stop()
    stored = (tmp_path / "webui_settings.json").read_bytes() + (
        tmp_path / "audit.jsonl"
    ).read_bytes()
    for blob in (*blobs, stored):
        for canary in (*CANARIES, "${secret"):
            assert canary.encode() not in blob
