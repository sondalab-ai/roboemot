"""The realtime route check against the fake server (SPEC 8, Realtime speech route; SPEC 12, 13).

One connect, ``session.updated``, one ``input_text`` turn with a client-chosen item id, the
first-audio latency, then ``conversation.item.delete`` of that id. No microphone audio, the
budget charged, and the report names hosts only: never a key, query, path, identifier value,
server error text or the reply.
"""

from __future__ import annotations

import json
import logging
import time

import pytest
from fake_server import ERROR_MESSAGE, REPLY_TEXT, FakeServer
from realtime_helpers import KEY
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.conformance import (
    RouteBudgetExhausted,
    RouteKeyUnavailable,
    RouteKindUnsupported,
)
from roboemot.llm.router import Router
from roboemot.realtime import check
from roboemot.realtime.session import Allocation, DirectSession

PLATFORM_ID = "HWID0123456789ab"
QUERY = "ticket=QUERYSECRET"


def make_route(base_url: str, *, keyed: bool = False, **kw: object) -> LLMRoute:
    kw.setdefault("locality", "local")
    return LLMRoute(
        name="rt",
        base_url=base_url,
        model="s2s",
        kind="realtime",
        api_key="${secret:rt_key}" if keyed else "none",
        **kw,
    )


def make_cfg(route: LLMRoute, *, mode: str = "direct", **kw: object) -> RoboEmotConfig:
    return RoboEmotConfig(llm_routes=[route], voice={"realtime_route": {"mode": mode}}, **kw)


def make_router(cfg: RoboEmotConfig, *, secret: str | None = KEY) -> Router:
    def secrets(name: str) -> str:
        if secret is None:
            raise KeyError(name)
        return secret

    return Router(cfg.llm_routes, cfg.safety, secrets=secrets)


class Allocator:
    """A platform provider: hands out ``url`` and says whether the identifier went with it."""

    key_hosts: tuple[str, ...] = ()
    declared: tuple[str, ...] = ()  # the key hosts this allocation names

    def __init__(self, url: str) -> None:
        self.url = url
        self.asked = 0

    async def allocate(self, request):
        self.asked += 1
        if request.platform_id:
            request.on_platform_id()  # a real provider would send PLATFORM_ID now
        return Allocation(self.url, self.declared)


def run(cfg: RoboEmotConfig, router: Router | None = None, provider=None, **kw):
    route = cfg.llm_routes[0]
    return check.check_realtime_route(
        route,
        cfg,
        router=router or make_router(cfg),
        provider=provider if provider is not None else DirectSession(),
        **kw,
    )


def leaks(report: check.RealtimeReport, caplog) -> list[str]:
    text = json.dumps(report.to_dict()) + caplog.text
    found = [s for s in (KEY, QUERY, PLATFORM_ID, ERROR_MESSAGE, REPLY_TEXT) if s in text]
    return found + (["/v1"] if "/v1" in json.dumps(report.to_dict()) else [])


def test_direct_mode_connects_answers_and_deletes_the_named_item(caplog):
    """Connect, session.updated, one input_text turn, first audio, and the item delete."""
    caplog.set_level(logging.DEBUG)
    with FakeServer(delete_reply="deleted") as server:
        cfg = make_cfg(make_route(server.http_base()))
        router = make_router(cfg)
        report = run(cfg, router)
    assert report.ok, report.to_dict()
    assert report.error == "" and report.failure is None
    assert report.mode == "direct" and report.allocator_host is None
    assert report.connect_host == "127.0.0.1"
    assert report.header_to_connect_host == "none"  # the route has no key
    assert report.update_seen is True
    assert report.first_audio_ms is not None and report.first_audio_ms >= 0
    assert report.item_delete == "accepted"
    assert report.platform_id_sent is False
    kinds = [e["type"] for e in server.received]
    assert kinds == [
        "session.update",
        "conversation.item.create",
        "response.create",
        "conversation.item.delete",
    ]
    item = server.received[1]["item"]
    assert item["content"][0]["type"] == "input_text"
    assert item["id"] == check.ITEM_ID == server.received[3]["item_id"]
    assert {c.name: c.status for c in report.checks} == {
        check.CONNECT: "pass",
        check.FIRST_AUDIO: "pass",
        check.ITEM_DELETE: "pass",
    }
    assert leaks(report, caplog) == []


def test_the_check_sends_no_microphone_audio_and_charges_the_budget():
    with FakeServer(delete_reply="deleted") as server:
        cfg = make_cfg(make_route(server.http_base()))
        router = make_router(cfg)
        report = run(cfg, router)
    assert not [e for e in server.received if e["type"].startswith("input_audio_buffer")]
    assert report.charged == 42 == router.used("rt")  # response.done's usage


def test_without_usage_the_estimate_is_charged(monkeypatch):
    with FakeServer(mode="stall") as server:
        cfg = make_cfg(make_route(server.http_base(), stall_timeout_s=1.5))
        router = make_router(cfg)
        report = run(cfg, router)
    assert report.failure == "stall" and report.first_audio_ms is None
    assert report.charged > 0 and router.used("rt") == report.charged
    assert not report.ok


def test_a_keyed_route_sends_the_key_to_its_own_host_only():
    with FakeServer(delete_reply="deleted") as server:
        cfg = make_cfg(make_route(server.http_base(), keyed=True))
        report = run(cfg)
        sent = server.requests[0]["headers"]
    assert sent.get("Authorization") == f"Bearer {KEY}"
    assert report.header_to_connect_host == "sent"
    assert KEY not in json.dumps(report.to_dict())


def test_platform_mode_names_the_allocator_and_the_connect_host_never_the_query(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer(delete_reply="deleted") as server:
        provider = Allocator(f"ws://127.0.0.1:{server.port}/v1/realtime?{QUERY}")
        route = make_route(f"http://localhost:{server.port}/session")
        cfg = make_cfg(route, mode="platform")
        report = run(cfg, provider=provider)
        assert server.requests[0]["path"] == f"/v1/realtime?{QUERY}"  # the query is used
    assert report.ok, report.to_dict()
    assert report.mode == "platform"
    assert report.allocator_host == "localhost"
    assert report.connect_host == "127.0.0.1"
    assert leaks(report, caplog) == []


def test_a_key_refused_for_the_allocated_host_is_reported_with_that_host(caplog):
    """The owner reads connect_host here to declare KEY_HOSTS exactly (SR-4 concern 1)."""
    caplog.set_level(logging.DEBUG)
    provider = Allocator(f"wss://connect.example.net/v1/realtime?{QUERY}")
    route = make_route("https://alloc.example.org/session", keyed=True, locality="cloud")
    report = run(make_cfg(route, mode="platform"), provider=provider)
    assert provider.asked == 1
    assert report.failure == "url_key_host"
    assert report.header_to_connect_host == "refused"
    assert report.allocator_host == "alloc.example.org"
    assert report.connect_host == "connect.example.net"
    assert report.update_seen is False and report.item_delete == "not_run"
    assert report.charged == 0
    assert {c.name: c.status for c in report.checks}[check.CONNECT] == "fail"
    assert leaks(report, caplog) == []


def test_the_platform_identifier_is_reported_as_sent_never_its_value(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer(delete_reply="deleted") as server:
        provider = Allocator(f"ws://127.0.0.1:{server.port}/v1/realtime")
        route = make_route(f"http://127.0.0.1:{server.port}/session")
        cfg = make_cfg(route, mode="platform", privacy={"realtime_platform_id": True})
        report = run(cfg, provider=provider)
    assert report.platform_id_sent is True
    assert leaks(report, caplog) == []


def test_an_item_delete_the_server_refuses_is_reported_refused_without_its_text(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer(delete_reply="error") as server:
        report = run(make_cfg(make_route(server.http_base())))
    assert report.item_delete == "refused"
    assert {c.name: c.status for c in report.checks}[check.ITEM_DELETE] == "fail"
    assert leaks(report, caplog) == []


def test_an_item_delete_with_no_answer_is_unconfirmed():
    with FakeServer(delete_reply="none") as server:
        route = make_route(server.http_base(), stall_timeout_s=1.5)
        report = run(make_cfg(route))
    assert report.item_delete == "unconfirmed"
    assert {c.name: c.status for c in report.checks}[check.ITEM_DELETE] == "skip"
    assert report.ok  # a server that never confirms deletes is not a failure of the route


def test_a_server_error_before_session_updated_drops_its_message(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer(mode="error_first") as server:
        report = run(make_cfg(make_route(server.http_base())))
        kinds = [e["type"] for e in server.received]
    assert report.failure == "server_error" and report.update_seen is False
    assert kinds == ["session.update"]  # no turn after a refused session
    assert leaks(report, caplog) == []


def test_pseudonyms_on_refuse_a_cloud_connect_host_before_anything_is_sent():
    provider = Allocator("wss://connect.example.net/v1/realtime")
    with FakeServer() as server:
        route = make_route(f"http://127.0.0.1:{server.port}/session")
        cfg = make_cfg(route, mode="platform", privacy={"pseudonyms_to_cloud": True})
        report = run(cfg, provider=provider)
        assert server.received == []
    assert report.failure == "pseudonyms"
    assert report.connect_host == "connect.example.net"


def test_platform_mode_without_a_provider_reports_it_and_sends_nothing():
    route = make_route("https://alloc.example.org/session", locality="cloud")
    cfg = make_cfg(route, mode="platform")
    report = check.check_realtime_route(route, cfg, router=make_router(cfg), provider=None)
    assert report.error == "provider" and not report.ok


def test_past_the_deadline_the_check_ends_as_timed_out():
    with FakeServer(mode="silent") as server:
        route = make_route(server.http_base(), first_token_timeout_s=30)
        report = run(make_cfg(route), deadline_s=0.5)
    assert report.error == "timeout" and not report.ok
    assert report.charged > 0  # what was sent is charged


def test_a_spent_budget_refuses_the_check_before_anything_is_sent():
    with FakeServer() as server:
        route = make_route(server.http_base(), daily_token_budget=10)
        cfg = make_cfg(route)
        router = make_router(cfg)
        router.charge("rt", 10)
        with pytest.raises(RouteBudgetExhausted):
            run(cfg, router)
        assert server.requests == []


def test_an_unreadable_key_refuses_the_check_before_anything_is_sent():
    with FakeServer() as server:
        cfg = make_cfg(make_route(server.http_base(), keyed=True))
        with pytest.raises(RouteKeyUnavailable):
            run(cfg, make_router(cfg, secret=None))
        assert server.requests == []


def test_a_chat_route_is_not_a_realtime_check():
    route = LLMRoute(name="chat", base_url="http://127.0.0.1:9/v1", model="m", api_key="none")
    cfg = RoboEmotConfig(llm_routes=[route])
    with pytest.raises(RouteKindUnsupported):
        check.check_realtime_route(route, cfg, router=make_router(cfg), provider=DirectSession())


# ----------------------------------------------------------------- CLI


def app_file(tmp_path, base_url: str, *, mode: str = "direct") -> str:
    import yaml

    route = {"name": "rt", "base_url": base_url, "model": "s2s", "kind": "realtime"}
    route |= {"locality": "local", "api_key": "none"}
    path = tmp_path / "app.yaml"
    body = {"llm_routes": [route], "voice": {"realtime_route": {"mode": mode}}}
    path.write_text(yaml.safe_dump(body))
    return str(path)


def test_the_cli_prints_the_hosts_and_the_outcome(tmp_path, capsys):
    from roboemot import cli

    with FakeServer(delete_reply="deleted") as server:
        app = app_file(tmp_path, server.http_base())
        assert cli.main(["realtime", "check", "rt", "--app", app]) == 0
    out = capsys.readouterr().out
    assert "connect_host            127.0.0.1" in out
    assert "header_to_connect_host  none" in out
    assert "item_delete             accepted" in out
    assert "pass  realtime_first_audio" in out
    assert "route rt (s2s): ok" in out
    assert "/v1" not in out and REPLY_TEXT not in out


def test_the_cli_json_report_and_exit_one_on_a_failure(tmp_path, capsys):
    from roboemot import cli

    with FakeServer(mode="error_first") as server:
        app = app_file(tmp_path, server.http_base())
        assert cli.main(["realtime", "check", "rt", "--app", app, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["failure"] == "server_error"
    assert report["update_seen"] is False and ERROR_MESSAGE not in json.dumps(report)


def test_the_cli_in_platform_mode_without_a_provider_says_so(tmp_path, capsys):
    from roboemot import cli

    app = app_file(tmp_path, "https://alloc.example.org/session", mode="platform")
    assert cli.main(["realtime", "check", "rt", "--app", app, "--robot", "no-such-robot"]) == 1
    assert "route rt (s2s): provider" in capsys.readouterr().out


def test_the_cli_refuses_a_chat_route(tmp_path, capsys):
    import yaml
    from roboemot import cli

    path = tmp_path / "app.yaml"
    chat = {"name": "lan", "base_url": "http://127.0.0.1:9/v1", "model": "m", "api_key": "none"}
    path.write_text(yaml.safe_dump({"llm_routes": [chat]}))
    assert cli.main(["realtime", "check", "lan", "--app", str(path)]) == 1
    assert "is a chat route" in capsys.readouterr().err


def test_the_chat_check_of_a_realtime_route_points_to_the_realtime_check(tmp_path, capsys):
    from roboemot import cli

    app = app_file(tmp_path, "http://127.0.0.1:9/v1")
    assert cli.main(["llm", "check", "rt", "--app", app]) == 1
    assert "roboemot realtime check rt" in capsys.readouterr().err


# ----------------------------------------------------------------- web UI Routes tab


def web_checker(cfg: RoboEmotConfig, store=None, **kw):
    from roboemot.webui.routes import RouteChecker

    return RouteChecker(lambda: cfg, store, **kw)


def served(tmp_path, cfg, checker):
    import test_webui_routes as web_routes

    return web_routes.serve(tmp_path, cfg, checker)


def test_the_routes_tab_runs_the_realtime_check_and_lists_its_report(tmp_path):
    """SR-5 replaces the 409 route_kind: the same admin-only endpoint runs the realtime check."""
    import test_webui_routes as web_routes

    with FakeServer(delete_reply="deleted") as server:
        cfg = make_cfg(make_route(server.http_base()), webui={"bind": "127.0.0.1", "port": 0})
        checker = web_checker(cfg, tmp_path / "route_checks.json")
        srv, client = served(tmp_path, cfg, checker)
        try:
            resp = client.request("POST", "/api/routes/rt/check", {})
            assert resp.status == 202 and resp.json == {"running": True}
            entry = web_routes.finish(client)
        finally:
            srv.stop()
        kinds = [e["type"] for e in server.received]
    report = entry["last_check"]
    assert report["kind"] == "realtime" and report["ok"] is True
    assert report["connect_host"] == "127.0.0.1" and report["item_delete"] == "accepted"
    assert report["update_seen"] is True and report["first_audio_ms"] is not None
    names = {c["name"] for c in report["checks"]}
    assert names == {check.CONNECT, check.FIRST_AUDIO, check.ITEM_DELETE}
    assert "input_audio_buffer.append" not in kinds
    saved = (tmp_path / "route_checks.json").read_text()
    for leaked in ("/v1", REPLY_TEXT):
        assert leaked not in json.dumps(report) and leaked not in saved


def test_the_routes_tab_realtime_check_needs_the_admin_session(tmp_path):
    from test_webui import Client

    with FakeServer() as server:
        cfg = make_cfg(make_route(server.http_base()), webui={"bind": "127.0.0.1", "port": 0})
        srv, _ = served(tmp_path, cfg, web_checker(cfg))
        try:
            resp = Client(srv.port).request("POST", "/api/routes/rt/check", {})
            assert resp.status == 401
        finally:
            srv.stop()
        assert server.requests == []


def test_the_routes_tab_uses_the_robots_session_provider_in_platform_mode(tmp_path):
    import test_webui_routes as web_routes

    with FakeServer(delete_reply="deleted") as server:
        provider = Allocator(f"ws://127.0.0.1:{server.port}/v1/realtime?{QUERY}")
        route = make_route(f"http://localhost:{server.port}/session")
        cfg = make_cfg(route, mode="platform", webui={"bind": "127.0.0.1", "port": 0})
        srv, client = served(tmp_path, cfg, web_checker(cfg, realtime_provider=lambda: provider))
        try:
            assert client.request("POST", "/api/routes/rt/check", {}).status == 202
            report = web_routes.finish(client)["last_check"]
        finally:
            srv.stop()
    assert provider.asked == 1
    assert report["allocator_host"] == "localhost" and report["connect_host"] == "127.0.0.1"
    assert QUERY not in json.dumps(report)


def test_the_routes_tab_in_platform_mode_without_a_provider_reports_it(tmp_path):
    import test_webui_routes as web_routes

    route = make_route("https://alloc.example.org/session", locality="cloud")
    cfg = make_cfg(route, mode="platform", webui={"bind": "127.0.0.1", "port": 0})
    srv, client = served(tmp_path, cfg, web_checker(cfg))
    try:
        assert client.request("POST", "/api/routes/rt/check", {}).status == 202
        report = web_routes.finish(client)["last_check"]
    finally:
        srv.stop()
    assert report["error"] == "provider" and report["ok"] is False


# ----------------------------------------------------------------- fix round 1


class SendFails:
    """A client whose connect succeeds and whose every send fails (the socket is gone)."""

    reason = None

    def __init__(self, route, settings, provider, **kw) -> None:
        self.sent: list[dict] = []

    def connect(self, update) -> bool:
        return True

    def snapshot(self) -> dict:
        return {"endpoint": "ws://127.0.0.1:9", "key_sent": False}

    def send(self, event):
        import concurrent.futures

        self.sent.append(event)
        done: concurrent.futures.Future = concurrent.futures.Future()
        done.set_result(False)
        return done

    def close(self) -> None:
        pass


def test_a_send_that_fails_is_reported_at_once_not_as_a_stall():
    """Review M6: a failed send of the turn is send_failed, without waiting for the route's
    first_token_timeout_s."""
    route = make_route("http://127.0.0.1:9/v1", first_token_timeout_s=30)
    cfg = make_cfg(route)
    started = time.monotonic()
    report = run(cfg, client_factory=SendFails)
    assert time.monotonic() - started < 5
    assert report.failure == "send_failed" and report.item_delete == "not_run"
    assert report.charged == 0  # nothing reached the server


def test_the_deadline_during_the_item_delete_leaves_it_unconfirmed():
    """Review M7: a route that answered, whose delete got no answer by the deadline, is not
    a check that took too long."""
    with FakeServer(delete_reply="none") as server:
        route = make_route(server.http_base(), stall_timeout_s=30)
        report = run(make_cfg(route), deadline_s=3.0)
    assert report.error == "" and report.item_delete == "unconfirmed"
    assert report.first_audio_ms is not None and report.ok


def test_the_mode_comes_from_the_provider_the_check_used():
    """Review M4: before the restart a mode edit is live in the config, the provider is not."""
    with FakeServer(delete_reply="deleted") as server:
        route = make_route(server.http_base())
        report = run(make_cfg(route, mode="platform"), provider=DirectSession())
    assert report.mode == "direct" and report.allocator_host is None


def test_a_pseudonym_refusal_before_the_handshake_does_not_read_as_key_sent():
    """Review M5: the key host is declared, but pseudonyms stop the connect before any socket."""
    provider = Allocator("wss://rt.connect.example.net/realtime")
    provider.key_hosts = (".connect.example.net",)
    provider.declared = (".connect.example.net",)
    with FakeServer() as server:
        route = make_route(f"http://127.0.0.1:{server.port}/session", keyed=True)
        cfg = make_cfg(route, mode="platform", privacy={"pseudonyms_to_cloud": True})
        report = run(cfg, provider=provider)
        assert server.requests == []
    assert report.failure == "pseudonyms"
    assert report.header_to_connect_host == "not_sent"


def test_the_privacy_gates_follow_the_live_config():
    """Review M3: pseudonyms turned on during a check refuse the cloud connect host."""
    provider = Allocator("wss://connect.example.net/v1/realtime")
    with FakeServer() as server:
        route = make_route(f"http://127.0.0.1:{server.port}/session")
        cfg = make_cfg(route, mode="platform")
        live = make_cfg(route, mode="platform", privacy={"pseudonyms_to_cloud": True})
        report = run(cfg, provider=provider, config=lambda: live)
        assert server.received == []
    assert report.failure == "pseudonyms"


def test_the_routes_tab_check_reads_the_privacy_gates_from_the_live_config(tmp_path):
    """Review M3: the web check passes the committed config, not the snapshot at its start."""
    from roboemot.webui.routes import RouteChecker

    provider = Allocator("wss://connect.example.net/v1/realtime")
    with FakeServer() as server:
        route = make_route(f"http://127.0.0.1:{server.port}/session")
        off = make_cfg(route, mode="platform")
        on = make_cfg(route, mode="platform", privacy={"pseudonyms_to_cloud": True})
        calls = []

        def cfg():
            calls.append(1)
            return off if len(calls) == 1 else on  # turned on right after the check started

        checker = RouteChecker(cfg, None, realtime_provider=lambda: provider)
        checker.start("rt")
        checker.thread.join(10)
        assert server.received == []
    assert checker.list_routes()[0]["last_check"]["failure"] == "pseudonyms"
