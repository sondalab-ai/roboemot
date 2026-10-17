"""Web UI route list and background conformance check: auth, CSRF, deadline, errors, no
secrets (SPEC 8, 12)."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.llm.conformance import RouteBudgetExhausted
from roboemot.webui import make_server
from roboemot.webui.routes import RouteBusy, RouteChecker
from test_llm_conformance import ITALIAN, SECRET_KEY, Fake, reply, text_chunks
from test_webui import PIN, Client, FakeService

ROUTE = {
    "name": "lan",
    "base_url": "https://user:pw@lan.local:8443/v1?token=abc",
    "api_key": SECRET_KEY,
    "model": "m",
    "first_token_timeout_s": 5,
    "daily_token_budget": 100_000,
}
PROTECTED = [("GET", "/api/routes"), ("POST", "/api/routes/lan/check")]


def make_cfg(route=None, check=None, **web) -> RoboEmotConfig:
    return RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0, **web},
        llm_routes=[route or ROUTE],
        llm_check=check or {},
    )


def checker_for(cfg, fake, store=None, **kw) -> RouteChecker:
    return RouteChecker(
        lambda: cfg,
        store,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
        **kw,
    )


def serve(tmp_path, cfg, routes):
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path, routes=routes)
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    assert client.login().status == 200
    return srv, client


def finish(client: Client, timeout: float = 10.0) -> dict:
    """Poll the list until the route is no longer running; the route entry."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        entry = client.request("GET", "/api/routes").json["routes"][0]
        if not entry["running"]:
            return entry
        time.sleep(0.05)
    raise AssertionError("check still running")


@pytest.fixture
def served(tmp_path):
    cfg, fake = make_cfg(), Fake()
    checker = checker_for(cfg, fake, tmp_path / "route_checks.json")
    srv, client = serve(tmp_path, cfg, checker)
    yield srv, client, checker, fake
    srv.stop()


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_requires_a_session(served, method, path):
    resp = Client(served[0].port).request(method, path, {} if method == "POST" else None)
    assert resp.status == 401
    assert served[3].bodies == []


def test_check_requires_the_csrf_token(served):
    resp = served[1].request("POST", "/api/routes/lan/check", {}, csrf=False)
    assert resp.status == 403 and resp.json == {"error": "csrf"}
    assert served[3].bodies == []


def test_list_shows_endpoint_only_and_no_key(served):
    resp = served[1].request("GET", "/api/routes")
    assert resp.status == 200
    assert resp.json["routes"] == [
        {
            "name": "lan",
            "model": "m",
            "endpoint": "lan.local:8443",
            "allowed_for_minors": False,
            "running": False,
            "last_check": None,
        }
    ]
    for leaked in (SECRET_KEY, "user", "pw", "token=abc", "/v1"):
        assert leaked.encode() not in resp.body


def test_check_runs_in_the_background_and_the_list_shows_the_report(served, tmp_path):
    _, client, _, _ = served
    resp = client.request("POST", "/api/routes/lan/check", {})
    assert resp.status == 202 and resp.json == {"running": True}
    entry = finish(client)
    report = entry["last_check"]
    assert report["ok"] is True and report["endpoint"] == "lan.local:8443"
    assert {c["name"] for c in report["checks"]} == {
        "streaming",
        "tool_round_trip",
        "first_token_latency",
        "italian",
    }
    saved = (tmp_path / "route_checks.json").read_text()
    assert SECRET_KEY not in saved and "lan" in saved
    assert SECRET_KEY not in json.dumps(entry)


def test_check_slower_than_the_request_deadline_still_reports(tmp_path):
    """Regression: the check (1.5 s here) outlives webui.request_deadline_s (1 s) and used to
    lose its response; now the POST returns at once and the result arrives through the list."""
    cfg = make_cfg(request_deadline_s=1)
    fake = Fake(plain=reply(text_chunks(ITALIAN), delay=1.5))
    srv, client = serve(tmp_path, cfg, checker_for(cfg, fake))
    try:
        started = time.monotonic()
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202
        assert time.monotonic() - started < 1
        report = finish(client)["last_check"]
        assert report["ok"] is True and report["first_token_s"] >= 1.5
    finally:
        srv.stop()


def test_check_past_its_own_deadline_is_reported_as_timed_out(tmp_path):
    base = make_cfg()
    cfg = base.model_copy(
        update={"llm_check": base.llm_check.model_copy(update={"deadline_s": 0.3})}
    )
    fake = Fake(plain=reply(text_chunks(ITALIAN), delay=30))
    srv, client = serve(tmp_path, cfg, checker_for(cfg, fake))
    try:
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202
        report = finish(client)["last_check"]
        assert report["error"] == "timeout" and report["ok"] is False
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202  # not stuck busy
        finish(client)
    finally:
        srv.stop()


def test_second_check_after_the_budget_is_spent_is_409(tmp_path):
    cfg = make_cfg(route=ROUTE | {"daily_token_budget": 50})
    fake = Fake()
    srv, client = serve(tmp_path, cfg, checker_for(cfg, fake))
    try:
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202
        finish(client)
        sent = len(fake.bodies)
        resp = client.request("POST", "/api/routes/lan/check", {})
        assert resp.status == 409 and resp.json == {"error": "route_budget"}
        assert len(fake.bodies) == sent
    finally:
        srv.stop()


def test_a_second_check_while_one_runs_is_busy(tmp_path):
    cfg = make_cfg()
    fake = Fake(plain=reply(text_chunks(ITALIAN), delay=0.5))
    srv, client = serve(tmp_path, cfg, checker_for(cfg, fake))
    try:
        assert client.request("POST", "/api/routes/lan/check", {}).status == 202
        assert client.request("GET", "/api/routes").json["routes"][0]["running"] is True
        resp = client.request("POST", "/api/routes/lan/check", {})
        assert resp.status == 503 and resp.json == {"error": "route_busy"}
        finish(client)
    finally:
        srv.stop()


def test_unknown_route_is_404(served):
    resp = served[1].request("POST", "/api/routes/nope/check", {})
    assert resp.status == 404 and resp.json == {"error": "route_unknown"}


def test_routes_unavailable_without_a_checker(tmp_path):
    cfg = make_cfg()
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/routes").status == 503
    finally:
        srv.stop()


class Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def list_routes(self):
        return []

    def start(self, name):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (RouteBusy("lan"), 503, "route_busy"),
        (RouteBudgetExhausted("lan"), 409, "route_budget"),
    ],
)
def test_error_mapping(tmp_path, exc, status, code):
    srv, client = serve(tmp_path, make_cfg(), Failing(exc))
    try:
        resp = client.request("POST", "/api/routes/lan/check", {})
        assert (resp.status, resp.json) == (status, {"error": code})
    finally:
        srv.stop()


def test_report_survives_a_restart_and_goes_stale_when_the_route_changes(tmp_path):
    store = tmp_path / "route_checks.json"
    cfg = make_cfg()
    checker = checker_for(cfg, Fake(), store)
    checker.start("lan")
    checker.thread.join(10)
    assert checker.list_routes()[0]["last_check"]["ok"] is True

    reborn = RouteChecker(lambda: cfg, store)
    assert reborn.list_routes()[0]["last_check"]["ok"] is True
    for change in ({"model": "other"}, {"base_url": "https://lan.local:9000/v1"}):
        moved = make_cfg(route=ROUTE | change)
        reader = RouteChecker(lambda cfg=moved: cfg, store)
        assert reader.list_routes()[0]["last_check"] is None


def test_malformed_stored_entries_are_dropped(tmp_path):
    store = tmp_path / "route_checks.json"
    good = {"model": "m", "endpoint": "lan.local:8443", "checks": [], "ok": True}
    store.write_text(json.dumps({"lan": good, "a": "x", "b": {"checks": "no"}, "c": [1]}))
    checker = RouteChecker(lambda: make_cfg(), store)
    assert list(checker._reports) == ["lan"]
    store.write_text("[1, 2]")
    assert RouteChecker(lambda: make_cfg(), store)._reports == {}


def test_the_data_folder_is_created_private_and_the_file_too(tmp_path):
    store = tmp_path / "fresh" / "route_checks.json"
    checker = checker_for(make_cfg(), Fake(), store)
    checker.start("lan")
    checker.thread.join(10)
    assert store.parent.stat().st_mode & 0o777 == 0o700
    assert store.stat().st_mode & 0o777 == 0o600
