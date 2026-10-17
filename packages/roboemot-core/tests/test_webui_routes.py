"""Web UI route list and conformance check: auth, CSRF, errors, no secrets (SPEC 8, 12)."""

from __future__ import annotations

import httpx
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.llm.conformance import RouteBudgetExhausted
from roboemot.webui import make_server
from roboemot.webui.routes import RouteBusy, RouteChecker
from test_llm_conformance import SECRET_KEY, Fake
from test_webui import PIN, Client, FakeService

ROUTE = {
    "name": "lan",
    "base_url": "https://user:pw@lan.local:8443/v1?token=abc",
    "api_key": SECRET_KEY,
    "model": "m",
    "first_token_timeout_s": 0.5,
    "daily_token_budget": 100_000,
}
PROTECTED = [("GET", "/api/routes"), ("POST", "/api/routes/lan/check")]


def make(tmp_path, fake=None, routes=None):
    fake = fake or Fake()
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0}, llm_routes=[ROUTE])
    checker = routes or RouteChecker(
        lambda: cfg,
        tmp_path / "route_checks.json",
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path, routes=checker)
    srv.set_pin(PIN)
    srv.start()
    return srv, checker, fake


@pytest.fixture
def served(tmp_path):
    srv, checker, fake = make(tmp_path)
    yield srv, checker, fake
    srv.stop()


@pytest.fixture
def authed(served):
    client = Client(served[0].port)
    assert client.login().status == 200
    return client


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_requires_a_session(served, method, path):
    resp = Client(served[0].port).request(method, path, {} if method == "POST" else None)
    assert resp.status == 401
    assert served[2].bodies == []


def test_check_requires_the_csrf_token(served, authed):
    resp = authed.request("POST", "/api/routes/lan/check", {}, csrf=False)
    assert resp.status == 403 and resp.json == {"error": "csrf"}
    assert served[2].bodies == []


def test_list_shows_host_only_and_no_key(authed):
    resp = authed.request("GET", "/api/routes")
    assert resp.status == 200
    assert resp.json["routes"] == [
        {
            "name": "lan",
            "model": "m",
            "host": "lan.local",
            "allowed_for_minors": False,
            "last_check": None,
        }
    ]
    for leaked in (SECRET_KEY, "user", "pw", "token=abc", "8443", "/v1"):
        assert leaked.encode() not in resp.body


def test_check_runs_stores_and_lists_the_report(served, authed, tmp_path):
    resp = authed.request("POST", "/api/routes/lan/check", {})
    assert resp.status == 200 and resp.json["ok"] is True
    assert {c["name"] for c in resp.json["checks"]} == {
        "streaming",
        "tool_round_trip",
        "first_token_latency",
        "italian",
    }
    assert SECRET_KEY.encode() not in resp.body
    listed = authed.request("GET", "/api/routes").json["routes"][0]
    assert listed["last_check"]["checked_at"] == resp.json["checked_at"]
    saved = (tmp_path / "route_checks.json").read_text()
    assert SECRET_KEY not in saved and "lan" in saved


def test_report_survives_a_restart(served, authed, tmp_path):
    authed.request("POST", "/api/routes/lan/check", {})
    reborn = RouteChecker(lambda: served[1].cfg(), tmp_path / "route_checks.json")
    assert reborn.list_routes()[0]["last_check"]["ok"] is True


def test_unknown_route_is_404(authed):
    resp = authed.request("POST", "/api/routes/nope/check", {})
    assert resp.status == 404 and resp.json == {"error": "route_unknown"}


def test_routes_unavailable_without_a_checker(tmp_path):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/routes").status == 503
    finally:
        srv.stop()


def test_spent_budget_is_409_and_sends_nothing(tmp_path):
    fake = Fake()
    cfg = RoboEmotConfig(
        webui={"bind": "127.0.0.1", "port": 0}, llm_routes=[ROUTE | {"daily_token_budget": 0}]
    )
    srv, _, _ = make(
        tmp_path,
        fake,
        RouteChecker(
            lambda: cfg,
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(fake.handler)
            ),
        ),
    )
    try:
        client = Client(srv.port)
        client.login()
        resp = client.request("POST", "/api/routes/lan/check", {})
        assert resp.status == 409 and resp.json == {"error": "route_budget"}
        assert fake.bodies == []
    finally:
        srv.stop()


class Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def list_routes(self):
        return []

    def check(self, name):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (RouteBusy("lan"), 503, "route_busy"),
        (TimeoutError(), 504, "route_timeout"),
        (RouteBudgetExhausted("lan"), 409, "route_budget"),
    ],
)
def test_error_mapping(tmp_path, exc, status, code):
    srv, _, _ = make(tmp_path, routes=Failing(exc))
    try:
        client = Client(srv.port)
        client.login()
        resp = client.request("POST", "/api/routes/lan/check", {})
        assert (resp.status, resp.json) == (status, {"error": code})
    finally:
        srv.stop()


def test_a_second_check_while_one_runs_is_busy(tmp_path):
    checker = RouteChecker(lambda: RoboEmotConfig(llm_routes=[ROUTE]))
    assert checker._lock.acquire(blocking=False)
    with pytest.raises(RouteBusy):
        checker.check("lan")
