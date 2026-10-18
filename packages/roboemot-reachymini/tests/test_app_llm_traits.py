"""LLM-rated persona traits wired into the app (SPEC 8, 11A Persona, 4A Installer)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import httpx
import yaml
from roboemot import paths
from roboemot.config.schema import SafetyConfig
from roboemot.llm import Router
from roboemot.secret_store import SecretStore
from roboemot_reachymini import app as app_module
from test_app_perception import data, new_app, start, wait_for  # noqa: F401  (data is a fixture)
from test_app_webui import Browser, webui_data  # noqa: F401  (webui_data is a fixture)

ROUTE = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m1"}
RATED = "warmth=0.9\nenergy=0.8\nplayfulness=0.6\nformality=0.1"
MANUAL_ENERGY = 0.5  # the default trait value: the app config sets none


class FakeRoute:
    def __init__(self, status: int = 200, hang: bool = False) -> None:
        self.status, self.hang = status, hang
        self.requests: list[dict] = []
        self.auth: list[str] = []
        self.hosts: list[str] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.auth.append(request.headers.get("authorization", ""))
        self.hosts.append(request.url.host)
        if self.hang:
            await asyncio.sleep(3600)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "x"})
        chunk = {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": RATED}, "finish_reason": None}],
        }  # fmt: skip
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )


class LLMApp(app_module.RoboEmotApp):
    route: FakeRoute
    monitor = None

    def make_router(self, cfg, monitor):
        self.monitor = monitor
        router = super().make_router(cfg, monitor)  # the real one: it reads the secret store
        router.http_client_factory = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(self.route.handler)
        )
        return router


def app_with(route: FakeRoute) -> LLMApp:
    app = new_app(LLMApp)
    app.route = route
    return app


def configure(data, source: str | None, routes: bool = True) -> None:  # noqa: F811
    path = data / "app.yaml"
    cfg = yaml.safe_load(path.read_text()) or {}
    if source is not None:
        cfg["persona"] = {**cfg.get("persona", {}), "traits_source": source}
    if routes:
        cfg["llm_routes"] = [ROUTE]
    path.write_text(yaml.safe_dump(cfg))


def energy(app) -> float:
    return app.traits_worker.stack.get("persona.traits.energy")


def rated(app) -> bool:
    return app.traits_worker is not None and energy(app) == 0.8


def test_startup_rates_the_traits_and_caches_them_in_the_data_folder(data):  # noqa: F811
    configure(data, "llm")
    route = FakeRoute()
    app = app_with(route)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: rated(app))
        assert paths.traits_cache_path().exists()
        assert paths.traits_cache_path().parent == paths.data_path()
        assert len(route.requests) == 1
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()


def test_manual_source_makes_no_llm_call(data):  # noqa: F811
    configure(data, None)
    route = FakeRoute()
    app = app_with(route)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        time.sleep(0.5)
        assert route.requests == [] and not paths.traits_cache_path().exists()
        assert energy(app) == MANUAL_ENERGY
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_a_reloaded_config_clears_and_applies_the_rated_traits(data):  # noqa: F811
    configure(data, "llm")
    app = app_with(FakeRoute())
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: rated(app))
        configure(data, "manual")  # the watcher reloads the file within a second
        assert wait_for(lambda: energy(app) == MANUAL_ENERGY)
        configure(data, "llm")
        assert wait_for(lambda: rated(app))
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_a_failing_route_keeps_the_manual_traits_and_the_app_running(data):  # noqa: F811
    configure(data, "llm")
    route = FakeRoute(status=500)
    app = app_with(route)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: route.requests)
        assert wait_for(lambda: app.diagnostics is not None)
        time.sleep(0.3)
        assert energy(app) == MANUAL_ENERGY and thread.is_alive() and not errors
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()


def test_shutdown_does_not_wait_for_a_stuck_route(data):  # noqa: F811
    configure(data, "llm")
    route = FakeRoute(hang=True)
    app = app_with(route)
    stop, thread, errors = start(app)
    assert wait_for(lambda: route.requests)
    t0 = time.monotonic()
    stop.set()
    thread.join(15)
    assert not thread.is_alive() and not errors
    assert time.monotonic() - t0 < 10
    assert not app.traits_worker._thread.is_alive()


def test_the_router_shares_the_apps_network_monitor(data):  # noqa: F811
    configure(data, "llm")
    app = app_with(FakeRoute())
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: rated(app))
        assert app.monitor is not None and app.traits_worker.router.monitor is app.monitor
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def secret_route(data) -> None:  # noqa: F811
    configure(data, "llm", routes=False)
    path = data / "app.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["llm_routes"] = [
        {**ROUTE, "api_key": "${secret:cloud_key}"},
        {**ROUTE, "name": "b", "base_url": "http://b.local/v1"},
    ]
    path.write_text(yaml.safe_dump(cfg))


def test_a_missing_secret_makes_its_route_unavailable_with_one_warning(data, caplog):  # noqa: F811
    """Regression: such a route used to go out with an empty key (``Bearer none``)."""
    secret_route(data)
    route = FakeRoute()
    app = app_with(route)
    with caplog.at_level(logging.WARNING, logger="roboemot.llm.router"):
        stop, thread, errors = start(app)
        try:
            assert wait_for(lambda: rated(app))  # route "b" has no secret and answers
            assert [r.name for r in app.router_hub.router.eligible(False)] == ["b"]
        finally:
            stop.set()
            thread.join(15)
    warned = [r.getMessage() for r in caplog.records if r.name == "roboemot.llm.router"]
    assert len(warned) == 1 and "lan" in warned[0] and "cloud_key" in warned[0]
    assert set(route.hosts) == {"b.local"}  # nothing was sent to the route without its key
    assert not errors


def test_a_stored_secret_is_the_route_key_at_startup(data, caplog):  # noqa: F811
    secret_route(data)
    SecretStore.for_data_dir().set("cloud_key", "sk-app-1")
    route = FakeRoute()
    app = app_with(route)
    with caplog.at_level(logging.DEBUG):
        stop, thread, errors = start(app)
        try:
            assert wait_for(lambda: rated(app))
            assert [r.name for r in app.router_hub.router.eligible(False)] == ["lan", "b"]
            report = json.dumps(app.diagnostics.report())
        finally:
            stop.set()
            thread.join(15)
    assert route.auth and route.auth[0] == "Bearer sk-app-1"
    assert "sk-app-1" not in report and "sk-app-1" not in caplog.text
    assert not errors


def test_the_data_folder_is_private_when_the_app_creates_it(data):  # noqa: F811
    configure(data, "llm")
    assert not paths.data_path().exists()
    app = app_with(FakeRoute())
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: rated(app))
        assert paths.data_path().stat().st_mode & 0o777 == 0o700
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_route_check_router_is_the_shared_hubs_or_none():
    app = app_with(FakeRoute())
    assert app.route_check_router() is None
    router = Router([], SafetyConfig())
    app.router_hub = SimpleNamespace(router=router)
    assert app.route_check_router() is router


def test_web_route_check_is_charged_to_the_traits_workers_router(webui_data, data):  # noqa: F811
    configure(data, "llm")
    app = app_with(FakeRoute())
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.webui is not None and rated(app))
        page = Browser(app.webui.port)
        assert page.login() == 200
        before = app.traits_worker.router.used("lan")
        assert page.call("POST", "/api/routes/lan/check", {})[0] == 202

        def done():
            routes = page.call("GET", "/api/routes")[1]["routes"]
            return routes[0]["last_check"] is not None and not routes[0]["running"]

        assert wait_for(done)
        assert app.traits_worker.router.used("lan") > before
    finally:
        stop.set()
        thread.join(15)
    assert not errors


def test_a_reload_that_adds_a_route_gets_it_probed(data):  # noqa: F811
    configure(data, None)
    app = app_with(FakeRoute())
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.monitor is not None and app.diagnostics is not None)
        probed: list[str] = []
        app.monitor.probe = lambda host, port, timeout: probed.append(host) or True
        path = data / "app.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["llm_routes"] = [
            ROUTE,
            # a cloud host: a reload that adds a local route needs the admin (SPEC 11A, 12)
            {"name": "extra", "base_url": "http://extra.example.com/v1", "model": "m2"},
        ]
        path.write_text(yaml.safe_dump(cfg))  # the watcher reloads the file within a second
        assert wait_for(lambda: "extra" in app.monitor.links)
        assert wait_for(lambda: "extra.example.com" in probed)
    finally:
        stop.set()
        thread.join(15)
    assert not errors
