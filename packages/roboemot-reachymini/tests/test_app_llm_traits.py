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

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
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
        return Router(
            cfg.llm_routes,
            cfg.safety,
            monitor=monitor,
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(self.route.handler)
            ),
        )


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


def test_a_secret_reference_without_a_secret_store_warns_once_at_startup(data, caplog):  # noqa: F811
    configure(data, None)
    path = data / "app.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["llm_routes"] = [{**ROUTE, "api_key": "${secret:cloud_key}"}, {**ROUTE, "name": "b"}]
    path.write_text(yaml.safe_dump(cfg))
    app = app_with(FakeRoute())
    with caplog.at_level(logging.WARNING, logger=app_module.log.name):
        stop, thread, errors = start(app)
        try:
            assert wait_for(lambda: app.diagnostics is not None)
        finally:
            stop.set()
            thread.join(15)
    warned = [r.getMessage() for r in caplog.records if "secret reference" in r.getMessage()]
    assert len(warned) == 1 and "lan" in warned[0] and "cloud_key" not in warned[0]
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


def test_route_check_router_is_the_traits_workers_or_none():
    app = app_with(FakeRoute())
    assert app.route_check_router() is None
    router = Router([], SafetyConfig())
    app.traits_worker = SimpleNamespace(router=router)
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
