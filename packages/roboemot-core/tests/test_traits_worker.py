"""SPEC 11A, Persona: the worker that rates LLM traits at startup and on config.changed."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest
from roboemot import paths
from roboemot.config import ConfigStack
from roboemot.config.schema import SafetyConfig
from roboemot.config.stack import ConfigChange
from roboemot.llm import Router
from roboemot.persona import llm_traits
from roboemot.persona.llm_traits import TraitCache, TraitsWorker

ROUTE = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m1"}
RATED = "warmth=0.9\nenergy=0.8\nplayfulness=0.6\nformality=0.1"


class Server:
    """A fake route: counts requests, replies with ``reply`` or fails, or hangs on demand."""

    def __init__(self, reply: str = RATED, status: int = 200, hang: bool = False) -> None:
        self.reply, self.status, self.hang = reply, status, hang
        self.requests: list[dict] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.hang:
            await asyncio.sleep(3600)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "x"})
        chunk = {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": self.reply}, "finish_reason": None}],
        }  # fmt: skip
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    def router(self, cfg) -> Router:
        return Router(
            cfg.llm_routes,
            cfg.safety,
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(self.handler)
            ),
        )


def wait_for(predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make(tmp_path, server: Server, source: str = "llm", **app):
    stack = ConfigStack(
        app={"persona": {"traits_source": source, "traits": {"energy": 0.1}}, "llm_routes": [ROUTE]}
        | app
    )
    worker = TraitsWorker(
        stack, server.router(stack.effective()), TraitCache(tmp_path / "t.json"), server.router
    )
    stack.subscribe(worker.on_change)
    return stack, worker


@pytest.fixture()
def runs(monkeypatch):
    """How many times the worker ran apply_llm_traits."""
    count = [0]
    real = llm_traits.apply_llm_traits

    async def counting(*args):
        count[0] += 1
        return await real(*args)

    monkeypatch.setattr(llm_traits, "apply_llm_traits", counting)
    return count


@pytest.fixture()
def started():
    workers: list[TraitsWorker] = []
    yield workers
    for w in workers:
        w.close()


def energy(stack) -> float:
    return stack.get("persona.traits.energy")


def test_startup_applies_the_rated_traits(tmp_path, started):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    assert len(server.requests) == 1
    assert any(e.source == llm_traits.SOURCE for e in stack.audit)


def test_a_config_change_applies_again(tmp_path, started):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    server.reply = "warmth=0.1\nenergy=0.2\nplayfulness=0.3\nformality=0.4"
    stack.set("llm_routes", [{**ROUTE, "model": "m2"}], layer="app")  # new model: new rating
    assert wait_for(lambda: energy(stack) == 0.2)
    assert len(server.requests) == 2


def test_switching_to_manual_clears_the_rated_traits_without_a_call(tmp_path, started):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    stack.set("persona.traits_source", "manual", layer="app")
    assert wait_for(lambda: energy(stack) == 0.1)
    assert len(server.requests) == 1


def test_manual_source_never_calls_a_route(tmp_path, started, runs):
    server = Server()
    stack, worker = make(tmp_path, server, source="manual")
    started.append(worker)
    worker.start()
    assert wait_for(lambda: runs[0] == 1)
    time.sleep(0.1)
    assert server.requests == [] and energy(stack) == 0.1
    assert not (tmp_path / "t.json").exists()


def test_a_failing_route_keeps_the_manual_traits_and_the_worker_alive(tmp_path, started, runs):
    server = Server(status=500)
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: runs[0] == 1 and server.requests)
    assert energy(stack) == 0.1
    server.status = 200  # the route recovers; a later change is rated normally
    stack.set("persona.traits_source", "manual", layer="app")
    stack.set("persona.traits_source", "llm", layer="app")
    assert wait_for(lambda: energy(stack) == 0.8)


def test_an_unexpected_error_is_logged_and_does_not_stop_the_worker(
    tmp_path, started, monkeypatch, caplog
):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)

    async def boom(*args):
        raise RuntimeError("secret prompt text")

    monkeypatch.setattr(llm_traits, "apply_llm_traits", boom)
    worker.start()
    assert wait_for(lambda: "RuntimeError" in caplog.text)
    assert "secret prompt text" not in caplog.text
    assert worker._thread.is_alive()


def test_its_own_derived_change_does_not_loop(tmp_path, started, runs):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    time.sleep(0.3)
    assert runs[0] == 1 and len(server.requests) == 1


def test_a_listener_call_from_the_worker_thread_is_ignored(tmp_path, started, runs):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: runs[0] == 1)
    seen = []

    async def from_worker():
        worker.on_change(ConfigChange(["persona.traits.energy"], [], stack.effective()))
        seen.append(worker._wake.is_set())

    asyncio.run_coroutine_threadsafe(from_worker(), worker._loop).result(2)
    assert seen == [False]


def test_unrelated_changes_do_not_wake_it(tmp_path, started, runs):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    worker.start()
    assert wait_for(lambda: runs[0] == 1)
    stack.set("conversation.language", "en", layer="app")
    time.sleep(0.2)
    assert runs[0] == 1


def test_a_burst_of_requests_during_a_run_coalesces_into_one_more_run(tmp_path, monkeypatch):
    runs = [0]
    server = Server()
    stack, worker = make(tmp_path, server)
    gate = threading.Event()
    real = llm_traits.apply_llm_traits

    async def gated(*args):
        runs[0] += 1
        while not gate.is_set():  # the run in flight takes a while
            await asyncio.sleep(0.01)
        return await real(*args)

    monkeypatch.setattr(llm_traits, "apply_llm_traits", gated)
    worker.start()
    try:
        assert wait_for(lambda: runs[0] == 1)
        for _ in range(20):
            worker.request()
        gate.set()
        assert wait_for(lambda: runs[0] == 2)
        time.sleep(0.2)
        assert runs[0] == 2
    finally:
        gate.set()
        worker.close()


def test_close_does_not_wait_for_a_slow_route(tmp_path, runs):
    server = Server(hang=True)
    stack, worker = make(tmp_path, server)
    worker.start()
    assert wait_for(lambda: server.requests)
    t0 = time.monotonic()
    worker.close()
    assert time.monotonic() - t0 < llm_traits.JOIN_TIMEOUT_S + 0.5
    assert not worker._thread.is_alive()
    assert energy(stack) == 0.1


def test_close_before_start_and_twice_is_harmless(tmp_path):
    stack, worker = make(tmp_path, Server())
    worker.close()
    worker.request()
    worker.close()


def test_the_router_is_rebuilt_on_new_routes_and_keeps_its_budgets(tmp_path, started):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    first = worker.router
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    used = first.used("lan")
    assert used > 0
    stack.set("llm_routes", [{**ROUTE, "model": "m2"}], layer="app")
    assert wait_for(lambda: worker.router is not first)
    assert worker.router.routes[0].model == "m2"
    assert wait_for(lambda: worker.router.used("lan") == 2 * used)  # carried, then charged again


def test_the_router_is_kept_while_routes_and_safety_do_not_change(tmp_path, started, runs):
    server = Server()
    stack, worker = make(tmp_path, server)
    started.append(worker)
    first = worker.router
    worker.start()
    assert wait_for(lambda: energy(stack) == 0.8)
    stack.set("persona.traits_source", "manual", layer="app")
    assert wait_for(lambda: runs[0] == 2)
    assert worker.router is first and worker.router.safety == SafetyConfig()


def test_the_cache_file_lives_in_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DIR, str(tmp_path / "data"))
    path = paths.traits_cache_path()
    assert path == tmp_path / "data" / paths.TRAITS_CACHE_FILE
    assert path.parent == paths.data_path()
    assert paths.traits_cache_path(str(tmp_path / "cfg")).parent == tmp_path / "cfg"
