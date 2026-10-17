"""SPEC 8: per-route conformance test (streaming, tool round trip, latency, Italian)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from roboemot.config.schema import LLMCheckConfig, LLMRoute, SafetyConfig
from roboemot.llm import Router
from roboemot.llm.conformance import (
    FAIL,
    PASS,
    SKIP,
    RouteBudgetExhausted,
    check_route,
    italian_ratio,
)
from test_llm_router import chunk, sse

SECRET_KEY = "sk-secret-do-not-leak"
ITALIAN = ["Ciao, ", "oggi c'è ", "il sole ", "e il cielo è bello."]
ENGLISH = ["Hello, ", "today the ", "sun is ", "shining."]
CALL = [
    chunk(
        tool_calls=[
            {
                "index": 0,
                "id": "c1",
                "type": "function",
                "function": {"name": "get_time", "arguments": "{}"},
            }
        ]
    )
]


def reply(chunks: list[dict], delay: float = 0.0):
    async def make() -> httpx.Response:
        if delay:
            await asyncio.sleep(delay)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(chunks)
        )

    return make


def text_chunks(parts: list[str]) -> list[dict]:
    return [chunk(p) for p in parts]


class Fake:
    """Answers the Italian request (no tools), the tool request and the round-trip request."""

    def __init__(self, *, plain=None, tool=None, final=None, status=None):
        self.plain = plain if plain is not None else reply(text_chunks(ITALIAN))
        self.tool = tool if tool is not None else reply(CALL)
        self.final = final if final is not None else reply(text_chunks(["Sono ", "le 10:30."]))
        self.status = status
        self.bodies: list[dict] = []
        self.auth: list[str] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.auth.append(request.headers.get("authorization", ""))
        if self.status:
            return httpx.Response(self.status, json={"error": {"message": "prompt echo"}})
        if "tools" not in body:
            return await self.plain()
        if any(m["role"] == "tool" for m in body["messages"]):
            return await self.final()
        return await self.tool()


def route(**kw) -> LLMRoute:
    kw.setdefault("first_token_timeout_s", 0.5)
    return LLMRoute(name="lan", base_url="http://lan.local/v1", model="m", api_key=SECRET_KEY, **kw)


async def run(fake: Fake, r: LLMRoute | None = None, **kw):
    return await check_route(
        r or route(),
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
        **kw,
    )


def status(report) -> dict[str, str]:
    return {c.name: c.status for c in report.checks}


async def test_conformant_route_passes_everything():
    fake = Fake()
    report = await run(fake)
    assert status(report) == {
        "streaming": PASS,
        "tool_round_trip": PASS,
        "first_token_latency": PASS,
        "italian": PASS,
    }
    assert report.ok and report.first_token_s is not None and report.tokens > 0
    for body in fake.bodies:
        assert body["stream"] is True and "response_format" not in body and "seed" not in body
    assert fake.bodies[1]["tool_choice"] == "auto"
    sent_back = fake.bodies[2]["messages"]
    assert sent_back[-1]["role"] == "tool" and sent_back[-2]["tool_calls"][0]["id"] == "c1"


async def test_single_chunk_is_not_streaming():
    report = await run(Fake(plain=reply([chunk("Ciao, oggi c'è il sole e il cielo è bello.")])))
    assert status(report)["streaming"] == FAIL and not report.ok


async def test_no_tool_call_fails_the_round_trip():
    report = await run(Fake(tool=reply(text_chunks(["Non ", "lo so."]))))
    result = next(c for c in report.checks if c.name == "tool_round_trip")
    assert result.status == FAIL and "no tool call" in result.reason


async def test_malformed_tool_arguments_fail():
    bad = [
        chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{oops"},
                }
            ]
        )
    ]
    report = await run(Fake(tool=reply(bad)))
    result = next(c for c in report.checks if c.name == "tool_round_trip")
    assert result.status == FAIL and "not valid JSON" in result.reason


async def test_no_answer_after_tool_result_fails():
    report = await run(Fake(final=reply([chunk(None)])))
    result = next(c for c in report.checks if c.name == "tool_round_trip")
    assert result.status == FAIL and "no text answer" in result.reason


async def test_slow_first_token_fails_latency_only():
    fake = Fake(plain=reply(text_chunks(ITALIAN), delay=0.4))
    report = await run(fake, route(first_token_timeout_s=0.1))
    s = status(report)
    assert s["first_token_latency"] == FAIL
    assert s["streaming"] == FAIL and s["italian"] == SKIP
    assert s["tool_round_trip"] == PASS


async def test_english_answer_fails_italian():
    report = await run(Fake(plain=reply(text_chunks(ENGLISH))))
    s = status(report)
    assert s["italian"] == FAIL and s["streaming"] == PASS


async def test_http_error_fails_without_server_text_or_key():
    report = await run(Fake(status=401))
    s = status(report)
    assert s["streaming"] == FAIL and s["tool_round_trip"] == FAIL
    dumped = json.dumps(report.to_dict())
    assert "HTTP 401" in dumped
    for leaked in ("prompt echo", SECRET_KEY, "sunny", "get_time"):
        assert leaked not in dumped


async def test_unreachable_route_skips_the_tool_check():
    async def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    report = await check_route(
        route(),
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    s = status(report)
    assert s["tool_round_trip"] == SKIP and s["streaming"] == FAIL and not report.ok


async def test_budget_exhausted_is_refused_before_any_request():
    fake = Fake()
    r = route(daily_token_budget=100)
    router = Router(
        [r],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    router._charge("lan", 100)
    with pytest.raises(RouteBudgetExhausted):
        await check_route(r, SafetyConfig(), router=router)
    assert fake.bodies == []


async def test_tokens_are_charged_to_the_shared_router():
    fake = Fake()
    r = route()
    router = Router(
        [r],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    report = await check_route(r, SafetyConfig(), router=router)
    assert router.used("lan") == report.tokens > 0


async def test_budget_spent_midway_skips_the_tool_check():
    fake = Fake()
    r = route(daily_token_budget=1)
    report = await run(fake, r)
    assert status(report)["tool_round_trip"] == SKIP and len(fake.bodies) == 1


def test_italian_heuristic():
    cfg = LLMCheckConfig()
    assert (
        italian_ratio("Ciao, oggi c'è il sole e il cielo è bello.", cfg.italian_stopwords)[1] > 0.3
    )
    assert italian_ratio("Hello, today the sun is shining.", cfg.italian_stopwords)[1] == 0
    assert italian_ratio("", cfg.italian_stopwords) == (0, 0.0)
