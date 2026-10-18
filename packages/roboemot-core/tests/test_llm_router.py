"""SPEC 8: OpenAI-compatible routes, fallback, minors, budgets, tools, text-only messages."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import time

import httpx
import pytest
from roboemot.config.schema import LLMRoute, SafetyConfig
from roboemot.llm import Done, NoRouteAvailable, Router, TextDelta, ToolCall, build_messages


def sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def chunk(content=None, tool_calls=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    c = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    if usage:
        c["usage"] = usage
    return c


class Server:
    """Fake OpenAI-compatible servers keyed by host."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        host = request.url.host
        self.requests.append((host, body))
        if host == "slow.local":
            await asyncio.sleep(1.0)
        if host == "down.local":
            return httpx.Response(503, json={"error": {"message": "down"}})
        if host == "tools.local":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse(
                    [
                        chunk(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "play", "arguments": '{"prim'},
                                }
                            ]
                        ),
                        chunk(
                            tool_calls=[{"index": 0, "function": {"arguments": 'itive": "nod"}'}}]
                        ),
                    ]
                ),
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                [
                    chunk("Ciao"),
                    chunk(" Marcello!"),
                    chunk(usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}),
                ]
            ),
        )


def route(name, host, **kw) -> LLMRoute:
    return LLMRoute(
        name=name,
        base_url=f"http://{host}/v1",
        model="m",
        first_token_timeout_s=kw.pop("timeout", 0.3),
        **kw,
    )


def router(server: Server, routes, **kw) -> Router:
    return Router(
        routes,
        kw.pop("safety", SafetyConfig()),
        http_client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(server.handler)
        ),
        **kw,
    )


async def collect(r: Router, **kw):
    msgs = [{"role": "user", "content": "ciao"}]
    return [ev async for ev in r.stream(msgs, **kw)]


async def test_streams_text_and_charges_reported_usage():
    srv = Server()
    r = router(srv, [route("lan", "lan.local")])
    evs = await collect(r)
    assert "".join(e.text for e in evs if isinstance(e, TextDelta)) == "Ciao Marcello!"
    done = evs[-1]
    assert isinstance(done, Done) and done.tokens == 25 and not done.estimated
    body = srv.requests[0][1]
    assert body["stream"] is True and "response_format" not in body and "seed" not in body


async def test_first_token_timeout_excludes_client_setup():
    """Regression: a slow client construction (cold imports) is not a slow first token."""
    srv = Server()

    def slow_factory() -> httpx.AsyncClient:
        time.sleep(0.4)  # longer than the route's 0.3 s first-token timeout
        return httpx.AsyncClient(transport=httpx.MockTransport(srv.handler))

    r = Router([route("lan", "lan.local")], SafetyConfig(), http_client_factory=slow_factory)
    evs = await collect(r)
    assert isinstance(evs[-1], Done) and r.failures == []


async def test_fallback_on_timeout_and_error():
    srv = Server()
    r = router(
        srv, [route("slow", "slow.local"), route("down", "down.local"), route("lan", "lan.local")]
    )
    evs = await collect(r)
    assert evs[-1].route == "lan"
    assert [f[0] for f in r.failures] == ["slow", "down"]


async def test_no_route_raises_for_fallback_phrase():
    r = router(Server(), [route("down", "down.local")])
    with pytest.raises(NoRouteAvailable):
        await collect(r)


async def test_minor_present_requires_route_and_safety_clearance():
    srv = Server()
    routes = [route("cloud", "lan.local", allowed_for_minors=True)]
    with pytest.raises(NoRouteAvailable):
        await collect(router(srv, routes), minor_present=True)  # safety default: not cleared
    ok = router(srv, routes, safety=SafetyConfig(minors_allowed_llm=True))
    assert isinstance((await collect(ok, minor_present=True))[-1], Done)


async def test_daily_budget_skips_route_and_resets_next_day():
    srv = Server()
    day = [dt.date(2026, 10, 14)]
    r = router(
        srv,
        [route("lan", "lan.local", daily_token_budget=30), route("b", "lan.local")],
        today=lambda: day[0],
    )
    assert (await collect(r))[-1].route == "lan"  # 25 used
    assert (await collect(r))[-1].route == "lan"  # 25 < 30 still eligible, now 50
    assert (await collect(r))[-1].route == "b"
    day[0] = dt.date(2026, 10, 15)
    assert (await collect(r))[-1].route == "lan"


async def test_tool_calls_assembled_with_auto_choice():
    srv = Server()
    r = router(srv, [route("t", "tools.local")])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "play",
                "parameters": {"type": "object", "properties": {"primitive": {"type": "string"}}},
            },
        }
    ]
    evs = [ev async for ev in r.stream([{"role": "user", "content": "annuisci"}], tools)]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.name == "play" and call.arguments == {"primitive": "nod"}
    assert srv.requests[0][1]["tool_choice"] == "auto"
    assert evs[-1].estimated  # this server reported no usage


async def test_secret_resolution():
    srv = Server()
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        return await srv.handler(request)

    r = Router(
        [route("c", "lan.local", api_key="${secret:cloud_key}")],
        SafetyConfig(),
        secrets=lambda n: {"cloud_key": "sk-test"}[n],
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await collect(r)
    assert seen["auth"] == "Bearer sk-test"


def test_build_messages_text_only_with_pseudonyms():
    msgs = build_messages(
        system="S",
        profile_summary="Marcello, adult",
        scene_summary="Marcello and a child",
        history=[{"role": "assistant", "content": "Ciao Marcello"}],
        user_text="sono Marcello",
        names={"Marcello": "Person A"},
    )
    text = json.dumps(msgs)
    assert "Marcello" not in text and "Person A" in text
    assert all(isinstance(m["content"], str) for m in msgs)


# ----------------------------------------------------------------------------- review fixes


class StallServer(Server):
    """First chunk carries only the role, then the server hangs (review finding 8)."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "stall.local":
            return await super().handler(request)
        server = self

        async def body():
            try:
                role = {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(role)}\n\n".encode()
                await asyncio.sleep(5)
                yield b"data: [DONE]\n\n"
            finally:
                server.closed += 1

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())


async def test_role_only_first_chunk_does_not_satisfy_first_token_timeout():
    srv = StallServer()
    r = router(srv, [route("stall", "stall.local", timeout=0.3), route("lan", "lan.local")])
    t0 = asyncio.get_running_loop().time()
    evs = await collect(r)
    assert evs[-1].route == "lan"
    assert asyncio.get_running_loop().time() - t0 < 1.5
    assert r.failures[0] == ("stall", "TimeoutError")
    assert srv.closed == 1  # abandoned stream closed


async def test_barge_in_still_charges_budget_and_closes():
    srv = Server()
    r = router(srv, [route("lan", "lan.local")])
    async with contextlib.aclosing(r.stream([{"role": "user", "content": "x" * 400}])) as gen:
        async for ev in gen:
            assert isinstance(ev, TextDelta)
            break  # user barges in after the first words
    assert r.used("lan") >= 100  # prompt (400 chars) + text estimate, charged on close


def _tc(index, call_id=None, name=None, args=None):
    fn = {k: v for k, v in (("name", name), ("arguments", args)) if v is not None}
    item = {"index": index, "function": fn}
    if call_id:
        item.update(id=call_id, type="function")
    return chunk(tool_calls=[item])


def _finish(c):
    c["choices"][0]["finish_reason"] = "tool_calls"
    return c


async def _held_stream(chunks):
    """A router on a server that sends ``chunks``, then holds the stream open until released."""
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n".encode()
            await release.wait()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    r = Router(
        [route("t", "tools.local", timeout=2.0)],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return r, release


async def test_a_tool_call_is_yielded_when_the_next_one_starts_not_when_the_stream_ends():
    r, release = await _held_stream(
        [
            _tc(0, "c1", "play", '{"gesture": "joy"}'),
            _tc(1, "c2", "look_at", '{"target"'),  # still being written when we look
        ]
    )
    stream = r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])
    first = await asyncio.wait_for(stream.__anext__(), 1.0)  # the stream is still open
    assert isinstance(first, ToolCall) and first.id == "c1"
    assert first.arguments == {"gesture": "joy"}
    release.set()
    rest = [ev async for ev in stream]
    assert [type(e) for e in rest] == [ToolCall, Done]  # the second, completed at the end


async def test_finish_reason_completes_the_last_call_and_events_keep_their_order():
    r, release = await _held_stream(
        [
            chunk("Ecco."),
            _tc(0, "c1", "play", '{"gesture": "joy"}'),
            _tc(1, "c2", "look_at", '{"target": "speaker"}'),
            _finish(chunk(content="")),  # finish_reason arrives, the HTTP stream stays open
        ]
    )
    seen = []
    stream = r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])
    while len([e for e in seen if isinstance(e, ToolCall)]) < 2:
        seen.append(await asyncio.wait_for(stream.__anext__(), 1.0))
    assert [type(e).__name__ for e in seen] == ["TextDelta", "ToolCall", "ToolCall"]
    assert [e.id for e in seen[1:]] == ["c1", "c2"]
    release.set()
    tail = [ev async for ev in stream]
    assert [type(e) for e in tail] == [Done]  # no call twice, Done last


async def test_bad_json_arguments_come_back_raw_and_the_stream_goes_on():
    r, release = await _held_stream([_tc(0, "c1", "play", '{"gesture": '), _finish(chunk(""))])
    release.set()
    evs = [ev async for ev in r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.arguments == {"_raw": '{"gesture": '}
