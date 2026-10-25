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
from roboemot.llm.router import FirstToken


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


async def test_first_token_timeout_excludes_openai_lazy_imports(monkeypatch):
    """Regression: the first ``client.chat`` access imports the resources lazily (tens of ms warm,
    far more on a loaded machine); it is not a slow first token."""
    import openai

    real = openai.AsyncOpenAI.chat

    def slow_chat(self):
        time.sleep(0.4)  # longer than the route's 0.3 s first-token timeout
        return real.__get__(self)

    monkeypatch.setattr(openai.AsyncOpenAI, "chat", property(slow_chat))
    r = router(Server(), [route("lan", "lan.local")])
    evs = await collect(r)
    assert isinstance(evs[-1], Done) and r.failures == []


async def test_a_failing_client_setup_closes_the_client_and_falls_back(monkeypatch):
    """The lazy ``client.chat`` access sits inside the stream's ``try``: when it raises, the
    client is still closed and the router goes on to the next route."""
    import openai

    real = openai.AsyncOpenAI.chat
    closed = []
    real_close = openai.AsyncOpenAI.close

    def chat(self):
        if self.base_url.host == "broken.local":
            raise RuntimeError("lazy import failed")
        return real.__get__(self)

    async def close(self):
        closed.append(str(self.base_url.host))
        await real_close(self)

    monkeypatch.setattr(openai.AsyncOpenAI, "chat", property(chat))
    monkeypatch.setattr(openai.AsyncOpenAI, "close", close)
    srv = Server()
    r = router(srv, [route("broken", "broken.local"), route("lan", "lan.local")])
    evs = await collect(r)
    assert evs[-1].route == "lan" and "broken.local" in closed
    assert [h for h, _ in srv.requests] == ["lan.local"]


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


async def test_hostile_tool_arguments_do_not_fail_the_route_or_get_parsed():
    huge_int = "9" * 6000  # json.loads raises ValueError past the interpreter's digit limit
    deep = "[" * 20_000 + "]" * 20_000  # RecursionError in the parser
    for text, expect in (
        ('{"n": ' + huge_int + "}", "raw"),
        (deep, "raw"),
        (
            '{"a":' * 5_000 + "1" + "}" * 5_000,
            "raw",
        ),  # parses everywhere; restoration would recurse
        ("[1, 2]", "raw"),  # valid JSON that is not an object
        ('{"a": "' + "x" * 60_000 + '"}', "dropped"),
    ):
        r, release = await _held_stream([_tc(0, "c1", "play", text), _finish(chunk(""))])
        release.set()
        evs = [
            ev async for ev in r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])
        ]
        call = next(e for e in evs if isinstance(e, ToolCall))
        assert list(call.arguments) == ["_raw"]
        assert (call.arguments["_raw"] == "") == (expect == "dropped")
        assert isinstance(evs[-1], Done) and r.failures == []  # no failover, charged once


async def test_deltas_without_an_index_are_grouped_by_their_id():
    def noindex(call_id, name, args):
        fn = {k: v for k, v in (("name", name), ("arguments", args)) if v is not None}
        item = {"function": fn}
        if call_id:
            item.update(id=call_id, type="function")
        return chunk(tool_calls=[item])

    r, release = await _held_stream(
        [noindex("c1", "play", '{"gesture"'), noindex(None, None, ': "joy"}'),
         noindex("c2", "look_at", '{"target": "away"}'), _finish(chunk(""))]
    )  # fmt: skip
    release.set()
    evs = [ev async for ev in r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])]
    calls = [e for e in evs if isinstance(e, ToolCall)]
    assert [(c.id, c.arguments) for c in calls] == [
        ("c1", {"gesture": "joy"}),
        ("c2", {"target": "away"}),
    ]


def _nested(depth: int, kind: str) -> object:
    """A JSON value ``depth`` containers deep: lists, dicts, or alternating dict/list."""
    value: object = 1
    for level in range(depth):
        value = [value] if kind == "list" or (kind == "mixed" and level % 2) else {"a": value}
    return value


@pytest.mark.parametrize("kind", ["list", "dict", "mixed"])
def test_nesting_depth_boundaries(kind):
    from roboemot.llm.router import TOOL_ARGS_MAX_DEPTH, _nesting

    assert _nesting(1) == 0 and _nesting({}) == 1
    assert _nesting(_nested(TOOL_ARGS_MAX_DEPTH, kind)) == TOOL_ARGS_MAX_DEPTH
    assert _nesting(_nested(TOOL_ARGS_MAX_DEPTH + 1, kind)) > TOOL_ARGS_MAX_DEPTH


async def test_arguments_at_the_depth_limit_are_kept_and_one_deeper_is_raw():
    from roboemot.llm.router import TOOL_ARGS_MAX_DEPTH

    for depth, kept in ((TOOL_ARGS_MAX_DEPTH, True), (TOOL_ARGS_MAX_DEPTH + 1, False)):
        text = json.dumps({"a": _nested(depth - 1, "mixed")})
        r, release = await _held_stream([_tc(0, "c1", "play", text), _finish(chunk(""))])
        release.set()
        evs = [
            ev async for ev in r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])
        ]
        call = next(e for e in evs if isinstance(e, ToolCall))
        assert ("_raw" not in call.arguments) is kept


async def test_a_wide_flat_list_under_the_char_cap_is_kept():
    from roboemot.llm.router import TOOL_ARGS_MAX_CHARS

    items = TOOL_ARGS_MAX_CHARS // 2 - 10
    text = json.dumps({"xs": [1] * items}, separators=(",", ":"))
    assert len(text) <= TOOL_ARGS_MAX_CHARS
    r, release = await _held_stream([_tc(0, "c1", "play", text), _finish(chunk(""))])
    release.set()
    evs = [ev async for ev in r.stream([{"role": "user", "content": "x"}], [{"type": "function"}])]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert len(call.arguments["xs"]) == items


async def test_deep_arguments_from_a_cloud_route_with_pseudonyms_come_back_raw():
    """End to end: the restoration recurses, so too-deep arguments must never reach it."""
    from roboemot.llm.pseudonyms import Pseudonyms
    from roboemot.llm.router import TOOL_ARGS_MAX_DEPTH

    text = json.dumps(_nested(TOOL_ARGS_MAX_DEPTH + 1, "dict"))
    body = sse([_tc(0, "c1", "play", text), _finish(chunk(""))])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    names = Pseudonyms()
    names.sync(["Marta"], {"en": "Person-{label}"}, "en", "en")
    r = Router(
        [route("cloud", "api.example.com", timeout=2.0)],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    messages = [{"role": "user", "content": "ciao Marta"}]
    evs = [ev async for ev in r.stream(messages, [{"type": "function"}], pseudonyms=names)]
    call = next(e for e in evs if isinstance(e, ToolCall))
    assert call.arguments == {"_raw": text} and r.failures == []


def test_first_token_timeout_default_fits_cloud_routes():
    # robot 2026-10-23: Claude Haiku answered in 0.8 to 1.8 s, so 1.5 s failed 5 turns of 8
    assert (
        LLMRoute(name="c", base_url="https://api.example.com/v1", model="m").first_token_timeout_s
        == 3.0
    )


def _canned(chunks: list[dict]):
    """A server that answers every request with ``chunks`` and records the request bodies."""
    bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(chunks)
        )

    return handler, bodies


def _reasoning(text: str, field: str = "reasoning") -> dict:
    c = chunk()
    c["choices"][0]["delta"][field] = text
    return c


def _ended(c: dict, reason: str) -> dict:
    c["choices"][0]["finish_reason"] = reason
    return c


async def test_reasoning_deltas_are_counted_never_yielded_or_requested():
    handler, bodies = _canned(
        [_reasoning("let me think"), _reasoning("... ok", "reasoning_content"), chunk("Ciao")]
    )
    r = Router(
        [route("lan", "lan.local", max_tokens=300)],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    timing = FirstToken()
    msgs = [{"role": "user", "content": "x"}]
    evs = [ev async for ev in r.stream_route(r.routes[0], msgs, None, timing)]
    assert [e.text for e in evs if isinstance(e, TextDelta)] == ["Ciao"]
    assert timing.reasoning_chars == len("let me think") + len("... ok")
    assert evs[-1].reasoning_chars == timing.reasoning_chars
    body = bodies[0]
    assert body["max_tokens"] == 300
    assert not [k for k in body if "reason" in k or k == "include_reasoning"]


async def test_length_finish_with_only_reasoning_is_recorded():
    handler, _ = _canned([_reasoning("x" * 40), _ended(chunk(), "length")])
    r = Router(
        [route("lan", "lan.local")],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    timing = FirstToken()
    msgs = [{"role": "user", "content": "x"}]
    evs = [ev async for ev in r.stream_route(r.routes[0], msgs, None, timing)]
    assert [type(e) for e in evs] == [Done]
    assert timing.seconds is None  # reasoning is not a first token
    assert timing.reasoning_chars == 40 and timing.finish_reason == "length"
    assert evs[0].finish_reason == "length"


async def test_reasoning_counts_in_the_estimated_charge_without_usage():
    handler, _ = _canned([_reasoning("x" * 400), _ended(chunk(), "length")])
    r = Router(
        [route("lan", "lan.local")],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    msgs = [{"role": "user", "content": "x"}]
    evs = [ev async for ev in r.stream_route(r.routes[0], msgs, None, FirstToken())]
    assert r.used("lan") == (1 + 400) // 4 == evs[-1].tokens and evs[-1].estimated


async def test_reasoning_streamed_in_both_fields_counts_once():
    both = _reasoning("abcdef")
    both["choices"][0]["delta"]["reasoning_content"] = "abcdef"
    handler, _ = _canned([both, chunk("Ciao")])
    r = Router(
        [route("lan", "lan.local")],
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    timing = FirstToken()
    msgs = [{"role": "user", "content": "x"}]
    _ = [ev async for ev in r.stream_route(r.routes[0], msgs, None, timing)]
    assert timing.reasoning_chars == 6
