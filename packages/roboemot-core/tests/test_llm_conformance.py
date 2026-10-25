"""SPEC 8: per-route conformance test (streaming, tool round trip, latency, Italian)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import APIStatusError
from roboemot.config.schema import LLMCheckConfig, LLMRoute, RoboEmotConfig, SafetyConfig
from roboemot.llm import Router
from roboemot.llm.conformance import (
    ECHO_WITHHELD,
    FAIL,
    PASS,
    SERVER_TEXT_MAX_CHARS,
    SKIP,
    USER_TURN,
    RouteBudgetExhausted,
    _why,
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
    assert result.status == FAIL and "not a JSON object" in result.reason


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


async def test_http_error_fails_with_the_server_message_and_no_key():
    report = await run(Fake(status=401))
    s = status(report)
    assert s["streaming"] == FAIL and s["tool_round_trip"] == FAIL
    dumped = json.dumps(report.to_dict())
    assert "HTTP 401 at /v1/chat/completions: prompt echo" in dumped
    for leaked in (SECRET_KEY, "sunny", "get_time"):
        assert leaked not in dumped


def status_error(body, url="https://api.example.com/v1/chat/completions"):
    request = httpx.Request("POST", url)
    return APIStatusError("x", response=httpx.Response(404, request=request), body=body)


R = LLMRoute(name="c", base_url="https://api.example.com/v1", model="m")


def test_http_error_names_a_retired_model():
    # the 2026-10-22 robot test: a preset still on a retired ":free" model gave a bare 404
    message = "The `:free` variant has been retired; use `minimax/minimax-m3.1-flash` instead."
    assert _why(status_error({"message": message, "code": 404}), R) == (
        f"HTTP 404 at /v1/chat/completions: {message}"
    )


def test_http_error_reads_an_unwrapped_error_body():
    assert _why(status_error({"error": {"message": "no endpoints"}}), R).endswith(": no endpoints")


@pytest.mark.parametrize(
    "body", [None, "plain text", {"message": 404}, {"error": "x"}, {"message": "  \n "}]
)
def test_http_error_without_a_message_shows_the_status_only(body):
    assert _why(status_error(body), R) == "HTTP 404 at /v1/chat/completions"


def test_http_error_message_masks_keys_and_ids():
    key = "sk-or-v1-" + "ab12" * 16
    reason = _why(status_error({"message": f"bad key {key} for user_3GidxfYvzBvtsyfIfp6Y"}), R)
    assert key not in reason and "3GidxfYvzBvtsyfIfp6Y" not in reason
    assert reason.endswith(": bad key [masked] for [masked]")


def test_http_error_message_masks_a_short_literal_key():
    r = LLMRoute(name="c", base_url="https://api.example.com/v1", model="m", api_key="k-123")
    assert _why(status_error({"message": "key k-123 is invalid"}), r).endswith(
        ": key [masked] is invalid"
    )


def test_http_error_message_is_one_bounded_line():
    reason = _why(status_error({"message": "a\nb\x1b[31m  c " + "word " * 100}), R)
    shown = reason.split(": ", 1)[1]
    assert shown.startswith("a b [31m c word")
    assert len(shown) == SERVER_TEXT_MAX_CHARS and shown.endswith("…")


async def test_http_error_names_the_path_called_without_host_or_credentials():
    r = LLMRoute(
        name="c", base_url="https://u:p@api.example.com/api", model="m", api_key=SECRET_KEY
    )
    report = await run(Fake(status=404), r)
    reasons = [c.reason for c in report.checks if c.status == FAIL]
    assert "no streamed text: HTTP 404 at /api/chat/completions: prompt echo" in reasons
    dumped = json.dumps(report.to_dict())
    for leaked in ("u:p", "api.example.com/api", SECRET_KEY):
        assert leaked not in dumped


async def test_http_error_path_leaves_out_the_query():
    r = LLMRoute(name="c", base_url="https://api.example.com/v1?key=k1", model="m")
    report = await run(Fake(status=404), r)
    assert "k1" not in json.dumps(report.to_dict())


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("https://api.example.com/v1/chat/completions", " at /v1/chat/completions"),
        ("https://api.example.com/tok_s3cr3t/x", ""),  # a redirect elsewhere
        ("https://api.example.com/v1x/chat/completions", ""),  # not under /v1/
        ("https://api.example.com/v1/%3Cb%3E", ""),  # not plain characters
        ("https://api.example.com/v1/" + "a" * 200, ""),  # too long
    ],
)
def test_http_error_path_only_under_the_route_base_path(url, shown):
    request = httpx.Request("POST", url)
    exc = APIStatusError("x", response=httpx.Response(404, request=request), body=None)
    r = LLMRoute(name="c", base_url="https://api.example.com/v1", model="m")
    assert _why(exc, r) == f"HTTP 404{shown}"


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


async def test_stream_that_breaks_midway_fails_streaming():
    class Cut(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse(text_chunks(["Ciao, ", "oggi "]))[: -len(b"data: [DONE]\n\n")]
            raise httpx.ReadError("cut")

    async def handler(request):
        body = json.loads(request.content)
        if "tools" in body:
            return await Fake().handler(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Cut())

    report = await check_route(
        route(),
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = next(c for c in report.checks if c.name == "streaming")
    assert result.status == FAIL and "stream broke after 2 chunks" in result.reason


async def test_report_carries_model_and_endpoint_without_credentials():
    r = LLMRoute(
        name="lan", base_url="https://u:p@lan.local:8443/v1?k=1", model="m", api_key=SECRET_KEY
    )
    report = await run(Fake(), r)
    assert (report.model, report.endpoint) == ("m", "lan.local:8443")


def _named(name: str):
    from roboemot.llm.pseudonyms import Pseudonyms

    p = Pseudonyms()
    p.sync([name], {"en": "Person-{label}"}, "en", "en")
    return p


async def test_a_cloud_route_check_goes_through_the_pseudonyms():
    fake = Fake()
    cloud = LLMRoute(
        name="c", base_url="http://api.example.com/v1", model="m", first_token_timeout_s=0.5
    )
    report = await run(fake, cloud, pseudonyms=_named("Italian"))
    assert report.ok
    assert fake.bodies and all("Italian" not in json.dumps(b) for b in fake.bodies)
    assert "Person-" in json.dumps(fake.bodies[0])


async def test_a_local_route_check_keeps_the_prompt_as_it_is():
    fake = Fake()
    await run(fake, pseudonyms=_named("Italian"))
    assert "Italian" in json.dumps(fake.bodies[0])


async def test_a_leaking_request_fails_the_check_and_is_not_sent(monkeypatch):
    p = _named("Italian")
    real_prepare = p.prepare

    def unscrubbed(messages, tools):
        out = real_prepare(messages, tools)
        out.messages, out.tools = messages, tools
        return out

    monkeypatch.setattr(p, "prepare", unscrubbed)
    fake = Fake()
    cloud = LLMRoute(
        name="c", base_url="http://api.example.com/v1", model="m", first_token_timeout_s=0.5
    )
    report = await run(fake, cloud, pseudonyms=p)
    assert all("Italian" not in json.dumps(b) for b in fake.bodies)  # the leaking request: not sent
    assert not report.ok
    streaming = next(c for c in report.checks if c.name == "streaming")
    assert streaming.status == FAIL and "PseudonymLeak" in streaming.reason


async def test_http_error_masks_a_stored_key_and_its_fragments():
    stored = "Ab9+xY/2"  # short, with "/" (ends a key-like run): only the sent key masks it
    label = "sk-or-v1-74c0d2e9"

    class Echo(Fake):
        async def handler(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"message": f"key {stored} (sk-or-v1-74c...e9) no ${{secret:k}}"}},
            )

    r = LLMRoute(name="c", base_url="https://api.example.com/v1", model="m", api_key="${secret:k}")
    report = await run(Echo(), r, secrets=lambda name: stored if name == "k" else "")
    dumped = json.dumps(report.to_dict())
    assert stored not in dumped and "${secret:k}" not in dumped
    assert "no streamed text: HTTP 404 at /v1/chat/completions: key [masked]" in dumped
    r2 = LLMRoute(name="c", base_url="https://api.example.com/v1", model="m", api_key=label)
    report = await run(Echo(), r2)
    assert "74c" not in json.dumps(report.to_dict())


# ------------------------------------------------------------ realistic turn, reasoning models


def thought(text: str) -> dict:
    c = chunk()
    c["choices"][0]["delta"]["reasoning"] = text
    return c


def ended(c: dict, reason: str = "length") -> dict:
    c["choices"][0]["finish_reason"] = reason
    return c


def reason(report, name: str) -> str:
    return next(c.reason for c in report.checks if c.name == name)


async def test_first_request_is_a_realistic_turn_at_the_routes_max_tokens():
    fake = Fake()
    cfg = RoboEmotConfig.model_validate(
        {"conversation": {"language": "en"}, "persona": {"prompt": "PERSONA-4c1d"}}
    )
    await run(fake, route(max_tokens=300), config=cfg)
    body = fake.bodies[0]
    system, user = body["messages"][0], body["messages"][-1]
    assert system["role"] == "system" and "PERSONA-4c1d" in system["content"]
    assert "Always reply in Italian." in system["content"]  # whatever the conversation language
    assert user == {"role": "user", "content": USER_TURN}
    assert body["max_tokens"] == 300 and "tools" not in body
    assert not [k for k in body if "reason" in k]  # nothing non-standard is asked for


async def test_reasoning_that_uses_the_whole_budget_fails_with_a_hint():
    plain = reply([thought("x" * 25), thought("y" * 15), ended(chunk())])
    report = await run(Fake(plain=plain), route(max_tokens=300))
    assert status(report)["streaming"] == FAIL and not report.ok
    assert reason(report, "streaming") == (
        "ran out of tokens before answering (max_tokens 300); "
        "the model reasoned for 40 chars before answering"
    )
    assert status(report)["italian"] == SKIP
    assert report.reasoning_chars == 40 and report.finish_reason == "length"


async def test_length_with_no_text_fails_without_reasoning_fields_too():
    report = await run(Fake(plain=reply([ended(chunk())])), route(max_tokens=64))
    assert reason(report, "streaming") == "ran out of tokens before answering (max_tokens 64)"
    assert report.reasoning_chars == 0


async def test_reasoning_then_text_passes_and_shows_the_hint():
    plain = reply([thought("z" * 30), *text_chunks(ITALIAN)])
    report = await run(Fake(plain=plain))
    assert report.ok and status(report)["streaming"] == PASS
    assert reason(report, "first_token_latency").endswith(
        "; the model reasoned for 30 chars before answering"
    )
    assert report.reasoning_chars == 30


async def test_plain_text_has_no_reasoning_hint():
    report = await run(Fake())
    assert report.reasoning_chars == 0 and report.finish_reason is None
    assert all("reasoned" not in c.reason for c in report.checks)


async def test_reasoning_past_the_first_token_timeout_fails_latency_with_a_hint():
    gate = asyncio.Event()  # never set: the stream stays open until the router closes it

    async def slow() -> httpx.Response:
        async def body():
            yield sse([thought("w" * 12)])[: -len(b"data: [DONE]\n\n")]
            await gate.wait()
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    report = await run(Fake(plain=slow), route(first_token_timeout_s=0.3))
    assert status(report)["first_token_latency"] == FAIL
    assert reason(report, "first_token_latency") == (
        "no content within 0.3 s; the model reasoned for 12 chars before answering"
    )
    assert reason(report, "streaming").endswith(
        "; the model reasoned for 12 chars before answering"
    )
    assert report.reasoning_chars == 12 and report.first_token_s is None


async def test_answer_cut_at_max_tokens_still_streams():
    plain = reply([*text_chunks(ITALIAN[:-1]), ended(chunk(ITALIAN[-1]))])
    report = await run(Fake(plain=plain), route(max_tokens=50))
    assert status(report)["streaming"] == PASS
    assert reason(report, "streaming") == "4 chunks, cut at max_tokens 50"


async def test_tool_answer_out_of_tokens_is_named():
    report = await run(Fake(final=reply([thought("q" * 9), ended(chunk())])), route(max_tokens=80))
    assert status(report)["tool_round_trip"] == FAIL
    assert reason(report, "tool_round_trip") == (
        "ran out of tokens before answering (max_tokens 80); "
        "the model reasoned for 9 chars before answering"
    )


async def test_mood_tags_do_not_count_against_italian():
    tagged = ["[mood=joy] Ciao, ", "oggi c'è ", "[mood=calm] il sole ", "e il cielo è bello."]
    report = await run(Fake(plain=reply(text_chunks(tagged))))
    assert status(report)["italian"] == PASS


async def test_reasons_never_hold_the_prompt_or_the_reasoning():
    cfg = RoboEmotConfig.model_validate({"persona": {"prompt": "PERSONA-4c1d"}})
    plain = reply([thought("THOUGHT-8e2b"), ended(chunk())])
    report = await run(Fake(plain=plain), config=cfg)
    dumped = json.dumps(report.to_dict())
    for secret in ("PERSONA-4c1d", "THOUGHT-8e2b", USER_TURN, SECRET_KEY):
        assert secret not in dumped


PRIVATE_PERSONA = "You live with Marta in a small flat and you love her cat."


def cloud_route(**kw) -> LLMRoute:
    kw.setdefault("first_token_timeout_s", 0.5)
    return LLMRoute(name="cloud", base_url="https://api.example.com/v1", model="m", **kw)


async def test_pseudonyms_on_without_a_name_map_keep_free_text_off_a_cloud_route():
    cfg = RoboEmotConfig.model_validate(
        {"persona": {"prompt": PRIVATE_PERSONA}, "privacy": {"pseudonyms_to_cloud": True}}
    )
    fake = Fake()
    await run(fake, cloud_route(), config=cfg)
    sent = json.dumps(fake.bodies)
    assert "Marta" not in sent and "Always reply in Italian." in sent


async def test_a_local_route_gets_the_owners_persona_with_pseudonyms_on():
    cfg = RoboEmotConfig.model_validate(
        {"persona": {"prompt": PRIVATE_PERSONA}, "privacy": {"pseudonyms_to_cloud": True}}
    )
    fake = Fake()
    await run(fake, route(), config=cfg)
    assert "Marta" in fake.bodies[0]["messages"][0]["content"]


def test_an_echoed_prompt_withholds_the_whole_server_message():
    message = f"Invalid request: '{PRIVATE_PERSONA.upper()}' is too long for model org/m-3.5"
    reason = _why(status_error({"message": message}), R, sent=(PRIVATE_PERSONA, USER_TURN))
    assert reason == f"HTTP 404 at /v1/chat/completions: {ECHO_WITHHELD}"


def test_a_server_message_that_quotes_nothing_is_shown():
    message = "The `:free` variant has been retired; use `minimax/minimax-m3.1-flash` instead."
    reason = _why(status_error({"message": message}), R, sent=(PRIVATE_PERSONA, USER_TURN))
    assert reason.endswith(f": {message}")


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_an_echo_written_as_json_is_withheld(ensure_ascii):
    system = "Sei Neve.\nLa tua padrona è Marta Rossi, c'è sempre il sole.\nYou live in Torino."
    message = "bad request body: " + json.dumps({"content": system}, ensure_ascii=ensure_ascii)
    reason = _why(status_error({"message": message}), R, sent=(system,))
    assert reason.endswith(f": {ECHO_WITHHELD}")


async def test_an_echo_of_the_realistic_turn_never_reaches_the_report():
    cfg = RoboEmotConfig.model_validate({"persona": {"prompt": PRIVATE_PERSONA}})

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        echo = " ".join(str(m.get("content") or "") for m in body["messages"])
        return httpx.Response(400, json={"error": {"message": f"bad: {echo}"[:150]}})

    report = await check_route(
        route(),
        SafetyConfig(),
        config=cfg,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    dumped = json.dumps(report.to_dict())
    assert "Marta" not in dumped and "small flat" not in dumped and "HTTP 400" in dumped
    assert ECHO_WITHHELD in dumped


async def test_a_json_echo_of_the_request_body_never_reaches_the_report():
    # review 2026-10-25: "\n" in the escaped JSON broke the word runs, a short line leaked
    persona = "You are Neve, a small robot.\nYour owner is Marta Rossi.\nYou live in Torino."
    cfg = RoboEmotConfig.model_validate({"persona": {"prompt": persona}})

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            400, json={"error": {"message": "bad request body: " + json.dumps(body)}}
        )

    report = await check_route(
        route(),
        SafetyConfig(),
        config=cfg,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    dumped = json.dumps(report.to_dict())
    for word in ("Torino", "Marta", "Neve", "owner"):
        assert word not in dumped
    assert ECHO_WITHHELD in dumped


async def test_a_tool_request_out_of_tokens_is_named():
    report = await run(Fake(tool=reply([thought("r" * 7), ended(chunk())])), route(max_tokens=90))
    assert reason(report, "tool_round_trip") == (
        "ran out of tokens before answering (max_tokens 90); "
        "the model reasoned for 7 chars before answering"
    )


async def test_owner_persona_names_are_aliased_on_a_cloud_route_with_a_name_map():
    cfg = RoboEmotConfig.model_validate(
        {"persona": {"prompt": PRIVATE_PERSONA}, "privacy": {"pseudonyms_to_cloud": True}}
    )
    fake = Fake()
    await run(fake, cloud_route(), config=cfg, pseudonyms=_named("Marta"))
    system = fake.bodies[0]["messages"][0]["content"]
    assert "Marta" not in system and "Person-" in system and "small flat" in system
