"""SPEC 8, 11A, 12: every branch of the route conformance check keeps the English ``reason`` the
CLI prints (the texts below are the ones from before the web UI codes), and gives the ``code``
and ``params`` the page words in its own language."""

from __future__ import annotations

import asyncio

import pytest
from roboemot.config.schema import LLMCheckConfig, LLMRoute
from roboemot.llm import conformance as c
from roboemot.llm.router import Done, FirstToken, TextDelta, ToolCall

ROUTE = LLMRoute(name="lan", base_url="https://lan.local/v1", model="m", max_tokens=64)
WHY = {"why": "timeout"}


def answer(chunks=0, text="", error=None, finish=None, seconds=None, reasoned=0, calls=()):
    ans = c._Answer(text=text, chunks=chunks, finish_reason=finish, calls=list(calls))
    ans.timing = FirstToken(seconds=seconds, reasoning_chars=reasoned)
    if error is not None:
        ans.error, ans.why, ans.why_params = error, "timed out", dict(WHY)
    return ans


def verdict(result):
    return result.status, result.reason, result.code, result.params


LIMIT = ROUTE.first_token_timeout_s


@pytest.mark.parametrize(
    ("ans", "expected"),
    [
        (answer(error=TimeoutError(), reasoned=7), (
            "fail", "no streamed text: timed out; the model reasoned for 7 chars before answering",
            "no_stream", {"why": "timeout", "reasoned": 7})),
        (answer(chunks=3, error=TimeoutError()), (
            "fail", "stream broke after 3 chunks: timed out", "stream_broke",
            {"chunks": 3, "why": "timeout"})),
        (answer(), ("fail", "no text in the stream", "no_text_stream", {})),
        (answer(finish="length"), (
            "fail", "ran out of tokens before answering (max_tokens 64)", "out_of_tokens",
            {"max_tokens": 64})),
        (answer(chunks=1), ("fail", "text came as a single chunk", "single_chunk", {})),
        (answer(chunks=5), ("pass", "5 chunks", "chunks", {"chunks": 5})),
        (answer(chunks=5, finish="length"), (
            "pass", "5 chunks, cut at max_tokens 64", "chunks_cut",
            {"chunks": 5, "max_tokens": 64})),
    ],
)  # fmt: skip
def test_streaming_reasons(ans, expected):
    assert verdict(c._streaming(ans, ROUTE)) == expected


def test_latency_reasons():
    timed_out = answer(error=TimeoutError(), reasoned=3)
    assert verdict(c._latency(timed_out, ROUTE)) == (
        "fail", f"no content within {LIMIT:g} s; the model reasoned for 3 chars before answering",
        "no_content_within", {"limit": LIMIT, "reasoned": 3},
    )  # fmt: skip
    assert verdict(c._latency(answer(), ROUTE)) == (
        "skip", "no answer to measure", "no_answer", {}
    )  # fmt: skip
    assert verdict(c._latency(answer(seconds=0.4567), ROUTE)) == (
        "pass", f"0.46 s of {LIMIT:g} s allowed (one cold sample, TLS setup included)",
        "latency", {"seconds": 0.46, "limit": LIMIT},
    )  # fmt: skip


def test_italian_reasons():
    cfg = LLMCheckConfig()
    assert verdict(c._italian(answer(), cfg)) == ("skip", "no reply to judge", "no_reply", {})
    assert verdict(c._italian(answer(text="ciao"), cfg)) == (
        "fail", "reply too short to judge (1 words)", "too_short", {"count": 1}
    )  # fmt: skip
    text = "il cane e il gatto sono nel giardino con la palla e il sole " * 3
    result = c._italian(answer(text=text), cfg)
    count, ratio = c.italian_ratio(text, cfg.italian_stopwords)
    need = cfg.italian_min_stopword_ratio
    assert result.reason == f"{ratio:.0%} Italian stopwords, {need:.0%} needed"
    assert result.code == "italian_ratio"
    assert result.params == {"ratio": round(ratio * 100), "need": round(need * 100)}


class Router:
    """Two scripted answers for the tool round trip, and the budget it reads."""

    def __init__(self, *answers, used=0) -> None:
        self.answers, self.spent = list(answers), used

    def used(self, name: str) -> int:
        return self.spent

    def auth_headers(self, route) -> dict:
        return {}

    async def stream_route(self, route, messages, tools, timing, pseudonyms=None):
        events = self.answers.pop(0)
        if isinstance(events, Exception):
            raise events
        for ev in events:
            yield ev


CALL = ToolCall(id="c1", name=c.TOOL_NAME, arguments={})
DONE = Done("lan", 1, False)
TEXT = [TextDelta("Sono le dieci e mezza."), DONE]


@pytest.mark.parametrize(
    ("router", "unreachable", "expected"),
    [
        (Router(), True, ("skip", "route unreachable", "unreachable", {})),
        (Router(used=ROUTE.daily_token_budget), False, (
            "skip", "daily token budget spent by the earlier request", "budget_spent", {})),
        (Router(TimeoutError()), False, (
            "fail", "tool request failed: timed out", "tool_failed", {"why": "timeout"})),
        (Router([DONE]), False, ("fail", "no tool call in the reply", "no_tool_call", {})),
        (Router([TextDelta("no"), DONE]), False, (
            "fail", "no tool call in the reply", "no_tool_call", {})),
        (Router([ToolCall("c1", "other", {}), DONE]), False, (
            "fail", "tool call names an unknown tool", "unknown_tool", {})),
        (Router([ToolCall("c1", c.TOOL_NAME, {"_raw": "x"}), DONE]), False, (
            "fail", "tool call arguments are not a JSON object within limits", "bad_arguments",
            {})),
        (Router([ToolCall("", c.TOOL_NAME, {}), DONE]), False, (
            "fail", "tool call has no id", "no_call_id", {})),
        (Router([CALL, DONE], TimeoutError()), False, (
            "fail", "tool result rejected: timed out", "tool_result_rejected", {"why": "timeout"})),
        (Router([CALL, DONE], [DONE]), False, (
            "fail", "no text answer after the tool result", "no_text_after", {})),
        (Router([CALL, DONE], TEXT), False, (
            "pass", "call, result and answer", "round_trip_ok", {})),
    ],
)  # fmt: skip
def test_tool_round_trip_reasons(router, unreachable, expected):
    result = asyncio.run(c._tool_round_trip(router, ROUTE, unreachable))
    assert verdict(result) == expected


def test_a_spent_budget_before_the_answer():
    class Spending(Router):
        def used(self, name: str) -> int:
            return ROUTE.daily_token_budget if len(self.answers) < 2 else 0  # after the call

    result = asyncio.run(c._tool_round_trip(Spending([CALL, DONE], TEXT), ROUTE, False))
    assert verdict(result) == (
        "skip", "daily token budget spent before the answer", "budget_before_answer", {}
    )  # fmt: skip
