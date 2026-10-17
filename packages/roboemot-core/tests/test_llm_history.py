"""SPEC 8, Conversation state: per-route history trimming and the load-time budget check."""

from __future__ import annotations

import json

import httpx
import pytest
from roboemot.config import ConfigStack
from roboemot.config.schema import ConversationConfig, LLMRoute, RoboEmotConfig, SafetyConfig
from roboemot.config.stack import ConfigError
from roboemot.llm import Router
from roboemot.llm.prompt import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    split_turns,
    system_tokens_worst_case,
    trim_history,
)

ROUTE = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m1"}
LONG = {"prompt": "p" * 1500, "prompt_max_chars": 4000}


def msg(role: str, tokens: int, **extra) -> dict:
    return {"role": role, "content": "x" * tokens * CHARS_PER_TOKEN, **extra}


def convo(turns: int, tokens: int = 100) -> list[dict]:
    out = [msg("system", 50)]
    for i in range(turns):
        out += [msg("user", tokens, n=i), msg("assistant", tokens, n=i)]
    return out + [msg("user", tokens, n="now")]


def test_estimate_rounds_up_at_four_chars_per_token():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_within_budget_nothing_is_dropped():
    msgs = convo(3)
    out, dropped = trim_history(msgs, 10_000)
    assert out is msgs and dropped == 0


def test_oldest_turns_go_first_and_order_is_kept():
    msgs = convo(5)  # 50 + 5 * 200 + 100 = 1150 tokens
    out, dropped = trim_history(msgs, 750)
    assert dropped == 2
    assert [m.get("n") for m in out if "n" in m] == [2, 2, 3, 3, 4, 4, "now"]
    assert out[0] is msgs[0] and out[-1] is msgs[-1]


def test_trim_stops_as_soon_as_it_fits():
    assert trim_history(convo(5), 1150)[1] == 0
    assert trim_history(convo(5), 1150 - 1)[1] == 1


def test_a_turn_is_never_split():
    msgs = convo(4)
    for budget in range(0, 1000, 25):
        out, _ = trim_history(msgs, budget)
        kept = [m["n"] for m in out if "n" in m]
        assert kept[-1] == "now"
        for i in range(4):  # a turn is both messages or neither
            assert kept.count(i) in (0, 2)


def test_tool_exchanges_stay_with_their_turn():
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    msgs = [
        msg("system", 50),
        msg("user", 100, n=0),
        {"role": "assistant", "content": None, "tool_calls": calls, "n": 0},
        {"role": "tool", "tool_call_id": "c1", "content": "ok", "n": 0},
        msg("assistant", 100, n=0),
        msg("user", 100, n=1),
        msg("assistant", 100, n=1),
        msg("user", 100, n="now"),
    ]
    out, dropped = trim_history(msgs, 450)
    assert dropped == 1
    assert [m.get("n") for m in out if "n" in m] == [1, 1, "now"]
    assert not any(m["role"] == "tool" for m in out)  # never a tool result without its call


def test_the_system_prompt_and_the_newest_turn_survive_any_budget():
    msgs = convo(3)
    out, dropped = trim_history(msgs, 1)
    assert dropped == 3 and out == [msgs[0], msgs[-1]]


def test_the_newest_turn_keeps_its_tool_exchange():
    msgs = [
        msg("system", 50),
        msg("user", 100, n=0),
        msg("assistant", 100, n=0),
        msg("user", 100, n="now"),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}], "n": "now"},
        {"role": "tool", "tool_call_id": "c", "content": "ok", "n": "now"},
    ]
    out, _ = trim_history(msgs, 1)
    assert [m.get("n") for m in out if "n" in m] == ["now", "now", "now"]


def test_orphaned_leading_messages_are_the_first_to_go():
    msgs = [msg("system", 10), msg("assistant", 100, n="orphan"), msg("user", 100, n="now")]
    head, turns = split_turns(msgs)
    assert len(head) == 1 and [len(t) for t in turns] == [1, 1]
    out, dropped = trim_history(msgs, 150)
    assert dropped == 1 and out == [msgs[0], msgs[2]]


def test_several_system_messages_are_all_kept():
    msgs = [msg("system", 10), msg("system", 10), msg("user", 10)]
    assert trim_history(msgs, 1)[0] == msgs


# ----------------------------------------------------------------------------- per route


def recording_router(routes: list[LLMRoute], seen: dict[str, list]) -> Router:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen[body["model"]] = body["messages"]
        if body["model"] == "small":  # the first route fails: the second gets the same request
            return httpx.Response(500, json={"error": "down"})
        chunk = {
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }  # fmt: skip
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    return Router(
        routes,
        SafetyConfig(),
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_each_route_gets_the_history_that_fits_its_own_budget():
    small = LLMRoute(**{**ROUTE, "name": "small", "model": "small"}, context_budget_tokens=750)
    big = LLMRoute(**{**ROUTE, "name": "big", "model": "big"}, context_budget_tokens=5000)
    seen: dict[str, list] = {}
    router = recording_router([small, big], seen)
    msgs = convo(5)  # 1150 tokens in all
    events = [ev async for ev in router.stream(msgs)]
    assert events
    assert len(seen["small"]) == 1 + 2 * 3 + 1  # the two oldest turns are dropped
    assert len(seen["big"]) == len(msgs)  # the fallback route's larger budget keeps them all
    assert seen["small"][0] == msgs[0] and seen["big"][0] == msgs[0]


# ----------------------------------------------------------------------------- load time


def test_the_defaults_leave_room_on_a_default_route():
    cfg = RoboEmotConfig(llm_routes=[ROUTE])
    need = system_tokens_worst_case(cfg) + cfg.conversation.turn_reserve_tokens
    assert need < cfg.llm_routes[0].context_budget_tokens


def test_a_budget_too_small_for_the_system_prompt_is_rejected_naming_the_route():
    tight = {**ROUTE, "name": "tiny", "context_budget_tokens": 256}
    with pytest.raises(ValueError, match="llm route 'tiny'") as err:
        RoboEmotConfig(llm_routes=[tight], persona=LONG)
    assert "context_budget_tokens is 256" in str(err.value)
    assert "pppppppppp" not in str(err.value)  # no prompt content in the message


def test_only_the_route_that_is_too_small_is_named():
    ok = {**ROUTE, "name": "ok", "context_budget_tokens": 4000}
    tight = {**ROUTE, "name": "tiny", "context_budget_tokens": 256}
    with pytest.raises(ValueError) as err:
        RoboEmotConfig(llm_routes=[ok, tight], persona=LONG)
    assert "'tiny'" in str(err.value) and "'ok'" not in str(err.value)
    RoboEmotConfig(llm_routes=[ok], persona=LONG)


def test_the_turn_reserve_counts():
    need = system_tokens_worst_case(RoboEmotConfig(llm_routes=[ROUTE]))
    route = {**ROUTE, "context_budget_tokens": need + 300}
    RoboEmotConfig(llm_routes=[route], conversation={"turn_reserve_tokens": 300})
    with pytest.raises(ValueError, match="room for one turn"):
        RoboEmotConfig(llm_routes=[route], conversation={"turn_reserve_tokens": 301})


def test_extra_safety_rules_and_summary_caps_count_toward_the_budget():
    route = {**ROUTE, "context_budget_tokens": 1500}
    RoboEmotConfig(llm_routes=[route])
    with pytest.raises(ValueError, match="'lan'"):
        RoboEmotConfig(llm_routes=[route], safety={"extra_prompt_rules": ["r" * 400] * 12})
    with pytest.raises(ValueError, match="'lan'"):
        RoboEmotConfig(
            llm_routes=[route],
            conversation={"profile_summary_max_chars": 4000, "scene_summary_max_chars": 4000},
        )


def test_the_worst_register_is_checked_not_the_active_one():
    route = {**ROUTE, "context_budget_tokens": 800}
    RoboEmotConfig(llm_routes=[route])
    with pytest.raises(ValueError, match="'lan'"):
        RoboEmotConfig(llm_routes=[route], register={"child": {"persona_fragment": "f" * 3000}})


def test_no_routes_means_nothing_to_check():
    RoboEmotConfig(
        persona={"prompt": "p" * 4000, "prompt_max_chars": 4000},
        conversation={"turn_reserve_tokens": 8192},
    )


def test_a_hot_reload_with_a_too_small_budget_is_refused_and_the_old_config_stays():
    stack = ConfigStack(app={"llm_routes": [ROUTE]})
    with pytest.raises(ConfigError, match="llm route 'lan'"):
        stack.set("llm_routes", [{**ROUTE, "context_budget_tokens": 256}], layer="app")
    assert stack.effective().llm_routes[0].context_budget_tokens == 8192


def test_budget_params_have_defaults_and_ranges():
    budget = LLMRoute.model_fields["context_budget_tokens"]
    reserve = ConversationConfig.model_fields["turn_reserve_tokens"]
    assert budget.default == 8192 and reserve.default == 256
    assert budget.json_schema_extra["unit"] == "tokens"
    assert reserve.json_schema_extra["unit"] == "tokens"
    with pytest.raises(ValueError):
        LLMRoute(**ROUTE, context_budget_tokens=10)
