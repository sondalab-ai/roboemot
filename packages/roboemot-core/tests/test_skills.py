"""SPEC 11, LLM tools: skill schemas from type hints, the registry, the executor and the default
skills."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Literal, Optional

import pytest
from roboemot.affect import AffectEngine
from roboemot.config.schema import RoboEmotConfig
from roboemot.skills import (
    Desc,
    MaxLen,
    Range,
    SkillContext,
    SkillRegistry,
    SkillResult,
    execute,
    skill,
)
from roboemot.skills.defaults import SkillEnv, default_skills

CTX = SkillContext(speaker="p1", language="it", minor=False, now=10.0)
EXEC = {"timeout_s": 1.0, "max_chars": 50, "enum_max": 20}


def tools(*fns, minor=False, allow=None):
    reg = SkillRegistry(fns)
    offered = reg.offered(allow or reg.names(), minor=minor)
    return reg, offered, SkillRegistry.tools(offered)


def test_schema_from_type_hints():
    @skill(llm_visible=True, description="d")
    def demo(
        word: Annotated[str, MaxLen(10), Desc("a word")],
        mode: Literal["a", "b"],
        count: Annotated[int, Range(1, 5)] = 2,
        level: Annotated[float, Range(0, 1)] = 0.5,
        loud: bool = False,
        note: Optional[str] = None,  # noqa: UP045
        other: str | None = None,
    ) -> str:
        return "x"

    _, _, [tool] = tools(demo)
    fn = tool["function"]
    assert tool["type"] == "function" and fn["name"] == "demo" and fn["description"] == "d"
    p = fn["parameters"]
    assert p["additionalProperties"] is False and p["required"] == ["word", "mode"]
    props = p["properties"]
    assert props["word"] == {"type": "string", "maxLength": 10, "description": "a word"}
    assert props["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert props["count"] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert props["level"]["type"] == "number" and props["loud"] == {"type": "boolean"}
    assert props["note"] == {"type": "string"} and props["other"] == {"type": "string"}


def test_unsupported_hints_are_refused_at_registration():
    with pytest.raises(TypeError):

        @skill()
        def a(x: list[str]) -> None: ...

    with pytest.raises(TypeError):

        @skill()
        def b(x: int | str) -> None: ...

    with pytest.raises(TypeError):

        @skill()
        def c(x) -> None: ...

    with pytest.raises(TypeError):

        @skill()
        def d(*x: int) -> None: ...

    with pytest.raises(TypeError):

        @skill(choices={"nope": lambda: []})
        def e(x: str) -> None: ...


def test_the_context_is_not_in_the_schema_and_a_skill_without_name_uses_the_function():
    @skill(llm_visible=True)
    def who(ctx: SkillContext, a: int = 1) -> str:
        """Tell who. More text."""
        return ctx.speaker or ""

    _, _, [tool] = tools(who)
    assert list(tool["function"]["parameters"]["properties"]) == ["a"]
    assert tool["function"]["description"] == "Tell who. More text."


def test_only_visible_listed_skills_are_offered_and_minors_filter_per_skill():
    @skill(llm_visible=True)
    def seen() -> str:
        return "1"

    @skill()
    def hidden() -> str:
        return "2"

    @skill(llm_visible=True, allowed_with_minor=False)
    def adults() -> str:
        return "3"

    reg = SkillRegistry([seen, hidden, adults])
    names = lambda **kw: [s.name for s in reg.offered(reg.names(), **kw)]  # noqa: E731
    assert names(minor=False) == ["seen", "adults"]
    assert names(minor=True) == ["seen"]
    assert [s.name for s in reg.offered(["adults"], minor=False)] == ["adults"]
    assert reg.offered([], minor=False) == []
    with pytest.raises(ValueError):
        reg.add(seen)


def test_dynamic_choices_become_the_enum_and_are_capped_and_empty_hides_the_skill():
    names = ["a", "b", "c", "d", "e", "f", "g"]

    @skill(llm_visible=True, choices={"who": lambda: names})
    def pick(who: str) -> str:
        return who

    reg = SkillRegistry([pick])
    [tool] = reg.tools(reg.offered(["pick"], minor=False), enum_max=3)
    assert tool["function"]["parameters"]["properties"]["who"]["enum"] == ["a", "b", "c"]
    names.clear()
    assert reg.offered(["pick"], minor=False) == []


async def run(reg, name, args, ctx=CTX, offered=None):
    return await execute(reg, reg.names() if offered is None else offered, name, args, ctx, **EXEC)


@pytest.fixture()
def demo():
    calls = []

    @skill(llm_visible=True, choices={"mode": lambda: ["x", "y"]})
    def demo(
        mode: str,
        count: Annotated[int, Range(1, 5)] = 2,
        level: Annotated[float, Range(0, 1)] = 0.5,
        loud: bool = False,
        text: str = "",
    ) -> dict:
        calls.append((mode, count, level, loud, text))
        return {"ok": True}

    return SkillRegistry([demo]), calls


async def test_the_executor_runs_valid_calls_and_clamps_ranges(demo):
    reg, calls = demo
    out = await run(reg, "demo", {"mode": "x", "count": 99, "level": -3})
    assert out.ok and json.loads(out.content) == {"ok": True}
    assert calls == [("x", 5, 0.0, False, "")]
    await run(reg, "demo", {"mode": "y", "level": 1, "count": 3, "loud": True})  # int for number
    assert calls[-1] == ("y", 3, 1.0, True, "")


@pytest.mark.parametrize(
    "args",
    [
        {"mode": "x", "extra": 1},  # unknown argument
        {"mode": "z"},  # not among the choices
        {"mode": "x", "count": "2"},  # wrong type
        {"mode": "x", "count": True},  # a bool is not an int
        {"mode": "x", "count": 2.5},
        {"mode": "x", "level": float("nan")},
        {"mode": "x", "loud": 1},
        {"mode": "x", "text": "t" * 51},  # over the cap
        {},  # missing
        {"_raw": '{"mode"'},  # the router's marker for broken JSON
        ["x"],
        "x",
    ],
)
async def test_invalid_arguments_are_refused_without_running_the_skill(demo, args):
    reg, calls = demo
    out = await run(reg, "demo", args)
    assert not out.ok and json.loads(out.content)["error"] == "invalid_arguments"
    assert calls == []


async def test_error_results_never_echo_the_arguments(demo):
    reg, _ = demo
    secret = "my-secret-value"
    out = await run(reg, "demo", {"mode": secret, "text": secret, secret: 1})
    assert secret not in out.content
    out = await run(reg, "demo", {"mode": "x", "text": secret * 20})
    assert secret not in out.content


async def test_unknown_and_not_offered_skills_do_not_run(demo):
    reg, calls = demo
    assert json.loads((await run(reg, "nope", {})).content)["error"] == "unknown_skill"
    out = await run(reg, "demo", {"mode": "x"}, offered=[])
    assert json.loads(out.content)["error"] == "unknown_skill" and calls == []


async def test_timeouts_and_exceptions_become_results_and_log_no_text(caplog):
    @skill(llm_visible=True)
    def slow() -> str:
        time.sleep(0.5)
        return "late"

    @skill(llm_visible=True)
    async def boom(text: str) -> str:
        raise RuntimeError(f"user said {text}")

    reg = SkillRegistry([slow, boom])
    with caplog.at_level(logging.DEBUG):
        t0 = time.monotonic()
        out = await execute(reg, ["slow"], "slow", {}, CTX, timeout_s=0.05, max_chars=9, enum_max=1)
        assert time.monotonic() - t0 < 0.4 and json.loads(out.content)["error"] == "timeout"
        out = await run(reg, "boom", {"text": "hunter2"})
    assert json.loads(out.content)["error"] == "failed"
    assert "hunter2" not in caplog.text and "RuntimeError" in caplog.text


async def test_skill_results_and_end_turn():
    @skill(llm_visible=True)
    def ending(ctx: SkillContext) -> SkillResult:
        return SkillResult({"status": "asked"}, end_turn=True)

    @skill(llm_visible=True)
    def plain() -> str:
        return "done"

    reg = SkillRegistry([ending, plain])
    out = await run(reg, "ending", {})
    assert out.end_turn and json.loads(out.content) == {"status": "asked"}
    out = await run(reg, "plain", {})
    assert out.content == "done" and not out.end_turn


# ------------------------------------------------------------------ default skills


class Env:
    def __init__(self):
        self.engine = AffectEngine(RoboEmotConfig().affect, RoboEmotConfig().appraisal)
        self.played, self.looked = [], []
        self.can_play = True
        self.env = SkillEnv(
            gestures=lambda: ["joy", "curiosity"],
            play=lambda g: self.played.append(g) or self.can_play,
            look_names=lambda: ["Marta", "Rex"],
            look_at=lambda t: self.looked.append(t) or True,
            affect=self.engine,
        )
        self.reg = default_skills(self.env)


def test_the_default_skills_are_the_five_of_the_spec_and_all_visible():
    reg = Env().reg
    assert reg.names() == [
        "play",
        "look_at",
        "express_emotion",
        "remember_preference",
        "stop_talking",
    ]
    allow = RoboEmotConfig().conversation.llm_skills
    offered = reg.offered(allow, minor=True)
    assert [s.name for s in offered] == reg.names()
    by = {t["function"]["name"]: t["function"]["parameters"] for t in reg.tools(offered)}
    assert by["play"]["properties"]["gesture"]["enum"] == ["joy", "curiosity"]
    assert by["look_at"]["properties"]["target"]["enum"] == [
        "speaker", "person", "animal", "object", "away", "Marta", "Rex",
    ]  # fmt: skip
    assert by["express_emotion"]["properties"]["emotion"]["enum"] == list(
        RoboEmotConfig().affect.labels
    )
    assert by["express_emotion"]["properties"]["intensity"]["maximum"] == 1.0
    assert by["remember_preference"]["required"] == ["key", "value"]
    assert by["stop_talking"]["properties"] == {}
    for params in by.values():  # no joint, no free-form object anywhere
        assert "joint" not in json.dumps(params)


async def test_play_and_look_at_only_take_known_names():
    e = Env()
    assert (await run(e.reg, "play", {"gesture": "joy"})).ok
    assert e.played == ["joy"]
    assert not (await run(e.reg, "play", {"gesture": "head.yaw=90"})).ok
    assert (await run(e.reg, "look_at", {"target": "Marta"})).ok and e.looked == ["Marta"]
    assert not (await run(e.reg, "look_at", {"target": "Mallory"})).ok
    e.can_play = False
    assert json.loads((await run(e.reg, "play", {"gesture": "joy"})).content) == {
        "error": "not_now"
    }


async def test_express_emotion_uses_the_per_turn_impulse_budget():
    e = Env()
    ctx = SkillContext("p1", "it", False, 100.0)
    e.engine.new_turn()
    before = (e.engine.valence, e.engine.arousal)
    out = await run(e.reg, "express_emotion", {"emotion": "joy", "intensity": 1.0}, ctx)
    assert json.loads(out.content)["applied"] is True
    moved = ((e.engine.valence - before[0]) ** 2 + (e.engine.arousal - before[1]) ** 2) ** 0.5
    assert 0 < moved <= RoboEmotConfig().affect.llm_impulse_max + 1e-9
    out = await run(e.reg, "express_emotion", {"emotion": "sadness", "intensity": 1.0}, ctx)
    assert json.loads(out.content)["applied"] is False  # budget spent: the state is unchanged
    e.engine.new_turn()
    assert json.loads((await run(e.reg, "express_emotion", {"emotion": "sadness"}, ctx)).content)[
        "applied"
    ]
    assert not (await run(e.reg, "express_emotion", {"emotion": "joy", "intensity": "a lot"})).ok


async def test_remember_preference_asks_and_ends_the_turn_and_stop_talking_stops():
    e = Env()
    asked, stopped = [], []
    ctx = SkillContext(
        "p1", "it", False, 1.0,
        stop_speech=lambda: stopped.append(1),
        ask_confirmation=lambda k, v: asked.append((k, v)) or "asked",
    )  # fmt: skip
    out = await run(e.reg, "remember_preference", {"key": "colore", "value": "blu"}, ctx)
    assert (
        out.end_turn
        and json.loads(out.content) == {"status": "asked"}
        and asked == [("colore", "blu")]
    )
    ctx.ask_confirmation = lambda k, v: "not_allowed"
    out = await run(e.reg, "remember_preference", {"key": "colore", "value": "blu"}, ctx)
    assert not out.end_turn
    out = await run(e.reg, "stop_talking", {}, ctx)
    assert out.end_turn and stopped == [1]
    assert asyncio.get_running_loop()  # (keeps the module async-aware)


async def test_null_arguments_mean_no_arguments_and_other_non_objects_are_refused():
    @skill(llm_visible=True)
    def quiet() -> str:
        return "done"

    reg = SkillRegistry([quiet])
    assert (await run(reg, "quiet", None)).ok  # "null"
    for bad in (5, True, "x", [1]):
        out = await run(reg, "quiet", bad)
        assert not out.ok and json.loads(out.content)["error"] == "invalid_arguments"


async def test_a_choice_is_checked_against_its_list_not_against_the_text_cap():
    @skill(llm_visible=True, choices={"who": lambda: ["a" * 80]})
    def pick(who: str) -> str:
        return "ok"

    reg = SkillRegistry([pick])
    out = await execute(
        reg, ["pick"], "pick", {"who": "a" * 80}, CTX, timeout_s=1, max_chars=10, enum_max=5
    )
    assert out.ok  # an offered name longer than tool_arg_max_chars still works


def test_the_default_skills_run_inline_on_the_loop():
    assert all(reg_skill.inline for reg_skill in Env().reg._skills.values())


async def test_inline_skills_run_on_the_calling_thread():
    import threading

    seen = []

    @skill(llm_visible=True, inline=True)
    def here() -> str:
        seen.append(threading.current_thread())
        return "x"

    reg = SkillRegistry([here])
    await run(reg, "here", {})
    assert seen == [threading.current_thread()]
