"""The stricter conversation start (SPEC 8, Conversation start): with the skip tool, each
realtime turn's instructions say whether a conversation with the robot is in progress. With
none in progress the strict start rule applies (only words clearly meant for the robot are
answered); the robot's recent answer, a waiting confirmation, the wake word or the name open
it. Turns and skips are counted by state. The server's judgment itself is not tested here: it
needs the real server (the robot run checks it)."""

from __future__ import annotations

import dataclasses
import time

import pytest
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RegisterParams
from roboemot.llm import prompt as prompt_module
from roboemot.llm.prompt import (
    ADDRESSED,
    CONVERSATION_STATES,
    IN_PROGRESS,
    NO_CONVERSATION,
    open_prompts,
    realtime_instructions,
    skip_prompts,
    start_prompts,
)
from roboemot.realtime.turn import SKIP_TOOL
from rt_rig import build_rt, wait_for
from test_conversation_worker import utt
from turn_server import TurnServer

SKIP = {"skip_tool": True}
SPAN_S = 30.0


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def instructions(server, n: int) -> str:
    return server.of("response.create")[n]["response"]["instructions"]


def submit(rig, utterance) -> None:
    n = rig.worker.snapshot().get("turns", 0)
    assert rig.worker.submit(utterance)
    assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)


def zero(**counts: int) -> dict[str, int]:
    return {s: counts.get(s, 0) for s in CONVERSATION_STATES}


# ------------------------------------------------------------------ the turn's state


def test_with_no_conversation_the_turn_carries_the_strict_start_rule(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("eh, va bene, sì")
        text = instructions(srv, 0)
        assert start_prompts()["it"] in text and open_prompts()["it"] not in text
        assert text.index(skip_prompts()["it"]) < text.index(start_prompts()["it"])
        assert rig.route.snapshot()["turns_by_state"] == zero(none=1)
        update = srv.of("session.update")[-1]["session"]["instructions"]
        assert start_prompts()["it"] not in update  # per turn only, never the session's


def test_a_recent_answer_opens_the_conversation_and_the_span_closes_it(rigs):
    with TurnServer(["reply", "reply", "reply"]) as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("ciao robot")  # none: nothing said yet
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        rig.turn("sì, va bene")  # the robot just answered: in progress
        second = instructions(srv, 1)
        assert open_prompts()["it"] in second and start_prompts()["it"] not in second
        rig.worker.clock = lambda: time.monotonic() + SPAN_S + 1  # past the span
        rig.turn("okay, facciamo così")
        third = instructions(srv, 2)
        assert start_prompts()["it"] in third and open_prompts()["it"] not in third
        assert rig.route.snapshot()["turns_by_state"] == zero(none=2, in_progress=1)


def test_the_wake_word_or_the_name_marks_the_turn_addressed(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv, rt=SKIP)
        stamp = time.monotonic()
        submit(rig, dataclasses.replace(utt("ehi robot", stamp=stamp), wake_word=True))
        submit(rig, utt("Reachy che ore sono adesso", keywords={"name"}, stamp=stamp + 1))
        for n in (0, 1):
            text = instructions(srv, n)
            assert open_prompts()["it"] in text and start_prompts()["it"] not in text
        assert rig.route.snapshot()["turns_by_state"] == zero(addressed=2)


def test_a_waiting_confirmation_counts_as_in_progress(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        rig.worker._confirm = object()  # a confirmation question was asked
        state = rig.worker._conversation_state(utt("sì"), rig.stack.effective(), rig.worker.clock())
        assert state == IN_PROGRESS
        rig.worker._confirm = None
        state = rig.worker._conversation_state(utt("sì"), rig.stack.effective(), rig.worker.clock())
        assert state == NO_CONVERSATION


def test_a_skip_is_counted_in_its_state_and_never_opens_the_conversation(rigs):
    with TurnServer(["tool", "tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn("eh, va bene, sì")
        rig.turn("okay, facciamo così")
        snap = rig.route.snapshot()
        assert snap["skipped"] == 2
        assert snap["skipped_by_state"] == zero(none=2)
        assert snap["turns_by_state"] == zero(none=2)  # the first skip left no answer
        assert start_prompts()["it"] in instructions(srv, 1)


def test_without_the_skip_tool_no_hint_and_no_count(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv)
        rig.turn("eh, va bene, sì")
        rig.turn("okay")
        hints = [*start_prompts().values(), *open_prompts().values()]
        for create in srv.of("response.create"):
            assert all(h not in create["response"]["instructions"] for h in hints)
        snap = rig.route.snapshot()
        assert snap["turns_by_state"] == zero() and snap["skipped_by_state"] == zero()


def test_the_span_follows_a_reload(rigs):
    with TurnServer(["reply", "reply"]) as srv:
        rig = rigs(srv, rt={**SKIP, "conversation_span_s": SPAN_S})
        rig.turn("ciao robot")
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        rig.worker.clock = lambda: time.monotonic() + SPAN_S / 2  # within the old span
        rig.stack.set("voice.realtime_route.conversation_span_s", 1.0, source="test", admin=True)
        rig.turn("sì, va bene")
        assert start_prompts()["it"] in instructions(srv, 1)


# ------------------------------------------------------------------ the prompt files


@pytest.mark.parametrize("language", ["it", "en"])
def test_the_hints_load_for_each_language(language):
    cfg = ConfigStack(
        app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}}
    ).effective()
    strict = start_prompts()[language]
    assert SKIP_TOOL in strict  # the strict rule points at the tool by name
    reg = RegisterParams()
    for state, text in ((NO_CONVERSATION, strict), (IN_PROGRESS, open_prompts()[language])):
        built = realtime_instructions(cfg, reg, language=language, conversation=state)
        assert built.endswith(text)
    assert realtime_instructions(cfg, reg, language=language, conversation=ADDRESSED).endswith(
        open_prompts()[language]
    )
    line = realtime_instructions(
        cfg, reg, language=language, skip_rule=False, conversation=NO_CONVERSATION
    )
    assert strict not in line  # a fixed line carries no hint


@pytest.mark.parametrize(
    ("loader", "stem"),
    [("start_prompts", "conversation_start"), ("open_prompts", "conversation_open")],
)
def test_a_missing_hint_file_fails_loud_at_load(monkeypatch, loader, stem):
    monkeypatch.setattr(prompt_module, loader, lambda: {"en": "x ignore_utterance"})
    with pytest.raises(ConfigError, match=f"{stem}.it.txt"):
        ConfigStack(app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}})
    ConfigStack(app={"conversation": {"language": "it"}})  # skip tool off: no file needed
