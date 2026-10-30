"""The skip tool (SPEC 8, Realtime speech route): with ``voice.realtime_route.skip_tool`` the
server's LLM may call ``ignore_utterance`` for speech not meant for the robot. Nothing more
plays, the turn's items are deleted on the server and in the history mirror, and the route
counts ``skipped`` (or ``skipped_late`` once audio played)."""

from __future__ import annotations

import dataclasses

import pytest
from roboemot.config import ConfigStack
from roboemot.config.schema import RegisterParams
from roboemot.llm.prompt import realtime_instructions, skip_prompts
from roboemot.realtime.turn import SKIP_TOOL
from rt_rig import build_rt, wait_for
from test_conversation_worker import utt
from turn_server import TurnServer

SKIP = {"skip_tool": True}


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


def deleted(server) -> set[str]:
    return {e["item_id"] for e in server.of("conversation.item.delete")}


def text_items(server) -> list[str]:
    return [
        e["item"]["id"]
        for e in server.of("conversation.item.create")
        if e["item"]["type"] == "message"
    ]


def tool_names(update: dict) -> list[str]:
    return [t["name"] for t in update["session"].get("tools") or []]


def assert_skipped(rig, server, items: set[str], counter: str) -> None:
    assert wait_for(lambda: deleted(server) == items)
    snap = rig.route.snapshot()
    other = "skipped_late" if counter == "skipped" else "skipped"
    assert snap[counter] == 1 and snap[other] == 0
    assert rig.history() == []  # not a turn of history_turns
    assert rig.worker.snapshot()["history_turns"] == 0
    assert rig.events[-1]["outcome"] == "skipped"
    assert len(server.of("response.create")) == 1  # no follow-up response
    outputs = [e for e in server.of("conversation.item.create") if e["item"]["type"] != "message"]
    assert outputs == []  # no tool result: the call item is deleted instead
    assert rig.chat_hosts() == []  # the chat routes do not answer it either


def test_a_skip_before_any_audio_plays_nothing_and_deletes_the_items(rigs):
    with TurnServer(["tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn("allora ci vediamo domani alle tre")
        assert rig.sink.pushed == []
        assert_skipped(rig, srv, {*text_items(srv), "fc_1"}, "skipped")
        create = srv.of("response.create")[0]["response"]
        assert create["tool_choice"] == "auto"  # offered without any skill
        assert SKIP_TOOL in tool_names(srv.of("session.update")[-1])
        assert skip_prompts()["it"] in create["instructions"]


def test_audio_after_the_skip_call_never_plays(rigs):
    """The run is muted on the client's thread as the call arrives: the audio that follows in
    the same response is not played, and its item is deleted too."""
    with TurnServer(["tool_then_audio"], tool_name=SKIP_TOOL, deltas=5) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        assert rig.sink.pushed == []
        assert_skipped(rig, srv, {*text_items(srv), "item_1", "fc_1"}, "skipped")


def test_a_skip_after_audio_played_cuts_it_like_a_barge_in(rigs):
    with TurnServer(["audio_then_tool"], tool_name=SKIP_TOOL, deltas=5, gap_s=0.1) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        assert rig.sink.pushed and rig.sink.stops >= 1
        assert_skipped(rig, srv, {*text_items(srv), "item_1", "fc_1"}, "skipped_late")


def test_a_skipped_audio_utterance_deletes_the_servers_own_user_items(rigs):
    app = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
    with TurnServer(["tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt={**SKIP, "input": "audio"}, app=app)
        assert rig.route.ready()
        heard = dataclasses.replace(utt("la TV parla"), realtime=(rig.route.generation, ("in_a",)))
        n = rig.worker.snapshot().get("turns", 0)
        assert rig.worker.submit(heard)
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)
        assert text_items(srv) == []  # the server holds the words
        assert_skipped(rig, srv, {"in_a", "fc_1"}, "skipped")


def test_a_skip_leaves_the_earlier_turns_alone(rigs):
    with TurnServer(["reply", "tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn("ciao robot")
        first = rig.history()
        rig.turn("pronto, sì, sono al telefono")
        assert wait_for(lambda: len(deleted(srv)) == 2)
        assert deleted(srv) == {text_items(srv)[1], "fc_2"}
        assert rig.history() == first and rig.worker.snapshot()["history_turns"] == 1
        assert rig.route.snapshot()["skipped"] == 1


def test_without_a_skip_call_the_turn_answers_and_nothing_is_deleted(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert [m["role"] for m in rig.history()[0]] == ["user", "assistant"]
        assert srv.of("conversation.item.delete") == []
        snap = rig.route.snapshot()
        assert snap["skipped"] == 0 and snap["skipped_late"] == 0
        assert rig.events[-1]["outcome"] == "ok"


def test_skip_tool_off_sends_neither_the_tool_nor_the_instruction(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        rig.turn()
        rule = skip_prompts()["it"]
        for update in srv.of("session.update"):
            assert SKIP_TOOL not in tool_names(update)
            assert rule not in update["session"].get("instructions", "")
        for create in srv.of("response.create"):
            assert rule not in create["response"]["instructions"]
            assert "tool_choice" not in create["response"]  # no tool offered at all


def test_a_skip_call_with_the_option_off_is_not_a_skip(rigs):
    """The tool was never offered: a call is an unknown tool, not a skip."""
    with TurnServer(["tool", "reply"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.route.snapshot()["skipped"] == 0
        assert rig.events[-1]["outcome"] != "skipped"


# ------------------------------------------------------------------ the prompt files


@pytest.mark.parametrize("language", ["it", "en"])
def test_the_instruction_loads_from_the_prompt_files(language):
    rule = skip_prompts()[language]
    assert SKIP_TOOL in rule  # a translation keeps the tool's name
    cfg = ConfigStack(
        app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}}
    ).effective()
    text = realtime_instructions(cfg, RegisterParams(), language=language)
    assert text.endswith(rule)
    assert rule not in realtime_instructions(cfg, RegisterParams(), language=language,
                                             skip_rule=False)  # fmt: skip


def test_the_instruction_follows_the_option_and_falls_back_by_language():
    off = ConfigStack(app={}).effective()
    assert all(
        r not in realtime_instructions(off, RegisterParams()) for r in skip_prompts().values()
    )
    cfg = ConfigStack(
        app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}}
    ).effective()
    assert realtime_instructions(cfg, RegisterParams(), language="de").endswith(
        skip_prompts()["it"]
    )
