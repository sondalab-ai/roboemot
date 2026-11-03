"""The skip tool (SPEC 8, Realtime speech route): with ``voice.realtime_route.skip_tool`` the
server's LLM may call ``ignore_utterance`` for speech not meant for the robot. Nothing more
plays, the turn's items are deleted on the server and in the history mirror, and the route
counts ``skipped`` (or ``skipped_late`` once audio played)."""

from __future__ import annotations

import dataclasses

import pytest
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RegisterParams
from roboemot.llm import prompt as prompt_module
from roboemot.llm.prompt import realtime_instructions, skip_prompts
from roboemot.realtime.turn import SKIP_TOOL
from rt_rig import build_rt, wait_for
from test_conversation_tools import Env
from test_conversation_worker import utt
from test_realtime_listener import FACE_RAD, Ear
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
    line = realtime_instructions(cfg, RegisterParams(), language=language, skip_rule=False)
    assert rule not in line


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


def test_a_missing_prompt_file_fails_loud_at_load(monkeypatch):
    """The tool is never offered without its rule: the configuration is rejected."""
    monkeypatch.setattr(prompt_module, "skip_prompts", lambda: {"en": "x ignore_utterance"})
    with pytest.raises(ConfigError, match="ignore_utterance.it.txt"):
        ConfigStack(app={"voice": {"realtime_route": SKIP}, "conversation": {"language": "it"}})
    ConfigStack(app={"conversation": {"language": "it"}})  # off: no file needed


# ------------------------------------------------------------------ fix round 1


def tooled(rigs, server, **kw):
    rig = rigs(server, rt=SKIP, **kw)
    env = Env()
    rig.worker.skills = env.registry
    rig.worker._speaker = lambda: "p1"
    return rig, env


def outputs(server) -> list[str]:
    return [
        e["item"]["output"]
        for e in server.of("conversation.item.create")
        if e["item"]["type"] == "function_call_output"
    ]


def assert_refused(rig, server, env) -> None:
    assert env.played == ["joy"]  # the skill acted
    snap = rig.route.snapshot()
    assert snap["skips_refused"] == 1 and snap["skipped"] == 0 and snap["skipped_late"] == 0
    assert wait_for(lambda: any("skip_unavailable" in o for o in outputs(server)))
    roles = [m["role"] for m in rig.history()[0]]
    assert roles[:3] == ["user", "assistant", "tool"] and "tool" in roles
    assert "fc_1" not in deleted(server)  # the skill's call stays on the server
    assert rig.events[-1]["outcome"] != "skipped"


def test_a_skip_in_a_later_round_after_a_skill_is_refused(rigs):
    names = ["play", SKIP_TOOL]
    with TurnServer(
        ["tool", "tool", "reply"], tool_names=names, tool_args='{"gesture": "joy"}'
    ) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert_refused(rig, srv, env)
        assert len(srv.of("response.create")) == 3  # the error result is answered


def test_a_skip_after_a_skill_in_the_same_response_is_refused(rigs):
    names = ["play", SKIP_TOOL]
    with TurnServer(["tools", "reply"], tool_names=names, tool_args='{"gesture": "joy"}') as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert_refused(rig, srv, env)
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)  # not muted


def test_a_skip_first_in_a_response_stops_the_calls_after_it(rigs):
    names = [SKIP_TOOL, "play"]
    with TurnServer(["tools"], tool_names=names, tool_args='{"gesture": "joy"}') as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert env.played == []
        assert_skipped(rig, srv, {*text_items(srv), "fc_1", "fc_1_1"}, "skipped")


def test_with_no_tool_rounds_only_the_skip_tool_is_offered(rigs):
    with TurnServer(["reply"]) as srv:
        rig, _ = tooled(rigs, srv, app={"conversation": {"language": "it", "max_tool_rounds": 0}})
        rig.turn()
        assert tool_names(srv.of("session.update")[-1]) == [SKIP_TOOL]
        assert srv.of("response.create")[0]["response"]["tool_choice"] == "auto"


def test_a_session_lost_right_after_the_skip_call_is_not_answered_elsewhere(rigs):
    with TurnServer(["tool_close"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        assert rig.chat_hosts() == []
        assert rig.route.snapshot()["skipped"] == 1
        assert rig.route.snapshot()["fallbacks_mid_turn"] == 0
        assert rig.history() == [] and rig.events[-1]["outcome"] == "skipped"


def test_a_barge_in_during_a_skipped_response_keeps_the_skip(rigs):
    """The skip came first: the turn's outcome is ``skipped``, counted once."""
    with TurnServer(["tool_stall"], tool_name=SKIP_TOOL, stall_s=2.0) as srv:
        rig = rigs(srv, rt=SKIP)
        n = rig.say()
        assert wait_for(lambda: srv.of("response.cancel"))  # the skip cancelled the response
        rig.worker.mark_interrupting()
        rig.speech.stop()
        assert rig.worker.interrupt()
        # well before the stalled response.done: the barge-in cut the turn
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n, 1.0)
        assert rig.events[-1]["outcome"] == "skipped"
        assert wait_for(lambda: deleted(srv) == {*text_items(srv), "fc_1"})
        snap = rig.route.snapshot()
        assert snap["skipped"] == 1 and snap["skipped_late"] == 0
        assert rig.history() == [] and rig.chat_hosts() == []


def test_an_audio_skip_through_the_realtime_input_counts_no_transcript_timeout(rigs):
    app = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
    with TurnServer(["tool"], tool_name=SKIP_TOOL, heard=[["la TV parla"]]) as srv:
        rig = rigs(srv, rt={**SKIP, "input": "audio"}, app=app)
        ear = Ear(rig, script=["x"], bearing=FACE_RAD, to_worker=True)
        try:
            ear.say()
            assert ear.settle(lambda: len(rig.events) == 1, 10.0)
            assert ear.settle(lambda: deleted(srv) == {"in_1_0", "fc_1"}, 5.0)
            ear.settle(lambda: False, 0.5)  # past any late transcript wait
            snap = rig.route.snapshot()
            assert snap["skipped"] == 1 and snap["transcript_timeouts"] == 0
        finally:
            ear.close()


def test_a_hot_reload_turns_the_tool_and_the_rule_off_for_the_next_turn(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        rule = skip_prompts()["it"]
        assert SKIP_TOOL in tool_names(srv.of("session.update")[-1])
        assert rule in srv.of("response.create")[-1]["response"]["instructions"]
        rig.stack.set("voice.realtime_route.skip_tool", False, source="test")
        rig.turn()
        assert SKIP_TOOL not in tool_names(srv.of("session.update")[-1])
        assert rule not in srv.of("session.update")[-1]["session"]["instructions"]
        assert rule not in srv.of("response.create")[-1]["response"]["instructions"]


def test_a_fixed_line_in_a_skip_session_has_no_rule_and_no_skip_tool(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt=SKIP)
        assert rig.route.ready()
        assert rig.worker.say_line("Sono di nuovo online.", "it")
        assert wait_for(lambda: srv.of("response.create"))
        line = srv.of("response.create")[-1]["response"]
        assert line["tool_choice"] == "none"
        assert skip_prompts()["it"] not in line["instructions"]
        assert "tools" not in line
