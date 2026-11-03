"""The tone tool (SPEC 8, Tone tool; SPEC 7): with ``voice.realtime_route.tone_tool`` the server's
LLM reports the user's tone of voice with ``note_tone`` beside its spoken reply. The robot turns
the first valid label of a turn into one appraisal event ``speech.tone.<label>``. The call is
fire-and-forget: it gets a ``function_call_output`` and never starts a tool round, except one
more response, without the tool, after a response of tone calls and no speech."""

from __future__ import annotations

import json

import pytest
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RegisterParams
from roboemot.conversation.worker import TONE_INVALID, TONE_NOTED
from roboemot.llm import prompt as prompt_module
from roboemot.llm.prompt import realtime_instructions, skip_prompts, tone_prompts
from roboemot.realtime.turn import SKIP_TOOL, TONE_TOOL
from rt_rig import build_rt, wait_for
from test_conversation_tools import Env
from turn_server import TurnServer

TONE = {"tone_tool": True}
HAPPY = '{"tone": "happy"}'
PLAY = '{"gesture": "joy"}'


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


def tool_names(update: dict) -> list[str]:
    return [t["name"] for t in update["session"].get("tools") or []]


def outputs(server) -> list[str]:
    return [
        e["item"]["output"]
        for e in server.of("conversation.item.create")
        if e["item"]["type"] == "function_call_output"
    ]


def creates(server) -> list[dict]:
    return [e["response"] for e in server.of("response.create")]


def tooled(rigs, server, **kw):
    rig = rigs(server, **kw)
    env = Env()
    rig.worker.skills = env.registry
    rig.worker._speaker = lambda: "p1"
    return rig, env


# ------------------------------------------------------------------ the round loop


def test_a_tone_beside_the_spoken_reply_starts_no_new_round(rigs):
    with TurnServer(
        ["audio_then_tool", "reply"], tool_name=TONE_TOOL, tool_args=HAPPY, strict_calls=True
    ) as srv:
        rig = rigs(srv, rt=TONE)
        rig.turn()
        assert len(srv.of("response.create")) == 1
        assert wait_for(lambda: outputs(srv) == [TONE_NOTED])
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert rig.sink.pushed  # the reply played
        snap = rig.route.snapshot()
        assert snap["tool_calls"] == 0  # never a skill call
        assert rig.worker.snapshot()["tones"] == 1
        assert rig.events[-1]["rounds"] == 0 and rig.events[-1]["outcome"] == "ok"
        assert "note_tone" not in json.dumps(rig.history())  # not in the chat history
        rig.turn()  # the call has its output: the next response is not refused
        assert srv.pending_refusals == 0
        assert len(srv.of("response.create")) == 2


def test_a_tone_without_speech_gets_one_more_response_without_the_tool(rigs):
    with TurnServer(
        ["tool", "reply"], tool_name=TONE_TOOL, tool_args=HAPPY, strict_calls=True
    ) as srv:
        rig = rigs(srv, rt=TONE)
        rig.turn()
        first, second = creates(srv)
        assert first["tool_choice"] == "auto"  # offered without any skill
        assert tone_prompts()["it"] in first["instructions"]
        assert tone_prompts()["it"] not in second["instructions"]
        assert "tool_choice" not in second  # no tool left in the session
        updates = srv.of("session.update")
        assert TONE_TOOL in tool_names(updates[-2]) and TONE_TOOL not in tool_names(updates[-1])
        assert tone_prompts()["it"] not in updates[-1]["session"]["instructions"]
        assert outputs(srv) == [TONE_NOTED]  # sent before the second response
        assert srv.pending_refusals == 0
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert rig.events[-1]["rounds"] == 0
        rig.turn()  # the next turn offers the tool again
        assert TONE_TOOL in tool_names(srv.of("session.update")[-1])


def test_the_extra_response_runs_once(rigs):
    with TurnServer(["tool"], tool_name=TONE_TOOL, tool_args=HAPPY, strict_calls=True) as srv:
        rig = rigs(srv, rt=TONE)
        rig.turn()
        assert len(srv.of("response.create")) == 2
        assert wait_for(lambda: len(outputs(srv)) == 2)
        assert srv.pending_refusals == 0
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]


def test_a_tone_beside_a_skill_runs_the_normal_next_round(rigs):
    names = [TONE_TOOL, "play"]
    args = {"call_1": HAPPY, "call_1_1": PLAY}
    with TurnServer(["tools", "reply"], tool_names=names, call_args=args, strict_calls=True) as srv:
        rig, env = tooled(rigs, srv, rt=TONE, app={"conversation": {"max_tool_rounds": 1}})
        rig.turn()
        assert env.played == ["joy"]
        assert len(srv.of("response.create")) == 2
        assert creates(srv)[1]["tool_choice"] == "none"  # the skill used the one round
        assert sorted(outputs(srv)) == sorted([TONE_NOTED, outputs(srv)[0]])
        assert TONE_NOTED in outputs(srv) and len(outputs(srv)) == 2
        assert rig.route.snapshot()["tool_calls"] == 1  # the skill only
        assert rig.events[-1]["rounds"] == 1
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert "note_tone" not in json.dumps(rig.history())


def test_a_tone_only_response_does_not_use_a_tool_round(rigs):
    """With ``max_tool_rounds: 1`` the skill called in the extra response still runs."""
    names = [TONE_TOOL, "play", "x"]
    args = {"call_1": HAPPY, "call_2": PLAY}
    with TurnServer(
        ["tool", "tool", "reply"], tool_names=names, call_args=args, strict_calls=True
    ) as srv:
        rig, env = tooled(rigs, srv, rt=TONE, app={"conversation": {"max_tool_rounds": 1}})
        rig.turn()
        assert [c["tool_choice"] for c in creates(srv)] == ["auto", "auto", "none"]
        assert env.played == ["joy"]
        assert TONE_TOOL not in tool_names(srv.of("session.update")[-1])
        assert "play" in tool_names(srv.of("session.update")[-1])
        assert rig.events[-1]["rounds"] == 1
        assert srv.pending_refusals == 0


def test_tone_tool_off_offers_neither_the_tool_nor_the_rule(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        rig.turn()
        rule = tone_prompts()["it"]
        for update in srv.of("session.update"):
            assert TONE_TOOL not in tool_names(update)
            assert rule not in update["session"].get("instructions", "")
        for create in creates(srv):
            assert rule not in create["instructions"]
            assert "tool_choice" not in create
        assert rig.emitted == []


def test_the_tool_offers_the_configured_labels(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, rt={**TONE, "tone_labels": ["calm", "tired"]})
        rig.turn()
        (tool,) = [t for t in srv.of("session.update")[-1]["session"]["tools"]]
        assert tool["name"] == TONE_TOOL
        assert tool["parameters"]["properties"]["tone"]["enum"] == ["calm", "tired"]
        assert tool["parameters"]["required"] == ["tone"]


# ------------------------------------------------------------------ no event


@pytest.mark.parametrize("arguments", ['{"tone": "furious"}', "not json", '{"tone": 3}', "[]"])
def test_an_invalid_tone_emits_no_event_and_is_counted(rigs, arguments):
    with TurnServer(
        ["audio_then_tool"], tool_name=TONE_TOOL, tool_args=arguments, strict_calls=True
    ) as srv:
        rig = rigs(srv, rt=TONE)
        rig.turn()
        assert wait_for(lambda: outputs(srv) == [TONE_INVALID])
        assert rig.emitted == []
        assert rig.worker.snapshot()["tone_invalid"] == 1
        assert [e for e in rig.diag if e["reason"] == "tone_invalid"]
        assert "furious" not in json.dumps(rig.diag)  # the model's text never leaves
        assert len(srv.of("response.create")) == 1


def test_a_second_tone_in_one_utterance_is_ignored(rigs):
    names = [TONE_TOOL, TONE_TOOL]
    args = {"call_1": HAPPY, "call_1_1": '{"tone": "sad"}'}
    with TurnServer(
        ["tools_then_audio"], tool_names=names, call_args=args, strict_calls=True
    ) as srv:
        rig = rigs(srv, rt=TONE)
        rig.turn()
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert outputs(srv) == [TONE_NOTED, TONE_NOTED]
        assert rig.worker.snapshot()["tone_repeated"] == 1
        assert len(srv.of("response.create")) == 1


def test_a_tone_before_a_skip_emits_nothing_and_the_skip_mutes(rigs):
    """The tone call is not the response's first call: the skip still mutes the audio after it."""
    rt = {**TONE, "skip_tool": True}
    with TurnServer(
        ["tools_then_audio"], tool_names=[TONE_TOOL, SKIP_TOOL], tool_args=HAPPY
    ) as srv:
        rig = rigs(srv, rt=rt)
        rig.turn()
        assert rig.sink.pushed == []
        assert rig.route.snapshot()["skipped"] == 1
        assert rig.events[-1]["outcome"] == "skipped"
        assert rig.emitted == []
        assert len(srv.of("response.create")) == 1


def test_a_tone_after_a_skip_emits_nothing(rigs):
    rt = {**TONE, "skip_tool": True}
    with TurnServer(["tools"], tool_names=[SKIP_TOOL, TONE_TOOL], tool_args=HAPPY) as srv:
        rig = rigs(srv, rt=rt)
        rig.turn()
        assert rig.route.snapshot()["skipped"] == 1
        assert rig.emitted == []
        assert len(srv.of("response.create")) == 1


def test_a_skip_in_the_extra_response_is_honoured_and_emits_nothing(rigs):
    rt = {**TONE, "skip_tool": True}
    with TurnServer(["tool", "tool"], tool_names=[TONE_TOOL, SKIP_TOOL], tool_args=HAPPY) as srv:
        rig = rigs(srv, rt=rt)
        rig.turn()
        assert rig.route.snapshot()["skipped"] == 1
        assert rig.events[-1]["outcome"] == "skipped"
        assert rig.emitted == []


# ------------------------------------------------------------------ the prompt files


@pytest.mark.parametrize("language", ["it", "en"])
def test_the_rule_loads_from_the_prompt_files(language):
    rule = tone_prompts()[language]
    assert TONE_TOOL in rule  # a translation keeps the tool's name
    cfg = ConfigStack(
        app={"voice": {"realtime_route": TONE}, "conversation": {"language": "it"}}
    ).effective()
    assert rule in realtime_instructions(cfg, RegisterParams(), language=language)
    off = realtime_instructions(cfg, RegisterParams(), language=language, tone_rule=False)
    assert rule not in off


def test_the_tone_rule_comes_before_the_skip_rule():
    rt = {**TONE, "skip_tool": True}
    cfg = ConfigStack(app={"voice": {"realtime_route": rt}, "conversation": {"language": "it"}})
    text = realtime_instructions(cfg.effective(), RegisterParams(), language="it")
    assert text.index(tone_prompts()["it"]) < text.index(skip_prompts()["it"])
    assert text.endswith(skip_prompts()["it"])


def test_a_missing_prompt_file_fails_loud_at_load(monkeypatch):
    """The tool is never offered without its rule: the configuration is rejected."""
    monkeypatch.setattr(prompt_module, "tone_prompts", lambda: {"en": "x note_tone"})
    with pytest.raises(ConfigError, match="note_tone.it.txt"):
        ConfigStack(app={"voice": {"realtime_route": TONE}, "conversation": {"language": "it"}})
    ConfigStack(app={"conversation": {"language": "it"}})  # off: no file needed


# ------------------------------------------------------------------ configuration


@pytest.mark.parametrize(
    "labels", [[], ["calm", "calm"], ["Calm"], ["calm.x"], [f"l{c}" for c in "abcdefghijklm"]]
)
def test_tone_labels_must_be_event_names(labels):
    with pytest.raises(ConfigError, match="tone_labels"):
        ConfigStack(app={"voice": {"realtime_route": {"tone_labels": labels}}})


def test_every_default_label_has_an_appraisal_rule_and_custom_ones_validate():
    cfg = ConfigStack(app={}).effective()
    events = {r.event for r in cfg.appraisal.rules}
    assert {f"speech.tone.{label}" for label in cfg.voice.realtime_route.tone_labels} <= events
    rule = {"event": "speech.tone.tired", "arousal": -0.1}
    ConfigStack(app={"appraisal": {"rules": [rule]}})
