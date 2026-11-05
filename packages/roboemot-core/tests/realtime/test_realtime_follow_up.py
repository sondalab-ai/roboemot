"""Fire-and-forget skills on the realtime route (SPEC 8, Realtime speech route, Tools; SPEC 11,
LLM tools): a response that spoke and whose calls are all skills with ``follow_up=False``
(``play``, ``look_at``, ``express_emotion``) gets one ``function_call_output`` per call and no
further ``response.create``: one more round only invited the model to say it again (robot run
2026-11-05: the same sentence three times)."""

from __future__ import annotations

import json
import time

import pytest
from roboemot.affect import AffectEngine
from roboemot.config.schema import RoboEmotConfig
from roboemot.llm.prompt import tone_prompts
from roboemot.realtime.turn import TONE_TOOL
from roboemot.skills import skill
from roboemot.skills.defaults import SkillEnv, default_skills
from rt_rig import build_rt, wait_for
from turn_server import REPLY, TurnServer

PLAY = '{"gesture": "joy"}'
LOOK = '{"target": "person"}'
SKILLS = ["play", "look_at", "express_emotion", "remember_preference", "stop_talking", "recall"]


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


class Env:
    """The default skills, with a ``look_at`` that may find nothing, plus ``recall``: a plugin
    skill that keeps the default ``follow_up`` (its result is what the model answers with)."""

    def __init__(self, look: bool = True) -> None:
        cfg = RoboEmotConfig()
        self.played: list[str] = []
        self.looked: list[str] = []
        env = SkillEnv(
            gestures=lambda: ["joy", "curiosity"],
            play=lambda g: self.played.append(g) or True,
            look_names=lambda: [],
            look_at=lambda t: self.looked.append(t) or look,
            affect=AffectEngine(cfg.affect, cfg.appraisal),
        )
        self.registry = default_skills(env)

        @skill(llm_visible=True, inline=True, description="Recall a fact.")
        def recall() -> str:
            return "the fact"

        self.registry.add(recall)


def tooled(rigs, server, env=None, **kw):
    app = kw.pop("app", {})
    app = {**app, "conversation": {"llm_skills": SKILLS, **app.get("conversation", {})}}
    rig = rigs(server, app=app, **kw)
    env = env or Env()
    rig.worker.skills = env.registry
    rig.worker._speaker = lambda: "p1"
    return rig, env


def creates(srv) -> list[dict]:
    return srv.of("response.create")


def outputs(srv) -> dict[str, str]:
    return {
        e["item"]["call_id"]: e["item"]["output"]
        for e in srv.of("conversation.item.create")
        if e["item"]["type"] == "function_call_output"
    }


def next_turn_on_the_route(rig, srv) -> None:
    """No call was left open: the next turn is answered by the server, not refused."""
    served = srv.responses
    rig.turn("ancora", stamp=time.monotonic() + 1.0)
    assert srv.responses == served + 1
    assert srv.pending_refusals == 0 and rig.route.snapshot()["refused"] == 0
    assert rig.chat_hosts() == [] and rig.events[-1]["route"] == "rt"


def test_speech_and_a_gesture_end_the_turn_after_one_response(rigs):
    """The robot-run case: the reply was spoken and ``play`` called in the same response."""
    script = ["audio_then_tool", "audio_then_tool", "audio_then_tool", "reply"]
    with TurnServer(script, tool_name="play", tool_args=PLAY, strict_calls=True) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn("hey ciao neve")
        assert len(creates(srv)) == 1
        assert env.played == ["joy"]
        assert outputs(srv) == {"call_1": '{"ok": true}'}
        snap = rig.worker.snapshot()
        assert snap["follow_ups_skipped"] == 1 and snap["tool_calls"] == 1
        assert rig.events[-1]["outcome"] == "ok" and rig.events[-1]["rounds"] == 0
        assert rig.events[-1]["tool_calls"] == 1
        # the call and its result, then the spoken reply as the closing message
        turn = rig.history()[0]
        assert [m["role"] for m in turn] == ["user", "assistant", "tool", "assistant"]
        assert turn[1]["content"] is None and turn[1]["tool_calls"][0]["id"] == "call_1"
        assert turn[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
        assert turn[3]["content"] == REPLY and "interrupted" not in turn[3]
        next_turn_on_the_route(rig, srv)


def test_a_barge_in_after_a_fire_and_forget_end_still_cuts_what_plays(rigs):
    with TurnServer(
        ["audio_then_tool", "reply"],
        tool_name="play",
        tool_args=PLAY,
        deltas=20,
        delta_ms=100,  # 2 s of audio, sent at once: it still plays after the turn
        strict_calls=True,
    ) as srv:
        rig, _ = tooled(rigs, srv)
        rig.turn()
        assert len(creates(srv)) == 1
        assert wait_for(lambda: rig.speech.stream_progress(1) is not None)
        rig.worker.mark_interrupting()
        rig.speech.stop()
        assert rig.worker.interrupt()
        assert wait_for(lambda: srv.of("conversation.item.truncate"))
        assert rig.worker.snapshot()["history_trimmed"] == 1  # the closing reply was the tail
        last = rig.history()[0][-1]
        assert last["role"] == "tool" or last["interrupted"] is True  # nothing heard: dropped


def test_a_gesture_without_speech_still_gets_the_follow_up(rigs):
    with TurnServer(["tool", "reply"], tool_name="play", tool_args=PLAY, strict_calls=True) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert len(creates(srv)) == 2 and env.played == ["joy"]
        assert rig.worker.snapshot().get("follow_ups_skipped", 0) == 0
        assert rig.events[-1]["rounds"] == 1
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)  # spoke after it


def test_speech_and_a_skill_that_needs_its_result_keep_the_extra_round(rigs):
    names = ["play", "recall"]
    args = {"call_1": PLAY, "call_1_1": "{}"}
    with TurnServer(
        ["tools_then_audio", "reply"], tool_names=names, call_args=args, strict_calls=True
    ) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert len(creates(srv)) == 2
        assert set(outputs(srv)) == {"call_1", "call_1_1"}
        assert rig.worker.snapshot().get("follow_ups_skipped", 0) == 0
        assert rig.events[-1]["rounds"] == 1
        next_turn_on_the_route(rig, srv)


def test_a_gesture_and_a_look_at_nothing_beside_speech_end_the_turn(rigs):
    """``look_at`` found nothing: the error goes back as its result, and no round follows (the
    robot already answered; one more round would only repeat it or apologise)."""
    names = ["play", "look_at"]
    args = {"call_1": PLAY, "call_1_1": LOOK}
    with TurnServer(
        ["tools_then_audio", "reply"], tool_names=names, call_args=args, strict_calls=True
    ) as srv:
        rig, env = tooled(rigs, srv, env=Env(look=False))
        rig.turn()
        assert len(creates(srv)) == 1
        assert env.played == ["joy"] and env.looked == ["person"]
        assert outputs(srv) == {
            "call_1": '{"ok": true}',
            "call_1_1": '{"error": "nothing_to_look_at"}',
        }
        snap = rig.worker.snapshot()
        assert snap["follow_ups_skipped"] == 1
        assert snap["tool_calls"] == 2  # both ran; the error is the skill's own result
        tools = [m for m in rig.history()[0] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tools] == ["call_1", "call_1_1"]
        next_turn_on_the_route(rig, srv)


def test_a_fire_and_forget_end_answers_the_tone_call_too(rigs):
    names = [TONE_TOOL, "play"]
    args = {"call_1": '{"tone": "happy"}', "call_1_1": PLAY}
    with TurnServer(
        ["tools_then_audio", "reply"], tool_names=names, call_args=args, strict_calls=True
    ) as srv:
        rig, _ = tooled(rigs, srv, rt={"tone_tool": True})
        rig.turn()
        assert len(creates(srv)) == 1
        assert set(outputs(srv)) == {"call_1", "call_1_1"}
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert "note_tone" not in json.dumps(rig.history())
        next_turn_on_the_route(rig, srv)


def test_the_tone_tool_and_its_rule_are_offered_in_the_first_response_only(rigs):
    """A later round would invite a second ``note_tone`` for the same utterance (robot run
    2026-11-05: ``tone_repeated`` 2): the round after the first takes the tool and its rule
    away, and the next turn offers them again."""
    with TurnServer(["tool", "reply", "reply"], tool_name="recall", strict_calls=True) as srv:
        rig, _ = tooled(rigs, srv, rt={"tone_tool": True})
        rig.turn()
        rule = tone_prompts()["it"]
        first, second = creates(srv)
        assert rule in first["response"]["instructions"]
        assert rule not in second["response"]["instructions"]
        updates = srv.of("session.update")
        last = [t["name"] for t in updates[-1]["session"]["tools"]]
        assert TONE_TOOL not in last and "recall" in last
        assert rule not in updates[-1]["session"]["instructions"]
        rig.turn("ancora", stamp=time.monotonic() + 1.0)
        latest = srv.of("session.update")[-1]["session"]
        assert TONE_TOOL in [t["name"] for t in latest["tools"]]
        assert rule in creates(srv)[-1]["response"]["instructions"]
        assert srv.pending_refusals == 0
