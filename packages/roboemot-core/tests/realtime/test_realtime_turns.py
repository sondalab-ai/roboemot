"""Whole conversation turns through the realtime route against the fake server: text in, the
server's voice out through ``SpeechOutput``, tools, barge-in, budget, fallbacks, history and the
session end (SPEC 8, Realtime speech route; Conversation state; Voice, expressivity and mood)."""

from __future__ import annotations

import time

import pytest
from roboemot.config.schema import StateEffects
from roboemot.soothe.synth import db_to_gain
from rt_rig import build_rt, rt_route, wait_for
from test_conversation_tools import Env, Store
from turn_server import REPLY, TurnServer


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


def response_creates(server):
    return server.of("response.create")


# ------------------------------------------------------------------ a whole turn


def test_a_whole_turn_text_in_and_the_server_voice_out(rigs):
    with TurnServer(transcript="Come stai oggi?") as server:
        rig = rigs(server)
        rig.turn("ciao robot")
        kinds = server.kinds()
        assert kinds[0] == "session.update"
        items = server.of("conversation.item.create")
        assert items[0]["item"]["content"][0] == {"type": "input_text", "text": "ciao robot"}
        assert items[0]["item"]["id"]
        create = response_creates(server)[0]["response"]
        assert "tool_choice" not in create and "[mood=" not in create["instructions"]
        update = server.of("session.update")[-1]["session"]
        assert "voice" not in update["audio"]["output"]  # empty: the server's default
        assert update["audio"]["input"]["turn_detection"]["create_response"] is False
        # the audio played through the speech output: timeline, echo reference, question flag
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.sink.pushed and rig.sink.peak() > 0
        ref = rig.sink.reference.window(time.monotonic() - 10, time.monotonic() + 10)
        assert ref.size > 0 and float(abs(ref).max()) > 0
        counts = rig.speech.timeline.snapshot()
        assert counts.get("sentences", 0) == 1 and counts.get("frames", 0) > 0
        assert rig.speech.question_end() is not None
        with rig.speech.timeline._lock:
            assert [s.question for s in rig.speech.timeline._sentences] == [True]
        # history mirror: the user's words and the reply's transcript
        history = rig.history()
        assert [m["role"] for m in history[0]] == ["user", "assistant"]
        assert history[0][1]["content"] == "Come stai oggi?"
        # the budget is charged from usage
        assert rig.hub.router.used("rt") == 42
        assert rig.chat_hosts() == []  # the chat routes were not asked
        snap = rig.route.snapshot()
        assert snap["state"] == "connected" and snap["utterances_sent"] == 1
        assert snap["responses"] == {"completed": 1}
        assert rig.events[-1]["route"] == "rt"


def test_the_reply_volume_is_applied_as_gain(rigs):
    with TurnServer() as server:
        rig = rigs(server)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        loud = rig.sink.peak()
        rig.sink.pushed.clear()
        rig.circadian[0] = StateEffects(volume_db=-6.0)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 2)
        assert rig.sink.peak() == pytest.approx(loud * db_to_gain(-6.0), rel=0.05)


# ------------------------------------------------------------------ tools


def tooled(rigs, server, store=None, **kw):
    rig = rigs(server, **kw)
    env = Env()
    rig.worker.skills = env.registry
    rig.worker.preferences = store
    rig.worker._speaker = lambda: "p1"
    return rig, env


def test_a_tool_round_trip(rigs):
    with TurnServer(["tool", "reply"], tool_name="play", tool_args='{"gesture": "joy"}') as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert env.played == ["joy"]
        outputs = [
            e["item"] for e in srv.of("conversation.item.create") if e["item"]["type"] != "message"
        ]
        assert outputs[0]["type"] == "function_call_output" and outputs[0]["call_id"] == "call_1"
        assert len(response_creates(srv)) == 2
        update = srv.of("session.update")[-1]["session"]
        assert "play" in [t["name"] for t in update["tools"]]
        assert rig.route.snapshot()["tool_calls"] == 1
        exchange = rig.history()[0]
        assert [m["role"] for m in exchange] == ["user", "assistant", "tool", "assistant"]


def test_the_tool_round_cap(rigs):
    with TurnServer(["tool"], tool_name="play", tool_args='{"gesture": "joy"}') as srv:
        rig, env = tooled(rigs, srv, app={"conversation": {"max_tool_rounds": 1}})
        rig.turn()
        creates = response_creates(srv)
        assert [c["response"]["tool_choice"] for c in creates] == ["auto", "none"]
        outputs = [
            e["item"]["output"]
            for e in srv.of("conversation.item.create")
            if e["item"]["type"] == "function_call_output"
        ]
        assert "too_many_rounds" in outputs[-1]
        assert env.played == ["joy"]  # the first call ran, the one past the cap did not


def test_stop_talking_ends_the_turn(rigs):
    with TurnServer(["tool"], tool_name="stop_talking") as srv:
        rig, _ = tooled(rigs, srv)
        rig.turn()
        assert len(response_creates(srv)) == 1  # no response after the results
        assert rig.sink.stops >= 1
        assert rig.worker.snapshot().get("stopped") == 1


def test_remember_preference_asks_in_the_server_voice(rigs):
    args = '{"key": "colore", "value": "blu"}'
    with TurnServer(["tool", "reply"], tool_name="remember_preference", tool_args=args) as srv:
        rig, _ = tooled(rigs, srv, store=Store())
        rig.turn()
        creates = response_creates(srv)
        assert len(creates) == 2  # the turn's response, then the question as a line
        line = creates[1]["response"]
        assert line["tool_choice"] == "none" and "blu" in line["instructions"]
        assert rig.worker._confirm is not None  # pending on the line's speech job
        assert rig.route.snapshot()["lines"] == 1


# ------------------------------------------------------------------ barge-in


def barge_in(rig):
    rig.worker.mark_interrupting()
    rig.speech.stop()
    return rig.worker.interrupt()


def test_barge_in_mid_response_cancels_truncates_and_marks_the_history(rigs):
    with TurnServer(deltas=30, delta_ms=100, gap_s=0.05) as srv:
        rig = rigs(srv)
        n = rig.say()
        assert wait_for(lambda: rig.sink.pushed and rig.speech.stream_progress(1) is not None)
        time.sleep(0.3)
        assert barge_in(rig)
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n)
        assert wait_for(lambda: srv.of("conversation.item.truncate"))
        assert srv.of("response.cancel")
        cut = srv.of("conversation.item.truncate")[0]
        assert cut["item_id"] == "item_1" and 0 < cut["audio_end_ms"] < 3000
        assistant = rig.history()[0][-1]
        assert assistant["interrupted"] is True
        assert len(assistant["content"].split()) < len(REPLY.split())


def test_barge_in_after_the_response_ended_truncates_what_still_plays(rigs):
    with TurnServer(deltas=20, delta_ms=100) as srv:  # 2 s of audio, sent at once
        rig = rigs(srv)
        rig.turn()  # the response is done; its audio still plays
        time.sleep(0.3)
        assert barge_in(rig)
        assert wait_for(lambda: srv.of("conversation.item.truncate"))
        cut = srv.of("conversation.item.truncate")[0]
        assert 0 < cut["audio_end_ms"] < 2000
        assert rig.history()[0][-1]["interrupted"] is True


# ------------------------------------------------------------------ budget


def test_the_budget_is_estimated_without_usage(rigs):
    with TurnServer(usage=None) as srv:
        rig = rigs(srv)
        rig.turn()
        used = rig.hub.router.used("rt")
        assert used > 0 and used != 42


# ------------------------------------------------------------------ fallbacks


def test_fallback_at_start_without_the_extra(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        rig.turn()
        assert rig.chat_hosts() == ["lan.local"] and srv.requests == []
        snap = rig.route.snapshot()
        assert snap["state"] == "blocked" and snap["reason"] == "no_extra"
        assert snap["fallbacks_start"] == 1


def test_fallback_at_start_when_the_connect_is_refused(rigs):
    with TurnServer() as srv:
        dead = rt_route(srv, base_url="http://127.0.0.1:9/v1")
        rig = rigs(srv, route=dead)
        rig.turn()
        assert rig.chat_hosts() == ["lan.local"]
        assert rig.route.snapshot()["state"] == "fallback_local"
        assert rig.route.snapshot()["connect_failures"] == 1


def test_fallback_at_start_when_the_budget_is_spent(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, route=rt_route(srv, daily_token_budget=1000))
        rig.hub.router.charge("rt", 1000)
        rig.turn()
        assert rig.chat_hosts() == ["lan.local"] and srv.requests == []
        assert rig.route.snapshot()["reason"] == "budget"


def test_a_session_lost_before_any_audio_answers_through_the_chat_routes(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.chat_hosts() == ["lan.local"]
        assert rig.route.snapshot()["fallbacks_mid_turn"] == 1


def test_a_session_lost_after_audio_cuts_the_reply(rigs):
    with TurnServer(["close_mid"], delta_ms=1500, gap_s=0.3) as srv:  # plays, then closes
        rig = rigs(srv)
        rig.turn()
        assert rig.chat_hosts() == []  # not repeated
        assert rig.sink.stops >= 1
        assert rig.route.snapshot()["cut_mid_turn"] == 1
        assert rig.history()[0][-1]["interrupted"] is True


# ------------------------------------------------------------------ session, history


def test_the_session_end_closes_the_server_session(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.worker.expire(time.monotonic() + 3600)
        assert wait_for(lambda: rig.route.snapshot()["state"] == "closed")
        assert wait_for(lambda: all(ws.state.name == "CLOSED" for ws in srv.connections))
        rig.turn()  # the next turn opens a new session
        assert len(srv.connections) == 2


def test_history_beyond_the_cap_deletes_the_server_items(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, app={"conversation": {"history_turns": 1}})
        rig.turn()
        first = srv.of("conversation.item.create")[0]["item"]["id"]
        rig.turn()
        assert wait_for(lambda: srv.of("conversation.item.delete"))
        deleted = {e["item_id"] for e in srv.of("conversation.item.delete")}
        assert deleted == {first, "item_1"}
        assert rig.route.snapshot()["items_deleted"] == 2


def test_an_unrequested_response_is_cancelled_and_counted(rigs):
    with TurnServer(unrequested=True) as srv:
        rig = rigs(srv)
        assert rig.route.ready()
        assert wait_for(lambda: rig.route.snapshot()["unrequested"] == 1)
        assert wait_for(lambda: srv.of("response.cancel"))


# ------------------------------------------------------------------ review round 1


def test_a_barge_in_charges_the_cancelled_responses_usage(rigs):
    with TurnServer(deltas=30, delta_ms=100, gap_s=0.05) as srv:
        rig = rigs(srv)
        n = rig.say()
        assert wait_for(lambda: rig.sink.pushed and rig.speech.stream_progress(1) is not None)
        barge_in(rig)
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n)
        assert wait_for(lambda: rig.route.snapshot()["responses"].get("cancelled") == 1)
        assert rig.hub.router.used("rt") == 42  # the usage of response.done, not an estimate


def test_the_realtime_link_recovers_with_the_default_policy(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        rig.network.set_routes([r for r in rig.stack.effective().llm_routes if r.name == "rt"])
        for _ in range(rig.network.policy.failures_to_offline):
            rig.network.report_realtime(False)
        assert not rig.network.online
        rig.turn()  # one connect and one finished response: two successes
        assert rig.network.online


# ------------------------------------------------------------------ fixed lines


def test_the_offline_cue_uses_the_server_voice_while_the_route_is_up(rigs):
    from roboemot.config.schema import RegisterParams
    from roboemot.health.speech_cue import SpeechCue

    with TurnServer() as srv:
        rig = rigs(srv)
        assert rig.route.ready()
        cue = SpeechCue(
            lambda: rig.speech,
            lambda: (RegisterParams(), StateEffects(), "joy", 0.5),
            lambda: False,
            rig.worker.say_line,
        )
        assert cue.say("Sono senza rete.", "it")
        assert wait_for(lambda: response_creates(srv))
        assert "Sono senza rete." in response_creates(srv)[0]["response"]["instructions"]
        assert rig.speech.backend.texts == []  # Piper said nothing


def test_the_offline_cue_uses_piper_while_the_route_is_down(rigs):
    from roboemot.config.schema import RegisterParams
    from roboemot.health.speech_cue import SpeechCue

    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        cue = SpeechCue(
            lambda: rig.speech,
            lambda: (RegisterParams(), StateEffects(), "joy", 0.5),
            lambda: False,
            rig.worker.say_line,
        )
        assert cue.say("Sono senza rete.", "it")
        assert wait_for(lambda: rig.speech.backend.texts == ["Sono senza rete."])


def test_a_confirmation_reply_line_is_spoken_in_the_server_voice(rigs):
    args = '{"key": "colore", "value": "blu"}'
    script = ["tool", "reply", "reply"]
    with TurnServer(script, tool_name="remember_preference", tool_args=args, deltas=1) as srv:
        store = Store()
        rig, _ = tooled(rigs, srv, store=store)
        rig.turn()
        pending = rig.worker._confirm
        assert pending is not None
        assert wait_for(lambda: (rig.speech.ended_at(pending.job) or 1e18) < time.monotonic())
        rig.turn("sì", stamp=time.monotonic() + 1.0)
        assert store.saved == [("p1", "colore", "blu")]
        creates = response_creates(srv)
        assert len(creates) == 3 and creates[2]["response"]["tool_choice"] == "none"
        assert rig.speech.backend.texts == []  # every line in the server voice


def test_a_line_falls_back_to_piper_when_the_session_drops_before_it(rigs):
    args = '{"key": "colore", "value": "blu"}'
    script = ["tool", "close_before"]
    with TurnServer(script, tool_name="remember_preference", tool_args=args) as srv:
        rig, _ = tooled(rigs, srv, store=Store())
        rig.turn()
        assert wait_for(lambda: any("blu" in t for t in rig.speech.backend.texts))
        assert rig.worker._confirm is not None  # pending on the Piper sentence


# ------------------------------------------------------------------ pseudonyms (SPEC 12)


class CloudAllocator:
    """A provider that allocates a cloud host for a local base_url."""

    key_hosts = ()

    def __init__(self):
        self.calls = 0

    async def allocate(self, request):
        from roboemot.realtime.session import Allocation

        self.calls += 1
        return Allocation("wss://rt.example.com/v1/realtime")


def test_pseudonyms_refuse_a_cloud_host_a_provider_allocated(rigs):
    with TurnServer() as srv:
        provider = CloudAllocator()
        rig = rigs(srv, provider=provider, app={"privacy": {"pseudonyms_to_cloud": True}})
        rig.turn()
        assert provider.calls == 1
        assert rig.chat_hosts() == ["lan.local"]  # the local pipeline answered
        snap = rig.route.snapshot()
        assert snap["state"] == "blocked" and snap["reason"] == "pseudonyms"
        assert srv.requests == []


def test_repr_holds_no_reply_text_or_ids():
    from roboemot.realtime.turn import Reply

    reply = Reply(texts={"item_9": "Quintavalle"}, jobs=[(1, "item_9")], response_id="resp_9")
    assert "Quintavalle" not in repr(reply) and "item_9" not in repr(reply)
    assert "resp_9" not in repr(reply)
