"""Server turns, second part (S1b; SPEC 8, Realtime speech route, Server turns; SPEC 12; SPEC
13): with ``turns: server`` the session's server VAD answers on its own (``create_response``),
and the response it created for a stream that ended whole is the turn's answer: it plays once,
the robot sends no ``response.create`` for it, and the stream's transcript fills the history
whenever it comes. A response for a fragment (a stream cut, revoked, from an old session, or
ended by the server inside the segment) is cancelled and never reaches the speaker; a second
response for an answered stream, or one while a turn's response runs, is a duplicate."""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest
from realtime_helpers import wait_until
from roboemot.realtime import protocol
from roboemot.realtime.input import InputJob, _Burst, _Owed
from roboemot.realtime.route import session_update
from test_realtime_follow_up import LOOK, SKILLS, Env
from test_realtime_listener import ears  # noqa: F401 (fixture)
from test_realtime_server_turns import HEARD, OPT_INS, SERVER, tail
from test_voice_listener import voice
from turn_server import REPLY, TurnServer


def stream_job():
    job = InputJob(np.zeros(0), 16000)
    job.stream = True
    return job


def detect(update: dict) -> dict:
    return update["session"]["audio"]["input"]["turn_detection"]


def pushed(ear) -> int:
    """Audio frames the speech output handed to the speaker."""
    return len(ear.rt.sink.pushed)


def turns(ear) -> int:
    return ear.rt.worker.snapshot().get("turns", 0)


def snap(ear) -> dict:
    return ear.rt.route.snapshot()


def cut_by_speaking(ear) -> None:
    """Half a segment streams, then the robot starts speaking: the next piece is refused and
    the stream is cut (its closing silence goes, and the server ends the turn on it)."""
    ear.r.feed(voice(seconds=0.5))
    assert wait_until(lambda: ear.rt.server.sent_of("input_audio_buffer.speech_started"))
    ear.input._speaking = lambda: True
    ear.r.feed(voice(seconds=0.3))
    assert wait_until(lambda: snap(ear)["streams_cut"] == 1)


REPLY_S = 0.5  # longer than the fake's reply takes to stream: any of it would have played


def fragment_cancelled(ear, server) -> None:
    """One fragment, cancelled once (no other response was live: the plain cancel)."""
    assert snap(ear)["server_fragments"] == 1 and len(server.auto_responses) == 1
    assert wait_until(lambda: len(server.of("response.cancel")) == 1)
    assert "response_id" not in server.of("response.cancel")[0]


def never_played(ear, server, *, seen: bool = True) -> None:
    """The server's own answer to a fragment: dropped (a fragment, or an old session's stale
    callback; ``seen`` False: a reset may have closed its socket first) and none of it played.
    No turn answered it (a turn that showed the did-not-get-it cue is no answer)."""
    if seen:
        assert wait_until(lambda: snap(ear)["server_fragments"] + snap(ear)["stale_callbacks"] >= 1)
    assert not wait_until(lambda: pushed(ear) > 0, REPLY_S)
    assert snap(ear)["server_responses"] == 0
    assert all(e["outcome"] != "ok" for e in ear.rt.events)
    assert server.of("response.create") == []


# ------------------------------------------------------------------ the session


def test_server_turns_let_the_server_answer_and_robot_turns_do_not(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        updates = server.of("session.update")
        assert all(detect(u)["create_response"] is True for u in updates)  # connect, opening
        assert all(detect(u)["interrupt_response"] is False for u in updates)  # S2 changes it
        cfg = ear.rt.stack.effective()
        assert detect(session_update(cfg, "x", create_response=False))["create_response"] is False
    with TurnServer() as server:
        ear = ears(server, script=["x"])
        assert all(detect(u)["create_response"] is False for u in server.of("session.update"))


def test_the_realtime_check_never_lets_the_server_answer_on_its_own():
    """The check asks for its one response itself, also with server turns configured."""
    from fake_server import FakeServer
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.realtime.route import server_turns
    from test_realtime_check import make_route, make_router, run

    with FakeServer(delete_reply="deleted") as server:
        cfg = RoboEmotConfig(
            llm_routes=[make_route(server.http_base())],
            voice={"realtime_route": {**SERVER, "input": "audio"}},
            privacy=OPT_INS["privacy"],
        )
        assert server_turns(cfg)
        assert run(cfg, make_router(cfg)).ok
    (update,) = [e for e in server.received if e["type"] == "session.update"]
    assert detect(update)["create_response"] is False


def test_robot_turns_back_at_runtime_turn_the_servers_own_answers_off(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=[["ciao"]], vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], bearing=0.0)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        stack.set("voice.realtime_route.audio_gate", "addressed", source="t", admin=True)
        ear.r.listener.apply_config(stack.effective())
        server.stream_vad = False  # robot turns: the fake answers the burst's commit
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        kinds = server.kinds()
        commit = kinds.index("input_audio_buffer.commit")
        last = max(i for i, k in enumerate(kinds[:commit]) if k == "session.update")
        # the burst's own update, before its audio, turns the server's answers off
        assert detect(server.received[last])["create_response"] is False


# ------------------------------------------------------------------ the server's own answer


def test_in_server_turns_the_servers_own_response_plays_once(ears):  # noqa: F811
    """Server-turns counterpart of ``test_realtime_audio_turns``' first test: the same input
    (a server that answers on its own) is the answer, played once, never cancelled."""
    with TurnServer(stream_vad=True, heard=[["ciao Neve"]], vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 1, 10.0)
        assert server.of("response.create") == [] and server.of("response.cancel") == []
        s = snap(ear)
        assert s["server_responses"] == 1 and s["unrequested"] == 0
        assert s["server_duplicates"] == 0 and s["server_fragments"] == 0
        assert ear.rt.speech.snapshot()["spoken"] == 1 and pushed(ear) > 0
        assert ear.rt.events[-1]["outcome"] == "ok"
        assert wait_until(lambda: ear.rt.history() != [])
        (entry,) = ear.rt.history()
        assert [m["role"] for m in entry] == ["user", "assistant"]
        assert entry[0]["content"] == "ciao Neve" and entry[1]["content"] == REPLY
        assert entry[0]["rt"][1] == ["in_1_0", "item_1"]  # the stream's item and the reply's


def test_a_bare_server_response_on_a_server_turns_session_is_still_cancelled(ears):  # noqa: F811
    """Server-turns counterpart of ``test_realtime_turns``' unrequested test: a response with
    no stream speech behind it answers nothing, so it is a duplicate, cancelled by its id."""
    with TurnServer(stream_vad=True, unrequested=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        assert wait_until(lambda: snap(ear)["server_duplicates"] == 1)
        assert wait_until(lambda: server.of("response.cancel"))
        assert "response_id" not in server.of("response.cancel")[0]  # alone: the plain cancel
        assert snap(ear)["unrequested"] == 0 and pushed(ear) == 0


def test_the_answer_starts_before_the_transcript_which_fills_the_history_later(ears):  # noqa: F811
    hold = threading.Event()
    with TurnServer(
        stream_vad=True, heard=[[HEARD]], vad_answers=True, hold=hold, hold_at="transcript"
    ) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)  # answered with no transcript yet
        assert ear.utterances == [] and pushed(ear) > 0
        assert ear.rt.history()[0][0]["content"] == ""
        hold.set()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert wait_until(lambda: ear.rt.history()[0][0]["content"] == HEARD)
        assert turns(ear) == 1 and server.of("response.create") == []
        assert server.of("conversation.item.delete") == []  # its words stay with the turn


def test_a_transcript_before_the_answer_waits_for_the_server_turn(ears):  # noqa: F811
    with TurnServer(
        stream_vad=True, heard=[[HEARD]], vad_answers=True, answer_delay_s=0.5
    ) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert turns(ear) == 0  # the words alone start no turn
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert ear.rt.history()[0][0]["content"] == HEARD
        assert server.of("response.create") == []


def test_the_turn_latency_runs_from_the_end_of_speech_in_both_modes(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        latency = snap(ear)["latency_ms"]
        assert latency["turn"]["count"] == 1
        assert set(latency["turn"]) == set(latency["first_audio"]) == {"count", "p50", "p95", "max"}
        # from the segment's last speech as the listener heard it (carried by its stream, so
        # known before any transcript) to the first audio handed to the speech output
        job = next(iter(ear.rt.worker._claims))
        last = ear.rt.worker._last
        assert job.start_stamp < job.speech_end <= ear.r.now and last["heard"] == job.speech_end
        expected = (last["token"] - job.speech_end) * 1000
        assert latency["turn"]["p50"] == pytest.approx(expected)
    with TurnServer() as server:  # robot turns, text in
        ear2 = ears(server, script=["x"])
        ear2.rt.turn("ciao")
        latency = snap(ear2)["latency_ms"]
        last = ear2.rt.worker._last
        assert latency["turn"]["count"] == 1
        assert latency["turn"]["p50"] == pytest.approx((last["token"] - last["heard"]) * 1000)


def test_the_listening_cue_ends_when_the_answer_starts(ears):  # noqa: F811
    hold = threading.Event()
    with TurnServer(stream_vad=True, vad_answers=True, hold=hold, hold_at="transcript") as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert ear.rt.cues.ended >= 1  # before any transcript
        hold.set()


def test_an_answered_stream_whose_transcript_never_comes_shows_no_cue(ears):  # noqa: F811
    hold = threading.Event()
    rt = {**SERVER, "transcript_timeout_s": 1.0}
    with TurnServer(stream_vad=True, vad_answers=True, hold=hold, hold_at="transcript") as server:
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        ear.r.now += 2.0  # the listener's clock: past its transcript deadline
        assert ear.settle(
            lambda: ear.r.listener.snapshot()["realtime_audio"]["answered_untranscribed"] == 1,
            5.0,
        )
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["local_fallbacks"] == 0 and audio["decodes_lost"] == 0
        assert ear.utterances == [] and "not_heard" not in ear.rt.cues.played
        assert server.of("conversation.item.delete") == []  # the turn keeps its item
        hold.set()


# ------------------------------------------------------------------ a pause, duplicates


def test_a_pause_gives_one_answer_per_server_response_and_no_robot_request(ears):  # noqa: F811
    """S1a answered a sentence with a pause twice (the robot's request on top of the server's
    commit). Now: the first answer is done before the second segment; each server response is
    one turn, played once."""
    heard = [["prima"], ["seconda"]]
    with TurnServer(stream_vad=True, heard=heard, vad_answers=True, deltas=1) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 1, 10.0)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 2 and len(ear.utterances) == 2, 10.0)
        assert server.of("response.create") == [] and server.of("response.cancel") == []
        assert snap(ear)["server_responses"] == 2
        assert [t[0]["content"] for t in ear.rt.history()] == ["prima", "seconda"]
        assert ear.rt.speech.snapshot()["spoken"] == 2


def test_a_server_response_while_the_turns_answer_runs_is_a_duplicate(ears):  # noqa: F811
    """The person goes on while the first answer is still streaming: the server's second
    response is cancelled by its id and never plays; the second segment's words join the turn
    that answered (they stay on the server: the next answer hears them)."""
    heard = [["prima"], ["seconda"]]
    with TurnServer(stream_vad=True, heard=heard, vad_answers=True, deltas=10, gap_s=0.1) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        ear.say()
        assert wait_until(lambda: len(server.auto_responses) == 2)
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 2, 15.0)
        s = snap(ear)
        assert s["server_responses"] == 1 and s["server_duplicates"] == 1
        cancels = server.of("response.cancel")
        assert [c.get("response_id") for c in cancels] == [server.auto_responses[1]]
        assert server.of("response.create") == []
        assert ear.rt.speech.snapshot()["spoken"] == 1  # one answer played
        assert wait_until(lambda: ear.rt.history()[0][0]["content"] == "prima seconda")
        assert {"in_1_0", "in_2_0"} <= set(ear.rt.history()[0][0]["rt"][1])
        # the dropped answer's own items go at its response.done; the person's words stay
        assert wait_until(lambda: server.of("conversation.item.delete"))
        deleted = {e["item_id"] for e in server.of("conversation.item.delete")}
        assert deleted <= {"item_2", "fc_2"} and not deleted & {"in_1_0", "in_2_0"}


def test_a_second_response_for_an_answered_stream_is_a_duplicate(ears):  # noqa: F811
    from roboemot.realtime.input import _Burst, _Owed

    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        inp, gen = ear.input, ear.rt.route.generation
        job = next(iter(ear.rt.worker._claims))
        assert job.answered
        # the server ends one more turn for that same stream (a split): owed again
        stream = _Burst(job, gen, utt_ms=0.0)
        inp._owed.append(_Owed(gen, stream, 10_000, inp._clock()))
        claim = inp.claim(gen, 5.0)
        assert claim is not None and claim.verdict == "duplicate"


# ------------------------------------------------------------------ fragments never play


def test_a_stream_cut_when_the_robot_speaks_is_never_answered(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        cut_by_speaking(ear)
        assert wait_until(lambda: server.auto_responses)  # it answered the fragment
        never_played(ear, server)
        fragment_cancelled(ear, server)


def test_a_stream_the_budget_cuts_is_never_answered(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        rt = {**SERVER, "daily_upload_budget_s": 1.5}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        assert wait_until(lambda: server.auto_responses)
        never_played(ear, server)
        fragment_cancelled(ear, server)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert ear.utterances[0].local_only  # not answered, as in S1a


def test_a_stream_a_chunk_gap_cuts_is_never_answered(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.r.feed(voice(seconds=0.1), gap=True)
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        assert wait_until(lambda: server.auto_responses)
        never_played(ear, server)
        fragment_cancelled(ear, server)


def test_a_stream_revoked_before_its_answer_is_never_answered(ears):  # noqa: F811
    """The stream ended whole, the server holds its answer back; the opt-in goes off meanwhile.
    The revoke resets the session, so the late answer is an old session's (or a revoked
    stream's): it never plays."""
    with TurnServer(stream_vad=True, vad_answers=True, answer_delay_s=0.5) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: server.sent_of("input_audio_buffer.committed"))
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.route.apply_config(stack.effective())
        assert wait_until(lambda: server.auto_responses)
        # robot turns now: the owed speech went with the revoke, the late answer is nobody's
        never_played(ear, server, seen=False)


def test_a_revoked_stream_on_a_live_session_is_a_fragment(ears):  # noqa: F811
    """The claim itself, without the reset: a stream of an older revocation epoch is never
    answered."""
    with TurnServer(stream_vad=True, vad_answers=True, answer_delay_s=0.5) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: server.sent_of("input_audio_buffer.committed"))
        ear.input._epoch += 1  # the revoke's epoch, its session reset aside
        assert wait_until(lambda: snap(ear)["server_fragments"] == 1)
        never_played(ear, server)


def test_a_reconnect_between_pieces_leaves_nothing_to_answer(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        assert ear.rt.route.reset_session(ear.rt.route.generation)
        assert wait_until(lambda: ear.rt.route.up(), 10.0)
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert snap(ear)["streams_cut"] == 1
        assert ear.utterances[0].local_only  # its audio left, it is not answered
        never_played(ear, server, seen=False)  # nothing of it reached the new session


def test_an_old_sessions_answer_never_plays(ears):  # noqa: F811
    """The server answered on the session a reconnect left: its callbacks are stale."""
    with TurnServer(stream_vad=True, vad_answers=True, answer_delay_s=0.5) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: server.sent_of("input_audio_buffer.committed"))
        assert ear.rt.route.reset_session(ear.rt.route.generation)
        assert wait_until(lambda: ear.rt.route.up(), 10.0)
        assert wait_until(lambda: len(server.sent_of("response.created")) == 1)
        never_played(ear, server, seen=False)  # a stale callback, or never delivered


def test_a_cut_stream_on_a_server_without_item_ids_is_never_answered(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True, ids=False) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        cut_by_speaking(ear)
        assert wait_until(lambda: snap(ear)["input_resets"] == 1)  # S1a: reset, no ids
        assert wait_until(lambda: server.auto_responses)
        # a fragment if it came before the reset, an old session's after it, or never sent
        never_played(ear, server, seen=False)


def test_a_whole_stream_on_a_server_without_item_ids_is_answered(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True, ids=False, heard=[[HEARD]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 1, 10.0)
        assert snap(ear)["server_responses"] == 1 and pushed(ear) > 0
        assert wait_until(lambda: ear.rt.history()[0][0]["content"] == HEARD)


def test_a_server_that_ends_the_turn_inside_the_segment_answered_a_fragment(ears):  # noqa: F811
    """The claim's coverage: the server's end lies before the segment's end (more speech came
    after it), so its answer is a fragment."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation

        def verdict(end_ms):
            stream = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
            inp._owed.append(_Owed(gen, stream, end_ms, inp._clock()))
            return inp.claim(gen, 5.0).verdict

        assert verdict(800) == "fragment"  # ended at 0.8 s of 2 s
        assert verdict(1500) == "answer"  # within the end silence
        assert verdict(None) == "fragment"  # no end position: never whole


# ------------------------------------------------------------------ tools


def tooled(ear) -> None:
    stack = ear.rt.stack
    stack.set("conversation.llm_skills", SKILLS, source="test", admin=True)
    ear.rt.worker.skills = Env().registry
    ear.rt.worker._speaker = lambda: "p1"


def test_the_streams_opening_gives_the_server_the_turns_tools(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt={**SERVER, "skip_tool": True}, app=OPT_INS, script=["x"])
        tooled(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        opening = server.of("session.update")[-1]["session"]
        names = {t["name"] for t in opening["tools"]}
        assert {"play", "look_at", "ignore_utterance"} <= names
        assert "ignore_utterance" in opening["instructions"]  # the skip rule


def test_a_fire_and_forget_call_in_the_servers_answer_asks_nothing_more(ears):  # noqa: F811
    with TurnServer(
        ["audio_then_tool"], tool_name="look_at", tool_args=LOOK, stream_vad=True,
        vad_answers=True,
    ) as server:  # fmt: skip
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        tooled(ear)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert server.of("response.create") == []
        outputs = [
            e for e in server.of("conversation.item.create")
            if e["item"]["type"] == "function_call_output"
        ]  # fmt: skip
        assert len(outputs) == 1
        assert ear.rt.worker.snapshot()["follow_ups_skipped"] == 1


def test_a_skill_that_needs_a_follow_up_still_asks_once(ears):  # noqa: F811
    with TurnServer(
        ["audio_then_tool", "reply"], tool_name="recall", stream_vad=True, vad_answers=True
    ) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        tooled(ear)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert len(server.of("response.create")) == 1  # the follow-up, after the output
        kinds = server.kinds()
        assert kinds.index("response.create") > max(
            i for i, e in enumerate(server.received)
            if e.get("type") == "conversation.item.create"
        )  # fmt: skip
        assert not ear.rt.route.input_held()  # the follow-up's hold is released


# ------------------------------------------------------------------ review round


def test_a_server_turn_lost_before_any_audio_says_the_fallback_locally(ears):  # noqa: F811
    """The session goes between the server's response.created and its first audio: nothing
    played and no words reached the robot, so Piper says the fallback line; the server is not
    asked to say it on a turn it answers itself."""
    with TurnServer(["created_close"], stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert ear.rt.events[-1]["outcome"] == "realtime_lost"
        assert ear.rt.worker.snapshot()["fallbacks"] == 1
        assert wait_until(lambda: ear.rt.speech.snapshot()["spoken"] == 1)  # Piper said it
        assert server.of("response.create") == []


def test_a_late_transcript_of_a_skipped_server_turn_deletes_nothing_twice(ears):  # noqa: F811
    from roboemot.realtime.turn import SKIP_TOOL

    hold = threading.Event()
    with TurnServer(
        ["tool"], tool_name=SKIP_TOOL, stream_vad=True, vad_answers=True, hold=hold,
        hold_at="transcript",
    ) as server:  # fmt: skip
        ear = ears(
            server, rt={**SERVER, "skip_tool": True}, app=OPT_INS, script=["x"], to_worker=True
        )
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert ear.rt.events[-1]["outcome"] == "skipped"
        hold.set()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        deleted = [e["item_id"] for e in server.of("conversation.item.delete")]
        assert deleted.count("in_1_0") == 1 and ear.rt.history() == []


def test_a_server_response_while_a_robot_turn_runs_joins_its_words_to_that_turn(ears):  # noqa: F811
    """A robot turn (text in) is in flight: the server's own answer to a stream is refused,
    and that stream's words join the robot's turn; its listening cue ends at once."""
    from roboemot.realtime.input import InputJob
    from roboemot.realtime.route import Request, ServerResponse
    from roboemot.voice.listener import Utterance
    from roboemot.voice.speech_features import Prosody

    with TurnServer(["silent"]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        worker = ear.rt.worker
        ear.rt.say("prima")  # its response never comes: the turn stays in flight
        assert wait_until(lambda: worker._active_turn is not None)
        job = InputJob(np.zeros(0), 16000)
        job.start_stamp = job.speech_end = 1.0
        offer = ServerResponse(Request("", None, 0.0), ear.rt.route.generation, job, ())
        ended = ear.rt.cues.ended
        assert not worker._server_response(offer)
        assert ear.rt.cues.ended == ended + 1
        quiet = Prosody(0.0, None, None, 0.0)
        words = Utterance("seconda", "it", 1.0, 1.0, 1.0, True, False, frozenset(), quiet,
                          stream=job)  # fmt: skip
        assert worker.submit(words)
        assert worker._active_turn.utterance.text == "prima seconda"


# ------------------------------------------------------------------ review round 2 (PR 196)


def test_two_fresh_owed_entries_fail_closed(ears):  # noqa: F811
    """Finding 1: the server skipped A's answer (an error, an active response), C was cut and
    its closing silence ended a turn; C's response must not be taken for A's."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        whole = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
        cut = _Burst(stream_job(), gen, start_ms=3000.0, utt_ms=800.0, silence_ms=600.0, cut=True)
        now = inp._clock()
        inp._owed += [_Owed(gen, whole, 1900, now), _Owed(gen, cut, 3900, now)]
        assert inp.claim(gen, 5.0).verdict == "fragment"
        assert not whole.job.claimed  # nothing was taken for A


def test_an_error_about_no_client_event_resolves_the_oldest_owed_speech(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        stream = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
        inp._owed.append(_Owed(gen, stream, 1900, inp._clock()))
        refusal = protocol.ServerError("error", "conversation_already_has_active_response")
        ear.rt.route._error(refusal, None)
        # declined: nothing is waited for, and a response that comes anyway matches nothing
        assert inp._owed == [] and inp.claim(gen, 5.0) is None and not stream.job.claimed


def test_a_cut_stream_is_unsettled_until_the_server_reports_it(ears):  # noqa: F811
    """Finding 2: after a cut, before the server's speech_stopped, a robot request must wait."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, route = ear.input, ear.rt.route
        cut = _Burst(stream_job(), route.generation, utt_ms=800.0, silence_ms=600.0, cut=True)
        cut.sent_ms, cut.end_at = 800.0, inp._clock()
        inp._streams = [cut]
        assert inp.unsettled(route.generation, 5.0) and route.busy()
        cut.end_at -= route.stack.effective().voice.realtime_route.server_answer_wait_s
        assert not route.busy()  # bounded: the server had its time


def test_a_request_past_the_wait_while_the_server_may_answer_resets_the_session(ears):  # noqa: F811
    """Finding 2: the 5 s escape of wait_in_order never binds a robot request to a session the
    server may still answer on its own: the session is reset."""
    import asyncio

    from roboemot.realtime.turn import wait_in_order

    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        route = ear.rt.route
        ear.r.feed(voice(seconds=0.5))  # a stream that still streams
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        assert route.busy()
        ticks = iter(range(1000))
        asyncio.run(wait_in_order(route, lambda: float(next(ticks))))  # past stall at once
        assert wait_until(lambda: snap(ear)["unsettled_resets"] == 1)


def test_a_response_while_its_stream_still_streams_is_judged_at_its_end(ears):  # noqa: F811
    """Finding 3a: the server ended the turn on the streamed trailing silence before the
    listener closed the stream: the response is held, then judged whole at the close."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        judged = []
        inp.route = type("R", (), {"judged": lambda self, job, v, items: judged.append(v)})()
        s = _Burst(stream_job(), gen, utt_ms=math.inf, silence_ms=600.0)
        s.sent_ms = 2000.0
        inp._streams = [s]
        inp._owed.append(_Owed(gen, s, 1900, inp._clock()))
        assert inp.claim(gen, 5.0).verdict == "wait" and inp.unsettled(gen, 5.0)
        s.utt_ms = 2000.0  # the close
        inp._judge(s)
        assert judged == ["answer"]
        cut = _Burst(stream_job(), gen, utt_ms=math.inf, silence_ms=600.0)
        inp._owed.append(_Owed(gen, cut, 900, inp._clock()))
        assert inp.claim(gen, 5.0).verdict == "wait"
        cut.cut = True
        inp._judge(cut)
        assert judged == ["answer", "fragment"]


def test_a_held_response_plays_once_its_stream_ends_whole(ears):  # noqa: F811
    """End to end through the route: the server's VAD (its silence as short as the local
    one) ends the turn on the streamed trailing silence, and its response comes before the
    stream thread runs the listener's close: it is held, then answered at the close."""
    with TurnServer(stream_vad=True, vad_answers=True, heard=[[HEARD]]) as server:
        end_ms = 500.0
        rt = {**SERVER, "server_silence_ms": end_ms}
        app = {**OPT_INS, "voice": {"vad": {"end_silence_ms": end_ms}}}
        ear = ears(server, rt=rt, app=app, script=["x"], to_worker=True)
        op = ear.input._stream_op

        def close_after_the_answer(kind, job, arg):
            if kind == "close":
                assert wait_until(lambda: server.sent_of("response.created"))
                assert wait_until(lambda: ear.rt.route._held)  # held while it streams
            op(kind, job, arg)

        ear.input._stream_op = close_after_the_answer
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 1, 10.0)
        assert snap(ear)["server_responses"] == 1 and pushed(ear) > 0
        assert not ear.rt.route._held


def late_answer_cfg() -> dict:
    return {**SERVER, "server_answer_wait_s": 1.0}


def test_a_late_answer_is_dropped_and_the_person_gets_the_not_heard_cue(ears):  # noqa: F811
    """Finding 3b: the server answers past server_answer_wait_s: its answer is nobody's, the
    transcript waits no longer, and the person sees the did-not-get-it cue."""
    with TurnServer(stream_vad=True, vad_answers=True, answer_delay_s=1.5) as server:
        ear = ears(server, rt=late_answer_cfg(), app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: "not_heard" in ear.rt.cues.played, 10.0)
        assert wait_until(lambda: server.auto_responses and snap(ear)["server_duplicates"] == 1)
        assert pushed(ear) == 0 and ear.rt.worker.snapshot()["server_unanswered"] == 1
        assert "in_1_0" in {e["item_id"] for e in server.of("conversation.item.delete")}
        assert not ear.rt.worker._unclaimed


def test_a_server_that_never_answers_gets_the_not_heard_cue(ears):  # noqa: F811
    """Finding 3c: a server that ignores create_response: no answer, no stuck transcript."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=late_answer_cfg(), app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: "not_heard" in ear.rt.cues.played, 10.0)
        assert not ear.rt.worker._unclaimed and server.of("response.create") == []
        assert "in_1_0" in {e["item_id"] for e in server.of("conversation.item.delete")}


def test_a_server_without_end_positions_answers_nothing_and_shows_the_cue(ears):  # noqa: F811
    """Finding 3d: no audio_end_ms, end to end: its answer is a fragment, and the listener's
    transcript deadline shows the did-not-get-it cue (the stream is never covered)."""
    with TurnServer(stream_vad=True, vad_answers=True, end_positions=False) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: snap(ear)["server_fragments"] == 1)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s + 1
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert ear.utterances[0].local_only  # not answered: the not_heard cue (listener)
        assert pushed(ear) == 0 and not ear.rt.worker._unclaimed


def test_a_refused_cancel_id_keeps_the_dropped_answer_silent_and_cleans_it_up(ears):  # noqa: F811
    """Finding 9: the server refuses response_id on response.cancel; the duplicate streams
    on at the server, never plays, and its items go at its response.done."""
    heard = [["prima"], ["seconda"]]
    with TurnServer(
        stream_vad=True, heard=heard, vad_answers=True, deltas=10, gap_s=0.1,
        refuse_cancel_id=True,
    ) as server:  # fmt: skip
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and len(ear.utterances) == 2, 15.0)
        assert snap(ear)["server_duplicates"] == 1 and snap(ear)["errors_unrelated"] >= 1
        assert ear.rt.speech.snapshot()["spoken"] == 1
        assert wait_until(lambda: "item_2" in {
            e["item_id"] for e in server.of("conversation.item.delete")
        })  # fmt: skip


def test_a_dropped_answer_with_audio_and_a_call_has_both_items_deleted(ears):  # noqa: F811
    """Finding 4: a fragment's response that streams on (the cancel ignored) made an audio
    item and a call before its done: both go, so no call stays open."""
    with TurnServer(
        ["audio_then_tool"], tool_name="look_at", stream_vad=True, vad_answers=True,
        ignore_cancel=True, strict_calls=True,
    ) as server:  # fmt: skip
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        cut_by_speaking(ear)
        assert wait_until(lambda: snap(ear)["server_fragments"] == 1)
        assert wait_until(lambda: {"item_1", "fc_1"} <= {
            e["item_id"] for e in server.of("conversation.item.delete")
        })  # fmt: skip
        assert pushed(ear) == 0 and server.open_calls == {}


def test_a_failure_before_the_run_adopts_releases_the_response(ears):  # noqa: F811
    """Finding 7: the server turn fails before its run follows the response: the response is
    cancelled and the next answer still plays."""
    with TurnServer(stream_vad=True, vad_answers=True, deltas=10, gap_s=0.05) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        worker, real = ear.rt.worker, ear.rt.worker._realtime_turn
        calls = []

        async def fail_once(turn):
            calls.append(turn)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return await real(turn)

        worker._realtime_turn = fail_once
        ear.say()
        assert wait_until(lambda: worker.snapshot().get("errors", 0) == 1)
        assert wait_until(lambda: server.of("response.cancel"))
        assert worker._active_turn is None and pushed(ear) == 0
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and pushed(ear) > 0, 10.0)


def test_a_server_response_while_serve_starts_a_turn_is_refused(ears):  # noqa: F811
    """Finding 6: _serve took a server turn off the queue but has not set _active_turn yet:
    a second whole stream's response is a duplicate, its words join that turn."""
    from roboemot.conversation.worker import _ServerTurn
    from roboemot.realtime.route import Request, ServerResponse

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        worker = ear.rt.worker
        first = _ServerTurn(ServerResponse(Request("", None, 0.0), 0, stream_job(), ()))
        with worker._lock:
            worker._serving = first
        job = stream_job()
        offer = ServerResponse(Request("", None, 0.0), ear.rt.route.generation, job, ())
        assert not worker._server_response(offer)
        assert worker._claims[job] is first and job.answered
        with worker._lock:
            worker._serving = None


def test_a_revoke_while_the_servers_answer_plays_stops_it(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True, deltas=30, gap_s=0.1) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: pushed(ear) > 0, 10.0)
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.route.apply_config(stack.effective())
        assert wait_until(lambda: snap(ear)["revoke_resets"] == 1)
        assert wait_until(lambda: ear.rt.sink.stops >= 1)  # the answer stops playing
        assert ear.settle(lambda: turns(ear) == 1, 10.0)


def test_a_stray_response_with_streams_off_counts_as_unrequested(ears):  # noqa: F811
    with TurnServer(unrequested=True) as server:
        rt = {**SERVER, "audio_gate": "addressed"}  # turns: server, but no stream may go
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"])
        assert wait_until(lambda: snap(ear)["unrequested"] == 1)
        assert snap(ear)["server_duplicates"] == 0


def test_robot_turns_back_clear_the_owed_speech(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen, stack = ear.input, ear.rt.route.generation, ear.rt.stack
        stream = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
        inp._owed.append(_Owed(gen, stream, 1900, inp._clock()))
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.route.apply_config(stack.effective())
        assert inp._owed == [] and inp.claim(gen, 5.0) is None


# ------------------------------------------------------------------ re-review follow-ups (PR 196)


class Judged:
    """A route stand-in that records the input's verdicts on held responses."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def judged(self, job, verdict, items) -> None:
        self.calls.append(verdict)


def held_stream(ear, end_ms: int, sent_ms: float = 1500.0):
    """A stream that still streams, with a server response held for it (WAIT)."""
    inp, gen = ear.input, ear.rt.route.generation
    s = _Burst(stream_job(), gen, utt_ms=math.inf, silence_ms=600.0)
    s.sent_ms = sent_ms
    inp._streams = [s]
    inp._owed.append(_Owed(gen, s, end_ms, inp._clock()))
    assert inp.claim(gen, 5.0).verdict == "wait"
    return s


def test_a_held_stream_closed_by_a_failure_is_a_fragment(ears):  # noqa: F811
    """L1: the stream thread fails mid-segment (or the session goes): _close ends it part
    way, so the response held for it answers a fragment, never a whole segment."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        judged = Judged()
        ear.input.route = judged
        s = held_stream(ear, 1400)  # the server ended on quiet inside the 1.5 s streamed
        ear.input._close(s.job)  # the stream thread's exception handler: no cancel
        assert judged.calls == ["fragment"]


def test_a_stream_the_server_splits_is_answered_once(ears):  # noqa: F811
    """L2: the server ends a stream inside the segment (held), then again at its end: the
    earlier hold is released as a fragment, and the later response is the stream's answer."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        judged = Judged()
        inp.route = judged
        s = held_stream(ear, 800)
        inp._owed.append(_Owed(gen, s, 2900, inp._clock()))  # the server's second end
        assert inp.claim(gen, 5.0).verdict == "wait"  # the stream's answer, held again
        assert judged.calls == ["fragment"]  # the first one released
        s.utt_ms = 3000.0  # the close
        inp._judge(s)
        assert judged.calls == ["fragment", "answer"]


def test_a_stream_judged_a_fragment_can_still_be_answered_later(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        judged = Judged()
        inp.route = judged
        s = held_stream(ear, 800)
        s.utt_ms = 3000.0
        inp._judge(s)  # the close judged the hold: the server ended inside the segment
        assert judged.calls == ["fragment"] and not s.job.claimed
        inp._owed.append(_Owed(gen, s, 2900, inp._clock()))
        assert inp.claim(gen, 5.0).verdict == "answer"


def test_a_judgement_racing_the_holds_registration_is_not_lost(ears):  # noqa: F811
    """L3: the stream ends between the claim (WAIT) and the route registering its hold:
    the route judges it itself once registered, so the answer plays."""
    with TurnServer(stream_vad=True, vad_answers=True, heard=[[HEARD]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        inp, claim = ear.input, ear.input.claim

        def claim_then_end(gen, wait):
            s = next(b for b in inp._streams if b.generation == gen)
            utt, s.utt_ms = s.utt_ms, math.inf  # still streaming at the claim
            result = claim(gen, wait)
            s.utt_ms = utt
            inp._judge(s)  # the close lands before the route registers the hold
            return result

        inp.claim = claim_then_end
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1 and pushed(ear) > 0, 10.0)
        assert snap(ear)["server_responses"] == 1 and not ear.rt.route._held


def test_an_error_about_a_client_event_does_not_poison_the_owed_speech(ears):  # noqa: F811
    """L4a: an error that echoes an event the robot sent (a delete, a truncate) is about that
    event, not a declined answer."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        stream = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
        inp._owed.append(_Owed(gen, stream, 1900, inp._clock()))
        ear.rt.route._error(protocol.ServerError("error", "item_not_found", "ev_7"), None)
        assert inp.claim(gen, 5.0).verdict == "answer"


def test_a_declined_answer_does_not_hold_or_reset_a_skill_follow_up(ears):  # noqa: F811
    """L4b: the server declines an answer (no event id) while a server turn runs: that owed
    speech is resolved at once, so the turn's follow-up neither waits nor resets."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, route, gen = ear.input, ear.rt.route, ear.rt.route.generation
        stream = _Burst(stream_job(), gen, utt_ms=2000.0, silence_ms=600.0)
        stream.add("in_9")
        stream.stopped("in_9", 1900)
        stream.sent_ms = 2000.0
        inp._streams = [stream]
        inp._owed.append(_Owed(gen, stream, 1900, inp._clock()))
        assert route.busy()
        declined = protocol.ServerError("error", "conversation_already_has_active_response")
        route._error(declined, None)
        assert not route.busy()
        route.expire()
        assert snap(ear)["unsettled_resets"] == 0


def test_a_revoke_during_a_server_turns_follow_up_stops_it(ears):  # noqa: F811
    """L5: the server's answer called a skill that needs a follow-up; the revoke lands while
    the robot's follow-up request is out: the session is reset."""
    with TurnServer(
        ["audio_then_tool", "silent"], tool_name="recall", stream_vad=True, vad_answers=True
    ) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        tooled(ear)
        ear.say()
        assert wait_until(lambda: len(server.of("response.create")) == 1, 10.0)
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.route.apply_config(stack.effective())
        assert wait_until(lambda: snap(ear)["revoke_resets"] == 1)
