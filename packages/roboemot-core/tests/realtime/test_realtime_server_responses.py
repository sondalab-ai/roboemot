"""Server turns, second part (S1b; SPEC 8, Realtime speech route, Server turns; SPEC 12; SPEC
13): with ``turns: server`` the session's server VAD answers on its own (``create_response``),
and the response it created for a stream that ended whole is the turn's answer: it plays once,
the robot sends no ``response.create`` for it, and the stream's transcript fills the history
whenever it comes. A response for a fragment (a stream cut, revoked, from an old session, or
ended by the server inside the segment) is cancelled and never reaches the speaker; a second
response for an answered stream, or one while a turn's response runs, is a duplicate."""

from __future__ import annotations

import threading

import numpy as np
from realtime_helpers import wait_until
from roboemot.realtime.route import session_update
from test_realtime_follow_up import LOOK, SKILLS, Env
from test_realtime_listener import ears  # noqa: F401 (fixture)
from test_realtime_server_turns import HEARD, OPT_INS, SERVER, tail
from test_voice_listener import voice
from turn_server import REPLY, TurnServer


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
        assert latency["turn"]["count"] == 1 and latency["first_audio"]["count"] == 1
        # from the segment's last speech as the listener heard it (the Ear's own clock here)
        job = next(iter(ear.rt.worker._claims))
        assert job.start_stamp < job.speech_end <= ear.r.now
        assert set(latency["turn"]) == set(latency["first_audio"]) == {"count", "p50", "p95", "max"}
    with TurnServer() as server:  # robot turns, text in
        ear2 = ears(server, script=["x"])
        ear2.rt.turn("ciao")
        latency = snap(ear2)["latency_ms"]
        assert latency["turn"]["count"] == 1
        assert latency["turn"]["p50"] == latency["first_audio"]["p50"]  # one measure


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
        assert server.of("conversation.item.delete") == []


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
        never_played(ear, server)


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
    from roboemot.realtime.input import InputJob, _Burst, _Owed

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        job = InputJob(np.zeros(0), 16000)
        job.stream = True
        stream = _Burst(job, gen, utt_ms=2000.0, silence_ms=600.0)
        inp._owed.append(_Owed(gen, stream, 800, inp._clock()))  # ended at 0.8 s of 2 s
        assert inp.claim(gen, 5.0).verdict == "fragment"
        inp._owed.append(_Owed(gen, stream, 1500, inp._clock()))  # within the end silence
        assert inp.claim(gen, 5.0).verdict == "answer"
        inp._owed.append(_Owed(gen, stream, None, inp._clock()))  # no end position: never
        assert inp.claim(gen, 5.0).verdict == "fragment"


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
