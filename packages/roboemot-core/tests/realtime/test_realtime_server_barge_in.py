"""Server turns, barge-in (S2; SPEC 8, Realtime speech route, Server turns; SPEC 12; SPEC 13):
the session's server VAD interrupts its own answer (``interrupt_response``) when it hears the
person; a barge-in's speech streams over the robot's voice (a playback stream, opened only at
the barge-in's cut-off point, which passed the echo guard) and the server's ``speech_started``
for it confirms the barge-in. The robot then stops, truncates what played and plays nothing
more of the cancelled answer, and it never sends a second ``response.cancel``. The consent and
the budget of every stream hold for a playback stream too."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from realtime_helpers import wait_until
from roboemot.realtime import protocol
from roboemot.realtime.route import Request
from test_realtime_listener import appended, ears  # noqa: F401 (fixture)
from test_realtime_server_responses import detect, pushed, snap, turns
from test_realtime_server_turns import OPT_INS, SERVER, tail
from test_voice_listener import SpyMusic, voice
from turn_server import TurnServer

# a reply that cannot end before the barge-in, also under load: 400 deltas of 100 ms, 50 ms
# apart (the server's interrupt ends it); the barge-in's own answer is short
LONG = {"stream_vad": True, "vad_answers": True, "deltas": [400, 20], "gap_s": 0.05}
HEARD = [["ciao Neve"], ["aspetta un attimo"]]


def speaking(ear) -> None:
    """The robot speaks its answer: the listener and the audio input see it."""
    ear.r.speech.extend(ear.r.now + 60.0)
    ear.input._speaking = lambda: True


def silent(ear) -> None:
    """The robot stopped (what the speaker's stop does to the speaking window)."""
    ear.r.speech.reset(ear.r.now)
    ear.input._speaking = lambda: False


def answer_playing(ear) -> None:
    """The first segment streams and the server's own answer starts to play."""
    ear.say()
    assert wait_until(lambda: pushed(ear) > 0, 10.0)
    speaking(ear)
    ear.r.feed(np.zeros(160, np.float32))  # the listener mutes


def barge_in(ear) -> threading.Event:
    """The barge-in's cut-off point (its echo guard passed): the listener takes the speech
    over the robot's voice; returns the event the server's confirmation sets."""
    confirmed = threading.Event()
    ear.input.on_confirm = confirmed.set
    assert ear.r.listener.stream_playback(np.zeros(0, np.float32), ear.r.t)
    return confirmed


def cut(ear) -> int:
    """What the barge-in does at the confirmation (``BargeIn._server_cut``); returns the
    frames the speaker had then."""
    worker = ear.rt.worker
    worker.mark_interrupting()
    ear.rt.speech.stop()
    at_stop = pushed(ear)
    worker.interrupt()
    silent(ear)
    return at_stop


# ------------------------------------------------------------------ the session


def test_server_turns_let_the_servers_vad_interrupt_its_own_answer(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        updates = server.of("session.update")
        assert updates and all(detect(u)["interrupt_response"] is True for u in updates)


# ------------------------------------------------------------------ confirm, stop, truncate


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_a_confirmed_barge_in_stops_truncates_and_never_cancels_twice(ears, interrupt_first):  # noqa: F811
    """The server's cancel of its own answer comes before or after its speech_started; either
    way the robot stops on the confirmation, truncates what played and sends no cancel. The
    barge-in's own answer then plays."""
    with TurnServer(**LONG, heard=HEARD, interrupt_first=interrupt_first) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        answer_playing(ear)
        confirmed = barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert confirmed.wait(10.0)
        assert server.interrupted == ["resp_1"]  # the server cancelled its answer itself
        cut(ear)
        ear.r.feed(voice(seconds=0.3))
        tail(ear)  # the person stops: the stream closes whole
        assert ear.settle(lambda: snap(ear)["server_responses"] == 2, 10.0)
        assert ear.settle(lambda: turns(ear) == 2, 10.0)
        assert server.of("response.cancel") == []  # never a second cancel
        assert server.of("response.create") == []
        (truncate,) = server.of("conversation.item.truncate")
        assert truncate["item_id"] == "item_1"
        assert 0 < truncate["audio_end_ms"] < 400 * 100  # at what played, not the whole reply
        s = snap(ear)
        assert s["playback_confirms"] == 1
        # live at the start only when the server's cancel came after it
        assert s["server_interrupts"] == (0 if interrupt_first else 1)
        assert s["server_duplicates"] == 0 and s["server_fragments"] == 0
        assert ear.r.listener.snapshot()["realtime_audio"]["streamed_during_playback"] == 1


def test_no_audio_of_the_interrupted_answer_plays_after_the_servers_start(ears):  # noqa: F811
    """Deltas the server still sends after its speech_started (its cancel landed late) never
    reach the speaker: the route drops them on the client thread, before the robot's stop."""
    with TurnServer(**LONG, heard=HEARD, late_deltas=3) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        answer_playing(ear)
        confirmed = barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert confirmed.wait(10.0)
        assert wait_until(lambda: snap(ear)["late_audio_dropped"] == 3)
        at_stop = cut(ear)
        assert not wait_until(lambda: pushed(ear) > at_stop, 0.5)  # nothing more played
        assert server.of("response.cancel") == []


def test_a_cancel_of_an_interrupted_answer_is_never_sent(ears):  # noqa: F811
    """Unit of the rule: an answer live when the server's VAD heard the person is cancelled by
    the server; the robot's abandon (its barge-in) sends no cancel. With robot turns it does."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        route = ear.rt.route
        req = Request("rq_1", None, 0.0, token=route._token, generation=route.generation)
        req.response_id = "resp_x"
        route._live["resp_x"] = req
        started = protocol.SpeechBoundary(protocol.SPEECH_STARTED, None, 0)
        route._on_event(started, route._token)
        route.abandon(req, 0)
        assert not wait_until(lambda: server.of("response.cancel"), 0.3)
        assert snap(ear)["cancels_skipped"] == 1
    with TurnServer() as server:
        ear = ears(server, script=["x"])
        route = ear.rt.route
        req = Request("rq_1", None, 0.0, token=route._token, generation=route.generation)
        req.response_id = "resp_x"
        route._live["resp_x"] = req
        route._on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, None, 0), route._token)
        route.abandon(req, 0)
        assert wait_until(lambda: server.of("response.cancel"))  # robot turns: ours


def test_a_speech_started_with_nothing_playing_interrupts_nothing(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        confirmed: list[bool] = []
        ear.input.on_confirm = lambda: confirmed.append(True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert server.interrupted == [] and server.of("response.cancel") == []
        assert confirmed == []  # not a barge-in's stream: nothing to confirm
        assert snap(ear)["server_interrupts"] == 0


def test_a_turn_a_barge_in_cut_never_hosts_the_barge_ins_own_answer(ears):  # noqa: F811
    """The barge-in's answer comes while the cut turn still ends: it is queued as the next
    turn, never dropped as a duplicate of the turn it interrupted. A turn that runs on (not
    cut) still hosts it, as in S1b."""
    from types import SimpleNamespace

    from roboemot.conversation.worker import _ServerTurn
    from roboemot.realtime.route import ServerResponse
    from test_realtime_server_responses import stream_job

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        worker, gen = ear.rt.worker, ear.rt.route.generation
        running = SimpleNamespace(interrupted=False, server=None)
        with worker._lock:
            worker._active_turn = running
        job = stream_job()
        assert not worker._server_response(ServerResponse(Request("", None, 0.0), gen, job, ()))
        running.interrupted = True  # the barge-in cut it; it is still ending
        job = stream_job()
        offer = ServerResponse(Request("", None, 0.0), gen, job, ())
        assert worker._server_response(offer)  # taken: its own turn, next
        host = worker._claims[job]
        assert isinstance(host, _ServerTurn) and host.offer is offer and job.answered
        with worker._lock:
            if worker._active_turn is running:
                worker._active_turn = None


# ------------------------------------------------------------------ the playback stream


def test_a_barge_ins_stream_goes_while_the_robot_speaks_and_only_one(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        tail(ear)  # its segment ends while the robot still speaks
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_stopped"))
        ear.r.feed(voice(seconds=0.5))  # more speech over the robot: dropped again
        tail(ear)
        started = server.sent_of("input_audio_buffer.speech_started")
        assert not wait_until(lambda: len(started) > 1, 0.5)  # no second turn reached it
        s = snap(ear)
        assert s["streams"] == 1 and s["stream_pieces_refused"] == 0 and s["streams_cut"] == 0
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["streamed_during_playback"] == 1 and audio["streamed"] == 1


def test_a_release_ends_the_barge_ins_stream_and_deletes_its_item(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.r.listener.end_playback()  # the barge-in's release: no confirmation in time
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        assert wait_until(lambda: server.of("conversation.item.delete"))
        sent = len(appended(server))
        ear.r.feed(voice(seconds=0.5))  # the robot still speaks: nothing more goes
        assert not wait_until(lambda: len(appended(server)) > sent, 0.3)


def test_a_robot_turns_session_never_streams_over_the_robot(ears):  # noqa: F811
    """Robot turns: the barge-in is local, as before; no playback stream is offered."""
    with TurnServer() as server:
        ear = ears(server, script=["x"])
        speaking(ear)
        assert not ear.r.listener.stream_playback(np.zeros(0, np.float32), ear.r.t)
        assert not ear.r.listener.server_turns()
        assert ear.r.listener.snapshot()["realtime_audio"]["playback_refused"] == 0


# ------------------------------------------------------------------ consent and budget


def refused(ear) -> None:
    """The barge-in's stream is refused: nothing leaves, the barge-in cuts locally."""
    before = len(appended(ear.rt.server))
    speaking(ear)
    assert not ear.r.listener.stream_playback(np.zeros(0, np.float32), ear.r.t)
    ear.r.feed(voice(seconds=0.5))
    assert not wait_until(lambda: len(appended(ear.rt.server)) > before, 0.3)


@pytest.mark.parametrize("opt_in", ["realtime_audio", "realtime_audio_unaddressed"])
def test_a_playback_stream_needs_both_opt_ins(ears, opt_in):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.rt.stack.set(f"privacy.{opt_in}", False, source="test", admin=True)
        ear.r.listener.apply_config(ear.rt.stack.effective())
        refused(ear)
        assert snap(ear)["streams"] == 0


def test_a_playback_stream_is_refused_while_asleep(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], asleep=lambda: True)
        refused(ear)
        assert ear.r.listener.snapshot()["realtime_audio"]["playback_refused"] == 1


def test_a_playback_stream_is_refused_with_the_budget_spent(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "daily_upload_budget_s": 1}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"])
        route = ear.rt.route
        with route._lock:
            route._uploaded()  # today's count
            route._upload_s = 1.0  # spent
        refused(ear)
        assert ear.r.listener.snapshot()["realtime_audio"]["playback_refused"] == 1


def test_a_playback_stream_is_refused_by_a_server_gate(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.listener.music = SpyMusic(playing=True)
        refused(ear)
        assert ear.r.listener.snapshot()["realtime_audio"]["playback_refused"] == 1


def test_a_gate_that_needs_the_voice_direction_refuses_a_playback_stream(ears):  # noqa: F811
    """The voice direction is muted while the robot speaks: such a gate cannot decide."""
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": ["at_face"]}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], bearing=0.0)
        refused(ear)


def test_a_playback_stream_is_refused_while_a_turn_holds_the_input(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.rt.route.hold_input()  # a robot turn's tool round is in flight
        try:
            refused(ear)
        finally:
            ear.rt.route.release_input()


def test_a_revoke_during_a_playback_stream_cuts_it(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio", False, source="test", admin=True)
        ear.rt.route.apply_config(stack.effective())
        before = snap(ear)["upload_s_today"]
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        s = snap(ear)
        assert s["streams_revoked"] + s["streams_cut"] == 1  # the reload, or a queued piece
        assert s["upload_s_today"] == before  # nothing more left, the closing silence neither
