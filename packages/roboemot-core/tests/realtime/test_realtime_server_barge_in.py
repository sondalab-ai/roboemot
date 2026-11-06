"""Server turns, barge-in (S2; SPEC 8, Realtime speech route, Server turns; SPEC 12; SPEC 13):
a barge-in's speech streams over the robot's voice (a playback stream, opened only at the
barge-in's cut-off point, which passed the echo guard) and the server's ``speech_started`` for
it confirms the barge-in. Only that stream's opening lets the server interrupt its own answer
(``interrupt_response``); every other stream keeps it off, so S1b's turns are unchanged. On the
confirmation the robot stops, truncates what played and plays nothing more of the interrupted
answer; it still cancels it (a fail-safe for a server that ignores the flag), with an event id
of its own, so the server's "nothing active" error is benign. The consent and the budget of
every stream hold for a playback stream too."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from realtime_helpers import wait_until
from roboemot.realtime.route import Request
from roboemot.voice.bargein import BargeIn
from roboemot.voice.vad import EnergyVAD
from test_bargein import user_voice
from test_realtime_listener import FACE_RAD, appended, ears  # noqa: F401 (fixture)
from test_realtime_server_responses import detect, pushed, snap, stream_job, turns
from test_realtime_server_turns import OPT_INS, SERVER, tail
from test_voice_listener import FRAME, RATE, SpyMusic, voice
from turn_server import TurnServer

# a reply that cannot end before the barge-in, also under load: 400 deltas of 300 ms, 50 ms
# apart (the server's interrupt ends it). It arrives six times faster than it plays, so when
# the robot stops on the confirmation (seconds after the reply began, under CI load too) it is
# still playing what it holds, and its truncation is at a point inside it; the barge-in's own
# answer is short
LONG = {
    "stream_vad": True, "vad_answers": True, "deltas": [400, 20], "gap_s": 0.05, "delta_ms": 300.0,
}  # fmt: skip
HEARD = [["ciao Neve"], ["aspetta un attimo"]]
NO_KEYWORD = {
    **OPT_INS,
    "voice": {"wake_word_policy": "never", "barge_in": {"keyword_enabled": False}},
}


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
    ear.input.on_confirm = lambda job: confirmed.set()
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


def settled_appends(server) -> int:
    """The server's appends once what the robot already sent has arrived (the socket lags
    the robot's sends): the count stops changing."""
    count = -1
    while count != len(appended(server)):
        count = len(appended(server))
        time.sleep(0.2)
    return count


def openings(server) -> list[bool]:
    """``interrupt_response`` of each session.update, in order."""
    return [detect(u)["interrupt_response"] for u in server.of("session.update")]


def wired(ear) -> BargeIn:
    """The real barge-in on the ear's listener, input and worker, as the runtime binds it
    (``bind_barge_in``), with no echo reference (the fake microphone hears no robot)."""
    cfg = ear.rt.stack.effective()
    bi = BargeIn(
        cfg, EnergyVAD(RATE, FRAME), ear.r.asr, None, language=lambda: "it",
        clock=lambda: ear.r.now,
    )  # fmt: skip
    listener, worker = ear.r.listener, ear.rt.worker
    listener._on_speaking_chunk, listener._on_speaking_end = bi.on_chunk, bi.reset
    listener._before_asr_reset = bi.handback_asr

    def stop() -> None:
        ear.rt.speech.stop()
        silent(ear)

    bi.bind(
        stop, worker.interrupt, listener.resume, worker.mark_interrupting,
        playback=listener.stream_playback, end_playback=listener.end_playback,
        server_mode=listener.server_turns, playback_job=listener.playback_job,
    )  # fmt: skip
    ear.input.on_confirm = bi.confirm
    return bi


def talk_until(ear, cond, seconds: float = 3.0) -> bool:
    """The person talks in short chunks, near real time, until ``cond`` (the server's
    confirmation reaches the listener thread between chunks)."""
    piece = user_voice(0.05, 0.21)
    for _ in range(round(seconds / 0.05)):
        ear.r.feed(piece)
        if cond():
            return True
        time.sleep(0.03)
    return cond()


# ------------------------------------------------------------------ the session


def test_only_a_barge_ins_stream_lets_the_server_interrupt_its_answer(ears):  # noqa: F811
    """Finding 1: interrupt_response is per stream opening. The connect and an ordinary
    stream keep it off (S1b's turns: a pause never cuts the answer), a barge-in's stream turns
    it on, and the next ordinary stream turns it off again."""
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert openings(server) and not any(openings(server))
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: openings(server)[-1] is True)
        silent(ear)
        tail(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 3, 10.0)
        assert openings(server)[-1] is False


def test_an_ordinary_streams_start_never_interrupts_the_answer_playing(ears):  # noqa: F811
    """Findings 1 and 6: a person who pauses and goes on (a second stream while the first
    answer plays, or a late start of a stream cut when the robot began to speak) leaves the
    answer alone: the server keeps it, nothing is marked, nothing is dropped."""
    long = {**LONG, "deltas": [30, 5]}
    with TurnServer(**long, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert wait_until(lambda: pushed(ear) > 0, 10.0)
        ear.r.feed(voice(seconds=0.5))  # the input does not see the robot speak: it streams
        assert wait_until(lambda: len(server.sent_of("input_audio_buffer.speech_started")) == 2)
        tail(ear)
        assert ear.settle(lambda: turns(ear) >= 1, 10.0)
        assert server.interrupted == [] and snap(ear)["server_interrupts"] == 0
        first = [
            e["response"]["status"] for e in server.sent
            if e.get("type") == "response.done" and e["response"]["id"] == "resp_1"
        ]  # fmt: skip
        assert first == ["completed"]  # the answer played on, whole
        assert snap(ear)["late_audio_dropped"] == 0


# ------------------------------------------------------------------ confirm, stop, truncate


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_a_confirmed_barge_in_stops_truncates_and_its_own_answer_plays(ears, interrupt_first):  # noqa: F811
    """The server's cancel of its own answer comes before or after its speech_started; either
    way the robot stops on the confirmation and truncates what played. Its own cancel carries
    its event id: the server's "nothing active" error about it is benign (Findings 4, 8), so
    the barge-in's own answer still plays and no did-not-get-it cue shows."""
    server_kw = {**LONG, "heard": HEARD, "interrupt_first": interrupt_first, "strict_cancel": True}
    with TurnServer(**server_kw) as server:
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
        assert server.of("response.create") == []
        assert wait_until(lambda: server.of("conversation.item.truncate"))  # off the socket
        (truncate,) = server.of("conversation.item.truncate")
        assert truncate["item_id"] == "item_1"
        assert 0 < truncate["audio_end_ms"] < 400 * 300  # at what played, not the whole reply
        for cancel in server.of("response.cancel"):  # the fail-safe: named, never bare
            assert cancel["event_id"].startswith("rc_")
        s = snap(ear)
        assert s["playback_confirms"] == 1 and s["errors_unrelated"] == 0
        assert s["cancels_not_active"] == server.cancel_errors
        assert s["server_interrupts"] == (0 if interrupt_first else 1)
        assert s["server_duplicates"] == 0 and s["server_fragments"] == 0
        assert ear.rt.worker.snapshot().get("server_unanswered", 0) == 0
        assert ear.r.listener.snapshot()["realtime_audio"]["streamed_during_playback"] == 1


@pytest.mark.parametrize("ids", [True, False], ids=["named", "unnamed"])
def test_an_error_about_the_robots_own_cancel_never_resolves_an_owed_answer(ears, ids):  # noqa: F811
    """Finding 4/8: the robot's cancel lands after the server's own interrupt; the server
    answers ``response_cancel_not_active``, naming the cancel or not. Benign either way: the
    barge-in's stream keeps its owed answer."""
    with TurnServer(**LONG, heard=HEARD, strict_cancel=True, cancel_error_ids=ids) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        answer_playing(ear)
        confirmed = barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert confirmed.wait(10.0)
        assert wait_until(lambda: any(e.get("type") == "response.done" for e in server.sent))
        ear.rt.route.cancel_response()  # a keyword or local cut racing the server's interrupt
        assert wait_until(lambda: server.cancel_errors >= 1)
        cut(ear)
        tail(ear)
        assert ear.settle(lambda: snap(ear)["server_responses"] == 2, 10.0)
        assert snap(ear)["cancels_not_active"] >= 1 and snap(ear)["errors_unrelated"] == 0


def test_a_server_that_ignores_interrupt_response_is_stopped_by_the_robots_cancel(ears):  # noqa: F811
    """Finding 4: the server's session.updated says interrupt_response false (or nothing): the
    robot marks nothing, and its own cancel stops the answer; the barge-in's answer plays and
    no did-not-get-it cue shows."""
    with TurnServer(**LONG, heard=HEARD, ignore_interrupt=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        answer_playing(ear)
        confirmed = barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert confirmed.wait(10.0)
        cut(ear)
        assert wait_until(lambda: server.of("response.cancel"))
        assert wait_until(
            lambda: any(
                e.get("type") == "response.done" and e["response"]["status"] == "cancelled"
                for e in server.sent
            )
        )  # the robot's cancel stopped it, not the server's VAD
        assert server.interrupted == [] and snap(ear)["server_interrupts"] == 0
        tail(ear)
        assert ear.settle(lambda: snap(ear)["server_responses"] == 2, 10.0)
        assert ear.rt.worker.snapshot().get("server_unanswered", 0) == 0
        assert "not_heard" not in ear.rt.cues.played


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


def test_a_round_the_server_interrupted_asks_nothing_more(ears):  # noqa: F811
    """M10: a round whose call needs a follow-up, interrupted by the server on a barge-in's
    stream before the robot's own stop: no follow-up request goes (it would answer over the
    person)."""
    from test_realtime_server_responses import tooled

    with TurnServer(
        ["tool_then_audio", "reply"], tool_name="recall", heard=HEARD, **LONG
    ) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        tooled(ear)
        answer_playing(ear)
        barge_in(ear)  # no confirmation is acted on: only the server's interrupt
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.interrupted == ["resp_1"], 10.0)
        assert wait_until(lambda: snap(ear)["server_interrupts"] == 1)
        silent(ear)
        tail(ear)  # the stream closes: nothing makes a follow-up wait any more
        assert not wait_until(lambda: server.of("response.create"), 5.0)


def test_a_turn_a_barge_in_cut_never_hosts_its_streams_own_answer(ears):  # noqa: F811
    """Finding 9: the barge-in's stream's answer comes while the cut turn still ends: it is
    queued as the next turn. Any other answer then is a duplicate, as in S1b; so is the
    barge-in's while a turn that was not cut runs."""
    from types import SimpleNamespace

    from roboemot.conversation.worker import _ServerTurn
    from roboemot.realtime.route import ServerResponse

    def offer(playback: bool) -> ServerResponse:
        job = stream_job()
        job.playback = playback
        return ServerResponse(Request("", None, 0.0), ear.rt.route.generation, job, ())

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        worker = ear.rt.worker
        running = SimpleNamespace(interrupted=False, server=None)
        with worker._lock:
            worker._active_turn = running
        assert not worker._server_response(offer(True))  # not cut: it hosts it
        running.interrupted = True  # the barge-in cut it; it is still ending
        assert not worker._server_response(offer(False))  # another stream's: a duplicate
        mine = offer(True)
        assert worker._server_response(mine)  # the barge-in's own: taken, its own turn next
        host = worker._claims[mine.job]
        assert isinstance(host, _ServerTurn) and host.offer is mine and mine.job.answered
        with worker._lock:
            if worker._active_turn is running:
                worker._active_turn = None


def test_a_barge_in_after_the_last_response_ended_still_truncates_in_robot_turns(ears):  # noqa: F811
    """M5: robot turns; the reply's response ended while its audio still plays and the turn
    has not finished: a barge-in then still truncates the audio item at what played."""
    with TurnServer(deltas=20) as server:  # 2 s of audio, stopped tens of ms in
        ear = ears(server, script=["x"], to_worker=True)
        worker = ear.rt.worker
        lines = worker._speak_lines
        playing: list[bool] = []

        async def barge_then_lines(turn):  # after the rounds, before _finish
            progress = [ear.rt.speech.stream_progress(j) for j, _ in turn.rt_jobs]
            playing.append(bool(progress) and all(p is not None and p[0] < p[1] for p in progress))
            worker.mark_interrupting()
            ear.rt.speech.stop()
            worker.interrupt()
            await lines(turn)

        worker._speak_lines = barge_then_lines
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert playing == [True]  # its item was known to the turn and still playing at the stop
        # the truncate went with the turn's end (the route's count is synchronous with the send);
        # the server reads it from its socket a moment later: wait for that event, not a time
        assert snap(ear)["items_truncated"] == 1
        assert wait_until(lambda: server.of("conversation.item.truncate"))
        (truncate,) = server.of("conversation.item.truncate")
        assert truncate["item_id"] == "item_1" and truncate["audio_end_ms"] < 20 * 100


# ------------------------------------------------------------------ the real barge-in


def test_the_real_barge_in_ducks_hands_off_and_stops_on_the_servers_confirmation(ears):  # noqa: F811
    """End to end: BargeIn → listener (stream over the robot) → input → on_confirm → the
    barge-in's stop at the next chunk; the barge-in's own answer then plays."""
    with TurnServer(**LONG, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=NO_KEYWORD, script=["x"], to_worker=True)
        bi = wired(ear)
        answer_playing(ear)
        assert talk_until(ear, lambda: bi.snapshot()["barge_ins"]["server_confirmed"] == 1)
        b = bi.snapshot()
        assert b["barge_ins"]["playback_streams"] == 1 and b["duck"]["cut"] == 1
        assert b["latency"]["onset_to_stop_server"]["n"] == 1 and b["triggers"] == 0
        assert wait_until(lambda: server.of("conversation.item.truncate"))
        tail(ear)
        assert ear.settle(lambda: snap(ear)["server_responses"] == 2, 10.0)


def test_the_stop_keyword_during_the_wait_cuts_the_stream_and_its_words_go_unanswered(ears):  # noqa: F811
    """Finding 3: the stop keyword while the server has not confirmed: the robot stops, the
    barge-in's stream is cut (a fragment: its items deleted, its answer never plays)."""
    with TurnServer(stream_vad=True, vad_answers=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=NO_KEYWORD, script=["x"], to_worker=True)
        bi = wired(ear)
        ear.input.on_confirm = lambda job: None  # the server's confirmation never reaches it
        speaking(ear)
        assert talk_until(ear, lambda: bi._await is not None)
        ear.r.feed(user_voice(0.5, 0.21))  # the listener's segment starts: its stream opens
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        bi._trigger("keyword", ear.r.now, ear.r.now, np.empty(0, np.float32), path="keyword")
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        silent(ear)
        tail(ear)
        assert not wait_until(lambda: snap(ear)["server_responses"] > 0, 1.0)
        assert turns(ear) == 0


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
        sent = settled_appends(server)
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
        # M1: a face at the voice; the last segment's samples (before the robot spoke) would
        # pass the check, but they are not this speech's: the gate refuses the stream
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert appended(server)  # it streamed: the check passed on those samples
        refused(ear)
        assert ear.r.listener.snapshot()["realtime_audio"]["playback_refused"] == 1


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


def test_a_revoke_and_restore_while_the_opening_waits_sends_nothing_of_it(ears):  # noqa: F811
    """M14: the opening of a barge-in's stream is past its epoch check when a reload forbids
    streams and another allows them again: the piece's own hold refuses it (its epoch)."""
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        gate, inside, sweep = threading.Event(), threading.Event(), ear.input._sweep

        def held_sweep(cfg):  # the opening passed its epoch check and waits here
            inside.set()
            gate.wait(5.0)
            sweep(cfg)

        ear.input._sweep = held_sweep
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert inside.wait(5.0)
        stack, route = ear.rt.stack, ear.rt.route
        opened = route.snapshot()["opened"]
        # a gate change forbids streams without closing the session (an opt-in would close
        # it, which refuses the opening anyway): only the epoch can stop the piece
        key = "voice.realtime_route.audio_gate"
        stack.set(key, "addressed", source="test", admin=True)
        route.apply_config(stack.effective())
        stack.set(key, "server_gates", source="test", admin=True)
        route.apply_config(stack.effective())
        assert route.connected and route.snapshot()["opened"] == opened
        gate.set()
        assert not wait_until(lambda: appended(server), 0.5)


def test_the_listener_ends_an_unconfirmed_playback_stream_on_its_own(ears):  # noqa: F811
    """Finding 2: whatever the barge-in does (off by a reload, its frames failing), the
    listener ends the barge-in's stream ``duck.confirm_ms`` after its start when the server
    has not confirmed it."""
    with TurnServer() as server:  # no stream VAD: the server never confirms
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        speaking(ear)
        barge_in(ear)
        confirm_s = ear.rt.stack.effective().voice.barge_in.duck.confirm_ms / 1000
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: appended(server))
        ear.r.feed(voice(seconds=confirm_s))
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        sent = settled_appends(server)
        ear.r.feed(voice(seconds=0.5))
        assert not wait_until(lambda: len(appended(server)) > sent, 0.3)
        assert not ear.r.listener._playback


def test_a_playback_stream_uses_the_upload_budget_and_is_cut_when_it_runs_out(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        rt = {**SERVER, "daily_upload_budget_s": 3}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"])
        route = ear.rt.route
        with route._lock:
            route._uploaded()
            route._upload_s = 1.5  # 1.5 s left: the opening fits with its closing silence
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=1.0))
        assert wait_until(lambda: snap(ear)["streams_cut"] == 1)
        s = snap(ear)
        assert s["stream_pieces_refused"] >= 1
        assert 1.5 < s["upload_s_today"] <= 3.0  # it counted, and never beyond the budget


# ------------------------------------------------------------------ approval follow-ups


def test_the_listener_never_ends_the_stream_before_the_barge_ins_own_release(ears):  # noqa: F811
    """Follow-up 1: both deadlines count ``confirm_ms`` from the same point, at the earliest
    the barge-in's cut-off: a confirmation the barge-in still takes as on time always finds
    the listener's stream there (else the person's words would be lost: no resume)."""
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=NO_KEYWORD, script=["x"])
        bi = wired(ear)
        ear.input.on_confirm = lambda job: None
        speaking(ear)
        assert talk_until(ear, lambda: bi._await is not None)
        assert ear.r.listener._playback_until >= bi._await.detected + bi._confirm_s


def test_a_barge_ins_segment_whose_opening_is_refused_later_is_never_decoded(ears):  # noqa: F811
    """Follow-up 2 (N10, privacy): the opening is refused on the stream thread after the
    hand-off (a turn took the input meanwhile): the segment heard over the robot's voice is
    dropped, never decoded locally nor given to the wake gate."""
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], decoded="ciao Neve")
        gate, inside, sweep = threading.Event(), threading.Event(), ear.input._sweep

        def held_sweep(cfg):
            inside.set()
            gate.wait(5.0)
            sweep(cfg)

        ear.input._sweep = held_sweep
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert inside.wait(5.0)
        ear.rt.route.hold_input()  # the opening is refused: turn
        try:
            gate.set()
            assert wait_until(lambda: ear.r.listener._playback_job.refused == "turn")
            tail(ear)
            assert ear.settle(
                lambda: ear.r.listener.snapshot()["realtime_audio"]["playback_unsent"] == 1
            )
        finally:
            ear.rt.route.release_input()
        assert ear.r.asr.decodes == 0 and ear.utterances == [] and appended(server) == []


def test_a_round_cancelled_without_the_servers_interrupt_still_asks_its_follow_up(ears):  # noqa: F811
    """Follow-up 3 (N13): the quiet end is for a response the server interrupted on a
    barge-in only; one that ends cancelled otherwise keeps its follow-up."""
    from test_realtime_server_responses import tooled

    with TurnServer(
        ["tool", "reply"], tool_name="recall", stream_vad=True, vad_answers=True,
        statuses=["cancelled", "completed"],
    ) as server:  # fmt: skip
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        tooled(ear)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert len(server.of("response.create")) == 1  # the follow-up after the output
        assert snap(ear)["server_interrupts"] == 0


def test_any_error_about_the_robots_own_cancel_is_benign_whatever_its_code(ears):  # noqa: F811
    """Follow-up 4 (N4): an error naming one of the robot's own cancels (its rc_ id) is
    about that cancel, whatever code the server gives it: benign, never unrelated."""
    from roboemot.realtime import protocol

    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        route = ear.rt.route
        assert route.cancel_response()
        event_id = route._cancel_ids[-1]
        assert event_id.startswith("rc_")
        error = protocol.ServerError(protocol.ERROR, "invalid_request_error", event_id)
        route._on_event(error, route._token)
        assert snap(ear)["cancels_not_active"] == 1 and snap(ear)["errors_unrelated"] == 0


def test_a_reply_whose_audio_streams_longer_than_the_stall_timeout_is_no_stall(ears):  # noqa: F811
    """CI root cause of 174cb45: the run's watchdog counted only queued events, never the audio
    deltas, so a reply that streamed audio for longer than ``stall_timeout_s`` (the rig's 3 s)
    with no other event was given up as a stall and cancelled by the robot itself mid-sentence.
    Audio arriving is the answer going on."""
    with TurnServer(deltas=80, gap_s=0.05) as server:  # 4 s of deltas, nothing else between
        ear = ears(server, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 15.0)
        assert ear.rt.events[0]["outcome"] == "ok"
        assert server.of("response.cancel") == []
        assert snap(ear)["cut_mid_turn"] == 0
