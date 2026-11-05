"""Server turns, first part (SPEC 8, Realtime speech route; SPEC 12): with
``voice.realtime_route.turns: server`` a speech segment that passed the server gates streams
while it is heard, the server's VAD ends the turn and the robot sends no commit. The gates
decide only whether a segment streams, before any of its audio leaves; the upload budget and
the opt-in hold on every piece. The answer is still asked for by the robot (``response.create``)
once the transcript came, until the server-created answers land."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from realtime_helpers import wait_until
from roboemot.config.schema import RealtimeRouteConfig
from test_realtime_listener import FACE_RAD, OFF_FACE_RAD, appended, ears  # noqa: F401 (fixture)
from test_voice_listener import RATE, SpyMusic, silence, voice
from turn_server import TurnServer

SERVER = {"turns": "server", "audio_gate": "server_gates"}
OPT_INS = {"privacy": {"realtime_audio": True, "realtime_audio_unaddressed": True}}
HEARD = "ciao Neve come stai"


def tail(ear) -> None:
    """The silence after the speech: the local VAD ends the segment."""
    ear.r.feed(silence(ear.rt.stack.effective().voice.vad.end_silence_ms / 1000 + 0.2))


def appended_s(server) -> float:
    return len(server.audio()) / 2 / RATE


def streamed_s(cfg, speech_s: float = 1.0) -> float:
    """What one segment streams when the speech starts the audio (no silence before it for
    the pre-roll): the speech, the local end silence and the closing silence."""
    vad, rt = cfg.voice.vad, cfg.voice.realtime_route
    return speech_s + vad.end_silence_ms / 1000 + rt.close_silence_ms / 1000


# ------------------------------------------------------------------ streaming


def test_a_segment_streams_while_it_is_heard_and_the_robot_sends_no_commit(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=[[HEARD]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.r.feed(voice(seconds=0.5))
        # the speech is not over, and its audio is at the server already
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        assert ear.utterances == [] and not server.sent_of("input_audio_buffer.speech_stopped")
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.rt.events) == 1, 10.0)
        (u,) = ear.utterances
        assert u.addressed and u.text == HEARD
        assert u.realtime == (ear.rt.route.generation, ("in_1_0",))
        assert ear.r.asr.started == []  # Vosk skipped: the server transcribes
        assert server.of("input_audio_buffer.commit") == []  # the server's VAD ended the turn
        assert len(server.of("response.create")) == 1  # the robot still asks (until S1b)
        assert server.of("conversation.item.create") == []  # the words are the server's item
        kinds = server.kinds()
        first = kinds.index("input_audio_buffer.append")
        assert kinds[first - 1] == "session.update"  # fresh instructions before any audio
        snap = ear.rt.route.snapshot()
        assert snap["turns"] == "server" and snap["streams"] == 1
        assert ear.r.listener.snapshot()["realtime_audio"]["streamed"] == 1


def test_the_stream_ends_with_the_closing_silence(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        cfg = ear.rt.stack.effective()
        pcm = np.frombuffer(server.audio(), "<i2")
        closing = round(RATE * cfg.voice.realtime_route.close_silence_ms / 1000)
        assert pcm.size > closing and not pcm[-closing:].any()
        # the speech, the local end silence and the closing silence: nothing else
        assert appended_s(server) == pytest.approx(streamed_s(cfg), abs=0.15)
        assert ear.rt.route.snapshot()["upload_s_today"] == pytest.approx(appended_s(server))


def test_robot_turns_still_send_one_burst_with_its_commit(ears):  # noqa: F811
    """The default: nothing of the server-turn path runs."""
    with TurnServer() as server:
        ear = ears(server, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        kinds = [k for k in server.kinds() if k.startswith(("input_audio", "session"))]
        appends = [i for i, k in enumerate(kinds) if k == "input_audio_buffer.append"]
        assert kinds[appends[0] - 1] == "session.update"
        assert kinds[appends[-1] + 1] == "input_audio_buffer.commit"
        snap = ear.rt.route.snapshot()
        assert snap["turns"] == "robot" and snap["streams"] == 0 and snap["audio_sent"] == 1
        assert ear.r.listener.snapshot()["realtime_audio"]["streamed"] == 0


def test_text_input_never_has_server_turns():
    from roboemot.config import ConfigStack
    from roboemot.realtime.route import server_turns

    stack = ConfigStack(app={"voice": {"realtime_route": {"turns": "server"}}})
    assert not server_turns(stack.effective())


# ------------------------------------------------------------------ gates


def test_a_gate_that_is_on_keeps_the_segment_local_before_any_append(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["come stai"])
        ear.r.listener.music = SpyMusic(playing=True)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == [] and server.of("session.update")[1:] == []
        assert not ear.utterances[0].addressed  # policy never, music: the wake word needed
        assert ear.r.asr.started  # Vosk heard it from its start
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["stream_gated_music"] == 1 and audio["streamed"] == 0


def test_a_gate_that_is_off_lets_the_segment_stream(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt={**SERVER, "server_gates": []}, app=OPT_INS, script=["x"])
        ear.r.listener.music = SpyMusic(playing=True)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert appended(server) and ear.utterances[0].text == "ciao Neve come stai"
        assert ear.r.listener.snapshot()["realtime_audio"]["stream_gated_music"] == 0


def test_the_voice_at_face_gate_is_off_by_default(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], bearing=OFF_FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert appended(server) and ear.utterances[0].addressed
        assert sum(ear.r.listener.snapshot()["wake"]["decisions"].values()) == 0  # no check


def test_the_voice_at_face_gate_holds_then_keeps_an_off_face_voice_local(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": ["at_face"]}
        ear = ears(server, rt=rt, app=OPT_INS, script=["come stai"], bearing=OFF_FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert not ear.utterances[0].addressed and ear.utterances[0].text == "come stai"
        assert ear.r.listener.snapshot()["realtime_audio"]["stream_gated_at_face"] == 1


def test_the_voice_at_face_gate_streams_a_voice_at_a_face_with_its_held_audio(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": ["at_face"]}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        cfg = ear.rt.stack.effective()
        # the held audio went too: nothing of the speech was lost to the hold
        assert appended_s(server) == pytest.approx(streamed_s(cfg), abs=0.15)


def test_speech_during_a_realtime_turn_does_not_stream(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["come stai"])
        ear.rt.route.hold_input()  # a turn's tool round is in flight
        try:
            ear.say()
            assert ear.settle(lambda: len(ear.utterances) == 1)
        finally:
            ear.rt.route.release_input()
        assert appended(server) == [] and ear.r.asr.started
        assert ear.r.listener.snapshot()["realtime_audio"]["streams_refused"] == 1


# ------------------------------------------------------------------ cut part way


def test_a_stream_cut_when_the_robot_speaks_sends_the_closing_silence_and_deletes_its_item(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.input._speaking = lambda: True  # the robot starts speaking: the next piece stops
        ear.r.feed(voice(seconds=0.3))
        assert wait_until(lambda: ear.rt.route.snapshot()["streams_cut"] == 1)
        # the server heard the closing silence, ended the turn, and its item is deleted
        assert wait_until(lambda: server.of("conversation.item.delete"))
        assert server.of("conversation.item.delete")[0]["item_id"] == "in_1_0"
        assert server.of("input_audio_buffer.commit") == []


def test_an_opt_in_revoked_mid_stream_stops_the_next_piece(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.rt.stack.set("privacy.realtime_audio", False, source="test", admin=True)
        # the reload bumps the route's revision: a piece that read the old config re-reads it,
        # so none passes from here on
        ear.rt.route.apply_config(ear.rt.stack.effective())
        before = ear.rt.route.snapshot()["upload_s_today"]
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1)
        snap = ear.rt.route.snapshot()
        assert snap["streams_revoked"] + snap["streams_cut"] == 1  # the reload, or a queued piece
        # nothing more left, the closing silence neither
        assert snap["upload_s_today"] == before


# ------------------------------------------------------------------ second review round


def test_a_refused_opening_is_decided_on_its_own_voice_direction(ears):  # noqa: F811
    """Default gates (no hold gate): the opening of an off-face segment is refused, and its
    local decode is slow; the next segment, at a face, starts before that decode ends. The
    wake gate must judge the first one on its own samples: it stays unaddressed."""
    with TurnServer(stream_vad=True, heard=[["seconda"]]) as server:
        ear = ears(
            server, rt=SERVER, app=OPT_INS, script=["x"], decoded="la tv parla",
            bearing=OFF_FACE_RAD, background=True, decode_s=0.5,
        )  # fmt: skip
        gate, sweep = threading.Event(), ear.input._sweep
        ear.input._sweep = lambda cfg: (gate.wait(5.0), sweep(cfg))[1]  # the opening waits
        ear.r.feed(voice(seconds=0.5))
        job = ear.r.listener._stream.job
        ear.rt.route.hold_input()  # a turn begins: the opening is refused (turn)
        gate.set()
        assert wait_until(lambda: job.done and job.refused == "turn")
        ear.rt.route.release_input()
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        ear.bearing = FACE_RAD  # the next speaker is at a face
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        first, second = ear.utterances
        assert first.text == "la tv parla" and not first.addressed
        assert second.addressed and second.text == "seconda"


def test_a_revoke_right_after_the_opening_went_resets_the_session(ears):  # noqa: F811
    """The opening's send and its return are apart (the upload count's save): a revoke in
    between still knows the stream holds audio."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        route, stack = ear.rt.route, ear.rt.stack
        save, fired = route._save_upload, []

        def revoke_then_save(*args):
            if not fired:
                fired.append(True)
                stack.set("privacy.realtime_audio_unaddressed", False, source="test")
                route.apply_config(stack.effective())
            save(*args)

        route._save_upload = revoke_then_save
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: route.snapshot()["streams_revoked"] == 1)  # counted last
        assert route.snapshot()["input_resets"] == 1


def test_a_cut_on_a_session_without_item_ids_resets_it(ears):  # noqa: F811
    """Without ids the cut turn's transcript cannot be deleted: the next stream must not land
    on a session that may still answer it."""
    with TurnServer(stream_vad=True, ids=False, heard=[["dopo"]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.input._speaking = lambda: True  # the robot starts speaking: the stream is cut
        ear.r.feed(voice(seconds=0.3))
        assert wait_until(lambda: ear.rt.route.snapshot()["streams_cut"] == 1)
        assert wait_until(lambda: ear.rt.route.snapshot()["input_resets"] == 1)
        ear.input._speaking = lambda: False
        tail(ear)
        assert wait_until(lambda: ear.rt.route.up(), 10.0)
        ear.say()
        assert ear.settle(lambda: any(u.text == "dopo" for u in ear.utterances), 10.0)
        assert len(server.connections) == 2


# ------------------------------------------------------------------ upload budget


def test_no_stream_opens_without_budget_for_its_closing_silence(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(
            server, rt={**SERVER, "daily_upload_budget_s": 0.5}, app=OPT_INS, script=["come stai"]
        )
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["streams_refused"] == 1


def test_the_budget_cuts_a_stream_and_its_closing_silence_still_fits(ears):  # noqa: F811
    budget = 1.5
    with TurnServer(stream_vad=True) as server:
        ear = ears(
            server, rt={**SERVER, "daily_upload_budget_s": budget}, app=OPT_INS, script=["x"]
        )
        ear.say()
        assert wait_until(lambda: ear.rt.route.snapshot()["streams_cut"] == 1)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        snap = ear.rt.route.snapshot()
        assert snap["stream_pieces_refused"] >= 1 and snap["audio_blocked"] == "upload_budget"
        # its audio left without a transcript: not answered (the not_heard cue), as a timeout
        assert ear.utterances[0].local_only
        assert appended_s(server) <= budget
        assert snap["upload_s_today"] == pytest.approx(appended_s(server))
        cfg = ear.rt.stack.effective()
        closing = round(RATE * cfg.voice.realtime_route.close_silence_ms / 1000)
        assert not np.frombuffer(server.audio(), "<i2")[-closing:].any()


# ------------------------------------------------------------------ config


def test_the_server_gates_default_and_check():
    rt = RealtimeRouteConfig()
    assert rt.turns == "robot" and rt.server_gates == ["music"] and rt.audio_gate == "addressed"
    with pytest.raises(ValueError):
        RealtimeRouteConfig(server_gates=["asleep"])  # asleep is no gate: it always holds
    with pytest.raises(ValueError, match="repeats"):
        RealtimeRouteConfig(server_gates=["music", "music"])
    with pytest.raises(ValueError):
        RealtimeRouteConfig(server_gates=["wake_word"])
    with pytest.raises(ValueError):
        RealtimeRouteConfig(server_gate_hold_ms=5000)


# ------------------------------------------------------------------ several segments, the skip


def test_two_segments_in_a_row_each_get_their_own_transcript(ears):  # noqa: F811
    """The second segment streams while the first one's transcript is still on its way."""
    heard = [["prima"], ["seconda"]]
    hold = threading.Event()
    with TurnServer(stream_vad=True, heard=heard, hold=hold, hold_at="transcript") as srv:
        ear = ears(srv, rt=SERVER, app=OPT_INS, script=["x"])
        ear.say()
        ear.say()
        assert wait_until(lambda: len(srv.sent_of("input_audio_buffer.committed")) == 2)
        hold.set()  # both streams wait for their transcripts now
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        assert [u.text for u in ear.utterances] == ["prima", "seconda"]
        assert [u.realtime[1] for u in ear.utterances] == [("in_1_0",), ("in_2_0",)]
        assert srv.of("conversation.item.delete") == []
        assert ear.rt.route.snapshot()["streams"] == 2


def test_a_skip_mutes_and_deletes_the_streamed_words(ears):  # noqa: F811
    from roboemot.realtime.turn import SKIP_TOOL

    with TurnServer(["tool"], tool_name=SKIP_TOOL, stream_vad=True) as srv:
        rt = {**SERVER, "skip_tool": True}
        ear = ears(srv, rt=rt, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: len(ear.rt.events) == 1, 10.0)
        assert ear.rt.events[0]["outcome"] == "skipped"
        deleted = {e["item_id"] for e in srv.of("conversation.item.delete")}
        assert "in_1_0" in deleted  # the user's item goes with the skip
        assert ear.rt.history() == []


def test_a_second_segment_reaches_the_server_before_the_first_turns_request(ears):  # noqa: F811
    """Known S1a behaviour, replaced by S1b's server-created answers: a person who pauses and
    goes on has the second segment committed before the first turn's response.create, so that
    response hears both, and the second turn asks once more."""
    heard = [["prima"], ["seconda"]]
    hold = threading.Event()
    with TurnServer(stream_vad=True, heard=heard, hold=hold, hold_at="transcript") as srv:
        ear = ears(srv, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        ear.say()
        assert wait_until(lambda: len(srv.sent_of("input_audio_buffer.committed")) == 2)
        hold.set()
        assert ear.settle(lambda: len(ear.rt.events) == 2, 15.0)
        committed = srv.sent_of("input_audio_buffer.committed")
        assert [e["item_id"] for e in committed] == ["in_1_0", "in_2_0"]
        assert srv.commits_at_create == [2, 2]  # both held when the first answer was asked


# ------------------------------------------------------------------ consent (SPEC 12)


def test_server_turns_without_the_unaddressed_opt_in_run_as_robot_turns(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, rt={"turns": "server"}, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert server.of("input_audio_buffer.commit")  # the burst, after the wake gate
        snap = ear.rt.route.snapshot()
        assert snap["turns"] == "robot" and snap["turns_fallback"] == "no_unaddressed_opt_in"
        assert snap["streams"] == 0
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["turns_fallbacks"] == 1 and audio["streamed"] == 0


def test_the_opt_in_turned_off_at_runtime_brings_robot_turns_back_at_once(ears):  # noqa: F811
    """Turning the consent off needs no admin and no restart: the next utterance is a burst
    whose server events reach it (not the streams of before)."""
    with TurnServer(stream_vad=True, heard=[["ciao"]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert ear.rt.route.snapshot()["streams"] == 1
        ear.rt.stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.stack.set("voice.realtime_route.audio_gate", "addressed", source="t", admin=True)
        ear.r.listener.apply_config(ear.rt.stack.effective())
        server.stream_vad = False  # the fake answers bursts on their commit only
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        u = ear.utterances[1]
        assert u.text == "ciao" and u.realtime is not None and not u.local_only
        assert len(server.of("input_audio_buffer.commit")) == 1
        snap = ear.rt.route.snapshot()
        assert snap["turns"] == "robot" and snap["audio_sent"] == 1 and snap["streams"] == 1


def test_a_stream_needs_the_server_gates_audio_gate(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt={"turns": "server"}, app=OPT_INS, script=["come stai"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == [] and server.of("input_audio_buffer.commit") == []
        assert ear.r.listener.snapshot()["realtime_audio"]["streams_refused"] == 1
        pcm = np.zeros(160, "<i2").tobytes()
        assert ear.rt.route.stream_audio(pcm, instructions="x") == "gate"  # the send gate too


def test_nothing_streams_while_asleep_whatever_the_gates(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": []}
        ear = ears(server, rt=rt, app=OPT_INS, script=["come stai"], asleep=lambda: True)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["streams_refused"] == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("privacy.realtime_audio_unaddressed", True),
        ("voice.realtime_route.turns", "server"),
        ("voice.realtime_route.server_gates", []),
        ("voice.realtime_route.audio_gate", "server_gates"),
    ],
)
def test_a_runtime_change_that_widens_the_stream_needs_the_admin(key, value):
    from roboemot.config import ConfigStack
    from roboemot.config.stack import GuardrailError

    stack = ConfigStack(app={})
    with pytest.raises(GuardrailError):
        stack.set(key, value, source="test")
    stack.set(key, value, source="test", admin=True)  # the admin's confirmation


# ------------------------------------------------------------------ a stream and a burst


def test_a_streams_item_never_joins_a_burst_of_the_same_session(ears):  # noqa: F811
    """A runtime switch back to robot turns can leave a stream's transcript on its way while a
    burst of the same session waits: the stream keeps its words, the burst gets none."""
    from roboemot.realtime import protocol
    from roboemot.realtime.input import InputJob, _Burst

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen = ear.input, ear.rt.route.generation
        caps = {"silence_ms": 600.0, "max_items": 8, "max_chars": 2000}
        stream = _Burst(InputJob(np.zeros(0), RATE), gen, utt_ms=1000.0, **caps)
        stream.add("s1")
        stream.stopped("s1", 1000)
        burst = _Burst(InputJob(np.zeros(0), RATE), gen, start_ms=0.0, utt_ms=1000.0, **caps)
        with inp._lock:
            inp._streams.append(stream)
            inp._burst = burst
        done = protocol.InputTranscript(protocol.TRANSCRIPTION_COMPLETED, "s1", "la tv", True)
        inp.on_event(done, gen)
        assert stream.job.result is not None and stream.job.result.text == "la tv"
        assert burst.texts == {} and not burst.job.done


def test_a_revoked_opt_in_cancels_the_open_stream_and_the_next_burst_is_clean(ears):  # noqa: F811
    """The revoked stream's audio never reaches a commit: the session is reset, so the next
    burst (robot turns again) holds its own words only, and they are not lost."""
    with TurnServer(stream_vad=True, heard=[["la seconda"]]) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        before = ear.rt.route.snapshot()["upload_s_today"]  # no piece passes the gate now
        ear.rt.route.apply_config(stack.effective())  # the runtime's hot reload
        snap = ear.rt.route.snapshot()
        # cancelled by the reload, or cut by a queued piece the gate refused: once, either way
        assert wait_until(lambda: ear.rt.route.snapshot()["input_resets"] == 1)
        snap = ear.rt.route.snapshot()
        assert snap["streams_revoked"] + snap["streams_cut"] == 1
        ear.r.listener.apply_config(stack.effective())
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert ear.utterances[0].local_only  # its audio left: not answered
        assert ear.rt.route.snapshot()["upload_s_today"] == before
        assert wait_until(lambda: ear.rt.route.up(), 10.0)  # the reset's new session
        server.stream_vad = False  # robot turns: the fake answers the burst's commit
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        u = ear.utterances[1]
        assert u.text == "la seconda" and u.realtime is not None and not u.local_only
        assert len(server.connections) == 2
        commit = server.connections[1]
        assert commit is not None and len(server.of("input_audio_buffer.commit")) == 1


# ------------------------------------------------------------------ a refused opening (fix round)


def held_onset(ear) -> None:
    """Speech past the VAD's start, before ``server_gate_hold_ms``: the gates have not decided
    and nothing is queued for the stream yet."""
    ear.r.feed(voice(seconds=0.25))
    assert ear.r.listener._stream is not None and ear.r.listener._stream.job is None


HOLD = {**SERVER, "server_gates": ["at_face"], "server_gate_hold_ms": 400}


def test_an_opening_refused_for_a_turn_is_decided_by_the_wake_gate(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(
            server, policy="always", rt=HOLD, app=OPT_INS, script=["x"], decoded="come stai",
            bearing=FACE_RAD,
        )  # fmt: skip
        held_onset(ear)
        ear.rt.route.hold_input()  # a realtime turn begins before the opening goes
        try:
            ear.r.feed(voice(seconds=0.75))
            tail(ear)
            assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        finally:
            ear.rt.route.release_input()
        u = ear.utterances[0]
        assert not u.addressed and u.text == "come stai" and not u.local_only  # no wake word
        assert appended(server) == []
        snap = ear.r.listener.snapshot()
        assert sum(snap["wake"]["decisions"].values()) == 1  # the wake gate decided it
        assert snap["realtime_audio"]["streamed"] == 0 and snap["realtime_audio"]["sent"] == 0
        assert ear.rt.route.snapshot()["streams_refused"] == 1


def test_an_opening_refused_without_a_session_is_decided_by_the_wake_gate(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(
            server, policy="always", rt=HOLD, app=OPT_INS, script=["x"], decoded="come stai",
            bearing=FACE_RAD,
        )  # fmt: skip
        held_onset(ear)
        ear.rt.route.close_soon("conversation_end")  # the session goes before the opening
        ear.r.feed(voice(seconds=0.75))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert not ear.utterances[0].addressed
        assert appended(server) == []
        assert sum(ear.r.listener.snapshot()["wake"]["decisions"].values()) == 1


def test_an_opening_after_this_inputs_reset_waits_for_the_new_session(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=[["dopo il reset"]]) as server:
        ear = ears(server, rt=HOLD, app=OPT_INS, script=["x"], bearing=FACE_RAD)
        held_onset(ear)
        ear.input._stream_reopening = True  # as after a sweep's reset
        route, reopened = ear.rt.route, threading.Event()
        reopen = route._reopen

        def slow_reopen(closes):  # the reconnect is still in flight when the opening comes
            reopened.wait(5.0)
            reopen(closes)

        route._reopen = slow_reopen
        assert route.reset_session(route.generation)
        ear.r.feed(voice(seconds=0.75))
        tail(ear)
        assert wait_until(lambda: route.snapshot()["audio_reopen_wait"] == 1)  # it waits
        reopened.set()  # the reconnect lands
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        u = ear.utterances[0]
        assert u.addressed and u.text == "dopo il reset" and u.realtime is not None
        assert len(server.connections) == 2


def test_a_revoke_while_the_opening_is_queued_sends_nothing(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, policy="always", rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.listener._cfg = ear.rt.stack.effective()
        gate, sweep = threading.Event(), ear.input._sweep
        ear.input._sweep = lambda cfg: (gate.wait(5.0), sweep(cfg))[1]  # the opening waits
        ear.r.feed(voice(seconds=0.5))
        assert ear.r.listener._stream is not None and ear.r.listener._stream.job is not None
        stack = ear.rt.stack
        stack.set("privacy.realtime_audio_unaddressed", False, source="test")
        ear.rt.route.apply_config(stack.effective())
        gate.set()
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert appended(server) == []
        assert not ear.utterances[0].addressed  # the wake gate decided: no wake word
        assert sum(ear.r.listener.snapshot()["wake"]["decisions"].values()) == 1


# ------------------------------------------------------------------ sessions and fallbacks


def test_a_reconnect_between_pieces_cuts_the_stream(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        assert ear.rt.route.reset_session(ear.rt.route.generation)
        assert wait_until(lambda: ear.rt.route.up(), 10.0)
        before = len(server.of("input_audio_buffer.append"))
        ear.r.feed(voice(seconds=0.5))
        tail(ear)
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        assert ear.rt.route.snapshot()["streams_cut"] == 1
        assert len(server.of("input_audio_buffer.append")) == before  # none on the new session


def test_the_fallback_with_the_stream_gate_set_is_robot_turns(ears):  # noqa: F811
    """turns: server and audio_gate: server_gates without the unaddressed opt-in: bursts."""
    with TurnServer() as server:
        ear = ears(server, rt=SERVER, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        u = ear.utterances[0]
        assert u.realtime is not None and u.text == "ciao Neve come stai"
        assert len(server.of("input_audio_buffer.commit")) == 1
        assert ear.rt.route.snapshot()["turns_fallback"] == "no_unaddressed_opt_in"


# ------------------------------------------------------------------ gates fail closed


def test_the_music_gate_with_the_detector_off_keeps_the_segment_local(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        app = {**OPT_INS, "music": {"enabled": False}}
        ear = ears(server, rt=SERVER, app=app, script=["come stai"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["stream_gated_music_off"] == 1


def test_the_facing_gate_with_the_facing_check_off_keeps_the_segment_local(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": ["facing"]}
        ear = ears(server, rt=rt, app=OPT_INS, script=["come stai"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["stream_gated_facing_off"] == 1


def test_the_facing_gate_keeps_a_face_looking_away_local(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        rt = {**SERVER, "server_gates": ["facing"]}
        app = {**OPT_INS, "voice": {"wake_word_policy": "never", "facing": {"enabled": True}}}
        ear = ears(server, rt=rt, app=app, script=["x"], bearing=FACE_RAD, yaw=0.7)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        snap = ear.r.listener.snapshot()
        assert snap["realtime_audio"]["stream_gated_facing"] == 1
        assert sum(snap["direction"]["checks"].values()) == 1  # the wake gate reused the verdict


def test_the_other_speaker_gate_keeps_another_known_person_local(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=[["prima"]]) as server:
        rt = {**SERVER, "server_gates": ["other_speaker"]}
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"])
        ear.r.listener._speaker_id = lambda: "anna"
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        ear.r.listener._speaker_id = lambda: "bruno"
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 2, 10.0)
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["streamed"] == 1 and audio["stream_gated_other_speaker"] == 1


# ------------------------------------------------------------------ short segments, cuts


def test_a_segment_shorter_than_the_hold_is_decided_at_its_end(ears):  # noqa: F811
    rt = {**SERVER, "server_gates": ["at_face"], "server_gate_hold_ms": 2000}
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=rt, app=OPT_INS, script=["x"], bearing=FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1, 10.0)
        cfg = ear.rt.stack.effective()
        assert appended_s(server) == pytest.approx(streamed_s(cfg), abs=0.15)  # decided late
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=rt, app=OPT_INS, script=["come stai"], bearing=OFF_FACE_RAD)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert appended(server) == []
        assert ear.r.listener.snapshot()["realtime_audio"]["stream_gated_at_face"] == 1


def test_a_chunk_gap_cuts_the_stream_in_progress(ears):  # noqa: F811
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.r.feed(voice(seconds=0.1), gap=True)  # lost audio: the segment is dropped
        assert wait_until(lambda: ear.rt.route.snapshot()["streams_cut"] == 1)
        assert wait_until(lambda: server.of("conversation.item.delete"))  # its words go
        assert server.of("conversation.item.delete")[0]["item_id"] == "in_1_0"


def test_an_old_stream_never_covered_resets_the_session_at_the_next_opening(ears):  # noqa: F811
    from roboemot.realtime.input import InputJob, _Burst

    with TurnServer() as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, gen, cfg = ear.input, ear.rt.route.generation, ear.rt.stack.effective()
        caps = {"silence_ms": 600.0, "max_items": 8, "max_chars": 2000}
        late = cfg.voice.realtime_route.transcript_timeout_s + 1.0
        heard = _Burst(InputJob(np.zeros(0), RATE), gen, utt_ms=900.0, sent_ms=900.0, **caps)
        heard.add("s1")
        heard.open.add("s1")  # started, never stopped: not covered
        silent = _Burst(InputJob(np.zeros(0), RATE), gen, utt_ms=900.0, sent_ms=900.0, **caps)
        for stream in (heard, silent):
            stream.closed, stream.ended_at = True, inp._clock() - late
        inp._streams += [silent]
        inp._sweep(cfg)
        assert inp._streams == [] and ear.rt.route.snapshot()["input_resets"] == 0  # nothing heard
        inp._streams += [heard]
        inp._sweep(cfg)
        assert inp._streams == [] and ear.rt.route.snapshot()["input_resets"] == 1


# ------------------------------------------------------------------ a stream ends once


def revoke(ear) -> None:
    stack = ear.rt.stack
    stack.set("privacy.realtime_audio_unaddressed", False, source="test")
    ear.rt.route.apply_config(stack.effective())


def test_a_revoke_inside_a_cut_ends_the_stream_once_as_revoked(ears):  # noqa: F811
    """Forced: the robot starts speaking, a piece is refused and the cut sends its closing
    silence; the revoke lands right then, before the cut retires the stream."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        route, send, fired = ear.rt.route, ear.rt.route.stream_audio, []

        def revoke_in_the_cut(pcm, **kw):
            if kw.get("closing") and not fired:
                fired.append(True)
                revoke(ear)  # the reload ends the stream first
            return send(pcm, **kw)

        route.stream_audio = revoke_in_the_cut
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        ear.input._speaking = lambda: True  # the next piece is refused: a cut
        ear.r.feed(voice(seconds=0.3))
        assert wait_until(lambda: fired and ear.input._ops.empty())
        assert wait_until(lambda: route.snapshot()["streams_revoked"] == 1)
        snap = route.snapshot()
        assert (snap["streams_revoked"], snap["streams_cut"]) == (1, 0)


def test_a_cut_before_the_revoke_ends_the_stream_once_as_cut(ears):  # noqa: F811
    """Forced the other way: the cut retires the stream, then the revoke finds it over."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, close, fired = ear.input, ear.input._close, []

        def close_then_revoke(job):
            retired = close(job)
            if job.stream and not fired:  # the cut retired it: the revoke lands before its count
                fired.append(True)
                revoke(ear)
            return retired

        inp._close = close_then_revoke
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        inp._speaking = lambda: True
        ear.r.feed(voice(seconds=0.3))
        assert wait_until(lambda: route_snapshot(ear)["streams_cut"] == 1)
        assert wait_until(lambda: bool(fired))
        snap = route_snapshot(ear)
        assert (snap["streams_revoked"], snap["streams_cut"]) == (0, 1)


def route_snapshot(ear) -> dict:
    return ear.rt.route.snapshot()


def test_a_cut_learnt_idless_only_afterwards_still_resets_the_session(ears):  # noqa: F811
    """Forced: the cut runs before the robot has heard the server's id-less speech_started;
    that event, once it comes, still resets the session."""
    with TurnServer(stream_vad=True, ids=False) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        inp, deliver, held = ear.input, ear.input.on_event, []
        inp.on_event = lambda event, gen=None: held.append((event, gen))  # not heard yet
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: server.sent_of("input_audio_buffer.speech_started"))
        inp._speaking = lambda: True
        ear.r.feed(voice(seconds=0.3))
        assert wait_until(lambda: route_snapshot(ear)["streams_cut"] == 1)
        assert route_snapshot(ear)["input_resets"] == 0  # nothing said it was id-less yet
        inp.on_event = deliver
        for event, gen in held:
            deliver(event, gen)
        assert wait_until(lambda: route_snapshot(ear)["input_resets"] == 1)
