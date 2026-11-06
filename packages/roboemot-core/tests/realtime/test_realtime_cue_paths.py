"""S3: the listening cue of a barge-in's stream (S2) never starts while the robot speaks
(SPEC 8, Non-verbal cues; Barge-in with server turns). It begins, at most, once the server
confirmed the stream and the robot's own speech stopped. A barge-in's stream the server never
confirmed shows no cue at all: neither the listening cue nor the did-not-get-it cue, and its
audio, heard over the robot's voice, is never decoded locally."""

from __future__ import annotations

import threading

from realtime_helpers import wait_until
from test_realtime_listener import ears  # noqa: F401 (fixture)
from test_realtime_server_barge_in import HEARD, barge_in, silent, speaking
from test_realtime_server_turns import OPT_INS, SERVER, tail
from test_voice_listener import silence, voice
from turn_server import TurnServer

SHORT_WAIT = {**SERVER, "transcript_timeout_s": 0.5}


def recorded(ear) -> list[tuple[str, float]]:
    """The listener's audio events (what the runtime turns into the listening cue)."""
    events: list[tuple[str, float]] = []
    ear.r.listener.bind_realtime(ear.input, lambda event, key: events.append((event, key)))
    return events


def after_the_echo_tail(ear):
    """Silence past the speaking window's echo tail (the robot counts as speaking within it)."""
    return silence(ear.rt.stack.effective().voice.echo_tail_ms / 1000 + 0.1)


def audio(ear) -> dict:
    return ear.r.listener.snapshot()["realtime_audio"]


def test_an_ordinary_stream_begins_the_listening_cue_at_its_end(ears):  # noqa: F811
    with TurnServer(stream_vad=True, heard=HEARD) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        events = recorded(ear)
        ear.say(0.5)
        assert ear.settle(lambda: [e for e, _ in events][:1] == ["sent"])


def test_a_barge_ins_stream_begins_the_cue_only_once_the_robot_stopped(ears):  # noqa: F811
    hold = threading.Event()
    with TurnServer(stream_vad=True, heard=HEARD, hold=hold, hold_at="transcript") as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"])
        events = recorded(ear)
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        assert wait_until(lambda: ear.r.listener._playback_job is not None)
        assert wait_until(lambda: ear.r.listener._playback_job.confirmed)
        tail(ear)  # its segment ends while the robot still speaks
        assert ear.settle(lambda: audio(ear)["streamed"] == 1)
        assert audio(ear)["sent"] == 1  # the audio left: counted at once
        assert not ear.settle(lambda: events, 0.3)  # but no cue over the robot's voice
        silent(ear)  # the robot stopped on the confirmation
        ear.r.feed(silence(0.05))
        assert not ear.settle(lambda: events, 0.2)  # within its echo tail: still the robot's
        ear.r.feed(after_the_echo_tail(ear))
        assert ear.settle(lambda: [e for e, _ in events] == ["sent"])
        hold.set()


def test_an_unconfirmed_barge_ins_stream_never_cues_and_is_never_decoded(ears):  # noqa: F811
    with TurnServer() as server:  # no stream VAD: the server never confirms
        ear = ears(server, rt=SHORT_WAIT, app=OPT_INS, script=["x"], decoded="ciao Neve")
        events = recorded(ear)
        speaking(ear)
        barge_in(ear)
        ear.r.feed(voice(seconds=0.5))
        tail(ear)  # its segment ends while the robot still speaks
        assert ear.settle(lambda: audio(ear)["streamed"] == 1)
        silent(ear)  # the robot's answer ended on its own: still no confirmation
        ear.r.feed(after_the_echo_tail(ear))
        assert ear.settle(lambda: audio(ear)["playback_unconfirmed"] == 1)
        assert events == []  # no listening cue, no did-not-get-it cue
        assert ear.r.asr.decodes == 0 and ear.utterances == []
        assert audio(ear)["local_fallbacks"] == 0
