"""The robot's ``thinking`` clip fills a slow cloud (SPEC 8, Realtime speech route, Lost reply
retry): a transcript not in ``retry.filler_after_s`` after an addressed utterance's audio left,
or a turn's first reply audio not in that long after its request, plays it once (one status
clip per wait: never twice for one utterance, never after the ``retrying`` clip). A transcript
that comes after the listener stopped waiting is counted (SPEC 12A,
``transcripts_after_timeout``)."""

from __future__ import annotations

import dataclasses
import threading

import pytest
from roboemot.config.schema import RealtimeRouteConfig
from rt_rig import wait_for
from test_conversation_worker import utt
from test_realtime_listener import FACE_RAD, ears  # noqa: F401 (fixture)
from test_realtime_retry import FakeClips, rigs  # noqa: F401 (fixture)
from turn_server import TurnServer

FILLER_S = 0.5  # the shortest filler delay: the reply tests wait in wall time
SLOW_S = 1.2  # a first reply delta this late is past it
FAST_REPLY = {"retry": {"filler_after_s": FILLER_S}}


class Slow:
    """The listener's ``on_slow_transcript``: counts its calls, says the clip played."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return True


def held_ear(ears, server, slow):  # noqa: F811
    ear = ears(server, script=["x"], decoded="che tempo fa", bearing=FACE_RAD)
    ear.r.listener.on_slow_transcript = slow
    ear.say()
    assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
    return ear


def fillers(ear) -> dict:
    return ear.r.listener.snapshot()["fillers"]


def test_a_transcript_slower_than_the_filler_delay_plays_thinking_once(ears):  # noqa: F811
    hold = threading.Event()
    slow = Slow()
    with TurnServer(hold=hold, hold_at="transcript") as server:
        ear = held_ear(ears, server, slow)
        assert ear.settle(lambda: ear.r.listener._queue[0].rt.sent)
        ear.r.listener.poll_handoff()
        assert slow.calls == 0  # not yet: its clock has not moved
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.retry.filler_after_s
        assert ear.settle(lambda: slow.calls == 1)
        ear.r.now += 1.0
        ear.r.listener.poll_handoff()
        assert slow.calls == 1  # once per utterance
        hold.set()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].filled
        assert fillers(ear) == {"transcript": 1}


def test_a_fast_transcript_plays_no_filler(ears):  # noqa: F811
    slow = Slow()
    with TurnServer() as server:
        ear = held_ear(ears, server, slow)
        assert ear.settle(lambda: len(ear.utterances) == 1)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.retry.filler_after_s
        ear.r.listener.poll_handoff()
        assert slow.calls == 0 and not ear.utterances[0].filled
        assert fillers(ear) == {"transcript": 0}


def test_a_filler_that_did_not_play_leaves_the_utterance_unfilled(ears):  # noqa: F811
    hold = threading.Event()
    with TurnServer(hold=hold, hold_at="transcript") as server:
        ear = held_ear(ears, server, lambda: False)  # the clip is missing
        assert ear.settle(lambda: ear.r.listener._queue[0].rt.sent)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.retry.filler_after_s
        ear.r.listener.poll_handoff()
        hold.set()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].filled
        assert fillers(ear) == {"transcript": 0}


def test_a_transcript_after_the_timeout_is_counted(ears):  # noqa: F811
    hold = threading.Event()
    with TurnServer(hold=hold, hold_at="transcript") as server:
        ear = held_ear(ears, server, Slow())
        assert ear.settle(lambda: ear.r.listener._queue[0].rt.sent)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].local_only
        assert ear.rt.route.snapshot()["transcripts_after_timeout"] == 0
        hold.set()  # the server's words come after the robot stopped waiting
        assert wait_for(lambda: ear.rt.route.snapshot()["transcripts_after_timeout"] == 1)
        assert ear.rt.route.snapshot()["transcript_timeouts"] == 1


def reply_fillers(rig) -> dict:
    return rig.worker.snapshot()["fillers"]


def test_a_slow_first_reply_plays_thinking_once_and_the_reply_follows_it(rigs):  # noqa: F811
    with TurnServer(first_delta_s=SLOW_S) as srv:
        rig = rigs(srv, rt=FAST_REPLY)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)  # the reply played
        assert rig.clips.played == {"thinking": 1}
        assert reply_fillers(rig) == {"reply": 1}
        assert rig.events[-1]["outcome"] == "ok"


def test_a_fast_first_reply_plays_no_filler(rigs):  # noqa: F811
    with TurnServer() as srv:
        rig = rigs(srv, rt=FAST_REPLY)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.clips.played == {}
        assert reply_fillers(rig) == {"reply": 0}


def test_no_second_filler_when_the_transcript_already_had_one(rigs):  # noqa: F811
    with TurnServer(first_delta_s=SLOW_S) as srv:
        rig = rigs(srv, rt=FAST_REPLY)
        n = rig.worker.snapshot().get("turns", 0) + 1
        u = dataclasses.replace(utt("ciao", stamp=1000.0), filled=True)
        assert rig.worker.submit(u)
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n, 10.0)
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.clips.played == {}
        assert reply_fillers(rig) == {"reply": 0}


def test_no_thinking_filler_after_the_retrying_clip(rigs):  # noqa: F811
    with TurnServer(["close_before", "reply"], first_delta_s=SLOW_S) as srv:
        rig = rigs(srv, rt=FAST_REPLY)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.clips.played == {"retrying": 1}  # one status clip per wait
        assert reply_fillers(rig) == {"reply": 0}


def test_a_missing_retrying_clip_leaves_the_thinking_filler(rigs):  # noqa: F811
    with TurnServer(["close_before", "reply"], first_delta_s=SLOW_S) as srv:
        rig = rigs(srv, FakeClips(missing=("retrying",)), rt=FAST_REPLY)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
        assert rig.clips.played == {"thinking": 1}
        assert reply_fillers(rig) == {"reply": 1}


def test_the_filler_must_come_before_the_transcript_timeout():
    with pytest.raises(ValueError, match="filler_after_s"):
        RealtimeRouteConfig(transcript_timeout_s=2.0, retry={"filler_after_s": 2.5})
    # without the retry no clip is recorded or played: the pair is not checked
    RealtimeRouteConfig(transcript_timeout_s=2.0, retry={"enabled": False, "filler_after_s": 2.5})
