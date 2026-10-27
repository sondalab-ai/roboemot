"""The realtime audio input against the fake server (SPEC 8, Realtime speech route; SPEC 12):
one burst per utterance, every transcript it produces, joined in order, the upload budget,
and the items of an utterance the robot does not answer through the route deleted."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.realtime.input import RealtimeInput
from rt_rig import build_rt, wait_for
from turn_server import RATE, TurnServer

AUDIO = {"input": "audio"}
OPT_IN = {"privacy": {"realtime_audio": True}}


def speech(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


@pytest.fixture
def rig_for():
    made = []

    def make(server, rt=None, app=None, speaking=lambda: False):
        rig = build_rt(server, rt={**AUDIO, **(rt or {})}, app={**OPT_IN, **(app or {})})
        rig.input = RealtimeInput(rig.route, speaking=speaking)
        made.append(rig)
        assert rig.route.ready()
        return rig

    yield make
    for rig in made:
        rig.input.close()
        rig.close()


def test_one_burst_one_transcript(rig_for):
    with TurnServer(heard=[["ciao Neve"]]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is not None and job.result.text == "ciao Neve"
        assert job.result.items == ("in_1_0",)
        assert job.result.generation == rig.route.generation
        kinds = server.kinds()
        burst = kinds[kinds.index("input_audio_buffer.append") - 1 :]
        assert burst[0] == "session.update" and burst[-1] == "input_audio_buffer.commit"
        silence = rig.stack.effective().voice.realtime_route.close_silence_ms
        pcm = server.audio()
        assert len(pcm) == 2 * (RATE + round(RATE * silence / 1000))
        assert pcm[-2 * round(RATE * silence / 1000) :] == bytes(2 * round(RATE * silence / 1000))
        assert "response.create" not in kinds
        snap = rig.route.snapshot()
        assert snap["audio_sent"] == 1 and snap["transcripts"] == 1
        assert snap["upload_s_today"] == pytest.approx(1.0 + silence / 1000, abs=1e-3)


def test_a_split_burst_gives_every_transcript_joined_in_order(rig_for):
    with TurnServer(heard=[["ciao Neve", "come stai"]], transcript_delay_s=0.2) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: job.done)
        assert job.result.text == "ciao Neve come stai"
        assert job.result.items == ("in_1_0", "in_1_1")


def test_no_transcript_leaves_the_job_open_and_a_cancel_sends_nothing_more(rig_for):
    with TurnServer(heard=[[]]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        assert not job.done
        rig.input.cancel(job)
        assert job.done and job.result is None
        assert server.of("conversation.item.delete") == []


def test_a_transcript_after_the_cancel_has_its_item_deleted(rig_for):
    with TurnServer(heard=[["tardi"]], transcript_delay_s=0.5) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.input.cancel(job)
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 1)
        assert server.of("conversation.item.delete")[0]["item_id"] == "in_1_0"
        assert rig.route.snapshot()["audio_items_deleted"] == 1


def test_nothing_is_sent_while_the_robot_speaks(rig_for):
    with TurnServer() as server:
        rig = rig_for(server, speaking=lambda: True)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None and job.refused == "speaking"
        assert server.of("input_audio_buffer.append") == []


def test_nothing_is_sent_without_the_opt_in(rig_for):
    with TurnServer() as server:
        rig = rig_for(server)
        rig.stack.set("privacy.realtime_audio", False, source="test", admin=True)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None and job.refused is not None
        assert server.of("input_audio_buffer.append") == []


def test_the_upload_budget_is_enforced(rig_for):
    rt = {"daily_upload_budget_s": 1.9}
    with TurnServer(heard=[["uno"]]) as server:
        rig = rig_for(server, rt=rt)
        first = rig.input.submit(speech(1.0), RATE)  # 1.9 s with the closing silence: all of it
        assert wait_for(lambda: first.done) and first.result is not None
        second = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is None and second.refused == "upload_budget"
        assert len(server.of("input_audio_buffer.commit")) == 1
        assert not rig.input.wants(rig.stack.effective())


def test_a_lost_session_ends_the_job(rig_for):
    with TurnServer(heard=[[]]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.route.close_soon("conversation_end")
        assert wait_for(lambda: rig.input.settled(job))
        assert job.result is None


def test_wants_follows_the_config_and_the_session(rig_for):
    with TurnServer() as server:
        rig = rig_for(server)
        assert rig.input.wants(rig.stack.effective())
        rig.stack.set("voice.realtime_route.input", "text", source="test", admin=True)
        assert not rig.input.wants(rig.stack.effective())
        rig.stack.set("voice.realtime_route.input", "audio", source="test", admin=True)
        rig.route.close_soon("conversation_end")
        assert not rig.input.wants(rig.stack.effective())


def test_the_job_never_shows_the_transcript(rig_for):
    with TurnServer(heard=[["segreto"]]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert "segreto" not in repr(job) and "segreto" not in repr(job.result)
        assert "in_1_0" not in repr(job.result)
