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

    def make(server, rt=None, app=None, speaking=lambda: False, asleep=lambda: False):
        rig = build_rt(server, rt={**AUDIO, **(rt or {})}, app={**OPT_IN, **(app or {})})
        rig.input = RealtimeInput(rig.route, speaking=speaking, asleep=asleep)
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


def test_startup_binds_the_input_to_the_listener_and_closes_it_before_the_route():
    from contextlib import ExitStack
    from types import SimpleNamespace

    from roboemot.runtime.startup import bind_realtime_input
    from roboemot.voice.activity import SpeechActivity

    bound = []
    route = SimpleNamespace(input=None)
    host = SimpleNamespace(speech=SpeechActivity(0.0))
    with ExitStack() as cleanup:
        bind_realtime_input(host, route, SimpleNamespace(bind_realtime=bound.append), cleanup)
        (audio_in,) = bound
        assert route.input is audio_in
        assert not audio_in._speaking()
        host.speech.extend(10**9)
        assert audio_in._speaking()  # nothing goes while the robot speaks
    job = audio_in.submit(speech(0.1), RATE)
    assert job.done and job.refused == "closed"


# ------------------------------------------------------------------ fix round 1


def test_a_job_cancelled_while_it_is_prepared_is_never_sent(rig_for):
    """I-1: the cancel lands while the input thread builds the instructions."""
    with TurnServer() as server:
        rig = rig_for(server)
        jobs = []
        base = rig.route.base_instructions

        def slow():
            rig.input.cancel(jobs[0])  # the listener's deadline passes right now
            return base()

        rig.route.base_instructions = slow
        jobs.append(rig.input.submit(speech(0.5), RATE))
        assert wait_for(lambda: rig.route.snapshot()["audio_refused"] == 1)
        assert rig.route.snapshot()["audio_blocked"] == "cancelled"
        assert server.of("input_audio_buffer.append") == []


def test_a_job_past_its_listener_deadline_is_never_sent(rig_for):
    with TurnServer() as server:
        rig = rig_for(server)
        now = [100.0]
        job = rig.input.submit(speech(0.5), RATE, deadline=99.0, clock=lambda: now[0])
        assert wait_for(lambda: job.done)
        assert job.refused == "cancelled" and not job.sent
        assert server.of("input_audio_buffer.append") == []


def test_nothing_is_sent_while_the_robot_sleeps(rig_for):
    """B-1: asleep is a send condition, whatever the pause does."""
    with TurnServer() as server:
        rig = rig_for(server, asleep=lambda: True)
        assert not rig.input.wants(rig.stack.effective())
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.refused == "asleep"
        assert server.of("input_audio_buffer.append") == []


def test_close_ends_every_queued_job_and_sends_nothing_more(rig_for):
    """M-2: a realtime turn holds the first burst; close() ends it and the one behind it."""
    with TurnServer() as server:
        rig = rig_for(server)
        rig.route.hold_input()
        first = rig.input.submit(speech(0.5), RATE)
        second = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: rig.input.held >= 1)
        rig.input.close()
        assert first.done and second.done
        rig.route.release_input()
        assert server.of("input_audio_buffer.append") == []
        assert not rig.input._thread_alive()


def test_a_turn_in_flight_holds_the_burst_until_it_ends(rig_for):
    """M-7: another utterance's words never land inside a turn's tool round."""
    with TurnServer(heard=[["dopo"]]) as server:
        rig = rig_for(server)
        rig.route.hold_input()
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: rig.input.held >= 1)
        assert server.of("input_audio_buffer.append") == []
        rig.route.release_input()
        assert wait_for(lambda: job.done)
        assert job.result is not None and job.result.text == "dopo"


def test_a_server_without_item_ids_still_joins_and_counts_what_it_cannot_delete(rig_for):
    """I-2: fail safe: the text is kept, the missing ids are counted."""
    with TurnServer(heard=[["ciao Neve", "come stai"]], ids=False) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: job.done)
        assert job.result.text == "ciao Neve come stai"
        assert job.result.items == () and job.result.unnamed == 2
        snap = rig.route.snapshot()
        assert snap["audio_items_unnamed"] == 2


def test_an_item_without_id_that_must_go_is_counted_as_undeletable(rig_for):
    with TurnServer(heard=[["tardi"]], ids=False, speech_delay_s=[0.3]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.input.cancel(job)
        assert wait_for(lambda: rig.route.snapshot()["audio_items_undeletable"] == 1)
        assert server.of("conversation.item.delete") == []


def test_a_late_item_of_a_cancelled_burst_never_joins_the_next_one(rig_for):
    """I-7: the item's position in the server's input buffer says which burst it is from."""
    with TurnServer(heard=[["uno"], ["due"]], speech_delay_s=[0.3, 0.8]) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.input.cancel(first)
        second = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.text == "due"
        assert second.result.items == ("in_2_0",)
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 1)
        assert server.of("conversation.item.delete")[0]["item_id"] == "in_1_0"


def test_a_burst_over_the_item_cap_ends_without_a_transcript(rig_for):
    """M-5: a server cannot grow the robot's state without bound."""
    with TurnServer(heard=[["a", "b", "c"]]) as server:
        rig = rig_for(server, rt={"transcript_max_items": 2})
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 3)


def test_a_transcript_over_the_character_cap_ends_without_a_transcript(rig_for):
    with TurnServer(heard=[["x" * 300]]) as server:
        rig = rig_for(server, rt={"transcript_max_chars": 200})
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 1)


def test_items_of_an_older_session_are_not_deleted_on_the_new_one():
    """M-4: a delete names its session; after a reset the old ids are left alone."""
    import threading
    from types import SimpleNamespace

    from roboemot.config.schema import RoboEmotConfig
    from roboemot.realtime import protocol

    cfg = RoboEmotConfig.model_validate(
        {"voice": {"realtime_route": {"input": "audio"}}, "privacy": {"realtime_audio": True}}
    )
    deletes, sent = [], threading.Event()

    def send_audio(pcm, *, gate, instructions, hold, track):
        track(1, 0.0, 1000.0)
        sent.set()

    route = SimpleNamespace(
        generation=1,
        connected=True,
        stack=SimpleNamespace(effective=lambda: cfg),
        base_instructions=lambda: "",
        send_audio=send_audio,
        delete_items=lambda gen, items: deletes.append((gen, items)) or len(items),
        count=lambda *a: None,
        input_held=lambda: False,
    )
    inp = RealtimeInput(route)
    try:
        job = inp.submit(speech(0.5), RATE)
        assert sent.wait(5)
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, "a", 10))
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, "b", 20))
        inp.on_event(protocol.InputTranscript(protocol.TRANSCRIPTION_COMPLETED, "a", "x", True))
        route.generation = 2  # the session was reset meanwhile
        inp.cancel(job)
        assert deletes == [(1, ["a"])]  # named with its session: delete_items drops it
    finally:
        inp.close()


def test_the_burst_and_its_items_never_show_in_repr(rig_for):
    from roboemot.realtime.input import _Burst

    burst = _Burst(None, 1)
    burst.order["in_secret"] = None
    burst.open.add("in_secret")
    assert "in_secret" not in repr(burst)
