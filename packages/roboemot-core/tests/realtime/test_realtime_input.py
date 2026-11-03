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
    with TurnServer(heard=[[], ["dopo"]]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        assert not job.done
        rig.input.cancel(job)
        assert job.done and job.result is None
        later = rig.input.submit(speech(0.5), RATE)  # waits for the session the reset opens
        assert wait_for(lambda: later.done) and later.result.text == "dopo"
        assert server.of("conversation.item.delete") == []
        assert rig.route.snapshot()["input_resets"] == 1


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
    sleep = SimpleNamespace(asleep=False)
    with ExitStack() as cleanup:
        bind_realtime_input(
            host,
            route,
            SimpleNamespace(bind_realtime=lambda audio_in, on_audio=None: bound.append(audio_in)),
            cleanup,
            sleep,
        )
        (audio_in,) = bound
        assert route.input is audio_in
        assert not audio_in._speaking() and not audio_in._asleep()
        host.speech.extend(10**9)
        assert audio_in._speaking()  # nothing goes while the robot speaks
        sleep.asleep = True
        assert audio_in._asleep()  # nor while it sleeps (SleepGate.asleep, B-1)
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
        assert not rig.input._thread_alive()
        later = RealtimeInput(rig.route)  # a later burst: its commit comes after any late send
        try:
            job = later.submit(speech(0.5), RATE)
            assert wait_for(lambda: job.done)
            assert len(server.of("input_audio_buffer.commit")) == 1
            assert len(server.of("session.update")) == 2  # the connect's and the later burst's
        finally:
            later.close()


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
        snap = rig.route.snapshot()
        assert snap["audio_held"] == 1  # once per burst, however long the hold
        assert snap["audio_refused"] == 0  # a hold is not a refusal


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
    """The burst is covered (its speech reported) but cancelled before its transcript: the
    id-less transcript that comes later cannot be deleted, it is counted."""
    with TurnServer(heard=[["tardi"]], ids=False, transcript_delay_s=1.0) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: rig.input._burst is not None and rig.input._burst.covered())
        rig.input.cancel(job)
        assert rig.input._dead_anon == 1  # its transcript is still to come
        assert wait_for(lambda: rig.route.snapshot()["audio_items_undeletable"] == 1)
        assert rig.input._dead_anon == 0
        assert server.of("conversation.item.delete") == []
        assert rig.route.snapshot()["input_resets"] == 0


def test_an_id_less_burst_cancelled_before_the_server_reports_resets_the_session(rig_for):
    with TurnServer(heard=[["tardi"]], ids=False, speech_delay_s=[1.0]) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.input.cancel(job)  # well before the server's VAD speaks
        assert rig.input._dead_anon == 0  # nothing of it heard yet
        # nothing of it reported: the session resets, so its late item is stale, never joined
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        assert server.of("conversation.item.delete") == []
        assert rig.route.snapshot()["audio_items_undeletable"] == 0


def test_a_late_item_of_a_cancelled_burst_never_joins_the_next_one(rig_for):
    """I-7: a cancelled burst the server had not reported resets the session; its late item is
    stale, and the next burst waits for the new session."""
    with TurnServer(heard=[["uno"], ["due"]], speech_delay_s=[0.3, 0.8]) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        rig.input.cancel(first)
        second = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.text == "due"
        assert second.result.items == ("in_2_0",)
        assert second.result.generation == rig.route.generation
        assert rig.route.snapshot()["input_resets"] == 1
        assert server.of("conversation.item.delete") == []


def test_a_burst_over_the_item_cap_ends_without_a_transcript(rig_for):
    """M-5: a server cannot grow the robot's state without bound."""
    with TurnServer(heard=[["a", "b", "c"]]) as server:
        rig = rig_for(server, rt={"transcript_max_items": 2})
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None
        # it ends at its third item's start: if the input looks before that item's stop came,
        # the burst is not covered and the session resets (the server drops all three); if
        # after, it is covered and each item is deleted once its transcript comes
        assert wait_for(
            lambda: (
                rig.route.snapshot()["input_resets"] == 1
                or len(server.of("conversation.item.delete")) == 3
            )
        )


def test_a_transcript_over_the_character_cap_ends_without_a_transcript(rig_for):
    with TurnServer(heard=[["x" * 300]]) as server:
        rig = rig_for(server, rt={"transcript_max_chars": 200})
        job = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 1)
        assert rig.route.snapshot()["input_resets"] == 0  # covered: deleted, no reset


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
        track(1)
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
        opening=lambda: False,
        reset_session=lambda gen: False,
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


def test_the_upload_count_lives_in_the_data_folder(tmp_path):
    from roboemot import paths

    assert paths.realtime_upload_path(str(tmp_path)) == tmp_path / "realtime_upload.json"


# ------------------------------------------------------------------ fix round 2


def stub_input(generations, resets=False):
    """A RealtimeInput over a stub route whose gate always sends: ``generations`` gives each
    burst's session; counts go to the returned dict; ``resets``: a reset always happens."""
    import threading
    from types import SimpleNamespace

    from roboemot.config.schema import RoboEmotConfig

    cfg = RoboEmotConfig.model_validate(
        {"voice": {"realtime_route": {"input": "audio"}}, "privacy": {"realtime_audio": True}}
    )
    counts: dict[str, int] = {}
    sent = threading.Semaphore(0)
    order = iter(generations)

    def send_audio(pcm, *, gate, instructions, hold, track):
        track(next(order))
        sent.release()

    route = SimpleNamespace(
        generation=1,
        connected=True,
        stack=SimpleNamespace(effective=lambda: cfg),
        base_instructions=lambda: "",
        send_audio=send_audio,
        delete_items=lambda gen, items: len(items),
        count=lambda key, n=1: counts.__setitem__(key, counts.get(key, 0) + n),
        input_held=lambda: False,
        opening=lambda: False,
        reset_session=lambda gen: resets,
    )
    return RealtimeInput(route), sent, counts


def test_an_id_less_transcript_that_may_be_a_cancelled_bursts_is_never_joined():
    """I-N2: no ids, burst A cancelled with an item still to come, burst B live: the next
    id-less transcript could be A's. B ends without a result (a local decode), never with
    A's words."""
    from roboemot.realtime import protocol

    inp, sent, counts = stub_input([1, 1])
    started, done = protocol.SPEECH_STARTED, protocol.TRANSCRIPTION_COMPLETED
    try:
        a = inp.submit(speech(0.5), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(started, None, 100))
        inp.cancel(a)
        b = inp.submit(speech(0.5), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(started, None, 1100))
        inp.on_event(protocol.InputTranscript(done, None, "LATE-A", True))
        inp.on_event(protocol.InputTranscript(done, None, "B words", True))
        assert b.done and b.result is None
        assert counts["audio_items_ambiguous"] == 1
    finally:
        inp.close()


def test_an_id_less_burst_with_nothing_pending_still_joins():
    """The fail-safe applies only while a cancelled burst's id-less item may still come."""
    from roboemot.realtime import protocol

    inp, sent, counts = stub_input([1])
    try:
        job = inp.submit(speech(0.5), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, None, 100))
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STOPPED, None, 600))
        inp.on_event(protocol.InputTranscript(protocol.TRANSCRIPTION_COMPLETED, None, "ciao", True))
        assert job.done and job.result.text == "ciao"
        assert "audio_items_ambiguous" not in counts
    finally:
        inp.close()


def test_a_decode_job_keeps_no_audio_once_it_ran_or_was_dropped():
    """Minor (M-N5): the task closure holds a view of the utterance's audio."""
    from roboemot.voice.decode_worker import DecodeJob

    ran = DecodeJob(lambda: "ciao", 1.0)
    ran.run()
    assert ran.result == "ciao" and ran._task is None
    dropped = DecodeJob(lambda: "ciao", 1.0)
    dropped.drop()
    assert dropped._task is None


TRIM = {"trim_ms": 900.0}  # Pollen's server drops each burst's closing silence


def test_every_burst_gets_its_transcript_when_the_server_drops_the_closing_silence(rig_for):
    """Robot run 2026-10-29: the server counts its buffer up to each speech end, not the
    appended audio, so the robot's own count ran ahead and every utterance after the first
    was deleted as an earlier burst's (10 of 11 fell back to the local pipeline)."""
    with TurnServer(heard=[["uno"], ["due"], ["tre"]], **TRIM) as server:
        rig = rig_for(server)
        for word in ("uno", "due", "tre"):
            job = rig.input.submit(speech(1.0), RATE)
            assert wait_for(lambda job=job: job.done)
            assert job.result is not None and job.result.text == word
        assert server.of("conversation.item.delete") == []
        snap = rig.route.snapshot()
        assert snap["transcripts"] == 3 and snap["audio_items_ambiguous"] == 0


def test_a_cancelled_burst_with_its_speech_reported_needs_no_reset(rig_for):
    """The server reported the cancelled burst's whole speech: its transcript is deleted when it
    comes, the session stays, and the next burst starts at the server's end of speech."""
    with TurnServer(heard=[["uno"], ["due"]], transcript_delay_s=0.5, **TRIM) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.sent_of("input_audio_buffer.speech_stopped")) == 1)
        assert wait_for(lambda: rig.input._burst is not None and rig.input._burst.covered())
        rig.input.cancel(first)
        second = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.text == "due"
        assert wait_for(lambda: len(server.of("conversation.item.delete")) == 1)
        assert server.of("conversation.item.delete")[0]["item_id"] == "in_1_0"
        assert rig.route.snapshot()["input_resets"] == 0


def test_the_next_burst_starts_where_the_server_ended_the_speech(rig_for):
    """The start really moves: burst 2 starts at burst 1's audio_end_ms, not at the end of the
    audio the robot appended."""
    with TurnServer(heard=[["uno"], ["due"]], **TRIM) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: first.done)
        appended_ms = 1000 + rig.stack.effective().voice.realtime_route.close_silence_ms
        second = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: second.done) and second.result.text == "due"
        assert second.result.items == ("in_2_0",)
        first_end = server.sent_of("input_audio_buffer.speech_stopped")[0]["audio_end_ms"]
        second_start = server.sent_of("input_audio_buffer.speech_started")[1]["audio_start_ms"]
        assert second_start == first_end == appended_ms - TRIM["trim_ms"] < appended_ms


# ------------------------------------------------------------------ fix round 3: coverage


def test_pollens_measured_positions_count_as_covered(rig_for):
    """Probe on Pollen's server, 2026-10-29: 1.8 s of speech and 1.0 s of closing silence
    gave speech ends 2432 ms apart (the server keeps about 630 ms after the speech). Every
    burst is covered with the default server_silence_ms (600 ms): no reset, every transcript."""
    pollen = {"trim_ms": 2800.0 - 2432.0}  # the server's end: 2432 ms after the burst's start
    with TurnServer(heard=[["uno"], ["due"], ["tre"]], **pollen) as server:
        rig = rig_for(server, rt={"close_silence_ms": 1000.0})
        assert rig.stack.effective().voice.realtime_route.server_silence_ms == 600.0
        for word in ("uno", "due", "tre"):
            job = rig.input.submit(speech(1.8), RATE)
            assert wait_for(lambda job=job: job.done)
            assert job.result is not None and job.result.text == word
        starts = [e["audio_start_ms"] for e in server.sent_of("input_audio_buffer.speech_started")]
        assert starts == [0, 2432, 4864]
        snap = rig.route.snapshot()
        assert snap["transcripts"] == 3 and snap["input_resets"] == 0
        assert server.of("conversation.item.delete") == []


def test_a_cancelled_burst_with_a_late_second_segment_never_joins_the_next(rig_for):
    """Every item it opened stopped, but the server's last end lies far before the utterance's
    end: a second segment may still come (at the next burst's start or after). Not covered:
    the session resets, the late segment is stale, the next burst hears only its own words."""
    heard = [["uno", "tardi"], ["due"]]
    with TurnServer(heard=heard, segment_gap_s=1.0, **TRIM) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(1.0), RATE)
        done = "conversation.item.input_audio_transcription.completed"
        assert wait_for(lambda: len(server.sent_of(done)) == 1)  # "uno" in, "tardi" to come
        assert not first.done  # not covered: no result yet
        rig.input.cancel(first)
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        second = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.text == "due"
        assert second.result.items == ("in_2_0",)
        assert second.result.generation == rig.route.generation


def test_a_late_second_segment_after_a_transcript_is_waited_for(rig_for):
    """The first segment's transcript is in, but the burst is not covered yet: the job waits
    (within its deadline) for the second segment instead of ending with half the words, which
    would then land in the next burst."""
    with TurnServer(heard=[["uno", "tardi"]], segment_gap_s=0.5, **TRIM) as server:
        rig = rig_for(server)
        job = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is not None and job.result.text == "uno tardi"
        assert job.result.items == ("in_1_0", "in_1_1")
        assert rig.route.snapshot()["input_resets"] == 0


def test_a_server_without_end_positions_never_covers_a_burst(rig_for):
    """Start positions but no audio_end_ms: the robot cannot tell the server is done with a
    burst, so it never takes its transcript (a local decode) and resets the session."""
    with TurnServer(heard=[["uno"]], end_positions=False) as server:
        rig = rig_for(server, rt={"transcript_timeout_s": 0.5})
        job = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: job.done)
        assert job.result is None
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        assert rig.route.snapshot()["transcripts"] == 0


def test_without_item_ids_a_reset_forgets_the_pending_items(rig_for):
    """A reset ends the old session's id-less items with it: the next session's first burst is
    not ambiguous (none of them can reach it)."""
    heard = [["uno"], ["due"]]
    with TurnServer(heard=heard, ids=False, transcript_delay_s=1.0, trim_ms=10_000.0) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(1.0), RATE)
        stopped = "input_audio_buffer.speech_stopped"
        assert wait_for(lambda: len(server.sent_of(stopped)) == 1)  # ends at 0: not covered
        server.trim_ms = 0.0  # the next session's burst is reported in full
        rig.input.cancel(first)  # its id-less item's transcript is still to come
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        assert rig.input._dead_anon == 0
        second = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.text == "due"
        assert rig.route.snapshot()["audio_items_ambiguous"] == 0


def test_an_item_id_repeated_by_the_next_session_is_not_taken_for_a_dead_one(rig_for):
    """Ids are per session: a buried id of the reset session must not delete the new
    session's item of the same name."""
    heard = [["uno"], ["due"]]
    kw = {"ids_per_session": True, "transcript_delay_s": 1.0, "trim_ms": 10_000.0}
    with TurnServer(heard=heard, **kw) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(1.0), RATE)
        stopped = "input_audio_buffer.speech_stopped"
        assert wait_for(lambda: len(server.sent_of(stopped)) == 1)
        server.trim_ms = 0.0
        rig.input.cancel(first)  # "in_1_0" of session 1 is buried
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        second = rig.input.submit(speech(1.0), RATE)
        assert wait_for(lambda: second.done)
        assert second.result is not None and second.result.items == ("in_1_0",)
        assert second.result.text == "due"
        assert server.of("conversation.item.delete") == []


def test_a_slow_reconnect_ends_the_next_burst_at_its_deadline_without_inflated_counts(rig_for):
    """The reset's reconnect hangs: the next burst waits for it (counted once, as a reopen
    wait, not as a hold or a refusal) and ends at its listener deadline, unsent."""
    import time

    with TurnServer(heard=[[]]) as server:
        rig = rig_for(server)
        first = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        server.mode = "no_session"  # the new session never answers its session.update
        rig.input.cancel(first)
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        later = rig.input.submit(
            speech(0.5), RATE, deadline=time.monotonic() + 0.3, clock=time.monotonic
        )
        assert wait_for(lambda: later.done)
        assert later.refused == "cancelled" and not later.sent
        snap = rig.route.snapshot()
        assert snap["audio_reopen_wait"] == 1
        assert snap["audio_refused"] == 0 and snap["audio_held"] == 0
        assert rig.input.held == 0
        assert len(server.of("input_audio_buffer.commit")) == 1


def test_a_failed_reconnect_stops_the_wait_and_the_burst_is_refused_once(rig_for):
    """The reset's reconnect fails: the next burst stops waiting once the route gave up and is
    refused (no session) once, then decoded locally."""
    with TurnServer(heard=[[]]) as server:
        rig = rig_for(server, rt={"connect_timeout_s": 1})
        first = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        server.mode = "no_session"
        rig.input.cancel(first)
        assert wait_for(lambda: rig.route.snapshot()["input_resets"] == 1)
        later = rig.input.submit(speech(0.5), RATE)
        assert wait_for(lambda: later.done, 5.0)
        assert later.refused == "no_session"
        snap = rig.route.snapshot()
        assert snap["audio_reopen_wait"] == 1 and snap["audio_refused"] == 1


def test_the_next_burst_starts_at_the_end_the_server_reported_for_its_session():
    """The start is read under the input's lock with the send, from the stops of the burst's
    own session: an item before it is an earlier burst's (deleted), one at it is the burst's;
    a new session starts at zero."""
    from roboemot.realtime import protocol

    started, stopped = protocol.SPEECH_STARTED, protocol.SPEECH_STOPPED
    done = protocol.TRANSCRIPTION_COMPLETED
    inp, sent, counts = stub_input([1, 1, 2])
    try:
        a = inp.submit(speech(1.8), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(started, "a", 0), 1)
        inp.on_event(protocol.SpeechBoundary(stopped, "a", 2432), 1)
        inp.on_event(protocol.InputTranscript(done, "a", "uno", True), 1)
        assert a.done and a.result.items == ("a",)
        inp.on_event(protocol.SpeechBoundary(stopped, "x", 9000), 0)  # an older session's
        b = inp.submit(speech(1.8), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(started, "early", 2400), 1)  # before: not b's
        inp.on_event(protocol.SpeechBoundary(started, "b", 2432), 1)
        inp.on_event(protocol.SpeechBoundary(stopped, "b", 4864), 1)
        inp.on_event(protocol.InputTranscript(done, "early", "vecchio", True), 1)
        inp.on_event(protocol.InputTranscript(done, "b", "due", True), 1)
        assert b.done and b.result.items == ("b",) and b.result.text == "due"
        assert counts["audio_items_deleted"] == 1
        c = inp.submit(speech(1.8), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(started, "c", 0), 2)  # a new session: from 0
        inp.on_event(protocol.SpeechBoundary(stopped, "c", 2432), 2)
        inp.on_event(protocol.InputTranscript(done, "c", "tre", True), 2)
        assert c.done and c.result.items == ("c",) and c.result.generation == 2
    finally:
        inp.close()


def test_a_reset_sessions_event_still_in_flight_is_dropped():
    """An event of the reset session that was already on its way (the route checked its token
    before the reset) ended with that session: it never counts as a pending item."""
    from roboemot.realtime import protocol

    inp, sent, counts = stub_input([1], resets=True)
    try:
        job = inp.submit(speech(1.0), RATE)
        assert sent.acquire(timeout=5)
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, None, 0), 1)
        inp.cancel(job)  # never stopped: not covered, the session resets
        assert wait_for(lambda: inp._reset_gen == 1)
        assert inp._dead_anon == 0
        inp.on_event(protocol.SpeechBoundary(protocol.SPEECH_STARTED, None, 50), 1)
        inp.on_event(protocol.InputTranscript(protocol.TRANSCRIPTION_COMPLETED, None, "x", True), 1)
        assert inp._dead_anon == 0 and "audio_items_undeletable" not in counts
    finally:
        inp.close()
