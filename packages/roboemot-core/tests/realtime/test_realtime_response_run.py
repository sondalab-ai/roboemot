"""Response correlation (SPEC 8, Realtime speech route): a run gets only its own response's
events, through the real ``RealtimeRoute`` (a stub client); a refusal is an ``error`` that
echoes its request's ``event_id``; an abandoned request never captures a later response."""

from __future__ import annotations

import asyncio

from roboemot.realtime import protocol as P
from roboemot.realtime.turn import ResponseRun
from test_realtime_route import rt_stack, stub_route

PCM = b"\x01\x00" * 160


class Speech:
    def __init__(self):
        self.streams = []

    def stream(self, feed, **kw):
        self.streams.append(feed)
        return len(self.streams)

    def mark_question(self, job):
        pass


def connected_route():
    stack = rt_stack()
    route, made = stub_route(stack)
    assert route.ready()
    sent = []
    made[0].send = sent.append
    return route, sent


def make_run(route, speech, **kw):
    return ResponseRun(
        route,
        speech,
        utterance=1,
        mood=None,
        volume_db=0.0,
        language="it",
        clock=asyncio.get_running_loop().time,
        **kw,
    )


def created(rid):
    return P.ResponseCreated(P.RESPONSE_CREATED, rid)


def delta(rid, item="item"):
    return P.AudioDelta(P.AUDIO_DELTA, rid, item, PCM)


def done(rid, tokens=5):
    return P.ResponseDone(P.RESPONSE_DONE, rid, "completed", tokens)


async def until(cond, timeout=2.0):
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < end
        await asyncio.sleep(0.005)


def creates(sent):
    return [e for e in sent if e["type"] == "response.create"]


async def test_events_of_another_response_are_dropped_before_and_after_created():
    route, sent = connected_route()
    speech, calls = Speech(), []

    async def on_call(ev):
        calls.append(ev)
        return False

    run = make_run(route, speech, on_call=on_call)
    task = asyncio.ensure_future(run.run("x"))
    await until(lambda: creates(sent))
    route._on_event(delta("stale", "old"))  # before the run's own created: dropped
    route._on_event(created("r1"))
    route._on_event(delta("stale", "old"))
    route._on_event(P.ReplyTranscript(P.REPLY_TRANSCRIPT_DONE, "stale", "old", "Eh?", True))
    route._on_event(P.FunctionCall(P.FUNCTION_CALL_DONE, "stale", "c", "play", "{}"))
    await asyncio.sleep(0.01)
    assert speech.streams == [] and run.reply.texts == {} and calls == []
    route._on_event(delta("r1"))
    route._on_event(done("r1"))
    reply = await asyncio.wait_for(task, 2)
    assert len(speech.streams) == 1 and reply.status == "completed"
    assert route.snapshot()["foreign_events"] == 4


async def test_no_stream_starts_after_the_run_ended_its_feeds():
    route, sent = connected_route()
    speech = Speech()
    run = make_run(route, speech)
    task = asyncio.ensure_future(run.run("x"))
    await until(lambda: creates(sent))
    route._on_event(created("r1"))
    route._on_event(delta("r1"))
    route._on_event(done("r1"))
    await asyncio.wait_for(task, 2)
    route._on_event(delta("r1", "late"))  # a late delta: no feed that never ends
    run.deliver(delta("r1", "late"))
    assert len(speech.streams) == 1


async def test_an_abandoned_request_never_created_does_not_capture_the_next_turn():
    """probe_owed: turn A is cut before its response.created and the server never creates it;
    turn B's response is B's, and B completes."""
    route, sent = connected_route()
    a = make_run(route, Speech())
    task_a = asyncio.ensure_future(a.run("a"))
    await until(lambda: creates(sent))
    task_a.cancel()  # barge-in before response.created
    await asyncio.gather(task_a, return_exceptions=True)
    assert route.busy()
    speech = Speech()
    b = make_run(route, speech)
    task_b = asyncio.ensure_future(b.run("b"))
    await asyncio.sleep(0.01)
    assert len(creates(sent)) == 1  # B waits while A is in flight (correlation by order)
    route._on_event(done("unrelated"))  # any response.done expires a never-created request
    await until(lambda: len(creates(sent)) == 2)
    route._on_event(created("B"))
    route._on_event(delta("B"))
    route._on_event(done("B"))
    reply = await asyncio.wait_for(task_b, 2)
    assert reply.status == "completed" and len(speech.streams) == 1


async def test_an_abandoned_request_expires_after_the_stall_timeout(monkeypatch):
    route, sent = connected_route()
    a = make_run(route, Speech())
    task_a = asyncio.ensure_future(a.run("a"))
    await until(lambda: creates(sent))
    task_a.cancel()
    await asyncio.gather(task_a, return_exceptions=True)
    monkeypatch.setattr(route, "timeouts", lambda: (2.0, 0.1))  # a short stall timeout
    b = make_run(route, Speech())
    task_b = asyncio.ensure_future(b.run("b"))
    await until(lambda: len(creates(sent)) == 2)  # expired: B goes on
    assert not route.busy()
    route._on_event(created("B"))
    route._on_event(done("B"))
    assert (await asyncio.wait_for(task_b, 2)).status == "completed"


async def test_a_refusal_echoes_the_request_and_an_unrelated_error_is_not_one():
    route, sent = connected_route()
    run = make_run(route, Speech())
    task = asyncio.ensure_future(run.run("x"))
    await until(lambda: creates(sent))
    route._on_event(P.ServerError(P.ERROR, "other", "ev_999"))  # about another event
    await asyncio.sleep(0.01)
    assert not task.done()
    route._on_event(P.ServerError(P.ERROR, "refused", creates(sent)[0]["event_id"]))
    reply = await asyncio.wait_for(task, 2)
    assert reply.status == "error"
    snap = route.snapshot()
    assert snap["refused"] == 1 and snap["errors_unrelated"] == 1


async def test_a_run_never_hangs_without_its_created():
    route, sent = connected_route()
    run = make_run(route, Speech())
    run.route.timeouts = lambda: (0.05, 0.05)
    reply = await asyncio.wait_for(run.run("x"), 2)
    assert reply.closed == "stall" and reply.abandoned
    assert any(e["type"] == "response.cancel" for e in sent)
