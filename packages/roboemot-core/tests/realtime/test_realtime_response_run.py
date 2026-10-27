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


async def test_a_never_created_request_that_expires_resets_the_session(monkeypatch):
    """probe_owed / probe_r3_misbind: turn A is cut before its response.created. Once it has
    had no event for the stall timeout the session is reset (closed and reopened): A's late
    response.created can never take B's place, and the next turn completes."""
    route, sent = connected_route()
    a = make_run(route, Speech())
    task_a = asyncio.ensure_future(a.run("a"))
    await until(lambda: creates(sent))
    task_a.cancel()  # barge-in before response.created
    await asyncio.gather(task_a, return_exceptions=True)
    assert route.busy()
    monkeypatch.setattr(route, "timeouts", lambda: (2.0, 0.1))
    first_client = route._client
    reopen = route.start
    monkeypatch.setattr(route, "start", lambda: None)  # reopened below, deterministically
    b = make_run(route, Speech())
    reply_b = await asyncio.wait_for(b.run("b"), 2)
    assert len(creates(sent)) == 1  # B was never sent on the old session
    assert reply_b.closed == "no_session"  # B falls back locally (the B1 rules)
    route._on_event(created("A"), None)  # A's late created: no run takes it
    assert route.snapshot()["resets"] == 1 and not route.connected
    reopen()
    await until(lambda: route.connected and route._client is not first_client)
    sent2 = []
    route._client.send = sent2.append
    speech = Speech()
    c = make_run(route, speech)
    task_c = asyncio.ensure_future(c.run("c"))
    await until(lambda: creates(sent2))
    route._on_event(created("C"))
    route._on_event(delta("C"))
    route._on_event(done("C"))
    reply = await asyncio.wait_for(task_c, 2)
    assert reply.status == "completed" and len(speech.streams) == 1


async def test_a_late_done_after_expiry_is_charged_once(monkeypatch):
    """probe_r3_double_charge: an abandoned live response expires (charged its estimate);
    its late response.done is not charged again. A response still streaming does not expire
    (the expiry ages from its last event)."""
    route, sent = connected_route()
    charged = []
    monkeypatch.setattr(route, "charge", charged.append)
    monkeypatch.setattr(route, "timeouts", lambda: (2.0, 0.2))
    a = make_run(route, Speech())
    task_a = asyncio.ensure_future(a.run("a" * 400))
    await until(lambda: creates(sent))
    route._on_event(created("A"))
    await asyncio.sleep(0.01)
    task_a.cancel()  # barge-in after created
    await asyncio.gather(task_a, return_exceptions=True)
    for _ in range(4):  # still streaming, gaps under the stall timeout: not expired
        await asyncio.sleep(0.08)
        route._on_event(delta("A", "iA"))
        route.expire()
    assert charged == []
    route._on_event(P.ResponseDone(P.RESPONSE_DONE, "A", "cancelled", 77))
    assert charged == [77]
    # and once expired, a late done is not charged again
    b = make_run(route, Speech())
    task_b = asyncio.ensure_future(b.run("b" * 400))
    await until(lambda: len(creates(sent)) == 2)
    route._on_event(created("B2"))
    task_b.cancel()
    await asyncio.gather(task_b, return_exceptions=True)
    await asyncio.sleep(0.25)
    route.expire()
    assert len(charged) == 2  # its estimate
    route._on_event(P.ResponseDone(P.RESPONSE_DONE, "B2", "cancelled", 99))
    assert len(charged) == 2 and 99 not in charged


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


async def test_a_usage_less_done_is_charged_once_even_if_the_run_is_cut_right_after(monkeypatch):
    route, sent = connected_route()
    charged = []
    monkeypatch.setattr(route, "charge", charged.append)
    run = make_run(route, Speech())
    task = asyncio.ensure_future(run.run("x" * 400))
    await until(lambda: creates(sent))
    route._on_event(created("R"))
    route._on_event(P.ResponseDone(P.RESPONSE_DONE, "R", "completed", None))
    task.cancel()  # cut before the run read its done
    await asyncio.gather(task, return_exceptions=True)
    route.expire()
    assert charged == [100]  # its estimate (400 chars / 4), charged by the route, once


async def test_a_request_pending_on_a_closed_session_is_settled_at_once():
    """probe_r4_orphan: turn A is cut before its response.created; the conversation ends
    (the session is detached, its close comes later). Turn B on the new session is sent at
    once and completes: no wait, no reset; A is charged once."""
    route, sent = connected_route()
    charged = []
    route.charge = charged.append
    route.timeouts = lambda: (1.0, 0.3)
    a = make_run(route, Speech())
    task_a = asyncio.ensure_future(a.run("a" * 400))
    await until(lambda: creates(sent))
    task_a.cancel()
    await asyncio.gather(task_a, return_exceptions=True)
    old = route._client
    route.close_soon("conversation_end")
    assert not route.busy() and charged == [100]
    await until(lambda: old.closed == 1)
    assert await asyncio.to_thread(route.ready)
    sent2 = []
    route._client.send = sent2.append
    b = make_run(route, Speech())
    t0 = asyncio.get_running_loop().time()
    task_b = asyncio.ensure_future(b.run("b"))
    await until(lambda: creates(sent2), 0.2)
    assert asyncio.get_running_loop().time() - t0 < 0.2
    route._on_event(created("B"))
    route._on_event(delta("B"))
    route._on_event(done("B"))
    reply = await asyncio.wait_for(task_b, 2)
    assert reply.status == "completed"
    snap = route.snapshot()
    assert snap["resets"] == 0 and snap["stale_callbacks"] == 0 and charged == [100, 5]


async def test_a_run_on_a_session_closed_by_a_reload_falls_back_at_once():
    """probe_r4_midrun: pseudonyms come on mid-stream; the run ends now (closed), its
    response is charged once, and nothing stays in flight."""
    route, sent = connected_route()
    charged = []
    route.charge = charged.append
    route.timeouts = lambda: (1.0, 1.0)
    run = make_run(route, Speech())
    task = asyncio.ensure_future(run.run("x" * 40))
    await until(lambda: creates(sent))
    route._on_event(created("R"), route._token)
    route._on_event(delta("R"), route._token)
    t0 = asyncio.get_running_loop().time()
    route.stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)
    route.apply_config(route.stack.effective())
    reply = await asyncio.wait_for(task, 0.5)
    assert asyncio.get_running_loop().time() - t0 < 0.2
    assert reply.closed == "reconfigured"
    assert route._live == {} and not route.busy() and charged == [10]


async def test_the_estimate_counts_the_utterance_too(monkeypatch):
    """As the old worker estimate did: instructions + utterance + reply, chars / 4."""
    route, sent = connected_route()
    charged = []
    monkeypatch.setattr(route, "charge", charged.append)
    monkeypatch.setattr(route, "timeouts", lambda: (2.0, 0.05))
    run = make_run(route, Speech(), input_chars=40)
    task = asyncio.ensure_future(run.run("x" * 400))
    await until(lambda: creates(sent))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.06)
    route.expire()
    assert charged == [110]
