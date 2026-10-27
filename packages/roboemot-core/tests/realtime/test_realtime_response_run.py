"""``ResponseRun`` keeps to its own response (SPEC 8, Realtime speech route): events of another
response are dropped, and nothing starts a stream after the run ended its feeds."""

from __future__ import annotations

import asyncio
import contextlib

from roboemot.realtime import protocol
from roboemot.realtime.turn import ResponseRun

PCM = b"\x01\x00" * 160


class Route:
    output_rate_hz = 16000

    def __init__(self):
        self.handlers = None
        self.sent = []
        self.current_response = None

    @contextlib.contextmanager
    def listen(self, on_event, on_close):
        self.handlers = (on_event, on_close)
        yield
        self.handlers = None

    def request_response(self, instructions, tool_choice=None):
        return True

    def send(self, event):
        self.sent.append(event)
        return True

    def note_response(self, done):
        pass


class Speech:
    def __init__(self):
        self.streams = []
        self.questions = []

    def stream(self, feed, **kw):
        self.streams.append(feed)
        return len(self.streams)

    def mark_question(self, job):
        self.questions.append(job)


def delta(rid, item):
    return protocol.AudioDelta(protocol.AUDIO_DELTA, rid, item, PCM)


async def start(route, speech, calls=None):
    async def on_call(ev):
        calls.append(ev)
        return False

    run = ResponseRun(
        route,
        speech,
        utterance=1,
        mood=None,
        volume_db=0.0,
        language="it",
        clock=lambda: 0.0,
        on_call=on_call if calls is not None else None,
    )
    task = asyncio.ensure_future(run.run("x"))
    while route.handlers is None:
        await asyncio.sleep(0)
    return run, task


async def test_events_of_another_response_are_dropped():
    route, speech, calls = Route(), Speech(), []
    run, task = await start(route, speech, calls)
    on_event = route.handlers[0]
    on_event(protocol.ResponseCreated(protocol.RESPONSE_CREATED, "r1"))
    await asyncio.sleep(0.01)
    on_event(delta("stale", "old_item"))  # an unrequested or stale response's audio
    on_event(protocol.ReplyTranscript(protocol.REPLY_TRANSCRIPT_DONE, "stale", "i", "Eh?", True))
    on_event(protocol.FunctionCall(protocol.FUNCTION_CALL_DONE, "stale", "c", "play", "{}"))
    assert speech.streams == [] and run.reply.texts == {}
    on_event(delta("r1", "item"))
    on_event(protocol.ResponseDone(protocol.RESPONSE_DONE, "r1", "completed"))
    reply = await asyncio.wait_for(task, 2)
    assert len(speech.streams) == 1 and reply.status == "completed" and calls == []


async def test_no_stream_starts_after_the_run_ended_its_feeds():
    route, speech = Route(), Speech()
    run, task = await start(route, speech)
    on_event = route.handlers[0]
    on_event(protocol.ResponseCreated(protocol.RESPONSE_CREATED, "r1"))
    on_event(delta("r1", "item"))
    on_event(protocol.ResponseDone(protocol.RESPONSE_DONE, "r1", "completed"))
    await asyncio.wait_for(task, 2)
    on_event(delta("r1", "late_item"))  # would make a feed that never ends
    assert len(speech.streams) == 1
    assert speech.streams[0].closed or speech.streams[0]._ended
