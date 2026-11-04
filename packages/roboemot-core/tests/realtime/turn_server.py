"""A fake realtime server for whole turns (SPEC 8, Realtime speech route): the SR-1 fake, plus
responses scripted one per ``response.create``, served on their own task so a
``response.cancel`` is read while a reply streams.

Scripts (``scripts``, used in order; the last one repeats):

- ``reply``: ``response.created``, ``deltas`` audio deltas of ``delta_ms`` each ``gap_s`` apart,
  the reply transcript (``transcript``), ``response.output_audio.done`` and ``response.done``
  with usage ``usage`` (None leaves usage out);
- ``tool``: ``response.created``, one ``response.function_call_arguments.done`` (``tool_name``,
  ``tool_args``) and ``response.done``;
- ``audio_then_tool``: the ``reply`` audio, then the ``tool`` call, then ``response.done``;
- ``tool_then_audio``: the ``tool`` call, then the ``reply`` audio (both stop at a cancel);
- ``tools``: one call per name of ``tool_names`` in one response, then ``response.done``;
- ``tools_then_audio``: the ``tools`` calls, then the ``reply`` audio;
- ``tool_close``: the ``tool`` call, then the socket is closed;
- ``tool_stall``: the ``tool`` call, then ``response.done`` only after ``stall_s``;
- ``close_before``: the socket is closed on ``response.create``;
- ``close_mid``: ``response.created``, one audio delta, then the socket is closed;
- ``error``: an ``error`` instead of ``response.created``;
- ``updated_then_refused``: ``session.updated``, then the ``error`` that echoes the request;
- ``unrelated_error``: ``session.updated``, then an ``error`` about another event;
- ``empty``: ``response.created`` and ``response.done``, no audio;
- ``silent``: nothing at all;
- ``pending``: the ``error`` ``function_call_output_pending`` that echoes the request;
- ``cancel_then_tool``: the ``reply`` audio; at a cancel, the ``tool`` call, then
  ``response.done`` ``cancelled`` (a call that arrives after the cut).

``strict_calls`` refuses a ``response.create`` with ``function_call_output_pending`` (echoing it,
no script used) while a call it sent has neither a ``function_call_output`` nor a delete of its
item, as the platform's server does (robot run 2026-11-02); each connection starts with none.
``reject_outputs`` answers every ``function_call_output`` with an ``error`` echoing it and
keeps its call open (a call item the cancel left missing, or a response still active).

``unrequested`` sends a ``response.created`` nobody asked for right after ``session.updated``.
A ``response.cancel`` stops the reply in progress with ``response.done`` ``cancelled``.

The server's VAD (``heard``, one list of transcripts per committed burst, the last one
repeats) answers each ``input_audio_buffer.commit``: one ``speech_started`` and
``speech_stopped`` per transcript (a split when there are several), then their
``transcription.completed`` after ``transcript_delay_s``; an empty list hears nothing.
``vad_answers`` (when set) makes it answer by itself, as a backend that ignores
``create_response: false`` would. ``speech_delay_s`` (one per burst, the last repeats) holds
everything a burst's VAD says; ``speech_started`` carries ``audio_start_ms``, its position in
the session's input buffer (from where the previous burst's speech ended), unless
``positions`` is off, and ``speech_stopped`` its ``audio_end_ms``. ``trim_ms`` makes it
count like Pollen's server (2026-10-29): a burst's last ``trim_ms`` (its closing silence) is
dropped, so the next burst starts at the previous speech end, not at the end of the appended
audio. ``end_positions`` off leaves ``audio_end_ms`` out (start positions only).
``segment_gap_s`` (when set) sends a burst's items one at a time, each with its transcript,
that long apart: a late second segment. ``ids`` off leaves every ``item_id`` out (a server
that does not send them); ``ids_per_session`` numbers the items per connection, so an id
repeats across sessions. ``tool_names`` (when set) names the call of each response in turn
(the last repeats) instead of ``tool_name``; ``call_args`` (call id -> arguments) overrides
``tool_args`` for one call. ``sent`` records every event the server sent.

``hold`` (a ``threading.Event``) holds what the VAD says at ``hold_at`` until the test sets it:
``speech`` (everything a burst's VAD says), ``transcript`` (the ``transcription.completed``
events) or ``segment`` (every segment after a burst's first, with ``segment_gap_s``). A test
that cancels a burst before the server's next word sets it after the cancel, so the order
does not depend on how fast the test runs. A hold nobody released fails the test when the
server closes (``__exit__`` raises): one that ran past ``hold_max_s`` (the server went on and
counted it in ``hold_timeouts``) or one still waiting.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from typing import Any

import numpy as np
from fake_server import FakeServer
from websockets.asyncio.server import ServerConnection

RATE = 16000
REPLY = "Ecco la mia risposta per te"
PENDING = "function_call_output_pending"
HOLD_MAX_S = 10.0  # default bound of a hold nobody releases (hold_timeouts)
HOLD_POLL_S = 0.005
HOLD_POINTS = ("speech", "transcript", "segment")


def pcm(ms: float, level: float = 0.25) -> bytes:
    """A PCM16 tone of ``ms`` at ``RATE``."""
    n = round(RATE * ms / 1000)
    t = np.arange(n) / RATE
    wave = (np.sin(2 * np.pi * 220 * t) * level * 32767).astype("<i2")
    return wave.tobytes()


class TurnServer(FakeServer):
    def __init__(
        self,
        scripts: list[str] | None = None,
        *,
        deltas: int = 3,
        delta_ms: float = 100.0,
        gap_s: float = 0.0,
        transcript: str = REPLY,
        usage: int | None = 42,
        tool_name: str = "stop_talking",
        tool_args: str = "{}",
        unrequested: bool = False,
        heard: list[list[str]] | None = None,
        transcript_delay_s: float = 0.0,
        vad_answers: bool = False,
        speech_delay_s: list[float] | None = None,
        positions: bool = True,
        ids: bool = True,
        trim_ms: float = 0.0,
        end_positions: bool = True,
        segment_gap_s: float = 0.0,
        ids_per_session: bool = False,
        tool_names: list[str] | None = None,
        stall_s: float = 1.5,
        strict_calls: bool = False,
        reject_outputs: bool = False,
        call_args: dict[str, str] | None = None,
        hold: threading.Event | None = None,
        hold_at: str = "transcript",
        hold_max_s: float = HOLD_MAX_S,
        **kw: Any,
    ) -> None:
        super().__init__("normal", **kw)
        assert hold_at in HOLD_POINTS
        self.hold, self.hold_at, self.hold_max_s = hold, hold_at, hold_max_s
        self.hold_timeouts = 0
        self.holding = 0  # tasks waiting at the hold now
        self.scripts = list(scripts or ["reply"])
        self.deltas, self.delta_ms, self.gap_s = deltas, delta_ms, gap_s
        self.transcript, self.usage = transcript, usage
        self.tool_name, self.tool_args = tool_name, tool_args
        self.tool_names, self.stall_s = tool_names, stall_s
        self.call_args = call_args or {}
        self.strict_calls = strict_calls
        self.reject_outputs = reject_outputs
        self.open_calls: dict[str, str] = {}  # call id -> its item id, on this connection
        self.pending_refusals = 0  # response.create refused for an open call (strict_calls)
        self.unrequested = unrequested
        self.heard = heard if heard is not None else [["ciao Neve come stai"]]
        self.transcript_delay_s = transcript_delay_s
        self.vad_answers = vad_answers
        self.speech_delay_s = speech_delay_s or [0.0]
        self.positions, self.ids = positions, ids
        self.trim_ms = trim_ms
        self.end_positions, self.segment_gap_s = end_positions, segment_gap_s
        self.ids_per_session = ids_per_session
        self.session_bursts = 0  # bursts committed on the current connection
        self.sent: list[dict[str, Any]] = []
        self.buffer_ms = 0.0  # audio appended in this session so far
        self.bursts = 0
        self._burst_start = 0.0
        self.responses = 0
        self._cancel: asyncio.Event | None = None
        self._tasks: set[asyncio.Task] = set()

    async def _send(self, ws: ServerConnection, event: dict[str, Any]) -> None:
        self.sent.append(event)
        await super()._send(ws, event)

    def sent_of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.sent if e.get("type") == kind]

    def kinds(self) -> list[str]:
        return [e.get("type") for e in self.received]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.received if e.get("type") == kind]

    def audio(self) -> bytes:
        """Every PCM byte appended so far."""
        return b"".join(base64.b64decode(e["audio"]) for e in self.of("input_audio_buffer.append"))

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections.append(ws)
        self.buffer_ms = self._burst_start = 0.0  # a new session's input buffer
        self.session_bursts = 0
        self.open_calls = {}  # a new session holds no call
        await self._send(ws, {"type": "session.created", "session": {}})
        async for frame in ws:
            event = json.loads(frame)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                await self._on_session_update(ws, event)
                if self.unrequested and len(self.of("session.update")) == 1:
                    await self._send(ws, {"type": "response.created", "response": {"id": "x"}})
            elif kind == "conversation.item.create":
                item = event.get("item", {})
                if item.get("type") == "function_call_output" and self.reject_outputs:
                    error = {"code": "invalid_value", "event_id": event.get("event_id")}
                    await self._send(ws, {"type": "error", "error": error})
                elif item.get("type") == "function_call_output":
                    self.open_calls.pop(item.get("call_id"), None)
            elif kind == "conversation.item.delete":
                gone = event.get("item_id")
                self.open_calls = {c: i for c, i in self.open_calls.items() if i != gone}
            elif kind == "response.create" and self.strict_calls and self.open_calls:
                self.pending_refusals += 1
                error = {"code": PENDING, "event_id": event.get("event_id")}
                await self._send(ws, {"type": "error", "error": error})
            elif kind == "response.create":
                script = self.scripts[min(self.responses, len(self.scripts) - 1)]
                self.responses += 1
                if script == "close_before":
                    await ws.close()
                    return
                self._cancel = asyncio.Event()
                task = asyncio.ensure_future(
                    self._respond(ws, script, self._cancel, event.get("event_id"))
                )
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif kind == "response.cancel" and self._cancel is not None:
                self._cancel.set()
            elif kind == "input_audio_buffer.append":
                self.buffer_ms += len(base64.b64decode(event["audio"])) / 2 / RATE * 1000
            elif kind == "input_audio_buffer.commit":
                texts = self.heard[min(self.bursts, len(self.heard) - 1)]
                delay = self.speech_delay_s[min(self.bursts, len(self.speech_delay_s) - 1)]
                self.bursts += 1
                self.session_bursts += 1
                number = self.session_bursts if self.ids_per_session else self.bursts
                start = self._burst_start
                end = max(start, self.buffer_ms - self.trim_ms)  # where its speech ends
                self.buffer_ms = self._burst_start = end
                task = asyncio.ensure_future(self._hear(ws, number, texts, delay, start, end))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

    async def _hear(
        self,
        ws: ServerConnection,
        burst: int,
        texts: list[str],
        delay: float,
        start: float,
        end: float,
    ) -> None:
        items = [f"in_{burst}_{i}" for i in range(len(texts))]
        if delay:
            await asyncio.sleep(delay)
        await self._held("speech")
        events = []
        for i, item in enumerate(items):
            started: dict[str, Any] = {"type": "input_audio_buffer.speech_started"}
            stopped: dict[str, Any] = {"type": "input_audio_buffer.speech_stopped"}
            done: dict[str, Any] = {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": texts[i],
            }
            if self.ids:
                started["item_id"] = stopped["item_id"] = done["item_id"] = item
            if self.positions:
                started["audio_start_ms"] = int(start + 10 * i)
                if self.end_positions:
                    last = i == len(items) - 1
                    stopped["audio_end_ms"] = int(end) if last else int(start + 10 * i + 5)
            events.append((started, stopped, done))
        if self.segment_gap_s:  # one segment at a time, each with its transcript
            for i, (started, stopped, done) in enumerate(events):
                if i:
                    await asyncio.sleep(self.segment_gap_s)
                    await self._held("segment")
                await self._send(ws, started)
                await self._send(ws, stopped)
                if self.transcript_delay_s:
                    await asyncio.sleep(self.transcript_delay_s)
                await self._held("transcript")
                await self._send(ws, done)
            return
        for started, stopped, _ in events:
            await self._send(ws, started)
            await self._send(ws, stopped)
        if self.vad_answers and items:  # its VAD ended the turn: it answers by itself
            await self._send(ws, {"type": "response.created", "response": {"id": f"auto_{burst}"}})
        if self.transcript_delay_s:
            await asyncio.sleep(self.transcript_delay_s)
        await self._held("transcript")
        for _, _, done in events:
            await self._send(ws, done)

    async def _held(self, point: str) -> None:
        """Wait at ``point`` while ``hold`` is unset (at most ``hold_max_s``)."""
        if self.hold is None or self.hold_at != point:
            return
        loop = asyncio.get_running_loop()
        end = loop.time() + self.hold_max_s
        self.holding += 1
        try:
            while not self.hold.is_set():
                if loop.time() >= end:
                    self.hold_timeouts += 1
                    return
                await asyncio.sleep(HOLD_POLL_S)
        finally:
            self.holding -= 1

    def __exit__(self, *exc: object) -> None:
        """Close the server; a hold nobody released (timed out, or still waiting) fails."""
        # read before the close ends the waiting tasks; a set hold's waiters are on their way
        holding = self.holding if self.hold is not None and not self.hold.is_set() else 0
        super().__exit__(*exc)
        if self.hold_timeouts or holding:
            raise AssertionError(
                f"TurnServer hold at {self.hold_at!r} never released: "
                f"{self.hold_timeouts} timed out, {holding} still waiting"
            )

    async def _respond(
        self, ws: ServerConnection, script: str, cancel: asyncio.Event, event_id: str | None
    ) -> None:
        rid = f"resp_{self.responses}"
        item = f"item_{self.responses}"
        if script == "silent":
            return
        if script in ("updated_then_refused", "unrelated_error"):
            session = {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}
            await self._send(ws, {"type": "session.updated", "session": session})
            about = event_id if script == "updated_then_refused" else "ev_someone_else"
            await self._send(ws, {"type": "error", "error": {"code": "x", "event_id": about}})
            return
        if script in ("error", "pending"):  # refused before it began
            code = PENDING if script == "pending" else "refused"
            error = {"code": code, "event_id": event_id}  # echoes the request
            await self._send(ws, {"type": "error", "error": error})
            return
        await self._send(ws, {"type": "response.created", "response": {"id": rid}})
        if script == "empty":  # a response with no audio
            await self._done(ws, rid, "completed")
            return
        if script == "tool":
            await self._call(ws, rid)
            await self._done(ws, rid, "completed")
            return
        if script in ("tools", "tools_then_audio"):
            for i, name in enumerate(self.tool_names or [self.tool_name]):
                await self._call(ws, rid, name, i)
        if script == "tools":
            await self._done(ws, rid, "completed")
            return
        if script == "tool_close":
            await self._call(ws, rid)
            await ws.close()
            return
        if script == "tool_stall":
            await self._call(ws, rid)
            await asyncio.sleep(self.stall_s)
            await self._done(ws, rid, "cancelled" if cancel.is_set() else "completed")
            return
        if script == "tool_then_audio":
            await self._call(ws, rid)
        delta = base64.b64encode(pcm(self.delta_ms)).decode()
        for i in range(1 if script == "close_mid" else self.deltas):
            if cancel.is_set():
                if script == "cancel_then_tool":
                    await self._call(ws, rid)
                await self._done(ws, rid, "cancelled")
                return
            await self._send(
                ws,
                {
                    "type": "response.output_audio.delta",
                    "response_id": rid,
                    "item_id": item,
                    "delta": delta,
                },
            )
            if i == 0:
                await self._send(
                    ws,
                    {
                        "type": "response.output_audio_transcript.done",
                        "response_id": rid,
                        "item_id": item,
                        "transcript": self.transcript,
                    },
                )
            if self.gap_s:
                await asyncio.sleep(self.gap_s)
        if script == "close_mid":
            await ws.close()
            return
        if script == "audio_then_tool":
            await self._call(ws, rid)
        await self._send(
            ws, {"type": "response.output_audio.done", "response_id": rid, "item_id": item}
        )
        await self._done(ws, rid, "cancelled" if cancel.is_set() else "completed")

    async def _call(
        self, ws: ServerConnection, rid: str, name: str | None = None, index: int = 0
    ) -> None:
        if name is None and self.tool_names:
            name = self.tool_names[min(self.responses - 1, len(self.tool_names) - 1)]
        suffix = f"_{index}" if index else ""
        call_id = f"call_{self.responses}{suffix}"
        self.open_calls[call_id] = f"fc_{self.responses}{suffix}"
        await self._send(
            ws,
            {
                "type": "response.function_call_arguments.done",
                "response_id": rid,
                "item_id": f"fc_{self.responses}{suffix}",
                "call_id": call_id,
                "name": name or self.tool_name,
                "arguments": self.call_args.get(call_id, self.tool_args),
            },
        )

    async def _done(self, ws: ServerConnection, rid: str, status: str) -> None:
        response: dict[str, Any] = {"id": rid, "status": status}
        if self.usage is not None:
            response["usage"] = {"total_tokens": self.usage}
            response["metadata"] = {"speech_to_speech.turn_latency": {"stt": 0.1, "e2e": 0.9}}
        await self._send(ws, {"type": "response.done", "response": response})
