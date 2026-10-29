"""A fake realtime server for whole turns (SPEC 8, Realtime speech route): the SR-1 fake, plus
responses scripted one per ``response.create``, served on their own task so a
``response.cancel`` is read while a reply streams.

Scripts (``scripts``, used in order; the last one repeats):

- ``reply``: ``response.created``, ``deltas`` audio deltas of ``delta_ms`` each ``gap_s`` apart,
  the reply transcript (``transcript``), ``response.output_audio.done`` and ``response.done``
  with usage ``usage`` (None leaves usage out);
- ``tool``: ``response.created``, one ``response.function_call_arguments.done`` (``tool_name``,
  ``tool_args``) and ``response.done``;
- ``close_before``: the socket is closed on ``response.create``;
- ``close_mid``: ``response.created``, one audio delta, then the socket is closed;
- ``error``: an ``error`` instead of ``response.created``;
- ``updated_then_refused``: ``session.updated``, then the ``error`` that echoes the request;
- ``unrelated_error``: ``session.updated``, then an ``error`` about another event;
- ``empty``: ``response.created`` and ``response.done``, no audio;
- ``silent``: nothing at all.

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
repeats across sessions. ``sent`` records every event the server sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import numpy as np
from fake_server import FakeServer
from websockets.asyncio.server import ServerConnection

RATE = 16000
REPLY = "Ecco la mia risposta per te"


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
        **kw: Any,
    ) -> None:
        super().__init__("normal", **kw)
        self.scripts = list(scripts or ["reply"])
        self.deltas, self.delta_ms, self.gap_s = deltas, delta_ms, gap_s
        self.transcript, self.usage = transcript, usage
        self.tool_name, self.tool_args = tool_name, tool_args
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
        await self._send(ws, {"type": "session.created", "session": {}})
        async for frame in ws:
            event = json.loads(frame)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                await self._on_session_update(ws, event)
                if self.unrequested and len(self.of("session.update")) == 1:
                    await self._send(ws, {"type": "response.created", "response": {"id": "x"}})
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
                await self._send(ws, started)
                await self._send(ws, stopped)
                if self.transcript_delay_s:
                    await asyncio.sleep(self.transcript_delay_s)
                await self._send(ws, done)
            return
        for started, stopped, _ in events:
            await self._send(ws, started)
            await self._send(ws, stopped)
        if self.vad_answers and items:  # its VAD ended the turn: it answers by itself
            await self._send(ws, {"type": "response.created", "response": {"id": f"auto_{burst}"}})
        if self.transcript_delay_s:
            await asyncio.sleep(self.transcript_delay_s)
        for _, _, done in events:
            await self._send(ws, done)

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
        if script == "error":  # refused before it began
            error = {"code": "refused", "event_id": event_id}  # echoes the request
            await self._send(ws, {"type": "error", "error": error})
            return
        await self._send(ws, {"type": "response.created", "response": {"id": rid}})
        if script == "empty":  # a response with no audio
            await self._done(ws, rid, "completed")
            return
        if script == "tool":
            await self._send(
                ws,
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": rid,
                    "item_id": f"fc_{self.responses}",
                    "call_id": f"call_{self.responses}",
                    "name": self.tool_name,
                    "arguments": self.tool_args,
                },
            )
            await self._done(ws, rid, "completed")
            return
        delta = base64.b64encode(pcm(self.delta_ms)).decode()
        for i in range(1 if script == "close_mid" else self.deltas):
            if cancel.is_set():
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
        await self._send(
            ws, {"type": "response.output_audio.done", "response_id": rid, "item_id": item}
        )
        await self._done(ws, rid, "cancelled" if cancel.is_set() else "completed")

    async def _done(self, ws: ServerConnection, rid: str, status: str) -> None:
        response: dict[str, Any] = {"id": rid, "status": status}
        if self.usage is not None:
            response["usage"] = {"total_tokens": self.usage}
            response["metadata"] = {"speech_to_speech.turn_latency": {"stt": 0.1, "e2e": 0.9}}
        await self._send(ws, {"type": "response.done", "response": response})
