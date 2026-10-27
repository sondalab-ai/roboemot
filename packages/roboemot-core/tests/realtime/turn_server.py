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
- ``empty``: ``response.created`` and ``response.done``, no audio;
- ``silent``: nothing at all.

``unrequested`` sends a ``response.created`` nobody asked for right after ``session.updated``.
A ``response.cancel`` stops the reply in progress with ``response.done`` ``cancelled``.
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
        **kw: Any,
    ) -> None:
        super().__init__("normal", **kw)
        self.scripts = list(scripts or ["reply"])
        self.deltas, self.delta_ms, self.gap_s = deltas, delta_ms, gap_s
        self.transcript, self.usage = transcript, usage
        self.tool_name, self.tool_args = tool_name, tool_args
        self.unrequested = unrequested
        self.responses = 0
        self._cancel: asyncio.Event | None = None
        self._tasks: set[asyncio.Task] = set()

    def kinds(self) -> list[str]:
        return [e.get("type") for e in self.received]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.received if e.get("type") == kind]

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections.append(ws)
        await self._send(ws, {"type": "session.created", "session": {}})
        async for frame in ws:
            event = json.loads(frame)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                await self._on_session_update(ws)
                if self.unrequested and len(self.of("session.update")) == 1:
                    await self._send(ws, {"type": "response.created", "response": {"id": "x"}})
            elif kind == "response.create":
                script = self.scripts[min(self.responses, len(self.scripts) - 1)]
                self.responses += 1
                if script == "close_before":
                    await ws.close()
                    return
                self._cancel = asyncio.Event()
                task = asyncio.ensure_future(self._respond(ws, script, self._cancel))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif kind == "response.cancel" and self._cancel is not None:
                self._cancel.set()

    async def _respond(self, ws: ServerConnection, script: str, cancel: asyncio.Event) -> None:
        rid = f"resp_{self.responses}"
        item = f"item_{self.responses}"
        if script == "silent":
            return
        if script == "error":  # refused before it began
            await self._send(ws, {"type": "error", "error": {"code": "refused"}})
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
