"""A scripted realtime server for tests (SPEC 8, Realtime speech route): ``websockets.serve`` on
``127.0.0.1:0`` speaking the speech-to-speech subset, on its own thread and loop.

``mode`` picks the script:

- ``normal``: ``session.created`` on connect, ``session.updated`` after ``session.update``, and
  for ``response.create``: ``response.created``, one audio delta, the reply transcript and
  ``response.done``;
- ``error_first``: an ``error`` (its message quotes the request, its ``event_id`` names the
  ``session.update``) instead of ``session.updated``;
- ``no_session``: never answers ``session.update``;
- ``stall``: ``response.created`` then nothing; ``silent``: nothing after ``response.create``;
- ``unknown_only``: after ``response.create``, an unknown event every 0.05 s for 1 s;
- ``close_mid``: ``response.created``, one audio delta, then the socket is closed;
- ``oversize``: after ``session.updated``, one message of ``oversize_bytes``;
- ``stop_reading``: after ``session.updated`` the server stops reading the socket;
- ``redirect``: the handshake is answered with 302 to ``redirect_to``.

``delete_reply`` answers ``conversation.item.delete``: ``none`` (the default: nothing),
``deleted`` (``conversation.item.deleted`` naming the item) or ``error`` (an ``error`` echoing
the delete's ``event_id``, its message quoting the request).

Every handshake is recorded (path with its query, headers) in ``requests``; client events in
``received``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from http import HTTPStatus
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

PCM = b"\x01\x00\x02\x00\x03\x00\x04\x00"
REPLY_TEXT = "reply words"
ERROR_MESSAGE = "you said: the secret request text"
CLOSE_TIMEOUT_S = 0.5
UNKNOWN_EVENTS = 20
UNKNOWN_GAP_S = 0.05
# The fake's own wire log (it sees the key and the text it is sent) stays out of caplog, so the
# leak tests check what the client logs.
SERVER_LOG = logging.getLogger("tests.realtime.fake_server")
SERVER_LOG.setLevel(logging.CRITICAL + 1)
SERVER_LOG.propagate = False


class FakeServer:
    def __init__(
        self,
        mode: str = "normal",
        *,
        host: str = "127.0.0.1",
        redirect_to: str = "",
        oversize_bytes: int = 0,
        delete_reply: str = "none",
    ) -> None:
        self.mode, self.host, self.redirect_to = mode, host, redirect_to
        self.delete_reply = delete_reply
        self.oversize_bytes = oversize_bytes
        self.requests: list[dict[str, Any]] = []
        self.received: list[dict[str, Any]] = []
        self.port = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.connections: list[ServerConnection] = []

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def http_base(self, path: str = "/v1", query: str = "") -> str:
        return f"http://{self.host}:{self.port}{path}" + (f"?{query}" if query else "")

    def __enter__(self) -> FakeServer:
        self._thread.start()
        assert self._ready.wait(5)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(5)

    def _run(self) -> None:
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(
            self._handler,
            self.host,
            0,
            process_request=self._process_request,
            max_size=None,
            max_queue=1,
            compression=None,
            close_timeout=CLOSE_TIMEOUT_S,
            logger=SERVER_LOG,
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def _process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.requests.append({"path": request.path, "headers": dict(request.headers.raw_items())})
        if self.mode == "redirect":
            headers = Headers({"Location": self.redirect_to})
            return Response(HTTPStatus.FOUND, "Found", headers, b"")
        return None

    async def _send(self, ws: ServerConnection, event: dict[str, Any]) -> None:
        await ws.send(json.dumps(event))

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections.append(ws)
        await self._send(ws, {"type": "session.created", "session": {}})
        async for frame in ws:
            event = json.loads(frame)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                await self._on_session_update(ws, event)
                if self.mode == "stop_reading":
                    ws.transport.pause_reading()
                    await ws.wait_closed()
                    return
            elif kind == "response.create":
                await self._on_response(ws)
            elif kind == "conversation.item.delete":
                await self._on_delete(ws, event)

    async def _on_delete(self, ws: ServerConnection, event: dict[str, Any]) -> None:
        if self.delete_reply == "deleted":
            await self._send(ws, {"type": "conversation.item.deleted", "item_id": event["item_id"]})
        elif self.delete_reply == "error":
            error = {"code": "item_not_found", "message": ERROR_MESSAGE}
            error["event_id"] = event["event_id"]
            await self._send(ws, {"type": "error", "error": error})

    async def _on_session_update(self, ws: ServerConnection, event: dict[str, Any]) -> None:
        if self.mode == "no_session":
            return
        if self.mode == "error_first":
            await self._send(
                ws,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "bad_session",
                        "message": ERROR_MESSAGE,
                        "event_id": event.get("event_id"),
                    },
                },
            )
            return
        await self._send(
            ws,
            {
                "type": "session.updated",
                "session": {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}},
            },
        )
        if self.mode == "oversize":
            await ws.send(json.dumps({"type": "x.big", "pad": "a" * self.oversize_bytes}))

    async def _on_response(self, ws: ServerConnection) -> None:
        if self.mode == "silent":
            return
        if self.mode == "unknown_only":
            for _ in range(UNKNOWN_EVENTS):
                await self._send(ws, {"type": "rate_limits.updated", "rate_limits": []})
                await asyncio.sleep(UNKNOWN_GAP_S)
            return
        await self._send(ws, {"type": "response.created", "response": {"id": "resp_1"}})
        if self.mode == "stall":
            return
        delta = base64.b64encode(PCM).decode()
        await self._send(
            ws,
            {
                "type": "response.output_audio.delta",
                "response_id": "resp_1",
                "item_id": "item_1",
                "delta": delta,
            },
        )
        if self.mode == "close_mid":
            await ws.close()
            return
        await self._send(
            ws,
            {
                "type": "response.output_audio_transcript.done",
                "response_id": "resp_1",
                "item_id": "item_1",
                "transcript": REPLY_TEXT,
            },
        )
        await self._send(
            ws,
            {
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {"total_tokens": 42},
                    "metadata": {"speech_to_speech.turn_latency": {"stt": 0.2, "e2e": 1.1}},
                },
            },
        )
