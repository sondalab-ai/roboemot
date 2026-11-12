"""A scripted Gemini Live server for tests (SPEC 8, Gemini Live wire): ``websockets.serve`` over
TLS on ``127.0.0.1:0`` (a ``gemini_live`` route takes ``wss`` only; ``tls`` makes a self-signed
certificate the client trusts through ``SSL_CERT_FILE``), on its own thread and loop, speaking
the message shapes the S0 probe saw on ``gemini-3.8-live`` (2026-11-11), every one JSON in a
binary frame.

- The first message must be ``setup``: answered ``setupComplete`` and an unasked
  ``sessionResumptionUpdate``; with ``refuse`` > 0 the connect is closed instead with 1008
  "The caller does not have permission" (S0 saw it once), ``refuse`` times.
- ``realtimeInput.activityStart``: ``voiceActivity`` ``ACTIVITY_START`` at the session's audio
  clock (``audioOffset``: the 16 kHz PCM received so far, as a protobuf Duration); a reply still
  streaming is cut (``interrupted``), as Gemini does on new input.
- ``activityEnd``: the next ``heard`` transcript (the last repeats) as ``inputTranscription``,
  then ``ACTIVITY_END``, then the next answer.
- ``clientContent`` with ``turnComplete`` true is a text turn: the next answer.
- An answer (``answers``, used in order, the last repeats) is a list of steps:
  ``call:<name>[,<name>]`` sends one ``toolCall`` (``args`` per name from ``call_args``) and waits
  for the ``toolResponse``; ``audio`` sends ``chunks`` ``modelTurn`` audio parts at 24 kHz, each
  ``chunk_ms`` long and ``gap_s`` apart, the first with ``outputTranscription`` ``reply``;
  ``interrupt`` sends ``interrupted`` on its own (Gemini cutting its answer) and ends the turn;
  ``go_away`` sends ``goAway``. After the steps: ``generationComplete``, then after
  ``turn_complete_s`` ``turnComplete`` with ``usageMetadata`` ``usage``.
- ``delay_s`` holds each answer back that long.
- ``idle_close_s``: a connection with no client message for that long is closed with
  ``close_code`` (1008), as Gemini closes a session idle for about two minutes (S3 run
  2026-11-12).
- ``usage_gate`` (a ``threading.Event``) holds every ``turnComplete`` until it is set.

``setups`` holds every ``setup`` (one per connection), ``received`` every client message,
``requests`` every handshake (path with its query, headers), ``tool_responses`` every
``toolResponse``'s function responses. Nothing is logged.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import ssl
import subprocess
import threading
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request

INPUT_RATE_HZ = 16000
OUTPUT_RATE_HZ = 24000
REPLY = "Ciao! Sto benissimo, grazie."
HEARD = "Ciao Neve, come stai oggi?"
REFUSAL = "The caller does not have permission"
IDLE = "idle"
POLICY = 1008
SAMPLE_PAIR = b"\x10\x00\x20\x00"  # two PCM16 samples
SERVER_LOG = logging.getLogger("tests.realtime.gemini_server")
SERVER_LOG.setLevel(logging.CRITICAL + 1)
SERVER_LOG.propagate = False
TOOL_WAIT_S = 5.0
GATE_POLL_S = 0.01
CLOSE_TIMEOUT_S = 0.5


def tls(directory: Path) -> tuple[Path, Path]:
    """A self-signed certificate (and its key) for 127.0.0.1 in ``directory``."""
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "ec",
            "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
            "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
            "-days", "2", "-keyout", str(key), "-out", str(cert),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return cert, key


def duration(ms: float) -> str:
    """Milliseconds as a protobuf Duration, the way Gemini writes ``audioOffset``."""
    return f"{ms / 1000:.6f}".rstrip("0").rstrip(".") + "s"


class GeminiServer:
    def __init__(
        self,
        tls_files: tuple[Path, Path],
        answers: list[list[str]] | None = None,
        *,
        heard: list[str] | None = None,
        reply: str = REPLY,
        chunks: int = 3,
        chunk_ms: int = 10,
        gap_s: float = 0.0,
        usage: int = 1785,
        turn_complete_s: float = 0.2,
        call_args: dict[str, dict[str, Any]] | None = None,
        refuse: int = 0,
        delay_s: float = 0.0,
        time_left: str = "5s",
        idle_close_s: float | None = None,
        close_code: int = POLICY,
        usage_gate: threading.Event | None = None,
    ) -> None:
        self.tls_files = tls_files
        self.answers = answers or [["audio"]]
        self.heard = heard or [HEARD]
        self.reply, self.chunks, self.gap_s = reply, chunks, gap_s
        self.chunk = SAMPLE_PAIR * (OUTPUT_RATE_HZ * chunk_ms // 1000 // 2)
        self.usage, self.turn_complete_s = usage, turn_complete_s
        self.call_args = call_args or {}
        self.refuse, self.delay_s, self.time_left = refuse, delay_s, time_left
        self.idle_close_s, self.close_code = idle_close_s, close_code
        self.usage_gate = usage_gate
        self.idle_closes = 0
        self.requests: list[dict[str, Any]] = []
        self.received: list[dict[str, Any]] = []
        self.setups: list[dict[str, Any]] = []
        self.tool_responses: list[list[dict[str, Any]]] = []
        self.closes = 0  # connects refused with 1008
        self.connections = 0
        self.port = 0
        self._turns = 0  # answers started (all connections)
        self._bursts = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def base_url(self, path: str = "/ws/BidiGenerateContent") -> str:
        return f"wss://127.0.0.1:{self.port}{path}"

    def of(self, key: str) -> list[dict[str, Any]]:
        """The client messages that carry ``key`` (``realtimeInput`` parts count by their own
        key: ``activityStart``, ``audio``, ``activityEnd``)."""
        out = []
        for msg in self.received:
            if key in msg or key in msg.get("realtimeInput", {}):
                out.append(msg)
        return out

    def __enter__(self) -> GeminiServer:
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
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*self.tls_files)
        async with serve(
            self._handler,
            "127.0.0.1",
            0,
            ssl=context,
            process_request=self._process_request,
            max_size=None,
            compression=None,
            close_timeout=CLOSE_TIMEOUT_S,
            logger=SERVER_LOG,
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def _process_request(self, conn: ServerConnection, request: Request) -> None:
        self.requests.append({"path": request.path, "headers": dict(request.headers.raw_items())})

    async def _handler(self, ws: ServerConnection) -> None:
        self.connections += 1
        session = _Session(self, ws)
        try:
            await session.serve()
        except ConnectionClosed:
            pass
        finally:
            session.stop()


class _Session:
    """One connection: its audio clock, the answer streaming and the tool wait."""

    def __init__(self, server: GeminiServer, ws: ServerConnection) -> None:
        self.server, self.ws = server, ws
        self.audio_bytes = 0
        self.answer: asyncio.Task[None] | None = None
        self.tool = asyncio.Event()
        self.pending: list[asyncio.Task[None]] = []

    def clock(self) -> str:
        return duration(self.audio_bytes / 2 / INPUT_RATE_HZ * 1000)

    async def send(self, message: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(message).encode())

    async def serve(self) -> None:
        srv = self.server
        first = json.loads(await self.ws.recv())
        srv.received.append(first)
        assert "setup" in first, first
        srv.setups.append(first["setup"])
        if srv.refuse > 0:
            srv.refuse -= 1
            srv.closes += 1
            await self.ws.close(POLICY, REFUSAL)
            return
        await self.send({"setupComplete": {}})
        await self.send({"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}})
        while True:
            try:
                frame = await asyncio.wait_for(self.ws.recv(), srv.idle_close_s)
            except TimeoutError:  # Gemini closes a session idle for about two minutes
                srv.idle_closes += 1
                await self.ws.close(srv.close_code, IDLE)
                return
            msg = json.loads(frame)
            srv.received.append(msg)
            await self.on_message(msg)

    async def on_message(self, msg: dict[str, Any]) -> None:
        srv = self.server
        rt = msg.get("realtimeInput", {})
        if "activityStart" in rt:
            if self.answer is not None and not self.answer.done():
                self.answer.cancel()
                await self.send({"serverContent": {"interrupted": True}})
            await self.send(
                {"voiceActivity": {"type": "ACTIVITY_START", "audioOffset": self.clock()}}
            )
        if "audio" in rt:
            self.audio_bytes += len(base64.b64decode(rt["audio"]["data"]))
        if "activityEnd" in rt:
            heard = srv.heard[min(srv._bursts, len(srv.heard) - 1)]
            srv._bursts += 1
            await self.send({"serverContent": {"inputTranscription": {"text": heard}}})
            await self.send(
                {"voiceActivity": {"type": "ACTIVITY_END", "audioOffset": self.clock()}}
            )
            self.start_answer()
        if msg.get("clientContent", {}).get("turnComplete") is True:
            self.start_answer()
        if "toolResponse" in msg:
            srv.tool_responses.append(msg["toolResponse"]["functionResponses"])
            self.tool.set()

    def start_answer(self) -> None:
        srv = self.server
        steps = srv.answers[min(srv._turns, len(srv.answers) - 1)]
        srv._turns += 1
        self.answer = asyncio.ensure_future(self.run_answer(steps))

    async def run_answer(self, steps: list[str]) -> None:
        srv = self.server
        if srv.delay_s:
            await asyncio.sleep(srv.delay_s)
        for step in steps:
            if step.startswith("call:"):
                await self.call(step.removeprefix("call:").split(","))
            elif step == "audio":
                await self.audio()
            elif step == "go_away":
                await self.send({"goAway": {"timeLeft": srv.time_left}})
            elif step == "interrupt":
                await self.send({"serverContent": {"interrupted": True}})
                await self.turn_complete()
                return
        await self.send({"serverContent": {"generationComplete": True}})
        self.pending.append(asyncio.ensure_future(self.turn_complete(srv.turn_complete_s)))

    async def call(self, names: list[str]) -> None:
        self.tool.clear()
        calls = [
            {"name": n, "args": self.server.call_args.get(n, {}), "id": f"fc_{i}_{n}"}
            for i, n in enumerate(names)
        ]
        await self.send({"toolCall": {"functionCalls": calls}})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.tool.wait(), TOOL_WAIT_S)

    async def audio(self) -> None:
        srv = self.server
        for i in range(srv.chunks):
            part = {"inlineData": {"mimeType": f"audio/pcm;rate={OUTPUT_RATE_HZ}",
                                   "data": base64.b64encode(srv.chunk).decode()}}  # fmt: skip
            content: dict[str, Any] = {"modelTurn": {"parts": [part]}}
            if i == 0:
                content["outputTranscription"] = {"text": srv.reply}
            await self.send({"serverContent": content})
            if srv.gap_s:
                await asyncio.sleep(srv.gap_s)

    async def turn_complete(self, after_s: float = 0.0) -> None:
        if after_s:
            await asyncio.sleep(after_s)
        gate = self.server.usage_gate
        while gate is not None and not gate.is_set():
            await asyncio.sleep(GATE_POLL_S)
        await self.send(
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {"totalTokenCount": self.server.usage},
            }
        )

    def stop(self) -> None:
        for task in (self.answer, *self.pending):
            if task is not None:
                task.cancel()
