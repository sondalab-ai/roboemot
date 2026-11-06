"""A fake realtime server for whole turns (SPEC 8, Realtime speech route): the SR-1 fake, plus
responses scripted one per ``response.create``, served on their own task so a
``response.cancel`` is read while a reply streams.

Scripts (``scripts``, used in order; the last one repeats):

- ``reply``: ``response.created``, ``deltas`` audio deltas (a list: per response, the last
  repeats) of ``delta_ms`` each ``gap_s`` apart, the reply transcript (``transcript``),
  ``response.output_audio.done`` and ``response.done`` with usage ``usage`` (None leaves usage
  out);
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
- ``created_close``: ``response.created``, then the socket is closed (no audio);
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
A ``response.cancel`` stops the reply it names (``response_id``), else the latest one, with
``response.done`` ``cancelled``.

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
``tool_args`` for one call. ``statuses`` (when set) is the ``response.done`` status of each
response in turn that would end ``completed`` (the last repeats), for example ``incomplete``
(a token cap). ``sent`` records every event the server sent.

``stream_vad`` makes the VAD listen to the appended audio itself, as for live streams (turns:
server): a turn starts at the first loud block and ends after the session's
``silence_duration_ms`` of quiet, with no commit from the robot (``_vad_feed``); its transcript
is the turn's ``heard`` list, joined. With ``vad_answers`` it also answers each turn it ended
by itself, right after ``committed``, with the next script (as a server with
``create_response: true`` does, or one that ignores ``false``; ``auto_responses`` lists their
ids), before or after the transcript (``transcript_delay_s``); ``answer_delay_s`` holds each
of those answers back that long. ``end_positions`` off leaves its ``audio_end_ms`` out too.
``refuse_cancel_id`` answers a ``response.cancel`` that names a ``response_id`` with an
``error`` ``unknown_or_invalid_event`` (no event id) and cancels nothing, as Pollen's server
answers unexpected values; ``ignore_cancel`` ignores every cancel (the reply streams on).

With the last ``session.update``'s ``interrupt_response`` true (server turns, S2) the stream VAD
cancels the responses still running when it starts a turn, as a server does on its own: each
ends ``response.done`` ``cancelled`` at its next delta (``interrupted`` lists their ids).
``interrupt_first`` waits for those ``response.done`` before it sends the ``speech_started``
(the server's cancel before the start); by default the start goes first. ``late_deltas``
audio deltas still go after a cancel, before its ``response.done`` (a cancel that lands late).

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
VAD_BLOCK_MS = 10  # the stream VAD's block
VAD_LEVEL = 1000  # a block with a sample this loud (PCM16) is speech


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
        deltas: int | list[int] = 3,
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
        statuses: list[str] | None = None,
        stream_vad: bool = False,
        answer_delay_s: float = 0.0,
        refuse_cancel_id: bool = False,
        ignore_cancel: bool = False,
        interrupt_first: bool = False,
        late_deltas: int = 0,
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
        self.statuses = statuses
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
        self.stream_vad = stream_vad
        self.answer_delay_s = answer_delay_s
        self.refuse_cancel_id, self.ignore_cancel = refuse_cancel_id, ignore_cancel
        self.interrupt_first, self.late_deltas = interrupt_first, late_deltas
        self.interrupted: list[str] = []  # responses its VAD cancelled (interrupt_response)
        self._running: dict[str, asyncio.Event] = {}  # response id -> set at its done
        self.stream_turns = 0  # turns the stream VAD heard, over all sessions
        self.commits_at_create: list[int] = []  # the stream VAD's commits at each response.create
        self._vad_item: str | None = None  # the stream VAD's turn in progress
        self._quiet_ms = 0.0
        self._speech_end = 0.0
        self.buffer_ms = 0.0  # audio appended in this session so far
        self.bursts = 0
        self._burst_start = 0.0
        self.responses = 0
        self.auto_responses: list[str] = []  # the ids of the responses its VAD created
        self._cancel: asyncio.Event | None = None
        self._cancels: dict[str, asyncio.Event] = {}  # response id -> its cancel
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
        self._vad_item, self._quiet_ms = None, 0.0  # nor a turn of the old one
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
                self.commits_at_create.append(len(self.sent_of("input_audio_buffer.committed")))
                if not await self._start(ws, event.get("event_id")):
                    return
            elif kind == "response.cancel" and self.ignore_cancel:
                pass
            elif kind == "response.cancel" and self.refuse_cancel_id and "response_id" in event:
                refused = {"code": "unknown_or_invalid_event"}
                await self._send(ws, {"type": "error", "error": refused})
            elif kind == "response.cancel" and event.get("response_id") in self._cancels:
                self._cancels[event["response_id"]].set()
            elif kind == "response.cancel" and self._cancel is not None:
                self._cancel.set()
            elif kind == "input_audio_buffer.append" and self.stream_vad:
                await self._vad_feed(ws, base64.b64decode(event["audio"]))
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

    async def _start(
        self, ws: ServerConnection, event_id: str | None, delay_s: float = 0.0
    ) -> bool:
        """Start the next script's response on its own task; False when it closed the
        socket (``close_before``)."""
        script = self.scripts[min(self.responses, len(self.scripts) - 1)]
        self.responses += 1
        if script == "close_before":
            await ws.close()
            return False
        self._cancel = self._cancels[f"resp_{self.responses}"] = asyncio.Event()
        task = asyncio.ensure_future(
            self._respond(ws, script, self._cancel, event_id, self.responses, delay_s)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

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

    async def _vad_feed(self, ws: ServerConnection, data: bytes) -> None:
        """The stream VAD (``stream_vad``): appended audio in blocks of ``VAD_BLOCK_MS``, by
        sample count, never wall time. The first loud block starts a turn (``speech_started``
        at its position); ``silence_duration_ms`` of quiet blocks (the last ``session.update``)
        ends it (``speech_stopped`` where the quiet began, ``committed``), then its transcript
        (``heard``, one list per turn, joined) after ``transcript_delay_s`` and the hold."""
        samples = np.frombuffer(data, "<i2")
        block = round(RATE * VAD_BLOCK_MS / 1000)
        silence_ms = self._silence_ms()
        for i in range(0, samples.size, block):
            part = samples[i : i + block]
            loud = part.size and int(np.max(np.abs(part.astype(np.int32)))) >= VAD_LEVEL
            pos = self.buffer_ms
            self.buffer_ms += part.size / RATE * 1000
            if loud:
                self._quiet_ms = 0.0
                if self._vad_item is None:
                    self.stream_turns += 1
                    self._vad_item = f"in_{self.stream_turns}_0"
                    started: dict[str, Any] = {"type": "input_audio_buffer.speech_started"}
                    if self.ids:
                        started["item_id"] = self._vad_item
                    started["audio_start_ms"] = int(pos)
                    interrupts = self._interrupts()
                    if interrupts and self.interrupt_first:
                        await self._interrupt(wait=True)
                    await self._send(ws, started)
                    if interrupts and not self.interrupt_first:
                        await self._interrupt(wait=False)
                self._speech_end = self.buffer_ms
                continue
            if self._vad_item is None:
                continue
            self._quiet_ms += part.size / RATE * 1000
            if self._quiet_ms >= silence_ms:
                item, self._vad_item = self._vad_item, None
                stopped: dict[str, Any] = {"type": "input_audio_buffer.speech_stopped"}
                committed: dict[str, Any] = {"type": "input_audio_buffer.committed"}
                if self.ids:
                    stopped["item_id"] = committed["item_id"] = item
                if self.end_positions:
                    stopped["audio_end_ms"] = int(self._speech_end)
                await self._send(ws, stopped)
                await self._send(ws, committed)
                if self.vad_answers:  # it answers the turn it ended by itself
                    self.auto_responses.append(f"resp_{self.responses + 1}")
                    if not await self._start(ws, None, self.answer_delay_s):
                        return
                texts = self.heard[min(self.stream_turns - 1, len(self.heard) - 1)]
                task = asyncio.ensure_future(self._vad_transcript(ws, item, " ".join(texts)))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        if self._vad_item is None:  # a later committed burst starts after this audio
            self._burst_start = self.buffer_ms

    def _interrupts(self) -> bool:
        updates = self.of("session.update")
        if not updates:
            return False
        detect = updates[-1]["session"]["audio"]["input"]["turn_detection"]
        return bool(detect.get("interrupt_response"))

    async def _interrupt(self, *, wait: bool) -> None:
        """Cancel every response still running (its done ``cancelled`` at its next delta);
        ``wait``: until each one sent its ``response.done``."""
        running = [rid for rid, done in self._running.items() if not done.is_set()]
        for rid in running:
            self.interrupted.append(rid)
            self._cancels[rid].set()
        if wait:
            for rid in running:
                await asyncio.wait_for(self._running[rid].wait(), HOLD_MAX_S)

    def _silence_ms(self) -> float:
        updates = self.of("session.update")
        if not updates:
            return 600.0
        detect = updates[-1]["session"]["audio"]["input"]["turn_detection"]
        return float(detect["silence_duration_ms"])

    async def _vad_transcript(self, ws: ServerConnection, item: str, text: str) -> None:
        if self.transcript_delay_s:
            await asyncio.sleep(self.transcript_delay_s)
        await self._held("transcript")
        done: dict[str, Any] = {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text,
        }
        if self.ids:
            done["item_id"] = item
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
        self,
        ws: ServerConnection,
        script: str,
        cancel: asyncio.Event,
        event_id: str | None,
        number: int,
        delay_s: float = 0.0,
    ) -> None:
        if delay_s:
            await asyncio.sleep(delay_s)
        rid = f"resp_{number}"
        item = f"item_{number}"
        self._running[rid] = asyncio.Event()
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
        if script == "created_close":
            await ws.close()
            return
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
        count = (
            self.deltas
            if isinstance(self.deltas, int)
            else self.deltas[min(number, len(self.deltas)) - 1]
        )
        for i in range(1 if script == "close_mid" else count):
            if cancel.is_set():
                if script == "cancel_then_tool":
                    await self._call(ws, rid)
                for _ in range(self.late_deltas):
                    late = {"type": "response.output_audio.delta", "response_id": rid}
                    await self._send(ws, {**late, "item_id": item, "delta": delta})
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
        if status == "completed" and self.statuses:
            status = self.statuses[min(self.responses - 1, len(self.statuses) - 1)]
        response: dict[str, Any] = {"id": rid, "status": status}
        if self.usage is not None:
            response["usage"] = {"total_tokens": self.usage}
            response["metadata"] = {"speech_to_speech.turn_latency": {"stt": 0.1, "e2e": 0.9}}
        await self._send(ws, {"type": "response.done", "response": response})
        if rid in self._running:  # after the send: an interrupt that waits sends after it
            self._running[rid].set()
