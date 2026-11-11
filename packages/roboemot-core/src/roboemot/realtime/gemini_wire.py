"""Gemini Live's wire as the realtime subset the route speaks (SPEC 8, Realtime speech route).

:class:`GeminiWire` is a pure, stateful codec (``wire.Wire``): no socket, no thread, no clock.
One instance serves one session; a reconnect starts a new one. Message shapes are the ones the
S0 probe saw on ``gemini-3.8-live`` (2026-11-11).

Out (``encode``):

- the first ``session.update`` is ``setup`` (manual activity, input and output transcription,
  sliding-window compression, BLOCKING function declarations); later ones only keep changed
  instructions, sent as context text (``clientContent``, ``turnComplete`` false) right before
  the next activity or text turn, never between a ``toolCall`` and its ``toolResponse``;
- a burst's appends are ``realtimeInput.audio`` after an ``activityStart``, its commit
  ``activityEnd``; ``conversation.item.create`` text and ``function_call_output`` are kept
  until the next ``response.create``;
- ``response.create`` releases the held answer, or sends the buffered outputs as one
  ``toolResponse``, or a text turn (``turnComplete`` true: the text items, else the
  response's instructions, a fixed line); its ``response.created`` is answered locally, so a
  response never reaches the route before the route asked for it;
- ``response.cancel`` (Gemini has none) ends the open response locally (``cancelled``) and
  drops the rest of that model turn; item deletes and truncates are local acks marked
  ``emulated``; ``tool_choice`` is dropped.

In (``decode``): ``voiceActivity`` is the speech boundaries (``audioOffset`` is the session's
cumulative input clock, in the route's ms), the input transcript is completed at the
activity's end (or at the answer's first content when it is late), ``modelTurn`` audio and
``outputTranscription`` are one output item per response, ``toolCall`` ends its response with
the calls, ``generationComplete`` ends the response, ``interrupted`` cancels it.

Hold and release: Gemini answers an ``activityEnd`` by itself, while the route asks only once
it has the transcript. So the answer's events are held from the commit until the route's
``response.create`` and released in order as one response; a delete of that turn's item (the
route declined it) or a new burst discards them. Input events are never held. The server's
``ACTIVITY_START`` is the fence between model turns: content before it is the older turn's
(a reply still streaming, or one dropped after a cancel), content after it the new turn's.

Usage: ``turnComplete`` carries ``usageMetadata`` seconds after ``generationComplete`` ended the
response, so each ``response.done`` has ``total_tokens`` None (the route charges its estimate)
and the turn's total comes as a ``UsageReport`` naming the responses it covers.

A BLOCKING call Gemini waits on is never left open: before the next activity or text turn,
and when the route deletes an unanswered call's item, the codec answers it ``skipped`` and
drops what Gemini says after.

Events keep text and audio out of ``repr`` (``protocol``); the codec logs nothing (SPEC 12).
"""

from __future__ import annotations

import base64
import binascii
import json
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import count
from typing import Any

from roboemot.realtime import protocol

INPUT_RATE_HZ = 16000  # Gemini Live's input PCM16 rate (an update that names a rate wins)
OUTPUT_RATE_HZ = 24000  # Gemini Live's reply PCM16 (audio/pcm;rate=24000, S0 2026-11-11)
AUDIO_MIME = "audio/pcm;rate={rate}"
MODEL_PREFIX = "models/"
TOOL_BEHAVIOR = "BLOCKING"  # Gemini waits for the toolResponse (S0 gate 4)
EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}
MS_PER_S = 1000
DURATION_UNIT = "s"  # protobuf Duration as JSON: "1.689375s"
ACTIVITY_START = "ACTIVITY_START"
ACTIVITY_END = "ACTIVITY_END"
CONTEXT_PREFIX = "[Updated instructions]\n"
OUTPUT_KEY = "result"  # a function output that is not a JSON object goes under it
SKIPPED = "skipped"  # the answer to a call the route never answered
RESPONSE_PREFIX = "gresp_"
OUTPUT_ITEM_PREFIX = "gout_"
CALL_ITEM_PREFIX = "gcall_"
INPUT_ITEM_PREFIX = "gitem_"
UNKNOWN_TYPE = "gemini.unknown"
MALFORMED = "malformed"
COUNTERS = (
    "empty_frames",
    "empty_content",
    "resumption_updates",
    "tool_call_cancellations",
    "thought_parts",
    "text_parts",
    "other_parts",
    "unknown_messages",
    "unknown_activity",
    "late_input_transcripts",
    "held_discarded",
    "dropped_after_cancel",
    "unrequested_dropped",
    "calls_skipped",
    "outputs_without_call",
    "tools_update_ignored",
    "tool_choice_dropped",
    "items_unsupported",
    "events_unsupported",
    "commits_without_audio",
    "cancels_idle",
)


class _Bad(Exception):
    """A field of the wrong type (internal: the frame becomes ``Malformed``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Response:
    """One codec response: its id, its one output item, whether audio or text came
    (``started``), the reply text so far, and whether its ``response.done`` went."""

    id: str = field(repr=False)
    item: str = field(repr=False)
    started: bool = False
    output: bool = False
    text: str = field(default="", repr=False)
    done: bool = False


@dataclass
class _Activity:
    """One input activity as the server reports it: its item, the transcript so far, and
    whether it ended and its transcript was completed."""

    item: str = field(repr=False)
    text: str = field(default="", repr=False)
    ended: bool = False
    final: bool = False


@dataclass
class _Hold:
    """An answer held from a commit until the route's ``response.create``: the turn's input
    item, the response the answer opened (None until content came), its events, and whether
    the model turn ended (its ``turnComplete`` came)."""

    item: str = field(repr=False)
    response: _Response | None = None
    events: list[protocol.ServerEvent] = field(default_factory=list, repr=False)
    ended: bool = False


class GeminiWire:
    """See the module docstring. ``model`` is the route's model (``models/`` added when
    missing); ``voice`` (the route's own) wins over the update's; ``max_bytes`` caps a frame
    as the client caps the socket. ``counts`` holds what was ignored, dropped or emulated."""

    def __init__(self, model: str, max_bytes: int, voice: str | None = None) -> None:
        self.model = model if model.startswith(MODEL_PREFIX) else MODEL_PREFIX + model
        self.voice = voice or None
        self.max_bytes = max_bytes
        self.counts: Counter[str] = Counter({name: 0 for name in COUNTERS})
        self._responses = count(1)
        self._items = count(1)
        self._setup = False
        self._input_rate = INPUT_RATE_HZ
        self._context_sent = ""
        self._context: str | None = None
        self._texts: list[str] = []
        self._outputs: dict[str, dict[str, Any]] = {}
        self._calls: dict[str, str] = {}  # call id -> name: calls Gemini waits on
        self._call_items: dict[str, str] = {}  # call item -> call id, while it is open
        self._skipped: set[str] = set()
        self._burst: str | None = None  # the item of the activity being sent
        self._sent: list[str] = []  # items of activities the server has not started yet
        self._heard: _Activity | None = None
        self._hold: _Hold | None = None
        self._open: _Response | None = None
        self._dropping = False
        self._closed: list[str] = []  # responses ended since the last usage report

    def __repr__(self) -> str:
        return f"GeminiWire(holding={self._hold is not None}, dropping={self._dropping})"

    # ----------------------------------------------------------------- encode

    def encode(
        self, event: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[protocol.ServerEvent]]:
        """One client event of the subset as Gemini messages, and the events answered here
        (see the module docstring). The client's ``event_id`` never goes to Gemini. An event
        outside the subset sends nothing (counted)."""
        handlers = {
            protocol.SESSION_UPDATE: self._session_update,
            protocol.AUDIO_APPEND: self._append,
            protocol.AUDIO_COMMIT: self._commit,
            protocol.ITEM_CREATE: self._item_create,
            protocol.ITEM_DELETE: self._item_delete,
            protocol.ITEM_TRUNCATE: self._item_truncate,
            protocol.RESPONSE_CREATE: self._response_create,
            protocol.RESPONSE_CANCEL: self._response_cancel,
        }
        handler = handlers.get(event.get("type", ""))
        if handler is None:
            self.counts["events_unsupported"] += 1
            return [], []
        return handler(event)

    def _session_update(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """The first update is ``setup``; a later one keeps changed instructions as context
        for the next turn (Gemini cannot change a session's setup)."""
        session = _dict(event.get("session"))
        audio = _dict(session.get("audio"))
        rate = _dict(_dict(audio.get("input")).get("format")).get("rate")
        if isinstance(rate, int) and not isinstance(rate, bool) and rate > 0:
            self._input_rate = rate
        instructions = session.get("instructions")
        if not self._setup:
            self._setup = True
            self._context_sent = instructions if isinstance(instructions, str) else ""
            voice = self.voice or _dict(audio.get("output")).get("voice")
            return [self._setup_message(session, voice)], []
        if "tools" in session:
            self.counts["tools_update_ignored"] += 1
        if isinstance(instructions, str):
            self._note_context(instructions)
        return [], []

    def _setup_message(self, session: dict[str, Any], voice: Any) -> dict[str, Any]:
        """The ``setup`` message the S0 probe verified: audio replies, the voice, the system
        instruction, the function declarations (BLOCKING), manual activity, both
        transcriptions and sliding-window context compression."""
        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if isinstance(voice, str) and voice:
            generation["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            }
        setup: dict[str, Any] = {"model": self.model, "generationConfig": generation}
        if self._context_sent:
            setup["systemInstruction"] = {"parts": [{"text": self._context_sent}]}
        tools = session.get("tools")
        declarations = [
            _declaration(t)
            for t in (tools if isinstance(tools, list) else [])
            if isinstance(t, dict) and t.get("type") == "function"
        ]
        if declarations:
            setup["tools"] = [{"functionDeclarations": declarations}]
        setup["realtimeInputConfig"] = {"automaticActivityDetection": {"disabled": True}}
        setup["inputAudioTranscription"] = {}
        setup["outputAudioTranscription"] = {}
        setup["contextWindowCompression"] = {"slidingWindow": {}}
        return {"setup": setup}

    def _note_context(self, instructions: str) -> None:
        """Keep ``instructions`` for the next turn when the session does not hold them yet
        (the latest wins: each update carries the whole instructions)."""
        self._context = None if instructions == self._context_sent else instructions

    def _append(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """Audio of a burst; its first piece starts the activity, after what must go before
        new input (``_before_input``)."""
        frames: list[dict[str, Any]] = []
        local: list[protocol.ServerEvent] = []
        if self._burst is None:
            frames, local = self._before_input()
            turns = self._context_turns()
            if turns:
                frames.append({"clientContent": {"turns": turns, "turnComplete": False}})
            frames.append({"realtimeInput": {"activityStart": {}}})
            self._burst = f"{INPUT_ITEM_PREFIX}{next(self._items)}"
            self._sent.append(self._burst)
        mime = AUDIO_MIME.format(rate=self._input_rate)
        data = event.get("audio", "")
        frames.append({"realtimeInput": {"audio": {"data": data, "mimeType": mime}}})
        return frames, local

    def _commit(self, _event: dict[str, Any]) -> tuple[list[dict], list]:
        """End the activity; Gemini answers it by itself, so its answer is held from now."""
        if self._burst is None:
            self.counts["commits_without_audio"] += 1
            return [], []
        self._hold = _Hold(self._burst)
        self._burst = None
        return [{"realtimeInput": {"activityEnd": {}}}], []

    def _item_create(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """A user text is kept for the next turn; a tool result until the next
        ``response.create`` (one ``toolResponse``)."""
        item = _dict(event.get("item"))
        if item.get("type") == "function_call_output":
            call_id, output = item.get("call_id"), item.get("output", "")
            if isinstance(call_id, str) and call_id in self._calls:
                self._outputs[call_id] = _response_object(output)
            else:
                self.counts["outputs_without_call"] += 1
            return [], []
        content = item.get("content")
        texts = [
            part["text"]
            for part in (content if isinstance(content, list) else [])
            if isinstance(part, dict)
            and part.get("type") == "input_text"
            and isinstance(part.get("text"), str)
        ]
        if item.get("type") == "message" and texts:
            self._texts.append("".join(texts))
        else:
            self.counts["items_unsupported"] += 1
        return [], []

    def _item_delete(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """Acked here (Gemini keeps its own history). The held turn's item discards the
        held answer; an unanswered call's item answers that call ``skipped``."""
        item = event.get("item_id")
        local: list[protocol.ServerEvent] = [
            protocol.ItemDeleted(protocol.ITEM_DELETED, item, emulated=True)
        ]
        frames: list[dict[str, Any]] = []
        if self._hold is not None and item == self._hold.item:
            local += self._discard()
        call_id = self._call_items.pop(item, None) if isinstance(item, str) else None
        if call_id is not None and call_id in self._calls and call_id not in self._outputs:
            self._skipped.add(call_id)
            if all(c in self._outputs or c in self._skipped for c in self._calls):
                frames.append(self._tool_response())
                self._dropping = True  # what Gemini says after is nobody's
        return frames, local

    def _item_truncate(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """Acked here: Gemini's history keeps the whole reply (accepted drift)."""
        item = event.get("item_id")
        return [], [protocol.ItemTruncated(protocol.ITEM_TRUNCATED, item, emulated=True)]

    def _response_create(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """Release the held answer, else answer Gemini's open calls, else send a text turn;
        ``response.created`` is answered here at once. Instructions that cannot reach an
        answer under way are kept as context for the next turn."""
        response = _dict(event.get("response"))
        instructions = response.get("instructions")
        instructions = instructions if isinstance(instructions, str) else None
        if "tool_choice" in response:
            self.counts["tool_choice_dropped"] += 1
        if self._hold is not None:
            hold, self._hold = self._hold, None
            resp = hold.response or self._new_response()
            if not resp.done:
                self._open = resp
            if instructions is not None:
                self._note_context(instructions)
            return [], [protocol.ResponseCreated(protocol.RESPONSE_CREATED, resp.id), *hold.events]
        frames: list[dict[str, Any]] = []
        if self._outputs:
            frames.append(self._tool_response())
            if instructions is not None:
                self._note_context(instructions)
        else:
            if self._calls:  # calls the route never saw (a discarded answer): not this turn's
                frames.append(self._tool_response())
                self._dropping = True
            if instructions is not None and self._texts:
                self._note_context(instructions)
            line = instructions if instructions is not None and not self._texts else None
            turns = self._context_turns()
            if line is not None:
                turns.append(_user(line))
            content: dict[str, Any] = {"turnComplete": True}
            if turns:
                content = {"turns": turns, **content}
            frames.append({"clientContent": content})
        self._open = self._new_response()
        return frames, [protocol.ResponseCreated(protocol.RESPONSE_CREATED, self._open.id)]

    def _response_cancel(self, event: dict[str, Any]) -> tuple[list[dict], list]:
        """End the open response here (Gemini has no cancel) and drop the rest of its model
        turn; with nothing open (or another response named) nothing happens."""
        rid = event.get("response_id")
        resp = self._open
        if resp is None or resp.done or (rid and rid != resp.id):
            self.counts["cancels_idle"] += 1
            return [], []
        self._open = None
        self._dropping = True
        return [], [self._done(resp, "cancelled")]

    def _before_input(self) -> tuple[list[dict], list[protocol.ServerEvent]]:
        """What goes before new user input: an answer never released is discarded, and every
        call Gemini still waits on is answered (the route's output, else ``skipped``)."""
        local = self._discard() if self._hold is not None else []
        frames = [self._tool_response()] if self._calls else []
        if frames:
            self._dropping = True
        return frames, local

    def _discard(self) -> list[protocol.ServerEvent]:
        """Drop the held answer; a usage report it held still goes (without the discarded
        response). Unless its model turn ended, the rest of it is dropped as it comes."""
        hold, self._hold = self._hold, None
        if hold is None:
            return []
        rid = hold.response.id if hold.response is not None else None
        if rid is not None:
            self.counts["held_discarded"] += 1
            if rid in self._closed:
                self._closed.remove(rid)
        if not hold.ended:
            self._dropping = True
        return [
            protocol.UsageReport(
                protocol.USAGE_REPORT,
                tuple(i for i in e.response_ids if i != rid),
                e.total_tokens,
            )
            for e in hold.events
            if isinstance(e, protocol.UsageReport)
        ]

    def _tool_response(self) -> dict[str, Any]:
        """One ``toolResponse`` for every call Gemini waits on, in its order: the route's
        output, else ``skipped``."""
        answers = []
        for call_id, name in self._calls.items():
            result = self._outputs.pop(call_id, None)
            if result is None:
                self.counts["calls_skipped"] += 1
                result = {OUTPUT_KEY: SKIPPED}
            answers.append({"id": call_id, "name": name, "response": result})
        self._calls.clear()
        self._skipped.clear()
        self._call_items.clear()
        return {"toolResponse": {"functionResponses": answers}}

    def _context_turns(self) -> list[dict[str, Any]]:
        """The kept context (changed instructions first, then user texts) as user turns."""
        turns = []
        if self._context is not None:
            turns.append(_user(CONTEXT_PREFIX + self._context))
            self._context_sent, self._context = self._context, None
        turns += [_user(text) for text in self._texts]
        self._texts.clear()
        return turns

    def _new_response(self) -> _Response:
        n = next(self._responses)
        return _Response(f"{RESPONSE_PREFIX}{n}", f"{OUTPUT_ITEM_PREFIX}{n}")

    # ----------------------------------------------------------------- decode

    def decode(self, frame: str | bytes) -> list[protocol.ServerEvent]:
        """One Gemini server message (JSON in a text or binary frame) as subset events. The
        whole frame is read before any state changes, so a bad one (``Malformed``, with a
        short reason) leaves the codec as it was."""
        if isinstance(frame, bytes | bytearray | memoryview):
            raw = bytes(frame)
            if len(raw) > self.max_bytes:
                return [protocol.Malformed(MALFORMED, "oversize")]
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return [protocol.Malformed(MALFORMED, "json")]
        else:
            text = frame
            if len(text) > self.max_bytes or len(text.encode("utf-8", "surrogatepass")) > (
                self.max_bytes
            ):
                return [protocol.Malformed(MALFORMED, "oversize")]
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            return [protocol.Malformed(MALFORMED, "json")]
        if not isinstance(data, dict):
            return [protocol.Malformed(MALFORMED, "shape")]
        try:
            facts = _facts(data)
        except _Bad as bad:
            return [protocol.Malformed(MALFORMED, bad.reason)]
        out: list[protocol.ServerEvent] = []
        for kind, value in facts:
            out += self._apply(kind, value)
        return out

    def _apply(self, kind: str, value: Any) -> list[protocol.ServerEvent]:
        """One fact of a server message (``_facts``) as events, moving the codec's state."""
        if kind == "ignored":
            self.counts[value] += 1
            return []
        if kind == "setup":
            return [protocol.SessionEvent(protocol.SESSION_UPDATED, OUTPUT_RATE_HZ)]
        if kind == "unknown":
            self.counts["unknown_messages"] += 1
            return [protocol.Unknown(UNKNOWN_TYPE)]
        if kind == "go_away":
            return [protocol.GoAway(protocol.GO_AWAY, value)]
        if kind == "activity":
            return self._activity(*value)
        if kind == "heard":
            return self._input_text(value)
        if kind == "usage":
            return self._usage(value)
        if kind == "turn_complete":
            return self._turn_complete()
        if kind == "interrupted":
            return self._interrupted()
        out = self._complete_input()
        if kind == "audio":
            out += self._content(pcm=value)
        elif kind == "said":
            out += self._content(said=value)
        elif kind == "calls":
            out += self._tool_call(value)
        elif kind == "generated":
            out += self._generated()
        return out

    def _activity(self, kind: str, ms: int) -> list[protocol.ServerEvent]:
        """``voiceActivity``: a start is the next sent activity's item (and fences off what
        an older model turn still sends); an end completes a transcript already in."""
        if kind == ACTIVITY_START:
            item = self._sent.pop(0) if self._sent else f"{INPUT_ITEM_PREFIX}{next(self._items)}"
            self._heard = _Activity(item)
            self._dropping = False
            return [protocol.SpeechBoundary(protocol.SPEECH_STARTED, item, ms)]
        if kind != ACTIVITY_END:
            self.counts["unknown_activity"] += 1
            return []
        heard = self._heard_activity()
        heard.ended = True
        out: list[protocol.ServerEvent] = []
        if heard.text and not heard.final:
            out.append(self._final(heard))
        out.append(protocol.SpeechBoundary(protocol.SPEECH_STOPPED, heard.item, ms))
        return out

    def _input_text(self, text: str) -> list[protocol.ServerEvent]:
        """A piece of the input transcript: a delta of the current activity's item (and its
        completion when the activity already ended); after the completion only counted."""
        heard = self._heard_activity()
        if heard.final:
            self.counts["late_input_transcripts"] += 1
            return []
        heard.text += text
        out: list[protocol.ServerEvent] = [
            protocol.InputTranscript(protocol.TRANSCRIPTION_DELTA, heard.item, text)
        ]
        if heard.ended:
            out.append(self._final(heard))
        return out

    def _heard_activity(self) -> _Activity:
        if self._heard is None:
            self._heard = _Activity(f"{INPUT_ITEM_PREFIX}{next(self._items)}")
        return self._heard

    def _final(self, heard: _Activity) -> protocol.InputTranscript:
        heard.final = True
        return protocol.InputTranscript(
            protocol.TRANSCRIPTION_COMPLETED, heard.item, heard.text, True
        )

    def _complete_input(self) -> list[protocol.ServerEvent]:
        """The answer began: an ended activity whose transcript never came is completed
        empty (the route's wait ends: no words)."""
        heard = self._heard
        if heard is None or not heard.ended or heard.final:
            return []
        return [self._final(heard)]

    def _target(self) -> _Response | None:
        """The response model content belongs to: the held one (opened by its first
        content) once the server started the held turn's activity, else the open one; None
        drops it (a cancelled turn, nobody asked)."""
        if self._dropping:
            self.counts["dropped_after_cancel"] += 1
            return None
        hold = self._hold
        if hold is not None and hold.item not in self._sent:
            if hold.response is None:
                hold.response = self._new_response()
            resp: _Response | None = hold.response
        else:  # before the server started the held activity, content is the older turn's
            resp = self._open
        if resp is None or resp.done:
            self.counts["unrequested_dropped"] += 1
            return None
        return resp

    def _ending(self) -> _Response | None:
        """The response a model turn's end applies to: the open one it started (older than
        any hold), else the held one it started; one with no content yet is the next
        turn's (the end came late) and is left open."""
        for resp in (self._open, self._hold.response if self._hold is not None else None):
            if resp is not None and not resp.done and resp.started:
                return resp
        return None

    def _emit(
        self, events: list[protocol.ServerEvent], resp: _Response | None = None
    ) -> list[protocol.ServerEvent]:
        """Events of the held response (and usage, while a hold is on) are held, others
        passed on."""
        hold = self._hold
        if hold is not None and (resp is None or resp is hold.response):
            hold.events += events
            return []
        return events

    def _content(self, *, pcm: bytes | None = None, said: str | None = None) -> list:
        resp = self._target()
        if resp is None:
            return []
        resp.started = resp.output = True
        if pcm is not None:
            event: protocol.ServerEvent = protocol.AudioDelta(
                protocol.AUDIO_DELTA, resp.id, resp.item, pcm
            )
        else:
            resp.text += said or ""
            event = protocol.ReplyTranscript(
                protocol.REPLY_TRANSCRIPT_DELTA, resp.id, resp.item, said or ""
            )
        return self._emit([event], resp)

    def _tool_call(self, calls: list[tuple[str, str, str]]) -> list[protocol.ServerEvent]:
        """``toolCall``: Gemini now waits on every call (also when its response is dropped:
        the codec answers it later); the response ends with the calls."""
        for call_id, name, _ in calls:
            self._calls[call_id] = name
        resp = self._target()
        if resp is None:
            return []
        resp.started = True
        events = self._end_output(resp)
        for call_id, name, arguments in calls:
            item = f"{CALL_ITEM_PREFIX}{next(self._items)}"
            self._call_items[item] = call_id
            events.append(
                protocol.FunctionCall(
                    protocol.FUNCTION_CALL_DONE, resp.id, call_id, name, arguments, item
                )
            )
        events.append(self._done(resp, "completed"))
        return self._emit(events, resp)

    def _generated(self) -> list[protocol.ServerEvent]:
        """``generationComplete`` ends the response (S0: the audio is over; ``turnComplete``
        and its usage follow seconds later); one with no content still ends."""
        resp = self._target()
        if resp is None:
            return []
        resp.started = True
        return self._emit([*self._end_output(resp), self._done(resp, "completed")], resp)

    def _interrupted(self) -> list[protocol.ServerEvent]:
        """Gemini cut its own answer (new input came): the open response is cancelled; after
        a local cancel it only ends the drop."""
        if self._dropping:
            self._dropping = False
            return []
        resp = self._ending()
        if resp is None:
            return []
        return self._emit([*self._end_output(resp), self._done(resp, "cancelled")], resp)

    def _turn_complete(self) -> list[protocol.ServerEvent]:
        """The model turn ended: the response it started and never ended ends now
        (``_ending``); a held answer's turn is marked ended (its discard drops nothing more)."""
        if self._dropping:
            self._dropping = False
            return []
        hold = self._hold
        older = self._open is not None and not self._open.done and self._open.started
        if hold is not None and hold.response is not None and hold.response.started and not older:
            hold.ended = True
        resp = self._ending()
        if resp is None:
            return []
        return self._emit([*self._end_output(resp), self._done(resp, "completed")], resp)

    def _usage(self, total: int) -> list[protocol.ServerEvent]:
        """The model turn's tokens, for the responses ended since the last report."""
        ids, self._closed = tuple(self._closed), []
        return self._emit([protocol.UsageReport(protocol.USAGE_REPORT, ids, total)])

    def _end_output(self, resp: _Response) -> list[protocol.ServerEvent]:
        """The end of the response's output item: its audio and its whole transcript."""
        if not resp.output:
            return []
        resp.output = False
        return [
            protocol.AudioDone(protocol.AUDIO_DONE, resp.id, resp.item),
            protocol.ReplyTranscript(
                protocol.REPLY_TRANSCRIPT_DONE, resp.id, resp.item, resp.text, True
            ),
        ]

    def _done(self, resp: _Response, status: str) -> protocol.ResponseDone:
        """The response's one ``response.done`` (no usage: ``UsageReport`` carries it)."""
        resp.done = True
        self._closed.append(resp.id)
        if self._open is resp:
            self._open = None
        return protocol.ResponseDone(protocol.RESPONSE_DONE, resp.id, status)


# --------------------------------------------------------------------------- reading


def _facts(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """A server message as (kind, value) facts, in the order they apply; raises ``_Bad`` on
    a field of the wrong type. ``usageMetadata`` applies after the content it came with."""
    if not data:
        return [("ignored", "empty_frames")]
    facts: list[tuple[str, Any]] = []
    usage: list[tuple[str, Any]] = []
    for key, value in data.items():
        if key == "setupComplete":
            facts.append(("setup", None))
        elif key == "serverContent":
            facts += _server_content(_obj(value))
        elif key == "toolCall":
            facts.append(("calls", _calls(_obj(value))))
        elif key == "toolCallCancellation":
            facts.append(("ignored", "tool_call_cancellations"))
        elif key == "voiceActivity":
            activity = _obj(value)
            kind = activity.get("type")
            if not isinstance(kind, str):
                raise _Bad("field")
            facts.append(("activity", (kind, _duration_ms(activity.get("audioOffset")))))
        elif key == "usageMetadata":
            total = _obj(value).get("totalTokenCount")
            if total is not None:
                if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                    raise _Bad("field")
                usage.append(("usage", total))
        elif key == "goAway":
            left = _obj(value).get("timeLeft")
            facts.append(("go_away", None if left is None else _duration_ms(left)))
        elif key == "sessionResumptionUpdate":
            facts.append(("ignored", "resumption_updates"))
        else:
            facts.append(("unknown", None))
    return facts + usage


def _server_content(content: dict[str, Any]) -> list[tuple[str, Any]]:
    facts: list[tuple[str, Any]] = []
    if "inputTranscription" in content:
        facts.append(("heard", _text(_obj(content["inputTranscription"]))))
    turn = content.get("modelTurn")
    if turn is not None:
        parts = _obj(turn).get("parts", [])
        if not isinstance(parts, list):
            raise _Bad("field")
        for part in parts:
            facts.append(_part(_obj(part)))
    if "outputTranscription" in content:
        facts.append(("said", _text(_obj(content["outputTranscription"]))))
    if content.get("generationComplete") is True:
        facts.append(("generated", None))
    if content.get("interrupted") is True:
        facts.append(("interrupted", None))
    if content.get("turnComplete") is True:
        facts.append(("turn_complete", None))
    return facts or [("ignored", "empty_content")]


def _part(part: dict[str, Any]) -> tuple[str, Any]:
    if part.get("thought") is True:
        return ("ignored", "thought_parts")  # never spoken, never reply text
    if "inlineData" in part:
        return ("audio", _pcm(_obj(part["inlineData"]).get("data")))
    if "text" in part:
        return ("ignored", "text_parts")
    return ("ignored", "other_parts")


def _calls(tool_call: dict[str, Any]) -> list[tuple[str, str, str]]:
    calls = tool_call.get("functionCalls", [])
    if not isinstance(calls, list):
        raise _Bad("field")
    out = []
    for call in calls:
        call = _obj(call)
        call_id, name, args = call.get("id"), call.get("name"), call.get("args", {})
        if not isinstance(call_id, str) or not call_id or len(call_id) > protocol.ID_MAX_CHARS:
            raise _Bad("field")
        if not isinstance(name, str) or not name:
            raise _Bad("field")
        out.append((call_id, name, json.dumps(_obj(args), ensure_ascii=False)))
    return out


def _obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _Bad("field")
    return value


def _text(data: dict[str, Any]) -> str:
    value = data.get("text", "")
    if not isinstance(value, str):
        raise _Bad("field")
    return value


def _pcm(value: Any) -> bytes:
    if not isinstance(value, str):
        raise _Bad("field")
    try:
        pcm = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _Bad("base64") from None
    if len(pcm) % protocol.PCM_SAMPLE_BYTES:
        raise _Bad("odd_pcm")
    return pcm


def _duration_ms(value: Any) -> int:
    """A protobuf Duration ("1.689375s") in whole ms, exactly (no float error), floored the
    same way for every position, so an activity that starts where the last ended matches."""
    if not isinstance(value, str) or not value.endswith(DURATION_UNIT):
        raise _Bad("field")
    try:
        seconds = Decimal(value[: -len(DURATION_UNIT)])
    except InvalidOperation:
        raise _Bad("field") from None
    if not seconds.is_finite() or seconds < 0:
        raise _Bad("field")
    return int(seconds * MS_PER_S)


# --------------------------------------------------------------------------- writing


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def _declaration(tool: dict[str, Any]) -> dict[str, Any]:
    """A realtime function tool as a Gemini function declaration (BLOCKING)."""
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters") or EMPTY_PARAMETERS,
        "behavior": TOOL_BEHAVIOR,
    }


def _response_object(output: Any) -> dict[str, Any]:
    """A tool output as Gemini's ``response`` object: the output when it is a JSON object,
    else the output (parsed when it is JSON) under ``result``."""
    if not isinstance(output, str):
        return {OUTPUT_KEY: output}
    try:
        value = json.loads(output)
    except (ValueError, RecursionError):
        return {OUTPUT_KEY: output}
    return value if isinstance(value, dict) else {OUTPUT_KEY: value}
