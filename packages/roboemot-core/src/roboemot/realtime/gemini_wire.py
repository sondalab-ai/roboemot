"""Gemini Live's wire as the realtime subset the route speaks (SPEC 8, Realtime speech route).

:class:`GeminiWire` is a pure, stateful codec (``wire.Wire``): no socket, no thread, no clock.
One instance serves one session; a reconnect starts a new one. Message shapes are the ones the
S0 probe saw on ``gemini-3.8-live`` (2026-11-11).

Out (``encode``):

- the first ``session.update`` is ``setup`` (manual activity, input and output transcription,
  sliding-window compression, BLOCKING function declarations with their JSON Schema); later
  ones only keep changed instructions: the lines the session does not hold yet go as context
  text (``clientContent``, ``turnComplete`` false) right before the next activity or text
  turn, never between a ``toolCall`` and its ``toolResponse`` (a removed line is not sent);
- a burst's appends are ``realtimeInput.audio`` after an ``activityStart``, its commit
  ``activityEnd``; ``conversation.item.create`` text (which supersedes an answer still held)
  and ``function_call_output`` are kept until the next ``response.create``;
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
(a reply still streaming, or one dropped after a cancel), content after it the new turn's. A
server that skips the start (``ACTIVITY_END``, a transcript or the answer of a sent activity
first) gets one synthesized, without a position.

Usage and turn ends: every ``generationComplete`` owes one ``turnComplete``, which comes with
``usageMetadata`` seconds later, possibly once the next turn is under way. So each
``response.done`` has ``total_tokens`` None (the route charges its estimate), an owed
``turnComplete`` only reports its turn's total as a ``UsageReport`` naming the responses that
ended before it (it never ends a newer response), and an output transcript that comes after
its response ended is dropped (counted: the route follows no response after its done).

Drops: after a cancel, a discarded answer, or calls answered ``skipped``, what Gemini still
says is dropped until its next ``generationComplete``, ``turnComplete``, ``interrupted`` or
``ACTIVITY_START``. A text turn sent right after skipped calls is not dropped: Gemini may
answer both in one reply, which is the text turn's.

A BLOCKING call Gemini waits on is never left open: before the next activity or text turn,
and when the route deletes an unanswered call's item, the codec answers it ``skipped``.

Known limits: on a barge-in the route settles a cut round's calls after the next burst
began, so their real outputs never reach Gemini (the calls were answered ``skipped``). The
``skipped`` payload (no spoken reply follows) and the skills' JSON Schema declarations were
accepted by the live server (S3 run, 2026-11-12).

Events keep text and audio out of ``repr`` (``protocol``); the codec logs nothing (SPEC 12).
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import count
from typing import Any

from roboemot.realtime import protocol

# the input rate the audio's mimeType names when the update asks for the native rate (null)
INPUT_RATE_HZ = 16000
OUTPUT_RATE_HZ = 24000  # Gemini Live's reply PCM16 (audio/pcm;rate=24000, S0 2026-11-11)
AUDIO_MIME = "audio/pcm;rate={rate}"
AUDIO_MIME_PREFIX = "audio/pcm"
MIME_RATE = re.compile(r"rate=([0-9]{1,6})")
MODEL_PREFIX = "models/"
KEY_HEADER = "x-goog-api-key"  # the key's only place: never the URL (S0 2026-11-11)
TOOL_BEHAVIOR = "BLOCKING"  # Gemini waits for the toolResponse (S0 gate 4)
EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}
MS_PER_S = 1000
# protobuf Duration as JSON ("1.689375s"); bounded, so no huge number is ever built
DURATION = re.compile(r"([0-9]{1,12})(\.[0-9]{1,9})?s")
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
    "text_answer_parts",  # text in an audio session's model turn: never spoken, dropped
    "other_parts",
    "audio_rate_mismatch",
    "bad_durations",
    "unknown_messages",
    "unknown_activity",
    "activity_starts_missing",
    "late_input_transcripts",
    "late_reply_transcripts",
    "context_removals_unsent",
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


class _FrameError(Exception):
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
    item, the response the answer opened (None until content came) and its events."""

    item: str = field(repr=False)
    response: _Response | None = None
    events: list[protocol.ServerEvent] = field(default_factory=list, repr=False)


class GeminiWire:
    """See the module docstring. ``model`` is the route's model (``models/`` added when
    missing); ``voice`` (the route's own) wins over the update's; ``max_bytes`` caps a frame
    as the client caps the socket. ``session`` (the client's session count) goes into every id
    the codec makes, so no id of one session repeats in the next: the route keys its charges by
    response id across sessions. ``counts`` holds what was ignored, dropped or emulated."""

    def __init__(
        self, model: str, max_bytes: int, voice: str | None = None, session: int | None = None
    ) -> None:
        self.model = model if model.startswith(MODEL_PREFIX) else MODEL_PREFIX + model
        self._session = "" if session is None else f"{session}_"
        self.voice = voice or None
        self.max_bytes = max_bytes
        self.counts: Counter[str] = Counter({name: 0 for name in COUNTERS})
        self._responses = count(1)
        self._items = count(1)
        self._setup = False
        self._input_rate = INPUT_RATE_HZ
        self._context_sent = ""
        self._context: str | None = None
        self._texts: list[tuple[str | None, str]] = []  # (item id, text) of user texts kept
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
        # one entry per generationComplete whose turnComplete is still to come: the responses
        # that ended in that turn (its usage report names them)
        self._owed: list[list[str]] = []
        self._usage_ids: list[str] | None = None  # set by an owed turnComplete for its usage

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
        (the latest wins: each update carries the whole instructions; only the lines the
        session lacks are sent, ``_context_turns``)."""
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
            self._burst = self._item_id(INPUT_ITEM_PREFIX)
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
        if item.get("type") != "message" or not texts:
            self.counts["items_unsupported"] += 1
            return [], []
        item_id = item.get("id")
        self._texts.append((item_id if isinstance(item_id, str) else None, "".join(texts)))
        return [], self._discard()  # new input: an answer still held is superseded

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
        if item is not None:
            self._texts = [(i, t) for i, t in self._texts if i != item]  # never sent later
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
            if self._calls:
                # calls the route never saw (a discarded answer) are answered first; Gemini
                # may answer both in one reply, so nothing after is dropped (it is this turn's)
                frames.append(self._tool_response())
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
        response). While Gemini may still generate it (no response yet, or one not ended)
        the rest is dropped as it comes. Its activity, if never started, is forgotten."""
        hold, self._hold = self._hold, None
        if hold is None:
            return []
        if hold.item in self._sent:
            self._sent.remove(hold.item)
        rid = hold.response.id if hold.response is not None else None
        if rid is not None:
            self.counts["held_discarded"] += 1
            for ids in (self._closed, *self._owed):
                if rid in ids:
                    ids.remove(rid)
        if hold.response is None or not hold.response.done:
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
        """The kept context as user turns: the instructions' lines the session does not hold
        yet first (a line only removed is not sent: counted), then the user texts."""
        turns = []
        if self._context is not None:
            added = _added_lines(self._context_sent, self._context)
            if added:
                turns.append(_user(CONTEXT_PREFIX + added))
            else:
                self.counts["context_removals_unsent"] += 1
            self._context_sent, self._context = self._context, None
        turns += [_user(text) for _, text in self._texts]
        self._texts.clear()
        return turns

    def _new_response(self) -> _Response:
        n = f"{self._session}{next(self._responses)}"
        return _Response(f"{RESPONSE_PREFIX}{n}", f"{OUTPUT_ITEM_PREFIX}{n}")

    def _item_id(self, prefix: str) -> str:
        return f"{prefix}{self._session}{next(self._items)}"

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
        except _FrameError as bad:
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
        out = self._adopt_held() + self._complete_input()
        if kind == "audio":
            out += self._content(pcm=value)
        elif kind == "said":
            if self._owed and not self._current_started():
                self.counts["late_reply_transcripts"] += 1  # its response is done already
            else:
                out += self._content(said=value)
        elif kind == "calls":
            out += self._tool_call(value)
        elif kind == "generated":
            out += self._generated()
        return out

    def _activity(self, kind: str, ms: int | None) -> list[protocol.ServerEvent]:
        """``voiceActivity``: a start is the next sent activity's item (and fences off what
        an older model turn still sends); an end completes a transcript already in (one with
        no start before it gets a synthesized one, ``_adopt``)."""
        if kind == ACTIVITY_START:
            item = self._sent.pop(0) if self._sent else self._item_id(INPUT_ITEM_PREFIX)
            self._heard = _Activity(item)
            self._dropping = False
            return [protocol.SpeechBoundary(protocol.SPEECH_STARTED, item, ms)]
        if kind != ACTIVITY_END:
            self.counts["unknown_activity"] += 1
            return []
        out: list[protocol.ServerEvent] = []
        if self._heard is None or self._heard.ended:
            out += self._adopt()
        heard = self._heard_activity()
        heard.ended = True
        if heard.text and not heard.final:
            out.append(self._final(heard))
        out.append(protocol.SpeechBoundary(protocol.SPEECH_STOPPED, heard.item, ms))
        return out

    def _input_text(self, text: str) -> list[protocol.ServerEvent]:
        """A piece of the input transcript: a delta of the current activity's item (and its
        completion when the activity already ended); after the completion only counted."""
        out: list[protocol.ServerEvent] = []
        if self._heard is None or self._heard.final:
            out += self._adopt()
        heard = self._heard_activity()
        if heard.final:
            self.counts["late_input_transcripts"] += 1
            return out
        heard.text += text
        out.append(protocol.InputTranscript(protocol.TRANSCRIPTION_DELTA, heard.item, text))
        if heard.ended:
            out.append(self._final(heard))
        return out

    def _adopt(self) -> list[protocol.ServerEvent]:
        """A sent activity the server reports without its ``ACTIVITY_START``: started now, as
        the start would (no position: the route's burst holds it), so nothing stalls and the
        next burst's start still lines up."""
        if not self._sent:
            return []
        item = self._sent.pop(0)
        self._heard = _Activity(item)
        self._dropping = False
        self.counts["activity_starts_missing"] += 1
        return [protocol.SpeechBoundary(protocol.SPEECH_STARTED, item, None)]

    def _adopt_held(self) -> list[protocol.ServerEvent]:
        """Model content while the held activity was never started and no older reply
        streams or is being dropped: the server skipped its start (``_adopt``)."""
        hold = self._hold
        out: list[protocol.ServerEvent] = []
        older = self._dropping or self._older_live()
        while hold is not None and hold.item in self._sent and not older:
            out += self._adopt()
        return out

    def _older_live(self) -> bool:
        """An open response that already has content: a reply still streaming."""
        return self._open is not None and not self._open.done and self._open.started

    def _current_started(self) -> bool:
        """Whether the response content would go to has content already."""
        hold = self._hold
        if hold is not None and hold.item not in self._sent:
            return hold.response is not None and hold.response.started
        return self._older_live()

    def _heard_activity(self) -> _Activity:
        if self._heard is None:
            self._heard = _Activity(self._item_id(INPUT_ITEM_PREFIX))
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
            item = self._item_id(CALL_ITEM_PREFIX)
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
        and its usage follow seconds later, so one is owed); one with no content still ends.
        It also ends a drop."""
        events: list[protocol.ServerEvent] = []
        if self._dropping:
            self.counts["dropped_after_cancel"] += 1
            self._dropping = False
        else:
            resp = self._target()
            if resp is not None:
                resp.started = True
                done = [*self._end_output(resp), self._done(resp, "completed")]
                events = self._emit(done, resp)
        self._owed.append(self._closed)
        self._closed = []
        return events

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
        """The model turn ended. One a ``generationComplete`` already ended only reports its
        usage (it may come once the next turn is under way: it never ends that one). Else the
        response it started and never ended ends now (``_ending``); it also ends a drop."""
        if self._owed:
            self._usage_ids = self._owed.pop(0)
            return []
        if self._dropping:
            self._dropping = False
            return []
        resp = self._ending()
        if resp is None:
            return []
        return self._emit([*self._end_output(resp), self._done(resp, "completed")], resp)

    def _usage(self, total: int) -> list[protocol.ServerEvent]:
        """The model turn's tokens, for the responses of that turn: those an owed
        ``turnComplete`` names, else those ended since the last report."""
        if self._usage_ids is not None:
            ids, self._usage_ids = tuple(self._usage_ids), None
        else:
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

BAD = object()  # a duration field that cannot be read (``_duration_ms``)


def _facts(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """A server message as (kind, value) facts, in the order they apply; raises ``_FrameError`` on
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
                raise _FrameError("field")
            ms = _duration_ms(activity.get("audioOffset"))
            if ms is BAD:
                facts.append(("ignored", "bad_durations"))
            facts.append(("activity", (kind, None if ms is BAD else ms)))
        elif key == "usageMetadata":
            total = _obj(value).get("totalTokenCount")
            if total is not None:
                if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                    raise _FrameError("field")
                usage.append(("usage", total))
        elif key == "goAway":
            left = _duration_ms(_obj(value).get("timeLeft"))
            if left is BAD:
                facts.append(("ignored", "bad_durations"))
            facts.append(("go_away", None if left is BAD else left))
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
            raise _FrameError("field")
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
        data = _obj(part["inlineData"])
        mime = data.get("mimeType")
        if isinstance(mime, str):
            if not mime.startswith(AUDIO_MIME_PREFIX):
                return ("ignored", "other_parts")
            rate = MIME_RATE.search(mime)
            if rate is not None and int(rate.group(1)) != OUTPUT_RATE_HZ:
                return ("ignored", "audio_rate_mismatch")  # the route plays it at 24 kHz
        return ("audio", _pcm(data.get("data")))
    if "text" in part:
        return ("ignored", "text_answer_parts")
    return ("ignored", "other_parts")


def _calls(tool_call: dict[str, Any]) -> list[tuple[str, str, str]]:
    calls = tool_call.get("functionCalls", [])
    if not isinstance(calls, list):
        raise _FrameError("field")
    out = []
    for call in calls:
        call = _obj(call)
        call_id, name, args = call.get("id"), call.get("name"), call.get("args", {})
        if not isinstance(call_id, str) or not call_id or len(call_id) > protocol.ID_MAX_CHARS:
            raise _FrameError("field")
        if not isinstance(name, str) or not name:
            raise _FrameError("field")
        out.append((call_id, name, json.dumps(_obj(args), ensure_ascii=False)))
    return out


def _obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _FrameError("field")
    return value


def _text(data: dict[str, Any]) -> str:
    value = data.get("text", "")
    if not isinstance(value, str):
        raise _FrameError("field")
    return value


def _pcm(value: Any) -> bytes:
    if not isinstance(value, str):
        raise _FrameError("field")
    try:
        pcm = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _FrameError("base64") from None
    if len(pcm) % protocol.PCM_SAMPLE_BYTES:
        raise _FrameError("odd_pcm")
    return pcm


def _duration_ms(value: Any) -> Any:
    """A protobuf Duration ("1.689375s") in whole ms, exactly (no float error), floored the
    same way for every position, so an activity that starts where the last ended matches.
    None when absent; ``BAD`` for anything that is not a bounded plain duration (the field
    is then ignored, never the frame). Never raises."""
    if value is None:
        return None
    if not isinstance(value, str) or DURATION.fullmatch(value) is None:
        return BAD
    return int(Decimal(value[:-1]) * MS_PER_S)


# --------------------------------------------------------------------------- writing


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "parts": [{"text": text}]}


def _declaration(tool: dict[str, Any]) -> dict[str, Any]:
    """A realtime function tool as a Gemini function declaration (BLOCKING). Its parameters
    go as ``parametersJsonSchema``: skill schemas use JSON Schema (``additionalProperties``,
    ``maxLength``, ``minimum``) outside the OpenAPI subset ``parameters`` takes."""
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parametersJsonSchema": tool.get("parameters") or EMPTY_PARAMETERS,
        "behavior": TOOL_BEHAVIOR,
    }


def _added_lines(held: str, new: str) -> str:
    """The lines of ``new`` that ``held`` does not have, in order."""
    known = set(held.splitlines())
    return "\n".join(line for line in new.splitlines() if line not in known)


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
