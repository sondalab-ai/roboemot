"""The subset of the OpenAI realtime protocol the realtime speech route speaks (SPEC 8, Realtime
speech route).

Builders return the client events as plain dicts (``encode`` makes the JSON text frame). The
parser turns one server text frame into a typed event, checking its size, its JSON shape and
the type of every field read. What is not understood never raises:

- an event type outside the subset is :class:`Unknown` (counted and ignored, the protocol drifts);
- a frame that is too big, binary, not JSON, or carries a field of the wrong type is
  :class:`Malformed` with a short reason code;
- the audio of ``response.output_audio.delta`` is base64 PCM16: a payload that is not base64 or
  whose length is odd is :class:`Malformed`;
- a server ``error`` keeps only its ``code`` (a short identifier), never its ``message``, which
  can quote what was sent.

Events that hold text, audio or tool arguments keep them out of ``repr``, so a log line or an
exception that shows an event never shows its content (SPEC 12). Identifiers (items, responses,
calls) are kept for the protocol and never logged either.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Client events (the subset the servers accept: SPEC 8, Realtime speech route).
SESSION_UPDATE = "session.update"
AUDIO_APPEND = "input_audio_buffer.append"
AUDIO_COMMIT = "input_audio_buffer.commit"
ITEM_CREATE = "conversation.item.create"
ITEM_DELETE = "conversation.item.delete"
ITEM_TRUNCATE = "conversation.item.truncate"
RESPONSE_CREATE = "response.create"
RESPONSE_CANCEL = "response.cancel"

# Server events.
SESSION_CREATED = "session.created"
SESSION_UPDATED = "session.updated"
ERROR = "error"
SPEECH_STARTED = "input_audio_buffer.speech_started"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
RESPONSE_CREATED = "response.created"
AUDIO_DELTA = "response.output_audio.delta"
AUDIO_DONE = "response.output_audio.done"
REPLY_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
REPLY_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
FUNCTION_CALL_DONE = "response.function_call_arguments.done"
RESPONSE_DONE = "response.done"

AUDIO_FORMAT = "audio/pcm"  # PCM16 little endian, mono
PCM_SAMPLE_BYTES = 2
TOOL_CHOICES = ("auto", "none")  # "none" only after the tool-round cap: it forbids no tool
RESPONSE_STATUSES = ("completed", "cancelled", "incomplete", "failed")
# Per-stage latencies a speech-to-speech server may put in response.done metadata (seconds).
LATENCY_METADATA_KEY = "speech_to_speech.turn_latency"
LATENCY_STAGES = ("vad_decision", "stt", "llm", "tts_ttfa", "e2e")
ID_MAX_CHARS = 256  # longest item, response or call id kept (servers use about 30)
ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
EVENT_TYPE = re.compile(r"[a-z0-9_.]{1,128}")


# --------------------------------------------------------------------------- client events


def session_update(
    *,
    instructions: str,
    input_rate_hz: int,
    output_rate_hz: int | None,
    voice: str,
    vad_threshold: float,
    silence_ms: int,
    create_response: bool,
    interrupt_response: bool,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """``session.update``: instructions, PCM16 in and out (``output_rate_hz`` None asks for the
    server's native rate, a speech-to-speech extension), server VAD with its threshold and end
    silence, and the function tools (realtime format, ``function_tools``) with
    ``tool_choice: "auto"``."""
    session: dict[str, Any] = {
        "type": "realtime",
        "instructions": instructions,
        "audio": {
            "input": {
                "format": {"type": AUDIO_FORMAT, "rate": input_rate_hz},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": vad_threshold,
                    "silence_duration_ms": silence_ms,
                    "create_response": create_response,
                    "interrupt_response": interrupt_response,
                },
            },
            "output": {"format": {"type": AUDIO_FORMAT, "rate": output_rate_hz}, "voice": voice},
        },
    }
    if tools:
        session["tools"] = tools
        session["tool_choice"] = "auto"
    return {"type": SESSION_UPDATE, "session": session}


def function_tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat Completions tools (``{"type": "function", "function": {...}}``) in the realtime
    format (``type``, ``name``, ``description``, ``parameters`` at the top level)."""
    out = []
    for tool in chat_tools:
        fn = tool["function"]
        out.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def audio_append(pcm: bytes) -> dict[str, Any]:
    """``input_audio_buffer.append`` with PCM16 bytes (base64 on the wire)."""
    if len(pcm) % PCM_SAMPLE_BYTES:
        raise ValueError("PCM16 audio has an even number of bytes")
    return {"type": AUDIO_APPEND, "audio": base64.b64encode(pcm).decode("ascii")}


def audio_commit() -> dict[str, Any]:
    return {"type": AUDIO_COMMIT}


def text_item(text: str) -> dict[str, Any]:
    """A user message as ``input_text`` (the only message content the servers take)."""
    return {
        "type": ITEM_CREATE,
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def function_output(call_id: str, output: str) -> dict[str, Any]:
    """The result of a tool call, as a ``function_call_output`` item."""
    return {
        "type": ITEM_CREATE,
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def item_delete(item_id: str) -> dict[str, Any]:
    return {"type": ITEM_DELETE, "item_id": item_id}


def item_truncate(item_id: str, audio_end_ms: int, content_index: int = 0) -> dict[str, Any]:
    """Cut an assistant audio item at what was played (barge-in)."""
    return {
        "type": ITEM_TRUNCATE,
        "item_id": item_id,
        "content_index": content_index,
        "audio_end_ms": max(0, int(audio_end_ms)),
    }


def response_create(
    instructions: str | None = None, tool_choice: Literal["auto", "none"] | None = None
) -> dict[str, Any]:
    """``response.create``, with per-response ``instructions`` and ``tool_choice`` when given.
    ``tool_choice`` is ``auto`` or ``none`` only (never a forced tool, SPEC 8)."""
    response: dict[str, Any] = {}
    if instructions is not None:
        response["instructions"] = instructions
    if tool_choice is not None:
        if tool_choice not in TOOL_CHOICES:
            raise ValueError("tool_choice is auto or none")
        response["tool_choice"] = tool_choice
    event: dict[str, Any] = {"type": RESPONSE_CREATE}
    if response:
        event["response"] = response
    return event


def response_cancel() -> dict[str, Any]:
    return {"type": RESPONSE_CANCEL}


def encode(event: dict[str, Any]) -> str:
    """The text frame of a client event."""
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- server events


@dataclass(frozen=True)
class ServerEvent:
    type: str


@dataclass(frozen=True)
class SessionEvent(ServerEvent):
    """``session.created`` or ``session.updated``; ``output_rate_hz`` when the server says."""

    output_rate_hz: int | None = None


@dataclass(frozen=True)
class ServerError(ServerEvent):
    """A server ``error``: its ``code`` only (None when absent or not an identifier)."""

    code: str | None = None


@dataclass(frozen=True)
class SpeechBoundary(ServerEvent):
    """The server VAD saw speech start or stop in the input buffer."""

    item_id: str | None = field(default=None, repr=False)
    audio_ms: int | None = None


@dataclass(frozen=True)
class InputTranscript(ServerEvent):
    """The user's words as the server heard them: a ``delta`` or the ``final`` transcript."""

    item_id: str | None = field(default=None, repr=False)
    text: str = field(default="", repr=False)
    final: bool = False


@dataclass(frozen=True)
class ResponseCreated(ServerEvent):
    response_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AudioDelta(ServerEvent):
    """PCM16 bytes of the reply (decoded, even length)."""

    response_id: str | None = field(default=None, repr=False)
    item_id: str | None = field(default=None, repr=False)
    pcm: bytes = field(default=b"", repr=False)


@dataclass(frozen=True)
class AudioDone(ServerEvent):
    response_id: str | None = field(default=None, repr=False)
    item_id: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ReplyTranscript(ServerEvent):
    """The text of the reply's audio: a ``delta`` or the ``final`` transcript."""

    response_id: str | None = field(default=None, repr=False)
    item_id: str | None = field(default=None, repr=False)
    text: str = field(default="", repr=False)
    final: bool = False


@dataclass(frozen=True)
class FunctionCall(ServerEvent):
    """A complete tool call; ``arguments`` is the raw JSON text (parsed by the executor)."""

    response_id: str | None = field(default=None, repr=False)
    call_id: str = field(default="", repr=False)
    name: str = ""
    arguments: str = field(default="", repr=False)


@dataclass(frozen=True)
class ResponseDone(ServerEvent):
    """End of a response: its status (``RESPONSE_STATUSES``, else ``unknown``), the tokens it
    used when the server says, and the server's own stage latencies in seconds when sent."""

    response_id: str | None = field(default=None, repr=False)
    status: str = "unknown"
    total_tokens: int | None = None
    latency_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Unknown(ServerEvent):
    """An event outside the subset: counted, ignored."""


@dataclass(frozen=True)
class Malformed(ServerEvent):
    """A frame that could not be read; ``reason`` is a short code, never content."""

    reason: str = ""


class _Bad(Exception):
    """A field of the wrong type (internal: becomes :class:`Malformed`)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _obj(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _Bad("shape")
    return value


def _opt_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > ID_MAX_CHARS:
        raise _Bad("field")
    return value


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise _Bad("field")
    return value


def _opt_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Bad("field")
    return value


def _pcm(data: dict[str, Any]) -> bytes:
    value = data.get("delta")
    if not isinstance(value, str):
        raise _Bad("field")
    try:
        pcm = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise _Bad("base64") from None
    if len(pcm) % PCM_SAMPLE_BYTES:
        raise _Bad("odd_pcm")
    return pcm


def _session(kind: str, data: dict[str, Any]) -> SessionEvent:
    session = _obj(data.get("session", {}))
    audio = _obj(session.get("audio", {}))
    output = _obj(audio.get("output", {}))
    fmt = _obj(output.get("format", {}))
    return SessionEvent(kind, _opt_int(fmt, "rate"))


def _error(data: dict[str, Any]) -> ServerError:
    error = data.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    ok = isinstance(code, str) and ERROR_CODE.fullmatch(code) is not None
    return ServerError(ERROR, code if ok else None)


def _response_done(data: dict[str, Any]) -> ResponseDone:
    response = _obj(data.get("response", {}))
    status = response.get("status")
    usage = response.get("usage")
    tokens = _opt_int(usage, "total_tokens") if isinstance(usage, dict) else None
    metadata = response.get("metadata")
    stages = metadata.get(LATENCY_METADATA_KEY) if isinstance(metadata, dict) else None
    latency: dict[str, float] = {}
    if isinstance(stages, dict):
        for stage in LATENCY_STAGES:
            value = stages.get(stage)
            number = isinstance(value, int | float) and not isinstance(value, bool)
            if number and math.isfinite(value) and value >= 0:
                latency[stage] = float(value)
    return ResponseDone(
        RESPONSE_DONE,
        _opt_str(response, "id"),
        status if status in RESPONSE_STATUSES else "unknown",
        tokens if tokens is None or tokens >= 0 else None,
        latency,
    )


def _event(kind: str, data: dict[str, Any]) -> ServerEvent:
    if kind in (SESSION_CREATED, SESSION_UPDATED):
        return _session(kind, data)
    if kind == ERROR:
        return _error(data)
    if kind in (SPEECH_STARTED, SPEECH_STOPPED):
        ms_key = "audio_start_ms" if kind == SPEECH_STARTED else "audio_end_ms"
        return SpeechBoundary(kind, _opt_str(data, "item_id"), _opt_int(data, ms_key))
    if kind == TRANSCRIPTION_DELTA:
        return InputTranscript(kind, _opt_str(data, "item_id"), _text(data, "delta"))
    if kind == TRANSCRIPTION_COMPLETED:
        return InputTranscript(kind, _opt_str(data, "item_id"), _text(data, "transcript"), True)
    if kind == RESPONSE_CREATED:
        return ResponseCreated(kind, _opt_str(_obj(data.get("response", {})), "id"))
    if kind == AUDIO_DELTA:
        return AudioDelta(
            kind, _opt_str(data, "response_id"), _opt_str(data, "item_id"), _pcm(data)
        )
    if kind == AUDIO_DONE:
        return AudioDone(kind, _opt_str(data, "response_id"), _opt_str(data, "item_id"))
    if kind in (REPLY_TRANSCRIPT_DELTA, REPLY_TRANSCRIPT_DONE):
        final = kind == REPLY_TRANSCRIPT_DONE
        return ReplyTranscript(
            kind,
            _opt_str(data, "response_id"),
            _opt_str(data, "item_id"),
            _text(data, "transcript" if final else "delta"),
            final,
        )
    if kind == FUNCTION_CALL_DONE:
        name = _opt_str(data, "name")
        call_id = _opt_str(data, "call_id")
        if not name or not call_id:
            raise _Bad("field")
        return FunctionCall(
            kind, _opt_str(data, "response_id"), call_id, name, _text(data, "arguments")
        )
    if kind == RESPONSE_DONE:
        return _response_done(data)
    return Unknown(kind)


def parse(frame: str | bytes, max_bytes: int) -> ServerEvent:
    """One server frame as an event. ``max_bytes`` bounds the frame (the client sets the same
    cap on the socket); nothing raises, a bad frame is :class:`Malformed`."""
    if isinstance(frame, bytes | bytearray | memoryview):
        return Malformed("malformed", "binary")
    if len(frame) > max_bytes or len(frame.encode("utf-8", "surrogatepass")) > max_bytes:
        return Malformed("malformed", "oversize")
    try:
        data = json.loads(frame)
    except (ValueError, RecursionError):
        return Malformed("malformed", "json")
    if not isinstance(data, dict):
        return Malformed("malformed", "shape")
    kind = data.get("type")
    if not isinstance(kind, str) or EVENT_TYPE.fullmatch(kind) is None:
        return Malformed("malformed", "type")
    try:
        return _event(kind, data)
    except _Bad as bad:
        return Malformed("malformed", bad.reason)
