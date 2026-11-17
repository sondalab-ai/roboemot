"""The LLM prompts log: the conversation's recent cloud LLM requests, for the web UI's Live tab
(SPEC 11A, 12).

An in-memory ring of ``{seq, t, route, via, label, text}``, filled by the conversation's router
(one entry per chat request, after the history trim and the aliasing, recorded just before it is
sent: a request that then fails is still listed) and by the realtime session's client (one entry
per frame, after the wire codec: Gemini's ``setup``, ``clientContent``...). Route checks, the
realtime check and the voice preview build their own router or client and are not recorded.
Opt-in (``webui.live.prompts``, with the Live tab itself on): while either is off nothing is
recorded, and turning it off clears what was kept. Only the payload is kept: never a URL, a
header or a key. Audio is never kept: consecutive audio frames of a route, with the activity
markers and commits between them, become one ``[audio N bytes in K frames]`` entry that grows in
place; any other binary part (a ``data:`` URL, a ``data`` next to its ``mimeType`` or
``format``, an image, video or audio part) is replaced by its size, and a long list of numbers by
its length. Bounded by count and text length, from the configuration. Never persisted, logged,
repr-ed or put in a debug report: this module has no logger and no ``__repr__`` that shows an
entry.

Thread-safe: the router records on the asyncio loop, the realtime client on its own thread, the
web UI reads on its request threads. The "off" case returns before anything is rendered.
"""

from __future__ import annotations

import collections
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from roboemot.config.schema import LiveConfig

EPOCH_BYTES = 4  # random bytes naming one life of the ring (a restart gives the page a new one)
JSON_INDENT = 2  # realtime frames are shown as pretty JSON
BASE64_CHARS = 4  # every 4 base64 characters
BASE64_BYTES = 3  # carry 3 bytes
AUDIO_LABEL = "audio"
CHAT_LABEL = "chat.completions"
OPENAI_AUDIO_APPEND = "input_audio_buffer.append"
# frames that only mark the audio stream: folded into the audio summary entry
OPENAI_AUDIO_MARKERS = frozenset({"input_audio_buffer.commit", "input_audio_buffer.clear"})
GEMINI_AUDIO_MARKERS = frozenset({"activityStart", "activityEnd", "audioStreamEnd"})
GEMINI_INPUT = "realtimeInput"
# keys whose text value is a base64 payload: replaced by its size wherever it sits (``audio`` as
# an object is a session's audio settings and stays)
BLOB_TEXT_KEYS = frozenset({"audio", "image_url"})
# keys of a whole binary part: replaced by a marker whatever the value
BLOB_KEYS = frozenset({"inlineData", "mediaChunks", "input_audio", "video", "image", "media"})
BLOB_DATA_KEY = "data"  # a binary payload when its object names a ``mimeType`` or ``format``
BLOB_TYPE_KEYS = frozenset({"mimeType", "format"})
DATA_URL = "data:"
DATA_URL_BASE64 = ";base64,"
# a list of this many numbers or more is collapsed to its length: the web UI's vector guard
# (webui.api.VECTOR_GUARD_LEN, tests/test_prompt_log.py checks they agree) refuses such a reply
NUMBER_LIST_LEN = 32


@dataclass(repr=False)
class _Entry:
    seq: int
    wall: float  # epoch seconds, shown by the page
    route: str
    via: str  # the model (chat) or the wire protocol (realtime)
    label: str  # chat.completions, the frame's type, or audio
    text: str
    rev: int = 0  # the ring's revision when this entry was last added or changed
    audio_bytes: int = 0
    audio_frames: int = 0
    markers: collections.Counter[str] = field(default_factory=collections.Counter)
    size: int = 0  # the text's length as the web UI encodes it (JSON, ASCII escapes)


class PromptRing:
    """See the module docstring. ``chat`` and ``frame`` are the two taps."""

    def __init__(self, cfg: LiveConfig, *, wall: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self._enabled = cfg.enabled and cfg.prompts
        self._wall = wall
        self._lock = threading.Lock()
        self._entries: collections.deque[_Entry] = collections.deque()
        self._seq = 0
        self._rev = 0
        self._boot = secrets.token_hex(EPOCH_BYTES)
        self._generation = 0

    def __repr__(self) -> str:
        return f"PromptRing(enabled={self._enabled}, size={len(self._entries)})"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def attach(self, stack: Any) -> Callable[[], None]:
        """Follow ``webui.live`` on the configuration stack as a listener of its own, so turning
        the flag off stops the recording and clears the ring whatever the other listeners do."""
        return stack.subscribe(lambda change: self.apply_config(change.config.webui.live))

    def apply_config(self, cfg: LiveConfig) -> None:
        """Hot reload: new bounds apply at once; turning the flag off clears the ring."""
        with self._lock:
            self._cfg = cfg
            self._enabled = cfg.enabled and cfg.prompts
            if not self._enabled:
                if self._entries:
                    self._clear_locked()
            else:
                self._trim_locked()

    def clear(self) -> None:
        """Forget everything (the Clear button, shutdown). Sequence numbers keep rising."""
        with self._lock:
            self._clear_locked()

    def chat(
        self,
        route: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> None:
        """One chat request as it leaves: role-prefixed message lines and the tool names."""
        if not self._enabled:
            return
        self._add(route, model, CHAT_LABEL, render_chat(messages, tools))

    def frame(self, route: str, protocol: str, frame: dict[str, Any]) -> None:
        """One realtime frame as it leaves the codec. An audio frame or an audio marker only
        grows the route's audio summary entry (a new one when another frame came in between)."""
        if not self._enabled:
            return
        audio = frame_audio_bytes(frame)
        if audio is not None:
            self._audio(route, protocol, audio, None)
            return
        marker = audio_marker(frame)
        if marker is not None:
            self._audio(route, protocol, 0, marker)
            return
        text = json.dumps(redact(frame), ensure_ascii=False, indent=JSON_INDENT)
        self._add(route, protocol, frame_label(frame), text)

    def snapshot(self, after: int = 0, epoch: str = "") -> dict[str, Any]:
        """``enabled``, ``epoch`` (changes on clear and restart: the page then starts over),
        ``rev`` (the cursor to send back), ``max_entries``, ``min_seq`` (the oldest kept, 0 if
        none) and the entries added or changed after revision ``after`` (an audio summary that
        grew comes again with the same ``seq``). Everything is returned when ``after`` is ahead
        of the ring or ``epoch`` is given and is not the current one. The newest entries whose
        encoded text fits ``prompts_response_max_chars`` (at least one)."""
        with self._lock:
            current = f"{self._boot}.{self._generation}"
            stale = after > self._rev or bool(epoch and epoch != current)
            cursor = 0 if stale else after
            kept = self._within_budget([e for e in self._entries if e.rev > cursor])
            return {
                "enabled": self._enabled,
                "epoch": current,
                "rev": self._rev,
                "min_seq": self._entries[0].seq if self._entries else 0,
                "max_entries": self._cfg.prompts_max_entries,
                "entries": [
                    {
                        "seq": e.seq,
                        "t": e.wall,
                        "route": e.route,
                        "via": e.via,
                        "label": e.label,
                        "text": e.text,
                    }
                    for e in kept
                ],
            }

    def _audio(self, route: str, via: str, size: int, marker: str | None) -> None:
        """Grow the route's audio summary (the last entry), or start one."""
        with self._lock:
            if not self._enabled:
                return
            self._rev += 1
            last = self._entries[-1] if self._entries else None
            if last is None or last.label != AUDIO_LABEL or last.route != route:
                last = self._append_locked(route, via, AUDIO_LABEL, "")
            if marker is None:
                last.audio_bytes += size
                last.audio_frames += 1
            else:
                last.markers[marker] += 1
            last.text = audio_text(last.audio_bytes, last.audio_frames, last.markers)
            last.size = encoded_len(last.text)
            last.rev = self._rev

    def _add(self, route: str, via: str, label: str, text: str) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._rev += 1
            self._append_locked(route, via, label, self._cap(text))

    def _append_locked(self, route: str, via: str, label: str, text: str) -> _Entry:
        self._seq += 1
        entry = _Entry(self._seq, self._wall(), route, via, label, text, self._rev)
        entry.size = encoded_len(text)
        self._entries.append(entry)
        self._trim_locked()
        return entry

    def _cap(self, text: str) -> str:
        limit = self._cfg.prompts_max_chars
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n... [truncated {len(text) - limit} chars]"

    def _within_budget(self, entries: list[_Entry]) -> list[_Entry]:
        budget, out = self._cfg.prompts_response_max_chars, []
        for entry in reversed(entries):
            budget -= entry.size
            if budget < 0 and out:
                break
            out.append(entry)
        return out[::-1]

    def _clear_locked(self) -> None:
        self._entries.clear()
        self._generation += 1

    def _trim_locked(self) -> None:
        while len(self._entries) > self._cfg.prompts_max_entries:
            self._entries.popleft()


def encoded_len(text: str) -> int:
    """The length of ``text`` as the web UI sends it (``json.dumps``: ASCII escapes, quotes)."""
    return len(json.dumps(text))


def audio_text(size: int, frames: int, markers: collections.Counter[str] | None = None) -> str:
    line = f"audio {size} bytes in {frames} frame{'' if frames == 1 else 's'}"
    if markers:
        line += "; " + ", ".join(f"{name} {n}" for name, n in markers.items())
    return f"[{line}]"


def b64_bytes(text: str) -> int:
    """The bytes a base64 text carries, read from its length without decoding it."""
    padding = len(text) - len(text.rstrip("="))
    return len(text) // BASE64_CHARS * BASE64_BYTES - padding


def frame_audio_bytes(frame: dict[str, Any]) -> int | None:
    """The audio bytes of an audio frame (OpenAI ``input_audio_buffer.append``, Gemini
    ``realtimeInput.audio``), None for any other frame."""
    if frame.get("type") == OPENAI_AUDIO_APPEND:
        audio = frame.get("audio")
        return b64_bytes(audio) if isinstance(audio, str) else 0
    rt_input = frame.get(GEMINI_INPUT)
    if isinstance(rt_input, dict) and "audio" in rt_input:
        data = rt_input["audio"].get("data") if isinstance(rt_input["audio"], dict) else None
        return b64_bytes(data) if isinstance(data, str) else 0
    return None


def audio_marker(frame: dict[str, Any]) -> str | None:
    """The name of a frame that only marks the audio stream (a commit, an activity start or
    end), None for any other frame."""
    kind = frame.get("type")
    if kind in OPENAI_AUDIO_MARKERS:
        return str(kind).rsplit(".", 1)[-1]
    rt_input = frame.get(GEMINI_INPUT)
    if isinstance(rt_input, dict) and len(rt_input) == 1:
        inner = next(iter(rt_input))
        if inner in GEMINI_AUDIO_MARKERS:
            return inner
    return None


def frame_label(frame: dict[str, Any]) -> str:
    """The OpenAI event's ``type``, else Gemini's top-level key (``setup``...), with the inner
    key for ``realtimeInput`` (``realtimeInput.video``)."""
    kind = frame.get("type")
    if isinstance(kind, str):
        return kind
    top = next(iter(frame), "frame")
    inner = frame.get(top)
    if top == GEMINI_INPUT and isinstance(inner, dict) and inner:
        return f"{top}.{next(iter(inner))}"
    return top


def data_url_bytes(text: str) -> int | None:
    """The bytes of a base64 ``data:`` URL, None for any other text."""
    if not text.startswith(DATA_URL) or DATA_URL_BASE64 not in text:
        return None
    return b64_bytes(text.split(DATA_URL_BASE64, 1)[1])


def redact(value: Any) -> Any:
    """A copy of ``value`` without binary payloads: a base64 ``data:`` URL anywhere, a text under
    ``BLOB_TEXT_KEYS`` and a ``data`` next to a ``mimeType`` or ``format`` become ``[N bytes]``,
    a part under ``BLOB_KEYS`` becomes ``[binary]``, and a list of ``NUMBER_LIST_LEN`` numbers or
    more becomes ``[N numbers]``."""
    if isinstance(value, dict):
        typed = bool(BLOB_TYPE_KEYS & value.keys())
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in BLOB_KEYS:
                out[key] = "[binary]"
            elif (
                isinstance(item, str)
                and item
                and (key in BLOB_TEXT_KEYS or (typed and key == BLOB_DATA_KEY))
            ):
                size = data_url_bytes(item)
                out[key] = f"[{b64_bytes(item) if size is None else size} bytes]"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        numbers = sum(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
        if numbers >= NUMBER_LIST_LEN:
            return f"[{len(value)} numbers]"
        return [redact(item) for item in value]
    if isinstance(value, str):
        size = data_url_bytes(value)
        return value if size is None else f"[{size} bytes]"
    return value


def render_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """One ``[role] text`` block per message, an assistant's tool calls as
    ``[assistant] -> name(args)``, then ``[tools] name, ...``. Content parts other than text
    are redacted (``redact``)."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?"))
        content = redact(msg.get("content"))
        if isinstance(content, list):  # content parts: text as is, anything else redacted
            content = "\n".join(
                str(p.get("text")) if isinstance(p, dict) and "text" in p else json.dumps(p)
                for p in content
            )
        if content:
            lines.append(f"[{role}] {content}")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            lines.append(f"[{role}] -> {fn.get('name', '?')}({fn.get('arguments', '')})")
    if tools:
        names = [str(t.get("function", {}).get("name", "?")) for t in tools if isinstance(t, dict)]
        lines.append(f"[tools] {', '.join(names)}")
    return "\n".join(lines)
