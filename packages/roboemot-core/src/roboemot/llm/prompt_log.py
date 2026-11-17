"""The LLM prompts log: recent requests that left the robot for a cloud LLM, for the web UI's
Live tab (SPEC 11A, 12).

An in-memory ring of ``{seq, t, route, via, label, text}``, filled by the router (one entry per
chat request, after the history trim and the aliasing: what really leaves) and by the realtime
client (one entry per frame, after the wire codec: Gemini's ``setup``, ``clientContent``...).
Opt-in (``webui.live.prompts``, with the Live tab itself on): while either is off nothing is
recorded, and turning it off clears what was kept. Only the payload is kept: never a URL, a
header or a key. Audio is never kept: consecutive audio frames of a route become one
``[audio N bytes in K frames]`` entry that grows in place. Bounded by count and text length,
from the configuration. Never persisted, logged, repr-ed or put in a debug report: this module
has no logger and no ``__repr__`` that shows an entry.

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
from dataclasses import dataclass
from typing import Any

from roboemot.config.schema import LiveConfig

EPOCH_BYTES = 4  # random bytes naming one life of the ring (a restart gives the page a new one)
JSON_INDENT = 2  # realtime frames are shown as pretty JSON
BASE64_CHARS = 4  # every 4 base64 characters
BASE64_BYTES = 3  # carry 3 bytes
AUDIO_LABEL = "audio"
CHAT_LABEL = "chat.completions"
OPENAI_AUDIO_APPEND = "input_audio_buffer.append"
# keys whose text value is a base64 payload (audio, an image): replaced by its size wherever it
# sits (``audio`` as an object is a session's audio settings and stays)
BLOB_TEXT_KEYS = frozenset({"audio", "image_url"})
BLOB_KEYS = frozenset({"inlineData", "mediaChunks"})  # whole binary parts: always replaced


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
        self._add(route, model, CHAT_LABEL, render_chat(messages, tools), audio=0)

    def frame(self, route: str, protocol: str, frame: dict[str, Any]) -> None:
        """One realtime frame as it leaves the codec. An audio frame only grows the route's
        audio summary entry (a new one when another frame came in between)."""
        if not self._enabled:
            return
        audio = frame_audio_bytes(frame)
        if audio is not None:
            self._add(route, protocol, AUDIO_LABEL, "", audio=audio)
            return
        text = json.dumps(redact(frame), ensure_ascii=False, indent=JSON_INDENT)
        self._add(route, protocol, frame_label(frame), text, audio=0)

    def snapshot(self, after: int = 0, epoch: str = "") -> dict[str, Any]:
        """``enabled``, ``epoch`` (changes on clear and restart: the page then starts over),
        ``rev`` (the cursor to send back), ``max_entries``, ``min_seq`` (the oldest kept, 0 if
        none) and the entries added or changed after revision ``after`` (an audio summary that
        grew comes again with the same ``seq``). Everything is returned when ``after`` is ahead
        of the ring or ``epoch`` is given and is not the current one. The newest entries that
        fit ``prompts_response_max_chars`` (at least one)."""
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

    def _add(self, route: str, via: str, label: str, text: str, *, audio: int) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._rev += 1
            last = self._entries[-1] if self._entries else None
            merge = last is not None and last.label == AUDIO_LABEL and last.route == route
            if label == AUDIO_LABEL and last is not None and merge:
                last.audio_bytes += audio
                last.audio_frames += 1
                last.text = audio_text(last.audio_bytes, last.audio_frames)
                last.rev = self._rev
                return
            if label == AUDIO_LABEL:
                text = audio_text(audio, 1)
            self._seq += 1
            entry = _Entry(self._seq, self._wall(), route, via, label, self._cap(text), self._rev)
            entry.audio_bytes, entry.audio_frames = audio, int(label == AUDIO_LABEL)
            self._entries.append(entry)
            self._trim_locked()

    def _cap(self, text: str) -> str:
        limit = self._cfg.prompts_max_chars
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n... [truncated {len(text) - limit} chars]"

    def _within_budget(self, entries: list[_Entry]) -> list[_Entry]:
        budget, out = self._cfg.prompts_response_max_chars, []
        for entry in reversed(entries):
            budget -= len(entry.text)
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


def audio_text(size: int, frames: int) -> str:
    return f"[audio {size} bytes in {frames} frame{'' if frames == 1 else 's'}]"


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
    rt_input = frame.get("realtimeInput")
    if isinstance(rt_input, dict) and "audio" in rt_input:
        data = rt_input["audio"].get("data") if isinstance(rt_input["audio"], dict) else None
        return b64_bytes(data) if isinstance(data, str) else 0
    return None


def frame_label(frame: dict[str, Any]) -> str:
    """The OpenAI event's ``type``, else Gemini's one top-level key (``setup``...)."""
    kind = frame.get("type")
    if isinstance(kind, str):
        return kind
    return next(iter(frame), "frame")


def redact(value: Any) -> Any:
    """A copy of ``value`` with every base64 text under ``BLOB_TEXT_KEYS`` replaced by its size
    and every binary part under ``BLOB_KEYS`` (Gemini's ``inlineData``) by a marker."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in BLOB_TEXT_KEYS and isinstance(item, str) and item:
                out[key] = f"[{b64_bytes(item)} bytes]"
            elif key in BLOB_KEYS:
                out[key] = "[binary]"
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def render_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """One ``[role] text`` block per message, an assistant's tool calls as
    ``[assistant] -> name(args)``, then ``[tools] name, ...``."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        if isinstance(content, list):  # content parts: text as is, anything else redacted
            content = "\n".join(
                str(p.get("text")) if isinstance(p, dict) and "text" in p else json.dumps(redact(p))
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
