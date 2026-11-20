"""Decision events for the Live tab's "What's happening" panel (SPEC 11A, 12, 12A).

Every decision the owner may want explained (a cue played, an utterance not answered, a barge-in,
a session reset) publishes one small event on the bus topic ``diag.event``: a ``kind``, a
``reason`` code, a few scalars and the wall-clock time. The web UI keeps the last ones in an
:class:`EventRing` and builds the sentence from the codes in the page's language, so core stays
language-free.

Producers hold a settable ``events`` attribute (an :data:`EventSink`, :func:`no_events` until the
runtime wires :class:`BusEvents`), so publishing costs one small dict and never raises or waits on
I/O: the bus calls its subscribers in the publisher's thread and the ring holds a leaf lock only
for a compare, a deque append and a pass over its small ring.

Privacy (SPEC 12): an event holds codes and numbers only, by allow-list. ``kind`` and ``reason``
must be in ``REASONS``; :func:`clean` keeps only the declared fields (``FIELDS``): a number field
a finite number, a flag a bool, a code field one of its enumerated values (any other string
becomes ``OTHER``). A word, a name, a key, audio or an embedding never leaves the producer.

One exception, by allow-list too (``TEXT_EVENTS``): two radio events may carry words as
``heard``, so the owner can see what a recognizer wrote for the name (SPEC 8). The
``unanswered``/``media_unconfirmed`` event (a wake over the radio the server's transcript did not
confirm) carries the server's transcript; the ``unanswered``/``media`` event (the local wake gate
rejected an utterance over the radio for no wake word) the local recognizer's final, so it can
hold speech nobody addressed to the robot. The listener puts them there only while the Live tab
and its transcript are on (``webui.live.transcript``), cut to ``webui.live.transcript_max_chars``;
the ring keeps them under the transcript's own rules (off, asleep, ``transcript_max_age_s``).
Nothing here logs an event, and no debug report reads the ring.
"""

from __future__ import annotations

import collections
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

TOPIC = "diag.event"
OTHER = "other"  # a kind, reason or field value outside its allowed set
DIGITS = 2  # decimals kept in a float field

# The kinds and the reasons each one is published with (the page has a sentence for each).
REASONS: dict[str, tuple[str, ...]] = {
    # a cue was asked for: the reason is why (the turn's outcome for the fallback)
    "cue": (
        "offline",
        "no_transcript",
        "audio_sent",
        "no_voice",
        "no_route",
        "error",
        "no_text",
        "realtime_lost",
    ),
    # an utterance the robot heard and does not answer: the rule that decided it
    "unanswered": (
        "no_window",
        "other_speaker",
        "no_face_at_voice",
        "not_facing",
        "no_facing_data",
        "music",
        "media",
        "media_unconfirmed",  # a wake over the radio the server's transcript did not confirm
        "skipped",
        "skipped_late",
        "stop_consumed",
        "replaced",
    ),
    "bargein": ("cut", "cut_own", "released"),  # a candidate episode closed (cut_own: a hum)
    "session": (
        "reset",
        "reconnected",
        "connect_failed",
        "lost",
        "server_error",
        "go_away",
        "idle_close",
        "language_refused",  # a gemini_live setup closed for its languageCodes (SPEC 8)
    ),  # fmt: skip
    "audio": ("transcript_timeout",),  # the server's transcript of sent audio never came
    "facing": ("cpu_high",),  # the facing check did not trust the head yaw
    "alone": ("start", "face", "sound"),  # the robot holds still, or what ended it (SPEC 5A)
    "nap": ("start", "face", "speech", "light", "motion", "loud"),  # began, or what woke it (7A)
    "media": (
        "fist_stop",  # a fist held up to the robot stopped the radio (SPEC 5, 11)
        "voice_stop",  # a stop said over the radio without the name stopped it (SPEC 8, 11)
        "face_wake",  # a voice from a face over the radio went for the server to confirm (SPEC 8)
    ),
    "turn": (  # a turn ended (its outcome), the model's skip was refused or its tone was invalid
        "ok",
        "skip_refused",
        "tone_invalid",
        "not_heard",
        "offline",
        "no_voice",
        "no_route",
        "error",
        "no_text",
        "realtime_lost",
        "interrupted",
        "skipped",
        "stopped",
        "confirmed",
        "declined",
        "capped",
    ),
}

# Values a free-valued field may carry: the closed sets that feed it today. Anything else becomes
# ``OTHER``, so no future producer can put a word, a name or a key in an event by mistake.
ROUTE_REASONS = (
    # route states and blocks (roboemot.realtime.route STATES, BLOCKS)
    "off", "blocked", "connecting", "connected", "closed", "fallback_local", "no_route",
    "no_opt_in", "no_extra", "provider", "pseudonyms", "budget", "unavailable",
    # the robot's own closes and resets (OWN_CLOSES, the reset reasons)
    "conversation_end", "asleep", "shutdown", "reconfigured", "expired", "input_reset",
    "pending_calls", "go_away", "idle_close", "server_refused", "setup_unready",
    # the client's close and connect reasons (roboemot.realtime.client)
    "closed_by_server", "connect_failed", "connect_timeout", "connection_lost",
    "key_unavailable", "oversize", "receive_failed", "send_failed", "send_stuck", "stall",
    # a response run's own (roboemot.realtime.turn)
    "no_session",
)  # fmt: skip
SERVER_CODES = (  # server error codes of the realtime subset the page can name
    "function_call_output_pending", "invalid_request_error", "invalid_value", "server_error",
    "rate_limit_exceeded", "session_expired", "conversation_already_has_active_response",
    "response_cancel_not_active", "item_not_found", "input_audio_buffer_commit_empty",
)  # fmt: skip
EXCEPTIONS = {  # exception class name -> the code a turn failure carries
    "TimeoutError": "timeout",
    "ConnectionError": "connection",
    "OSError": "os_error",
    "RuntimeError": "runtime_error",
    "ValueError": "value_error",
    "KeyError": "key_error",
    "TypeError": "type_error",
    "CancelledError": "cancelled",
}
CUE_KINDS = ("listening", "offline", "not_heard", "fallback")
CUE_RESULTS = (
    "played", "disabled", "muted", "unsupported", "sound_dropped", "sound_skipped", "skipped",
    "dropped",
)  # fmt: skip
FAILURES = ("not_ready", "no_session", "refused", "closed", "exception")
PATHS = ("vad", "onset", "keyword")
NUMBER, FLAG = "number", "flag"  # a field that holds a number, a field that holds a bool
TEXT = "text"  # a field that holds words: kept only on an event of ``TEXT_EVENTS``
# the only ones text rides on: the server's words of an unconfirmed radio wake, the local final
# of a radio rejection
TEXT_EVENTS = frozenset({("unanswered", "media_unconfirmed"), ("unanswered", "media")})

FIELDS: dict[str, tuple[str, ...] | str] = {
    "cue": CUE_KINDS,
    "result": CUE_RESULTS,
    "failure": FAILURES,
    "detail": (*ROUTE_REASONS, *SERVER_CODES, *EXCEPTIONS.values()),
    "reset": ROUTE_REASONS,
    "policy": ("always", "never", "unless_enrolled_in_view"),
    "facing": ("passed", "not_facing", "cpu_high", "no_data"),
    "conversation": ("none", "in_progress", "addressed"),
    "path": ("heard", "no_source", "silent", "muted"),
    "voice": ("realtime", "local"),
    "fired": PATHS,
    "confirmed": PATHS,
    "released_by": ("quiet", "speech_end"),
    "why": ROUTE_REASONS,
    "after": ROUTE_REASONS,
    "counter": ("resets", "input_resets", "pending_call_resets"),
    "code": SERVER_CODES,
    "yaw_deg": NUMBER,
    "limit_deg": NUMBER,
    "limit": NUMBER,
    "cpu_percent": NUMBER,
    "cpu_max_percent": NUMBER,
    "timeout_s": NUMBER,
    "level_db": NUMBER,
    "vad_peak": NUMBER,
    "echo_corr": NUMBER,
    "speech_ms": NUMBER,
    "in_row": NUMBER,
    "delay_s": NUMBER,
    "rounds": NUMBER,
    "held_s": NUMBER,
    "robot_playing": FLAG,
    "moving": FLAG,
    "ducking": FLAG,
    "request": FLAG,
    "heard": TEXT,  # a recognizer's words of a radio wake or rejection (TEXT_EVENTS)
}
TEXT_FIELDS = tuple(key for key, spec in FIELDS.items() if spec == TEXT)

EventSink = Callable[..., None]


def no_events(kind: str, reason: str, **fields: Any) -> None:
    """The sink of a producer the runtime did not wire: nothing happens."""


def code(value: Any, allowed: tuple[str, ...]) -> str:
    """``value`` when it is one of ``allowed``, else ``OTHER``."""
    return value if isinstance(value, str) and value in allowed else OTHER


def exception_code(exc: BaseException) -> str:
    """The code of an exception's class (``EXCEPTIONS``), ``OTHER`` for any other."""
    return EXCEPTIONS.get(type(exc).__name__, OTHER)


def clean(fields: dict[str, Any], *, text: bool = False) -> dict[str, Any]:
    """The fields that may leave a producer (SPEC 12): only the declared ones (``FIELDS``), a
    number field with a finite number, a flag with a bool, a code field with one of its values
    (``OTHER`` for any other string); None for any of them. A text field is kept, as a string,
    only with ``text`` (an event of ``TEXT_EVENTS``). Everything else is dropped."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        spec = FIELDS.get(key) if isinstance(key, str) else None
        if spec is None or (spec == TEXT and not text):
            continue
        if value is None:
            out[key] = None
        elif spec == TEXT:
            if isinstance(value, str):
                out[key] = value
        elif spec == FLAG:
            if isinstance(value, bool):
                out[key] = value
        elif spec == NUMBER:
            number = isinstance(value, int | float) and not isinstance(value, bool)
            if number and math.isfinite(value):
                out[key] = round(value, DIGITS) if isinstance(value, float) else value
        elif isinstance(value, str):
            out[key] = code(value, spec)  # type: ignore[arg-type]
    return out


def event_kind(kind: Any, reason: Any) -> tuple[str, str]:
    """``kind`` and ``reason`` when the catalogue (``REASONS``) has them, else ``OTHER``."""
    if not (isinstance(kind, str) and kind in REASONS):
        return OTHER, OTHER
    return kind, code(reason, REASONS[kind])


class BusEvents:
    """The runtime's event sink: ``(kind, reason, **fields)`` -> one ``diag.event`` on ``bus``.
    Never raises: a failing subscriber is logged by type and the decision goes on."""

    def __init__(
        self,
        bus: Any,
        wall: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus, self._wall, self._clock = bus, wall, clock

    def __call__(self, kind: str, reason: str, **fields: Any) -> None:
        try:
            kind, reason = event_kind(kind, reason)
            text = (kind, reason) in TEXT_EVENTS
            event = {**clean(fields, text=text), "kind": kind, "reason": reason, "t": self._wall()}
            self._bus.publish(TOPIC, event, self._clock())
        except Exception as exc:
            log.warning("diag: event not published (%s)", type(exc).__name__)


class EventRing:
    """The last ``webui.live.activity_events`` events, newest first on read, each with a ``seq``.
    ``on_message`` (a bus subscriber, any thread) appends under a leaf lock held for a compare,
    an append and a pass over the (small) ring for the text age bound; ``apply_config``
    resizes, and turning the Live tab or the panel off forgets them and stops collecting. A text
    field (``TEXT_FIELDS``) follows the live transcript's rules (SPEC 12): kept only while
    ``webui.live.transcript`` is on and the robot is awake, cut to ``transcript_max_chars``, and
    forgotten past ``transcript_max_age_s`` (on append, on read and on the app tick's ``trim``),
    when the transcript turns off, on its Clear (``forget_text``) or when the robot falls asleep;
    its event stays, without the words."""

    def __init__(self, live_cfg: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seq = itertools.count(1)
        self._lock = threading.Lock()  # leaf: every read and write of the ring
        self._events: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=live_cfg.activity_events
        )
        self._on = False
        self._asleep = False
        self.apply_config(live_cfg)

    def apply_config(self, live_cfg: Any) -> None:
        """Hot reload of ``webui.live``: collect while ``enabled`` and ``activity`` are on, keep
        text while ``transcript`` is on too."""
        on = bool(live_cfg.enabled and live_cfg.activity)
        with self._lock:
            if not on:
                self._events.clear()
            if self._events.maxlen != live_cfg.activity_events:
                self._events = collections.deque(self._events, maxlen=live_cfg.activity_events)
            self._on = on
            self._text = bool(live_cfg.transcript)
            self._max_chars = live_cfg.transcript_max_chars
            self._max_age_s = live_cfg.transcript_max_age_s
            if not self._text:
                self._forget_text_locked(None)

    def set_asleep(self, asleep: bool) -> None:
        """The robot fell asleep (True: its events lose their words and no new words are kept,
        as the live transcript is cleared) or woke up."""
        with self._lock:
            self._asleep = asleep
        if asleep:
            self.forget_text()

    def forget_text(self) -> None:
        """Drop the words of every kept event; the events stay (the transcript's Clear, a
        sleep)."""
        with self._lock:
            self._forget_text_locked(None)

    def trim(self) -> None:
        """Forget the words of events older than ``transcript_max_age_s``. Called from the app
        tick, so they do not outlive the bound in memory when nobody polls."""
        if self._events:
            with self._lock:
                self._forget_text_locked(self._clock() - self._max_age_s)

    def _forget_text_locked(self, before: float | None) -> None:
        """Drop the text fields of the kept events (only those stamped before ``before``)."""
        for event in self._events:
            if before is None or event["_at"] < before:
                for key in TEXT_FIELDS:
                    event.pop(key, None)

    def _text_locked(self, event: dict[str, Any]) -> None:
        """Cut ``event``'s text fields, or drop them while text is not kept."""
        for key in TEXT_FIELDS:
            if key not in event:
                continue
            if self._text and not self._asleep and isinstance(event[key], str):
                event[key] = event[key][: self._max_chars]
            else:
                del event[key]

    def attach(self, bus: Any) -> Callable[[], None]:
        """Subscribe to ``diag.event`` on ``bus``; returns the unsubscribe."""
        return bus.subscribe(TOPIC, self.on_message)

    def on_message(self, msg: Any) -> None:
        """Keep one event. One that repeats the newest one (the same kind, reason and labels;
        its numbers may differ) replaces it with ``count`` one higher, so a run of the same
        rejection (a television) cannot push the rest out."""
        with self._lock:  # a leaf lock held for a compare, an append and the text age pass
            if not self._on:
                return
            event = {**msg.payload, "seq": next(self._seq), "_at": msg.stamp, "count": 1}
            self._text_locked(event)
            self._forget_text_locked(msg.stamp - self._max_age_s)
            events = self._events
            last = events[-1] if events else None
            if last is not None and _labels(last) == _labels(event):
                event["count"] = last["count"] + 1
                events.pop()
            events.append(event)

    def events(self, *, text: bool = True) -> list[dict[str, Any]]:
        """The events, newest first, without the internal monotonic stamp (and without the
        words of one older than ``transcript_max_age_s``); with ``text`` False, without any
        words at all."""
        with self._lock:
            self._forget_text_locked(self._clock() - self._max_age_s)
            rows = list(self._events)
        drop = ("_at",) if text else ("_at", *TEXT_FIELDS)
        return [{k: v for k, v in e.items() if k not in drop} for e in reversed(rows)]

    def age_of(self, kind: str, reason: str) -> float | None:
        """Seconds since the newest ``kind``/``reason`` event, None when none is kept."""
        with self._lock:
            rows = list(self._events)
        for e in reversed(rows):
            if e.get("kind") == kind and e.get("reason") == reason:
                return max(0.0, self._clock() - e["_at"])
        return None


RING_KEYS = ("t", "seq", "_at", "count")  # what the ring adds or what always differs


def _labels(event: dict[str, Any]) -> dict[str, Any]:
    """The fields that say what an event was (codes, flags, None), without its numbers and its
    words (``TEXT_FIELDS``): a run of unconfirmed wakes or radio rejections on a song's lyrics
    is one row with a count and the newest words, so it cannot push the rest out."""
    return {
        k: v
        for k, v in event.items()
        if k not in RING_KEYS and k not in TEXT_FIELDS and (v is None or isinstance(v, bool | str))
    }


__all__ = [
    "FIELDS",
    "OTHER",
    "REASONS",
    "ROUTE_REASONS",
    "SERVER_CODES",
    "TEXT_EVENTS",
    "TEXT_FIELDS",
    "TOPIC",
    "BusEvents",
    "EventRing",
    "EventSink",
    "clean",
    "code",
    "event_kind",
    "exception_code",
    "no_events",
]
