"""Decision events for the Live tab's "What's happening" panel (SPEC 11A, 12, 12A).

Every decision the owner may want explained (a cue played, an utterance not answered, a barge-in,
a session reset) publishes one small event on the bus topic ``diag.event``: a ``kind``, a
``reason`` code, a few scalars and the wall-clock time. The web UI keeps the last ones in an
:class:`EventRing` and builds the sentence from the codes in the page's language, so core stays
language-free.

Producers hold a settable ``events`` attribute (an :data:`EventSink`, :func:`no_events` until the
runtime wires :class:`BusEvents`), so publishing costs one small dict and never raises or waits on
I/O: the bus calls its subscribers in the publisher's thread and the ring holds a leaf lock only
for a compare and a deque append.

Privacy (SPEC 12): an event holds codes and numbers only. :func:`clean` keeps a field only when
it is a bool, an int, a finite float (rounded), None or a short lower-case code; anything else
(text, audio, an embedding, a key, a name) is dropped before the event leaves the producer.
"""

from __future__ import annotations

import collections
import itertools
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

TOPIC = "diag.event"
CODE = re.compile(r"[a-z][a-z0-9_.]{0,47}")  # a code: never a sentence, a name or a key
OTHER = "other"  # a kind or reason that is not a code
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
        "skipped",
        "skipped_late",
        "stop_consumed",
        "replaced",
    ),
    "bargein": ("cut", "released"),  # a candidate episode closed
    "session": ("reset", "reconnected", "connect_failed", "lost", "server_error"),
    "audio": ("transcript_timeout",),  # the server's transcript of sent audio never came
    "facing": ("cpu_high",),  # the facing check did not trust the head yaw
    "turn": (  # a turn ended (its outcome), or the model's skip was refused
        "ok",
        "skip_refused",
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

EventSink = Callable[..., None]


def no_events(kind: str, reason: str, **fields: Any) -> None:
    """The sink of a producer the runtime did not wire: nothing happens."""


def code(value: Any) -> str:
    """``value`` when it is a code, else ``OTHER``."""
    return value if isinstance(value, str) and CODE.fullmatch(value) else OTHER


def clean(fields: dict[str, Any]) -> dict[str, Any]:
    """The fields that may leave a producer: code keys with a bool, int, finite float, None or
    code value. Everything else is dropped (SPEC 12)."""
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if not (isinstance(key, str) and CODE.fullmatch(key)):
            continue
        if value is None or isinstance(value, bool | int):
            out[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                out[key] = round(value, DIGITS)
        elif isinstance(value, str) and CODE.fullmatch(value):
            out[key] = value
    return out


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
            event = {**clean(fields), "kind": code(kind), "reason": code(reason), "t": self._wall()}
            self._bus.publish(TOPIC, event, self._clock())
        except Exception as exc:
            log.warning("diag: event not published (%s)", type(exc).__name__)


class EventRing:
    """The last ``webui.live.activity_events`` events, newest first on read, each with a ``seq``.
    ``on_message`` (a bus subscriber, any thread) appends under a leaf lock held for a compare
    and an append; ``apply_config`` resizes, and turning the Live tab or the panel off forgets
    them and stops collecting."""

    def __init__(self, live_cfg: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seq = itertools.count(1)
        self._lock = threading.Lock()  # resize and clear only
        self._events: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=live_cfg.activity_events
        )
        self._on = False
        self.apply_config(live_cfg)

    def apply_config(self, live_cfg: Any) -> None:
        """Hot reload of ``webui.live``: collect while ``enabled`` and ``activity`` are on."""
        on = bool(live_cfg.enabled and live_cfg.activity)
        with self._lock:
            if not on:
                self._events.clear()
            if self._events.maxlen != live_cfg.activity_events:
                self._events = collections.deque(self._events, maxlen=live_cfg.activity_events)
            self._on = on

    def attach(self, bus: Any) -> Callable[[], None]:
        """Subscribe to ``diag.event`` on ``bus``; returns the unsubscribe."""
        return bus.subscribe(TOPIC, self.on_message)

    def on_message(self, msg: Any) -> None:
        """Keep one event. One that repeats the newest one's kind and reason replaces it with
        ``count`` one higher, so a run of the same rejection (a television) cannot push the
        rest out."""
        if not self._on:
            return
        event = {**msg.payload, "seq": next(self._seq), "_at": msg.stamp, "count": 1}
        with self._lock:  # a leaf lock held for a compare and an append
            events = self._events
            last = events[-1] if events else None
            if last is not None and (last["kind"], last["reason"]) == (
                event["kind"],
                event["reason"],
            ):
                event["count"] = last["count"] + 1
                events.pop()
            events.append(event)

    def events(self) -> list[dict[str, Any]]:
        """The events, newest first, without the internal monotonic stamp."""
        rows = list(self._events)
        return [{k: v for k, v in e.items() if k != "_at"} for e in reversed(rows)]

    def age_of(self, kind: str, reason: str) -> float | None:
        """Seconds since the newest ``kind``/``reason`` event, None when none is kept."""
        for e in reversed(list(self._events)):
            if e.get("kind") == kind and e.get("reason") == reason:
                return max(0.0, self._clock() - e["_at"])
        return None


__all__ = [
    "REASONS",
    "TOPIC",
    "BusEvents",
    "EventRing",
    "EventSink",
    "clean",
    "code",
    "no_events",
]
