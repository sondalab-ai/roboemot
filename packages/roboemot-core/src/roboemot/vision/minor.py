"""The "minor present" signal (SPEC 5, 5A Scanning, 12, glossary).

One tracker is the single source of truth for the motion loop, the scan context and, later, the
LLM router and prompt. It listens to ``percept.face`` and ``person.recognized``: a face whose
``possible_minor`` is set (declared or estimated child, or an age that is still ``unknown``,
SPEC 5) switches the signal on at once. It switches off only after ``safety.minor_hold_s`` without
such a percept, so a lost track or a dropped frame never relaxes the caps (SPEC 12: safety only
gets more conservative).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from ..bus import Bus, Message


class MinorPresence:
    """Latch-with-hold over the possible-minor flag of face percepts.

    ``hold`` returns the hold time in seconds and is read live, so a config reload applies at
    once. ``on_rise`` is called (from the publishing thread) the moment the signal turns on, so
    the motion caps engage without waiting for the next app tick. The clock is injectable.
    """

    def __init__(
        self,
        bus: Bus,
        hold: Callable[[], float],
        *,
        clock: Callable[[], float] = time.monotonic,
        on_rise: Callable[[], None] | None = None,
    ) -> None:
        self._hold = hold
        self._clock = clock
        self._on_rise = on_rise
        self._lock = threading.Lock()
        self._last: float | None = None
        self._unsubscribe = [
            bus.subscribe("percept.face", self._on_message),
            bus.subscribe("person.recognized", self._on_message),
        ]

    def _on_message(self, msg: Message) -> None:
        if not getattr(msg.payload, "possible_minor", False):
            return
        now = self._clock()
        with self._lock:
            rose = not self._active(now)
            self._last = now
        if rose and self._on_rise is not None:
            self._on_rise()

    def _active(self, now: float) -> bool:
        return self._last is not None and now - self._last < self._hold()

    def present(self, now: float | None = None) -> bool:
        """True while a possible minor was seen within the hold time."""
        with self._lock:
            return self._active(self._clock() if now is None else now)

    def close(self) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()

    def snapshot(self) -> dict[str, Any]:
        """For the debug report: the flag only, never who."""
        return {"minor_present": self.present()}
