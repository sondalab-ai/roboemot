"""The "minor present" signal (SPEC 5, 5A Scanning, 12, glossary).

One tracker is the single source of truth for the motion loop, the scan context and, later, the
LLM router and prompt. It fails safe: it starts present and a release needs positive evidence.

- A ``percept.face`` or ``person.recognized`` payload with ``possible_minor`` set (declared or
  estimated child, or an age still ``unknown``, SPEC 5) switches it on at once. A payload that
  does not say counts as a possible minor.
- ``percept.face_frame`` is published each time the face stage processed a frame, with or
  without faces. The signal turns off only after a full hold of such frames with no possible
  minor. No frames for the hold (no camera, vision off or dead, a stalled run) keep it on.
- The hold is ``max(safety.minor_hold_s, safety.minor_hold_periods x detection period)`` so a
  throttled detector cannot make the flag flap (SPEC 12: safety only gets more conservative).

Every write to the sink (the motion loop) happens under this object's lock, together with the
read of the state it writes, so the bus thread and the app tick cannot overwrite each other.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from ..bus import Bus, Message
from ..config.schema import SafetyConfig

log = logging.getLogger(__name__)


class MinorPresence:
    """Latch with a hold and fail-safe release over the face stage's output.

    ``period`` returns the face-detection period in seconds (infinite while the stage is off).
    ``sink`` receives the current state on every change of evidence and on ``apply``. ``configure``
    caches the hold settings (call it again on a config reload). The clock is injectable.
    """

    def __init__(
        self,
        bus: Bus,
        safety: SafetyConfig,
        *,
        period: Callable[[], float] = lambda: math.inf,
        sink: Callable[[bool], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._period = period
        self._sink = sink
        self._clock = clock
        self._lock = threading.Lock()
        self._hold_s = safety.minor_hold_s
        self._hold_periods = safety.minor_hold_periods
        self._last_frame: float | None = None
        self._clear_since: float | None = None
        self._unsubscribe = [
            bus.subscribe("percept.face", self._on_face),
            bus.subscribe("person.recognized", self._on_face),
            bus.subscribe("percept.face_frame", self._on_frame),
        ]

    def configure(self, safety: SafetyConfig) -> None:
        """Take new hold settings (hot reload)."""
        with self._lock:
            self._hold_s = safety.minor_hold_s
            self._hold_periods = safety.minor_hold_periods

    def _on_face(self, msg: Message) -> None:
        with self._lock:
            if getattr(msg.payload, "possible_minor", True):
                self._clear_since = None
            self._push(self._clock())

    def _on_frame(self, msg: Message) -> None:
        with self._lock:
            now = self._clock()
            if self._clear_since is None or not self._fresh(now):
                self._clear_since = now
            self._last_frame = now
            self._push(now)

    def _hold(self) -> float:
        period = self._period()
        return max(self._hold_s, self._hold_periods * period)

    def _fresh(self, now: float) -> bool:
        return self._last_frame is not None and now - self._last_frame < self._hold()

    def _active(self, now: float) -> bool:
        if self._clear_since is None or not self._fresh(now):
            return True
        return now - self._clear_since < self._hold()

    def _push(self, now: float) -> None:
        if self._sink is None:
            return
        try:
            self._sink(self._active(now))
        except Exception as exc:
            log.warning("minor-present sink failed: %s", type(exc).__name__)

    def apply(self, now: float | None = None) -> bool:
        """Write the current state to the sink (the app tick) and return it."""
        with self._lock:
            t = self._clock() if now is None else now
            state = self._active(t)
            if self._sink is not None:
                self._push(t)
            return state

    def present(self, now: float | None = None) -> bool:
        """True unless face frames ran without a possible minor for a full hold."""
        with self._lock:
            return self._active(self._clock() if now is None else now)

    def close(self) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()

    def snapshot(self) -> dict[str, Any]:
        """For the debug report: the flag and whether the face stage is delivering frames."""
        with self._lock:
            now = self._clock()
            return {"minor_present": self._active(now), "face_frames_fresh": self._fresh(now)}
