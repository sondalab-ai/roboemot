"""What the conversation is doing right now, for the web UI's Live tab (SPEC 8, 11A, 12).

One small thread-safe object. The voice listener reports "an utterance is in progress"
(``set_listening``), the conversation worker reports "an LLM stream is in flight" and which route
it is on (``begin_thinking``, ``set_route``, ``end_thinking``), and "speaking" is read from the
speech gate when asked. It never holds text, prompts or replies: only a phase name, a route name
and a start stamp.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

IDLE, LISTENING, THINKING, SPEAKING = "idle", "listening", "thinking", "speaking"


class ConversationPhase:
    """Phase priority: speaking, then thinking, then listening, then idle. A barge-in therefore
    goes speaking -> listening (the speech is cut) while the interrupted stream ends."""

    def __init__(
        self,
        speaking: Callable[[float], bool] = lambda now: False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._speaking = speaking
        self._clock = clock
        self._lock = threading.Lock()
        self._listening = False
        self._since: float | None = None  # start of the stream in flight; None: none in flight
        self._route: str | None = None

    def set_listening(self, active: bool) -> None:
        """The listener has (True) or no longer has (False) an utterance in progress."""
        with self._lock:
            self._listening = active

    @property
    def listening(self) -> bool:
        """An utterance is in progress, whatever the phase (a barge-in or a user who talks
        while the answer streams is listening and thinking or speaking at once)."""
        with self._lock:
            return self._listening

    def begin_thinking(self) -> None:
        """A request is about to be sent: the route is not known until the first event."""
        with self._lock:
            self._since, self._route = self._clock(), None

    def set_route(self, name: str) -> None:
        """The route the stream in flight is on (a route name from the configuration)."""
        with self._lock:
            if self._since is not None:
                self._route = name

    def end_thinking(self) -> None:
        """The turn ended (answer done, error, cancel or barge-in); safe to call twice."""
        with self._lock:
            self._since = self._route = None

    def snapshot(self) -> dict[str, Any]:
        """``phase`` plus, while a stream is in flight (even if the robot already speaks the
        first sentences), its ``route`` (None until known) and ``elapsed_s``."""
        now = self._clock()
        with self._lock:
            listening, since, route = self._listening, self._since, self._route
        speaking = self._speaking(now)
        phase = (
            SPEAKING
            if speaking
            else THINKING
            if since is not None
            else LISTENING
            if listening
            else IDLE
        )
        out: dict[str, Any] = {"phase": phase, "route": None, "elapsed_s": None}
        if since is not None:
            out["route"], out["elapsed_s"] = route, max(0.0, now - since)
        return out
