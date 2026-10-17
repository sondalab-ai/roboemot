"""Whether an utterance is addressed to the robot (SPEC 8 stage 3, ``voice.wake_word_policy``).

The wake word is *required* with policy ``always``, never with ``never``, and with
``unless_enrolled_in_view`` only while no enrolled person is in view (strangers must say it). An
utterance is addressed when it is not required, or it contains the wake word, or it starts
within ``voice.fastpath.addressed_window_s`` after the last speech of an addressed utterance
that needed it. Every addressed utterance that needed the wake word extends the window from its
own last speech, so a conversation does not time out mid-way; an utterance that did not need it
(an enrolled person in view) opens nothing for strangers.
"""

from __future__ import annotations

import math
import threading


class WakeGate:
    """Thread-safe addressed-window state machine; stamps are in the audio clock (seconds)."""

    def __init__(self, policy: str, window_s: float) -> None:
        self._lock = threading.Lock()
        self._policy = policy
        self._window = window_s
        self._until = -math.inf

    def apply(self, policy: str, window_s: float) -> None:
        """Take a reloaded policy and window length; an open window stays open."""
        with self._lock:
            self._policy, self._window = policy, window_s

    def decide(
        self, *, wake_word: bool, enrolled_in_view: bool, start: float, last_speech: float
    ) -> bool:
        """Whether the utterance (first and last speech stamps) is addressed to the robot."""
        with self._lock:
            required = self._policy == "always" or (
                self._policy == "unless_enrolled_in_view" and not enrolled_in_view
            )
            if not required:
                return True
            if wake_word or start <= self._until:
                self._until = last_speech + self._window
                return True
            return False
