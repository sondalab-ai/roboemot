"""Whether an utterance is addressed to the robot (SPEC 8 stage 3, ``voice.wake_word_policy``).

The wake word is *required* with policy ``always``, never with ``never``, and with
``unless_enrolled_in_view`` only while no enrolled person is in view (SPEC 8 stage 2).

Every utterance that would be addressed *without* the wake word (not required, or a follow-up in
the open window) must also pass ``at_face``, the voice-at-face check (``voice.direction``): its
sound direction pointed at a visible face. Otherwise it is ``rejected_no_face_at_voice`` and needs
the wake word, so music or a television is not answered. An utterance with the wake word never
runs the check.

When required, an utterance is addressed if it contains the wake word, or it starts within
``voice.fastpath.addressed_window_s`` after the last speech of an addressed one. A follow-up
extends the window from its own last speech, but never past ``window_max_s`` from the wake word
that opened it, and when both utterances' speakers are known people they must be the same person.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

DECISIONS = (
    "not_required",
    "wake_word",
    "window",
    "rejected_no_window",
    "rejected_other_speaker",
    "rejected_no_face_at_voice",
)


class WakeGate:
    """Thread-safe addressed-window state machine; stamps are in the audio clock (seconds)."""

    def __init__(self, policy: str, window_s: float, max_s: float = math.inf) -> None:
        self._lock = threading.Lock()
        self._policy = policy
        self._window = window_s
        self._max = max_s
        self._until = -math.inf
        self._opened = -math.inf
        self._speaker: str | None = None
        self._decisions = dict.fromkeys(DECISIONS, 0)

    def apply(self, policy: str, window_s: float, max_s: float = math.inf) -> None:
        """Take a reloaded policy and window lengths; an open window stays open."""
        with self._lock:
            self._policy, self._window, self._max = policy, window_s, max_s

    def decide(
        self,
        *,
        wake_word: bool,
        enrolled_in_view: bool,
        start: float,
        last_speech: float,
        speaker: str | None = None,
        at_face: Callable[[], bool] | None = None,
    ) -> bool:
        """Whether the utterance (first and last speech stamps, speaker id if known) is addressed.
        ``at_face`` is the voice-at-face check, called only when the utterance would be addressed
        without the wake word (None: no check)."""
        with self._lock:
            required = self._policy == "always" or (
                self._policy == "unless_enrolled_in_view" and not enrolled_in_view
            )
            if not required:
                outcome = "not_required"
                if not wake_word and at_face is not None and not at_face():
                    outcome = "rejected_no_face_at_voice"
                self._decisions[outcome] += 1
                return outcome == "not_required"
            if wake_word:
                self._opened, self._speaker = last_speech, speaker
                outcome = "wake_word"
            else:
                outcome = self._follow(start, speaker)
                if outcome == "window" and at_face is not None and not at_face():
                    outcome = "rejected_no_face_at_voice"
            self._decisions[outcome] += 1
            if outcome.startswith("rejected"):
                return False
            self._until = min(last_speech + self._window, self._opened + self._max)
            return True

    def _follow(self, start: float, speaker: str | None) -> str:
        """Outcome of an utterance without the wake word: ``window`` when it falls in the open
        window (same person if both are known), else the ``rejected_*`` reason (lock held)."""
        if start > self._until:
            return "rejected_no_window"
        if speaker is None or self._speaker is None or speaker == self._speaker:
            return "window"
        return "rejected_other_speaker"

    def snapshot(self, now: float) -> dict[str, Any]:
        """Policy, window state and decision counts for the debug report (no content)."""
        with self._lock:
            return {
                "policy": self._policy,
                "window_open": now <= self._until,
                "window_left_s": max(0.0, self._until - now) if math.isfinite(self._until) else 0.0,
                "decisions": dict(self._decisions),
            }
