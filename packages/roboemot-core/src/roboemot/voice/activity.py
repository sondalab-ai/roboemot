"""Whether the robot is speaking, for the gates that must not hear its own voice (SPEC 4A, 8, 10).

The speaker reports the monotonic stamp until which queued audio plays; the robot counts as
speaking until that stamp plus ``voice.echo_tail_ms`` (room echo and the microphone's own latency
outlive the last sample). The reflex engine skips ``loud_onset`` and the app mutes the
direction-of-arrival source while this holds. Barge-in ends it early with ``reset``.
"""

from __future__ import annotations

import math
import threading

MS_PER_S = 1000.0


class SpeechActivity:
    """Thread-safe "robot is speaking until ``playing_until + echo_tail``"."""

    def __init__(self, echo_tail_ms: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._tail_s = max(0.0, echo_tail_ms) / MS_PER_S
        self._playing_until = -math.inf

    def set_echo_tail_ms(self, echo_tail_ms: float) -> None:
        """Apply a reloaded ``voice.echo_tail_ms``."""
        with self._lock:
            self._tail_s = max(0.0, echo_tail_ms) / MS_PER_S

    def extend(self, until: float) -> None:
        """Audio is queued to play until ``until``; never moves the end earlier."""
        if math.isfinite(until):
            with self._lock:
                self._playing_until = max(self._playing_until, until)

    def reset(self, now: float) -> None:
        """Playback was cut at ``now`` (barge-in): the end moves to ``now``; the echo tail stays."""
        with self._lock:
            self._playing_until = now

    def until(self) -> float:
        """Stamp until which the robot counts as speaking (``-inf`` before any speech)."""
        with self._lock:
            return self._playing_until + self._tail_s

    def speaking(self, now: float) -> bool:
        """Whether ``now`` is inside the speaking window, echo tail included."""
        return now < self.until()
