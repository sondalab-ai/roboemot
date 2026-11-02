"""Whether the robot is speaking, for the gates that must not hear its own voice (SPEC 4A, 8, 10).

The speaker reports the monotonic stamp until which queued audio plays; the robot counts as
speaking until that stamp plus ``voice.echo_tail_ms`` (room echo and the microphone's own latency
outlive the last sample). The reflex engine skips ``loud_onset`` and the app mutes the
direction-of-arrival source while this holds. Barge-in ends it early with ``reset``.

A short sound that is not speech (the listening blip, SPEC 8, Non-verbal cues) opens only the
*own-sound* window (``extend_own``): ``sounding`` covers both and is what the gates that must
not hear the robot's own sound read (the reflex engine, the sound-direction source and samples).
``speaking`` stays speech only, for the gates about the robot's voice: the listener's drop and
barge-in, the realtime input's "nothing is sent while the robot speaks", the conversation phase.
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
        self._own_until = -math.inf  # a sound that is not speech plays until then

    def set_echo_tail_ms(self, echo_tail_ms: float) -> None:
        """Apply a reloaded ``voice.echo_tail_ms``."""
        with self._lock:
            self._tail_s = max(0.0, echo_tail_ms) / MS_PER_S

    def extend(self, until: float) -> None:
        """Audio is queued to play until ``until``; never moves the end earlier."""
        if math.isfinite(until):
            with self._lock:
                self._playing_until = max(self._playing_until, until)

    def extend_own(self, until: float) -> None:
        """A sound that is not speech is queued to play until ``until``; never moves it earlier."""
        if math.isfinite(until):
            with self._lock:
                self._own_until = max(self._own_until, until)

    def reset(self, now: float) -> None:
        """Playback was cut at ``now`` (barge-in): both ends move to ``now``; the echo tail stays
        on."""
        with self._lock:
            self._playing_until = now
            self._own_until = min(self._own_until, now)

    def until(self) -> float:
        """Stamp until which the robot counts as speaking (``-inf`` before any speech)."""
        with self._lock:
            return self._playing_until + self._tail_s

    def speaking(self, now: float) -> bool:
        """Whether ``now`` is inside the speaking window, echo tail included."""
        return now < self.until()

    def sounding(self, now: float) -> bool:
        """Whether the robot's speech or another sound of its own plays at ``now``, echo tail
        included: for the gates that must not take the robot's sound for the room's."""
        with self._lock:
            end = max(self._playing_until, self._own_until) + self._tail_s
        return now < end
