"""Speaker for the Reachy Mini: chunked, non-blocking playback with barge-in (SPEC 4A, 8).

Coded against reachy-mini 1.12's documented interface, not exercised against a real robot:
``mini.media.start_playing()``, ``push_audio_sample(float32)`` (an appsrc push that does not wait
for playback), ``get_output_audio_samplerate()`` and ``mini.media.audio.clear_player()``, which
drops queued playback at once (older SDKs lack it; ``stop_playing`` then ``start_playing`` is the
fallback). The SDK does not report how much is queued, so the speaker counts: every pushed
sample extends ``playing_until()``, a monotonic stamp, and ``stop`` resets it. The stamp feeds
``SpeechActivity`` (the echo gate) and tells barge-in what is still to be cut.

Privacy (SPEC 12): audio passes through; nothing is kept, logged or published.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from roboemot.voice.activity import SpeechActivity

from .resample import resample

log = logging.getLogger(__name__)
MS_PER_S = 1000.0


class ReachySpeaker:
    """Plays float32 mono audio of any sample rate through the SDK media."""

    def __init__(
        self,
        mini: Any,
        chunk_ms: float,
        activity: SpeechActivity | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mini, self._chunk_s, self._activity, self._clock = (
            mini,
            chunk_ms / MS_PER_S,
            activity,
            clock,
        )
        self._lock = threading.RLock()  # stop may run from inside a push hook
        self._until = -math.inf  # monotonic stamp at which the queued audio has played
        self._generation = 0  # bumped by stop: a push in flight abandons its remaining chunks
        self._out_hz: int | None = None
        self._playing = False
        self._pushed = 0

    def _media(self) -> Any:
        return getattr(self._mini, "media", None)

    def start_playing(self) -> bool:
        """Start the playback pipeline and read the output rate. False (and silent) when the
        robot has no audio system or the SDK reports no usable rate."""
        media = self._media()
        if media is None or getattr(media, "audio", True) is None:
            log.warning("speaker: the SDK reports no audio system")
            return False
        try:
            media.start_playing()
            rate = int(media.get_output_audio_samplerate())
        except Exception as exc:
            log.warning("speaker: start_playing: %s", type(exc).__name__)
            return False
        if rate <= 0:
            log.warning("speaker: the SDK reports no output rate")
            return False
        self._out_hz, self._playing = rate, True
        return True

    def close(self) -> None:
        """Flush anything queued and stop the playback pipeline."""
        self.stop()
        if not self._playing:
            return
        self._playing = False
        try:
            self._media().stop_playing()
        except Exception as exc:
            log.warning("speaker: stop_playing: %s", type(exc).__name__)

    def push_audio_sample(self, samples: np.ndarray, rate_hz: int) -> float:
        """Queue mono float32 ``samples`` recorded at ``rate_hz``: resampled to the output rate
        and pushed in pieces of ``chunk_ms`` without waiting for playback. Returns the stamp
        until which everything queued has played (0.0 if nothing was queued)."""
        if not self._playing or self._out_hz is None or samples.size == 0:
            return self.playing_until() if self._playing else 0.0
        out = resample(samples.astype(np.float32, copy=False).reshape(-1), rate_hz, self._out_hz)
        step = max(1, int(self._chunk_s * self._out_hz))
        with self._lock:
            generation = self._generation
        for i in range(0, out.size, step):
            piece = np.ascontiguousarray(out[i : i + step])
            with self._lock:
                if generation != self._generation:
                    break
                try:
                    self._media().push_audio_sample(piece)
                except Exception as exc:
                    log.warning("speaker: push_audio_sample: %s", type(exc).__name__)
                    break
                if generation != self._generation:  # stopped from inside the push
                    break
                self._until = max(self._until, self._clock()) + piece.size / self._out_hz
                self._pushed += piece.size
                until = self._until
            if self._activity is not None:
                self._activity.extend(until)
        return self.playing_until()

    def playing_until(self) -> float:
        """Monotonic stamp at which everything pushed so far has played (past if idle)."""
        with self._lock:
            return self._until

    def queued_s(self) -> float:
        """Seconds of pushed audio still to play."""
        return max(0.0, self.playing_until() - self._clock())

    def stop(self) -> None:
        """Barge-in: drop the queued audio now (``clear_player``, else restart the pipeline),
        abandon the pieces of a push in flight and reset the queued duration."""
        with self._lock:
            self._generation += 1
            self._until = self._clock()
            now = self._until
        if self._playing:
            audio = getattr(self._media(), "audio", None)
            clear = getattr(audio, "clear_player", None)
            try:
                if callable(clear):
                    clear()
                else:
                    self._media().stop_playing()
                    self._media().start_playing()
            except Exception as exc:
                log.warning("speaker: flush: %s", type(exc).__name__)
        if self._activity is not None:
            self._activity.reset(now)

    def snapshot(self) -> dict[str, Any]:
        """State for the debug report: counts and durations only."""
        return {
            "playing": self._playing,
            "output_rate_hz": self._out_hz,
            "queued_s": self.queued_s(),
            "pushed_samples": self._pushed,
        }
