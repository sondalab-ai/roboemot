"""Speaker for the Reachy Mini: chunked, non-blocking playback with barge-in (SPEC 4A, 8).

Coded against reachy-mini 1.12's documented interface, not exercised against a real robot:
``mini.media.start_playing()``, ``push_audio_sample(float32)`` (an appsrc push that does not wait
for playback), ``get_output_audio_samplerate()`` and ``mini.media.audio.clear_player()``, which
drops queued playback at once (``media/audio_base.py:319``, abstract: every audio backend has it;
the package pins reachy-mini>=1.12 for it). The SDK does not report how much is queued, so the
speaker counts: every pushed sample extends ``playing_until()``, a monotonic stamp, and ``stop``
resets it. The stamp feeds ``SpeechActivity`` (the echo gate) and tells barge-in what is still to
be cut.

Playback is resampled as a stream (:class:`~.resample.StreamResampler`): the speech output pushes
a sentence in pieces (Piper: ``voice.piper.chunk_ms``), and converting each piece on its own put an
edge transient, a click, at every piece boundary. The converter keeps its filter state across the
pushes of one stream and holds back a few ms of lookahead; ``end_stream`` (called by the speech
output after each sentence) flushes that tail and starts a fresh stream, and so do ``stop`` and a
change of input rate. Input rates off a ``RATE_GRID_HZ`` grid (Piper pitch-shifted by the DSP
backend, e.g. 23907 Hz) are snapped to it: a near-coprime rate would make the converter hold back
up to a second of input, and the snap costs at most half the grid in rate (about 0.1 %, under 2
cents of pitch).

With a ``reference`` (:class:`~roboemot.voice.bargein.PlaybackReference`) the speaker also keeps
the last seconds of what it played, resampled to the reference rate and stamped with the time each
piece starts to play, for the barge-in echo check (SPEC 8 stage 8); ``stop`` cuts it at the stop.

Privacy (SPEC 12): audio passes through; the reference is a RAM ring of a few seconds that is
never logged or published.
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
from roboemot.voice.bargein import PlaybackReference

from .resample import StreamResampler, resample
from .sdk import MiniPort

log = logging.getLogger(__name__)
MS_PER_S = 1000.0
RATE_GRID_HZ = 50  # stream input rates are snapped to it; 22050, 16000 and 48000 are on it


class ReachySpeaker:
    """Plays float32 mono audio of any sample rate through the SDK media."""

    def __init__(
        self,
        mini: MiniPort,
        chunk_ms: float,
        activity: SpeechActivity | None = None,
        clock: Callable[[], float] = time.monotonic,
        reference: PlaybackReference | None = None,
    ) -> None:
        self._mini, self._chunk_s, self._activity, self._clock = (
            mini,
            chunk_ms / MS_PER_S,
            activity,
            clock,
        )
        self._reference = reference
        self._lock = threading.RLock()  # stop may run from inside a push hook
        self._until = -math.inf  # monotonic stamp at which the queued audio has played
        self._generation = 0  # bumped by stop: a push in flight abandons its remaining chunks
        self._out_hz: int | None = None
        self._playing = False
        self._pushed = 0
        self._stream: StreamResampler | None = None  # the playback stream's converter
        self._stream_hz = 0  # its input rate as given (before snapping)

    def start_playing(self) -> bool:
        """Start the playback pipeline and read the output rate. False (and silent) when the
        robot has no audio system or the SDK reports no usable rate."""
        media = self._mini.media
        if media.audio is None:
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
            self._mini.media.stop_playing()
        except Exception as exc:
            log.warning("speaker: stop_playing: %s", type(exc).__name__)

    def push_audio_sample(self, samples: np.ndarray, rate_hz: int) -> float:
        """Queue mono float32 ``samples`` recorded at ``rate_hz``: resampled to the output rate
        and pushed in pieces of ``chunk_ms`` without waiting for playback. Returns the stamp
        until which everything queued has played; 0.0 if this call queued nothing."""
        if not self._playing or self._out_hz is None or samples.size == 0 or rate_hz <= 0:
            return 0.0
        x = samples.astype(np.float32, copy=False).reshape(-1)
        with self._lock:
            generation = self._generation
            tail = self._take_stream() if self._stream_hz != rate_hz else None
            if self._stream is None and rate_hz != self._out_hz:
                snapped = max(RATE_GRID_HZ, round(rate_hz / RATE_GRID_HZ) * RATE_GRID_HZ)
                self._stream = StreamResampler(snapped, self._out_hz)
            self._stream_hz, stream = rate_hz, self._stream
        # converted outside the lock; a stop meanwhile orphans ``stream`` and the pieces are
        # abandoned below, so old audio never reaches a new stream
        out = x if stream is None else stream.process(x)
        if tail is not None and tail.size:
            out = np.concatenate([tail, out])
        queued = self._push_pieces(out, generation)
        return self.playing_until() if queued else 0.0

    def end_stream(self) -> float:
        """End the playback stream (a sentence is complete): push the converter's held-back
        tail and start a fresh stream with the next push. Returns the stamp until which
        everything queued has played; 0.0 if there was nothing to flush."""
        with self._lock:
            generation = self._generation
            tail = self._take_stream()
        if tail is None or not tail.size or not self._playing:
            return 0.0
        return self.playing_until() if self._push_pieces(tail, generation) else 0.0

    def _take_stream(self) -> np.ndarray | None:
        """Detach the current stream and return its flushed tail (None without one); the lock
        is held."""
        stream, self._stream, self._stream_hz = self._stream, None, 0
        return None if stream is None else stream.flush()

    def _push_pieces(self, out: np.ndarray, generation: int) -> bool:
        """Push ``out`` (at the output rate) in pieces of ``chunk_ms``, extending the timeline,
        the reference and the speech gate; stops at a stop or a failing push. True if any piece
        was queued."""
        assert self._out_hz is not None
        step = max(1, int(self._chunk_s * self._out_hz))
        queued = False
        for i in range(0, out.size, step):
            piece = np.ascontiguousarray(out[i : i + step])
            with self._lock:
                if generation != self._generation:
                    break
                try:
                    self._mini.media.push_audio_sample(piece)
                except Exception as exc:
                    log.warning("speaker: push_audio_sample: %s", type(exc).__name__)
                    break
                if generation != self._generation:  # stopped from inside the push
                    break
                begins = max(self._until, self._clock())
                if self._reference is not None:
                    self._reference.add(
                        begins, resample(piece, self._out_hz, self._reference.rate_hz)
                    )
                self._until = begins + piece.size / self._out_hz
                self._pushed += piece.size
                queued = True
                if self._activity is not None:  # under the lock: a stop cannot interleave
                    self._activity.extend(self._until)
        return queued

    def playing_until(self) -> float:
        """Monotonic stamp at which everything pushed so far has played (past if idle)."""
        with self._lock:
            return self._until

    def queued_s(self) -> float:
        """Seconds of pushed audio still to play."""
        return max(0.0, self.playing_until() - self._clock())

    def stop(self) -> None:
        """Barge-in: drop the queued audio now (``clear_player``),
        abandon the pieces of a push in flight and reset the queued duration."""
        with self._lock:  # held throughout: no push lands between the flush and the reset
            self._generation += 1
            self._stream, self._stream_hz = None, 0  # the next push starts a fresh stream
            self._until = now = self._clock()
            if self._reference is not None:
                self._reference.truncate(now)
            if self._playing:
                audio = self._mini.media.audio
                try:
                    if audio is not None:
                        audio.clear_player()
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
