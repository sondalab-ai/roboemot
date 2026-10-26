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

Pacing (SPEC 8 stage 8, 13): with ``lead_ms`` > 0 (``voice.playback_lead_ms``, 200 ms) the
speaker hands the SDK player at most that much audio ahead of playback and keeps the rest pending;
a pacer thread hands the next piece as the player drains. That bounds what barge-in ducking cannot
reach: the duck (:class:`~roboemot.voice.duck.DuckGain`, given by the speech output through
``take_duck``) is applied at the hand-off, per piece, keyed to the piece's play time. The SDK's
appsrc is not capped (``set_max_output_buffers`` is leaky and would drop speech), and its sink
buffers about 50 ms, so the pacer counts an *underrun* when it hands a piece after the player ran
dry, and a *late* hand-off when the gap passed the SDK's 200 ms gap reset (the buffer is restamped
with DISCONT). ``playing_until`` still reports the end of everything accepted; the echo reference
and the speech gate are stamped at the hand-off, with the audio as played (after the gain).
``stop`` drops the pending pieces too (``pending_at_stop_*_ms``), so the SDK flush is at most the
lead. A lead of 0 hands everything over at once, as before pacing.

Privacy (SPEC 12): audio passes through; the reference is a RAM ring of a few seconds that is
never logged or published.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.duck import DuckGain

from .resample import StreamResampler, resample
from .sdk import MiniPort

log = logging.getLogger(__name__)
MS_PER_S = 1000.0
RATE_GRID_HZ = 50  # stream input rates are snapped to it; 22050, 16000 and 48000 are on it
# reachy-mini 1.12 ``AudioBase.GAP_RESET_NS``: a buffer this late is restamped (DISCONT)
SDK_GAP_RESET_S = 0.2
PACER_JOIN_S = 1.0


class ReachySpeaker:
    """Plays float32 mono audio of any sample rate through the SDK media."""

    def __init__(
        self,
        mini: MiniPort,
        chunk_ms: float,
        activity: SpeechActivity | None = None,
        clock: Callable[[], float] = time.monotonic,
        reference: PlaybackReference | None = None,
        lead_ms: float = 0.0,
        pace_thread: bool = True,
    ) -> None:
        self._mini, self._chunk_s, self._activity, self._clock = (
            mini,
            chunk_ms / MS_PER_S,
            activity,
            clock,
        )
        self._reference = reference
        self._lock = threading.RLock()  # stop may run from inside a push hook
        self._generation = 0  # bumped by stop: a push in flight abandons its remaining chunks
        self._out_hz: int | None = None
        self._playing = False
        self._pushed = 0
        self._stream: StreamResampler | None = None  # the playback stream's converter
        self._stream_hz = 0  # its input rate as given (before snapping)
        self._lead_s = lead_ms / MS_PER_S
        self._pace_thread = pace_thread
        self._cv = threading.Condition(self._lock)
        # pieces accepted but not handed to the player: (samples at the output rate, accepted at)
        self._pending: collections.deque[tuple[np.ndarray, float]] = collections.deque()
        self._pending_n = 0
        self._handed = -math.inf  # monotonic stamp at which the audio handed over has played
        self._duck: DuckGain | None = None
        self._pacer: threading.Thread | None = None
        self._closing = False
        self._counts = {"pieces": 0, "underruns": 0, "late": 0}
        self._underrun_ms = 0.0
        self._at_stop_ms: list[float] = []  # last and max pending audio dropped by a stop

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
        if self._lead_s > 0 and self._pace_thread and self._pacer is None:
            self._closing = False
            self._pacer = threading.Thread(target=self._pace, name="speaker-pacer", daemon=True)
            self._pacer.start()
        return True

    def close(self) -> None:
        """Flush anything queued, join the pacer and stop the playback pipeline."""
        self.stop()
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        pacer, self._pacer = self._pacer, None
        if pacer is not None:
            pacer.join(PACER_JOIN_S)
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
        """Accept ``out`` (at the output rate) in pieces of ``chunk_ms``: pending, then handed to
        the player as far as the lead allows (all at once with no lead). Stops at a stop or a
        failing push. True if any piece was accepted."""
        assert self._out_hz is not None
        step = max(1, int(self._chunk_s * self._out_hz))
        queued = False
        for i in range(0, out.size, step):
            piece = np.ascontiguousarray(out[i : i + step])
            with self._cv:
                if generation != self._generation:
                    break
                self._pending.append((piece, self._clock()))
                self._pending_n += piece.size
                queued = True
                if not self._feed_locked():
                    break
                if generation != self._generation:  # stopped from inside the push
                    break
                self._cv.notify_all()
        return queued

    def take_duck(self, duck: DuckGain | None) -> bool:
        """The speech output hands over barge-in's duck: the gain is applied here, at the
        hand-off (so the speech output does not apply it again)."""
        with self._lock:
            self._duck = duck
        return True

    def feed(self) -> float | None:
        """Hand the player what the lead allows now (the pacer's step; tests call it on a fake
        clock). Returns the seconds until the next piece is due, None when nothing is pending."""
        with self._cv:
            self._feed_locked()
            return self._next_due()

    def _feed_locked(self) -> bool:
        """Hand pending pieces while the player holds less than the lead (every piece with no
        lead); the lock is held. False when a push failed (the pending audio is dropped)."""
        while self._pending:
            now = self._clock()
            piece, accepted = self._pending[0]
            held = self._handed - now
            if self._lead_s > 0 and held > 0 and held + piece.size / self._out_hz > self._lead_s:
                return True
            self._pending.popleft()
            self._pending_n -= piece.size
            if not self._hand(piece, accepted, now):
                self._drop_pending()
                return False
        return True

    def _hand(self, piece: np.ndarray, accepted: float, now: float) -> bool:
        """Push one piece to the player with the duck's gain, keyed to when it plays; stamp the
        reference and the gate. The lock is held."""
        assert self._out_hz is not None
        begins = max(self._handed, now)
        if math.isfinite(self._handed) and now > self._handed and accepted <= self._handed:
            gap = now - self._handed  # it was pending in time: the player ran dry waiting
            self._counts["underruns"] += 1
            self._counts["late"] += gap > SDK_GAP_RESET_S
            self._underrun_ms += gap * MS_PER_S
        duck = self._duck
        played = piece if duck is None else duck.apply(piece, self._out_hz, begins)
        generation = self._generation
        try:
            self._mini.media.push_audio_sample(played)
        except Exception as exc:
            log.warning("speaker: push_audio_sample: %s", type(exc).__name__)
            return False
        if generation != self._generation:  # stopped from inside the push
            return True
        if self._reference is not None:
            self._reference.add(begins, resample(played, self._out_hz, self._reference.rate_hz))
        self._handed = begins + piece.size / self._out_hz
        self._pushed += piece.size
        self._counts["pieces"] += 1
        if self._activity is not None:  # under the lock: a stop cannot interleave
            self._activity.extend(self._handed)
        return True

    def _drop_pending(self) -> None:
        self._pending.clear()
        self._pending_n = 0

    def _next_due(self) -> float | None:
        """Seconds until the next pending piece may be handed; the lock is held."""
        if not self._pending or self._out_hz is None:
            return None
        piece = self._pending[0][0]
        return max(0.0, self._handed + piece.size / self._out_hz - self._lead_s - self._clock())

    def _pace(self) -> None:
        """The pacer thread: hand pieces as the player drains, until ``close``."""
        with self._cv:
            while not self._closing:
                try:
                    self._feed_locked()
                except Exception as exc:  # a pacer bug must not end playback silently
                    log.warning("speaker: pacer: %s", type(exc).__name__)
                    self._drop_pending()
                self._cv.wait(self._next_due())

    def pacer_alive(self) -> bool:
        """Whether the pacer thread runs (False with no lead, or after ``close``)."""
        pacer = self._pacer
        return pacer is not None and pacer.is_alive()

    def pacing(self) -> dict[str, Any]:
        """Pacing figures for the debug report: lead, pending audio, hand-offs, underruns (the
        player ran dry while a piece waited), late hand-offs (past the SDK gap reset) and the
        pending audio a stop dropped (last, max). Numbers only."""
        with self._lock:
            pending_ms = self._pending_s() * MS_PER_S
            at_stop = self._at_stop_ms
            return {
                "lead_ms": self._lead_s * MS_PER_S,
                "pending_ms": pending_ms,
                **self._counts,
                "underrun_ms": self._underrun_ms,
                "pending_at_stop_last_ms": at_stop[0] if at_stop else None,
                "pending_at_stop_max_ms": at_stop[1] if at_stop else None,
            }

    def _pending_s(self) -> float:
        return self._pending_n / self._out_hz if self._out_hz else 0.0

    def playing_until(self) -> float:
        """Monotonic stamp at which everything accepted so far has played (past if idle): the
        audio handed to the player, then the pending audio from when the player runs dry."""
        with self._lock:
            if not self._pending:
                return self._handed
            return max(self._handed, self._clock()) + self._pending_s()

    def queued_s(self) -> float:
        """Seconds of pushed audio still to play."""
        return max(0.0, self.playing_until() - self._clock())

    def stop(self) -> None:
        """Barge-in: drop the queued audio now (``clear_player``),
        abandon the pieces of a push in flight and reset the queued duration."""
        with self._lock:  # held throughout: no push lands between the flush and the reset
            self._generation += 1
            self._stream, self._stream_hz = None, 0  # the next push starts a fresh stream
            now = self._clock()
            if self._pending_n:
                dropped = self._pending_s() * MS_PER_S
                top = self._at_stop_ms[1] if self._at_stop_ms else dropped
                self._at_stop_ms = [dropped, max(top, dropped)]
            self._drop_pending()
            self._handed = now
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
            "pacing": self.pacing(),
        }
