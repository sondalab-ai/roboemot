"""Microphone hub for the Reachy Mini: the one reader of the SDK's audio (SPEC 4A, 8, 10, 12).

The app owns the SDK client and its media (one process, SPEC 4A); this reads from the
``ReachyMini`` the daemon passed to ``run`` and never creates a client. Coded against
reachy-mini 1.12's documented interface, not exercised against a real robot or the simulator:
``mini.media.start_recording()`` starts the capture, ``mini.media.get_audio_sample()`` returns the
next chunk as a float32 ``ndarray`` (``(n,)`` or ``(n, channels)``, full scale 1.0) or ``None``
when nothing is queued, and ``mini.media.audio`` is ``None`` when the robot has no audio system
(the SDK then logs a warning on every call, so it is checked once).

``get_audio_sample`` hands each chunk to one reader, so every consumer takes its audio from here
and nothing else may read the SDK. Two consumers:

* the reflex sink gets each chunk's RMS. Chunks carry no capture timestamp in the SDK, so the
  stamp is the read time. After a stall of the reading thread (a gap longer than ``max_gap_s``)
  the queue holds old audio: it is drained and only the newest chunk counts for the reflex, so a
  startle is never played from sound that is already old.
* the voice pipeline (``voice=True``) gets every chunk, lossless unless the bound overflows: mono
  float32 at ``voice_rate_hz``, from the input rate and channel count the SDK reports (not
  assumed; channels are averaged). ``read`` pops them from a queue bounded in seconds of audio.
  When the consumer lags the oldest chunks are dropped, counted (chunks and samples) in
  ``snapshot()``, and the next chunk carries ``gap=True``. A chunk with non-finite samples is
  replaced by silence of the same length, also marked ``gap``.

Voice stamps are approximate, monotone and not capture times: the newest chunk of a read ends
at the read time, earlier chunks of one drained batch end one chunk duration before the next,
and the resampler's lookahead is subtracted.

Privacy (SPEC 12): without ``voice`` a chunk is only held while its RMS is taken, and only
that scalar leaves this module. With ``voice`` the queue holds audio for the voice
pipeline by design and nothing else; frames are never published on the bus, written to disk or
logged, and ``snapshot()`` carries counts only.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from roboemot import shutdown

from .resample import StreamResampler

log = logging.getLogger(__name__)
JOIN_TIMEOUT_S = 2.0
IDLE_WAIT_S = 0.01
DRAIN_MAX = 1000  # most chunks thrown away in one catch-up


def chunk_rms(chunk: Any) -> float | None:
    """RMS of a float audio chunk (all channels), or None for anything but a non-empty ndarray."""
    if not isinstance(chunk, np.ndarray) or chunk.size == 0:
        return None
    x = chunk.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(x))))
    return rms if math.isfinite(rms) else None


class VoiceChunk(NamedTuple):
    """Mono float32 audio for the voice pipeline: ``rate_hz`` travels with the samples and
    ``stamp`` is the approximate monotonic time the chunk ends (not a capture time). ``gap`` is
    set on the first chunk after audio was lost (queue overflow, non-finite input)."""

    samples: np.ndarray
    rate_hz: int
    stamp: float
    gap: bool = False


def to_mono(chunk: np.ndarray) -> np.ndarray:
    """Average the channels of an ``(n,)`` or ``(n, channels)`` chunk into mono float32."""
    mono = chunk if chunk.ndim == 1 else chunk.reshape(chunk.shape[0], -1).mean(axis=1)
    return mono.astype(np.float32, copy=False)


class ReachyAudio:
    """Polls the SDK's audio chunks in a thread; the hub for every consumer of the microphone.

    ``sink(rms, stamp)`` (optional) gets each chunk's RMS for the reflexes. With ``voice`` every
    chunk also goes, downmixed and resampled to ``voice_rate_hz``, to a queue of at most
    ``voice_queue_s`` seconds that ``read`` drains. ``state`` is one of ``off`` (not started),
    ``unavailable`` (no audio system or start failed; ``reason`` says which), ``no_data``
    (running, nothing usable yet) and ``running``. The stamp is the monotonic time of the read.
    """

    def __init__(
        self,
        mini: Any,
        sink: Callable[[float, float], None] | None,
        poll_hz: float,
        clock: Callable[[], float] = time.monotonic,
        max_gap_s: float = math.inf,
        *,
        voice: bool = False,
        voice_rate_hz: int = 16000,
        voice_queue_s: float = 10.0,
    ) -> None:
        self._mini, self._sink, self._clock = mini, sink, clock
        self._period = 1.0 / poll_hz
        self._max_gap_s = max_gap_s
        self._dropped = 0
        self._voice = voice
        self._voice_rate = voice_rate_hz
        self._voice_cap = int(voice_queue_s * voice_rate_hz)
        self._voice_cond = threading.Condition()
        self._voice_q: deque[VoiceChunk] = deque()
        self._voice_queued = 0  # samples in the queue
        self._voice_dropped = 0
        self._voice_dropped_samples = 0
        self._voice_sanitised = 0
        self._voice_gap = False  # the next queued chunk follows a loss
        self._voice_closed = False
        self._voice_frames = 0
        self._in_rate: int | None = None
        self._in_channels: int | None = None
        self._resampler: StreamResampler | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state, self._reason = "off", None
        self._chunks = 0
        self._last_error: str | None = None

    def set_voice(self, enabled: bool) -> None:
        """Turn the voice queue on or off before ``start`` (the app learns whether a listener
        exists only after the hub is built); a hub without a consumer queues nothing."""
        self._voice = enabled

    def snapshot(self) -> dict[str, Any]:
        """State for the debug report: scalars and a reason, never audio."""
        with self._lock:
            snap: dict[str, Any] = {
                "state": self._state,
                "reason": self._reason,
                "chunks": self._chunks,
                "dropped_stale": self._dropped,
                "voice": None,
            }
        if self._voice:
            with self._voice_cond:
                snap["voice"] = {
                    "rate_hz": self._voice_rate,
                    "input_rate_hz": self._in_rate,
                    "input_channels": self._in_channels,
                    "queued_s": self._voice_queued / self._voice_rate,
                    "frames": self._voice_frames,
                    "dropped": self._voice_dropped,
                    "dropped_samples": self._voice_dropped_samples,
                    "sanitised": self._voice_sanitised,
                }
        return snap

    def read(self, timeout: float = 0.0) -> VoiceChunk | None:
        """Pop the oldest voice chunk, waiting up to ``timeout`` s; ``None`` if there is none
        (or ``voice`` is off). The consumer API of the M3 voice pipeline."""
        with self._voice_cond:
            if not self._voice_q and timeout > 0 and not self._voice_closed:
                self._voice_cond.wait_for(
                    lambda: bool(self._voice_q) or self._voice_closed, timeout
                )
            if not self._voice_q:
                return None
            chunk = self._voice_q.popleft()
            self._voice_queued -= chunk.samples.size
            return chunk

    def _voice_ready(self) -> bool:
        return bool(self._voice_q) or self._voice_closed

    def _set(self, state: str, reason: str | None = None) -> None:
        with self._lock:
            self._state, self._reason = state, reason

    def start(self) -> None:
        """Start capturing in a daemon thread; a second start while running is a no-op. Degrades
        to ``unavailable`` (and starts nothing) when the robot has no audio system."""
        if self._thread is not None and self._thread.is_alive():
            return
        media = getattr(self._mini, "media", None)
        if media is None or getattr(media, "audio", True) is None:
            self._set("unavailable", "the SDK reports no audio system")
            return
        try:
            media.start_recording()
        except Exception as exc:
            self._set("unavailable", f"start_recording failed: {type(exc).__name__}")
            return
        if self._voice:
            with self._voice_cond:
                self._voice_q.clear()
                self._voice_queued, self._voice_gap, self._voice_closed = 0, False, False
            self._open_voice(media)
        self._set("no_data", "no audio chunk read yet")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="roboemot-audio", daemon=True)
        self._thread.start()

    def _open_voice(self, media: Any) -> None:
        """Read the input rate and channel count from the SDK (never assumed) and set up the
        conversion to ``voice_rate_hz``; an unusable rate leaves the voice side silent."""
        self._resampler = None
        try:
            rate = int(media.get_input_audio_samplerate())
            self._in_channels = int(media.get_input_channels())
        except Exception as exc:
            log.warning("audio: input format: %s", type(exc).__name__)
            self._in_rate = None
            return
        self._in_rate = rate if rate > 0 else None
        if self._in_rate is None:
            log.warning("audio: the SDK reports no input rate; voice frames are off")
        elif self._in_rate != self._voice_rate:
            self._resampler = StreamResampler(self._in_rate, self._voice_rate)

    def _chunk_ends(self, batch: list[Any], read_at: float) -> list[float]:
        """End stamp of each chunk of one read: the newest ends at ``read_at``, each earlier one
        a chunk duration before the next."""
        rate = self._in_rate or 1
        ends, t = [], read_at
        for raw in reversed(batch):
            ends.append(t)
            t -= (raw.shape[0] if isinstance(raw, np.ndarray) and raw.ndim else 0) / rate
        return ends[::-1]

    def _push_voice(self, chunk: Any, end_stamp: float) -> bool:
        """Queue one raw chunk (ending at ``end_stamp``) for the voice consumer; the oldest
        chunks go when over the bound, and the first one left is marked as following a gap."""
        if self._in_rate is None or not isinstance(chunk, np.ndarray) or chunk.size == 0:
            return False
        mono = to_mono(chunk)
        bad = not np.isfinite(mono).all()
        if bad:
            mono = np.zeros_like(mono)
        if self._resampler is not None:
            mono = self._resampler.process(mono)
            stamp = end_stamp - self._resampler.lookahead_s
        else:
            stamp = end_stamp
        with self._voice_cond:
            self._voice_gap = self._voice_gap or bad
            self._voice_sanitised += bad
            if mono.size == 0:
                return True
            self._voice_q.append(VoiceChunk(mono, self._voice_rate, stamp, self._voice_gap))
            self._voice_gap = False
            self._voice_queued += mono.size
            self._voice_frames += mono.size
            lost = False
            while self._voice_queued > self._voice_cap and len(self._voice_q) > 1:
                gone = self._voice_q.popleft().samples.size
                self._voice_queued -= gone
                self._voice_dropped += 1
                self._voice_dropped_samples += gone
                lost = True
            if lost:
                self._voice_q[0] = self._voice_q[0]._replace(gap=True)
            self._voice_cond.notify()
        return True

    def stop(self) -> None:
        """Stop the thread, wait briefly, then stop the capture. A thread stuck in the SDK keeps
        its handle, so a later ``start`` does not run a second one beside it."""
        self._stop.set()
        with self._voice_cond:
            self._voice_closed = True
            self._voice_cond.notify_all()
        thread = self._thread
        if thread is not None:
            ended = shutdown.join(thread, JOIN_TIMEOUT_S)
            if ended:
                self._thread = None
            try:  # best effort even when the thread did not end: the capture must not stay open
                self._mini.media.stop_recording()
            except Exception as exc:
                log.warning("audio: stop_recording: %s", type(exc).__name__)
            if not ended:
                return
        self._set("off")

    def poll_once(self, drain: bool = False) -> bool:
        """Read one chunk; hand it to the voice queue and its RMS to the reflex sink. Returns
        whether a chunk was used. With ``drain`` (and a sink) everything queued is read: the
        voice side keeps all of it, the reflex only the newest chunk (the rest is counted)."""
        try:
            media = self._mini.media
            chunk = media.get_audio_sample()
            batch = [chunk] if self._voice and chunk is not None else []
            skipped = 0
            drain = drain and self._sink is not None
            while drain and chunk is not None and skipped < DRAIN_MAX:
                newer = media.get_audio_sample()
                if newer is None:
                    break
                chunk, skipped = newer, skipped + 1
                if self._voice:
                    batch.append(newer)
            if skipped:
                with self._lock:
                    self._dropped += skipped
            used = False
            ends = self._chunk_ends(batch, self._clock())
            for raw, end in zip(batch, ends, strict=True):
                used = self._push_voice(raw, end) or used
            if self._sink is not None:
                rms = chunk_rms(chunk)
                del chunk, batch
                if rms is None:
                    return False
                self._sink(rms, self._clock())
                used = True
            else:
                del chunk, batch
            if not used:
                return False
        except Exception as exc:
            msg = f"{type(exc).__name__}"
            if msg != self._last_error:  # once per kind, not per poll
                log.warning("audio: %s", msg)
                self._last_error = msg
            return False
        self._last_error = None
        with self._lock:
            self._chunks += 1
            self._state, self._reason = "running", None
        return True

    def _run(self) -> None:
        last = self._clock()
        while not self._stop.is_set():
            now = self._clock()
            # Read what is queued, then wait out the rest of the period; after a stall, catch up.
            used = self.poll_once(drain=now - last > self._max_gap_s)
            last = now
            self._stop.wait(IDLE_WAIT_S if used else self._period)
