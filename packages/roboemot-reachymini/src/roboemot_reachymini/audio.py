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
  stamp is the read time. After a stall of the reading thread (a gap longer than ``max_gap_s``
  from the end of one pass to the start of the next) the queue holds old audio: it is drained
  and only the newest chunk counts for the reflex (the others are ``dropped_stale``), so a
  startle is never played from sound that is already old. The drain reads only the backlog of
  that stall: it stops at an empty queue (``None``), at a read that had to wait for the
  microphone (SDK 1.12 blocks for the next chunk instead of returning ``None``), once it has
  read as much audio as the stall lasted, or at ``DRAIN_MAX`` chunks. Without a known input rate
  only the first and the last apply.
* the voice pipeline (``voice=True``) gets every chunk, lossless unless the bound overflows: mono
  float32 at ``voice_rate_hz``, from the input rate and channel count the SDK reports (not
  assumed; channels are averaged). ``read`` pops them from a queue bounded in seconds of audio.
  When the consumer lags the oldest chunks are dropped, counted (chunks and samples) in
  ``snapshot()``, and the next chunk carries ``gap=True``. A chunk with non-finite samples is
  replaced by silence of the same length, also marked ``gap``.

The levels of the two microphone channels (RMS in dBFS, mono mix, correlation) are kept per block
of ``level_block_s`` input seconds for the debug report (``snapshot()["levels"]``), to tell two
live microphones from one silent or anti-phase channel (SPEC 12A). Every chunk read counts,
drained ones included, with or without the voice pipeline (the input rate is read at start
either way); only running sums leave a chunk, never samples.

Every chunk goes to the voice queue as soon as it is read, drained ones included, so the
consumer never waits for a drain to end. Voice stamps are approximate and monotone: a chunk read
in a normal pass ends at the read time; a queued chunk read in a drain ends one chunk duration
after the previous chunk (its capture time when the SDK lost nothing), never after its read
time; the resampler's lookahead is subtracted.

Privacy (SPEC 12): without ``voice`` a chunk is only held while its RMS and the level sums are
taken, and only those scalars leave this module. With ``voice`` the queue holds audio for the voice
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
from roboemot.config.schema import RoboEmotConfig
from roboemot.stats import P10, P50, P90, summarise

from .resample import StreamResampler
from .sdk import MiniPort

log = logging.getLogger(__name__)
JOIN_TIMEOUT_S = 2.0
MS_PER_S = 1000.0
IDLE_WAIT_S = 0.01
DRAIN_MAX = 1000  # most chunks thrown away in one catch-up
# A read that takes this share of the chunk's own duration waited for the microphone: the SDK
# queue was empty. A queued chunk is handed over without blocking; a waiting read blocks until
# the capture of the next chunk (SDK 1.12), so this classifies the SDK's read, not a behaviour.
WAITED_READ_FRACTION = 0.5
LEVEL_FLOOR_DB = -120.0  # silence, and the lowest level reported
DB_PER_DECADE = 20.0  # amplitude ratio to dB
LEVEL_QUANTILES = {"p10": P10, "p50": P50, "p90": P90}


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
        mini: MiniPort,
        sink: Callable[[float, float], None] | None,
        poll_hz: float,
        clock: Callable[[], float] = time.monotonic,
        max_gap_s: float = math.inf,
        *,
        voice: bool = False,
        voice_rate_hz: int = 16000,
        voice_queue_s: float = 10.0,
        level_block_s: float = 1.0,
        level_window: int = 300,
    ) -> None:
        self._mini, self._sink, self._clock = mini, sink, clock
        self._period = 1.0 / poll_hz
        self.max_gap_s = max_gap_s
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
        self._level_block_s = level_block_s
        self._level_lock = threading.Lock()
        self._level_blocks: deque[tuple[float, float, float, float]] = deque(maxlen=level_window)
        self._level_reset()
        self._in_rate: int | None = None
        self._in_channels: int | None = None
        self._resampler: StreamResampler | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state, self._reason = "off", None
        self._chunks = 0
        self._last_error: str | None = None
        self._last_end: float | None = None  # end stamp of the last chunk read, before lookahead

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (SPEC 11A): the stall limit follows ``reflexes.max_age_ms``. The read rate,
        the voice queue and the level blocks are restart fields."""
        self.max_gap_s = cfg.reflexes.max_age_ms / MS_PER_S

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
        snap["levels"] = self._levels_report()
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

    def _level_reset(self) -> None:
        """Zero the running sums of the block in progress (the finished blocks stay)."""
        self._level_sums = [0.0] * 4  # x0^2, x1^2, mono^2, x0*x1
        self._level_n = 0

    def _accumulate_levels(self, chunk: Any) -> None:
        """Add a finite ``(n, 2+)`` chunk to the running sums; close the block once it holds
        ``level_block_s`` seconds of input (the chunk that crosses the limit completes it).
        Anything else (mono, empty, non-finite, no input rate known) is not added, so a block
        can span more wall time than that."""
        if (
            self._in_rate is None
            or not isinstance(chunk, np.ndarray)
            or chunk.ndim != 2
            or chunk.shape[1] < 2
            or chunk.shape[0] == 0
        ):
            return
        x = chunk.astype(np.float64, copy=False)
        if not np.isfinite(x).all():
            return
        x0, x1, mono = x[:, 0], x[:, 1], x.mean(axis=1)
        add = (float(x0 @ x0), float(x1 @ x1), float(mono @ mono), float(x0 @ x1))
        with self._level_lock:
            self._level_sums = [a + b for a, b in zip(self._level_sums, add, strict=True)]
            self._level_n += x.shape[0]
            if self._level_n >= self._level_block_s * self._in_rate:
                self._level_blocks.append(self._close_block())
                self._level_reset()

    def _close_block(self) -> tuple[float, float, float, float]:
        """(ch0 dBFS, ch1 dBFS, mono dBFS, correlation) of the running sums; the lock is held."""
        s0, s1, sm, cross = self._level_sums

        def dbfs(power_sum: float) -> float:
            rms = math.sqrt(power_sum / self._level_n)
            return (
                max(LEVEL_FLOOR_DB, DB_PER_DECADE * math.log10(rms)) if rms > 0 else LEVEL_FLOOR_DB
            )

        norm = math.sqrt(s0 * s1)
        corr = max(-1.0, min(1.0, cross / norm)) if norm > 0 else 0.0
        return dbfs(s0), dbfs(s1), dbfs(sm), corr

    def _levels_report(self) -> dict[str, Any] | None:
        """Block count and summaries of the channel levels, or None before the first block (and
        for a mono input, which never makes one)."""
        with self._level_lock:
            blocks = list(self._level_blocks)
        if not blocks:
            return None

        def summary(i: int, *, with_min: bool = False) -> dict[str, float]:
            out = summarise([b[i] for b in blocks], LEVEL_QUANTILES, with_min=with_min)
            del out["count"]
            return out

        return {
            "block_s": self._level_block_s,
            "blocks": len(blocks),
            "ch0_dbfs": summary(0),
            "ch1_dbfs": summary(1),
            "mono_dbfs": summary(2),
            "corr": summary(3, with_min=True),
        }

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
        media = self._mini.media
        if media.audio is None:
            self._set("unavailable", "the SDK reports no audio system")
            return
        try:
            media.start_recording()
        except Exception as exc:
            self._set("unavailable", f"start_recording failed: {type(exc).__name__}")
            return
        self._level_reset()
        self._last_end = None
        if self._voice:
            with self._voice_cond:
                self._voice_q.clear()
                self._voice_queued, self._voice_gap, self._voice_closed = 0, False, False
            self._open_voice(media)
        else:
            self._read_format(media)
        self._set("no_data", "no audio chunk read yet")
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="roboemot-audio", daemon=True)
        thread.start()
        self._thread = thread

    def _read_format(self, media: Any) -> None:
        """Read the input rate and channel count from the SDK (never assumed); an unusable rate
        is None (no voice frames, no level blocks)."""
        try:
            rate = int(media.get_input_audio_samplerate())
            self._in_channels = int(media.get_input_channels())
        except Exception as exc:
            # Without voice the format only feeds the level report: not worth a warning.
            level = logging.WARNING if self._voice else logging.DEBUG
            log.log(level, "audio: input format: %s", type(exc).__name__)
            self._in_rate = None
            return
        self._in_rate = rate if rate > 0 else None

    def _open_voice(self, media: Any) -> None:
        """Read the input format (``_read_format``) and set up the conversion to
        ``voice_rate_hz``; an unusable rate leaves the voice side silent."""
        self._resampler = None
        self._read_format(media)
        if self._in_rate is None:
            log.warning("audio: the SDK reports no input rate; voice frames are off")
        elif self._in_rate != self._voice_rate:
            self._resampler = StreamResampler(self._in_rate, self._voice_rate)

    def _duration(self, chunk: Any) -> float:
        """Seconds of input in ``chunk``; 0 when the input rate is unknown or it is no array."""
        if self._in_rate is None or not isinstance(chunk, np.ndarray) or chunk.ndim == 0:
            return 0.0
        return chunk.shape[0] / self._in_rate

    def _take(self, chunk: Any, read_at: float, queued: bool) -> bool:
        """Count one chunk read at ``read_at`` for the levels and queue it for the voice side at
        once. A ``queued`` chunk (read from a drain's backlog) ends a chunk after the previous
        one, never after ``read_at``; any other ends at ``read_at``. Returns whether the voice
        side used it."""
        self._accumulate_levels(chunk)
        end = read_at
        if queued and self._last_end is not None:
            end = min(read_at, self._last_end + self._duration(chunk))
        if chunk is not None:
            self._last_end = end
        return self._voice and self._push_voice(chunk, end)

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

    def _read(self, media: Any) -> tuple[Any, float, bool]:
        """One SDK read: the chunk, its read time, and whether the read waited for the
        microphone (it took ``WAITED_READ_FRACTION`` of the chunk's duration or more)."""
        before = self._clock()
        chunk = media.get_audio_sample()
        read_at = self._clock()
        limit = WAITED_READ_FRACTION * self._duration(chunk)
        return chunk, read_at, limit > 0 and read_at - before >= limit

    def poll_once(self, drain: bool = False, stall_s: float = math.inf) -> bool:
        """Read one chunk; hand it to the voice queue and its RMS to the reflex sink. Returns
        whether a chunk was used. With ``drain`` (and a sink) the backlog of a stall of
        ``stall_s`` is read (module docstring): every chunk goes to the voice queue as it is
        read, the reflex gets only the newest one and the rest is counted as ``dropped_stale``."""
        try:
            media = self._mini.media
            drain = drain and self._sink is not None
            chunk, read_at, waited = self._read(media)
            used = self._take(chunk, read_at, drain and not waited)
            covered, skipped = self._duration(chunk), 0
            while drain and chunk is not None and not waited and covered < stall_s:
                if skipped >= DRAIN_MAX:
                    break
                newer, read_at, waited = self._read(media)
                if newer is None:
                    break
                chunk, skipped = newer, skipped + 1
                used = self._take(newer, read_at, not waited) or used
                covered += self._duration(newer)
            if skipped:
                with self._lock:
                    self._dropped += skipped
            if self._sink is not None:
                rms = chunk_rms(chunk)
                del chunk
                if rms is None:
                    return False
                self._sink(rms, self._clock())
                used = True
            else:
                del chunk
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
        ended = self._clock()
        while not self._stop.is_set():
            # Read what is queued, then wait out the rest of the period; after a stall, catch up.
            # The gap runs from the end of the last pass, so a long pass (a drain) is no stall.
            gap = self._clock() - ended
            used = self.poll_once(drain=gap > self.max_gap_s, stall_s=gap)
            ended = self._clock()
            self._stop.wait(IDLE_WAIT_S if used else self._period)
