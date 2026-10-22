"""Voice activity detection and endpointing (SPEC 8 stage 2).

``VADBackend`` scores fixed-size mono float32 frames; ``SileroVAD`` is the production backend
(Silero v5 on ONNX Runtime), ``EnergyVAD`` a crude RMS stand-in for tests and as a fallback.
``Endpointer`` turns the frame scores into utterance events with hysteresis:

- ``speech_start``: ``min_speech_ms`` of frames at or above ``threshold``. Its stamp is the
  first of those frames; the event carries the audio before and including them (``pre_roll_ms``
  before the first one), so the first syllable is not cut.
- while speaking a frame counts as speech down to ``threshold - off_margin``;
  ``end_silence_ms`` of frames below that end the utterance. ``max_utterance_s`` cuts a long one
  (speech that continues starts a new utterance).
- ``speech_end`` carries ``last_speech_stamp`` (start of the last frame that counted as speech;
  the latency budgets of SPEC 13 are measured from it, not from the endpoint) and
  ``endpoint_stamp`` (the end of the frame that completed the silence), plus the utterance audio.

Stamps are in the caller's clock (seconds, the stamp of a chunk's first sample); chunks are
taken as contiguous, the sample count advances the clock. Chunks may have any size. Audio is
held in RAM for the consumer only and is never logged: events hide it from ``repr``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ..config.schema import VadConfig
from ..stats import P50, P90, P99, summarise

MS_PER_S = 1000.0
SILERO_RATE_HZ = 16000
SILERO_FRAME_SAMPLES = 512
SILERO_CONTEXT_SAMPLES = 64
SILERO_STATE_SHAPE = (2, 1, 128)
# Floor of the off-threshold: with threshold 0.05 and off_margin 0.5 it would reach 0 and make
# every frame count as speech, so the end-silence timer could never run. Interface constant.
MIN_OFF_THRESHOLD = 0.01
# Reference level of EnergyVAD (tests and fallback only): an RMS of 0.1 (-20 dBFS) scores 1.
ENERGY_FULL_SCALE_RMS = 0.1
PEAK_LEVELS = {"p50": P50, "p90": P90, "p99": P99}
RUN_LEVELS = {"p50": P50, "p90": P90}
WATCH_COUNTS = ("frames", "frames_on", "starts", "ends_silence", "ends_max_length")


@runtime_checkable
class VADBackend(Protocol):
    """Scores one frame at a time: probability that it holds speech, 0 to 1."""

    rate_hz: int
    frame_samples: int

    def speech_prob(self, frame: np.ndarray) -> float:
        """Speech probability of exactly ``frame_samples`` mono float32 samples at ``rate_hz``."""
        ...

    def reset(self) -> None:
        """Forget any recurrent state (between utterances, after a gap in the audio)."""
        ...


class EnergyVAD:
    """RMS level mapped to a probability (``full_scale_rms`` or more is 1). Tests, fallback."""

    def __init__(
        self, rate_hz: int, frame_samples: int, full_scale_rms: float = ENERGY_FULL_SCALE_RMS
    ) -> None:
        self.rate_hz = rate_hz
        self.frame_samples = frame_samples
        self._full_scale = full_scale_rms

    def speech_prob(self, frame: np.ndarray) -> float:
        _check_frame(frame, self.frame_samples)
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        return min(1.0, rms / self._full_scale)

    def reset(self) -> None:
        return None


class SileroVAD:
    """Silero VAD v5 (``silero_vad.onnx``): 512-sample frames at 16 kHz.

    The model inputs are ``input`` (batch of 64 samples of context from the previous frame
    followed by the 512 new ones), ``state`` (2x1x128, carried from one call to the next) and
    ``sr``. ``reset`` zeroes both the state and the context. ``threads`` is ``voice.vad.threads``.
    Load the file through ``roboemot.voice.models.model_file`` so its marker is checked first.
    """

    rate_hz = SILERO_RATE_HZ
    frame_samples = SILERO_FRAME_SAMPLES

    def __init__(self, model_path: Path, threads: int) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("SileroVAD needs the `voice` extra (onnxruntime)") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SILERO_RATE_HZ, dtype=np.int64)
        self.reset()

    def speech_prob(self, frame: np.ndarray) -> float:
        _check_frame(frame, self.frame_samples)
        window = np.concatenate([self._context, frame])[np.newaxis, :]
        out, self._state = self._session.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = frame[-SILERO_CONTEXT_SAMPLES:]
        return float(out[0, 0])

    def reset(self) -> None:
        self._state = np.zeros(SILERO_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(SILERO_CONTEXT_SAMPLES, dtype=np.float32)


def _check_frame(frame: np.ndarray, size: int) -> None:
    if frame.dtype != np.float32 or frame.shape != (size,):
        raise ValueError(f"expected {size} mono float32 samples, got {frame.dtype} {frame.shape}")


@dataclass(frozen=True)
class SpeechStart:
    """An utterance began. ``pre_roll`` is the audio from ``pre_roll_stamp`` up to now."""

    stamp: float
    pre_roll_stamp: float
    pre_roll: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SpeechAudio:
    """Audio of the running utterance after the ``SpeechStart`` pre-roll, in order."""

    stamp: float
    samples: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SpeechEnd:
    """An utterance ended; ``audio`` runs from ``audio_stamp`` to ``endpoint_stamp``.

    ``last_speech_stamp`` is the start of the last frame judged speech at the off threshold
    (conservative: the true end of speech lies up to one frame later). A silence ending is
    declared ``end_silence`` frames after that frame, so ``endpoint_stamp - last_speech_stamp``
    is the end-silence (rounded up to whole frames) plus one frame. SPEC 13 latencies count
    from ``last_speech_stamp``. A ``max_length`` cut has no silence; the next utterance, if the
    speech continues, starts with the cut frame.
    """

    start_stamp: float
    last_speech_stamp: float
    endpoint_stamp: float
    audio_stamp: float
    audio: np.ndarray = field(repr=False)
    reason: Literal["silence", "max_length"]


VADEvent = SpeechStart | SpeechAudio | SpeechEnd


class Endpointer:
    """Frame probabilities to utterance events (see the module docstring)."""

    def __init__(self, backend: VADBackend, config: VadConfig) -> None:
        self._vad = backend
        self._rate = backend.rate_hz
        self._size = backend.frame_samples
        self._frame_s = self._size / self._rate
        frame_ms = self._frame_s * MS_PER_S
        self._on = config.threshold
        self._off = max(config.threshold - config.off_margin, MIN_OFF_THRESHOLD)
        self._min_frames = max(1, math.ceil(config.min_speech_ms / frame_ms))
        self._end_frames = max(1, math.ceil(config.end_silence_ms / frame_ms))
        self._max_s = config.max_utterance_s
        pre_frames = math.ceil(config.pre_roll_ms / frame_ms)
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=pre_frames + self._min_frames)
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_stamp = 0.0
        # Report-only watch (SPEC 12A): feed() runs on the listener thread, snapshot() on the
        # report thread; the lock covers the counters and the deque of finished bursts.
        self._watch = min(config.watch_prob, config.threshold)
        self._frame_ms = frame_ms
        self._watch_lock = threading.Lock()
        self._counts = dict.fromkeys(WATCH_COUNTS, 0)
        self._bursts: deque[tuple[float, float, bool]] = deque(maxlen=config.watch_window)
        self.reset()

    @property
    def frame_s(self) -> float:
        """Length of one VAD frame in seconds."""
        return self._frame_s

    @property
    def in_speech(self) -> bool:
        """True between ``SpeechStart`` and ``SpeechEnd``."""
        return self._speaking

    def reset(self) -> None:
        """Drop the utterance in progress, buffered audio, the backend state and any speech
        burst being watched (not recorded); the report counters and finished bursts stay."""
        self._vad.reset()
        self._ring.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._speaking = False
        self._run = 0
        self._silence = 0
        self._utterance: list[np.ndarray] = []
        self._utterance_stamp = 0.0
        self._start_stamp = 0.0
        self._last_speech = 0.0
        self._burst_peak: float | None = None
        self._burst_run = 0
        self._burst_longest = 0

    def feed(self, samples: np.ndarray, stamp: float) -> list[VADEvent]:
        """Add a chunk (mono float32, any length) starting at ``stamp``; returns new events."""
        if samples.dtype != np.float32 or samples.ndim != 1:
            raise ValueError(f"expected mono float32 samples, got {samples.dtype} {samples.shape}")
        if len(self._pending) == 0:
            self._pending_stamp = stamp
        buf = np.concatenate([self._pending, samples]) if len(self._pending) else samples.copy()
        events: list[VADEvent] = []
        pos = 0
        while len(buf) - pos >= self._size:
            frame = buf[pos : pos + self._size]
            events += self._frame(frame, self._pending_stamp + pos / self._rate)
            pos += self._size
        self._pending_stamp += pos / self._rate
        self._pending = buf[pos:].copy()
        return events

    def _frame(self, frame: np.ndarray, stamp: float) -> list[VADEvent]:
        """Advance the state machine by one frame starting at ``stamp``."""
        prob = self._vad.speech_prob(frame)
        frame_end = stamp + self._frame_s
        self._watch_frame(prob)
        if not self._speaking:
            self._ring.append((stamp, frame))
            self._run = self._run + 1 if prob >= self._on else 0
            if self._run < self._min_frames:
                return []
            return [self._begin(stamp)]
        self._utterance.append(frame)
        events: list[VADEvent] = [SpeechAudio(stamp, frame)]
        if prob >= self._off:
            self._silence = 0
            self._last_speech = stamp
        else:
            self._silence += 1
        if self._silence >= self._end_frames:
            events.append(self._finish("silence", frame_end))
        elif frame_end - self._start_stamp >= self._max_s:
            events.append(self._finish("max_length", frame_end))
            if prob >= self._off:
                self._ring.append((stamp, frame))
                self._run = 1
                events.append(self._begin(stamp))
        return events

    def _watch_frame(self, prob: float) -> None:
        """Count a scored frame and, while not speaking, grow or close the burst at or above the
        watch level (report only; one short lock hold per frame)."""
        on = bool(prob >= self._on)
        with self._watch_lock:
            self._counts["frames"] += 1
            self._counts["frames_on"] += on
            if self._speaking:
                return
            if prob < self._watch:
                self._close_burst(started=False)
                return
            self._burst_peak = prob if self._burst_peak is None else max(self._burst_peak, prob)
            self._burst_run = self._burst_run + 1 if on else 0
            self._burst_longest = max(self._burst_longest, self._burst_run)

    def _close_burst(self, *, started: bool) -> None:
        """Record the open burst (peak, longest run above threshold in ms, started), if any;
        the watch lock is held."""
        if self._burst_peak is not None:
            self._bursts.append(
                (float(self._burst_peak), self._burst_longest * self._frame_ms, started)
            )
        self._burst_peak, self._burst_run, self._burst_longest = None, 0, 0

    def snapshot(self) -> dict[str, Any]:
        """Counters and burst summaries for the debug report: numbers only, no audio."""
        with self._watch_lock:
            counts = dict(self._counts)
            bursts = list(self._bursts)

        def part(started: bool) -> dict[str, Any]:
            chosen = [(peak, run) for peak, run, s in bursts if s is started]
            return {
                "peak_prob": summarise([p for p, _ in chosen], PEAK_LEVELS),
                "longest_on_run_ms": summarise([r for _, r in chosen], RUN_LEVELS),
            }

        return {
            **counts,
            "watch_prob": self._watch,
            "window": self._bursts.maxlen,
            "started": part(True),
            "not_started": part(False),
        }

    def _begin(self, stamp: float) -> SpeechStart:
        """Start an utterance: the ring (pre-roll and the confirming frames) is its first audio."""
        with self._watch_lock:
            self._counts["starts"] += 1
            self._close_burst(started=True)
        items = list(self._ring)
        self._ring.clear()
        self._speaking = True
        self._silence = 0
        self._start_stamp = items[len(items) - self._run][0]
        self._last_speech = stamp
        self._run = 0
        self._utterance = [f for _, f in items]
        self._utterance_stamp = items[0][0]
        return SpeechStart(
            self._start_stamp, self._utterance_stamp, np.concatenate(self._utterance)
        )

    def _finish(self, reason: Literal["silence", "max_length"], endpoint: float) -> SpeechEnd:
        end = SpeechEnd(
            start_stamp=self._start_stamp,
            last_speech_stamp=self._last_speech,
            endpoint_stamp=endpoint,
            audio_stamp=self._utterance_stamp,
            audio=np.concatenate(self._utterance),
            reason=reason,
        )
        self._speaking = False
        self._utterance = []
        self._silence = 0
        with self._watch_lock:
            self._counts["ends_silence" if reason == "silence" else "ends_max_length"] += 1
        if reason == "silence":
            self._vad.reset()
        return end
