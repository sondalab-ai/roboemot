"""Barge-in: the user speaks over the robot's voice (SPEC 8 stage 8; 7A stop keyword; 13).

The listener drops every microphone chunk while the robot speaks and hands it to
:meth:`BargeIn.on_chunk`. A frame counts as the user's speech when a *second* Silero VAD
(its recurrent state is not shared with the listener's) scores it at or above
``voice.barge_in.threshold`` **and** it is not the robot's own voice coming back through the
microphone. That second test is the residual-echo guard for robots without hardware AEC (the
Wireless has XMOS AEC): the speaker keeps a short reference of what it played with the stamp each
piece starts to play (:class:`PlaybackReference`), and the last ``echo_window_ms`` of microphone
audio is compared with it, lag searched up to ``echo_max_lag_ms``, by normalised
cross-correlation (FFT). Above ``echo_corr_max`` it is echo. User speech mixed with echo
correlates less the louder it is, so a user quieter than the echo is not heard (documented limit).

A second, faster path covers words that start with a fricative (English "stop": a ~130 ms /s/ that
Silero scores at 0.35 to 0.9 until the vowel). ``onset_min_speech_ms`` of frames that are
fricative-like (``onset_hf_min_ratio`` of the energy above ``onset_hf_cutoff_hz``), scored at or
above ``onset_threshold``, louder than ``onset_min_level_db`` and correlated under
``onset_echo_corr_max`` with the played audio, both in the full band and in the band above the
cutoff (where a user's /s/ is not hidden by the robot's low-frequency voice), cut the robot off
too; frames the main path accepts count as well, under the same stricter guard. The onset is
the first of those frames, so the 200 ms of SPEC 13 count from the first sound.
``onset_echo_frames`` counts the frames this guard dropped.

After ``min_speech_ms`` of such frames the robot is cut off: ``stop`` (the speech output: sink
flushed, the speaking gate reset), ``interrupt`` (the conversation cancels the LLM stream and
trims its history to what was heard) and ``resume`` (the listener restarts on the audio from
shortly before the onset, so the first words are kept). Before the stop, ``prepare`` tells the
conversation to say nothing more. The onset-to-stop latency is measured (SPEC 13: 200 ms) from
the first speech frame to the moment ``stop()`` returned, on the audio clock; it does not include
the microphone's capture latency or the SDK's ``clear_player`` latency, which only the robot can
tell. It shows in :meth:`BargeIn.snapshot` with ``keyword_seconds``, the audio fed to the keyword
recognizer (a CPU proxy).

The stop keyword path (SPEC 7A: always active): frames the VAD scores at the looser
``keyword_threshold`` and that pass the looser ``keyword_echo_corr_max`` are fed to the listener's
idle recognizer, and a ``stop`` phrase in its partials stops the robot (and emits
``speech.keyword.stop``) even when the barge-in threshold is never met. The recognizer is the
listener's, because only one runs at a time (cm4 budget), but it is *run* by a
:class:`~roboemot.voice.keyword_worker.KeywordWorker` thread (``keyword_background``): building
it and digesting the ring take 100 to 300 ms, which on the listener thread delayed the speech
path's cut-off. The listener thread only submits frames and polls for a hit, and still performs
every cut-off itself, so the barge-in state has a single thread. It hands the recognizer back
(``reset``) before the listener restarts on the barge-in audio.

Audio and text are never logged or published.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..config.schema import RoboEmotConfig
from .asr import ASRBackend
from .fastpath import Emit, LatencyStats
from .keyword_worker import KeywordWorker
from .keywords import KeywordSpotter
from .vad import VADBackend

log = logging.getLogger(__name__)
MS_PER_S = 1000.0
# Mean-square level below which audio counts as silence for the echo check (about -80 dBFS).
SILENT_POWER = 1e-8
# Decibels per decade of power, and the floor that keeps log10 finite on a silent frame.
DB_PER_DECADE = 10.0
LOG_FLOOR = 1e-12
# Float slack when comparing frame stamps with the pre-roll boundary.
EDGE_S = 1e-9


class VoiceChunkLike(Protocol):
    samples: np.ndarray
    rate_hz: int
    stamp: float
    gap: bool


class PlaybackReference:
    """The last seconds of played audio at ``rate_hz`` with the monotonic stamp each piece
    starts to play. Written by the speaker, read by the echo check; thread-safe."""

    def __init__(self, rate_hz: int, keep_s: float) -> None:
        self.rate_hz = rate_hz
        self._keep_s = keep_s
        self._lock = threading.Lock()
        self._pieces: collections.deque[tuple[float, np.ndarray]] = collections.deque()

    def set_keep_s(self, keep_s: float) -> None:
        with self._lock:
            self._keep_s = keep_s

    def add(self, start: float, samples: np.ndarray) -> None:
        """A piece of mono float32 ``samples`` (at ``rate_hz``) that starts playing at ``start``."""
        if samples.size == 0:
            return
        with self._lock:
            self._pieces.append((start, samples.astype(np.float32, copy=True)))
            horizon = start + samples.size / self.rate_hz - self._keep_s
            while self._pieces and self._end(self._pieces[0]) < horizon:
                self._pieces.popleft()

    def _end(self, piece: tuple[float, np.ndarray]) -> float:
        return piece[0] + piece[1].size / self.rate_hz

    def truncate(self, at: float) -> None:
        """The speaker was flushed at ``at``: what was queued to play after it never will."""
        with self._lock:
            kept: collections.deque[tuple[float, np.ndarray]] = collections.deque()
            for start, samples in self._pieces:
                if start >= at:
                    continue
                keep = math.floor((at - start) * self.rate_hz)
                kept.append((start, samples[: max(0, keep)]))
            self._pieces = kept

    def window(self, t0: float, t1: float) -> np.ndarray:
        """The audio that played during ``[t0, t1)`` (zeros where nothing did)."""
        n = max(0, round((t1 - t0) * self.rate_hz))
        out = np.zeros(n, np.float32)
        with self._lock:
            for start, samples in self._pieces:
                lo = round((start - t0) * self.rate_hz)
                a, b = max(lo, 0), min(lo + samples.size, n)
                if b > a:
                    out[a:b] = samples[a - lo : b - lo]
        return out


def echo_correlation(mic: np.ndarray, ref: np.ndarray) -> float:
    """Highest absolute normalised cross-correlation of ``mic`` with any ``mic.size`` long
    stretch of ``ref`` (which is the longer: the extra length is the lag searched). 0 when
    either is silent. Computed with FFTs, so a few hundred microseconds for the defaults."""
    n, m = mic.size, ref.size
    if n == 0 or m < n:
        return 0.0
    mic = mic.astype(np.float64) - float(np.mean(mic))
    ref = ref.astype(np.float64) - float(np.mean(ref))
    e_mic = float(np.dot(mic, mic))
    if e_mic / n < SILENT_POWER:
        return 0.0
    size = 1 << (m + n - 1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(mic, size)), size)[
        : m - n + 1
    ]
    cum = np.concatenate([[0.0], np.cumsum(ref * ref)])
    e_ref = cum[n:] - cum[: m - n + 1]
    ok = e_ref / n >= SILENT_POWER
    if not ok.any():
        return 0.0
    ncc = np.abs(cross[ok]) / np.sqrt(e_mic * e_ref[ok])
    return float(min(1.0, ncc.max()))


class _Session:
    """A keyword recognizer run: started at the first candidate frame."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.last_candidate = started


class BargeIn:
    """See the module docstring. Built before the speech output and the conversation exist;
    ``bind`` hands it the three actions (and it ignores chunks until then)."""

    def __init__(
        self,
        cfg: RoboEmotConfig,
        vad: VADBackend,
        asr: ASRBackend,
        reference: PlaybackReference | None,
        *,
        language: Callable[[], str],
        emit: Emit | None = None,
        clock: Callable[[], float] = time.monotonic,
        keyword_background: bool = False,
    ) -> None:
        self._vad, self._asr, self._reference = vad, asr, reference
        self._language, self._emit, self._clock = language, emit, clock
        self._rate = vad.rate_hz
        self._size = vad.frame_samples
        self._frame_s = self._size / self._rate
        self._stop: Callable[[], None] | None = None
        self._interrupt: Callable[[], Any] = lambda: None
        self._resume: Callable[[np.ndarray, float], None] | None = None
        self._prepare: Callable[[], Any] = lambda: None
        self.stats = LatencyStats()
        self._counts = {
            "triggers": 0,
            "keyword_triggers": 0,
            "echo_frames": 0,
            "onset_echo_frames": 0,
            "errors": 0,
        }
        self._keyword_s = 0.0  # audio fed to the keyword recognizer, a CPU proxy for the cm4
        self._last_corr: float | None = None
        self._lang = ""
        self._deaf_until = -math.inf
        self._session: _Session | None = None
        self._worker = KeywordWorker(
            asr,
            lambda text, lang: "stop" in self._spotter.spot(text, lang),
            self._frame_s,
            background=keyword_background,
        )
        self.apply_config(cfg)
        self.reset()

    def close(self) -> None:
        """Stop the keyword worker thread and hand the recognizer back."""
        self._worker.close()

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: thresholds, windows and the keyword lists."""
        b = cfg.voice.barge_in
        self._b = b
        self._tail_s = cfg.voice.echo_tail_ms / MS_PER_S
        # Only the stop keyword is used here: no name, so no name phrases to build or check.
        self._spotter = KeywordSpotter(cfg.voice.keywords, None)
        frame_ms = self._frame_s * MS_PER_S
        self._min_frames = max(1, math.ceil(b.min_speech_ms / frame_ms))
        self._onset_frames = max(1, math.ceil(b.onset_min_speech_ms / frame_ms))
        self._echo_frames = max(1, math.ceil(b.echo_window_ms / frame_ms))
        self._pre_frames = math.ceil(b.pre_roll_ms / frame_ms)
        ring = (
            max(self._pre_frames + max(self._min_frames, self._onset_frames), self._echo_frames) + 1
        )
        self._ring_len = ring  # applied on the listener thread, which owns the ring
        if not hasattr(self, "_ring"):
            self._ring: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=ring)
        if self._reference is not None:
            self._reference.set_keep_s(b.echo_reference_s)

    def bind(
        self,
        stop: Callable[[], None],
        interrupt: Callable[[], Any],
        resume: Callable[[np.ndarray, float], None] | None = None,
        prepare: Callable[[], Any] | None = None,
    ) -> None:
        """Connect the actions: ``stop`` the speech output, ``interrupt`` the conversation,
        ``resume`` the listener on the barge-in audio. ``prepare`` runs before ``stop``: it
        tells the conversation to say nothing more, so a sentence queued between the stop and
        ``interrupt`` cannot play."""
        self._stop, self._interrupt, self._resume = stop, interrupt, resume
        self._prepare = prepare or (lambda: None)

    @property
    def bound(self) -> bool:
        """Whether ``bind`` has connected the actions; chunks are ignored until then."""
        return self._stop is not None

    def reset(self) -> None:
        """Forget the stretch in progress (the robot stopped speaking, or the audio has a gap)."""
        self._end_session(block=True)
        self._vad.reset()
        self._ring.clear()
        self._pending = np.empty(0, dtype=np.float32)
        self._pending_stamp = 0.0
        self._run = 0
        self._onset = 0.0
        self._weak_run = 0
        self._weak_onset = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Counters and latencies (ms) for the debug report; no audio, no text."""
        return {
            "enabled": self._b.enabled,
            "bound": self.bound,
            "keyword_enabled": self._b.keyword_enabled,
            **{**self._counts, "errors": self._counts["errors"] + self._worker.errors},
            "keyword_seconds": self._keyword_s,
            "last_echo_corr": self._last_corr,
            "latency": self.stats.snapshot(),
        }

    # ----------------------------------------------------------------- chunks

    def on_chunk(self, chunk: VoiceChunkLike) -> None:
        """One microphone chunk heard while the robot speaks (the listener's hook)."""
        b = self._b
        if self._stop is None or not (b.enabled or b.keyword_enabled):
            return
        if chunk.rate_hz != self._rate:
            self._counts["errors"] += 1
            return
        n = chunk.samples.size
        start = chunk.stamp - n / self._rate
        if chunk.gap:
            self.reset()
        if self._ring.maxlen != self._ring_len:  # a hot reload changed the windows
            self._ring = collections.deque(self._ring, maxlen=self._ring_len)
        if self._session is not None:  # the worker heard "stop" since the last chunk
            detected = self._worker.take_hit()
            if detected is not None:
                self._trigger("keyword", detected, detected, np.empty(0, np.float32))
                return
        if start < self._deaf_until:  # the cut-off just happened: this is its echo tail
            return
        if self._pending.size == 0:
            self._pending_stamp = start
        buf = (
            np.concatenate([self._pending, chunk.samples]) if self._pending.size else chunk.samples
        )
        pos = 0
        while buf.size - pos >= self._size:
            frame = buf[pos : pos + self._size]
            stamp = self._pending_stamp + pos / self._rate
            pos += self._size
            if self._frame(frame, stamp, buf[pos:]):
                return
        self._pending_stamp += pos / self._rate
        self._pending = buf[pos:].copy()

    def _frame(self, frame: np.ndarray, stamp: float, rest: np.ndarray) -> bool:
        """Advance by one frame; True when it triggered (the rest of the chunk is dropped)."""
        b = self._b
        prob = self._vad.speech_prob(frame)
        self._ring.append((stamp, frame))
        speech = b.enabled and prob >= b.threshold
        fricative = b.enabled and b.onset_enabled and self._fricative(frame, prob)
        candidate = b.keyword_enabled and prob >= b.keyword_threshold
        corr = self._echo(stamp + self._frame_s) if (speech or fricative or candidate) else 0.0
        if speech and corr >= b.echo_corr_max:
            speech = False
            self._counts["echo_frames"] += 1
        # The onset path counts a frame that passed the main path (and then under the stricter
        # onset_echo_corr_max: a frame at threshold or more with corr in [onset_echo_corr_max,
        # echo_corr_max) counts on the main path only), or a fricative-like one.
        weak = b.enabled and b.onset_enabled and speech and corr < b.onset_echo_corr_max
        if fricative:
            weak = not self._fricative_is_echo(frame, corr, stamp + self._frame_s)
        elif speech and b.onset_enabled and not weak:
            self._counts["onset_echo_frames"] += 1
        candidate = candidate and corr < b.keyword_echo_corr_max
        if speech:
            if self._run == 0:
                self._onset = stamp
            self._run += 1
            if self._run >= self._min_frames:
                self._trigger("speech", self._onset, stamp + self._frame_s, rest)
                return True
        else:
            self._run = 0
        if weak:
            if self._weak_run == 0:
                self._weak_onset = stamp
            self._weak_run += 1
            if self._weak_run >= self._onset_frames:
                self._trigger("speech", self._weak_onset, stamp + self._frame_s, rest)
                return True
        else:
            self._weak_run = 0
        return b.keyword_enabled and self._keywords(frame, stamp, candidate)

    def _fricative(self, frame: np.ndarray, prob: float) -> bool:
        """A frame the main path does not count but the onset path may: scored from
        ``onset_threshold`` up to ``threshold`` and mostly high-frequency energy (an /s/)."""
        b = self._b
        if prob < b.onset_threshold:
            return False
        return self._hf_ratio(frame) >= b.onset_hf_min_ratio

    def _fricative_is_echo(self, frame: np.ndarray, corr: float, end: float) -> bool:
        """Whether a fricative-like frame is dropped: too quiet to tell from the echo of the
        robot's own "s" at the noise floor, or correlated with the played audio in the high band
        (the caller has checked the full band)."""
        b = self._b
        if (
            self._level_db(frame) < b.onset_min_level_db
            or self._echo_hf(end) >= b.onset_echo_corr_max
        ):
            self._counts["onset_echo_frames"] += 1
            return True
        return False

    @staticmethod
    def _level_db(frame: np.ndarray) -> float:
        return DB_PER_DECADE * math.log10(float(np.mean(frame * frame)) + LOG_FLOOR)

    def _hf_ratio(self, frame: np.ndarray) -> float:
        """Share of the frame's energy above ``onset_hf_cutoff_hz`` (0 for a silent frame)."""
        power = np.abs(np.fft.rfft(frame)) ** 2
        total = float(power.sum())
        if total <= 0.0:
            return 0.0
        freqs = np.fft.rfftfreq(frame.size, 1.0 / self._rate)
        return float(power[freqs >= self._b.onset_hf_cutoff_hz].sum()) / total

    def _echo_hf(self, end: float) -> float:
        """Echo correlation in the band above ``onset_hf_cutoff_hz`` only: that is where a
        fricative's evidence is, and a user's /s/ is not hidden by the robot's low-frequency
        voice in it (the full-band value is ~1 while the robot speaks)."""
        if self._reference is None:
            return 0.0
        frames = list(self._ring)[-self._echo_frames :]
        mic = np.concatenate([f for _, f in frames])
        lag_s = self._b.echo_max_lag_ms / MS_PER_S
        ref = self._reference.window(end - mic.size / self._rate - lag_s, end)
        return echo_correlation(self._high_pass(mic), self._high_pass(ref))

    def _high_pass(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(x)
        spec[np.fft.rfftfreq(x.size, 1.0 / self._rate) < self._b.onset_hf_cutoff_hz] = 0
        return np.fft.irfft(spec, x.size).astype(np.float32)

    def _echo(self, end: float) -> float:
        """Correlation of the latest microphone audio (ending at ``end``) with what played."""
        self._last_corr = 0.0
        if self._reference is None:
            return 0.0
        frames = list(self._ring)[-self._echo_frames :]
        mic = np.concatenate([f for _, f in frames])
        lag_s = self._b.echo_max_lag_ms / MS_PER_S
        ref = self._reference.window(end - mic.size / self._rate - lag_s, end)
        self._last_corr = echo_correlation(mic, ref)
        return self._last_corr

    # ----------------------------------------------------------------- keywords

    def _keywords(self, frame: np.ndarray, stamp: float, candidate: bool) -> bool:
        """Hand the frame to the keyword worker; True when it has heard "stop" (inline mode: at
        once; background: on a later frame or chunk) and the robot was cut off."""
        session = self._session
        b = self._b
        try:
            if session is None:
                if not candidate:
                    return False
                lang = self._language()
                self._lang = lang
                session = self._session = _Session(stamp)
                frames = list(self._ring)
                self._keyword_s += sum(f.size for _, f in frames) / self._rate
                self._worker.start(lang, frames)
            else:
                self._keyword_s += frame.size / self._rate
                self._worker.feed(stamp, frame)
            if candidate:
                session.last_candidate = stamp
        except Exception as exc:
            self._counts["errors"] += 1
            log.warning("barge-in: keyword recognizer: %s", type(exc).__name__)
            self._end_session()
            return False
        detected = self._worker.take_hit()
        if detected is not None:
            self._trigger("keyword", detected, detected, np.empty(0, np.float32))
            return True
        over = stamp - session.started >= b.keyword_window_s
        idle = stamp - session.last_candidate >= b.keyword_hold_ms / MS_PER_S
        if over or idle:
            self._end_session()
        return False

    def _end_session(self, *, block: bool = False) -> None:
        """Drop the keyword session. ``block`` also waits for the worker and resets the
        recognizer, which the listener may use next (``reset``, after a cut-off)."""
        self._session = None
        self._worker.end(block=block)

    # ----------------------------------------------------------------- the cut

    def _trigger(self, kind: str, onset: float, detected: float, rest: np.ndarray) -> None:
        """Cut the robot off. ``stop`` comes first (it is the budgeted step), then the
        conversation, then the listener's handoff."""
        audio_from = onset - self._pre_frames * self._frame_s
        audio = [f for t, f in self._ring if t >= audio_from - EDGE_S]
        audio_stamp = min((t for t, _ in self._ring if t >= audio_from - EDGE_S), default=onset)
        audio.append(rest)
        samples = np.concatenate(audio)
        self._end_session()  # not blocking: the stop is the budgeted step; ``reset`` waits later
        t0 = time.perf_counter()
        stop = self._stop
        if stop is None:  # unbound: on_chunk never gets here, but a caller may
            return
        try:
            self._prepare()
        except Exception as exc:
            log.warning("barge-in: prepare: %s", type(exc).__name__)
        stop()
        stopped = self._clock()
        self._deaf_until = stopped + self._tail_s  # the speaking window the stop leaves behind
        self.stats.record("stop_call", time.perf_counter() - t0)
        self.stats.record("onset_to_stop", stopped - onset)  # until ``stop()`` returned
        self.stats.record("detect_to_stop", stopped - detected)
        self._counts["keyword_triggers" if kind == "keyword" else "triggers"] += 1
        if kind == "keyword" and self._emit is not None:
            try:
                self._emit("speech.keyword.stop", stopped, {"language": self._lang})
            except Exception as exc:
                log.warning("barge-in: stop event: %s", type(exc).__name__)
        try:
            self._interrupt()
        except Exception as exc:
            log.warning("barge-in: interrupt: %s", type(exc).__name__)
        self.reset()
        if kind == "speech" and self._resume is not None:
            self._resume(samples, audio_stamp)
