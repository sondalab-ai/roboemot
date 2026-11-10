"""Music in the room, from the audio the listener hears (SPEC 8, 11A).

The listener feeds every chunk it ingests (never one heard while the robot speaks). Per hop of
``music.hop_ms`` one real FFT over the last ``music.fft_ms`` gives the energy in each of
``music.bands_hz``; the onset strength of a band is the half-wave-rectified rise of its log energy
(floored at ``music.min_level_db``, so a band with nothing in it carries no onsets). The summed
onsets form an envelope of about 31 values a second, kept for ``music.window_s``. Before an
estimate the envelope loses its moving average over ``music.detrend_s`` (half-wave rectified), so
a dense sound such as metal keeps only the rises above its own level, and a triangular smoothing
of ``music.smooth_radius_hops`` lines up a beat spread over two hops. Chunks wait in a batch of
``music.batch_ms`` and their hops are analysed together, because most of the CPU cost is per NumPy
call and not per sample (QA 2026-11-09); a hole in the audio analyses what waits first.

Every ``music.estimate_interval_s``, once the envelope holds two periods of the slowest tempo
and when the last interval's level is above ``music.min_level_db``, the mean-removed envelope is
autocorrelated over the lags of ``music.tempo_bpm``; the peak (refined between lags) is the
tempo, folded into ``music.prior_bpm`` when its double or half there is periodic enough. An
estimate is *periodic* when the peak reaches ``music.min_periodicity``, at least
``music.min_bands_active`` bands reach ``music.min_band_periodicity`` on their own envelope at
that lag (a ticking clock is one narrow band) and the window was heard for
``music.min_active_share``. It *holds* when it is periodic and its tempo is within
``music.tempo_tolerance`` of the previous periodic estimate's, or is twice or half that tempo
with the new envelope still periodic at the previous lag (an octave flip of real music).
The beat phase is the offset whose beat grid sums the most envelope (a comb).

While music plays, an estimate that reaches ``music.keep_periodicity`` also restarts the release,
unless ``speech_heard`` says someone is speaking: a song in a real room reads far below the start
threshold, and speech must not keep the music state alive. For the same reason an estimate made
while someone speaks cannot start music: it does not count as periodic while no music plays.

Music starts when an estimate holds and the evidence (back-dated to the first heard hop of the
window of the first periodic estimate in the chain) spans ``music.confirm_s``; the music interval
starts at the evidence. It ends after ``music.release_s`` of heard audio without a holding
estimate. A hole in the audio longer than ``music.max_gap_s`` clears the envelope; state, tempo
and phase run on (a flywheel) and the hole does not count toward the release. Nothing heard for
longer than ``music.blind_hold_s`` ends it.

``state`` and ``during`` read one immutable view (state and intervals together), replaced by
one assignment: no lock. ``snapshot`` has counts (with the reason of each estimate that did not
hold), tempo, confidence, cost (CPU ms of the feeding thread per second of audio,
``time.thread_time`` by default) and a summary of the last estimates only: no audio, no spectrum,
no envelope (SPEC 12).
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..config.schema import SECONDS_PER_MINUTE, MusicConfig
from ..stats import P90, quantile

MS_PER_S = 1000.0
DB_PER_DECADE = 10.0
COST_SAMPLES = 600  # report only, not behaviour: one per estimate interval, ten minutes
RECENT_ESTIMATES = 60  # report only, not behaviour: about the last minute of estimates
STARTED, STOPPED = "started", "stopped"
HOLD, KEEP = "hold", "keep"
QUIET, LOW_PERIODICITY, LOW_HEARD, FEW_BANDS = "quiet", "low_periodicity", "low_heard", "few_bands"
SPEECH = "speech"
FIRST_PERIODIC, UNSTABLE_TEMPO = "first_periodic", "unstable_tempo"
REJECTIONS = (
    QUIET,
    LOW_PERIODICITY,
    LOW_HEARD,
    FEW_BANDS,
    SPEECH,
    FIRST_PERIODIC,
    UNSTABLE_TEMPO,
)


@dataclass(frozen=True)
class MusicState:
    """Music at ``stamp``: whether it plays, its tempo, the time since the last beat
    (``beat_phase_s``, in [0, 60 / tempo_bpm)) and the last estimate's periodicity (0 to 1).
    Tempo and phase are None while no music plays."""

    playing: bool
    tempo_bpm: float | None
    beat_phase_s: float | None
    confidence: float
    stamp: float


Interval = tuple[float, float | None]  # (start, end); end None while the music plays


@dataclass(frozen=True)
class _View:
    """Everything another thread reads, swapped as one object: the state and the intervals
    can never be seen out of step."""

    playing: bool
    period_s: float | None
    anchor_s: float | None  # the stamp of one beat; the phase runs from it
    confidence: float
    heard_until: float  # end of the last audio fed
    intervals: tuple[Interval, ...] = ()


_SILENT = _View(False, None, None, 0.0, -math.inf)


@dataclass(frozen=True)
class _Estimate:
    period_s: float
    anchor_s: float
    periodicity: float
    periodic: bool  # periodicity, bands and heard share pass; stability is checked apart
    corr: np.ndarray = field(compare=False, repr=False)  # the envelope's autocorrelation by lag
    rejected: str | None = None  # the first rule the estimate failed, None when periodic
    bands: int | None = None  # bands carrying the beat; None when an earlier rule failed


@dataclass(frozen=True)
class _Outcome:
    """One estimate interval for the report: how it ended and the numbers behind it."""

    name: str
    periodicity: float | None
    bands: int | None
    level_db: float | None


class _Bands:
    """Band energies of analysis windows: one batched real FFT under a Hann window, summed per
    band and scaled so that a band's value is its share of the window's mean-square level."""

    def __init__(self, cfg: MusicConfig, rate_hz: int, size: int) -> None:
        nyquist = rate_hz / 2
        if max(hi for _, hi in cfg.bands_hz) > nyquist:
            raise ValueError(f"music.bands_hz reaches above the Nyquist frequency {nyquist} Hz")
        self._window = np.hanning(size)
        freqs = np.fft.rfftfreq(size, 1.0 / rate_hz)
        scale = 2 / (size * float(np.sum(self._window**2)))
        self._weights = np.stack(
            [((freqs >= lo) & (freqs < hi)) * scale for lo, hi in cfg.bands_hz], axis=1
        )

    def energy(self, frames: np.ndarray) -> np.ndarray:
        """(hops, window size) samples to (hops, bands) band energies."""
        spectrum = np.fft.rfft(frames * self._window, axis=1)
        power = spectrum.real**2 + spectrum.imag**2
        return power @ self._weights


class MusicDetector:
    """See the module docstring. ``feed``, ``gap`` and ``apply_config``'s effect run on the
    listener thread; ``state``, ``during`` and ``snapshot`` run on any thread."""

    def __init__(
        self,
        cfg: MusicConfig,
        rate_hz: int,
        timer: Callable[[], float] = time.thread_time,
    ) -> None:
        self._rate = rate_hz
        self._timer = timer
        self._lock = threading.Lock()  # leaf lock: counts and cost samples for snapshot
        self._counts = {
            "estimates": 0,
            "holds": 0,
            "keeps": 0,
            "starts": 0,
            "stops": 0,
            "resets": 0,
        }
        self.speech_heard: Callable[[], bool] | None = None  # "someone is speaking now"
        self._rejected = dict.fromkeys(REJECTIONS, 0)
        self._recent: deque[_Outcome] = deque(maxlen=RECENT_ESTIMATES)
        self._costs: deque[float] = deque(maxlen=COST_SAMPLES)
        self._cost_busy_s = 0.0
        self._cost_audio_s = 0.0
        self._pending: MusicConfig | None = None
        self._view = _SILENT
        self._last_hold_s = -math.inf
        self._blind_from: float | None = None
        self._configure(cfg)

    # ------------------------------------------------------------------ configuration

    def _configure(self, cfg: MusicConfig) -> None:
        """Size the hop, the rings and the band filter for ``cfg`` and clear the envelope.
        Raises ValueError when a band reaches above the Nyquist frequency."""
        self._cfg = cfg
        self._hop = _hop_samples(cfg, self._rate)
        self._hop_s = self._hop / self._rate
        self._fft = _fft_samples(cfg, self._rate)
        self._detrend_hops = round(cfg.detrend_s / self._hop_s)
        kernel = np.bartlett(2 * (cfg.smooth_radius_hops + 1) + 1)[1:-1]
        self._kernel = kernel / np.sum(kernel)
        self._bands = _Bands(cfg, self._rate, self._fft)
        self._batch_samples = round(cfg.batch_ms / MS_PER_S * self._rate)
        self._ring_n = max(2, round(cfg.window_s / self._hop_s))
        self._interval_hops = max(1, round(cfg.estimate_interval_s / self._hop_s))
        periods_s = [SECONDS_PER_MINUTE / bpm for bpm in cfg.tempo_bpm]
        self._lag_lo = max(1, math.floor(min(periods_s) / self._hop_s))
        self._lag_hi = math.ceil(max(periods_s) / self._hop_s)
        self._min_fill = min(self._ring_n, 2 * self._lag_hi)  # two slowest periods
        self._floor = DB_PER_DECADE ** (cfg.min_level_db / DB_PER_DECADE)
        self._clear()

    def apply_config(self, cfg: MusicConfig) -> None:
        """Hot reload (any thread): ``cfg`` takes effect at the next ``feed``, which rebuilds
        the analysis and clears the envelope; state, tempo and history are kept. Disabling ends
        music at that feed."""
        _Bands(cfg, self._rate, _fft_samples(cfg, self._rate))  # a bad band fails here, not later
        self._pending = cfg

    def _clear(self) -> None:
        """Empty the envelope rings, the hop carry and the estimate chain (after a gap or a
        reload); the published view and the interval history stay."""
        n, bands = self._ring_n, len(self._cfg.bands_hz)
        self._onsets = np.zeros((n, bands))
        self._stamps = np.zeros(n)
        self._heard = np.zeros(n, dtype=bool)
        self._filled = 0
        self._head = 0
        self._carry = np.zeros(0, dtype=np.float32)
        self._carry_start = 0.0
        self._tail = np.zeros(self._fft - self._hop)  # samples before the next hop, for its window
        self._batch: list[np.ndarray] = []
        self._batch_n = 0
        self._batch_start = self._batch_end = 0.0
        self._batch_hole: float | None = None
        self._prev_log: np.ndarray | None = None
        self._powers: deque[float] = deque(maxlen=self._interval_hops)
        self._since_estimate = self._interval_hops  # the first full-enough window estimates
        self._previous: _Estimate | None = None
        self._evidence_from: float | None = None

    # ------------------------------------------------------------------ listener thread

    def gap(self, stamp: float) -> None:
        """A hole in the audio (a chunk gap, the robot speaking) began at ``stamp``. The next
        ``feed`` measures it; one longer than ``music.max_gap_s`` clears the envelope."""
        if self._blind_from is None or stamp < self._blind_from:
            self._blind_from = stamp

    def feed(self, samples: np.ndarray, start: float) -> str | None:
        """Analyse one mono chunk that starts at ``start`` (monotonic seconds). Returns
        ``"started"`` or ``"stopped"`` when the music state changed with it, else None. A chunk
        after more than ``music.blind_hold_s`` of nothing ends music first."""
        t0 = self._timer()
        audio = np.asarray(samples, dtype=np.float32)
        try:
            return self._feed_batched(audio, start)
        finally:
            self._account(self._timer() - t0, audio.size / self._rate)

    def _feed_batched(self, samples: np.ndarray, start: float) -> str | None:
        """A pending reload first, then ``samples`` join the batch, which is analysed once it
        holds ``music.batch_ms``. A hole before the chunk analyses the waiting batch first, so
        the audio on each side of the hole stays apart."""
        if self._pending is not None:
            cfg, self._pending = self._pending, None
            self._configure(cfg)
        if not self._cfg.enabled:
            return self._analyse(samples, start)
        event = None
        joined = abs(start - self._batch_end) <= self._hop_s and self._blind_from is None
        if self._batch and not joined:
            event = self._flush()
        if not self._batch:
            self._batch_start, self._batch_hole = start, self._blind_from
            self._blind_from = None
        self._batch.append(samples)
        self._batch_n += samples.size
        self._batch_end = start + samples.size / self._rate
        if self._batch_n >= self._batch_samples:
            event = self._flush() or event
        return event

    def _flush(self) -> str | None:
        """Analyse the waiting batch with the hole that came before it. A hole reported after
        the batch began stays for the next batch."""
        audio = self._batch[0] if len(self._batch) == 1 else np.concatenate(self._batch)
        self._batch, self._batch_n = [], 0
        later, self._blind_from = self._blind_from, self._batch_hole
        self._batch_hole = None
        try:
            return self._analyse(audio, self._batch_start)
        finally:
            self._blind_from = later

    def _analyse(self, samples: np.ndarray, start: float) -> str | None:
        """The hole before the audio, its hops and the estimates that fall due, and the release
        check at its end."""
        cfg = self._cfg
        end = start + samples.size / self._rate
        if not cfg.enabled:
            self._blind_from = None
            return self._stop(start) if self._view.playing else None
        event = self._after_hole(start)
        for stamp, row, heard in self._hops(samples, start):
            self._push(stamp, row, heard)
            self._since_estimate += 1
            if self._since_estimate >= self._interval_hops and self._filled >= self._min_fill:
                self._since_estimate = 0
                event = self._estimate(stamp + self._hop_s) or event
        self._view = replace(self._view, heard_until=end)
        if self._view.playing and end - self._last_hold_s >= cfg.release_s:
            event = self._stop(end)
        return event

    def _after_hole(self, start: float) -> str | None:
        """Handle the hole before a chunk: end music after a blind hold, clear the envelope
        after a long hole, and keep the hole out of the release clock."""
        cfg = self._cfg
        heard_until = self._view.heard_until
        hole_from = heard_until if self._blind_from is None else min(self._blind_from, start)
        self._blind_from = None
        if not math.isfinite(hole_from):
            return None
        hole_s = max(0.0, start - hole_from)
        event = None
        if self._view.playing and start - heard_until > cfg.blind_hold_s:
            event = self._stop(heard_until + cfg.blind_hold_s)
        elif self._view.playing:
            self._last_hold_s += hole_s
        if hole_s > cfg.max_gap_s:
            self._clear()
            with self._lock:
                self._counts["resets"] += 1
        return event

    def _hops(self, samples: np.ndarray, start: float) -> list[tuple[float, np.ndarray, bool]]:
        """Whole hops of the carry plus ``samples``: (start stamp, onset per band, heard) each.
        A chunk that does not follow the carry drops it."""
        if self._carry.size and abs(start - self._expected_next()) > self._hop_s:
            self._carry = np.zeros(0, dtype=np.float32)
        if not self._carry.size:
            self._carry_start = start
        audio = np.concatenate([self._carry, samples]) if self._carry.size else samples
        count = audio.size // self._hop
        first = self._carry_start
        self._carry = audio[count * self._hop :].copy()
        self._carry_start = first + count * self._hop_s
        if count == 0:
            return []
        hops = audio[: count * self._hop].astype(np.float64)
        joined = np.concatenate([self._tail, hops])
        self._tail = joined[joined.size - self._tail.size :]
        starts = self._hop * np.arange(count)
        energy = self._bands.energy(joined[starts[:, None] + np.arange(self._fft)[None, :]])
        powers = np.mean(hops.reshape(count, self._hop) ** 2, axis=1)
        logs = np.log(energy + self._floor)
        prev = logs[0] if self._prev_log is None else self._prev_log
        rises = np.maximum(0.0, np.diff(np.vstack([prev, logs]), axis=0))
        self._prev_log = logs[-1]
        out = []
        for i in range(count):
            self._powers.append(float(powers[i]))
            heard = statistics.fmean(self._powers) >= self._floor
            out.append((first + i * self._hop_s, rises[i], heard))
        return out

    def _expected_next(self) -> float:
        """Stamp of the next sample a continuous chunk would start with (after the carry)."""
        return self._carry_start + self._carry.size / self._rate

    def _push(self, stamp: float, row: np.ndarray, heard: bool) -> None:
        """Write one hop into the rings; a hop not heard (level under ``music.min_level_db``)
        carries no onsets."""
        i = self._head
        self._onsets[i] = row if heard else 0.0
        self._stamps[i] = stamp
        self._heard[i] = heard
        self._head = (i + 1) % self._ring_n
        self._filled = min(self._ring_n, self._filled + 1)

    def _ordered(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The filled part of the rings, oldest first: onsets per band, stamps, heard flags."""
        n = self._filled
        idx = (self._head - n + np.arange(n)) % self._ring_n
        return self._onsets[idx], self._stamps[idx], self._heard[idx]

    # ------------------------------------------------------------------ estimate

    def _estimate(self, now: float) -> str | None:
        """One tempo estimate at ``now``; updates the evidence chain and the state."""
        cfg = self._cfg
        result = None
        level = statistics.fmean(self._powers) if self._powers else 0.0
        if level >= self._floor:
            result = self._unless_speech(self._periodic())
            with self._lock:
                self._counts["estimates"] += 1
        previous, self._previous = self._previous, result
        kept = self._keeps_playing(result)
        if kept:
            self._last_hold_s = now
        if result is None or not result.periodic:
            self._evidence_from = None
            reason = QUIET if result is None else str(result.rejected)
            self._note(KEEP if kept else reason, result, level)
            self._publish(result)
            return None
        chained = previous is not None and previous.periodic
        if not (chained and previous is not None and self._stable(previous, result)):
            self._evidence_from = self._first_heard()
            reason = UNSTABLE_TEMPO if chained else FIRST_PERIODIC
            self._note(KEEP if kept else reason, result, level)
            self._publish(result)
            return None
        self._last_hold_s = now
        self._note(HOLD, result, level)
        event = None
        since = self._evidence_from
        if not self._view.playing and since is not None and now - since >= cfg.confirm_s:
            event = self._start(since, now)
        self._publish(result)
        return event

    def _speaking(self) -> bool:
        """Whether the listener hears someone speaking now."""
        return self.speech_heard is not None and self.speech_heard()

    def _unless_speech(self, result: _Estimate) -> _Estimate:
        """``result``, no longer periodic when it could start music while someone speaks."""
        if result.periodic and not self._view.playing and self._speaking():
            return replace(result, periodic=False, rejected=SPEECH)
        return result

    def _keeps_playing(self, result: _Estimate | None) -> bool:
        """Whether this estimate keeps music that already plays: periodic enough for the lower
        ``music.keep_periodicity``, and nobody is speaking."""
        if result is None or not self._view.playing:
            return False
        if result.periodicity < self._cfg.keep_periodicity:
            return False
        return not self._speaking()

    def _note(self, name: str, result: _Estimate | None, level: float) -> None:
        """Count one estimate interval's outcome and keep its numbers for the report (SPEC 12A):
        no audio, spectrum or envelope. ``level`` is the interval's mean power."""
        outcome = _Outcome(
            name,
            None if result is None else result.periodicity,
            None if result is None else result.bands,
            DB_PER_DECADE * math.log10(level) if level > 0 else None,
        )
        with self._lock:
            if name == HOLD:
                self._counts["holds"] += 1
            elif name == KEEP:
                self._counts["keeps"] += 1
            else:
                self._rejected[name] += 1
            self._recent.append(outcome)

    def _shaped(self, onsets: np.ndarray) -> np.ndarray:
        """``onsets`` as the estimate reads them: above their moving average, then smoothed."""
        return np.convolve(_detrend(onsets, self._detrend_hops), self._kernel, "same")

    def _bands_carrying(self, onsets: np.ndarray, lag: float) -> int:
        """Bands that carry the beat at ``lag``: periodic on their own envelope and not nearly
        empty. The last check of an estimate, because it costs one correlation per band."""
        cfg = self._cfg
        strengths = onsets.mean(axis=0)
        least = cfg.min_band_onset_share * float(np.max(strengths))
        return sum(
            1
            for band, strength in zip(onsets.T, strengths, strict=True)
            if strength >= least
            and _autocorr_at(self._shaped(band), lag) >= cfg.min_band_periodicity
        )

    def _periodic(self) -> _Estimate:
        """Tempo, beat anchor and periodicity of the current envelope, and whether it is
        periodic on its own (periodicity, bands carrying the beat, heard share)."""
        cfg = self._cfg
        onsets, stamps, heard = self._ordered()
        envelope = self._shaped(onsets.sum(axis=1))
        corr = _autocorr(envelope, self._lag_hi + 1)
        lags = np.arange(self._lag_lo, self._lag_hi + 1)
        peak = int(lags[np.argmax(corr[lags])])
        lag = self._fold(corr, _refine(corr, peak))
        periodicity = float(_at(corr, lag))
        rejected, bands = None, None
        if periodicity < cfg.min_periodicity:
            rejected = LOW_PERIODICITY
        elif float(np.mean(heard)) < cfg.min_active_share:
            rejected = LOW_HEARD
        else:
            bands = self._bands_carrying(onsets, lag)
            if bands < cfg.min_bands_active:
                rejected = FEW_BANDS
        anchor = _comb(envelope, stamps, lag)
        return _Estimate(
            lag * self._hop_s, anchor, periodicity, rejected is None, corr, rejected, bands
        )

    def _fold(self, corr: np.ndarray, lag: float) -> float:
        """Fold ``lag`` into the prior tempo range when its half or double lies there and is
        periodic enough on its own; else keep it."""
        cfg = self._cfg
        lo, hi = cfg.prior_bpm
        for cand in (lag / 2, lag * 2):
            bpm = SECONDS_PER_MINUTE / (cand * self._hop_s)
            if lo <= bpm <= hi and 1 <= cand <= self._lag_hi:
                refined = _refine(corr, round(cand))
                if _at(corr, refined) >= cfg.min_periodicity:
                    return refined
        return lag

    def _stable(self, a: _Estimate, b: _Estimate) -> bool:
        """Whether ``b`` keeps ``a``'s tempo: its period within ``music.tempo_tolerance`` of
        ``a``'s, or an octave flip (twice or half) that ``b``'s own envelope backs, its
        autocorrelation at ``a``'s lag reaching ``music.min_periodicity``. Speech that peaks at
        unrelated lags from one estimate to the next does not chain this way."""
        tol = self._cfg.tempo_tolerance
        if abs(b.period_s - a.period_s) <= tol * a.period_s:
            return True
        flipped = any(abs(b.period_s * k - a.period_s) <= tol * a.period_s for k in (0.5, 2))
        backed = _at(b.corr, a.period_s / self._hop_s) >= self._cfg.min_periodicity
        return flipped and backed

    def _first_heard(self) -> float:
        """Stamp of the first heard hop in the window: where the evidence starts."""
        _, stamps, heard = self._ordered()
        hit = np.flatnonzero(heard)
        return float(stamps[hit[0]]) if hit.size else float(stamps[-1])

    # ------------------------------------------------------------------ state changes

    def _publish(self, result: _Estimate | None) -> None:
        """Swap in a new view: the estimate's tempo and anchor when it was periodic (``state``
        shows them only while music plays), its periodicity as the confidence always."""
        view = self._view
        confidence = 0.0 if result is None else min(1.0, max(0.0, result.periodicity))
        period, anchor = view.period_s, view.anchor_s
        if result is not None and result.periodic:
            period, anchor = result.period_s, result.anchor_s
        self._view = replace(view, period_s=period, anchor_s=anchor, confidence=confidence)

    def _start(self, since: float, now: float) -> str:
        """Music starts: open an interval back-dated to ``since`` (merged with a closed one that
        reaches it), drop intervals older than ``music.history_s`` before ``now``."""
        kept = self._pruned(now)
        if kept and kept[-1][1] is not None and kept[-1][1] >= since:
            kept = kept[:-1] + ((min(kept[-1][0], since), None),)
        else:
            kept = kept + ((since, None),)
        self._view = replace(self._view, playing=True, intervals=kept)
        with self._lock:
            self._counts["starts"] += 1
        return STARTED

    def _stop(self, stamp: float) -> str:
        """Music ends at ``stamp``: close the open interval there and forget tempo and phase;
        the confidence stays for the report."""
        kept = self._view.intervals
        if kept and kept[-1][1] is None:
            kept = kept[:-1] + ((kept[-1][0], stamp),)
        self._view = replace(
            self._view, playing=False, period_s=None, anchor_s=None, intervals=kept
        )
        self._last_hold_s = -math.inf
        with self._lock:
            self._counts["stops"] += 1
        return STOPPED

    def _pruned(self, now: float) -> tuple[Interval, ...]:
        """The intervals still open or ended within ``music.history_s`` before ``now``."""
        oldest = now - self._cfg.history_s
        return tuple(iv for iv in self._view.intervals if iv[1] is None or iv[1] >= oldest)

    def _account(self, busy_s: float, audio_s: float) -> None:
        """Add one feed's time; close a cost sample once an estimate interval of audio is in."""
        self._cost_busy_s += busy_s
        self._cost_audio_s += audio_s
        if self._cost_audio_s >= self._cfg.estimate_interval_s:
            sample = self._cost_busy_s / self._cost_audio_s * MS_PER_S
            self._cost_busy_s = self._cost_audio_s = 0.0
            with self._lock:
                self._costs.append(sample)

    # ------------------------------------------------------------------ any thread

    def state(self, now: float) -> MusicState:
        """Music at ``now`` (lock-free): the beat phase runs on from the last anchor, and music
        is off once nothing was heard for longer than ``music.blind_hold_s``."""
        view = self._view
        playing = view.playing and now - view.heard_until <= self._cfg.blind_hold_s
        if not playing or view.period_s is None or view.anchor_s is None:
            return MusicState(playing, None, None, view.confidence, now)
        tempo = SECONDS_PER_MINUTE / view.period_s
        phase = (now - view.anchor_s) % view.period_s
        return MusicState(True, tempo, phase, view.confidence, now)

    def during(self, start: float, end: float) -> bool:
        """Whether music played at any time in [start, end] (lock-free), over the back-dated
        interval history; an open interval lasts while ``state`` says playing."""
        view = self._view
        for begin, finish in view.intervals:
            if finish is None:
                finish = view.heard_until + self._cfg.blind_hold_s if view.playing else begin
            if begin <= end and start <= finish:
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        """Counts, the reasons estimates did not hold, state, tempo, confidence, cost (ms per
        second of audio) and a summary of the last estimates: numbers only, no audio (SPEC 12)."""
        with self._lock:
            counts = dict(self._counts)
            rejected = dict(self._rejected)
            costs = list(self._costs)
            recent = list(self._recent)
        view = self._view
        tempo = SECONDS_PER_MINUTE / view.period_s if view.playing and view.period_s else None
        return {
            "enabled": self._cfg.enabled,
            **counts,
            "rejected": rejected,
            "playing": view.playing,
            "tempo_bpm": tempo,
            "confidence": view.confidence,
            "cost_ms_per_audio_s": _spread(costs),
            "recent": _recent_report(recent),
        }


def _spread(values: list[float]) -> dict[str, Any]:
    """Number of ``values`` and their min, median, p90 and last, all None with no values."""
    if not values:
        return {"samples": 0, "min": None, "median": None, "p90": None, "last": None}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": quantile(ordered, P90),
        "last": values[-1],
    }


def _recent_report(recent: list[_Outcome]) -> dict[str, Any]:
    """The last estimate intervals for the report: how many ended each way, and the spread of
    the periodicity, the bands carrying the beat and the level (each over the intervals that
    have it)."""
    outcomes: dict[str, int] = {}
    for item in recent:
        outcomes[item.name] = outcomes.get(item.name, 0) + 1
    return {
        "outcomes": outcomes,
        "periodicity": _spread([i.periodicity for i in recent if i.periodicity is not None]),
        "bands_carrying": _spread([float(i.bands) for i in recent if i.bands is not None]),
        "level_db": _spread([i.level_db for i in recent if i.level_db is not None]),
    }


def _hop_samples(cfg: MusicConfig, rate_hz: int) -> int:
    """Samples in one hop of ``music.hop_ms`` at ``rate_hz`` (at least two, for the FFT)."""
    return max(2, round(cfg.hop_ms / MS_PER_S * rate_hz))


def _fft_samples(cfg: MusicConfig, rate_hz: int) -> int:
    """Samples in one analysis window of ``music.fft_ms`` at ``rate_hz`` (at least one hop)."""
    return max(_hop_samples(cfg, rate_hz), round(cfg.fft_ms / MS_PER_S * rate_hz))


def _detrend(x: np.ndarray, hops: int) -> np.ndarray:
    """``x`` above its centred moving average of ``hops`` values, half-wave rectified. The
    average shrinks at the edges; ``hops`` under 2 returns ``x`` unchanged."""
    if hops < 2:
        return x
    total = np.concatenate([[0.0], np.cumsum(x)])
    index = np.arange(x.size)
    lo, hi = np.maximum(0, index - hops // 2), np.minimum(x.size, index + hops // 2 + 1)
    return np.maximum(0.0, x - (total[hi] - total[lo]) / (hi - lo))


def _autocorr(x: np.ndarray, lags: int) -> np.ndarray:
    """Normalised, unbiased autocorrelation of mean-removed ``x`` for lags 0 to ``lags``
    (inclusive); all zero when ``x`` has no variance."""
    x = x - np.mean(x)
    n = x.size
    energy = float(np.dot(x, x)) / n
    out = np.zeros(lags + 1)
    if energy <= 0:
        return out
    full = np.correlate(x, x, "full")[n - 1 : n + lags]
    count = n - np.arange(full.size)
    out[: full.size] = full / count / energy
    return out


def _autocorr_at(x: np.ndarray, lag: float) -> float:
    """``_autocorr`` of ``x`` at one fractional lag, linearly between the two lags around it;
    0 when ``x`` has no variance. Costs two dot products instead of a full correlation."""
    x = x - np.mean(x)
    n = x.size
    energy = float(np.dot(x, x)) / n
    if energy <= 0:
        return 0.0
    below = math.floor(lag)
    weight = lag - below
    near, far = (float(np.dot(x[: n - k], x[k:])) / (n - k) / energy for k in (below, below + 1))
    return (1 - weight) * near + weight * far


def _refine(corr: np.ndarray, peak: int) -> float:
    """Fractional lag of a peak at integer ``peak`` by a parabola through its neighbours."""
    if peak < 1 or peak + 1 >= corr.size:
        return float(peak)
    a, b, c = corr[peak - 1], corr[peak], corr[peak + 1]
    curve = a - 2 * b + c
    if curve >= 0 or b < a or b < c:  # not a local maximum: the parabola's vertex is elsewhere
        return float(peak)
    return peak + 0.5 * (a - c) / curve


def _at(corr: np.ndarray, lag: float) -> float:
    """``corr`` at a fractional lag, linearly between the two lags around it."""
    return float(np.interp(lag, np.arange(corr.size), corr))


def _comb(envelope: np.ndarray, stamps: np.ndarray, lag: float) -> float:
    """The stamp of the most recent beat: the offset from the end whose grid of period ``lag``
    hops sums the most envelope."""
    n = envelope.size
    offsets = np.arange(math.ceil(lag))
    beats = np.arange(math.floor((n - 1) / lag) + 1)
    idx = np.rint(n - 1 - offsets[:, None] - beats[None, :] * lag).astype(int)
    sums = np.where(idx >= 0, envelope[np.clip(idx, 0, n - 1)], 0.0).sum(axis=1)
    return float(stamps[n - 1 - int(offsets[np.argmax(sums)])])
