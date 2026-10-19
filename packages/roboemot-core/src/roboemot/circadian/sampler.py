"""Light sampling for the circadian rhythm (SPEC 7A, Light sensing from the camera; SPEC 12).

``LightSampler`` measures a camera frame handed to it and keeps scalars only: one value per
minute (brightness index, luma, noise) in a RAM ring buffer of ``light.history_h`` hours. The
frame is measured inside :meth:`LightSampler.offer`, on the caller's thread, and is never kept,
copied into storage, logged or passed on. Two producers feed it:

- the vision pipeline's ``light_tap`` (frames vision already captures), at most one per
  ``light.sample_interval_s``;
- :class:`IdleLightReader`, which reads one frame through the app's frame source when no sample
  arrived for ``circadian.idle_sample_interval_s`` (vision idle, off or stalled).

The level (dark, dim, bright) is the estimator's, with its hysteresis, and counts as the room's
level only after the samples have agreed on it for ``light.confirm_s``.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from .. import shutdown
from ..config.schema import CircadianConfig, LightConfig
from .light import COLOR_NDIM, Level, LightEstimator, LightReading
from .model import MINUTES_PER_HOUR, SECONDS_PER_MINUTE

log = logging.getLogger(__name__)

JOIN_GRACE_S = 5.0


class LightMinute(NamedTuple):
    """One minute of light: the means of the samples taken in it (all plain floats)."""

    minute: int  # whole minutes of the sampler clock
    index: float
    luma: float
    noise: float


@dataclass
class _Acc:
    minute: int
    n: int = 0
    index: float = 0.0
    luma: float = 0.0
    noise: float = 0.0

    def add(self, r: LightReading) -> None:
        self.n += 1
        self.index += r.index
        self.luma += r.luma
        self.noise += r.noise

    def done(self) -> LightMinute:
        return LightMinute(
            self.minute, self.index / self.n, self.luma / self.n, self.noise / self.n
        )


class LightSampler:
    """Thread-safe light sampler; ``clock`` is the circadian clock's monotonic seconds."""

    def __init__(
        self, light: LightConfig, circadian: CircadianConfig, clock: Callable[[], float]
    ) -> None:
        self.cfg = light
        self.circadian = circadian
        self._clock = clock
        self._lock = threading.Lock()
        self._estimator = LightEstimator(light)
        self._ring: deque[LightMinute] = deque(maxlen=self._ring_len(light))
        self._acc: _Acc | None = None
        self._last: float | None = None  # clock time of the latest sample
        self._attempt: float | None = None  # clock time of the latest idle read, kept or not
        self._streak_level: Level | None = None
        self._streak_since = 0.0
        self._noise: float | None = None  # smoothed image noise
        self._noise_at = 0.0
        self._confirmed: Level | None = None
        self.samples = 0

    @staticmethod
    def _ring_len(light: LightConfig) -> int:
        return max(1, round(light.history_h * MINUTES_PER_HOUR))

    @property
    def enabled(self) -> bool:
        """False with ``light_source: clock_only``: the camera is never read."""
        return self.circadian.enabled and self.circadian.light_source == "camera"

    def apply_config(self, light: LightConfig, circadian: CircadianConfig) -> None:
        """Hot reload: new thresholds and intervals; the history is kept (resized if needed)."""
        with self._lock:
            self.cfg, self.circadian = light, circadian
            self._estimator.cfg = light
            if self._ring.maxlen != self._ring_len(light):
                self._ring = deque(self._ring, maxlen=self._ring_len(light))

    # ---- producers

    def due(self) -> bool:
        """Whether the vision tap should measure a frame now."""
        with self._lock:
            return self._due(self.cfg.sample_interval_s)

    def idle_due(self) -> bool:
        """Whether no sample arrived for ``idle_sample_interval_s``: read a frame ourselves."""
        with self._lock:
            return self._due(self.circadian.idle_sample_interval_s, self._attempt)

    def note_attempt(self) -> None:
        """An idle read was made (it may have returned no frame): wait a full interval."""
        with self._lock:
            self._attempt = self._clock()

    def _due(self, interval: float, attempt: float | None = None) -> bool:
        if not self.enabled:
            return False
        last = [t for t in (self._last, attempt) if t is not None]
        return not last or self._clock() - max(last) >= interval

    def offer(
        self,
        image: np.ndarray,
        *,
        exposure_s: float | None = None,
        gain: float | None = None,
        bgr: bool = True,
        force: bool = False,
    ) -> bool:
        """Measure ``image`` if a sample is due (or ``force``); True when it was measured.

        ``bgr`` says the channels are blue-green-red (the camera's order); the view is
        reversed, not copied. Only the numbers leave this call.
        """
        if not self.enabled or not (force or self.due()):
            return False
        img = np.asarray(image)
        if img.ndim < 2 or img.size == 0:  # not a picture: nothing to measure
            return False
        if bgr and img.ndim == COLOR_NDIM:
            img = img[..., ::-1]
        luma, noise = self._estimator.stats(img)
        self.offer_stats(luma, noise, exposure_s=exposure_s, gain=gain)
        return True

    def offer_stats(
        self,
        luma: float,
        noise: float,
        *,
        exposure_s: float | None = None,
        gain: float | None = None,
    ) -> LightReading | None:
        """Take an already-measured sample (a logged one in a replay), whenever it is due or
        not; None, and nothing kept, with ``light_source: clock_only``."""
        if not self.enabled:
            return None
        with self._lock:
            now = self._clock()
            smooth = self._smooth_noise(noise, now)
            reading = self._estimator.classify(luma, smooth, exposure_s=exposure_s, gain=gain)
            self._add(LightReading(reading.level, luma, noise, reading.index), now)
            return reading

    def _smooth_noise(self, noise: float, now: float) -> float:
        """Exponentially smoothed noise (``light.noise_smooth_s``) for the dark-noise rule."""
        tau = self.cfg.noise_smooth_s
        if self._noise is None or tau <= 0:
            self._noise = noise
        else:
            self._noise += (1 - math.exp(-max(0.0, now - self._noise_at) / tau)) * (
                noise - self._noise
            )
        self._noise_at = now
        return self._noise

    def _add(self, r: LightReading, now: float) -> None:
        if self._last is not None and now - self._last > self.cfg.stale_s:
            # A gap longer than stale_s: whatever was confirmed before is history, and a streak
            # across the gap proves nothing. Confirm anew.
            self._confirmed, self._streak_level = None, None
        self._last = now
        self.samples += 1
        minute = int(now // SECONDS_PER_MINUTE)
        if self._acc is not None and self._acc.minute != minute:
            self._ring.append(self._acc.done())
            self._acc = None
        if self._acc is None:
            self._acc = _Acc(minute)
        self._acc.add(r)
        if r.level != self._streak_level:
            self._streak_level, self._streak_since = r.level, now
        if now - self._streak_since >= self.cfg.confirm_s:
            self._confirmed = r.level

    # ---- readers

    def level(self) -> Level | None:
        """The confirmed light level, or None before the first confirmation or when the latest
        sample is older than ``light.stale_s`` (the clock then rules alone)."""
        with self._lock:
            if self._confirmed is None or self._last is None:
                return None
            return self._confirmed if self._clock() - self._last <= self.cfg.stale_s else None

    def history(self) -> list[LightMinute]:
        """The per-minute values, oldest first, including the minute in progress."""
        with self._lock:
            out = list(self._ring)
            if self._acc is not None:
                out.append(self._acc.done())
            return out[-(self._ring.maxlen or len(out)) :]

    def snapshot(self) -> dict[str, Any]:
        """Numbers for the debug report: counts and ages only."""
        with self._lock:
            age = None if self._last is None else round(self._clock() - self._last, 1)
            return {
                "enabled": self.enabled,
                "samples": self.samples,
                "minutes_kept": len(self._ring) + (self._acc is not None),
                "last_sample_age_s": age,
                "level": self._confirmed,
            }


class IdleLightReader:
    """Reads one frame through the app's frame source whenever the sampler has had none for
    ``circadian.idle_sample_interval_s``; call :meth:`stop` before the camera goes away.

    The read runs in a thread of its own so a slow SDK call never stalls the app loop. A read
    that fails is logged by type and retried at the next interval.
    """

    def __init__(self, sampler: LightSampler, source: Any) -> None:
        self.sampler = sampler
        self.source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reads = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="roboemot-light", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            shutdown.join(self._thread, self.sampler.circadian.idle_poll_s * 2 + JOIN_GRACE_S)
            if self._thread.is_alive():  # a camera read stuck in the SDK cannot be interrupted
                log.warning("light reader did not stop in time")

    def step(self) -> bool:
        """One check: read and measure a frame if one is due. True when it measured."""
        if not self.sampler.idle_due():
            return False
        self.sampler.note_attempt()
        try:
            frame = self.source.read()
        except Exception as exc:
            log.warning("light sampling: camera read failed: %s", type(exc).__name__)
            return False
        self.reads += 1
        if frame is None:
            return False
        try:
            return self.sampler.offer(frame.image, force=True)
        except Exception as exc:
            log.warning("light sampling: frame not measurable: %s", type(exc).__name__)
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(self.sampler.circadian.idle_poll_s)
