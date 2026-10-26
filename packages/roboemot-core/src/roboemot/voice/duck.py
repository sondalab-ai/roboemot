"""Barge-in ducking: the robot's voice gain while it may be hearing the user (SPEC 8 stage 8).

:class:`~roboemot.voice.bargein.BargeIn` marks an *episode* on the monotonic clock: the robot's
voice should be at ``voice.barge_in.duck.gain`` for every sample that plays from ``start`` on,
and at full gain before it. The episode stays open while candidate frames continue (the user may
still be talking, so nothing comes back up early) and closes only when barge-in decides the
release (``release(at)``: ``release_ms`` of non-candidate frames) or the cut-off clears it. The
gain is applied (:meth:`DuckGain.apply`) where audio leaves for the player, keyed to the time
each sample will *play*, not the time it is handed over: by the speaker at its paced hand-off
when it takes the duck (only a bounded lead is then out of reach), else by the speech output
before the sink. Audio already in the player cannot be changed.

The applied level never steps: it moves towards the target at ``(1 - gain) / attack_ms`` going
down and ``(1 - gain) / ramp_ms`` going up (both at least 1 ms), continuously across pushes (the
level of the last sample is kept, and moves during silence between pushes). ``clear`` (a stop
flushed the player, or the robot stopped speaking) forgets the episode and the level: the next
push plays at full gain.

Thread-safe: the listener thread writes the episode, the output thread applies it. ``apply``
holds the lock while it computes (a 100 ms piece is a few microseconds), so a ``clear`` from
the listener lands either before an apply, which then plays at full gain, or after it, never in
the middle with a stale gain written back.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from ..config.schema import BargeInDuckConfig

MS_PER_S = 1000.0
# Smallest gain span used for the slew rate, so a level left below a raised gain still recovers.
MIN_SPAN = 1e-3


class DuckGain:
    """See the module docstring."""

    def __init__(self, cfg: BargeInDuckConfig) -> None:
        self._lock = threading.Lock()
        self._start = self._until = -math.inf
        self._level = 1.0
        self._level_at = -math.inf  # play time just after the last sample ``apply`` saw
        self.apply_config(cfg)

    def apply_config(self, cfg: BargeInDuckConfig) -> None:
        """Hot reload: the gain and the two ramps (the episode in progress keeps going)."""
        span = max(1.0 - cfg.gain, MIN_SPAN)
        with self._lock:
            self._gain = cfg.gain
            self._down = span / (cfg.attack_ms / MS_PER_S)
            self._up = span / (cfg.ramp_ms / MS_PER_S)

    def duck(self, start: float) -> None:
        """A candidate: duck from ``start`` on, until ``release``. An open episode keeps its
        start; one already released is replaced."""
        with self._lock:
            if start >= self._until:
                self._start = start
            self._until = math.inf

    def release(self, at: float) -> None:
        """Close the episode at ``at``: later samples ramp back to full gain."""
        with self._lock:
            self._until = min(self._until, at)

    def clear(self) -> None:
        """No episode and full gain at once: the player was flushed or is idle."""
        with self._lock:
            self._start = self._until = -math.inf
            self._level, self._level_at = 1.0, -math.inf

    def pushed_until(self) -> float:
        """Play time just after the last sample ``apply`` saw (-inf after ``clear``): the end of
        the audio handed to the player, read without touching the sink's lock."""
        with self._lock:
            return self._level_at

    def ducking(self, at: float) -> bool:
        """Whether a sample that plays at ``at`` is in the episode."""
        with self._lock:
            return self._start <= at < self._until

    def apply(self, samples: np.ndarray, rate_hz: int, start: float) -> np.ndarray:
        """``samples`` (mono, ``rate_hz``) that start to play at ``start``, with the gain on
        them. Returned unchanged (the same array) when they play at full gain throughout."""
        n = samples.size
        if n == 0 or rate_hz <= 0:
            return samples
        end = start + n / rate_hz
        with self._lock:
            ep_start, until, gain = self._start, self._until, self._gain
            down, up, level = self._down, self._up, self._level
            if self._level_at < start:  # silence since the last push: the level kept moving
                level = self._slew(level, self._level_at, start, ep_start, until, gain, down, up)
            if level >= 1.0 and (gain >= 1.0 or end <= ep_start or start >= until):
                out, last = samples, 1.0
            else:
                gains = self._curve(level, start, n, rate_hz, ep_start, until, gain, down, up)
                out = (np.asarray(samples, np.float32) * gains).astype(np.float32, copy=False)
                last = self._slew(level, start, end, ep_start, until, gain, down, up)
            self._level, self._level_at = last, max(self._level_at, end)
        return out

    @staticmethod
    def _slew(
        level: float,
        t0: float,
        t1: float,
        ep_start: float,
        until: float,
        gain: float,
        down: float,
        up: float,
    ) -> float:
        """The level at ``t1`` given ``level`` at ``t0`` (``t0`` may be -inf: settled)."""
        if not math.isfinite(t0):
            return gain if ep_start <= t1 < until else 1.0
        cuts = sorted({t0, t1, *(t for t in (ep_start, until) if t0 < t < t1)})
        for a, b in zip(cuts, cuts[1:], strict=False):
            target = gain if ep_start <= a < until else 1.0
            rate = down if target < level else up
            step = rate * (b - a)
            level = max(target, level - step) if target < level else min(target, level + step)
        return level

    @staticmethod
    def _curve(
        level: float,
        start: float,
        n: int,
        rate_hz: int,
        ep_start: float,
        until: float,
        gain: float,
        down: float,
        up: float,
    ) -> np.ndarray:
        """The gain of each of ``n`` samples from ``start``: piecewise in at most three
        segments (before, inside and after the episode), each a clamped linear ramp."""
        t = start + np.arange(n) / rate_hz
        out = np.empty(n, np.float32)
        bounds = [0, *sorted({int(np.searchsorted(t, x)) for x in (ep_start, until)}), n]
        for a, b in zip(bounds, bounds[1:], strict=False):
            if b <= a:
                continue
            target = gain if ep_start <= t[a] < until else 1.0
            dt = t[a:b] - t[a]
            if target < level:
                out[a:b] = np.maximum(target, level - down * dt)
            else:
                out[a:b] = np.minimum(target, level + up * dt)
            # the level where the next segment starts (its first sample's play time)
            nxt = t[b] if b < n else t[-1] + 1.0 / rate_hz
            gap = nxt - t[a]
            if target < level:
                level = max(target, level - down * gap)
            else:
                level = min(target, level + up * gap)
        return out
