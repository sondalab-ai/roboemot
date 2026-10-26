"""Barge-in ducking: the robot's voice gain while it may be hearing the user (SPEC 8 stage 8).

:class:`~roboemot.voice.bargein.BargeIn` marks an *episode* on the monotonic clock: the robot's
voice should be at ``voice.barge_in.duck.gain`` for every sample that plays in
``[start, until)`` and at full gain outside it. Each candidate frame pushes ``until`` forward to
the moment the release would start if no other candidate came, so audio queued while ducked
already carries the release. The speech output applies the gain on the samples it hands to the
sink (:meth:`DuckGain.apply`), keyed to the time each sample will *play*, not the time it is
pushed: the sink queues ahead of playback, and audio already handed over cannot be changed.

The applied level never steps: it moves towards the target at ``(1 - gain) / attack_ms`` going
down and ``(1 - gain) / ramp_ms`` going up, continuously across pushes (the level of the last
pushed sample is kept, and moves during silence between pushes). ``clear`` (a stop flushed the
player, or the robot stopped speaking) forgets the episode and the level: the next push plays at
full gain.

Thread-safe: the listener thread writes the episode, the speech output thread applies it, and the
work on samples runs outside the lock so the listener never waits for it.
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
        self._level_at = -math.inf  # play time just after the last pushed sample
        self._generation = 0  # bumped by ``clear``: an ``apply`` in flight does not write back
        self.apply_config(cfg)

    def apply_config(self, cfg: BargeInDuckConfig) -> None:
        """Hot reload: the gain and the two ramps (the episode in progress keeps going)."""
        span = max(1.0 - cfg.gain, MIN_SPAN)
        attack_s, ramp_s = cfg.attack_ms / MS_PER_S, cfg.ramp_ms / MS_PER_S
        with self._lock:
            self._gain = cfg.gain
            self._down = span / attack_s if attack_s > 0 else math.inf
            self._up = span / ramp_s if ramp_s > 0 else math.inf

    def duck(self, start: float, until: float) -> None:
        """Duck the samples that play in ``[start, until)``. An episode still running at
        ``start`` is extended to ``until``; one already over is replaced."""
        with self._lock:
            if start >= self._until:
                self._start = start
            self._until = max(self._until, until)

    def release(self, at: float) -> None:
        """End the episode at ``at``: later samples ramp back to full gain."""
        with self._lock:
            self._until = min(self._until, at)

    def clear(self) -> None:
        """No episode and full gain at once: the player was flushed or is idle."""
        with self._lock:
            self._start = self._until = -math.inf
            self._level, self._level_at = 1.0, -math.inf
            self._generation += 1

    def pushed_until(self) -> float:
        """Play time just after the last sample ``apply`` saw (-inf after ``clear``): what the
        sink has queued up to, read without touching the sink's lock."""
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
        with self._lock:
            ep_start, until, gain = self._start, self._until, self._gain
            down, up, generation = self._down, self._up, self._generation
            level, level_at = self._level, self._level_at
        end = start + n / rate_hz if rate_hz > 0 else start
        if level_at < start:  # silence since the last push: the level kept moving
            level = self._slew(level, level_at, start, ep_start, until, gain, down, up)
        if level >= 1.0 and (n == 0 or end <= ep_start or start >= until):
            out, last = samples, 1.0
        else:
            gains = self._curve(level, start, n, rate_hz, ep_start, until, gain, down, up)
            out = (np.asarray(samples, np.float32) * gains).astype(np.float32, copy=False)
            last = self._slew(level, start, end, ep_start, until, gain, down, up)
        with self._lock:
            if generation == self._generation:
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
                seg = np.maximum(target, level - down * dt) if math.isfinite(down) else target
            else:
                seg = np.minimum(target, level + up * dt) if math.isfinite(up) else target
            out[a:b] = seg
            # the level where the next segment starts (its first sample's play time)
            nxt = t[b] if b < n else t[-1] + 1.0 / rate_hz
            gap = nxt - t[a]
            if target < level:
                level = max(target, level - down * gap)
            else:
                level = min(target, level + up * gap)
        return out
