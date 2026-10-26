"""The idle runner of the self-soothing sound (SPEC 7B, 11A).

Like the yawn scheduler (``circadian/motion.py``) it reads no clock and no global generator:
the app passes ``now`` to :meth:`SootheRunner.tick` and injects the ``random.Random``. It tracks
continuous idle time, schedules a sound lazily, picks a variant eligible in the circadian state,
synthesizes it, queues it on the speech output and publishes the appraisal event ``self.soothe``.
When idle ends while the sound plays it cancels exactly that sound.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..config.schema import SootheConfig, SootheVariant
from .synth import synthesize

EVENT = "self.soothe"
RESTING = "resting"


class RawOutput(Protocol):
    """What plays the sound: ``SpeechOutput`` (``play`` returns the job id, ``None`` when full)."""

    def play(self, samples: np.ndarray, rate_hz: int, volume_db: float = 0.0) -> int | None: ...
    def cancel(self, job_id: int) -> None: ...


class SootheRunner:
    """Schedules, plays and cancels the self-soothing sound."""

    def __init__(
        self,
        cfg: SootheConfig,
        output: RawOutput,
        *,
        on_event: Callable[[str, float, dict[str, Any]], None],
        rng: random.Random | None = None,
        volume_db: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.cfg = cfg
        self.output = output
        self.on_event = on_event
        self.rng = rng or random.Random()  # noqa: S311 (a pick and a jitter, not security)
        self.volume_db = volume_db  # the circadian volume offset now
        self._idle_since: float | None = None
        self._due: float | None = None
        self._cooldown_until = -math.inf
        self._last: SootheVariant | None = None
        self._job: int | None = None  # id of the sound queued or playing, until it has played
        self._sound_end = -math.inf  # when the last sound ends (or was cut)

    def apply_config(self, cfg: SootheConfig) -> None:
        """Hot reload: new parameters and variants, schedule redrawn. A sound in flight
        finishes, unless the runner was switched off (the next tick then cuts it)."""
        self.cfg = cfg
        self._due = None

    @property
    def job(self) -> int | None:
        """Id of our sound while it is queued or playing, else None."""
        return self._job

    def sounding(self, now: float, tail_s: float = 0.0) -> bool:
        """Whether our own sound plays at ``now``, or ended less than ``tail_s`` ago. The app
        uses it so that the "speaking" window our sound opens on the speaker does not count as
        somebody else's audio and cancel the sound (or reset the idle clock at its end)."""
        return now < self._sound_end + tail_s

    def tick(self, now: float, idle: bool, circadian_state: str, quiet: bool) -> None:
        """One app tick. ``idle`` is the app's predicate (SPEC 7B), ``quiet`` the quiet hours.
        A non-idle tick (or a disabled runner) resets the idle clock, clears the schedule and
        cuts a sound still playing."""
        if self._job is not None and now >= self._sound_end:
            self._job = None  # played to its end: nothing to cancel
        if not (idle and self.cfg.enabled):
            self._lose_idle(now)
            return
        if self._idle_since is None:
            self._idle_since = now
        if self._job is not None or now - self._idle_since < self.cfg.idle_min_s:
            return
        if self._due is None:
            self._schedule(now)
        elif now >= self._due:
            self._start(now, circadian_state, quiet)

    def close(self) -> None:
        """Shutdown: cut the sound if one is queued or playing."""
        if self._job is not None:
            self.output.cancel(self._job)
            self._job = None

    def _schedule(self, at: float) -> None:
        """Next sound at ``at + interval * (1 + jitter * u)``, never inside the cooldown."""
        c = self.cfg
        gap = c.interval_s * (1 + c.jitter * self.rng.uniform(-1, 1))
        self._due = max(at + gap, self._cooldown_until)

    def _lose_idle(self, now: float) -> None:
        """Idle ended: forget the idle clock and the schedule; cut our sound if it still plays
        (the cooldown then starts from the cut)."""
        if self._job is not None:
            self.output.cancel(self._job)
            self._job, self._sound_end = None, now
            self._cooldown_until = now + self.cfg.cooldown_s
        self._idle_since = self._due = None

    def _eligible(self, state: str) -> list[SootheVariant]:
        """Variants allowed in ``state``, without the previous pick unless it is the only one."""
        pool = [v for v in self.cfg.variants if state in v.states]
        fresh = [v for v in pool if v != self._last]
        return fresh or pool

    def _start(self, now: float, state: str, quiet: bool) -> None:
        """The sound is due: play one if the state allows, then schedule the next."""
        pool = [] if state == RESTING or quiet else self._eligible(state)
        if not pool:
            self._schedule(now)
            return
        variant = self.rng.choices(pool, weights=[v.weight for v in pool])[0]
        samples = synthesize(variant.sound, variant.shape, self.rng, self.cfg.rate_hz)
        job = self.output.play(samples, self.cfg.rate_hz, self.volume_db())
        if job is None:  # queue full: no event, try again at the next schedule
            self._schedule(now)
            return
        self._job, self._last = job, variant
        self._sound_end = now + samples.size / self.cfg.rate_hz
        self._cooldown_until = self._sound_end + self.cfg.cooldown_s
        self._schedule(self._sound_end)
        self.on_event(EVENT, now, {"label": variant.label, "sound": variant.sound})
