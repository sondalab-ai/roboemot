"""The idle runner of the self-soothing sound (SPEC 7B, 11A).

Like the yawn scheduler (``circadian/motion.py``) it reads no clock and no global generator:
the app passes ``now`` to :meth:`SootheRunner.tick` and injects the ``random.Random``. It tracks
continuous idle time, schedules a sound lazily, picks a variant eligible in the circadian state,
synthesizes it, queues it on the speech output and publishes the appraisal event ``self.soothe``.
When idle ends while the sound plays it cancels exactly that sound.

Someone speaking is never sung over (SPEC 7B, owner decision 2026-11-04): while the listener
hears speech, or speech may be starting (``tick``'s ``speech``), no sound starts and the idle
clock restarts (counted ``held_by_speech``, once per stretch of speech that ended an idle
stretch with no sound playing), and a sound playing is cut (``cut_by_speech``). Speech that
starts while the sound plays reaches the runner through ``cut_by_speech``, which the listener
(speech already in progress as the sound starts) and barge-in (speech over the sound) call.

Threads: ``tick`` runs on the app tick thread, ``apply_config`` on the config watcher's thread
(``ConfigStack.start_watcher``) and ``cut_by_speech`` on the listener thread, so they take one
lock (then the output's, never the other way round); ``sounding`` and ``job`` only read.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from ..config.schema import SootheConfig, SootheVariant
from ..motion.beat import Beat
from .synth import BlirpPlan, HumPlan, plan_sound, render_plan

EVENT = "self.soothe"
RESTING = "resting"  # the circadian state name (circadian.model.State)


class RawOutput(Protocol):
    """What plays the sound: ``SpeechOutput`` (``play`` returns the job id, ``None`` when full)."""

    def play(self, samples: np.ndarray, rate_hz: int, volume_db: float = 0.0) -> int | None: ...
    def cancel(self, job_id: int) -> None: ...
    def live(self, job_id: int) -> bool: ...
    def started_at(self, job_id: int) -> float | None: ...


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
        self._lock = threading.Lock()
        self._idle_since: float | None = None
        self._anchor: float | None = None  # where the next gap is measured from (None: now)
        self._due: float | None = None
        self._cooldown_until = -math.inf
        self._last: SootheVariant | None = None
        self._job: int | None = None  # id of the sound queued or playing
        self._plan: HumPlan | BlirpPlan | None = None  # the plan of that sound
        self._sound_end = -math.inf  # estimate of the end of the sound from its start
        self._ended = -math.inf  # when the last sound was seen to be over (or was cut)
        self._held = False  # speech holds the sound back now (counted once per stretch)
        self._counts = {"played": 0, "held_by_speech": 0, "cut_by_speech": 0}

    def apply_config(self, cfg: SootheConfig) -> None:
        """Hot reload: new parameters and variants, schedule redrawn from now. A sound in flight
        finishes, unless the runner was switched off (the next tick then cuts it)."""
        with self._lock:
            self.cfg = cfg
            self._due = self._anchor = None

    @property
    def job(self) -> int | None:
        """Id of our sound while it is queued or playing, else None."""
        return self._job

    def sounding(self, now: float, tail_s: float = 0.0) -> bool:
        """Whether our own sound is queued or playing at ``now``, or ended less than ``tail_s``
        ago. The app uses it so that the "speaking" window our sound opens on the speaker, and
        its end, do not count as somebody else's audio and cancel it or reset the idle clock."""
        return self._job is not None or now < self._ended + tail_s

    def beat_at(self, now: float) -> Beat | None:
        """The planned beat grid of the hum playing at ``now`` (SPEC 7B, 7 Dance motion): its
        ``beat_s`` from the moment the output really started it, not the queue time. None for
        a blirp (irregular), before the hum reached the output, once it is over and at once
        after a barge-in cut it (the next ``tick`` is not waited for)."""
        with self._lock:
            plan, job = self._plan, self._job
            if job is None or not isinstance(plan, HumPlan) or not self.output.live(job):
                return None
            start = self.output.started_at(job)
        if start is None or not start <= now < start + plan.length_s:
            return None
        return Beat(plan.beat_s, start, 1.0, "own")

    def accents(self, since: float, now: float) -> list[float]:
        """Stamps of the events of the blirp playing that fall in ``(since, now]``, from the
        moment the output really started it. Empty for a hum or with nothing playing."""
        with self._lock:
            plan, job = self._plan, self._job
            if job is None or not isinstance(plan, BlirpPlan) or not self.output.live(job):
                return []
            start = self.output.started_at(job)
        if start is None:
            return []
        return [start + e.start_s for e in plan.events if since < start + e.start_s <= now]

    def snapshot(self) -> dict[str, Any]:
        """Counts for the debug report (SPEC 12A): sounds played, held back and cut by speech."""
        with self._lock:
            return {"enabled": self.cfg.enabled, "playing": self._job is not None, **self._counts}

    def cut_by_speech(self, now: float) -> bool:
        """Someone speaks over our sound: cut it if it is queued or playing (counted
        ``cut_by_speech``; the cooldown starts from the cut). True when there was one."""
        with self._lock:
            if self._job is None:
                return False
            self._counts["cut_by_speech"] += 1
            self._lose_idle(now)
            return True

    def tick(
        self, now: float, idle: bool, circadian_state: str, quiet: bool, speech: bool = False
    ) -> None:
        """One app tick. ``idle`` is the app's predicate (SPEC 7B), ``quiet`` the quiet hours,
        ``speech`` whether the listener hears speech now or speech may be starting. A non-idle
        tick, speech or a disabled runner resets the idle clock, clears the schedule and cuts a
        sound still playing. A sound that stopped before its estimated end was cut by a
        barge-in: the cooldown starts from the cut."""
        with self._lock:
            self._check_job(now)
            if speech and idle and self.cfg.enabled:
                self._hold(now)
                return
            self._held = False
            if not (idle and self.cfg.enabled):
                self._lose_idle(now)
                return
            if self._idle_since is None:
                self._idle_since = self._anchor = now
            if self._job is not None:
                return
            if self._due is None:
                at, self._anchor = self._anchor if self._anchor is not None else now, None
                self._schedule(at)
            elif now >= self._due:
                self._start(now, circadian_state, quiet)

    def close(self) -> None:
        """Shutdown: cut the sound if one is queued or playing."""
        with self._lock:
            if self._job is not None:
                self.output.cancel(self._job)
                self._job = None

    def _hold(self, now: float) -> None:
        """Speech in an idle stretch: cut our sound (``cut_by_speech``) or hold the next one back
        (``held_by_speech``, once per stretch of speech); idle is lost either way."""
        if self._job is not None:
            self._counts["cut_by_speech"] += 1
        elif not self._held and self._idle_since is not None:
            self._counts["held_by_speech"] += 1
        self._held = True
        self._lose_idle(now)

    def _check_job(self, now: float) -> None:
        """Forget our job once the output no longer has it: played out, or (before the
        estimated end) cut by a stop."""
        if self._job is None or self.output.live(self._job):
            return
        if now < self._sound_end:
            self._cooldown_until = now + self.cfg.cooldown_s
        self._job, self._ended = None, now

    def _schedule(self, at: float) -> None:
        """Next sound at ``at + interval * (1 + jitter * u)``, but not before ``idle_min_s`` of
        idle time and never inside the cooldown."""
        c = self.cfg
        gap = c.interval_s * (1 + c.jitter * self.rng.uniform(-1, 1))
        floor = -math.inf if self._idle_since is None else self._idle_since + c.idle_min_s
        self._due = max(at + gap, floor, self._cooldown_until)

    def _lose_idle(self, now: float) -> None:
        """Idle ended: forget the idle clock and the schedule; cut our sound if it still plays
        (the cooldown then starts from the cut)."""
        if self._job is not None:
            self.output.cancel(self._job)
            self._job, self._ended = None, now
            self._cooldown_until = now + self.cfg.cooldown_s
        self._idle_since = self._anchor = self._due = None

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
        plan = plan_sound(variant.sound, variant.shape, self.rng)
        samples = render_plan(plan, self.cfg.rate_hz)
        job = self.output.play(samples, self.cfg.rate_hz, self.volume_db())
        if job is None:  # queue full: no event, try again at the next schedule
            self._schedule(now)
            return
        self._job, self._plan, self._last = job, plan, variant
        self._sound_end = now + samples.size / self.cfg.rate_hz
        self._cooldown_until = self._sound_end + self.cfg.cooldown_s
        self._schedule(self._sound_end)
        self._counts["played"] += 1
        self.on_event(EVENT, now, {"label": variant.label, "sound": variant.sound})
