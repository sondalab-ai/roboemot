"""Facing check: does the face the voice came from look at the robot? (SPEC 8 stage 2)

The voice-at-face check (``roboemot.voice.direction``) cannot tell a person from a screen in the
same direction: a video near the owner was answered and commented on (robot run, 2026-10-30).
Someone watching a screen looks at the screen, not at the robot, so an utterance that would be
addressed without the wake word must also come from a face looking at the robot: its head yaw
(``roboemot.vision.facing``, 0 looking into the camera) within ``voice.facing.max_yaw_deg`` for
at least ``min_share`` of the samples that have one. ``VoiceDirection`` picks the samples (the
face matched at the voice, or, degraded, the face attention is on), asks ``decide`` once per
utterance and turns its outcome into the gate's verdict (``direction.FACING_VERDICTS``).

``decide`` gives one of ``OUTCOMES``: ``passed``, ``not_facing``, ``cpu_high`` (the whole CPU
has stayed above ``cpu_max_percent`` for ``cpu_hold_s``: face detection runs late and the yaw
is not trusted) or ``no_data`` (no sample has a yaw: vision paused, no usable landmarks, a face
too small). The CPU is read from ``psutil.cpu_times`` deltas of its own, at most every
``cpu_sample_s`` and only while an utterance is heard; nothing blocks.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import psutil

from ..config.schema import VoiceFacingConfig
from ..stats import P90, quantile

OUTCOMES = ("passed", "not_facing", "cpu_high", "no_data")
SUMMARY_SIZE = 64  # utterances kept for the yaw summary of the debug report
PERCENT = 100.0
IDLE_FIELDS = ("idle", "iowait")
GUEST_FIELDS = ("guest", "guest_nice")  # already counted in user and nice on Linux

CpuReader = Callable[[], tuple[float, float]]


def cpu_times() -> tuple[float, float]:
    """Cumulative (busy, total) CPU seconds of the whole machine, every core."""
    t = psutil.cpu_times()._asdict()
    total = sum(t.values()) - sum(t.get(k, 0.0) for k in GUEST_FIELDS)
    return total - sum(t.get(k, 0.0) for k in IDLE_FIELDS), total


class CpuLoad:
    """Whether the CPU has stayed busy for a while: each ``poll`` (throttled) measures the busy
    share since the previous one; above the threshold, the high stretch starts at that
    previous poll, and one poll at or below it ends the stretch. A previous poll more than
    ``cpu_hold_s`` old is no evidence of a stretch (samples are taken only while an utterance
    is heard): the stretch then restarts at this poll. Listener thread only."""

    def __init__(
        self, reader: CpuReader = cpu_times, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._read, self._clock = reader, clock
        self._prev: tuple[float, float] | None = None
        self._at = -math.inf
        self._above_since: float | None = None
        self.percent: float | None = None

    def poll(self, cfg: VoiceFacingConfig) -> None:
        """Take a sample when ``cpu_sample_s`` has passed since the last one."""
        now = self._clock()
        if now - self._at < cfg.cpu_sample_s:
            return
        try:
            busy, total = self._read()
        except Exception:
            return
        if self._prev is not None and total > self._prev[1]:
            self.percent = PERCENT * (busy - self._prev[0]) / (total - self._prev[1])
            if self.percent <= cfg.cpu_max_percent:
                self._above_since = None
            elif now - self._at > cfg.cpu_hold_s:
                self._above_since = now
            elif self._above_since is None:
                self._above_since = self._at
        self._prev, self._at = (busy, total), now

    def high(self, cfg: VoiceFacingConfig) -> bool:
        """The CPU has been above ``cpu_max_percent`` for at least ``cpu_hold_s``. Polls first,
        so it may read the CPU times (``/proc/stat`` on Linux) on the listener thread inside
        the wake gate's lock: one cheap read, at most every ``cpu_sample_s``."""
        self.poll(cfg)
        since = self._above_since
        return since is not None and self._clock() - since >= cfg.cpu_hold_s


class FacingCheck:
    """See the module docstring. ``poll`` and ``decide`` run on the listener thread (``decide``
    inside the wake gate's lock); ``_lock`` is a leaf lock for the counts ``snapshot`` reads."""

    def __init__(self, cfg: VoiceFacingConfig, cpu: CpuLoad | None = None) -> None:
        self._cfg = cfg
        self._cpu = cpu if cpu is not None else CpuLoad()
        self._lock = threading.Lock()
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._yaws_deg: deque[float] = deque(maxlen=SUMMARY_SIZE)

    def apply_config(self, cfg: VoiceFacingConfig) -> None:
        """Hot reload: every field."""
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def config(self) -> VoiceFacingConfig:
        return self._cfg

    def poll(self) -> None:
        """A CPU sample when one is due (called on each piece of utterance audio)."""
        self._cpu.poll(self._cfg)

    def decide(self, yaws: Sequence[float | None]) -> str:
        """The outcome (one of ``OUTCOMES``) from the head yaw (radians, None: none near the
        sample's time) of the checked face in each sample; counts it."""
        cfg = self._cfg
        if self._cpu.high(cfg):
            return self._record("cpu_high")
        known = [abs(y) for y in yaws if y is not None]
        if not known:
            return self._record("no_data")
        limit = math.radians(cfg.max_yaw_deg)
        facing = sum(1 for y in known if y <= limit)
        passed = facing >= cfg.min_share * len(known)
        return self._record("passed" if passed else "not_facing", known)

    def _record(self, outcome: str, yaws: list[float] | None = None) -> str:
        with self._lock:
            self._outcomes[outcome] += 1
            if yaws:
                self._yaws_deg.append(math.degrees(statistics.median(yaws)))
        return outcome

    def snapshot(self) -> dict[str, Any]:
        """Outcome counts, the last CPU sample and the summary of each judged utterance's median
        head yaw of the checked face (degrees, 0 looking at the robot): numbers only."""
        with self._lock:
            outcomes = dict(self._outcomes)
            yaws = list(self._yaws_deg)
        percent = self._cpu.percent
        return {
            "enabled": self._cfg.enabled,
            "checks": outcomes,
            "cpu_percent": None if percent is None else round(percent, 1),
            "yaw_deg": summary(yaws),
        }


def summary(values: list[float]) -> dict[str, Any]:
    """Count, min, median, p90, max and last of ``values`` (in order), rounded to 0.1."""
    out: dict[str, Any] = {"count": len(values)}
    if values:
        ordered = sorted(values)
        out |= {
            "min": round(ordered[0], 1),
            "median": round(statistics.median(ordered), 1),
            "p90": round(quantile(ordered, P90), 1),
            "max": round(ordered[-1], 1),
            "last": round(values[-1], 1),
        }
    return out
