"""Rate governor: per-stage target rates scaled by circadian state and load (SPEC 5, 13, 7A).

Stage rates come from ``vision.rates`` (``<stage>_hz``). Scales multiply: the circadian
``vision_rate_scale`` and a load scale the CPU and thermal governor of SPEC 13 sets. A scale
of 0 turns every stage off. Decisions use capture stamps only, so replays are deterministic;
a frame that arrives before its stage is due is dropped, never queued (latest-wins).
"""

from __future__ import annotations

import math
import threading

from ..config.schema import VisionRates

CIRCADIAN = "circadian"
LOAD = "load"
EPSILON_S = 1e-9


class RateGovernor:
    """Decides, from capture stamps, whether a stage may run now."""

    def __init__(self, rates: VisionRates):
        self.rates = rates
        self._lock = threading.Lock()
        self._scales: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self.dropped: dict[str, int] = {}

    def set_scale(self, source: str, value: float) -> None:
        """Set one multiplier (``CIRCADIAN`` or ``LOAD``), clamped to [0, 1].

        The circadian model sets ``CIRCADIAN`` from its state's ``vision_rate_scale``; the CPU
        and thermal governor sets ``LOAD``. They multiply; 0 turns every stage off.
        """
        with self._lock:
            self._scales[source] = min(1.0, max(0.0, value))

    @property
    def scale(self) -> float:
        """Product of all scales; 1 when none is set."""
        with self._lock:
            return math.prod(self._scales.values())

    def period(self, stage: str) -> float:
        """Seconds between two runs of ``stage`` (infinite when the scale is 0).

        ``stage`` names a field of ``vision.rates`` without its ``_hz`` suffix, for example
        ``detection`` or ``embedding``. The pipeline also uses it for per-track schedules.
        """
        rate = getattr(self.rates, f"{stage}_hz") * self.scale
        return math.inf if rate <= 0 else 1.0 / rate

    def allow(self, stage: str, stamp: float) -> bool:
        """True and recorded when ``stage`` is due at ``stamp``; otherwise count a drop.

        Only capture stamps are compared, never the wall clock. A stage that is not due is
        skipped, not queued: the caller drops the frame for that stage (latest-wins).
        """
        period = self.period(stage)
        with self._lock:
            if math.isinf(period):
                self.dropped[stage] = self.dropped.get(stage, 0) + 1
                return False
            last = self._last.get(stage)
            if last is None or stamp - last >= period - EPSILON_S:
                self._last[stage] = stamp
                return True
            self.dropped[stage] = self.dropped.get(stage, 0) + 1
            return False

    def reset(self) -> None:
        """Forget when each stage last ran (new session or replay start)."""
        with self._lock:
            self._last.clear()
            self.dropped.clear()
