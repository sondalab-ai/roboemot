"""Rate governor: per-stage target rates scaled by circadian state, load and conversation
(SPEC 5, 13, 7A).

Stage rates come from ``vision.rates`` (``<stage>_hz``). Scales multiply: the circadian
``vision_rate_scale``, a load scale the CPU and thermal governor of SPEC 13 sets, and the
conversation scale (``vision.rates.conversation_scale`` while the robot thinks or speaks, SPEC 8,
so the TTS keeps its CPU, and ``listening_scale`` while the user speaks, so the listener keeps
up). A scale of 0 turns every stage off. Decisions use capture stamps
only, so replays are deterministic; a frame that arrives before its stage is due is dropped,
never queued (latest-wins).

A stage with an ``<stage>_active_hz`` field may be made active (``set_active``): it then runs at
that rate instead of ``<stage>_hz``, under the same scales. Attention does this for ``objects``
while a shown object is its target (SPEC 5A, Joint attention), so the gaze can follow it, and
passes the target's base-frame bearing (``focus``) so the pipeline knows which track is shown.
The object ceiling ``objects_max_hz`` holds by construction: the config rejects ``objects_hz`` or
``objects_active_hz`` above it, and every scale is at most 1.

A stage with an ``<stage>_idle_hz`` field may be made idle (``set_idle``): it then runs at the
lower of that rate and ``<stage>_hz``, under the same scales, so an idle rate can only lower
it. The pipeline does this for ``detection`` once ``vision.rates.detection_idle_after_s`` of
capture time passed with no face track, and clears it on the first face track (SPEC 5, 13,
owner decision 2026-11-11). An active rate wins over an idle one. A ``detection_idle_hz`` at
or above ``detection_hz`` has no effect, so an old config with a low ``detection_hz`` keeps
loading and keeps its rate.
"""

from __future__ import annotations

import math
import threading

from ..config.schema import VisionRates

CIRCADIAN = "circadian"
LOAD = "load"
CONVERSATION = "conversation"
EPSILON_S = 1e-9


class RateGovernor:
    """Decides, from capture stamps, whether a stage may run now."""

    def __init__(self, rates: VisionRates):
        self.rates = rates
        self._lock = threading.Lock()
        self._scales: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._active: set[str] = set()
        self._idle: set[str] = set()
        self._focus: dict[str, tuple[float, float]] = {}
        self.dropped: dict[str, int] = {}

    def set_scale(self, source: str, value: float) -> None:
        """Set one multiplier (``CIRCADIAN``, ``LOAD`` or ``CONVERSATION``), clamped to [0, 1].

        The circadian model sets ``CIRCADIAN`` from its state's ``vision_rate_scale``; the CPU
        and thermal governor sets ``LOAD``; the app tick sets ``CONVERSATION`` from the
        conversation phase. They multiply; 0 turns every stage off.
        """
        with self._lock:
            self._scales[source] = min(1.0, max(0.0, value))

    def set_active(self, stage: str, on: bool, focus: tuple[float, float] | None = None) -> None:
        """Run ``stage`` at its ``<stage>_active_hz`` rate (on) or its usual rate (off); a stage
        without that field ignores it. ``focus`` is the base-frame (azimuth, elevation) of what
        the active rate follows, in radians; None when unknown, and always cleared when off."""
        with self._lock:
            if on:
                self._active.add(stage)
            else:
                self._active.discard(stage)
            if on and focus is not None:
                self._focus[stage] = focus
            else:
                self._focus.pop(stage, None)

    def set_idle(self, stage: str, on: bool) -> None:
        """Run ``stage`` at its ``<stage>_idle_hz`` rate, never above ``<stage>_hz`` (on), or its
        usual rate (off); a stage without that field ignores it. The caller decides from capture
        stamps, so a replay stays deterministic."""
        with self._lock:
            if on:
                self._idle.add(stage)
            else:
                self._idle.discard(stage)

    def idle(self, stage: str) -> bool:
        """Whether ``stage`` is idle and has an idle rate (``period`` lets an active rate win)."""
        with self._lock:
            return stage in self._idle and hasattr(self.rates, f"{stage}_idle_hz")

    def focus(self, stage: str) -> tuple[float, float] | None:
        """The base-frame bearing the active ``stage`` follows (``set_active``), or None."""
        with self._lock:
            return self._focus.get(stage) if stage in self._active else None

    def active(self, stage: str) -> bool:
        """Whether ``stage`` runs at its active rate now."""
        with self._lock:
            return stage in self._active and hasattr(self.rates, f"{stage}_active_hz")

    @property
    def scale(self) -> float:
        """Product of all scales; 1 when none is set."""
        with self._lock:
            return math.prod(self._scales.values())

    def scales(self) -> dict[str, float]:
        """Each multiplier set so far by source, for the debug report (SPEC 12A)."""
        with self._lock:
            return dict(self._scales)

    def scale_of(self, source: str) -> float:
        """The multiplier one source set; 1 when it set none."""
        with self._lock:
            return self._scales.get(source, 1.0)

    def stretch(self, gap_s: float, source: str) -> float:
        """A minimum gap between two runs, stretched by one source's scale: ``gap_s / scale``,
        infinite when that scale is 0. A gap of 0 stays 0 while the scale is above 0 (no 0/0).
        The pipeline stretches the new-track burst gaps by ``CONVERSATION`` only, so the
        circadian and load scales keep bursts bounded by the detection period alone."""
        scale = self.scale_of(source)
        if scale <= 0:
            return math.inf
        return gap_s / scale

    def period(self, stage: str) -> float:
        """Seconds between two runs of ``stage`` (infinite when the scale is 0).

        ``stage`` names a field of ``vision.rates`` without its ``_hz`` suffix, for example
        ``detection`` or ``embedding``. The pipeline also uses it for per-track schedules. An
        active stage uses ``<stage>_active_hz``, an idle one the lower of ``<stage>_idle_hz``
        and ``<stage>_hz``.
        """
        if self.active(stage):
            base = getattr(self.rates, f"{stage}_active_hz")
        elif self.idle(stage):
            base = min(getattr(self.rates, f"{stage}_idle_hz"), getattr(self.rates, f"{stage}_hz"))
        else:
            base = getattr(self.rates, f"{stage}_hz")
        rate = base * self.scale
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
        """Forget when each stage last ran and which stages were idle (new session or replay
        start)."""
        with self._lock:
            self._last.clear()
            self._idle.clear()
            self.dropped.clear()
