"""Affect engine (SPEC 7, Affect state).

Continuous valence/arousal in [-1, 1] decaying toward a baseline (set by the circadian model,
SPEC 7A) with a configurable half-life; appraisal rules turn bus events into impulses with
cooldowns; LLM impulses are capped per conversational turn. The discrete label is the
appraisal label while it is held, otherwise the nearest label prototype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config.schema import AffectConfig, AppraisalConfig


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass
class AffectState:
    valence: float
    arousal: float
    label: str
    intensity: float


@dataclass
class AffectEngine:
    cfg: AffectConfig
    appraisal: AppraisalConfig
    valence: float = 0.0
    arousal: float = 0.0
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.0
    _last: float | None = None
    _event_label: tuple[str, float] | None = None  # (label, until)
    _cooldown: dict[str, float] = field(default_factory=dict)
    _llm_budget: float = 0.0

    def __post_init__(self) -> None:
        self._llm_budget = self.cfg.llm_impulse_max

    # ---- time

    def update(self, now: float) -> AffectState:
        if not math.isfinite(now):
            raise ValueError("affect update time must be finite")
        if self._last is not None and now > self._last:
            k = math.pow(0.5, (now - self._last) / self.cfg.half_life_s)
            self.valence = self.baseline_valence + (self.valence - self.baseline_valence) * k
            self.arousal = self.baseline_arousal + (self.arousal - self.baseline_arousal) * k
        self._last = now
        return self.state(now)

    def set_baseline(self, valence: float | None = None, arousal: float | None = None) -> None:
        if valence is not None:
            self.baseline_valence = _clamp(valence)
        if arousal is not None:
            self.baseline_arousal = _clamp(arousal)

    # ---- inputs

    def impulse(
        self, now: float, valence: float = 0.0, arousal: float = 0.0, label: str | None = None
    ) -> None:
        if not all(math.isfinite(x) for x in (now, valence, arousal)):
            raise ValueError("affect impulse must be finite")
        self.update(now)
        self.valence = _clamp(self.valence + valence)
        self.arousal = _clamp(self.arousal + arousal)
        if label:
            if label not in self.cfg.labels:
                raise KeyError(f"unknown affect label {label}")
            self._event_label = (label, now + self.cfg.event_label_hold_s)

    def on_event(self, event: str, now: float) -> bool:
        """Apply matching appraisal rules; returns True if any rule fired."""
        fired = False
        for i, rule in enumerate(self.appraisal.rules):
            if rule.event != event:
                continue
            key = f"{i}:{rule.event}"
            if now < self._cooldown.get(key, -math.inf):
                continue
            self._cooldown[key] = now + rule.cooldown_s
            self.impulse(now, rule.valence, rule.arousal, rule.label)
            fired = True
        return fired

    def new_turn(self) -> None:
        self._llm_budget = self.cfg.llm_impulse_max

    def llm_impulse(self, now: float, valence: float, arousal: float) -> tuple[float, float]:
        """LLM ``express_emotion``: clamped so conversation cannot override the robot's own
        state (SPEC 7). Returns the impulse actually applied."""
        if not (math.isfinite(valence) and math.isfinite(arousal)):
            raise ValueError("LLM impulse must be finite")  # json.loads accepts NaN
        mag = math.hypot(valence, arousal)
        if mag == 0 or self._llm_budget <= 0:
            return 0.0, 0.0
        scale = min(1.0, self._llm_budget / mag)
        dv, da = valence * scale, arousal * scale
        self._llm_budget -= math.hypot(dv, da)
        self.impulse(now, dv, da)
        return dv, da

    # ---- outputs

    def nearest_label(self, valence: float, arousal: float) -> str:
        return min(
            self.cfg.labels.items(),
            key=lambda kv: math.hypot(kv[1].valence - valence, kv[1].arousal - arousal),
        )[0]

    def state(self, now: float) -> AffectState:
        if self._event_label and now < self._event_label[1]:
            label = self._event_label[0]
        else:
            self._event_label = None
            label = self.nearest_label(self.valence, self.arousal)
        dev = math.hypot(self.valence - self.baseline_valence, self.arousal - self.baseline_arousal)
        return AffectState(self.valence, self.arousal, label, min(1.0, dev / math.sqrt(2)))
