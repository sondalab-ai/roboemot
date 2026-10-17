"""Affect engine (SPEC 7, Affect state).

Continuous valence/arousal in [-1, 1] decaying toward a baseline (set by the circadian model,
SPEC 7A) with a configurable half-life; appraisal rules turn bus events into impulses with
cooldowns; LLM impulses are capped per conversational turn. The discrete label is the
appraisal label while it is held, otherwise the nearest label prototype.
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..config.schema import AffectConfig, AppraisalConfig, rule_identity

SUBJECT_KEYS = ("person_id", "entity_id", "track_id")  # payload fields that name who/what


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _matches(when: Mapping[str, Any], payload: Mapping[str, Any] | None) -> bool:
    """Whether every ``when`` field equals the payload's (an empty condition always matches)."""
    return all(k in (payload or {}) and (payload or {})[k] == v for k, v in when.items())


@dataclass
class AffectState:
    valence: float
    arousal: float
    label: str
    intensity: float


F = TypeVar("F", bound=Callable[..., Any])


def _locked(fn: F) -> F:
    """Run a public method under the engine's lock: the bus, the responder and the app tick
    all reach the one shared engine from different threads."""

    @functools.wraps(fn)
    def wrapper(self: AffectEngine, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


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
    _cooldown: dict[tuple, float] = field(default_factory=dict)  # (rule, subject) -> until
    _llm_budget: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self._llm_budget = self.cfg.llm_impulse_max

    # ---- time

    @_locked
    def update(self, now: float) -> AffectState:
        if not math.isfinite(now):
            raise ValueError("affect update time must be finite")
        if self._last is not None and now > self._last:
            k = math.pow(0.5, (now - self._last) / self.cfg.half_life_s)
            self.valence = self.baseline_valence + (self.valence - self.baseline_valence) * k
            self.arousal = self.baseline_arousal + (self.arousal - self.baseline_arousal) * k
        # Time never goes back: bus handlers pass capture stamps older than the loop tick.
        self._last = now if self._last is None else max(self._last, now)
        return self.state(self._last)

    @_locked
    def set_baseline(self, valence: float | None = None, arousal: float | None = None) -> None:
        if valence is not None:
            self.baseline_valence = _clamp(valence)
        if arousal is not None:
            self.baseline_arousal = _clamp(arousal)

    # ---- inputs

    @_locked
    def impulse(
        self, now: float, valence: float = 0.0, arousal: float = 0.0, label: str | None = None
    ) -> None:
        if not all(math.isfinite(x) for x in (now, valence, arousal)):
            raise ValueError("affect impulse must be finite")
        if label and label not in self.cfg.labels:
            raise KeyError(f"unknown affect label {label}")
        self.update(now)
        now = self._last  # a stale stamp counts as now
        self.valence = _clamp(self.valence + valence)
        self.arousal = _clamp(self.arousal + arousal)
        if label:
            self._event_label = (label, now + self.cfg.event_label_hold_s)

    @_locked
    def on_event(self, event: str, now: float, payload: Mapping[str, Any] | None = None) -> bool:
        """Apply matching appraisal rules; returns True if any rule fired.

        A rule with ``when`` matches only if every listed field equals the same field of
        ``payload`` (a missing field never matches). A cooldown is kept per rule and, when the
        payload names a subject (``person_id``, ``entity_id`` or ``track_id``), per subject:
        a second person reacts, the same one does not retrigger within the cooldown.
        """
        if not math.isfinite(now):
            raise ValueError("affect event time must be finite")
        if self._last is not None:
            now = max(now, self._last)  # a stale stamp must not shorten a label hold
        self._cooldown = {k: until for k, until in self._cooldown.items() if until > now}
        subject = next((payload[k] for k in SUBJECT_KEYS if payload and k in payload), None)
        fired = False
        for rule in self.appraisal.rules:
            if rule.event != event or not _matches(rule.when, payload):
                continue
            key = (rule_identity(rule.event, rule.when), subject)
            if now < self._cooldown.get(key, -math.inf):
                continue
            self.impulse(now, rule.valence, rule.arousal, rule.label)
            if rule.cooldown_s > 0:
                self._cooldown[key] = now + rule.cooldown_s
            fired = True
        return fired

    @_locked
    def set_config(self, cfg: AffectConfig, appraisal: AppraisalConfig) -> None:
        """Hot reload: new parameters and rules; the state and the cooldowns of rules that
        still exist are kept, those of removed rules are dropped."""
        self.cfg, self.appraisal = cfg, appraisal
        alive = {rule_identity(r.event, r.when) for r in appraisal.rules}
        self._cooldown = {k: v for k, v in self._cooldown.items() if k[0] in alive}

    @_locked
    def new_turn(self) -> None:
        self._llm_budget = self.cfg.llm_impulse_max

    @_locked
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

    @_locked
    def state(self, now: float) -> AffectState:
        if self._event_label and now < self._event_label[1]:
            label = self._event_label[0]
        else:
            self._event_label = None
            label = self.nearest_label(self.valence, self.arousal)
        dev = math.hypot(self.valence - self.baseline_valence, self.arousal - self.baseline_arousal)
        return AffectState(self.valence, self.arousal, label, min(1.0, dev / math.sqrt(2)))
