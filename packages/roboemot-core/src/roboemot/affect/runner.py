"""Runs the one shared affect engine (SPEC 7, 11).

The runner owns the :class:`~roboemot.affect.AffectEngine` the whole app uses (bus appraisal,
connectivity responder, later the LLM tool and circadian model), decays it at
``affect.update_hz`` from the caller's tick and publishes each update on ``affect.state``.
There is no thread of its own: the app calls :meth:`tick` from its loop, so the publish rate
is ``update_hz`` capped by how often that loop runs. Decay is time-based, so a late tick
loses nothing.
"""

from __future__ import annotations

import math
from typing import Any

from ..bus import Bus
from ..config.schema import AffectConfig, AppraisalConfig, StateEffects
from . import AffectEngine, AffectState
from .appraisal import AppraisalAdapter

TOPIC = "affect.state"


class AffectRunner:
    """Owns the shared engine, ticks it and publishes ``affect.state``."""

    def __init__(
        self, cfg: AffectConfig, appraisal: AppraisalConfig, bus: Bus, now: float = 0.0
    ) -> None:
        self.bus = bus
        self.engine = AffectEngine(cfg, appraisal)
        self.adapter = AppraisalAdapter(bus, appraisal, self.on_event, start=now)
        self._due = -math.inf

    def on_event(self, event: str, now: float, payload: dict[str, Any] | None = None) -> None:
        """Run an appraisal event through the engine's rules."""
        self.engine.on_event(event, now, payload)

    def tick(self, now: float) -> AffectState | None:
        """Advance the time-based appraisal events and, when an update is due, the engine.

        Publishes and returns the new state, or returns None between updates.
        """
        self.adapter.tick(now)
        if now < self._due:
            return None
        period = 1.0 / self.engine.cfg.update_hz
        # Stay on the update grid while keeping up; after a stall restart it from now.
        self._due = (self._due if now - self._due < period else now) + period
        state = self.engine.update(now)
        self.bus.publish(TOPIC, state, now)
        return state

    def set_circadian(self, effects: StateEffects) -> None:
        """Move the arousal baseline to the circadian state's (SPEC 7A); the circadian runner
        calls this on a state change and the state then decays toward it."""
        self.engine.set_baseline(arousal=effects.arousal_baseline)

    def apply_config(self, affect: AffectConfig, appraisal: AppraisalConfig) -> None:
        """Hot reload: new rules, label regions and parameters; the state is kept."""
        self.engine.cfg = affect
        self.engine.appraisal = appraisal
        self.adapter.cfg = appraisal

    def close(self) -> None:
        """Detach from the bus."""
        self.adapter.close()
