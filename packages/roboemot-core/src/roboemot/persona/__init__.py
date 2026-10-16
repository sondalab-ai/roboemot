"""Persona traits turned into runtime effects (SPEC 11A, Persona).

Traits never write configuration: they produce multipliers (``_x``), offsets (``_add``) and
whole steps on enumerated parameters (``_steps``), applied at the same stage as the register
and the circadian state. Consumers clamp the results to their own ranges and limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import get_args

from ..config.schema import PersonaConfig, RegisterParams

REACTION_LEVELS: tuple[str, ...] = get_args(
    RegisterParams.model_fields["reaction_intensity"].annotation
)
VOCABULARY_LEVELS: tuple[str, ...] = get_args(RegisterParams.model_fields["vocabulary"].annotation)


@dataclass(frozen=True)
class PersonaEffects:
    motion_amplitude_x: float = 1.0
    idle_period_x: float = 1.0
    voice_rate_x: float = 1.0
    voice_pitch_st_add: float = 0.0
    expressivity_x: float = 1.0
    reaction_steps: int = 0
    voice_variability_add: float = 0.0
    vocabulary_steps: int = 0


def interpolate(curve: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear value of ``curve`` at ``x``; flat outside the first and last points."""
    if x <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1][1]


def _steps(value: float) -> int:
    """Nearest whole step, halves away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def effects(persona: PersonaConfig) -> PersonaEffects:
    """Effects of the persona's traits through its tables."""
    t, e = persona.traits, persona.trait_effects
    return PersonaEffects(
        motion_amplitude_x=interpolate(e.energy.motion_amplitude_x, t.energy),
        idle_period_x=interpolate(e.energy.idle_period_x, t.energy),
        voice_rate_x=interpolate(e.energy.voice_rate_x, t.energy),
        voice_pitch_st_add=interpolate(e.warmth.voice_pitch_st_add, t.warmth),
        expressivity_x=interpolate(e.warmth.expressivity_x, t.warmth),
        reaction_steps=_steps(interpolate(e.playfulness.reaction_steps, t.playfulness)),
        voice_variability_add=interpolate(e.playfulness.voice_variability_add, t.playfulness),
        vocabulary_steps=_steps(interpolate(e.formality.vocabulary_steps, t.formality)),
    )


def _step(levels: tuple[str, ...], current: str, steps: int) -> str:
    i = levels.index(current) + steps
    return levels[max(0, min(len(levels) - 1, i))]


def _clamp_field(model: type, name: str, value: float) -> float:
    lo, hi = -math.inf, math.inf
    for c in model.model_fields[name].metadata:
        lo = getattr(c, "ge", None) if getattr(c, "ge", None) is not None else lo
        hi = getattr(c, "le", None) if getattr(c, "le", None) is not None else hi
    return max(lo, min(hi, value))


def apply_to_register(register: RegisterParams, fx: PersonaEffects) -> RegisterParams:
    """The register as the persona shapes it: motion amplitude, reaction intensity and
    vocabulary level; each kept inside its schema range."""
    return register.model_copy(
        update={
            "motion_amplitude": _clamp_field(
                RegisterParams,
                "motion_amplitude",
                register.motion_amplitude * fx.motion_amplitude_x,
            ),
            "reaction_intensity": _step(
                REACTION_LEVELS, register.reaction_intensity, fx.reaction_steps
            ),
            "vocabulary": _step(VOCABULARY_LEVELS, register.vocabulary, fx.vocabulary_steps),
        }
    )


__all__ = ["PersonaEffects", "apply_to_register", "effects", "interpolate"]
