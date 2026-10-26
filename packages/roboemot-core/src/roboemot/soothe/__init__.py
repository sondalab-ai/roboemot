"""Self-soothing idle sounds (SPEC 7B)."""

from __future__ import annotations

from .runner import EVENT, SootheRunner
from .synth import BlirpEvent, BlirpPlan, HumNote, HumPlan, plan_blirp, plan_hum, synthesize

__all__ = [
    "EVENT",
    "SootheRunner",
    "BlirpEvent",
    "BlirpPlan",
    "HumNote",
    "HumPlan",
    "plan_blirp",
    "plan_hum",
    "synthesize",
]
