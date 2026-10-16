"""Aggregation of per-frame age and gender estimates over a track (SPEC 5, Rules).

Estimates only ever make behaviour more conservative: a declared value always wins, anything
below the confidence threshold or with too few frames is ``unknown``, and an unknown age group
counts as a possible minor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..config.schema import IdentityConfig
from .records import AGE_GROUPS, possible_minor

UNKNOWN = "unknown"
ORDERED_AGE_GROUPS = tuple(g for g in AGE_GROUPS if g != UNKNOWN)


@dataclass(frozen=True)
class Estimate:
    """An aggregated attribute value, where it came from and how sure the estimate is."""

    value: str
    source: Literal["declared", "estimated", "none"]
    confidence: float = 0.0

    @property
    def possible_minor(self) -> bool:
        return possible_minor(self.value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _settle(value: str, confidences: Sequence[float], config: IdentityConfig) -> Estimate:
    confidence = _mean(confidences)
    if confidence < config.estimate_confidence_min:
        return Estimate(UNKNOWN, "none", confidence)
    return Estimate(value, "estimated", confidence)


def estimate_age_group(
    samples: Sequence[tuple[str, float]], config: IdentityConfig, declared: str | None = None
) -> Estimate:
    """Median age group of ``(group, confidence)`` frames; declared values override.

    The median is taken on the ordered groups (child < teen < adult < senior); with an even
    number of frames the younger middle one is used, the conservative choice.
    """
    if declared is not None:
        return Estimate(declared, "declared", 1.0)
    frames = [(g, c) for g, c in samples if g in ORDERED_AGE_GROUPS]
    if len(frames) < config.estimate_frames_min:
        return Estimate(UNKNOWN, "none")
    ranks = sorted(ORDERED_AGE_GROUPS.index(g) for g, _ in frames)
    value = ORDERED_AGE_GROUPS[ranks[(len(ranks) - 1) // 2]]
    return _settle(value, [c for g, c in frames if g == value], config)


def estimate_gender(
    samples: Sequence[tuple[str, float]], config: IdentityConfig, declared: str | None = None
) -> Estimate:
    """Most frequent gender of ``(value, confidence)`` frames (no order exists, so no median).

    A tie, too few frames or a low mean confidence of the winning frames gives ``unknown``.
    """
    if declared is not None:
        return Estimate(declared, "declared", 1.0)
    frames = [(g, c) for g, c in samples if g != UNKNOWN]
    if len(frames) < config.estimate_frames_min:
        return Estimate(UNKNOWN, "none")
    ranked = Counter(g for g, _ in frames).most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return Estimate(UNKNOWN, "none")
    value = ranked[0][0]
    return _settle(value, [c for g, c in frames if g == value], config)
