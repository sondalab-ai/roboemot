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
from .records import AGE_GROUPS, GENDERS, check_age_group, check_gender, possible_minor

UNKNOWN = "unknown"
# Youngest-first with unknown below child: the median can only lean conservative.
ORDERED_AGE_GROUPS = (UNKNOWN, *(g for g in AGE_GROUPS if g != UNKNOWN))


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


def _settle(value: str, confidences: Sequence[float], confidence_min: float) -> Estimate:
    confidence = _mean(confidences)
    if confidence < confidence_min:
        return Estimate(UNKNOWN, "none", confidence)
    return Estimate(value, "estimated", confidence)


def _vocabulary(samples: Sequence[tuple[str, float]], allowed: tuple[str, ...]) -> None:
    for value, _ in samples:
        if value not in allowed:
            raise ValueError(f"unexpected estimate value {value!r}")


def estimate_age_group(
    samples: Sequence[tuple[str, float]],
    config: IdentityConfig,
    declared: str | None = None,
    *,
    confidence_min: float | None = None,
) -> Estimate:
    """Median age group of ``(group, confidence)`` frames; declared values override.

    Order is unknown < child < teen < adult < senior, so ``unknown`` frames pull the median
    down: mostly-unknown frames give unknown. With an even number of frames the lower middle
    one is used, the conservative choice. Fewer than ``estimate_frames_min`` frames, or a mean
    confidence of the winning frames below ``confidence_min``, give unknown. ``confidence_min``
    defaults to ``identity.estimate_confidence_min`` (the 0.8 rule); an estimator that already
    turns weak crops into ``unknown`` with its own score floor passes that floor instead.
    """
    if declared is not None:
        return Estimate(check_age_group(declared), "declared", 1.0)
    _vocabulary(samples, AGE_GROUPS)
    if len(samples) < config.estimate_frames_min:
        return Estimate(UNKNOWN, "none")
    ranks = sorted(ORDERED_AGE_GROUPS.index(g) for g, _ in samples)
    value = ORDERED_AGE_GROUPS[ranks[(len(ranks) - 1) // 2]]
    if value == UNKNOWN:
        return Estimate(UNKNOWN, "none")
    floor = config.estimate_confidence_min if confidence_min is None else confidence_min
    return _settle(value, [c for g, c in samples if g == value], floor)


def estimate_gender(
    samples: Sequence[tuple[str, float]], config: IdentityConfig, declared: str | None = None
) -> Estimate:
    """Most frequent gender of ``(value, confidence)`` frames (no order exists, so no median).

    ``unknown`` frames count as a value: a tie, an unknown winner, too few frames or a low
    mean confidence of the winning frames gives ``unknown``.
    """
    if declared is not None:
        return Estimate(check_gender(declared), "declared", 1.0)
    _vocabulary(samples, GENDERS)
    if len(samples) < config.estimate_frames_min:
        return Estimate(UNKNOWN, "none")
    ranked = Counter(g for g, _ in samples).most_common(2)
    if (len(ranked) > 1 and ranked[0][1] == ranked[1][1]) or ranked[0][0] == UNKNOWN:
        return Estimate(UNKNOWN, "none")
    value = ranked[0][0]
    return _settle(value, [c for g, c in samples if g == value], config.estimate_confidence_min)
