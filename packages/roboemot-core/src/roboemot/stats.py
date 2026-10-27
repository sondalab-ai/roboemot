"""Small order statistics for the debug report: numbers in, numbers out (SPEC 12A)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

P10 = 0.1
P50 = 0.5
P90 = 0.9
P95 = 0.95
P99 = 0.99
# The report's usual summary: median and 90th percentile (plus count and max).
SUMMARY_LEVELS = {"p50": P50, "p90": P90}


def quantile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank quantile ``q`` (0 to 1) of an ascending, non-empty sequence."""
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarise(
    values: Sequence[float], levels: Mapping[str, float], *, with_min: bool = False
) -> dict[str, Any]:
    """``{"count": n, <name>: quantile, ..., "max": m}`` (plus ``"min"``), or ``{"count": 0}``.

    ``levels`` maps a key to its quantile. Values come back as the plain floats or ints they
    were given, so the result serialises as JSON.
    """
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    out: dict[str, Any] = {"count": len(ordered)}
    if with_min:
        out["min"] = ordered[0]
    out.update({name: quantile(ordered, q) for name, q in levels.items()})
    out["max"] = ordered[-1]
    return out
