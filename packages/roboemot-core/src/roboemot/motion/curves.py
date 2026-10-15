"""Interpolation curves (SPEC 4A, 10). ``s`` is normalised time in [0, 1].

The same curves exist in ``rt/src/curves.cpp``; golden vectors keep them in sync.
"""

from __future__ import annotations

from collections.abc import Callable

# Overshoot of the "cartoon" curve (back-ease constant, Penner's standard value).
CARTOON_OVERSHOOT = 1.70158


def _clamp01(s: float) -> float:
    return 0.0 if s <= 0.0 else 1.0 if s >= 1.0 else s


def linear(s: float) -> float:
    return _clamp01(s)


def min_jerk(s: float) -> float:
    s = _clamp01(s)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def ease_in_out(s: float) -> float:
    s = _clamp01(s)
    return 3.0 * s * s - 2.0 * s * s * s


def cartoon(s: float) -> float:
    """Ease-out with overshoot: goes past the target and settles back (back-ease-out)."""
    s = _clamp01(s) - 1.0
    c = CARTOON_OVERSHOOT
    return 1.0 + (c + 1.0) * s * s * s + c * s * s


CURVES: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "minjerk": min_jerk,
    "ease_in_out": ease_in_out,
    "cartoon": cartoon,
}
