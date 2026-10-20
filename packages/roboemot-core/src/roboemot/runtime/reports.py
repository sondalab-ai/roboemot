"""Fields of the debug report that the runtime owns (SPEC 12A): numbers and labels only."""

from __future__ import annotations

from typing import Any

from roboemot.affect.register import RegisterTracker


def affect_report(engine: Any, now: float) -> dict[str, Any]:
    """Affect fields of the debug report: the numbers and the label, nothing about who or why."""
    state = engine.state(now)
    return {
        "valence": state.valence,
        "arousal": state.arousal,
        "label": state.label,
    }


def register_report(tracker: RegisterTracker) -> dict[str, Any]:
    """Register fields of the debug report: the group and its parameters, no person ids."""
    return tracker.active.summary() if tracker.active is not None else {}


def register_overrides(registry: Any, person_id: str) -> dict[str, Any]:
    """A registered person's ``register_overrides`` ({} without a registry). A missing record or
    a registry error raises: the tracker then resolves that person as ``unknown``."""
    if registry is None:
        return {}
    return dict(registry.get_person(person_id).register_overrides)
