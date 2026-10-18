"""The registry as the store of confirmed preferences (SPEC 6)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class RegistryPreferences:
    """``PreferenceStore`` over a registry; ``max_items`` is read on each call (hot reload)."""

    def __init__(self, registry: Any, max_items: Callable[[], int]) -> None:
        self._registry, self._max_items = registry, max_items

    def allowed(self, person_id: str, key: str) -> bool:
        try:
            record = self._registry.get_person(person_id)
        except Exception:
            return False
        return record.status == "active" and (
            key in record.preferences or len(record.preferences) < self._max_items()
        )

    def store(self, person_id: str, key: str, value: str) -> bool:
        try:
            self._registry.set_preference(person_id, key, value, max_items=self._max_items())
        except Exception as exc:  # forgotten meanwhile, not active, full, store failure
            log.warning("preferences: not stored (%s)", type(exc).__name__)
            return False
        return True
