"""Built-in spoken lines (SPEC 8): the "can't think" fallback.

All text comes from the configuration (``conversation.fallback_lines``), per language, with the
conversation language as the fallback entry. They are spoken without asking any route.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")


def pick(table: Mapping[str, T], language: str, default_language: str) -> T | None:
    """``table[language]``, else the default language's entry, else None."""
    return table.get(language) or table.get(default_language)
