"""Built-in spoken lines (SPEC 8, 12): child-safe lines and the "can't think" fallback.

All text comes from the configuration (``safety.minor_lines``, ``conversation.fallback_lines``),
per language, with the conversation language as the fallback entry. They are spoken without
asking any route.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import TypeVar

from ..config.schema import MinorLines

T = TypeVar("T")


def pick(table: Mapping[str, T], language: str, default_language: str) -> T | None:
    """``table[language]``, else the default language's entry, else None."""
    return table.get(language) or table.get(default_language)


def minor_line(
    lines: Mapping[str, MinorLines],
    language: str,
    default_language: str,
    keywords: Collection[str],
) -> str:
    """The child-safe line for a turn no cleared route can answer: a greeting back for a
    greeting, the robot's name when it was called, otherwise "ask a grown-up". Empty when the
    configuration has no lines."""
    chosen = pick(lines, language, default_language)
    if chosen is None:
        return ""
    if "greeting" in keywords:
        return chosen.greeting or chosen.ask_grown_up
    if "name" in keywords:
        return chosen.robot_name or chosen.ask_grown_up
    return chosen.ask_grown_up
