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


def line_text(
    lines: Mapping[str, MinorLines], language: str, default_language: str, field: str
) -> str:
    """One field of the language's lines; an empty or missing one falls back, field by field,
    to the default language's, then to any language's, so a child never gets silence."""
    order = [language, default_language, *lines]
    for name in order:
        entry = lines.get(name)
        text = getattr(entry, field, "") if entry is not None else ""
        if text.strip():
            return text
    return ""


def minor_line(
    lines: Mapping[str, MinorLines],
    language: str,
    default_language: str,
    keywords: Collection[str],
    name: str = "",
) -> str:
    """The child-safe line for a minor's turn no route can answer: a greeting back for a
    greeting, the robot's name when it was called (``name``, already sanitised, fills the line's
    ``{name}``; without one the line's nameless variant), otherwise "ask a grown-up"."""
    if "greeting" in keywords:
        field = "greeting"
    elif "name" in keywords:
        field = "robot_name" if name else "robot_unnamed"
    else:
        field = "ask_grown_up"
    text = line_text(lines, language, default_language, field) or line_text(
        lines, language, default_language, "ask_grown_up"
    )
    return text.replace("{name}", name) if name else text
