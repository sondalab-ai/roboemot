"""Speaker profile and scene summaries for the system prompt (SPEC 8, Conversation state; 12).

Plain text only: a registered speaker's display name and the preferences they confirmed, and the
entities the robot sees, by attention priority. No embeddings, ids, ages, genders or images. A
person is named only when the registry knows them, otherwise they are "a person": age and gender
estimates are never offered to the LLM as facts (SPEC 12), so there is no "a child" here either;
the minor content rules in the safety block cover that case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..world import Entry
from ..world.types import UNKNOWN_IDENTITY

KIND_ORDER = ("person", "animal", "object")
SCALARS = (str, int, float, bool)
SEPARATOR = ", "
VOWELS = "aeiou"


def one_line(text: str, max_chars: int) -> str:
    """``text`` on one line and at most ``max_chars`` long: a stored name or preference cannot
    add lines (rules) to the system prompt or flood it."""
    return " ".join(text.split())[:max_chars]


def profile_summary(
    name: str | None, preferences: Mapping[str, Any], max_chars: int, value_max_chars: int
) -> str | None:
    """The speaker's display name and confirmed preferences, cut to ``max_chars``; None when
    there is nothing to say. Only plain values are listed (strings, numbers, booleans)."""
    parts = [one_line(name, value_max_chars)] if name else []
    prefs = [
        f"{one_line(str(k), value_max_chars)}={one_line(str(v), value_max_chars)}"
        for k, v in sorted(preferences.items())
        if isinstance(v, SCALARS)
    ]
    if prefs:
        parts.append("preferences " + SEPARATOR.join(prefs))
    text = "; ".join(p for p in parts if p)[:max_chars]
    return text or None


def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in VOWELS else "a"


def label(entry: Entry, name_of: Callable[[Entry], str | None], value_max_chars: int) -> str:
    """How the LLM hears about ``entry``: its registry name when known, else its kind."""
    known = entry.identity not in ("", UNKNOWN_IDENTITY)
    name = name_of(entry) if known else None
    if entry.kind == "person":
        return one_line(name, value_max_chars) if name else "a person"
    noun = one_line(entry.category, value_max_chars) or entry.kind
    if name:
        return f"{one_line(name, value_max_chars)} ({_article(noun)} {noun})"
    return f"{_article(noun)} {noun}"


def scene_summary(
    entries: Iterable[Entry],
    *,
    now: float,
    stale_s: float,
    target_id: str | None,
    name_of: Callable[[Entry], str | None],
    max_chars: int,
    value_max_chars: int,
) -> str | None:
    """Entities in view by attention priority until ``max_chars``: the attention target first,
    then people, animals and objects, the more confident first. An entity that would cross the
    cap is left out whole (the summary is never cut mid-name). None when nothing is in view."""
    fresh = [e for e in entries if now - e.last_seen <= stale_s]

    def rank(e: Entry) -> tuple[bool, int, float]:
        return (e.entry_id != target_id, KIND_ORDER.index(e.kind), -e.confidence)

    out = ""
    for entry in sorted(fresh, key=rank):
        piece = label(entry, name_of, value_max_chars)
        joined = piece if not out else out + SEPARATOR + piece
        if len(joined) > max_chars:
            break
        out = joined
    return out or None


__all__ = ["label", "profile_summary", "scene_summary"]
