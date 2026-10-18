"""Spoken confirmation of a preference (SPEC 8, Conversation state; 6, 12).

``remember_preference`` never writes. It asks out loud ("Should I remember this: ...?") and
leaves a :class:`Pending` request; only the very next utterance, from the same person, within
``conversation.confirm_timeout_s``, can confirm it, by a ``yes`` phrase and no ``no`` phrase
(``conversation.confirm_yes`` / ``confirm_no``). Anything else, from anybody, drops it. The key
and value come from the model, and later land in prompts as data, so they are sanitised and
length-capped here before the question is even asked. Nothing in this module logs.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..voice.keywords import normalise

Reply = Literal["yes", "no", "other"]
KEY_FORBIDDEN = re.compile(r"[^a-z0-9 _-]")
VALUE_FORBIDDEN = re.compile(r"[\"'`<>{}\[\]\\|=;]")
SPACES = re.compile(r"\s+")


class PreferenceStore(Protocol):
    """Where confirmed preferences live (the registry, SPEC 6). ``allowed`` says whether a
    preference called ``key`` could be stored for this person now (an enrolled, active person
    with room); ``store`` writes it and returns whether it did."""

    def allowed(self, person_id: str, key: str) -> bool: ...
    def store(self, person_id: str, key: str, value: str) -> bool: ...


@dataclass
class Pending:
    """A question asked and not yet answered. The text is the person's data: hidden from repr."""

    person_id: str
    key: str = field(repr=False)
    value: str = field(repr=False)
    expires: float


def _clean(text: str) -> str:
    text = SPACES.sub(" ", unicodedata.normalize("NFKC", text))  # newlines and tabs: spaces
    text = "".join(c for c in text if c.isprintable() and unicodedata.category(c) != "Cf")
    return text.strip()


def sanitise_preference(
    key: Any, value: Any, key_max: int, value_max: int
) -> tuple[str, str] | None:
    """The key (lower case, letters, digits, space, ``_`` and ``-``) and the value (printable,
    one line, without quotes, brackets or the characters a prompt template could use), or
    None when either is not a string, empty after cleaning, or over its cap."""
    if not isinstance(key, str) or not isinstance(value, str):
        return None
    k = KEY_FORBIDDEN.sub("", _clean(key).casefold())
    k = SPACES.sub(" ", k).strip()
    v = SPACES.sub(" ", VALUE_FORBIDDEN.sub(" ", _clean(value))).strip()
    if not k or not v or len(k) > key_max or len(v) > value_max:
        return None
    return k, v


def reply_kind(
    text: str,
    language: str,
    yes: Mapping[str, Collection[str]],
    no: Mapping[str, Collection[str]],
) -> Reply:
    """``yes`` if the text holds a yes phrase and no no phrase (whole words), ``no`` if it
    holds a no phrase, else ``other``. Phrases are those of ``language``."""
    padded = f" {normalise(text)} "

    def has(table: Mapping[str, Collection[str]]) -> bool:
        return any(f" {p} " in padded for p in (normalise(x) for x in table.get(language, ())) if p)

    if has(no):
        return "no"
    return "yes" if has(yes) else "other"
