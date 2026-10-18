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
WORD = re.compile(r"\w+")


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
    job: int  # the speech job of the question


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


def _words(text: str) -> list[str]:
    """Lower-cased words, accents kept (``sì`` is not ``si``)."""
    return WORD.findall(unicodedata.normalize("NFC", text).casefold())


def _consume(words: list[str], phrases: Collection[list[str]]) -> int:
    """How many leading words ``phrases`` (token lists, longest first) cover at position 0."""
    for phrase in phrases:
        if words[: len(phrase)] == phrase:
            return len(phrase)
    return 0


def reply_kind(
    text: str,
    language: str,
    yes: Mapping[str, Collection[str]],
    no: Mapping[str, Collection[str]],
    fillers: Mapping[str, Collection[str]],
    max_words: int,
) -> Reply:
    """``no`` if the reply holds a no phrase (whole words, accents ignored); ``yes`` only if it
    is at most ``max_words`` long, starts with a yes phrase, and every other word belongs to a
    yes phrase or a filler (``sì grazie``): "Come si dice cane?", "si chiama Luca" and "Ok
    Google" are not confirmations. Else ``other``. Phrases are those of ``language``."""
    padded = f" {normalise(text)} "
    if any(f" {p} " in padded for p in (normalise(x) for x in no.get(language, ())) if p):
        return "no"
    words = _words(text)
    if not 0 < len(words) <= max_words:
        return "other"
    yes_phrases = sorted(
        (w for w in (_words(x) for x in yes.get(language, ())) if w), key=len, reverse=True
    )
    filler_phrases = sorted(
        (w for w in (_words(x) for x in fillers.get(language, ())) if w), key=len, reverse=True
    )
    first = _consume(words, yes_phrases)
    if not first:
        return "other"
    rest = words[first:]
    while rest:
        n = _consume(rest, [*yes_phrases, *filler_phrases])
        if not n:
            return "other"
        rest = rest[n:]
    return "yes"
