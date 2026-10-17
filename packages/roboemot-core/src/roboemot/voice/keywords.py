"""Keyword spotting on ASR text (SPEC 8 stage 3, fast path).

Text and phrases are normalised the same way (lower case, accents removed, anything that is not
a Unicode letter or digit becomes a space) and a phrase matches only as whole words, so ``stop``
does not match ``stopped`` and ``hi`` is not found inside ``history``. There is no stemming: list
the inflections you want. Categories are ``stop``, ``no``, ``greeting``, ``name``
(``voice.wake_word`` or one of the language's ``name_aliases``) and ``wake`` (one of its
``wake_phrases``). It runs on every partial, so it is a few substring tests.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from ..config.schema import KeywordSet

NON_WORD = re.compile(r"[\W_]+")
CATEGORIES = ("stop", "no", "greeting", "name", "wake")


def normalise(text: str) -> str:
    """Lower case, accents stripped, punctuation and runs of spaces collapsed to one space."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return NON_WORD.sub(" ", plain).strip()


class KeywordSpotter:
    """Finds the keyword categories present in a text, for the text's language."""

    def __init__(self, sets: Mapping[str, KeywordSet], wake_word: str) -> None:
        self._wake = normalise(wake_word)
        self._phrases: dict[str, dict[str, list[str]]] = {}
        for language, ks in sets.items():
            lists = {
                "stop": ks.stop,
                "no": ks.no,
                "greeting": ks.greeting,
                "name": ks.name_aliases,
                "wake": ks.wake_phrases,
            }
            self._phrases[language] = {
                cat: [p for p in (normalise(x) for x in phrases) if p]
                for cat, phrases in lists.items()
            }

    def spot(self, text: str, language: str) -> set[str]:
        """The categories whose phrases occur as whole words in ``text``."""
        padded = f" {normalise(text)} "
        if padded == "  ":
            return set()
        found: set[str] = set()
        if self._wake and f" {self._wake} " in padded:
            found.add("name")
        for cat, phrases in self._phrases.get(language, {}).items():
            if any(f" {p} " in padded for p in phrases):
                found.add(cat)
        return found
