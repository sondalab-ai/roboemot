"""Keyword spotting on ASR text (SPEC 8 stage 3, fast path).

Text and phrases are normalised the same way (lower case, accents removed, anything that is not
a Unicode letter or digit becomes a space) and a phrase matches only as whole words, so ``stop``
does not match ``stopped`` and ``hi`` is not found inside ``history``. There is no stemming: list
the inflections you want. Categories are ``stop``, ``yes``, ``no``, ``greeting``, ``name``
(the robot's name, see ``robot_name``, or one of the language's ``name_aliases``) and ``wake``
(one of its ``wake_phrases``, or a ``name_wake_templates`` phrase rendered with the name). It
runs on every partial, so it is a few substring tests.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..config.schema import KeywordSet

if TYPE_CHECKING:
    from .robot_name import ResolvedName, RobotName

NON_WORD = re.compile(r"[\W_]+")
CATEGORIES = ("stop", "yes", "no", "greeting", "name", "wake")


def normalise(text: str) -> str:
    """Lower case, accents stripped, punctuation and runs of spaces collapsed to one space."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return NON_WORD.sub(" ", plain).strip()


class KeywordSpotter:
    """Finds the keyword categories present in a text, for the text's language."""

    def __init__(
        self, sets: Mapping[str, KeywordSet], wake_word: str | None, names: RobotName | None = None
    ) -> None:
        from .robot_name import RobotName

        self._sets = dict(sets)
        self._configured = wake_word
        self._names = names if names is not None else RobotName()
        self._resolved: ResolvedName | None = None
        self._phrases: dict[str, dict[str, list[str]]] = {}
        self._oov: dict[str, int] = {}
        self._rebuild()

    def _rebuild(self) -> None:
        """Phrase lists for the current name; rebuilt when the name or its checked languages
        change (a model finished loading, the platform name arrived)."""
        resolved = self._names.resolve(self._configured, tuple(self._sets))
        if resolved == self._resolved:
            return
        self._resolved = resolved
        self._phrases = {}
        self._oov = {}
        aliases = resolved.aliases_apply()
        for language, ks in self._sets.items():
            wake = list(ks.wake_phrases)
            if resolved.usable(language):
                wake += [t.replace("{name}", resolved.matching) for t in ks.name_wake_templates]
            lists = {
                "stop": ks.stop,
                "yes": ks.yes,
                "no": ks.no,
                "greeting": ks.greeting,
                "name": ks.name_aliases if aliases else [],
                "wake": wake,
            }
            self._phrases[language] = {
                cat: [p for p in (normalise(x) for x in phrases) if p]
                for cat, phrases in lists.items()
            }
            bad = self._names.out_of_vocab(language, self._phrases[language]["wake"])
            if bad is not None:
                self._oov[language] = bad

    def wake_out_of_vocab(self) -> dict[str, int]:
        """Per language whose recognizer vocabulary is known, how many wake phrases (generic
        and rendered with the name) can never be heard (SPEC 8, 12A: numbers only)."""
        return dict(self._oov)

    def spot(self, text: str, language: str) -> set[str]:
        """The categories whose phrases occur as whole words in ``text``."""
        padded = f" {normalise(text)} "
        if padded == "  ":
            return set()
        self._rebuild()
        found: set[str] = set()
        name = self._resolved
        if name is not None and name.usable(language) and f" {name.matching} " in padded:
            found.add("name")
        for cat, phrases in self._phrases.get(language, {}).items():
            if any(f" {p} " in padded for p in phrases):
                found.add(cat)
        return found
