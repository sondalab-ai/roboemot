"""Keyword spotting on ASR text (SPEC 8 stage 3, fast path).

Text and phrases are normalised the same way (lower case, accents removed, anything that is not
a Unicode letter or digit becomes a space) and a phrase matches only as whole words, so ``stop``
does not match ``stopped`` and ``hi`` is not found inside ``history``. There is no stemming: list
the inflections you want. Categories are ``stop``, ``yes``, ``no``, ``greeting``, ``name``
(the robot's name, see ``robot_name``, or one of the language's ``name_aliases``) and ``wake``
(one of its ``wake_phrases``, or a ``name_wake_templates`` phrase rendered with the name). It
runs on every partial, so it is a few substring tests.

``grammar_phrases`` gives the same lists *folded* instead (lower case, accents kept): the form a
recognizer vocabulary spells them in, for the grammar pass (``roboemot.voice.grammar``).
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
FOLD_SPLIT = re.compile(r"[^\w']+|_")
CATEGORIES = ("stop", "yes", "no", "greeting", "name", "wake")


def normalise(text: str) -> str:
    """Lower case, accents stripped, punctuation and runs of spaces collapsed to one space."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in folded if not unicodedata.combining(c))
    return NON_WORD.sub(" ", plain).strip()


def fold(text: str) -> str:
    """Lower case with accents kept (NFC), punctuation other than apostrophes collapsed to one
    space: how a Vosk vocabulary spells words (``così``, ``c'è``)."""
    folded = unicodedata.normalize("NFC", text.casefold())
    return " ".join(FOLD_SPLIT.sub(" ", folded).split())


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
        self._folded: dict[str, dict[str, tuple[str, ...]]] = {}
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
        folded: dict[str, dict[str, tuple[str, ...]]] = {}
        oov: dict[str, int] = {}  # assigned once below: a report never reads a partial dict
        aliases = resolved.aliases_apply()
        spoken = fold(resolved.display)
        for language, ks in self._sets.items():
            usable = resolved.usable(language)
            lists = {
                "stop": ks.stop,
                "yes": ks.yes,
                "no": ks.no,
                "greeting": ks.greeting,
                "name": ks.name_aliases if aliases else [],
                "wake": list(ks.wake_phrases),
            }
            self._phrases[language] = {
                cat: [p for p in (normalise(x) for x in phrases) if p]
                for cat, phrases in lists.items()
            }
            templates = ks.name_wake_templates if usable else []
            rendered = (t.replace("{name}", resolved.matching) for t in templates)
            self._phrases[language]["wake"] += [p for p in map(normalise, rendered) if p]
            folded[language] = {
                cat: tuple(p for p in (fold(x) for x in phrases) if p)
                for cat, phrases in lists.items()
            }
            folded[language]["wake"] += tuple(fold(t.replace("{name}", spoken)) for t in templates)
            if usable and spoken:
                folded[language]["name"] += (spoken,)
            bad = self._names.out_of_vocab(language, self._phrases[language]["wake"])
            if bad is not None:
                oov[language] = bad
        self._oov = oov
        self._folded = folded

    def grammar_phrases(self, language: str) -> dict[str, tuple[str, ...]]:
        """Per category, the phrases of ``language`` folded (accents kept) for a recognizer
        grammar, the bare robot name under ``name`` when it is recognisable; current for the
        robot's name (rebuilt here when it changed)."""
        self._rebuild()
        return dict(self._folded.get(language, {}))

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
