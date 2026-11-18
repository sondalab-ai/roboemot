"""Keyword spotting on ASR text (SPEC 8 stage 3, fast path).

Text and phrases are normalised the same way (lower case, accents removed, anything that is not
a Unicode letter or digit becomes a space) and a phrase matches only as whole words, so ``stop``
does not match ``stopped`` and ``hi`` is not found inside ``history``. There is no stemming: list
the inflections you want. Categories are ``stop``, ``yes``, ``no``, ``greeting``, ``name``
(the robot's name, see ``robot_name``, or one of the language's ``name_aliases``),
``extra_name`` (one of ``voice.extra_names``: kept apart because over the robot's radio it does
not count, SPEC 8) and ``wake`` (one of its ``wake_phrases``, or a ``name_wake_templates`` phrase
rendered with the name or an extra name; a rendered phrase that repeats a wake phrase, such as
``ciao robot``, is listed once). It runs on every partial, so it is a few substring tests.

``name_first`` tells whether a text *opens* with a name (the robot's own, an extra name, an
alias), after at most one
lead-in (a ``name_lead_ins`` phrase or a ``greeting`` of the language): the name as the first
word wakes the robot at any sentence length (SPEC 8, owner decision 2026-11-04), while the name
elsewhere wakes only a short final. The grammar pass reports the same position in what it heard
with the ``NAME_FIRST`` flag (never a category: the fast path takes it out).

``name_close`` is for a server's transcript only (SPEC 8, Radio playback: the server must confirm a
wake over the robot's radio), never for the local recognizer. It tries each name (the robot's own
and the extra names): a word is close to a word of the name when its spelling, with ``i`` read as
``e`` and ``u`` as ``o`` (the vowels a recognizer swaps), differs only by inserted or dropped
vowels, at most ``max_edits`` of them (a swapped vowel class counts two: one dropped, one inserted).
The consonants must match in order, so "Nivea" and "Nieve" are close to "Neve" while "nave", "nove",
"deve", "nei" are not. A name word shorter than ``min_letters`` must match exactly; a name of
several words matches word by word, all of them, in order.

The spelling rules depend on the transcript's language (``SPELLING``, data, not code); a language
without its own rules uses the default ones above. English (owner decision 2026-11-17: a server
writes "Navy" or "Neevee" for "Neve") also reads ``y`` after the first letter as a vowel, drops a
final ``h`` after a vowel, reads ``a`` as ``e`` too, and collapses doubled letters, on both the
word and the name. So "Navy", "Neva", "Nevay", "Neevee", "Nevvy", "Neveh" and "Nev" are close to
"Neve", while "never", "knave", "eve", "even", "nerve", "novel", "have", "give", "live", "love",
"move" and "wave" are not. Known false accepts in English: "nav" and "nave" (and "navy" itself,
which the owner wants accepted).

``grammar_phrases`` gives the same lists *folded* instead (lower case, accents kept): the form a
recognizer vocabulary spells them in, for the grammar pass (``roboemot.voice.grammar``).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config.schema import KeywordSet

if TYPE_CHECKING:
    from .robot_name import ResolvedName, RobotName, SpokenName

NON_WORD = re.compile(r"[\W_]+")
FOLD_SPLIT = re.compile(r"[^\w']+|_")
EXTRA_NAME = "extra_name"  # one of ``voice.extra_names`` (the robot's own name is ``name``)
CATEGORIES = ("stop", "yes", "no", "greeting", "name", EXTRA_NAME, "wake")


@dataclass(frozen=True)
class Spelling:
    """How a language's transcripts spell a name (facts of the spelling, not tunables):
    ``before`` regex rewrites, then ``classes`` (vowels a recognizer swaps for one another,
    mapped to one class), then ``after`` rewrites; ``vowels`` are the classes left."""

    before: tuple[tuple[str, str], ...]
    classes: dict[int, str]
    after: tuple[tuple[str, str], ...]
    vowels: frozenset[str]

    def apply(self, word: str) -> str:
        """``word`` (normalised) as these rules read it, ready for the vowel-edit count."""
        for pattern, repl in self.before:
            word = re.sub(pattern, repl, word)
        word = word.translate(self.classes)
        for pattern, repl in self.after:
            word = re.sub(pattern, repl, word)
        return word


# ``name_close``'s spelling rules per transcript language, and the default (Italian and any
# language without its own): spelling facts, not tunables
DEFAULT_SPELLING = Spelling((), str.maketrans({"i": "e", "u": "o"}), (), frozenset("aeo"))
SPELLING = {
    "en": Spelling(
        before=((r"(?<=.)y", "i"), (r"(?<=[aeiou])h$", "")),  # y as a vowel; a final h off
        classes=str.maketrans({"i": "e", "u": "o", "a": "e"}),
        after=((r"(.)\1+", r"\1"),),  # doubled letters as one
        vowels=frozenset("eo"),
    ),
}
NAME_FIRST = "name_first"  # grammar pass flag in what it heard: the name opened the utterance
EXTRA_FIRST = "extra_name_first"  # the same for an extra name


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
        self._names_by_language: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._folded: dict[str, dict[str, tuple[str, ...]]] = {}
        self._oov: dict[str, int] = {}
        self._leads = {
            language: tuple(
                p for p in (normalise(x) for x in [*ks.name_lead_ins, *ks.greeting]) if p
            )
            for language, ks in self._sets.items()
        }
        self._rebuild()

    def _rebuild(self) -> None:
        """Phrase lists for the current name; rebuilt when the name or its checked languages
        change (a model finished loading, the platform name arrived)."""
        resolved = self._names.resolve(self._configured, tuple(self._sets))
        if resolved == self._resolved:
            return
        self._resolved = resolved
        self._phrases = {}
        by_language: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        folded: dict[str, dict[str, tuple[str, ...]]] = {}
        oov: dict[str, int] = {}  # assigned once below: a report never reads a partial dict
        aliases = resolved.aliases_apply()
        extras = [e.matching for e in resolved.extras]
        for language, ks in self._sets.items():
            names = resolved.names_for(language)
            by_language[language] = _split(names)
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
            # The grammar lists render a name with accents kept (the vocabulary's spelling), the
            # free-text lists with them stripped; ``names_for`` checked the stripped form.
            templates = ks.name_wake_templates
            rendered = [
                normalise(t.replace("{name}", n.matching)) for n in names for t in templates
            ]
            listed = self._phrases[language]
            listed["wake"] = _distinct([*listed["wake"], *rendered])
            grammar = {cat: _distinct([fold(x) for x in items]) for cat, items in lists.items()}
            spoken = [fold(n.display) for n in names]
            grammar["wake"] += [fold(t.replace("{name}", n)) for n in spoken for t in templates]
            grammar["name"] += [fold(n.display) for n in names if not n.extra]
            grammar[EXTRA_NAME] = [fold(n.display) for n in names if n.extra]
            folded[language] = {cat: tuple(_distinct(items)) for cat, items in grammar.items()}
            # an extra name out of this vocabulary is counted with the wake phrases, never named
            bad = self._names.out_of_vocab(language, [*listed["wake"], *extras])
            if bad is not None:
                oov[language] = bad
        self._oov = oov
        self._folded = folded
        self._names_by_language = by_language

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

    def lead_in(self, words: str, language: str) -> bool:
        """Whether ``words`` (any spelling) is empty or one lead-in phrase of ``language``: what
        may come before the name for ``name_first``."""
        plain = normalise(words)
        return not plain or plain in self._leads.get(language, ())

    def name_first(self, text: str, language: str, extras: bool = True) -> bool:
        """Whether ``text`` opens with a name the robot answers to or one of its aliases (whole
        words), alone or after one lead-in phrase of ``language`` (``lead_in``). ``extras``
        off: the robot's own name and its aliases only (over the radio, SPEC 8)."""
        padded = f" {normalise(text)} "
        if padded == "  ":
            return False
        self._rebuild()
        names = [
            *self._phrases.get(language, {}).get("name", []),
            *self._names_in(language, extras),
        ]
        starts = [" ", *(f" {lead} " for lead in self._leads.get(language, ()))]
        return any(padded.startswith(f"{s}{n} ") for s in starts for n in names)

    def name_close(
        self, text: str, language: str, max_edits: int, min_letters: int, extras: bool = True
    ) -> bool:
        """Whether ``text`` holds a name the robot answers to, or a close spelling of it, as whole
        words in order (module docstring): for a server's transcript only. False without a
        usable name. ``extras`` off: the robot's own name only (over the radio, SPEC 8)."""
        self._rebuild()
        words = normalise(text).split()
        spelling = SPELLING.get(language, DEFAULT_SPELLING)
        for name in self._names_in(language, extras):
            targets = name.split()
            for i in range(len(words) - len(targets) + 1):
                pairs = zip(words[i : i + len(targets)], targets, strict=True)
                if all(_close(w, t, max_edits, min_letters, spelling) for w, t in pairs):
                    return True
        return False

    def _names_in(self, language: str, extras: bool = True) -> tuple[str, ...]:
        """The normalised names usable in ``language`` (``ResolvedName.names_for``), computed
        once per rebuild: the robot's own name, then (with ``extras``) the extra names."""
        own, extra = self._split_names(language)
        return own + extra if extras else own

    def _split_names(self, language: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The robot's own and the extra names usable in ``language``: from the rebuild for a
        listed language, else resolved now (a name with no vocabulary check works in any)."""
        found = self._names_by_language.get(language)
        if found is not None:
            return found
        name = self._resolved
        return ((), ()) if name is None else _split(name.names_for(language))

    def spot(self, text: str, language: str) -> set[str]:
        """The categories whose phrases occur as whole words in ``text``."""
        padded = f" {normalise(text)} "
        if padded == "  ":
            return set()
        self._rebuild()
        found: set[str] = set()
        own, extra = self._split_names(language)
        if any(f" {n} " in padded for n in own):
            found.add("name")
        if any(f" {n} " in padded for n in extra):
            found.add(EXTRA_NAME)
        for cat, phrases in self._phrases.get(language, {}).items():
            if any(f" {p} " in padded for p in phrases):
                found.add(cat)
        return found


def _split(names: tuple[SpokenName, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The normalised own names and the extra names of ``names``."""
    own = tuple(n.matching for n in names if not n.extra)
    return own, tuple(n.matching for n in names if n.extra)


def _distinct(phrases: list[str]) -> list[str]:
    """``phrases`` without empty ones and repeats, in order (a repeat would be counted twice)."""
    return list(dict.fromkeys(p for p in phrases if p))


def _close(
    word: str, target: str, max_edits: int, min_letters: int, spelling: Spelling = DEFAULT_SPELLING
) -> bool:
    """Whether normalised ``word`` is a close spelling of normalised ``target`` (the module
    docstring's rule): exact when ``target`` is shorter than ``min_letters``."""
    if word == target:
        return True
    if len(target) < min_letters:
        return False
    edits = _vowel_edits(spelling.apply(word), spelling.apply(target), spelling.vowels)
    return edits <= max_edits


def _vowel_edits(a: str, b: str, vowels: frozenset[str]) -> float:
    """The fewest vowel insertions and deletions that turn ``a`` into ``b`` (infinite when a
    consonant differs): an edit distance with no substitution and no consonant edit, over two
    short words."""
    inf = float("inf")
    row = [0.0]
    for ch in b:
        row.append(row[-1] + 1 if ch in vowels else inf)
    for ca in a:
        prev = row
        row = [prev[0] + 1 if ca in vowels else inf]
        for j, cb in enumerate(b, 1):
            best = prev[j - 1] if ca == cb else inf
            if ca in vowels:
                best = min(best, prev[j] + 1)
            if cb in vowels:
                best = min(best, row[j - 1] + 1)
            row.append(best)
    return row[-1]
