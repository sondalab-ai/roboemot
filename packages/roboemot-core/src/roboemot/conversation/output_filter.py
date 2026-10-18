"""Output filter before TTS while a minor is present (SPEC 12, Behaviour and content).

A per-language blocklist (``safety.minor_output_blocklist``) matched on the same normalisation
the keyword spotter uses (case, accents and punctuation ignored) after removing invisible
format characters (zero-width spaces and joiners). An entry matches whole words; a trailing
``*`` makes it a prefix (``kill*`` also blocks killing, killed), and multi-word entries match as
a phrase. Every language's list applies to every sentence, so a model that answers in another
language than asked does not slip past. A phrase the splitter cut between two sentences is
caught by checking the tail of the previous sentences with the new one.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from ..voice.keywords import normalise

PREFIX = "*"


def clean(text: str) -> str:
    """``text`` without Unicode format characters (category Cf: zero-width space, joiners,
    bidi marks, soft hyphen), which would otherwise split a word for the matcher only."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


class OutputFilter:
    """``blocked(text)`` is True when ``text`` contains a blocklisted word or phrase."""

    def __init__(self, blocklist: Mapping[str, Sequence[str]]) -> None:
        self.source = {k: list(v) for k, v in blocklist.items()}
        patterns, words = [], 0
        for entry in sorted({e for entries in blocklist.values() for e in entries}):
            prefix = clean(entry).strip().endswith(PREFIX)
            norm = normalise(clean(entry).rstrip(PREFIX + " "))
            if not norm:
                continue
            words = max(words, norm.count(" ") + 1)
            tail = "" if prefix else r"(?=\s)"
            patterns.append(r"(?<=\s)" + re.escape(norm) + tail)
        self._regex = re.compile("|".join(patterns)) if patterns else None
        self._window = words - 1

    def blocked(self, text: str, previous: str = "") -> bool:
        """Whether ``text`` is blocked. ``previous`` is what the same answer already said: the
        tail of it (the longest phrase's words, less one) is checked together with ``text``."""
        if self._regex is None:
            return False
        words = normalise(clean(previous)).split()
        tail = words[-self._window :] if self._window > 0 else []
        padded = f" {' '.join([*tail, normalise(clean(text))])} "
        return self._regex.search(padded) is not None
