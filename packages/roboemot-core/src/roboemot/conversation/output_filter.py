"""Output filter before TTS while a minor is present (SPEC 12, Behaviour and content).

A per-language blocklist (``safety.minor_output_blocklist``) matched as whole words after the
same normalisation the keyword spotter uses (case, accents and punctuation ignored). Every
language's list applies to every sentence: a model that answers in another language than asked
does not slip past. Multi-word entries match as a phrase.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..voice.keywords import normalise


class OutputFilter:
    """``blocked(text)`` is True when ``text`` contains a blocklisted word or phrase."""

    def __init__(self, blocklist: Mapping[str, Sequence[str]]) -> None:
        self.source = {k: list(v) for k, v in blocklist.items()}
        entries = {normalise(w) for words in blocklist.values() for w in words}
        self._entries = tuple(f" {w} " for w in sorted(entries) if w)

    def blocked(self, text: str) -> bool:
        padded = f" {normalise(text)} "
        return any(w in padded for w in self._entries)
