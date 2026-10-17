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
        self._window = max((w.count(" ") for w in self._entries), default=0) - 1

    def blocked(self, text: str, previous: str = "") -> bool:
        """Whether ``text`` is blocked. ``previous`` is what the same answer already said: the
        splitter can cut a phrase between two sentences, so the tail of ``previous`` (as many
        words as the longest phrase needs, less one) is checked together with ``text``."""
        words = normalise(previous).split()
        tail = words[-self._window :] if self._window > 0 else []
        padded = f" {' '.join([*tail, normalise(text)])} "
        return any(w in padded for w in self._entries)
