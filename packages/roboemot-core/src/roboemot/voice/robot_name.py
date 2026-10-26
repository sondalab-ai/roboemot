"""The robot's name for the voice pipeline (SPEC 8 stage 3, Wake word).

The name is ``voice.wake_word`` when set, else the one the robot platform reports (``source``,
the driver's ``robot_name()``; for example the name its owner gave it in the robot's setup),
else none. (``voice.default_name``, the manifest's name, is never the effective name: it only
gates the manifest's name aliases.) It is shown in the prompt
and, per language, turned into wake phrases through the language's ``name_wake_templates``, but
only when every word of the name is in that language's recognizer vocabulary: a word the
recognizer cannot produce can never be heard, so the language then keeps its generic
``wake_phrases`` alone and the gap is logged once. The name is configuration or a device label,
not speech: it may be logged. It also comes from outside (the owner's phone), so the prompt and
the spoken lines only get ``prompt_name``: letters, digits, spaces and ``'.-``, capped.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .keywords import normalise

log = logging.getLogger(__name__)

# (language, words) -> the words the recognizer's vocabulary lacks; None when it cannot tell yet.
MissingWords = Callable[[str, list[str]], list[str] | None]
PROMPT_PUNCTUATION = "'.-"  # besides letters, digits and spaces, what a prompt name may hold
NO_THROTTLE_S = 0.0


@dataclass(frozen=True)
class ResolvedName:
    """The name in force: ``display`` for people and the prompt, ``matching`` (normalised) for
    spotting, ``languages`` where its words are recognisable (None: no vocabulary information,
    every language)."""

    display: str = ""
    matching: str = ""
    languages: frozenset[str] | None = None
    own: str = ""  # the manifest's default name, normalised: the name the aliases are spellings of

    def usable(self, language: str) -> bool:
        """Whether a name exists and the recognizer of ``language`` can produce it."""
        return bool(self.matching) and (self.languages is None or language in self.languages)

    def aliases_apply(self) -> bool:
        """Whether the manifest's name aliases are in force: the robot has no name, or its name
        starts with the first word of the manifest's default name."""
        if not self.matching:
            return True
        return bool(self.own) and self.matching.split()[0] == self.own.split()[0]


def display_name(raw: str | None) -> str:
    """``raw`` as a person would say it: underscores and hyphens become spaces, edges trimmed,
    and a name that is all lower case gets a capital per word ("desk head" -> "Desk Head")."""
    text = " ".join((raw or "").replace("_", " ").replace("-", " ").split())
    if text and text == text.lower():
        text = " ".join(w[:1].upper() + w[1:] for w in text.split())
    return text


def sanitise_name(shown: str, max_chars: int) -> str:
    """``shown`` reduced to what may sit inside a prompt or a spoken line: letters, digits,
    spaces and ``'.-`` (controls, format and bidi characters, line breaks and emoji drop out),
    cut to ``max_chars``. '' when nothing usable is left."""
    text = unicodedata.normalize("NFC", shown)
    kept = "".join(
        c if c.isalpha() or c.isdigit() or c in PROMPT_PUNCTUATION else " " for c in text
    )
    kept = " ".join(kept.split())[:max_chars].strip()
    return kept if normalise(kept) else ""


class RobotName:
    """Holds the platform's name and resolves the effective one against the configuration."""

    def __init__(
        self,
        platform: str | None = None,
        missing: MissingWords | None = None,
        source: Callable[[], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._lock = threading.Lock()
        self._source = source
        self._clock = clock
        self._poll_every_s = NO_THROTTLE_S
        self._max_chars = 0
        self._default = ""
        self._missing = missing
        self._generation = 0
        self._warned: set[tuple[str, str]] = set()
        self._oov_warned: set[tuple[str, int]] = set()  # (language, count) wake checks logged
        self._cache: tuple[tuple[Any, ...], ResolvedName] | None = None
        self._platform = display_name(platform) or display_name(self._poll())
        self._polled_at = clock()

    def configure(self, voice: Any) -> None:
        """Take ``voice.name_poll_interval_s``, ``voice.name_max_chars`` and
        ``voice.default_name`` (alias gating only; idempotent)."""
        with self._lock:
            self._poll_every_s = voice.name_poll_interval_s
            self._max_chars = voice.name_max_chars
            self._default = display_name(voice.default_name)

    def _poll(self) -> str | None:
        """Ask the platform for its name; a failure is no name (the platform may not be ready)."""
        if self._source is None:
            return None
        try:
            return self._source()
        except Exception:
            return None

    def set_platform(self, name: str | None) -> None:
        """The name the robot platform reports (None: it has none)."""
        with self._lock:
            self._platform = display_name(name)
            self._cache = None

    def attach_vocabulary(self, missing: MissingWords) -> None:
        """Use ``missing`` to check the name against each recognizer's vocabulary."""
        with self._lock:
            self._missing = missing
            self._cache = None

    def refresh(self) -> None:
        """A recognizer model finished loading: check the languages that were not checkable, and
        ask the platform again for a name it had not reported yet."""
        self._platform_now(force=True)
        with self._lock:
            self._generation += 1

    def _platform_now(self, force: bool = False) -> str:
        """The platform name; while there is none, asks again at most every
        ``voice.name_poll_interval_s`` (the daemon's status can arrive after the app starts)."""
        with self._lock:
            now = self._clock()
            if (
                self._platform
                or self._source is None
                or (not force and now - self._polled_at < self._poll_every_s)
            ):
                return self._platform
            self._polled_at = now
        polled = display_name(self._poll())
        with self._lock:
            if polled and not self._platform:
                self._platform = polled
                self._cache = None
            return self._platform

    def display(self, configured: str | None) -> str:
        """The effective name for people and logs ('' without one)."""
        return display_name(configured) or self._platform_now()

    def prompt_name(self, configured: str | None) -> str:
        """The effective name made safe for the prompt and the spoken lines ('' without a usable
        one): ``sanitise_name`` capped at ``voice.name_max_chars``."""
        shown = self.display(configured)
        with self._lock:
            limit = self._max_chars
        return sanitise_name(shown, limit) if limit > 0 else ""

    def resolve(self, configured: str | None, languages: tuple[str, ...]) -> ResolvedName:
        """The effective name and the ``languages`` (those with keyword lists) it works in.
        Cached until the configuration, the platform name or the loaded models change."""
        platform = self._platform_now()
        with self._lock:
            key = (configured, languages, self._generation, platform, self._default)
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            shown = display_name(configured) or platform
            matching = normalise(shown)
            own = normalise(self._default)
            missing = self._missing
            if not matching:
                resolved = ResolvedName(own=own)
            elif missing is None:
                resolved = ResolvedName(shown, matching, None, own)
            else:
                resolved = ResolvedName(
                    shown, matching, self._recognisable(missing, matching, languages), own
                )
            self._cache = (key, resolved)
            return resolved

    def out_of_vocab(self, language: str, phrases: list[str]) -> int | None:
        """How many of ``phrases`` (normalised wake phrases) have a word the ``language``
        recognizer can never output, so they can never wake the robot; None when it cannot tell
        (no vocabulary, or that model is not loaded). A non-zero count is logged once per
        language and count, without the phrases."""
        with self._lock:
            missing = self._missing
        if missing is None:
            return None
        bad = 0
        for phrase in phrases:
            absent = missing(language, phrase.split())
            if absent is None:
                return None
            bad += bool(absent)
        with self._lock:
            fresh = bad and (language, bad) not in self._oov_warned
            if fresh:
                self._oov_warned.add((language, bad))
        if fresh:
            log.warning(
                "voice: %d %s wake phrase(s) have a word missing from the recognizer vocabulary "
                "and can never wake the robot",
                bad,
                language,
            )
        return bad

    def _recognisable(
        self, missing: MissingWords, matching: str, languages: tuple[str, ...]
    ) -> frozenset[str]:
        """Languages whose vocabulary has every word of ``matching`` (lock held)."""
        words = matching.split()
        ok: set[str] = set()
        for language in languages:
            absent = missing(language, words)
            if absent is None:
                continue
            if not absent:
                ok.add(language)
            elif (language, matching) not in self._warned:
                self._warned.add((language, matching))
                log.warning(
                    "voice: the robot name %r has %d word(s) missing from the %s recognizer "
                    "vocabulary; only the generic wake phrases work in %s",
                    matching,
                    len(absent),
                    language,
                    language,
                )
        return frozenset(ok)
