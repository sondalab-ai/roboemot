"""The robot's name for the voice pipeline (SPEC 8 stage 3, Wake word).

The name is ``voice.wake_word`` when set, else the one the robot platform reports (``source``,
the driver's ``robot_name()``; Reachy Mini: the name its owner gave it in the daemon setup),
else none. It is shown in the prompt and, per language, turned into wake phrases through the language's
``name_wake_templates``, but only when every word of the name is in that language's recognizer
vocabulary: a word the recognizer cannot produce can never be heard, so the language then keeps
its generic ``wake_phrases`` alone and the gap is logged once. The name is configuration or a
device label, not speech: it may be logged.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .keywords import normalise

log = logging.getLogger(__name__)

# (language, words) -> the words the recognizer's vocabulary lacks; None when it cannot tell yet.
MissingWords = Callable[[str, list[str]], list[str] | None]


@dataclass(frozen=True)
class ResolvedName:
    """The name in force: ``display`` for people and the prompt, ``matching`` (normalised) for
    spotting, ``languages`` where its words are recognisable (None: no vocabulary information,
    every language)."""

    display: str = ""
    matching: str = ""
    languages: frozenset[str] | None = None

    def usable(self, language: str) -> bool:
        """Whether a name exists and the recognizer of ``language`` can produce it."""
        return bool(self.matching) and (self.languages is None or language in self.languages)


def display_name(raw: str | None) -> str:
    """``raw`` as a person would say it: underscores and hyphens become spaces, edges trimmed."""
    return " ".join((raw or "").replace("_", " ").replace("-", " ").split())


class RobotName:
    """Holds the platform's name and resolves the effective one against the configuration."""

    def __init__(
        self,
        platform: str | None = None,
        missing: MissingWords | None = None,
        source: Callable[[], str | None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._source = source
        self._platform = display_name(platform) or display_name(self._poll())
        self._missing = missing
        self._generation = 0
        self._warned: set[tuple[str, str]] = set()
        self._cache: tuple[tuple[object, ...], ResolvedName] | None = None

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
        polled = display_name(self._poll()) if not self._platform else ""
        with self._lock:
            self._generation += 1
            if polled:
                self._platform = polled

    def display(self, configured: str | None) -> str:
        """The effective name for people and the prompt ('' without one)."""
        return display_name(configured) or self._platform

    def resolve(self, configured: str | None, languages: tuple[str, ...]) -> ResolvedName:
        """The effective name and the ``languages`` (those with keyword lists) it works in.
        Cached until the configuration, the platform name or the loaded models change."""
        key = (configured, languages, self._generation)
        with self._lock:
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            shown = display_name(configured) or self._platform
            matching = normalise(shown)
            if not matching:
                resolved = ResolvedName()
            elif self._missing is None:
                resolved = ResolvedName(shown, matching, None)
            else:
                resolved = ResolvedName(shown, matching, self._recognisable(matching, languages))
            self._cache = (key, resolved)
            return resolved

    def _recognisable(self, matching: str, languages: tuple[str, ...]) -> frozenset[str]:
        """Languages whose vocabulary has every word of ``matching`` (lock held)."""
        assert self._missing is not None
        words = matching.split()
        ok: set[str] = set()
        for language in languages:
            absent = self._missing(language, words)
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
