"""The offline cue spoken through the runtime's speech output (SPEC 12A, Behaviour while offline;
SPEC 8, voice)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..config.schema import RegisterParams, StateEffects

log = logging.getLogger(__name__)

Delivery = tuple[RegisterParams, StateEffects, str, float]  # register, circadian, mood, intensity


class SpeechCue:
    """An :class:`~roboemot.health.responder.OfflineCue` that queues the phrase on the speech
    output with the register, circadian state and affect of this moment.

    Nothing is said (and nothing raised) when ``muted()`` (asleep or paused), or when
    ``output()`` is ``None``. The text is never logged here (SPEC 12)."""

    def __init__(
        self,
        output: Callable[[], Any | None],
        delivery: Callable[[], Delivery],
        muted: Callable[[], bool],
    ) -> None:
        self._output, self._delivery, self._muted = output, delivery, muted

    def say(self, text: str, language: str) -> bool:
        """Queue ``text``; True if the output took it. Called under the responder's lock, from
        any thread."""
        try:
            out = self._output()
            if out is None or self._muted():
                return False
            register, circadian, mood, intensity = self._delivery()
            job = out.say(
                text,
                language=language,
                register=register,
                circadian=circadian,
                mood=mood,
                intensity=intensity,
            )
            return job is not None
        except Exception as exc:
            log.warning("offline cue: not said (%s)", type(exc).__name__)
            return False


__all__ = ["Delivery", "SpeechCue"]
