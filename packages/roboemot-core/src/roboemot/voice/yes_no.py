"""Nod on a spoken yes, shake on a spoken no (SPEC 7, 8 stage 3).

The fast path hands :class:`YesNoGestures` the answer of an addressed utterance: a short final
text (``voice.fastpath.gesture_max_words``) holding exactly one yes or no keyword. Partials never
trigger a gesture, and the affect events (``speech.keyword.yes`` / ``.no``) are emitted by the
fast path as before, independent of this module.

The handler asks the expression renderer for ``nod`` / ``shake`` with the current time, so the
renderer's rules decide: a request inside ``expression.min_interval_s`` of the last expression is
refused rather than played late, and a resting robot (circadian) does not move; reflex priority,
register, minor cap and limiter apply to what does play.

No gesture when the yes or no is not for the robot to mirror:

- the robot is speaking (``SpeechActivity``, echo tail included): its own "Certo!" coming back
  through the microphone, or the person talking over it;
- the robot's last spoken sentence was a question and it ended no more than
  ``voice.fastpath.answer_window_s`` ago (or is still playing): the person is answering it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..config.schema import FastpathConfig

log = logging.getLogger(__name__)

GESTURE_FOR = {"yes": "nod", "no": "shake"}


class YesNoGestures:
    """Callable ``(category, now)`` for ``FastPath(on_answer=...)``; hot-reloadable config."""

    def __init__(
        self,
        request: Callable[[str, float], bool],
        question_end: Callable[[], float | None],
        speaking: Callable[[float], bool],
        cfg: FastpathConfig,
    ) -> None:
        self._request, self._question_end, self._speaking = request, question_end, speaking
        self._cfg = cfg

    def apply_config(self, cfg: FastpathConfig) -> None:
        """Hot reload of ``voice.fastpath`` (the answer window)."""
        self._cfg = cfg

    def _answering(self, now: float) -> bool:
        """Whether ``now`` is inside the window after the robot's last question."""
        end = self._question_end()
        return end is not None and now - end <= self._cfg.answer_window_s

    def __call__(self, category: str, now: float) -> None:
        gesture = GESTURE_FOR.get(category)
        if gesture is None:
            return
        try:  # runs on the listener thread: any failure means no gesture, never a dead listener
            if self._speaking(now) or self._answering(now):
                return
            self._request(gesture, now)
        except Exception as exc:
            log.warning("yes/no gesture: %s", type(exc).__name__)
