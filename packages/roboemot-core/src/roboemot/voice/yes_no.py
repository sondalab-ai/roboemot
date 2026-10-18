"""Nod on a spoken yes, shake on a spoken no (SPEC 7, 8 stage 3).

:class:`YesNoGestures` sits between the voice fast path and the affect runner. Every event goes
on to ``emit`` unchanged; ``speech.keyword.yes`` and ``speech.keyword.no`` also ask the
expression renderer for the ``nod`` / ``shake`` gesture. The renderer owns every rule about
whether and how it plays (minimum interval, register, minor cap, limiter), so a refused request
is just dropped.

The exception: when the robot's last spoken sentence was a question and it ended no more than
``voice.fastpath.answer_window_s`` ago (or is still playing), the yes or no is the person's
answer, not something to mirror, so no gesture is requested. The affect event is still emitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

GESTURE_FOR = {"speech.keyword.yes": "nod", "speech.keyword.no": "shake"}


class YesNoGestures:
    """An ``Emit`` that forwards to ``emit`` and requests a gesture on yes / no."""

    def __init__(
        self,
        emit: Callable[[str, float, dict[str, Any]], None],
        request: Callable[[str], bool],
        question_end: Callable[[], float | None],
        answer_window_s: Callable[[], float],
    ) -> None:
        self._emit, self._request = emit, request
        self._question_end, self._window = question_end, answer_window_s

    def answering(self, now: float) -> bool:
        """Whether ``now`` is inside the window after the robot's last question."""
        end = self._question_end()
        return end is not None and now - end <= self._window()

    def __call__(self, event: str, now: float, payload: dict[str, Any]) -> None:
        self._emit(event, now, payload)
        gesture = GESTURE_FOR.get(event)
        if gesture is not None and not self.answering(now):
            self._request(gesture)
