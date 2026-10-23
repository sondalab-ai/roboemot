"""How the robot behaves while a link is down (SPEC 12A, Behaviour while offline).

Everything local keeps running (idle, gaze, affect); connectivity only colours it:

- ``link.*`` events go to the affect engine's appraisal rules (subdued while the network is
  lost, perking up when it or the robot link comes back);
- on recovery the robot plays a small perk-up gesture; when someone speaks while the network is
  down (:meth:`OfflineResponder.speech_while_offline`) it plays a puzzled, slightly apologetic
  one instead of waiting for a model that cannot answer;
- an :class:`OfflineCue` says ``connectivity.offline_phrases[language]`` once per outage and on
  speech while offline, at most every ``cue_cooldown_s``. The runtime installs a
  :class:`~roboemot.health.speech_cue.SpeechCue` once the speech output exists; without a
  speaker the cue stays :class:`SilentCue`.

The gestures are the manifest's ``gesture_map`` entries ``perk_up`` and ``puzzled`` (SPEC 7):
additive primitives over whichever of their channels the robot has. A manifest without one of
them simply shows no motion for it.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Protocol

from ..affect import AffectEngine
from ..affect.expression import ExpressionRenderer
from ..config.schema import ConnectivityConfig
from ..motion.core import Command
from ..motion.primitives import Player, Primitive

NETWORK_LOST, NETWORK_RESTORED = "link.network.lost", "link.network.restored"
ROBOT_RESTORED = "link.robot.restored"
UNAVAILABLE = "conversation.unavailable"


class OfflineCue(Protocol):
    def say(self, text: str, language: str) -> bool:
        """Say ``text``; whether it was actually queued (muted or no output: ``False``)."""
        ...


class SilentCue:
    """The cue when there is no speech output (voice disabled, or before it is built)."""

    def say(self, text: str, language: str) -> bool:
        return False


PERK_UP, PUZZLED = "perk_up", "puzzled"


@dataclass
class OfflineResponder:
    cfg: ConnectivityConfig
    expression: ExpressionRenderer
    language: str
    affect: AffectEngine | None = None
    cue: OfflineCue = field(default_factory=SilentCue)
    _players: list[Player] = field(default_factory=list)
    _last_cue: float = -math.inf
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_event(self, event: str, now: float) -> None:
        """Link change from a :class:`~roboemot.health.Link` (any thread)."""
        with self._lock:
            if self.affect:
                self.affect.on_event(event, now)
            if event == NETWORK_LOST:
                self._say(now)
            elif event in (NETWORK_RESTORED, ROBOT_RESTORED):
                self._play(self.expression.gesture(PERK_UP), now)

    def speech_while_offline(self, now: float, language: str | None = None) -> None:
        """Someone spoke and the router is :class:`~roboemot.llm.Offline`: reply nonverbally and
        with the cue in ``language`` (the utterance's; default the configured one)."""
        with self._lock:
            if self.affect:
                self.affect.on_event(UNAVAILABLE, now)
            self._play(self.expression.gesture(PUZZLED), now)
            self._say(now, language)

    def commands(self, now: float) -> list[Command]:
        """Gesture commands for this tick; feed them to the motion loop with the idle ones."""
        with self._lock:
            self._players = [p for p in self._players if not p.done(now)]
            return [c for p in self._players if (c := p.command(now)) is not None]

    def _play(self, prim: Primitive | None, now: float) -> None:
        if prim is not None:  # the manifest has the gesture and the robot one of its channels
            self._players = [p for p in self._players if p.prim.name != prim.name]
            self._players.append(Player(prim, now, source=f"connectivity:{prim.name}"))

    def _say(self, now: float, language: str | None = None) -> None:
        """The phrase of ``language`` (else of the configured one), at most every cooldown."""
        phrases = self.cfg.offline_phrases
        lang = language if language in phrases else self.language
        text = phrases.get(lang)
        due = now - self._last_cue >= self.cfg.cue_cooldown_s
        if text and due and self.cue.say(text, lang):  # a muted cue leaves the cooldown free
            self._last_cue = now


__all__ = ["PERK_UP", "PUZZLED", "OfflineCue", "OfflineResponder", "SilentCue"]
