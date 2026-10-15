"""How the robot behaves while a link is down (SPEC 12A, Behaviour while offline).

Everything local keeps running (idle, gaze, affect); connectivity only colours it:

- ``link.*`` events go to the affect engine's appraisal rules (subdued while the network is
  lost, perking up when it or the robot link comes back);
- on recovery the robot plays a small perk-up gesture; when someone speaks while the network is
  down (:meth:`OfflineResponder.speech_while_offline`) it plays a puzzled, slightly apologetic
  one instead of waiting for a model that cannot answer;
- an :class:`OfflineCue` says ``connectivity.offline_phrases[language]`` once per outage and on
  speech while offline, at most every ``cue_cooldown_s``. Until speech output is wired the cue
  is :class:`SilentCue`.

Gestures are additive primitives over whichever of their dofs the manifest has, so robots
without antennas or head roll still get the part they can show.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Protocol

from ..affect import AffectEngine
from ..config.schema import ConnectivityConfig, GestureConfig
from ..motion.core import Command
from ..motion.primitives import Keyframe, Player, Primitive

# Gesture envelope: reach the pose at 30% of its length, leave it at 70%.
RISE = 0.3
HOLD = 0.7
NETWORK_LOST, NETWORK_RESTORED = "link.network.lost", "link.network.restored"
ROBOT_RESTORED = "link.robot.restored"
UNAVAILABLE = "conversation.unavailable"


class OfflineCue(Protocol):
    def say(self, text: str, language: str) -> None: ...


class SilentCue:
    """Placeholder until a TTS backend is wired (SPEC 8)."""

    def say(self, text: str, language: str) -> None:
        pass


def _gesture(name: str, peak: dict[str, float], duration_s: float, dofs: set[str]) -> Primitive:
    """Additive bump: zero, ``peak`` held between RISE and HOLD, back to zero."""
    peak = {k: v for k, v in peak.items() if k in dofs}
    zero = dict.fromkeys(peak, 0.0)
    times = (0.0, duration_s * RISE, duration_s * HOLD, duration_s)
    frames = (zero, peak, peak, zero)
    return Primitive(
        name, tuple(Keyframe(t, v) for t, v in zip(times, frames, strict=True)), additive=True
    )


def puzzled(g: GestureConfig, dofs: set[str]) -> Primitive:
    """Head tilt with antennas spreading outward (right negative, left positive at home)."""
    peak = {
        "head.roll": g.head_roll_rad,
        "antenna_right": -g.antenna_rad,
        "antenna_left": g.antenna_rad,
    }
    return _gesture("puzzled", peak, g.duration_s, dofs)


def perk_up(g: GestureConfig, dofs: set[str]) -> Primitive:
    """Head lifts (negative pitch) and antennas rise toward vertical."""
    peak = {
        "head.pitch": -g.head_pitch_rad,
        "antenna_right": g.antenna_rad,
        "antenna_left": -g.antenna_rad,
    }
    return _gesture("perk_up", peak, g.duration_s, dofs)


@dataclass
class OfflineResponder:
    cfg: ConnectivityConfig
    dofs: set[str]
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
                self._play(perk_up(self.cfg.gestures, self.dofs), now)

    def speech_while_offline(self, now: float) -> None:
        """Someone spoke and the router is :class:`~roboemot.llm.Offline`: reply nonverbally."""
        with self._lock:
            if self.affect:
                self.affect.on_event(UNAVAILABLE, now)
            self._play(puzzled(self.cfg.gestures, self.dofs), now)
            self._say(now)

    def commands(self, now: float) -> list[Command]:
        """Gesture commands for this tick; feed them to the motion loop with the idle ones."""
        with self._lock:
            self._players = [p for p in self._players if not p.done(now)]
            return [c for p in self._players if (c := p.command(now)) is not None]

    def _play(self, prim: Primitive, now: float) -> None:
        if prim.keyframes[0].values:  # the robot has at least one of the gesture's dofs
            self._players = [p for p in self._players if p.prim.name != prim.name]
            self._players.append(Player(prim, now, source=f"connectivity:{prim.name}"))

    def _say(self, now: float) -> None:
        text = self.cfg.offline_phrases.get(self.language)
        if text and now - self._last_cue >= self.cfg.cue_cooldown_s:
            self._last_cue = now
            self.cue.say(text, self.language)


__all__ = ["OfflineCue", "OfflineResponder", "SilentCue", "perk_up", "puzzled"]
