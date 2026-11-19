"""The radio's stops without its tools: a stop said without the name, and a fist (SPEC 5, 8, 11
Radio, 12A).

Both go through :func:`stop_and_nod`: the player's ``stop`` with the stop's reason, then one nod.

The fist stop (owner decision 2026-11-19, on by default): :meth:`FistStop.wanted` is the vision
pipeline's ``hands_wanted`` seam, true only while ``tools.radio.fist_stop`` is on, the robot is
awake and the music can be heard (``MediaPlayer.audible``), so the hand check costs no CPU at
any other time. A ``percept.hand`` fist (:meth:`FistStop.on_hand`) while a station is set stops
it with reason ``fist_stop`` and publishes the decision event ``media``/``fist_stop`` with the
hold's ``held_s`` only; a fist with no station set is counted as ignored.

Threads: the bus calls its subscribers in the publisher's thread, and ``percept.hand`` is
published by the vision pipeline on the camera thread (``roboemot-camera``: the hand check's own
thread hands its results back there). The seam is read on that thread too. The stop touches only
any-thread parts: the player's ``stop`` (its leaf lock, the decoder kill, the speech output's
``cut_media`` under that output's lock, then ``media.stopped``), the renderer's ``request`` (its
request lock) and the decision-event sink, which never waits. The counters are plain ints written
only on that thread and read by the debug report from any thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from ..bus import Message
from ..config.schema import RoboEmotConfig
from ..diag import EventSink, no_events
from ..vision.hands import HandGesture
from .host import RuntimeHost

STOP_GESTURE = "nod"  # the robot's answer to a radio stop without its tools (SPEC 11, Radio)
FIST_STOP = "fist_stop"  # ``media.stopped``'s reason, and the ``media`` event's (SPEC 11)
EVENT_KIND = "media"
FIST = "fist"


class Radio(Protocol):
    def audible(self) -> bool: ...

    def playing(self) -> bool: ...

    def stop(self, reason: str) -> None: ...


def stop_and_nod(host: RuntimeHost, player: Radio, reason: str) -> None:
    """Stop the station for ``reason``, then nod once through ``host.expression`` (the
    ``play`` skill's path, with the time, so the renderer refuses it rather than play it late);
    without a renderer it only stops. Any thread (the listener's for the voice stop, the
    camera thread for the fist stop): the player's stop and the renderer's request are
    thread-safe."""
    player.stop(reason)
    expression = host.expression
    if expression is not None:
        expression.request(STOP_GESTURE, time.monotonic())


class FistStop:
    """A fist stops the radio (see the module). ``media`` gives the run's player (None until
    start-up builds it, or without one), ``awake`` whether the robot is awake, ``cfg`` the
    effective config (read on each call, so a reload needs no wiring) and ``stop`` the
    stop-and-nod path."""

    def __init__(
        self,
        media: Callable[[], Radio | None],
        awake: Callable[[], bool],
        cfg: Callable[[], RoboEmotConfig],
        stop: Callable[[Radio, str], None],
    ) -> None:
        self._media, self._awake, self._cfg, self._stop = media, awake, cfg, stop
        self.events: EventSink = no_events
        self.fist_stops = 0
        self.fist_ignored = 0  # a fist with no station set
        self.last_held_s: float | None = None

    def wanted(self) -> bool:
        """The ``hands_wanted`` seam: the field on, the robot awake and the music audible."""
        if not self._cfg().tools.radio.fist_stop or not self._awake():
            return False
        media = self._media()
        return media is not None and media.audible()

    def on_hand(self, msg: Message) -> None:
        """A ``percept.hand`` event: a fist stops the station that is set, with its nod."""
        gesture = msg.payload
        if not isinstance(gesture, HandGesture) or gesture.gesture != FIST:
            return
        if not self._cfg().tools.radio.fist_stop:
            return
        media = self._media()
        if media is None or not media.playing():
            self.fist_ignored += 1
            return
        self._stop(media, FIST_STOP)
        self.fist_stops += 1
        self.last_held_s = gesture.held_s
        self.events(EVENT_KIND, FIST_STOP, held_s=gesture.held_s)

    def snapshot(self) -> dict[str, Any]:
        """The ``roboemot.fist_stop`` report section: counts only."""
        return {
            "fist_stops": self.fist_stops,
            "fist_ignored": self.fist_ignored,
            "last_held_s": None if self.last_held_s is None else round(self.last_held_s, 2),
        }
