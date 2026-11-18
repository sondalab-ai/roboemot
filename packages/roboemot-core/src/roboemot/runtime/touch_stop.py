"""A hand on the robot stops the radio (SPEC 4A, 11 Radio, 12A).

Over the music the recognizers mishear the robot's name, so "stop the radio" by voice is not
reliable (owner robot run 2026-11-18). The motion loop already detects external control: the
measured pose stops following its commands and the loop yields (``MotionLoop.external``,
SPEC 12A, External control). While the radio is audible (``MediaPlayer.audible``), external
control that lasts ``tools.radio.touch_stop_min_s`` stops the station (``media.stopped`` reason
``touch``) and publishes the decision event ``media``/``touch_stop``. The robot then nods once,
on the first tick the loop takes control back (the yield's grace after the hand is still): a
gesture requested while it yields would not be written. Motors off (a platform sleep) forgets
the nod.

Only the main loop's thread calls :meth:`tick` and :meth:`forget`; the counters are plain ints
read by the debug report from any thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ..diag import EventSink, no_events

STOP_REASON = "touch"  # ``media.stopped``'s reason (SPEC 11, Radio)
EVENT_KIND, EVENT_REASON = "media", "touch_stop"
NOD = "nod"


class Radio(Protocol):
    def audible(self) -> bool: ...

    def stop(self, reason: str) -> None: ...


class TouchStop:
    """Watches the loop's external control while the radio plays (see the module)."""

    def __init__(self, media: Callable[[], Radio | None], gesture: Callable[[str], bool]) -> None:
        self._media = media  # the radio's player, None while the run has none
        self._gesture = gesture  # the expression renderer's request (the play skill's path)
        self.events: EventSink = no_events
        self._since: float | None = None  # start of the current external run with the radio on
        self._fired = False  # this external run already stopped the radio
        self._nod = False  # a nod waits for the loop to take control back
        self.stops = 0
        self.nods = 0
        self.nods_refused = 0  # the renderer refused the nod (resting, no such gesture)
        self.nods_dropped = 0  # a sleep came before the loop took control back

    def tick(self, now: float, external: bool, cfg: Any, motors_off: bool = False) -> None:
        """One app-tick step: ``external`` and ``motors_off`` are the loop's now, ``cfg`` the
        effective ``tools.radio`` section (read each tick, so a reload needs no wiring). Motors
        off (a platform sleep after its move) is not a resume: the nod is dropped."""
        if motors_off:
            self.forget()
            return
        if not external:
            self._since, self._fired = None, False
            if self._nod:
                self._nod = False
                if self._gesture(NOD):
                    self.nods += 1
                else:
                    self.nods_refused += 1
            return
        if self._fired or not cfg.touch_stop:
            return
        media = self._media()
        if media is None or not media.audible():
            self._since = None
            return
        if self._since is None:
            self._since = now
        held_s = now - self._since
        if held_s < cfg.touch_stop_min_s:
            return
        self._fired, self._nod = True, True
        self.stops += 1
        media.stop(STOP_REASON)
        self.events(EVENT_KIND, EVENT_REASON, held_s=held_s)

    def forget(self) -> None:
        """The robot is asleep (motors off): the run ends and no nod is left for the wake."""
        if self._nod:
            self.nods_dropped += 1
        self._since, self._fired, self._nod = None, False, False

    def snapshot(self) -> dict[str, Any]:
        """The ``roboemot.touch_stop`` report section: counts only."""
        return {
            "stops": self.stops,
            "nods": self.nods,
            "nods_refused": self.nods_refused,
            "nods_dropped": self.nods_dropped,
            "nod_pending": self._nod,
        }
