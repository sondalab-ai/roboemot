"""A hand on the robot stops the radio (SPEC 4A, 11 Radio, 12A).

Over the music the recognizers mishear the robot's name, so "stop the radio" by voice is not
reliable (owner robot run 2026-11-18). The motion loop already detects external control: the
measured pose stops following its commands and the loop yields (``MotionLoop.external``,
SPEC 12A, External control). A yield lasts at least ``motion.platform_move_grace_s``, so its
length says nothing; what the pose does in it does. Once the music has been audible
(``MediaPlayer.audible``) for ``tools.radio.touch_stop_check_s`` of a yield, the yield is looked
at once (``MotionLoop.yield_view``):

- no part is away from the last target the loop wrote to it: a lag that settled (an old state
  sample, an overloaded CPU) or a hand already gone; no stop (``skipped_settled``);
- an away joint is a known unreachable stop, or a unit holds on one: our own target meeting the
  body; no stop (``skipped_unreachable``);
- every part is away, on a robot of several parts: a platform move of its own (go-to-sleep, an
  idle reset) shifts the whole robot, a hand usually one part; no stop (``skipped_platform``);
- else a hand holds the robot: the station stops (``media.stopped`` reason ``touch``) and the
  decision event ``media``/``touch_stop`` is published.

The robot then nods once, on the first tick the loop takes control back (the yield's grace after
the pose is still): a gesture requested while it yields would not be written. Motors off first
(a platform sleep) drops the nod. ``stops_then_asleep`` counts the stops the motors went off
within ``motion.platform_move_grace_s`` of: a platform move taken for a hand (report only).

Only the main loop's thread calls :meth:`tick`; the counters are plain ints read by the debug
report from any thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from ..diag import EventSink, no_events
from ..motion.external import YieldView

STOP_REASON = "touch"  # ``media.stopped``'s reason (SPEC 11, Radio)
EVENT_KIND, EVENT_REASON = "media", "touch_stop"
NOD = "nod"

log = logging.getLogger(__name__)


class Radio(Protocol):
    def audible(self) -> bool: ...

    def stop(self, reason: str) -> None: ...


class TouchStop:
    """Watches the loop's yields while the radio plays (see the module)."""

    def __init__(
        self,
        media: Callable[[], Radio | None],
        view: Callable[[], YieldView | None],
        gesture: Callable[[str, float], bool],
    ) -> None:
        self._media = media  # the radio's player, None while the run has none
        self._view = view  # the loop's ``yield_view``
        self._gesture = gesture  # the expression renderer's request (the play skill's path)
        self.events: EventSink = no_events
        self._since: float | None = None  # start of the audible part of the current yield
        self._checked = False  # this yield was looked at
        self._nod = False  # a nod waits for the loop to take control back
        self._stopped_t: float | None = None  # the last stop, until the grace after it
        self.stops = 0
        self.stops_then_asleep = 0
        self.last_held_s: float | None = None
        self.skipped = dict.fromkeys(("settled", "unreachable", "platform"), 0)
        self.nods = 0
        self.nods_refused = 0  # the renderer refused the nod (resting, too soon, no gesture)
        self.nods_dropped = 0  # the motors went off before the loop took control back

    def tick(self, now: float, external: bool, motors_off: bool, cfg: Any) -> None:
        """One app-tick step: ``external`` and ``motors_off`` are the loop's now (read in that
        order: the loop sets motors off before it clears a yield), ``cfg`` the effective config
        (read each tick, so a reload needs no wiring)."""
        if self._stopped_t is not None and now - self._stopped_t > cfg.motion.platform_move_grace_s:
            self._stopped_t = None
        if motors_off:
            self._asleep()
            return
        if not external:
            self._since, self._checked = None, False
            if self._nod:
                self._nod = False
                if self._gesture(NOD, now):
                    self.nods += 1
                else:
                    self.nods_refused += 1
            return
        radio = cfg.tools.radio
        if self._checked or not radio.touch_stop:
            return
        media = self._media()
        if media is None or not media.audible():
            self._since = None
            return
        if self._since is None:
            self._since = now
        held_s = now - self._since
        if held_s < radio.touch_stop_check_s:
            return
        self._checked = True
        skip = self._skip(self._view())
        if skip is not None:
            self.skipped[skip] += 1
            return
        media.stop(STOP_REASON)
        self._nod, self._stopped_t, self.last_held_s = True, now, held_s
        self.stops += 1
        self.events(EVENT_KIND, EVENT_REASON, held_s=held_s)

    @staticmethod
    def _skip(view: YieldView | None) -> str | None:
        """Why this yield is no hand (a key of ``skipped``), None for a hand."""
        if view is None or view.away == 0:
            return "settled"
        if view.unreachable:
            return "unreachable"
        if view.units > 1 and view.away == view.units:
            return "platform"
        return None

    def _asleep(self) -> None:
        """Motors off: the yield ends without a resume and no nod is left for the wake."""
        if self._stopped_t is not None:
            self.stops_then_asleep += 1
            self._stopped_t = None
        if self._nod:
            self.nods_dropped += 1
        self._since, self._checked, self._nod = None, False, False

    def snapshot(self) -> dict[str, Any]:
        """The ``roboemot.touch_stop`` report section: counts only."""
        return {
            "stops": self.stops,
            "stops_then_asleep": self.stops_then_asleep,
            "last_held_s": None if self.last_held_s is None else round(self.last_held_s, 2),
            "skipped": dict(self.skipped),
            "nods": self.nods,
            "nods_refused": self.nods_refused,
            "nods_dropped": self.nods_dropped,
            "nod_pending": self._nod,
        }


def touch_stop_tick(touch: TouchStop | None, now: float, loop: Any, cfg: Any) -> None:
    """The main loop's step (awake and asleep): ``loop.external`` read before
    ``loop.motors_off``. A failure is logged by type only and the tick goes on."""
    if touch is None:
        return
    try:
        external = loop.external
        touch.tick(now, external, loop.motors_off, cfg)
    except Exception as exc:
        log.warning("touch stop: %s", type(exc).__name__)
