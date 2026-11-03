"""Alone stillness (SPEC 5A, Scanning; 10, Idle).

With no face in view and no sound for ``world.scan.alone_after_s`` the robot is alone: the scan
policy stops sweeps and glances, breathing and the other small idle motions go on. A face
(``percept.face``, ``person.recognized``) or a sound (``percept.utterance``, ``sound.loud``, or the
caller's ``busy`` flag: speech in progress, a speaker hint, a conversation) ends it at once.
Reflexes are not touched. Every change is a decision event (``alone``: ``start``, ``face``,
``sound``) for the Live tab, and :meth:`snapshot` is the ``roboemot.alone`` report section.

The delay is never shorter than ``circadian.presence_periods`` face-detection periods at the
governor's current rate, so a still face seen once every 30 s (drowsy or resting vision) is not
taken for an empty room. ``sound.loud`` comes from the reflex engine: with ``reflexes.enabled``
off, a voice counts only through the listener (an utterance, speech in progress).

The robot's own voice never counts: the listener drops audio while it speaks and the reflex
engine's loud onsets ignore its own sounds, so a self-soothing sound never ends the stillness.
While the platform holds no motors (``robot.asleep``) the app tick does not run this watch;
``robot.awake`` restarts the delay, so the robot is never alone on the first tick after a wake.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

from ..bus import Bus, Message
from ..config.schema import RoboEmotConfig
from ..diag import EventSink, no_events

FACE_TOPICS = ("percept.face", "person.recognized")
SOUND_TOPICS = ("percept.utterance", "sound.loud")
EVENT_KIND = "alone"


class AloneWatch:
    """Tracks the last face or sound and says whether the robot is alone (see the module)."""

    def __init__(
        self,
        cfg: RoboEmotConfig,
        bus: Bus | None = None,
        clock: Callable[[], float] = time.monotonic,
        detection_period: Callable[[], float] = lambda: math.inf,
    ) -> None:
        self.cfg = cfg
        self._clock = clock
        self._period = detection_period
        self._lock = threading.Lock()
        self._last: float | None = None  # clock time of the last face or sound; set at 1st tick
        self._cause = "sound"  # what the last sign of company was
        self.alone = False
        self.since: float | None = None
        self.changes = 0
        self.events: EventSink = no_events
        self._unsubscribe: list[Callable[[], None]] = []
        if bus is not None:
            for topic in FACE_TOPICS:
                self._unsubscribe.append(bus.subscribe(topic, self._on_face))
            for topic in SOUND_TOPICS:
                self._unsubscribe.append(bus.subscribe(topic, self._on_sound))
            self._unsubscribe.append(bus.subscribe("robot.awake", self._on_awake))

    def _on_face(self, msg: Message) -> None:
        self.note("face")

    def _on_sound(self, msg: Message) -> None:
        self.note("sound")

    def _on_awake(self, msg: Message) -> None:
        """The robot woke from platform sleep: nothing was watched meanwhile, so the delay
        restarts and the robot is not alone (no event: the asleep status covered the gap)."""
        with self._lock:
            self._last = self._clock()
            if self.alone:
                self.alone, self.since = False, None

    def note(self, cause: str) -> None:
        """A sign of company now: ``face`` or ``sound``. Safe from any thread."""
        with self._lock:
            self._last, self._cause = self._clock(), cause

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (SPEC 11A): the new delay applies from the next tick."""
        self.cfg = cfg

    def close(self) -> None:
        """Detach from the bus."""
        for fn in self._unsubscribe:
            fn()
        self._unsubscribe.clear()

    def after_s(self) -> float:
        """The delay in force: ``alone_after_s``, or ``presence_periods`` face-detection periods
        when those are longer."""
        delay = self.cfg.world.scan.alone_after_s
        period = self._period()
        if math.isfinite(period):
            delay = max(delay, self.cfg.circadian.presence_periods * period)
        return delay

    def tick(self, now: float, busy: bool = False) -> bool:
        """One app-tick step: ``busy`` (speech in progress, a speaker hint, a conversation)
        counts as a sound now. Returns whether the robot is alone; a change raises one event."""
        if busy:
            self.note("sound")
        delay = self.after_s()
        with self._lock:
            if self._last is None:  # the delay counts from the first tick, never from boot
                self._last = self._clock()
            alone = self._clock() - self._last >= delay
            cause = self._cause
        if alone == self.alone:
            return alone
        self.alone, self.since = alone, now
        self.changes += 1
        self.events(EVENT_KIND, "start" if alone else cause)
        return alone

    def snapshot(self, now: float) -> dict[str, Any]:
        """The ``roboemot.alone`` report section: the state and the quiet time, numbers only.
        ``alone_for_s`` counts from the tick that found the robot alone (``since``), not from
        the moment the delay ran out."""
        with self._lock:
            last = self._last
            quiet = None if last is None else round(max(0.0, self._clock() - last), 1)
        since = self.since if self.alone and self.since is not None else now
        return {
            "alone": self.alone,
            "alone_for_s": round(now - since, 1),
            "quiet_for_s": quiet,
            "alone_after_s": round(self.after_s(), 1),
            "changes": self.changes,
        }
