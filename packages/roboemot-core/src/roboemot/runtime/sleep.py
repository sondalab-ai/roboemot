"""Pausing the heavy work while the robot sleeps (SPEC 12, 12A).

The robot is asleep while the platform does not hold its motors in position control (the owner's
app, a daemon reset). The motion loop already follows the measured pose meanwhile
(``MotionLoop.motors_off``); the rest of the runtime would keep running at full CPU for no effect:
camera perception, the idle light read, the microphone listener and the behaviour tick. A small
robot's SBC idles only a few degrees below the threshold where its fan starts, so a busy
"sleeping" robot is audible. :class:`SleepGate` watches the loop from the app tick (never from the
50 Hz thread), publishes ``robot.asleep`` and
``robot.awake`` on the bus (the pattern of ``link.robot.lost`` and ``.restored``) and, with
``runtime.pause_when_asleep``, pauses those parts and resumes them on wake. Pausing keeps every
thread alive and idle: nothing is stopped, restarted or leaked, and models stay loaded. Kept
running: the motion loop's motors probe (the wake-up check), the web UI, diagnostics and the
network and robot link supervision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.bus import Bus
from roboemot.world.scan import SOURCE as SCAN_SOURCE

log = logging.getLogger(__name__)


class SleepGate:
    """Tracks the asleep state and applies or lifts the pause. ``enabled`` is
    ``runtime.pause_when_asleep`` read live, so a reload can opt out while asleep."""

    def __init__(
        self,
        host: Any,
        bus: Bus,
        loop: Any,
        perception: Any,
        light_reader: Any,
        listener: Any,
        gaze: Any,
        scan: Any,
        enabled: Callable[[], bool],
    ) -> None:
        self._host, self._bus, self._loop = host, bus, loop
        self._perception, self._light, self._listener = perception, light_reader, listener
        self._gaze, self._scan, self._enabled = gaze, scan, enabled
        self.asleep = False
        self.paused = False  # the pause is applied (asleep, and the config asks for it)
        self.changes = 0  # asleep/awake transitions seen
        self.since: float | None = None

    def tick(self, now: float) -> bool:
        """One app-tick step: publish a change of the asleep state, then apply or lift the pause.
        Returns whether the heavy work is paused (the caller then skips its own)."""
        asleep = bool(self._loop.motors_off and self._loop.link.online)
        if asleep != self.asleep:
            self.asleep, self.since = asleep, now
            self.changes += 1
            log.info("robot %s", "asleep" if asleep else "awake")
            self._bus.publish("robot.asleep" if asleep else "robot.awake", None, now)
        want = asleep and self._enabled()
        if want != self.paused:
            self.paused = want
            (self._pause if want else self._resume)()
        return self.paused

    def paused_parts(self) -> list[str]:
        """Names of the parts paused right now (``app_tick``, ``vision``, ``light_reader``,
        ``listener``); reads flags only, never the configuration."""
        service = self._perception.service
        flags = {
            "app_tick": self.paused,
            "vision": service.paused if service is not None else False,
            "light_reader": self._light.paused,
            "listener": self._listener.paused if self._listener is not None else False,
        }
        return [name for name, on in flags.items() if on]

    def snapshot(self, now: float) -> dict[str, Any]:
        """The ``roboemot.sleep`` report section: the state and what is paused."""
        service = self._perception.service
        return {
            "asleep": self.asleep,
            "asleep_for_s": now - self.since if self.asleep and self.since is not None else 0.0,
            "pause_when_asleep": self._enabled(),
            "changes": self.changes,
            "paused": {
                "app_tick": self.paused,
                "vision": service.paused if service is not None else False,
                "light_reader": self._light.paused,
                "listener": self._listener.paused if self._listener is not None else False,
            },
        }

    def _each(self, name: str, step: Callable[[], None]) -> None:
        """Run one pause or resume step; a failure is logged (type only), the others still run."""
        try:
            step()
        except Exception as exc:
            log.warning("sleep: %s: %s", name, type(exc).__name__)

    def _pause(self) -> None:
        """Cut the speech in progress as a barge-in does (the conversation says nothing more,
        the output stops, the turn is cancelled), release the gaze and scan and pause the
        camera, the light reader and the listener."""
        host = self._host
        conv, output = host.conversation, host.speech_output
        if conv is not None:
            self._each("conversation", conv.mark_interrupting)
        if output is not None:
            self._each("speech", output.stop)
        if conv is not None:
            self._each("conversation", conv.interrupt)
        self._each("gaze", self._gaze.release)
        self._each("attention command", lambda: self._loop.clear(ATTENTION_SOURCE))
        if self._scan is not None:
            self._each("scan", self._scan.reset)
            self._each("scan command", lambda: self._loop.clear(SCAN_SOURCE))
        if self._perception.service is not None:
            self._each("vision", self._perception.service.pause)
        self._each("light reader", self._light.pause)
        if self._listener is not None:
            self._each("listener", self._listener.pause_listening)

    def _resume(self) -> None:
        if self._listener is not None:
            self._each("listener", self._listener.resume_listening)
        self._each("light reader", self._light.resume)
        if self._perception.service is not None:
            self._each("vision", self._perception.service.resume)
