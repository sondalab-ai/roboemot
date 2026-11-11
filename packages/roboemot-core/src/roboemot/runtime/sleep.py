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
thread alive and idle: nothing is leaked and models stay loaded. The one thing stopped is the
microphone capture, when the hub can (``CapturePause``) and :func:`mic_needed_asleep` says nothing
needs the sound (owner decision 2026-11-11: on the reference robot the capture kept about a third
of a core busy in sleep). It resumes on wake before the listener. Kept running: the motion loop's
motors probe (the wake-up check), the web UI, diagnostics and the network and robot link
supervision.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from roboemot.attention import SOURCE as ATTENTION_SOURCE
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.robot_platform import CapturePause
from roboemot.world.scan import SOURCE as SCAN_SOURCE

log = logging.getLogger(__name__)


def mic_needed_asleep(cfg: RoboEmotConfig) -> bool:
    """Whether a configured consumer needs the microphone while the platform sleeps (SPEC 12A).

    True exactly when the pause does not apply (``runtime.pause_when_asleep`` off): the listener
    then keeps running asleep, with every voice profile and wake-word policy. With the pause on,
    no consumer counts:

    * the voice listener (local speech, the wake word, the cloud voice's realtime route): paused
      with the rest, and the realtime session is closed;
    * the reflex sink (a loud sound: startle, the day nap's ``sound.loud`` wake): the behaviour
      tick is paused and the motors are off, so a reflex moves nothing; the nap clock restarts
      on ``robot.awake``;
    * the levels (Live tab meter, debug report): the meter says ``paused`` while asleep;
    * the wake word: it cannot wake a sleeping platform, since RoboEmot never calls the
      daemon's wake-up in app mode (SPEC 4A), whatever ``voice.wake_word_policy`` says.

    A later feature that wakes the platform by sound (a wake word policy, a sound reflex) must
    add its condition here, with its rows in ``tests/runtime/test_mic_needed_asleep.py``."""
    return not cfg.runtime.pause_when_asleep


class SleepGate:
    """Tracks the asleep state and applies or lifts the pause. ``enabled`` is
    ``runtime.pause_when_asleep`` read live, so a reload can opt out while asleep. ``audio`` is
    the microphone hub: its capture is paused while the pause applies and ``mic_needed`` (the
    live :func:`mic_needed_asleep`) is false, decided again on every tick. A hub without
    ``CapturePause`` is left alone."""

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
        realtime: Callable[[], Any] = lambda: None,
        *,
        audio: Any = None,
        mic_needed: Callable[[], bool] = lambda: True,
    ) -> None:
        self._host, self._bus, self._loop = host, bus, loop
        self._realtime = realtime  # the realtime speech route, when there is one
        self._perception, self._light, self._listener = perception, light_reader, listener
        self._gaze, self._scan, self._enabled = gaze, scan, enabled
        self._audio = audio if isinstance(audio, CapturePause) else None
        self._mic_needed = mic_needed
        self.asleep = False
        self.paused = False  # the pause is applied (asleep, and the config asks for it)
        self.mic_paused = False  # this gate paused the microphone capture
        self.changes = 0  # asleep/awake transitions seen
        self.since: float | None = None

    def tick(self, now: float) -> bool:
        """One app-tick step: publish a change of the asleep state, then apply or lift the pause.
        The microphone resumes before the listener and pauses after it (and after the speech
        stop). Returns whether the heavy work is paused (the caller then skips its own)."""
        asleep = bool(self._loop.motors_off and self._loop.link.online)
        if asleep != self.asleep:
            self.asleep, self.since = asleep, now
            self.changes += 1
            log.info("robot %s", "asleep" if asleep else "awake")
            self._bus.publish("robot.asleep" if asleep else "robot.awake", None, now)
        want = asleep and self._enabled()
        mic = want and self._audio is not None and not self._mic_needed()
        if self.mic_paused and not mic:
            self._set_mic(False)
        if want != self.paused:
            self.paused = want
            (self._pause if want else self._resume)()
        if mic and not self.mic_paused:
            self._set_mic(True)
        return self.paused

    def _set_mic(self, pause: bool) -> None:
        """Pause or resume the microphone capture; a failure is logged and the flag still
        moves (the hub retries a failed restart itself)."""
        self.mic_paused = pause
        audio = self._audio
        self._each("microphone", audio.pause_capture if pause else audio.resume_capture)

    def paused_parts(self) -> list[str]:
        """Names of the parts paused right now (``app_tick``, ``vision``, ``light_reader``,
        ``listener``, ``microphone``); reads flags only, never the configuration."""
        service = self._perception.service
        flags = {
            "app_tick": self.paused,
            "vision": service.paused if service is not None else False,
            "light_reader": self._light.paused,
            "listener": self._listener.paused if self._listener is not None else False,
            "microphone": self._audio.capture_paused if self._audio is not None else False,
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
                "vision": service.paused if service is not None else "absent",
                "light_reader": self._light.paused,
                "listener": self._listener.paused if self._listener is not None else "absent",
                "microphone": (self._audio.capture_paused if self._audio is not None else "absent"),
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
        realtime = self._realtime()
        if realtime is not None:  # the server session closes while asleep, reopens on wake
            self._each("realtime route", lambda: realtime.close_soon("asleep"))
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
        realtime = self._realtime()
        if realtime is not None:
            self._each("realtime route", realtime.start)
        if self._listener is not None:
            self._each("listener", self._listener.resume_listening)
        self._each("light reader", self._light.resume)
        if self._perception.service is not None:
            self._each("vision", self._perception.service.resume)
