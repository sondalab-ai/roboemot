"""The circadian runner: wall clock, light and presence in, state and effects out (SPEC 7A).

It drives :class:`CircadianModel` every ``circadian.update_period_s`` from the local hour of an
injectable clock, the elapsed time, the confirmed light level of the :class:`LightSampler` and
the presence cue, and on a state change it

- publishes ``circadian.state`` (also every ``circadian.publish_period_s``, and when the light
  level changes),
- moves the affect engine's arousal baseline (``AffectRunner.set_circadian``),
- starts the waking stretch when the robot leaves ``resting`` and, if that is inside the quiet
  hours because of a person or a wake call (not an app hold), raises the appraisal event
  ``circadian.woken_at_night``. The stretch plays at most every ``circadian.stretch_cooldown_s``.

A confirmed light level that rises from dark raises ``light.on``, one that falls to dark
``light.off``; the first level seen raises nothing. ``light_source: clock_only`` never touches
the camera and ``circadian.enabled: false`` makes the runner inert. Presence is the weak second
cue (SPEC 7A): a face, a recognised person or an attention target within
``circadian.presence_window_s`` (or ``presence_periods`` face-detection periods), or attention
engaged now. The app loop calls :meth:`tick`; the hold and wake calls come from any thread.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from ..bus import Bus, Message
from ..config.schema import RoboEmotConfig, StateEffects
from ..manifest import Dof
from ..motion.core import Command
from .model import (
    HOURS_PER_DAY,
    MINUTES_PER_HOUR,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    CircadianModel,
    CircadianState,
    State,
    in_window,
)
from .motion import CircadianMotion
from .sampler import LightSampler

TOPIC = "circadian.state"
STATES = ("awake", "drowsy", "resting")
REPORT_DIGITS = 3


class Clock(Protocol):
    """Time for the circadian rhythm: monotonic seconds and the local hour of day."""

    def monotonic(self) -> float: ...

    def hour(self) -> float: ...


class SystemClock:
    """Real time: ``time.monotonic`` and the local wall-clock hour."""

    def monotonic(self) -> float:
        return time.monotonic()

    def hour(self) -> float:
        now = datetime.now()
        return now.hour + now.minute / MINUTES_PER_HOUR + now.second / SECONDS_PER_HOUR


class AffectSink(Protocol):
    """What the runner needs of the affect runner."""

    def on_event(self, event: str, now: float, payload: dict[str, Any] | None = None) -> None: ...

    def set_circadian(self, effects: StateEffects) -> None: ...


class CircadianRunner:
    """See the module docstring."""

    def __init__(
        self,
        cfg: RoboEmotConfig,
        bus: Bus,
        affect: AffectSink | None = None,
        *,
        clock: Clock | None = None,
        dofs: dict[str, Dof] | None = None,
        detection_period: Callable[[], float] = lambda: math.inf,
        engaged: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.affect = affect
        self.clock: Clock = clock or SystemClock()
        self.detection_period = detection_period
        self.engaged = engaged
        self.model = CircadianModel(cfg.circadian)
        self.sampler = LightSampler(cfg.light, cfg.circadian, self.clock.monotonic)
        self.motion = CircadianMotion(cfg.circadian, cfg.circadian_motion, dofs) if dofs else None
        self._lock = threading.RLock()
        self._seen: float | None = None  # clock time of the latest sign of people
        self._last_update: float | None = None
        self._last_publish = -math.inf
        self._applied: State | None = None
        self._level: str | None = None
        self._force = False
        self._cause: str | None = None  # "wake" or "hold" since the last update
        self._last_stretch = -math.inf
        self.last: CircadianState | None = None
        self.woken_at_night = 0  # count, for the debug report
        self._unsubscribe = [
            bus.subscribe("percept.face", self._on_presence),
            bus.subscribe("person.recognized", self._on_presence),
            bus.subscribe("attention.target", self._on_target),
            bus.subscribe("percept.utterance", self._on_utterance),
        ]

    # ---- inputs

    def _on_presence(self, msg: Message) -> None:
        self._seen = self.clock.monotonic()

    def _on_utterance(self, msg: Message) -> None:
        """Speech is a sign of people; the wake word (``wake_word`` in the payload) also wakes
        the robot (SPEC 7A, Waking)."""
        self._seen = self.clock.monotonic()
        if getattr(msg.payload, "wake_word", False):
            self.wake()

    def _on_target(self, msg: Message) -> None:
        if msg.payload is not None:
            self._seen = self.clock.monotonic()

    def note_presence(self) -> None:
        """Record a sign of people now (face, recognised person, attention target)."""
        self._seen = self.clock.monotonic()

    def hold(self, state: str, minutes: float) -> None:
        """Keep the robot in ``state`` for ``minutes`` (an app, e.g. ``hold("awake", 30)`` during
        a party); the hold ends by itself and the model then follows its rhythm again."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {state!r}")
        top = self.cfg.circadian.hold_max_min
        if not (math.isfinite(minutes) and 0 < minutes <= top):
            raise ValueError(f"minutes must be in (0, {top}], got {minutes}")
        with self._lock:
            self.model.hold(state, minutes * SECONDS_PER_MINUTE, self._lag())  # type: ignore[arg-type]
            self._force = True
            self._cause = "hold"

    def wake(self) -> None:
        """Wake word or direct address: awake for ``circadian.wake_hold_s``. Process C is not
        reset, S stops decaying (SPEC 7A, Waking); the state changes within one tick."""
        with self._lock:
            self.model.wake(lag_s=self._lag())
            self._force = True
            self._cause = "wake"

    def _lag(self) -> float:
        """Seconds since the last model update: the model's clock stands that far behind, and a
        hold or wake counts from now (caller holds the lock)."""
        if self._last_update is None:
            return 0.0
        return max(0.0, self.clock.monotonic() - self._last_update)

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: parameters, thresholds and primitives; the model state is kept.

        Turning the rhythm off hands the robot back to the awake state at once: the arousal
        baseline, the rest posture and any yawn or stretch go."""
        with self._lock:
            was = self.cfg.circadian.enabled
            self.cfg = cfg
            if was and not cfg.circadian.enabled:
                self.model.state = "awake"
                self.model.clear_hold()
                self._applied = self.last = None
                self._last_update = None
                self._cause = None
                if self.motion is not None:
                    self.motion.reset()
                if self.affect is not None:
                    self.affect.set_circadian(cfg.circadian.states["awake"])
            self.model.cfg = cfg.circadian
            self.sampler.apply_config(cfg.light, cfg.circadian)
            if self.motion is not None:
                self.motion.apply_config(cfg.circadian, cfg.circadian_motion)
            if self.affect is not None and self._applied is not None and cfg.circadian.enabled:
                self.affect.set_circadian(self.model.effects())

    def close(self) -> None:
        """Detach from the bus."""
        for fn in self._unsubscribe:
            fn()
        self._unsubscribe.clear()

    # ---- outputs

    @property
    def enabled(self) -> bool:
        return self.cfg.circadian.enabled

    def amplitude_factor(self) -> float:
        """The circadian factor of the expression amplitude (SPEC 7): the state's
        ``expression_scale``, 1.0 while disabled."""
        return self.model.effects().expression_scale if self.enabled else 1.0

    def commands(self, now: float, attending: bool = False) -> list[Command]:
        """Rest posture, yawn and stretch for this tick (empty while disabled)."""
        if not self.enabled or self.motion is None:
            return []
        return self.motion.commands(now, self.model.state, attending)

    def present(self) -> bool:
        """Whether people were seen within the presence window, or attention is engaged."""
        if self.engaged():
            return True
        if self._seen is None:
            return False
        c = self.cfg.circadian
        period = self.detection_period()
        window = c.presence_window_s
        if math.isfinite(period):
            window = max(window, c.presence_periods * period)
        return self.clock.monotonic() - self._seen <= window

    # ---- the update

    def tick(self, now: float) -> CircadianState | None:
        """Run the model if an update is due; returns the new state or None between updates.

        ``now`` is the app's monotonic time, used for events and motion; the model's own time
        comes from the clock, so a replay can run it faster than real time.
        """
        c = self.cfg.circadian
        if not c.enabled:
            self._last_update = None
            return None
        events: list[tuple[str, dict[str, Any]]] = []
        publish: CircadianState | None = None
        t = self.clock.monotonic()
        with self._lock:
            if (
                not self._force
                and self._last_update is not None
                and t - self._last_update < c.update_period_s
            ):
                return None
            self._force = False
            dt = 0.0 if self._last_update is None else max(0.0, t - self._last_update)
            self._last_update = t
            hour = self.clock.hour()
            level = self.sampler.level()
            light_changed = False
            if level is not None:
                if self._level is not None and level != self._level:
                    light_changed = True
                    if self._level == "dark":
                        events.append(("light.on", {}))
                    elif level == "dark":
                        events.append(("light.off", {}))
                self._level = level
            prev = self.model.state
            st = self.model.update(
                hour,
                dt,
                light=level,  # type: ignore[arg-type]
                presence_active=self.present(),
            )
            changed = st.state != prev or self._applied is None
            if changed and self.affect is not None:
                self.affect.set_circadian(self.model.effects())
            self._applied = st.state
            cause, self._cause = self._cause, None
            if prev == "resting" and st.state != "resting":
                cooled = t - self._last_stretch >= c.stretch_cooldown_s
                if cooled and self.motion is not None and self.motion.start_waking(now):
                    self._last_stretch = t
                if in_window(hour, c.quiet_hours) and cause != "hold":
                    self.woken_at_night += 1
                    events.append(("circadian.woken_at_night", {}))
            self.last = st
            if changed or light_changed or t - self._last_publish >= c.publish_period_s:
                self._last_publish = t
                publish = st
        if publish is not None:  # outside the lock: a subscriber may call back into the runner
            self.bus.publish(TOPIC, publish, now)
        if self.affect is not None:
            for name, payload in events:
                self.affect.on_event(name, now, payload)
        return st

    # ---- report

    def snapshot(self) -> dict[str, Any]:
        """The ``robot.circadian.state`` information for the debug report (SPEC 7A, 12A):
        numbers and names only, nothing about frames or people."""
        with self._lock:
            st = self.last
            hold = self.model.hold_remaining(self._lag())
            c = self.cfg.circadian
            out: dict[str, Any] = {
                "enabled": c.enabled,
                "light_source": c.light_source,
                "state": self.model.state,
                "present": self.present(),
                "hold": None if hold is None else {"state": hold[0], "left_s": round(hold[1], 1)},
                "woken_at_night": self.woken_at_night,
                "waking": self.motion is not None and self.motion.playing == "stretch",
                "light": self.sampler.snapshot(),
            }
            if st is not None:
                out |= {
                    "hour": round(st.hour, REPORT_DIGITS),
                    "phase_h": round(
                        (c.peak_hour + st.phase_shift_h) % HOURS_PER_DAY, REPORT_DIGITS
                    ),
                    "phase_shift_h": round(st.phase_shift_h, REPORT_DIGITS),
                    "sleepiness": round(st.sleepiness, REPORT_DIGITS),
                    "process_s": round(st.s, REPORT_DIGITS),
                    "process_c": round(st.c, REPORT_DIGITS),
                    "light_level": st.light,
                }
            return out
