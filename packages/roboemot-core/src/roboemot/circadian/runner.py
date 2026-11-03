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

Day nap (SPEC 7A, Day nap; the rules in :class:`DayNap`): with no confirmed face and no addressed
speech for ``circadian.nap_after_s`` of time outside the quiet hours, nobody present (with
``presence_keeps_awake``), the robot rests (reason ``nap``). A face counts when confirmed (a
second sighting from another frame within the presence window); speech counts when addressed
(the wake word, a face in view: the listener's rules), so a television's speech neither holds a
nap off nor wakes one; a conversation in progress (``conversing``) counts too. Activity ends the
nap like the wake word (``wake_hold_s``). Quiet hours end it without a wake. No nap while the
robot is asleep (``robot.asleep``), held, resting on its own or attending to a person, and the
idle clock restarts after each of those.

Three stimuli also end a day nap (owner decision 2026-11-03), each behind its switch:

- ``nap_wake_on_light``: a confirmed light level that rises (dark to dim or bright, dim to
  bright), unless it was confirmed within ``nap_light_grace_s`` of the nap's start (the rest
  posture moved the camera to a brighter view); a rise inside the grace is not looked at again
  when it ends;
- ``nap_wake_on_motion``: ``percept.scene_motion`` from the vision pipeline's motion watch,
  which runs only while :meth:`wants_scene_motion` (a nap with the switch on, the robot not
  asleep) and handles the head and its settle time itself; a report within
  ``nap_motion_grace_s`` of the nap's start is ignored (auto-exposure still settling on the
  rest posture's view);
- ``nap_wake_on_loud``: ``sound.loud``, the reflex engine's loud onset, which is already
  silent while the robot's own speech or sound plays (``SpeechActivity.sounding``). So a loud
  television now wakes a nap, though its speech still does not.

They wake only a day nap, never the night rest of the quiet hours, a rest of the rhythm's own,
a hold, or a nap while the platform sleeps (``robot.asleep``: the motors are off). Sound and
frames arrive on their own threads between updates, so the handler checks the quiet hours itself
(the nap flag clears only at the next update after they begin). While the robot is awake they do
not restart the idle clock (see :mod:`.nap`). Each start and wake is a decision event (``nap``:
``start``, ``face``, ``speech``, ``light``, ``motion``, ``loud``) and the debug report counts the
wakes by reason.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from ..affect.clips import ClipLibrary
from ..bus import Bus, Message
from ..config.schema import RoboEmotConfig, StateEffects
from ..diag import EventSink, no_events
from ..manifest import Dof
from ..motion.core import Command
from .light import LEVELS
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
from .nap import DayNap
from .sampler import LightSampler

TOPIC = "circadian.state"
STATES = ("awake", "drowsy", "resting")
Stimulus = Literal["light", "motion", "loud"]
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
        body_yaw: str | None,
        affect_intensity: Callable[[], float],
        clips: ClipLibrary | None = None,
        detection_period: Callable[[], float] = lambda: math.inf,
        engaged: Callable[[], bool] = lambda: False,
        conversing: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cfg = cfg
        self.bus = bus
        self.affect = affect
        self.clock: Clock = clock or SystemClock()
        self.detection_period = detection_period
        self.engaged = engaged
        self.conversing = conversing
        self.events: EventSink = no_events
        self.model = CircadianModel(cfg.circadian)
        self.sampler = LightSampler(cfg.light, cfg.circadian, self.clock.monotonic)
        self.motion = (
            CircadianMotion(
                cfg.circadian,
                cfg.circadian_motion,
                dofs,
                body_yaw=body_yaw,
                affect_intensity=affect_intensity,
                clips=clips,
            )
            if dofs
            else None
        )
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
        self.nap = DayNap()
        self._asleep = False  # the platform holds no motors (robot.asleep): no nap
        self._unsubscribe = [
            bus.subscribe("robot.asleep", self._on_asleep),
            bus.subscribe("robot.awake", self._on_awake),
            bus.subscribe("percept.face", self._on_presence),
            bus.subscribe("person.recognized", self._on_presence),
            bus.subscribe("attention.target", self._on_target),
            bus.subscribe("percept.utterance", self._on_utterance),
            bus.subscribe("sound.loud", self._on_loud),
            bus.subscribe("percept.scene_motion", self._on_scene_motion),
        ]

    # ---- inputs

    def _on_presence(self, msg: Message) -> None:
        """A face or a recognised person: presence now; a second sighting from another frame
        within the presence window is confirmed activity for the day nap."""
        t, window = self.clock.monotonic(), self._window()
        with self._lock:
            self._seen = t
            woke = self.nap.face(t, msg.stamp, window) and self._activity(t, "face")
        if woke:
            self.events("nap", "face")

    def _on_utterance(self, msg: Message) -> None:
        """Speech addressed to the robot is a sign of people and activity for the day nap (it
        wakes one), and the wake word (``wake_word`` in the payload) wakes it (SPEC 7A, Waking).
        Unaddressed speech (a television) does none of these."""
        if not getattr(msg.payload, "addressed", False):
            return
        t = self.clock.monotonic()
        with self._lock:
            self._seen = t
            woke = self._activity(t, "speech")
        if woke:
            self.events("nap", "speech")
        if getattr(msg.payload, "wake_word", False):
            self.wake()

    def _on_loud(self, msg: Message) -> None:
        """A loud onset (the reflex engine's, never the robot's own sound): wakes a day nap."""
        self._on_stimulus("loud")

    def _on_scene_motion(self, msg: Message) -> None:
        """Motion in front of the still camera (the pipeline's motion watch): wakes a day nap."""
        self._on_stimulus("motion")

    def _on_stimulus(self, reason: Stimulus) -> None:
        """A stimulus from another thread: wake a day nap and publish its decision event."""
        with self._lock:
            woke = self._stimulus(self.clock.monotonic(), reason)
        if woke:
            self.events("nap", reason)

    def _on_asleep(self, msg: Message) -> None:
        with self._lock:
            self._asleep = True

    def _on_awake(self, msg: Message) -> None:
        """The platform holds the motors again: the idle clock restarts (nothing was seen or
        heard while asleep)."""
        with self._lock:
            self._asleep = False
            self.nap.restart(self.clock.monotonic())

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
            self.nap.cancel(self.clock.monotonic())
            self._force = True
            self._cause = "hold"

    def wake(self) -> None:
        """Wake word or direct address: awake for ``circadian.wake_hold_s``. Process C is not
        reset, S stops decaying (SPEC 7A, Waking); the state changes within one tick."""
        with self._lock:
            self.model.wake(lag_s=self._lag())
            self.nap.cancel(self.clock.monotonic())
            self._force = True
            self._cause = "wake"

    def _activity(self, t: float, reason: str, lag_s: float | None = None) -> bool:
        """Confirmed face, addressed speech or a conversation at ``t`` (``reason``): the nap
        clock restarts and a nap ends with a wake (caller holds the lock; ``lag_s`` as in
        :meth:`CircadianModel.hold`, default the time since the last update). Returns whether a
        nap ended."""
        if not self.nap.activity(t, reason):
            return False
        self._woken(lag_s)
        return True

    def _stimulus(self, t: float, reason: Stimulus, lag_s: float | None = None) -> bool:
        """The light rising, motion or a loud sound at ``t`` (``reason``): ends a day nap with a
        wake when its switch (``nap_wake_on_<reason>``) is on, the platform is not asleep, the
        stimulus' grace after the nap's start is over and the quiet hours have not begun; never
        touches the idle clock while awake (caller holds the lock). Returns whether a nap
        ended."""
        c = self.cfg.circadian
        switch, grace_s = {
            "light": (c.nap_wake_on_light, c.nap_light_grace_s),
            "motion": (c.nap_wake_on_motion, c.nap_motion_grace_s),
            "loud": (c.nap_wake_on_loud, 0.0),
        }[reason]
        if not (c.enabled and switch) or self._asleep:
            return False
        started = self.nap.started
        if started is not None and t - started < grace_s:
            return False
        if in_window(self.clock.hour(), c.quiet_hours) or not self.nap.stimulus(t, reason):
            return False
        self._woken(lag_s)
        return True

    def _woken(self, lag_s: float | None) -> None:
        """A nap ended: awake for ``wake_hold_s`` from now (caller holds the lock)."""
        self.model.wake(lag_s=self._lag() if lag_s is None else lag_s)
        self._force = True
        self._cause = "wake"

    def wants_scene_motion(self) -> bool:
        """Whether the vision pipeline's motion watch should run: a day nap is in progress,
        ``nap_wake_on_motion`` is on and the platform is not asleep. Called on the vision thread
        for every frame, so it reads plain attributes without the lock."""
        c = self.cfg.circadian
        return c.enabled and c.nap_wake_on_motion and self.nap.napping and not self._asleep

    def _nap_blocked(self, dt: float, engaged: bool) -> bool:
        """Whether no nap is possible now (caller holds the lock): the robot asleep, a hold, a
        rest of the rhythm's own (the quiet hours only pause the clock), or attention on a
        person."""
        resting = self.model.state == "resting" and self.model.reason != "quiet_hours"
        held = self.model.hold_remaining(dt) is not None
        return self._asleep or held or resting or engaged

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
            was = self.cfg.circadian
            self.cfg = cfg
            after, before = cfg.circadian.nap_after_s, was.nap_after_s
            if after is not None and (before is None or after < before):
                self.nap.restart(self.clock.monotonic())  # never a nap from a stale clock
            if not was.enabled and cfg.circadian.enabled:
                self.nap.restart(self.clock.monotonic())
            if was.enabled and not cfg.circadian.enabled:
                self.model.state = "awake"
                self.model.reason = None
                self.model.clear_hold()
                self.nap.cancel(self.clock.monotonic())
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
        return self.clock.monotonic() - self._seen <= self._window()

    def _window(self) -> float:
        """The presence window: ``presence_window_s``, or ``presence_periods`` face-detection
        periods at the governor's current rate when that is longer."""
        c = self.cfg.circadian
        period = self.detection_period()
        window = c.presence_window_s
        if math.isfinite(period):
            window = max(window, c.presence_periods * period)
        return window

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
        decisions: list[str] = []
        publish: CircadianState | None = None
        t = self.clock.monotonic()
        talking, engaged = self.conversing(), self.engaged()  # outside the lock: other parts
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
            light_changed = rose = False
            if level is not None:
                if self._level is not None and level != self._level:
                    light_changed = True
                    rose = LEVELS.index(level) > LEVELS.index(self._level)  # type: ignore[arg-type]
                    if self._level == "dark":
                        events.append(("light.on", {}))
                    elif level == "dark":
                        events.append(("light.off", {}))
                self._level = level
            if rose and self._stimulus(t, "light", dt):
                decisions.append("light")
            if talking and self._activity(t, "speech", dt):
                decisions.append("speech")
            started = self.nap.step(
                t,
                after_s=c.nap_after_s,
                quiet=in_window(hour, c.quiet_hours),
                blocked=self._nap_blocked(dt, engaged),  # engaged holds off, never wakes
                held=c.presence_keeps_awake and self.present(),
            )
            if started:
                decisions.append("start")
            prev = self.model.state
            st = self.model.update(
                hour,
                dt,
                light=level,  # type: ignore[arg-type]
                presence_active=self.present(),
                nap=self.nap.napping,
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
        for reason in decisions:
            self.events("nap", reason)
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
                "reason": self.model.reason,
                "nap": {
                    "napping": self.nap.napping,
                    "after_s": c.nap_after_s,
                    "idle_s": round(self.nap.idle_s, 1),  # as of the last update
                    "naps": self.nap.naps,
                    "wakes": dict(self.nap.wakes),
                },
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
