"""Two-process circadian model (SPEC 7A, Model), after Borbély.

- Process C: alertness oscillator, ``c_amplitude * cos(2π (h - peak - shift) / period)``.
- Process S: sleep pressure in [0, 1], rising toward 1 while not resting
  (``s_rise_tau_h``) and decaying toward 0 while resting (``s_decay_tau_h``).
- Sleepiness = (S - C + a) / (1 + 2a), in [0, 1].
- Light entrainment: light that contradicts the subjective day/night shifts the phase toward
  the observed pattern, at most ``max_shift_h_per_day`` per day.
- States awake / drowsy / resting with hysteresis; quiet hours force resting; people present
  and active prevent resting; ``hold`` and ``wake`` let the app or the wake word override.

Time is passed in explicitly (local hour of day and elapsed seconds), so the model is
deterministic and can be replayed at accelerated time (SPEC 14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..config.schema import CircadianConfig, StateEffects

State = Literal["awake", "drowsy", "resting"]
Light = Literal["dark", "dim", "bright"]
SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0


@dataclass
class CircadianState:
    hour: float
    phase_shift_h: float
    c: float
    s: float
    sleepiness: float
    state: State
    light: Light | None


def _hhmm(text: str) -> float:
    h, m = text.split(":")
    return int(h) + int(m) / MINUTES_PER_HOUR


def in_window(hour: float, window: tuple[str, str]) -> bool:
    start, end = _hhmm(window[0]), _hhmm(window[1])
    return start <= hour < end if start <= end else (hour >= start or hour < end)


class CircadianModel:
    def __init__(self, cfg: CircadianConfig, *, phase_shift_h: float = 0.0):
        self.cfg = cfg
        self.s = cfg.initial_sleep_pressure
        self.phase_shift_h = phase_shift_h
        self.state: State = "awake"
        self._hold: tuple[State, float] | None = None  # (state, until elapsed seconds)
        self._elapsed = 0.0
        # Unwrapped wall-clock hours: process C runs on it, so a period other than 24 h has no
        # jump at midnight, and it never drifts away from the wall clock.
        self.t_h: float | None = None
        self.last: CircadianState | None = None

    # ---- processes

    def process_c(self, t_h: float) -> float:
        """Alertness at internal time ``t_h`` (hours)."""
        cfg = self.cfg
        return cfg.c_amplitude * math.cos(
            2 * math.pi * (t_h - cfg.peak_hour - self.phase_shift_h) / cfg.period_h
        )

    def sleepiness(self, t_h: float) -> float:
        a = self.cfg.c_amplitude
        return max(0.0, min(1.0, (self.s - self.process_c(t_h) + a) / (1 + 2 * a)))

    # ---- inputs

    def hold(self, state: State, seconds: float) -> None:
        """App override, e.g. ``hold("awake", 1800)`` during a party (SPEC 7A API)."""
        self._hold = (state, self._elapsed + seconds)

    def wake(self, seconds: float | None = None) -> None:
        """Wake word or direct address while resting: awake for ``wake_hold_s``; C is not
        reset (SPEC 7A, Waking)."""
        self.hold("awake", self.cfg.wake_hold_s if seconds is None else seconds)

    def update(
        self, hour: float, dt_s: float, *, light: Light | None = None, presence_active: bool = False
    ) -> CircadianState:
        cfg = self.cfg
        self._elapsed += dt_s
        dt_h = dt_s / SECONDS_PER_HOUR
        if self.t_h is None:
            self.t_h = hour
        else:
            # Unwrapped wall-clock hour: continuous across midnight, and re-anchored to the
            # wall clock every update so DST, NTP corrections or imprecise dt never make
            # process C drift from the quiet hours.
            predicted = self.t_h + dt_h
            self.t_h = hour + HOURS_PER_DAY * round((predicted - hour) / HOURS_PER_DAY)
        if self.state == "resting":
            self.s *= math.exp(-dt_h / cfg.s_decay_tau_h)
        else:
            self.s = 1.0 - (1.0 - self.s) * math.exp(-dt_h / cfg.s_rise_tau_h)

        if light is not None and cfg.light_source == "camera":
            self._entrain(self.t_h, light, dt_h)

        sl = self.sleepiness(self.t_h)
        self.state = self._next_state(hour, sl, light, presence_active)
        self.last = CircadianState(
            hour, self.phase_shift_h, self.process_c(self.t_h), self.s, sl, self.state, light
        )
        return self.last

    def effects(self) -> StateEffects:
        return self.cfg.states[self.state]

    # ---- internals

    def _entrain(self, t_h: float, light: Light, dt_h: float) -> None:
        """Phase response: light in the subjective evening/night delays the rhythm, light in
        the subjective late night/morning advances it; darkness in the subjective day does the
        opposite. Rate bounded by max_shift_h_per_day."""
        c = self.process_c(t_h)
        subjective_day = c > 0
        if (light == "bright") == subjective_day or light == "dim":
            return  # light agrees with the rhythm
        # Hours since the alertness peak, wrapped to (-period/2, period/2].
        since_peak = (t_h - self.cfg.peak_hour - self.phase_shift_h + self.cfg.period_h / 2) % (
            self.cfg.period_h
        ) - self.cfg.period_h / 2
        direction = 1.0 if since_peak > 0 else -1.0  # after the peak: delay; before: advance
        if light == "dark":
            direction = -direction
        rate = self.cfg.max_shift_h_per_day / HOURS_PER_DAY
        self.phase_shift_h += direction * rate * dt_h
        half = self.cfg.period_h / 2
        self.phase_shift_h = (self.phase_shift_h + half) % self.cfg.period_h - half

    def _next_state(
        self, hour: float, sl: float, light: Light | None, presence_active: bool
    ) -> State:
        cfg = self.cfg
        if self._hold and self._elapsed < self._hold[1]:
            return self._hold[0]
        self._hold = None
        keep_awake = presence_active and cfg.presence_keeps_awake
        if in_window(hour, cfg.quiet_hours) and not keep_awake:
            return "resting"
        h = cfg.hysteresis
        st = self.state
        if st == "awake" and sl > cfg.drowsy_threshold + h:
            st = "drowsy"
        if st == "drowsy" and sl < cfg.drowsy_threshold - h:
            st = "awake"
        if (
            st == "drowsy"
            and sl > cfg.resting_threshold + h
            and light != "bright"
            and not keep_awake
        ):
            st = "resting"
        if st == "resting" and (sl < cfg.resting_threshold - h or light == "bright" or keep_awake):
            st = "drowsy"
        return st
