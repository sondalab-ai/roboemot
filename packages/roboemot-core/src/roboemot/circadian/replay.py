"""Light-log replay for the circadian rhythm (SPEC 7A, Replay; SPEC 14, Endurance).

A light log is JSON Lines, one sample per line, normally one per minute (SPEC 7A has the field
list): ``t`` (ISO 8601 local wall-clock time, or epoch seconds), ``luma`` (0..1), optional
``noise``, ``exposure_s``, ``gain`` and ``presence``. The samples go through the same
:class:`LightSampler` and :class:`LightEstimator` as live frames (hysteresis, confirmation),
and the :class:`CircadianRunner` runs on a fake clock that jumps from sample to sample, so 24
hours replay in a fraction of a second and the result is deterministic.

``synthetic_day`` writes a plausible day (dawn, daylight, an evening lamp, lights off at night,
with noise) for tests and for the exit check of milestone M2.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..bus import Bus
from ..config.schema import RoboEmotConfig, StateEffects
from .model import MINUTES_PER_HOUR, SECONDS_PER_HOUR, CircadianState
from .runner import CircadianRunner

SECONDS_PER_MINUTE = 60.0
LOG_DIGITS = 4
SMOOTH_A = 3

# The synthetic scenario: fixed facts of a made-up household, not tunable robot behaviour.
SYNTH_START = "2026-03-02T08:00:00"
SYNTH_DAWN_START_H = 6.5  # the room starts to brighten
SYNTH_DAWN_END_H = 7.5  # full daylight
SYNTH_DUSK_START_H = 17.5  # daylight fades
SYNTH_LAMP_ON_H = 18.5  # the evening lamp is on from here
SYNTH_LIGHTS_OFF_H = 22.25  # the household switches the lights off
SYNTH_DAY_LUMA = 0.55
SYNTH_LAMP_LUMA = 0.25
SYNTH_NIGHT_LUMA = 0.03
SYNTH_LUMA_SD = 0.02
SYNTH_NOISE = 0.01
SYNTH_NIGHT_NOISE = 0.02
SYNTH_NOISE_SD = 0.003
SYNTH_SEED = 7
SYNTH_HOURS = 24.0


@dataclass(frozen=True)
class LightSample:
    """One logged sample: when, and what the camera saw."""

    when: datetime  # local wall-clock time
    luma: float
    noise: float = 0.0
    exposure_s: float | None = None
    gain: float | None = None
    presence: bool = False


def parse_time(value: str | float) -> datetime:
    """ISO 8601 text as local wall-clock time (a UTC offset in it is ignored: the log holds the
    hour the robot's clock showed), or a number as epoch seconds in the machine's time zone."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    return datetime.fromisoformat(value).replace(tzinfo=None)


def parse_line(line: str) -> LightSample:
    """One JSONL line; raises ``ValueError`` naming the problem (never echoes the content)."""
    try:
        raw = json.loads(line)
        sample = LightSample(
            parse_time(raw["t"]),
            float(raw["luma"]),
            float(raw.get("noise") or 0.0),
            None if raw.get("exposure_s") is None else float(raw["exposure_s"]),
            None if raw.get("gain") is None else float(raw["gain"]),
            bool(raw.get("presence", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad light log line ({type(exc).__name__})") from exc
    if not 0.0 <= sample.luma <= 1.0:
        raise ValueError("bad light log line: luma outside 0..1")
    return sample


def read_log(source: str | Path | Iterable[str]) -> list[LightSample]:
    """Samples of a JSONL file (or of its lines), blank lines skipped, in time order."""
    lines = Path(source).read_text().splitlines() if isinstance(source, (str, Path)) else source
    out = [parse_line(line) for line in lines if line.strip()]
    if any(b.when < a.when for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("light log is not in time order")
    return out


def format_line(s: LightSample) -> str:
    """The JSONL line of a sample."""
    row = {
        "t": s.when.isoformat(timespec="seconds"),
        "luma": round(s.luma, LOG_DIGITS),
        "noise": round(s.noise, LOG_DIGITS),
        "exposure_s": s.exposure_s,
        "gain": s.gain,
        "presence": s.presence,
    }
    return json.dumps(row)


def write_log(samples: Iterable[LightSample], path: str | Path) -> None:
    """Write a light log as JSONL."""
    Path(path).write_text("".join(format_line(s) + "\n" for s in samples))


def _smooth(x: float) -> float:
    """Smoothstep of x in [0, 1]."""
    x = min(1.0, max(0.0, x))
    return x * x * (SMOOTH_A - 2 * x)


def _clean_luma(hour: float) -> float:
    """The noise-free brightness of the synthetic room at ``hour``."""
    if hour < SYNTH_DAWN_START_H or hour >= SYNTH_LIGHTS_OFF_H:
        return SYNTH_NIGHT_LUMA
    if hour < SYNTH_DAWN_END_H:
        k = _smooth((hour - SYNTH_DAWN_START_H) / (SYNTH_DAWN_END_H - SYNTH_DAWN_START_H))
        return SYNTH_NIGHT_LUMA + (SYNTH_DAY_LUMA - SYNTH_NIGHT_LUMA) * k
    if hour < SYNTH_DUSK_START_H:
        return SYNTH_DAY_LUMA
    if hour < SYNTH_LAMP_ON_H:
        k = _smooth((hour - SYNTH_DUSK_START_H) / (SYNTH_LAMP_ON_H - SYNTH_DUSK_START_H))
        return SYNTH_DAY_LUMA + (SYNTH_LAMP_LUMA - SYNTH_DAY_LUMA) * k
    return SYNTH_LAMP_LUMA


def synthetic_day(
    start: str = SYNTH_START,
    hours: float = SYNTH_HOURS,
    *,
    seed: int = SYNTH_SEED,
    presence: Iterable[tuple[float, float]] = (),
) -> list[LightSample]:
    """One sample per minute: dawn, daylight, an evening lamp, lights off at night, with noise.

    ``presence`` lists (from_hour, to_hour) windows of the clock in which people are around.
    The samples start at ``start`` (ISO, local time) and run for ``hours``.
    """
    rng = random.Random(seed)  # noqa: S311 (a synthetic scenario, not security)
    t0 = parse_time(start)
    windows = list(presence)
    out = []
    for i in range(round(hours * MINUTES_PER_HOUR)):
        when = t0 + timedelta(minutes=i)
        hour = when.hour + when.minute / MINUTES_PER_HOUR
        luma = min(1.0, max(0.0, _clean_luma(hour) + rng.gauss(0.0, SYNTH_LUMA_SD)))
        dark = hour < SYNTH_DAWN_START_H or hour >= SYNTH_LIGHTS_OFF_H
        noise = max(
            0.0, (SYNTH_NIGHT_NOISE if dark else SYNTH_NOISE) + rng.gauss(0.0, SYNTH_NOISE_SD)
        )
        out.append(LightSample(when, luma, noise, presence=any(a <= hour < b for a, b in windows)))
    return out


class ReplayClock:
    """A fake :class:`Clock`: ``set`` jumps to a sample time (seconds since the first sample
    and the local hour)."""

    def __init__(self) -> None:
        self.t = 0.0
        self._hour = 0.0

    def monotonic(self) -> float:
        return self.t

    def hour(self) -> float:
        return self._hour

    def set(self, t: float, hour: float) -> None:
        self.t, self._hour = t, hour


@dataclass
class RecordingSink:
    """An affect sink that records what the runner sends (events and baseline changes)."""

    events: list[tuple[float, str]] = field(default_factory=list)
    effects: list[tuple[float, StateEffects]] = field(default_factory=list)
    now: float = 0.0

    def on_event(self, event: str, now: float, payload: dict[str, Any] | None = None) -> None:
        self.events.append((now, event))

    def set_circadian(self, effects: StateEffects) -> None:
        self.effects.append((self.now, effects))


@dataclass
class Point:
    t_s: float  # seconds since the first sample
    hour: float
    state: str
    light: str | None
    sleepiness: float


@dataclass
class ReplayResult:
    timeline: list[Point]
    transitions: list[tuple[float, float, str, str]]  # (t_s, hour, from, to)
    events: list[tuple[float, str]]  # (t_s, event) as sent to the affect engine
    runner: CircadianRunner
    final: CircadianState | None

    def first_hour(self, state: str, after_h: float = 0.0) -> float | None:
        """The hour of day of the first transition into ``state`` at least ``after_h`` hours
        after the start, or None."""
        for t, hour, _, to in self.transitions:
            if to == state and t >= after_h * SECONDS_PER_HOUR:
                return hour
        return None

    def states(self) -> list[str]:
        """The distinct states in order of appearance, consecutive repeats collapsed."""
        out: list[str] = []
        for p in self.timeline:
            if not out or out[-1] != p.state:
                out.append(p.state)
        return out


def replay(
    samples: Iterable[LightSample],
    cfg: RoboEmotConfig | None = None,
    *,
    sink: RecordingSink | None = None,
) -> ReplayResult:
    """Run a :class:`CircadianRunner` over ``samples`` at accelerated virtual time.

    Each sample is fed to the sampler (so the estimator's hysteresis and the confirmation
    apply) and the runner is ticked once at the sample's time. Needs no camera, models or
    threads; ``cfg.circadian.light_source`` ``clock_only`` ignores the light, as live.
    """
    cfg = cfg or RoboEmotConfig()
    sink = sink or RecordingSink()
    clock = ReplayClock()
    runner = CircadianRunner(cfg, Bus(), sink, clock=clock)
    timeline: list[Point] = []
    transitions: list[tuple[float, float, str, str]] = []
    first: datetime | None = None
    prev = runner.model.state
    final = None
    for s in samples:
        first = first or s.when
        t = (s.when - first).total_seconds()
        hour = s.when.hour + s.when.minute / MINUTES_PER_HOUR + s.when.second / SECONDS_PER_HOUR
        clock.set(t, hour)
        sink.now = t
        if s.presence:
            runner.note_presence()
        runner.sampler.offer_stats(s.luma, s.noise, exposure_s=s.exposure_s, gain=s.gain)
        st = runner.tick(t)
        if st is None:
            continue
        final = st
        timeline.append(Point(t, hour, st.state, st.light, st.sleepiness))
        if st.state != prev:
            transitions.append((t, hour, prev, st.state))
            prev = st.state
    runner.close()
    return ReplayResult(timeline, transitions, list(sink.events), runner, final)
