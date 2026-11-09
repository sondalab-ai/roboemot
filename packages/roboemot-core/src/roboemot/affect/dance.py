"""Dance motion: small moves locked to a beat (SPEC 7, Dance motion).

One additive offset at EXPRESSION priority, fed from the app tick right after the speaking motion.
Roles, bound to channels by the manifest's ``dance_motion_map`` (core names no joint):

- ``bob``: one dip on the beat, min-jerk;
- ``sway``: one side per bar (``music.beats_per_bar``), slow;
- ``accent``: a flick on an accent stamp (a blirp's event), at most one per
  ``accent_min_interval_s``, peaking on the stamp.

A move half never lasts less than ``min_half_s``: a tempo too fast for a full bob moves on every
second, then every fourth beat. The whole layer reads the beat ``lead_ms`` ahead.

Amplitude, re-read every tick::

    role degrees x register.motion_amplitude x circadian factor x (1 + arousal_gain x arousal)
                 x beat strength

The product of the factors is capped by ``expression.max_amplitude``. Roles that share a channel
add up, so a channel's sum is capped too, at that factor times its largest role's degrees. Each
offset is then clamped to the manifest range about home. ``bob`` and ``sway`` ramp in over
``fade_in_s`` once the beat is there and out over ``fade_out_s`` when it goes; a beat that returns
during the fade-out fades in again while the old offsets finish fading (an accent has its own
envelope, which starts and ends at zero). A disabled layer, a circadian factor of 0 or the resting
state ramp the offsets to zero with no beat read. A channel that the playing expression or the
speaking motion drives leaves the dance over the speaking motion's ``fade_ms``, so the two never
add up for long. Never moves the body yaw or a translation: a map naming one is refused at load
and by :meth:`DanceRenderer.apply_config`.

Threads: the app tick calls everything except :meth:`DanceRenderer.apply_config`, which the config
reload calls, and :meth:`DanceRenderer.live_snapshot`, which the web thread calls. Each swaps or
reads one immutable object by assignment: no lock, and each tick reads the settings bundle once.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from ..config.schema import (
    SECONDS_PER_MINUTE,
    DanceConfig,
    DanceMotionMap,
    ExpressionEntry,
)
from ..manifest import Dof
from ..motion.beat import Beat, BeatSource
from ..motion.core import Command, Priority
from ..motion.curves import min_jerk
from ..motion.primitives import Primitive
from . import AffectState
from .expression import build_primitive
from .register import ActiveRegister

SOURCE = "dance"
MS_PER_S = 1000.0
GAIN_EPSILON = 1e-9  # a fade gain below this is zero (float rounding of the per-tick steps)
FOLD_FACTORS = (1, 2, 4)  # one move per this many beats: full, half and quarter time


@dataclass(frozen=True)
class _Shape:
    """The accent's primitive (radians of the map's shape), its largest offset and when."""

    prim: Primitive
    peak: float
    peak_t: float


@dataclass(frozen=True)
class _Settings:
    """Everything :meth:`DanceRenderer.apply_config` sets, swapped as one object."""

    cfg: DanceConfig
    max_amplitude: float
    beats_per_bar: int
    bob: dict[str, int]
    sway: dict[str, int]
    accent: _Shape | None
    ceilings: dict[str, float]  # per channel: the cap of the summed offset, radians


@dataclass(frozen=True)
class DanceLive:
    """The Live tab's view of one tick (SPEC 7, 12), swapped whole. ``active``: the offsets are
    not back at zero yet. ``beat``: the beat followed now (None: no beat, also during the
    fade-out). ``bob`` and ``sway``: each role's envelope as a share of its degrees in [0, 1],
    after the fade and the yield to other layers, 0 for a role that rests; not the waveform,
    which a poll would only alias. The beat's phase and the two stamps are on the monotonic
    clock and stay in the process."""

    active: bool = False
    beat: Beat | None = None
    bob: float = 0.0
    sway: float = 0.0
    last_accent_s: float = -math.inf
    stamp: float = -math.inf

    @property
    def source(self) -> str | None:
        """Which provider the beat comes from, None with no beat."""
        return self.beat.source if self.beat is not None else None

    def tempo_bpm(self) -> float | None:
        """The tempo of the beat followed, None with no beat or no period."""
        beat = self.beat
        if beat is None or beat.period_s <= 0:
            return None
        return SECONDS_PER_MINUTE / beat.period_s

    def beat_age_s(self, now: float) -> float | None:
        """Seconds since the last beat of the grid at ``now``, in [0, period); None with no
        beat or no period."""
        beat = self.beat
        if beat is None or beat.period_s <= 0:
            return None
        return (now - beat.phase_s) % beat.period_s

    def last_accent_age_s(self, now: float) -> float | None:
        """Seconds since the last accent played, None before the first. At least 0: an accent
        is found ahead of the clock so that it can peak on its stamp."""
        return None if math.isinf(self.last_accent_s) else max(0.0, now - self.last_accent_s)


@dataclass
class _Play:
    """The accent playing, on the lead clock; ``gain`` is radians per unit of the shape."""

    shape: _Shape
    start: float
    gain: float

    def done(self, t: float) -> bool:
        """Whether the flick has played to its end at lead time ``t``."""
        return t - self.start >= self.shape.prim.duration

    def offsets(self, t: float) -> dict[str, float]:
        """The flick's offset per channel at lead time ``t``, in radians, already scaled."""
        return {n: v * self.gain for n, v in self.shape.prim.sample(t - self.start).items()}


@dataclass
class _State:
    """Everything that is reset once the offsets are back at zero."""

    gain: float = 0.0
    frozen: dict[str, float] | None = None
    raw: dict[str, float] = field(default_factory=dict)
    tail: dict[str, float] = field(default_factory=dict)  # still fading after a re-arm
    tail_gain: float = 0.0
    duck: dict[str, float] = field(default_factory=dict)
    started: bool = False
    bob: float = 0.0  # share of the role's degrees before the fade, for the Live tab
    sway: float = 0.0


def fold(period_s: float, min_half_s: float) -> int | None:
    """How many beats one move spans so that each of its two halves lasts ``min_half_s``; None
    when even the longest fold is too short (the role then rests)."""
    for factor in FOLD_FACTORS:
        if factor * period_s >= 2 * min_half_s:
            return factor
    return None


class DanceRenderer:
    """Dance motion (see the module docstring). Call :meth:`set_register` every app tick,
    :meth:`set_affect` with each new affect state and :meth:`commands` every app tick.

    ``beat`` is read every tick; ``speaking_channels`` returns the channels the speaking motion
    emitted on its last tick and ``ducked`` those of the playing expression. ``yield_fade_s`` is
    how fast such a channel leaves the dance (the speaking motion's fade, read live).
    ``circadian_factor`` and ``resting`` are hooks the assembly sets.
    """

    def __init__(
        self,
        cfg: DanceConfig,
        dance_map: DanceMotionMap,
        dofs: Mapping[str, Dof],
        *,
        body_yaw: str | None,
        max_amplitude: float,
        beats_per_bar: int,
        beat: BeatSource,
        yield_fade_s: Callable[[], float],
        speaking_channels: Callable[[], frozenset[str]] = frozenset,
        ducked: Callable[[], frozenset[str]] = frozenset,
    ) -> None:
        self.dofs, self.body_yaw = dofs, body_yaw
        self.beat, self.speaking_channels, self.ducked = beat, speaking_channels, ducked
        self.yield_fade_s = yield_fade_s
        self.circadian_factor: Callable[[], float] = lambda: 1.0
        self.resting: Callable[[], bool] = lambda: False
        self._motion_amplitude = 0.0  # no motion before the first register
        self._arousal = 0.0
        self._last_now: float | None = None
        self._accent_since: float | None = None
        self._last_accent = -math.inf
        self._play: _Play | None = None
        self._sent: frozenset[str] = frozenset()  # channels of the last command sent
        self._s = _State()
        self._source: str | None = None
        self._live = DanceLive()
        self.counts = {"started": 0, "accent": 0, "skipped": 0}
        self.apply_config(cfg, dance_map, max_amplitude, beats_per_bar)

    # ------------------------------------------------------------------ config and inputs

    def apply_config(
        self,
        cfg: DanceConfig,
        dance_map: DanceMotionMap,
        max_amplitude: float,
        beats_per_bar: int,
    ) -> None:
        """Hot reload, from any thread: every field applies from the next tick. ValueError, and
        the old settings kept, when the map names the body yaw or a translation."""
        for role, names in dance_map.channels().items():
            for name in names:
                dof = self.dofs.get(name)
                if name == self.body_yaw or (dof is not None and dof.kind != "angle"):
                    raise ValueError(f"dance_motion_map.{role}: {name!r} may not move")
        bob, sway = self._signs(dance_map.bob), self._signs(dance_map.sway)
        accent = self._shape(dance_map.accent) if dance_map.accent is not None else None
        accent_names = accent.prim.keyframes[0].values if accent is not None else {}
        degrees: dict[str, float] = {}
        for deg, names in (
            (cfg.bob_deg, bob),
            (cfg.sway_deg, sway),
            (cfg.accent_deg, accent_names),
        ):
            for name in names:
                degrees[name] = max(degrees.get(name, 0.0), deg)
        ceilings = {name: math.radians(deg) * max_amplitude for name, deg in degrees.items()}
        self._b = _Settings(cfg, max_amplitude, beats_per_bar, bob, sway, accent, ceilings)

    @property
    def cfg(self) -> DanceConfig:
        """The dance settings in effect (the last :meth:`apply_config`)."""
        return self._b.cfg

    def set_register(self, register: ActiveRegister) -> None:
        """Take the register in effect now; it scales the very next offsets."""
        self._motion_amplitude = register.params.motion_amplitude

    def set_affect(self, state: AffectState) -> None:
        """Take the latest affect state (its arousal scales the amplitude)."""
        self._arousal = state.arousal

    def _signs(self, raw: Mapping[str, int]) -> dict[str, int]:
        """The channel-to-sign map without the channels this robot lacks or cannot rotate."""
        return {n: s for n, s in raw.items() if n in self.dofs and self.dofs[n].kind == "angle"}

    def _shape(self, entry: ExpressionEntry) -> _Shape | None:
        """The accent entry as a primitive with its peak, or None when no channel of it exists
        on this robot or its keyframes never leave zero (the role is then off)."""
        prim = build_primitive("accent", entry, self.dofs, body_yaw=self.body_yaw, body=False)
        if prim is None:
            return None
        k = max(prim.keyframes, key=lambda f: max(abs(v) for v in f.values.values()))
        peak = max(abs(v) for v in k.values.values())
        return _Shape(prim, peak, k.t) if peak > 0 else None

    def scale(self, strength: float, b: _Settings | None = None) -> float:
        """The amplitude factor now: register x circadian x arousal x strength, capped."""
        b = b or self._b
        arousal = max(0.0, 1 + b.cfg.arousal_gain * self._arousal)
        raw = self._motion_amplitude * self.circadian_factor() * arousal * strength
        return max(0.0, min(b.max_amplitude, raw))

    # ------------------------------------------------------------------ the tick

    def commands(self, now: float) -> list[Command]:
        """The dance offset for this tick, to feed to the motion loop (empty when idle)."""
        b = self._b
        cfg = b.cfg
        dt = self._step_clock(now, cfg)
        s = self._s
        t = now + cfg.lead_ms / MS_PER_S
        moving = cfg.enabled and not self.resting() and self.scale(1.0, b) > 0
        beat = self.beat.beat_at(now) if moving else None
        self._source = beat.source if beat is not None else None
        if beat is None and s.frozen is None:
            s.frozen = dict(s.raw)
        elif beat is not None and s.frozen is not None:
            self._rearm(s)
        raw = s.frozen if s.frozen is not None else self._render(t, beat, b)
        s.raw = raw
        target = 0.0 if s.frozen is not None else 1.0
        up, down = dt / cfg.fade_in_s, dt / cfg.fade_out_s
        s.gain = min(target, s.gain + up) if target > s.gain else max(target, s.gain - down)
        s.tail_gain = max(0.0, s.tail_gain - down)
        if s.gain > GAIN_EPSILON and not s.started:
            s.started = True
            self.counts["started"] += 1
        accent = self._accent_offsets(t, b, moving)
        if s.gain <= GAIN_EPSILON and s.tail_gain <= GAIN_EPSILON and not accent:
            s = self._s = _State()
            self._publish(now, beat, s, b, False)
            return self._emit(now, cfg, {})  # a last zero for every channel still held
        out = self._mix(raw, accent, s, b, dt)  # first: it moves the ducks the levels read
        self._publish(now, beat, s, b, True)
        return self._emit(now, cfg, out)

    def _publish(
        self, now: float, beat: Beat | None, s: _State, b: _Settings, active: bool
    ) -> None:
        """Swap in the Live tab's view of this tick: the only work the tab adds to the tick.
        Each level goes through the same fade curve as the offsets, and the role's most ducked
        channel, so a bar shows what the robot does."""
        w = min_jerk(s.gain)
        bob = s.bob * w * min((s.duck.get(n, 1.0) for n in b.bob), default=0.0)
        sway = s.sway * w * min((s.duck.get(n, 1.0) for n in b.sway), default=0.0)
        self._live = DanceLive(active, beat, bob, sway, self._last_accent, now)

    def _step_clock(self, now: float, cfg: DanceConfig) -> float:
        """Seconds since the last tick. A gap longer than the command's lifetime (the app tick
        stopped, for example while asleep) restarts the layer: it fades in again, never a jump."""
        last, self._last_now = self._last_now, now
        if last is None:
            return 0.0
        if now - last > cfg.command_ttl_s:
            self._s = _State()
            self._play, self._accent_since = None, None
            self._sent = frozenset()
            return 0.0
        return max(0.0, now - last)

    @staticmethod
    def _rearm(s: _State) -> None:
        """The beat came back during the fade-out: what is on the channels now keeps fading as
        the tail, and the beat fades in from zero, so the swap has no step."""
        frozen = s.frozen or {}
        w, tail_w = min_jerk(s.gain), min_jerk(s.tail_gain)
        s.tail = {
            n: frozen.get(n, 0.0) * w + s.tail.get(n, 0.0) * tail_w
            for n in frozen.keys() | s.tail.keys()
        }
        s.tail_gain, s.gain, s.frozen = 1.0, 0.0, None

    def _mix(
        self,
        raw: dict[str, float],
        accent: dict[str, float],
        s: _State,
        b: _Settings,
        dt: float,
    ) -> dict[str, float]:
        """The beat offsets through the fade, the tail and the accent: each channel's sum under
        its ceiling, then through its duck. A channel another layer drives leaves over
        ``yield_fade_s`` and comes back the same way; one already yielded starts at zero."""
        ducked = self.ducked() | self.speaking_channels()
        fade_s = self.yield_fade_s()
        step = dt / fade_s if fade_s > 0 else 1.0
        w, tail_w = min_jerk(s.gain), min_jerk(s.tail_gain)
        out: dict[str, float] = {}
        for n in raw.keys() | accent.keys() | s.tail.keys():
            d = s.duck.get(n, 0.0 if n in ducked else 1.0)
            d = max(0.0, d - step) if n in ducked else min(1.0, d + step)
            s.duck[n] = d
            total = raw.get(n, 0.0) * w + s.tail.get(n, 0.0) * tail_w + accent.get(n, 0.0)
            ceiling = b.ceilings.get(n, math.inf)  # a reload may have dropped the channel
            out[n] = self._clamp(n, max(-ceiling, min(ceiling, total)) * d)
        return out

    def _emit(self, now: float, cfg: DanceConfig, out: dict[str, float]) -> list[Command]:
        """The command for ``out``, with 0 for a channel the previous command set and this one
        does not: the arbiter keeps a source's last command until it expires."""
        targets = dict.fromkeys(self._sent - set(out), 0.0) | out
        self._sent = frozenset(out)
        if not targets:
            return []
        return [
            Command(
                SOURCE,
                Priority.EXPRESSION,
                targets,
                now,
                expires=now + cfg.command_ttl_s,
                additive=True,
            )
        ]

    def _render(self, t: float, beat: Beat | None, b: _Settings) -> dict[str, float]:
        """The ``bob`` and ``sway`` offsets at lead time ``t`` before the fade."""
        if beat is None:
            return {}
        k, cfg, s = self.scale(beat.strength, b), b.cfg, self._s
        share = min(1.0, k)
        raw: dict[str, float] = {}
        factor = fold(beat.period_s, cfg.min_half_s)
        bob = math.radians(cfg.bob_deg) * self._bob(t, beat, factor) * k
        for n, sign in b.bob.items():
            raw[n] = raw.get(n, 0.0) + sign * bob
        s.bob = share if b.bob and factor is not None and cfg.bob_deg > 0 else 0.0
        bar = beat.period_s * b.beats_per_bar
        factor = fold(bar, cfg.min_half_s)
        sway = math.radians(cfg.sway_deg) * self._sway(t, bar, beat.phase_s, factor) * k
        for n, sign in b.sway.items():
            raw[n] = raw.get(n, 0.0) + sign * sway
        s.sway = share if b.sway and factor is not None and cfg.sway_deg > 0 else 0.0
        return raw

    @staticmethod
    def _bob(t: float, beat: Beat, factor: int | None) -> float:
        """Depth of the dip in [0, 1]: 1 on the beat, 0 halfway between two moves. ``factor``
        is the fold of the beat; None rests."""
        if factor is None:
            return 0.0
        p = ((t - beat.phase_s) / (factor * beat.period_s)) % 1.0
        return min_jerk(abs(2 * p - 1))

    @staticmethod
    def _sway(t: float, bar: float, phase_s: float, factor: int | None) -> float:
        """Sway in [-1, 1]: out and back on one side during a bar, the other side the next.
        ``factor`` is the fold of the bar; None rests."""
        if factor is None:
            return 0.0
        q = ((t - phase_s) / (factor * bar)) % 2.0
        side = 1.0 if q < 1.0 else -1.0
        return side * min_jerk(1 - abs(2 * (q % 1.0) - 1))

    # ------------------------------------------------------------------ accent

    def _accent_offsets(self, t: float, b: _Settings, moving: bool) -> dict[str, float]:
        """The accent's offsets at ``t``, after starting one from a new stamp. Not ``moving``
        (disabled, resting, a factor of 0) starts none, but the one playing finishes: its shape
        returns to zero by itself within its length, so it never steps."""
        if moving and b.accent is not None:
            horizon = t + b.accent.peak_t
            since = self._accent_since if self._accent_since is not None else horizon
            self._accent_since = horizon
            for stamp in self.beat.accents(since, horizon):
                self._start_accent(stamp, t, b.accent, b)
        else:
            self._accent_since = None
        play = self._play
        if play is None:
            return {}
        if play.done(t):
            self._play = None
            return {}
        return play.offsets(t)

    def _start_accent(self, stamp: float, t: float, shape: _Shape, b: _Settings) -> None:
        """Start the flick for ``stamp`` unless the rate limit or a flick in progress says no.
        It starts at ``t``, from the shape's zero, never mid-shape: it peaks within one tick
        after the stamp, or later when a stalled tick found the stamp late."""
        cfg = b.cfg
        if stamp - self._last_accent < cfg.accent_min_interval_s or cfg.accent_deg <= 0:
            return
        if self._play is not None:
            self.counts["skipped"] += 1
            return
        self._last_accent = stamp
        gain = math.radians(cfg.accent_deg) / shape.peak * self.scale(1.0, b)
        self._play = _Play(shape, t, gain)
        self.counts["accent"] += 1

    # ------------------------------------------------------------------ output

    def _clamp(self, name: str, offset: float) -> float:
        """``offset`` limited to the manifest range of channel ``name`` about its home."""
        dof = self.dofs[name]
        return max(dof.min - dof.home, min(dof.max - dof.home, offset))

    def snapshot(self) -> dict[str, int | str | None]:
        """For the debug report (SPEC 12A): dance starts, accents played and skipped, and which
        beat source plays now (None: no beat). Counts and a label only."""
        return {**self.counts, "source": self._source}

    def live_snapshot(self, now: float) -> DanceLive:
        """The last tick's view for the Live tab (any thread: one read, no lock). A tick older
        than ``command_ttl_s`` (the app tick stopped, for example while asleep) means the
        offsets have expired: idle, with the accent stamp kept."""
        live = self._live
        if now - live.stamp <= self._b.cfg.command_ttl_s:
            return live
        return replace(live, active=False, beat=None, bob=0.0, sway=0.0)


__all__ = ["SOURCE", "DanceLive", "DanceRenderer", "fold"]
