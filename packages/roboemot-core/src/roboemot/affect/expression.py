"""Affect to motion: the expression renderer (SPEC 7, Expression map; SPEC 10, EXPRESSION).

An expression is data (``expression_map``: keyframes per channel in degrees). The renderer
starts one when the affect label changes or a labelled appraisal impulse fires, and plays it as
an additive offset at EXPRESSION priority, in radians:

    amplitude = register.motion_amplitude x intensity_factor[reaction_intensity]
                x (floor + (1 - floor) x affect intensity) x circadian factor

The register passed in is already persona-shaped (``resolve_register``), so the persona's
``motion_amplitude_x`` is not applied again. The product is capped by
``expression.max_amplitude``.

What stays true whatever the amplitude (SPEC 12):

- Offsets are clamped to the manifest's range about home, so the register's scaling can never
  ask the limiter for more than the hardware allows, and the minor cap, which the arbiter
  applies to the sum about home, always wins.
- Only angle dofs are driven: an expression never translates the robot. ``body_yaw`` is dropped
  while the register says ``approach_allowed`` is false, so no expression turns the body toward
  someone the robot must not approach.
- Each command lives ``expression.command_ttl_s``: when the renderer stops feeding it the offset
  is gone, and a new expression blends in from the current offset over the arbiter's crossfade.

Not scaled by speed: expressions play at the keyframe times.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from ..config.schema import ExpressionConfig, ExpressionEntry
from ..manifest import Dof
from ..motion.core import Command, Priority
from ..motion.curves import min_jerk
from ..motion.primitives import Keyframe, Player, Primitive
from . import AffectState
from .register import ActiveRegister

SOURCE = "expression"
BODY_CHANNEL = "body_yaw"


def _sample(frames: list[tuple[float, float]], t: float) -> float:
    """Piecewise-linear value of one channel's keyframes at ``t``."""
    for (t0, v0), (t1, v1) in zip(frames, frames[1:], strict=False):
        if t <= t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0))
    return frames[-1][1]


def build_primitive(
    label: str, entry: ExpressionEntry, dofs: Mapping[str, Dof], *, body: bool = True
) -> Primitive | None:
    """The entry as an additive primitive over the dofs this robot has (degrees to radians).

    Channels the robot lacks, translation dofs and, with ``body`` false, ``body_yaw`` are
    skipped; None when nothing is left. Keyframes of different channels are merged onto the
    union of their times and ``repeat`` unrolls the cycle.
    """
    use = {
        name: frames
        for name, frames in entry.channels.items()
        if name in dofs and dofs[name].kind == "angle" and (body or name != BODY_CHANNEL)
    }
    if not use:
        return None
    cycle = max(frames[-1][0] for frames in use.values())
    grid = sorted({t for frames in use.values() for t, _ in frames})
    frames = tuple(
        Keyframe(
            k * cycle + t,
            {n: math.radians(_sample(f, t)) for n, f in use.items()},
        )
        for k in range(entry.repeat)
        for t in grid
        if k == 0 or t > 0  # the end of one cycle is the start of the next: one keyframe
    )
    return Primitive(label, frames, method=entry.curve, additive=True)


class ExpressionRenderer:
    """Starts and plays expressions; call :meth:`update` with each affect state and
    :meth:`commands` every app tick.

    ``circadian_factor`` is a hook returning the circadian amplitude multiplier (1.0 until the
    circadian runner sets it). Single-threaded: both calls come from the app loop.
    """

    def __init__(
        self,
        cfg: ExpressionConfig,
        expression_map: Mapping[str, ExpressionEntry],
        dofs: Mapping[str, Dof],
        crossfade_s: float,
    ) -> None:
        self.dofs = dofs
        self.circadian_factor: Callable[[], float] = lambda: 1.0
        self.apply_config(cfg, expression_map, crossfade_s)
        self._label: str | None = None
        self._serial = 0
        self._pending = False
        self._last_start = -math.inf
        self._player: Player | None = None
        self._blend: tuple[float, dict[str, float]] | None = None
        self._last: dict[str, float] = {}
        self._minor_factor = 1.0  # safety.minor_motion_factor while a minor is present, else 1
        self._approach = False  # the register's latest approach_allowed: checked every tick
        self._scale_now = 0.0  # the register's latest motion factor
        self._scale_start = 0.0  # the same when the playing expression started
        self.last_label: str | None = None  # the label of the latest expression started
        self.last_amplitude = 0.0  # its amplitude, for tests and the debug report

    def apply_config(
        self,
        cfg: ExpressionConfig,
        expression_map: Mapping[str, ExpressionEntry],
        crossfade_s: float,
    ) -> None:
        """Hot reload: new parameters and map. An expression already playing finishes."""
        self.cfg, self.expression_map, self.crossfade_s = cfg, dict(expression_map), crossfade_s
        self._cache: dict[tuple[str, bool], Primitive | None] = {}

    def amplitude(self, register: ActiveRegister, state: AffectState) -> float:
        """The amplitude factor for ``state`` under ``register`` (see the module docstring)."""
        c = self.cfg
        reaction = getattr(c.intensity_factor, register.params.reaction_intensity)
        affect = c.intensity_floor + (1 - c.intensity_floor) * min(1.0, max(0.0, state.intensity))
        raw = register.params.motion_amplitude * reaction * affect * self.circadian_factor()
        return max(0.0, min(c.max_amplitude, raw))

    def set_register(self, register: ActiveRegister, minor_factor: float = 1.0) -> None:
        """Take the register in effect now; call it every app tick, whatever the affect rate.

        A stricter register (no approach, lower amplitude) acts on the playing expression at
        once. ``minor_factor`` is ``safety.minor_motion_factor`` while a minor is present (1.0
        otherwise): offsets are then kept inside that share of the range, on top of the cap the
        arbiter applies (defence in depth, SPEC 12).
        """
        self._approach = register.params.approach_allowed
        self._scale_now = self._register_scale(register)
        self._minor_factor = minor_factor

    def update(self, state: AffectState, register: ActiveRegister, now: float) -> bool:
        """Take a new affect state; starts an expression when one is due. Returns whether it did.

        Due: the label changed or a labelled impulse fired (held back until
        ``expression.min_interval_s`` after the last start), or the label has been held for
        ``retrigger_s`` with the affect intensity still at ``retrigger_min_intensity``.
        """
        self.set_register(register, self._minor_factor)
        if self._label is not None:
            changed = state.label != self._label or state.impulse != self._serial
            held = (
                now - self._last_start >= self.cfg.retrigger_s
                and state.intensity >= self.cfg.retrigger_min_intensity
            )
            self._pending = self._pending or changed or held
        self._label, self._serial = state.label, state.impulse
        if not self._pending or now - self._last_start < self.cfg.min_interval_s:
            return False
        self._pending = False
        return self._start(state, register, now)

    def _primitive(self, label: str, body: bool) -> Primitive | None:
        key = (label, body)
        if key not in self._cache:
            entry = self.expression_map.get(label)
            self._cache[key] = (
                build_primitive(label, entry, self.dofs, body=body) if entry else None
            )
        return self._cache[key]

    def _register_scale(self, register: ActiveRegister) -> float:
        """The register's share of the amplitude: motion amplitude x reaction factor."""
        p = register.params
        return p.motion_amplitude * getattr(self.cfg.intensity_factor, p.reaction_intensity)

    def _start(self, state: AffectState, register: ActiveRegister, now: float) -> bool:
        prim = self._primitive(state.label, register.params.approach_allowed)
        amp = self.amplitude(register, state)
        if prim is None or amp <= 0:
            return False
        if self._last and self.crossfade_s > 0:
            self._blend = (now, dict(self._last))
        self._player = Player(prim, now, source=SOURCE, amplitude=amp, priority=Priority.EXPRESSION)
        self._scale_start = self._scale_now
        self._last_start, self.last_label, self.last_amplitude = now, state.label, amp
        return True

    def commands(self, now: float) -> list[Command]:
        """The expression offset for this tick, to feed to the motion loop (empty when idle)."""
        if self._player is not None and self._player.done(now):
            self._player = None
        cmd = self._player.command(now) if self._player is not None else None
        out = dict(cmd.targets) if cmd is not None else {}
        if self._blend is not None:
            start, previous = self._blend
            s = (now - start) / self.crossfade_s if self.crossfade_s > 0 else 1.0
            if s >= 1.0:
                self._blend = None
            else:
                w = min_jerk(s)
                out = {
                    n: out.get(n, 0.0) * w + previous.get(n, 0.0) * (1 - w)
                    for n in out.keys() | previous.keys()
                }
        # The register can tighten while an expression plays (a child walks in): the body
        # channel goes at once and the amplitude follows the register down, never up.
        tighten = min(1.0, self._scale_now / self._scale_start) if self._scale_start > 0 else 1.0
        if self.last_amplitude > self.cfg.max_amplitude:  # the ceiling was lowered by a reload
            tighten = min(tighten, self.cfg.max_amplitude / self.last_amplitude)
        self._last = {
            n: self._clamp(n, v * tighten)
            for n, v in out.items()
            if self._approach or n != BODY_CHANNEL
        }
        if self._player is None and self._blend is None:
            self._last = {}
        if not self._last:
            return []
        return [
            Command(
                SOURCE,
                Priority.EXPRESSION,
                dict(self._last),
                now,
                expires=now + self.cfg.command_ttl_s,
                additive=True,
            )
        ]

    def _clamp(self, name: str, offset: float) -> float:
        dof = self.dofs[name]
        k = self._minor_factor
        return max((dof.min - dof.home) * k, min((dof.max - dof.home) * k, offset))

    @property
    def playing(self) -> bool:
        """Whether an expression is being fed right now."""
        return bool(self._last)


__all__ = ["SOURCE", "ExpressionRenderer", "build_primitive"]
