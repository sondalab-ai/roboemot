"""Motion of the circadian states (SPEC 7A, Effects by state; SPEC 10, 12).

- **Drowsy:** an occasional yawn (appendages droop, head tilts), every ``circadian.yawn_interval_s``
  with jitter. The slower, smaller idle breathing is the app's ``idle_command`` scaled by the
  state's ``StateEffects``.
- **Resting:** the rest posture (``circadian_motion.rest_pose_deg``), a plain IDLE command.
  IDLE is the tier of idle breathing, scan and attention's gaze, so it replaces breathing and
  attention still takes the head when it engages. No idle motion in this state; the motors
  stay enabled and keep holding the pose.
- **Waking:** a stretch (``circadian_motion.stretch``, 3 s by default) when the robot leaves
  resting because lights came on, someone approached or the app woke it.

Yawn and stretch are additive offsets at EXPRESSION priority, from the same keyframe format and
renderer code as affect expressions. Nothing here bypasses the safety path: offsets are clamped
to the manifest range about home (inside the minor share while a minor is present), the arbiter
scales the sum by the minor factor and the limiter rate-limits the result. The pose and primitive
angles are starting points, not verified on hardware.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import replace

from ..affect.expression import build_primitive
from ..config.schema import CircadianConfig
from ..config.schema import CircadianMotion as CircadianMotionConfig
from ..manifest import Dof
from ..motion.core import Command, Priority
from ..motion.primitives import Player, Primitive
from .model import State

REST_SOURCE = "circadian-rest"
FX_SOURCE = "circadian-motion"


class CircadianMotion:
    """Produces the rest posture, yawns and the waking stretch as motion commands."""

    def __init__(
        self,
        cfg: CircadianConfig,
        motion: CircadianMotionConfig,
        dofs: Mapping[str, Dof],
        *,
        body_yaw: str | None,
        rng: random.Random | None = None,
    ) -> None:
        self.dofs = dofs
        self.body_yaw = body_yaw
        self.rng = rng or random.Random()  # noqa: S311 (jitter, not security)
        self.minor_factor = 1.0
        self._scale_now = 1.0  # the register's motion amplitude now
        self._scale_start = 1.0  # the same when the playing primitive started
        self._amp = 1.0  # amplitude the playing primitive started with
        self._player: Player | None = None
        self._playing = ""
        self._next_yawn: float | None = None
        self.apply_config(cfg, motion)

    def apply_config(self, cfg: CircadianConfig, motion: CircadianMotionConfig) -> None:
        """Hot reload: new parameters and primitives; a primitive already playing finishes."""
        self.cfg = cfg
        self.rest = {
            name: min(max(math.radians(deg), self.dofs[name].min), self.dofs[name].max)
            for name, deg in motion.rest_pose_deg.items()
            if name in self.dofs and self.dofs[name].kind == "angle"
        }
        self._yawn = self._build("yawn", motion.yawn)
        self._stretch = self._build("stretch", motion.stretch)

    def _build(self, name: str, entry: object) -> Primitive | None:
        return (
            build_primitive(name, entry, self.dofs, body_yaw=self.body_yaw, body=False)  # type: ignore[arg-type]
            if entry is not None
            else None
        )

    def set_register(self, motion_amplitude: float, max_amplitude: float) -> None:
        """The register's motion amplitude and ``expression.max_amplitude`` in effect now; call
        it every tick. Yawn and stretch start at ``min(max_amplitude, motion_amplitude)`` and,
        like an expression, follow a stricter register down while they play, never up."""
        self._scale_now = motion_amplitude
        self.max_amplitude = max_amplitude

    max_amplitude = 2.0

    def reset(self) -> None:
        """Stop any yawn or stretch and forget the yawn schedule (the rhythm was switched off)."""
        self._player, self._playing, self._next_yawn = None, "", None

    def set_minor_factor(self, factor: float) -> None:
        """``safety.minor_motion_factor`` while a minor is present, else 1.0 (SPEC 12)."""
        self.minor_factor = factor

    @property
    def playing(self) -> str:
        """The primitive playing now (``yawn`` or ``stretch``), or an empty string."""
        return self._playing if self._player is not None else ""

    def start_waking(self, now: float) -> bool:
        """Start the waking stretch (it cancels a yawn). False when the robot has no stretch."""
        if self._stretch is None:
            return False
        self._play("stretch", self._stretch, now)
        return True

    def _play(self, name: str, prim: Primitive, now: float) -> None:
        self._amp = max(0.0, min(self.max_amplitude, self._scale_now))
        self._scale_start = self._scale_now
        self._player = Player(
            prim, now, source=FX_SOURCE, amplitude=self._amp, priority=Priority.EXPRESSION
        )
        self._playing = name

    def _schedule_yawn(self, now: float) -> None:
        c = self.cfg
        self._next_yawn = now + c.yawn_interval_s * (1 + c.yawn_jitter * self.rng.uniform(-1, 1))

    def commands(self, now: float, state: State, attending: bool = False) -> list[Command]:
        """The commands for this tick: feed them, and clear ``REST_SOURCE`` when it is absent.

        ``attending``: attention owns the head, so the rest posture is withheld (attention
        wins) and no yawn starts.
        """
        out: list[Command] = []
        if self._player is not None and self._player.done(now):
            self._player = None
            self._playing = ""
        if state == "drowsy" and not attending and self._player is None and self._yawn:
            if self._next_yawn is None:
                self._schedule_yawn(now)
            elif now >= self._next_yawn:
                self._play("yawn", self._yawn, now)
                self._schedule_yawn(now + self._yawn.duration)
        elif state != "drowsy":
            self._next_yawn = None
        if state == "resting" and not attending and self.rest:
            out.append(
                Command(
                    REST_SOURCE,
                    Priority.IDLE,
                    dict(self.rest),
                    now,
                    expires=now + self.cfg.command_ttl_s,
                )
            )
        cmd = self._player.command(now) if self._player is not None else None
        if cmd is not None:
            tighten = (
                min(1.0, self._scale_now / self._scale_start) if self._scale_start > 0 else 1.0
            )
            if self._amp > self.max_amplitude:  # the ceiling was lowered by a reload
                tighten = min(tighten, self.max_amplitude / self._amp)
            targets = {n: self._clamp(n, v * tighten) for n, v in cmd.targets.items()}
            out.append(replace(cmd, targets=targets, expires=now + self.cfg.command_ttl_s))
        return out

    def _clamp(self, name: str, offset: float) -> float:
        dof = self.dofs[name]
        k = self.minor_factor
        return max((dof.min - dof.home) * k, min((dof.max - dof.home) * k, offset))
