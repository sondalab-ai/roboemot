"""Ramps between the app tick's commands, on the motor loop's faster ticks (SPEC 10, Motor loop).

The app tick feeds a new command ``motion.app_tick_hz`` times a second and the motor loop runs
faster. With a high acceleration cap the limiter follows each step, so a slow move becomes stop
and go. This ramps a source's targets from where its last ramp stands to the new command over one
app tick, which delays those moves by at most that tick.

A command above COMMANDED priority (safety, reflex) is never ramped. Neither is the first command
of a source, nor one that arrives more than ``GAP_TICKS`` ramps after the previous one: the
arbiter's cross-fade covers those. Called under the motion loop's lock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .core import Command, Priority

GAP_TICKS = 2.0  # a pause longer than this many app ticks is a new start, not a late command


@dataclass
class _Ramp:
    """One source's command and where its ramp started at stamp ``t0``."""

    cmd: Command
    start: dict[str, float]
    t0: float
    done: bool = False

    def at(self, t: float, ramp_s: float) -> dict[str, float]:
        """The ramp's targets at ``t``, held at the command's targets once it ended."""
        share = min(1.0, max(0.0, (t - self.t0) / ramp_s))
        return {n: v + (self.cmd.targets[n] - v) * share for n, v in self.start.items()}


class CommandRamps:
    """Per-source ramps; ``ramp_s`` of 0 passes every command through unchanged."""

    def __init__(self, ramp_s: float = 0.0) -> None:
        self.ramp_s = ramp_s
        self._ramps: dict[str, _Ramp] = {}

    def admit(self, cmd: Command) -> Command:
        """The command to submit now for ``cmd``, and start its ramp when one applies."""
        if self.ramp_s <= 0 or cmd.priority < Priority.COMMANDED:
            self._ramps.pop(cmd.source, None)
            return cmd
        last = self._ramps.get(cmd.source)
        gap = None if last is None else cmd.stamp - last.cmd.stamp
        if last is None or gap is None or not 0 < gap <= GAP_TICKS * self.ramp_s:
            self._ramps[cmd.source] = _Ramp(cmd, dict(cmd.targets), cmd.stamp, done=True)
            return cmd
        standing = last.at(cmd.stamp, self.ramp_s)
        start = {n: standing.get(n, v) for n, v in cmd.targets.items()}
        self._ramps[cmd.source] = _Ramp(cmd, start, cmd.stamp)
        return replace(cmd, targets=start)

    def step(self, now: float) -> list[Command]:
        """The ramped commands to submit at ``now``; a ramp sends its last step once."""
        out: list[Command] = []
        for ramp in self._ramps.values():
            if ramp.done:
                continue
            ramp.done = now - ramp.t0 >= self.ramp_s
            out.append(replace(ramp.cmd, targets=ramp.at(now, self.ramp_s)))
        return out

    def clear(self, source: str) -> None:
        """Forget a source, so that a cleared command is not submitted again."""
        self._ramps.pop(source, None)


__all__ = ["CommandRamps", "GAP_TICKS"]
