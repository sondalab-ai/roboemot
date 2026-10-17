#!/usr/bin/env python3
"""Generate golden vectors from the Python motion reference for the C++ core (rt/tests).

Format (one record per line):
  dof <name> <min> <max> <max_vel> <max_acc> <home>
  yaw_delta <a> <b> <max_rad>
  config <crossfade_s> <watchdog_s> <minor_factor> <watchdog_return_factor> <margin> <dt>
  tick <now> <minor 0|1>
  cmd <source> <priority> <additive 0|1> <stamp> <expires|-1> <n> (<dof> <value>)*n
  expect <n> (<dof> <value>)*n
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from roboemot.manifest import Constraint, Dof
from roboemot.motion import Command, MotionCore, Priority

DOFS = {
    "head.yaw": Dof("head.yaw", "head", -math.pi, math.pi, 3.0, 20.0),
    "head.pitch": Dof("head.pitch", "head", -0.7, 0.7, 2.0, 15.0),
    "body_yaw": Dof("body_yaw", "body_yaw", -2.79, 2.79, 1.5, 6.0),
    "antenna_right": Dof("antenna_right", "antennas", -3.05, 3.05, 6.0, 40.0, -0.1745),
}
CONSTRAINTS = [Constraint(type="yaw_delta", a="head.yaw", b="body_yaw", max_deg=65.0)]
CFG = {
    "crossfade_s": 0.15,
    "watchdog_s": 0.2,
    "minor_factor": 0.6,
    "watchdog_return_factor": 0.3,
    "constraint_margin": math.radians(3.0),
}
DT = 0.01


def script(tick: int, now: float) -> tuple[list[Command], bool]:
    """Scenario: commanded move, expression on top, reflex pre-emption with cross-fade,
    yaw-delta conflict, minor present, expiry, then silence to trigger the watchdog."""
    cmds: list[Command] = []
    minor = 150 <= tick < 250 or 300 <= tick < 400
    if tick < 300:
        cmds.append(Command("idle", Priority.IDLE, {"head.pitch": 0.02 * math.sin(now)}, now))
        cmds.append(
            Command(
                "app",
                Priority.COMMANDED,
                {"head.yaw": 1.6, "body_yaw": 0.2, "antenna_right": 1.0},
                now,
            )
        )
        cmds.append(
            Command(
                "expr",
                Priority.EXPRESSION,
                {"head.pitch": 0.1, "antenna_right": 0.3},
                now,
                additive=True,
            )
        )
    if 60 <= tick < 120:
        cmds.append(
            Command(
                "startle",
                Priority.REFLEX,
                {"head.yaw": -0.5, "head.pitch": -0.3},
                now,
                expires=now + 0.05,
            )
        )
    if 300 <= tick < 400:  # winner near its limit plus an additive overflow, minor on (SPEC 12)
        cmds.append(Command("hold", Priority.COMMANDED, {"head.pitch": 0.7}, now))
        cmds.append(Command("bump", Priority.EXPRESSION, {"head.pitch": 0.2}, now, additive=True))
    if tick == 200:
        cmds.append(Command("prim", Priority.COMMANDED, {"body_yaw": -1.0}, now, expires=now + 0.4))
    return cmds, minor


def main(out: Path) -> None:
    core = MotionCore(DOFS, CONSTRAINTS, **CFG)
    lines = [
        f"dof {d.name} {d.min!r} {d.max!r} {d.max_vel!r} {d.max_acc!r} {d.home!r}"
        for d in DOFS.values()
    ]
    lines += [f"yaw_delta {c.a} {c.b} {math.radians(c.max_deg)!r}" for c in CONSTRAINTS]
    lines.append(
        f"config {CFG['crossfade_s']!r} {CFG['watchdog_s']!r} {CFG['minor_factor']!r} "
        f"{CFG['watchdog_return_factor']!r} {CFG['constraint_margin']!r} {DT!r}"
    )
    for tick in range(600):
        now = round(tick * DT, 10)
        cmds, minor = script(tick, now)
        lines.append(f"tick {now!r} {int(minor)}")
        core.minor_present = minor
        for c in cmds:
            exp = c.expires if c.expires is not None else -1
            pairs = " ".join(f"{k} {v!r}" for k, v in c.targets.items())
            lines.append(
                f"cmd {c.source} {int(c.priority)} {int(c.additive)} {c.stamp!r} {exp!r} "
                f"{len(c.targets)} {pairs}"
            )
            core.feed(c)
        out_vals = core.tick(now, DT)
        lines.append(
            f"expect {len(out_vals)} " + " ".join(f"{k} {v!r}" for k, v in out_vals.items())
        )
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rt/tests/golden_motion.txt"))
