"""SPEC 12: the minor band applies to the final summed target, additive offsets included."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.affect.expression import build_primitive
from roboemot.config.schema import ExpressionEntry
from roboemot.manifest import load_manifest
from roboemot.motion import Command, MotionCore, Player, Priority

REACHY = Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
DT = 0.02
FACTOR = 0.6


@pytest.fixture()
def reachy():
    return load_manifest(REACHY, allow_unmeasured_dynamics=True)


def make_core(reachy) -> MotionCore:
    return MotionCore(
        reachy.dofs,
        reachy.constraints,
        crossfade_s=0.0,
        watchdog_s=100.0,
        minor_factor=FACTOR,
        watchdog_return_factor=1.0,
    )


def test_a_winner_at_its_limit_plus_an_additive_overflow_stays_in_the_band(reachy):
    dof = reachy.dofs["head.pitch"]
    core = make_core(reachy)
    core.minor_present = True
    peak = 0.0
    for i in range(400):
        t = i * DT
        core.feed(Command("gaze", Priority.COMMANDED, {"head.pitch": dof.max}, t))
        core.feed(Command("expr", Priority.EXPRESSION, {"head.pitch": 0.3}, t, additive=True))
        peak = max(peak, core.tick(t, DT)["head.pitch"] - dof.home)
    assert 0 < peak <= FACTOR * (dof.max - dof.home) + 1e-9


def test_without_a_minor_the_sum_is_clamped_to_the_range(reachy):
    dof = reachy.dofs["head.pitch"]
    core = make_core(reachy)
    for i in range(400):
        t = i * DT
        core.feed(Command("gaze", Priority.COMMANDED, {"head.pitch": dof.max}, t))
        core.feed(Command("expr", Priority.EXPRESSION, {"head.pitch": 0.3}, t, additive=True))
        out = core.tick(t, DT)["head.pitch"]
    assert out == pytest.approx(dof.max)


def test_the_connectivity_gestures_stay_in_the_band(reachy):
    dof = reachy.dofs["head.roll"]
    core = make_core(reachy)
    core.minor_present = True
    puzzled = ExpressionEntry.model_validate(reachy.model.behaviour["gesture_map"]["puzzled"])
    prim = build_primitive("puzzled", puzzled, reachy.dofs, body_yaw="body_yaw")
    assert prim is not None
    player = Player(prim, 0.0)
    peak = 0.0
    for i in range(300):
        t = i * DT
        core.feed(Command("gaze", Priority.COMMANDED, {"head.roll": dof.max}, t))
        if (cmd := player.command(t)) is not None:
            core.feed(cmd)
        peak = max(peak, core.tick(t, DT)["head.roll"] - dof.home)
    assert peak <= FACTOR * (dof.max - dof.home) + 1e-9
