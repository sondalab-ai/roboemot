"""SPEC 10, 12: arbitration, cross-fade, limits, constraints, watchdog, primitives."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from roboemot.manifest import load_manifest
from roboemot.motion import (
    CURVES,
    Arbiter,
    Command,
    Keyframe,
    Limiter,
    MotionCore,
    Player,
    Primitive,
    Priority,
)

REACHY = Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
DT = 0.02


@pytest.fixture()
def reachy():
    return load_manifest(REACHY, allow_unmeasured_dynamics=True)


def cmd(src, prio, t, now=0.0, **kw):
    return Command(src, prio, t, now, **kw)


def test_curves_endpoints():
    for name, f in CURVES.items():
        assert f(0.0) == pytest.approx(0.0), name
        assert f(1.0) == pytest.approx(1.0), name
    assert max(CURVES["cartoon"](i / 100) for i in range(101)) > 1.0  # overshoot


def test_priority_wins_and_ties_go_to_latest(reachy):
    arb = Arbiter(reachy.dofs, crossfade_s=0.0)
    arb.submit(cmd("idle", Priority.IDLE, {"head.yaw": 0.1}))
    arb.submit(cmd("mimic", Priority.MIMICRY, {"head.yaw": 0.2}))
    arb.submit(cmd("app", Priority.COMMANDED, {"head.yaw": 0.3}, now=0.0))
    assert arb.step(0.0)["head.yaw"] == pytest.approx(0.3)
    arb.submit(cmd("app2", Priority.COMMANDED, {"head.yaw": 0.4}, now=0.01))
    assert arb.step(0.01)["head.yaw"] == pytest.approx(0.4)
    arb.submit(cmd("reflex", Priority.REFLEX, {"head.yaw": -0.5}, now=0.02))
    assert arb.step(0.02)["head.yaw"] == pytest.approx(-0.5)


def test_additive_expression_not_on_top_of_reflex(reachy):
    arb = Arbiter(reachy.dofs, crossfade_s=0.0)
    arb.submit(cmd("app", Priority.COMMANDED, {"head.pitch": 0.1}))
    arb.submit(cmd("expr", Priority.EXPRESSION, {"head.pitch": 0.05}, additive=True))
    assert arb.step(0.0)["head.pitch"] == pytest.approx(0.15)
    arb.submit(cmd("reflex", Priority.REFLEX, {"head.pitch": -0.2}))
    assert arb.step(0.0)["head.pitch"] == pytest.approx(-0.2)


def test_crossfade_is_smooth_and_safety_is_immediate(reachy):
    arb = Arbiter(reachy.dofs, crossfade_s=0.15)
    arb.submit(cmd("a", Priority.COMMANDED, {"body_yaw": 0.0}))
    arb.step(0.0)
    arb.submit(cmd("b", Priority.REFLEX, {"body_yaw": 1.0}, now=0.0))
    assert arb.step(0.0)["body_yaw"] == pytest.approx(0.0)  # fade starts when observed
    mid = arb.step(0.075)["body_yaw"]
    assert 0.0 < mid < 1.0
    assert arb.step(0.2)["body_yaw"] == pytest.approx(1.0)
    arb.submit(cmd("stop", Priority.SAFETY, {"body_yaw": 0.3}, now=0.2))
    assert arb.step(0.21)["body_yaw"] == pytest.approx(0.3)


def test_expired_commands_drop_and_dof_returns_home(reachy):
    arb = Arbiter(reachy.dofs, crossfade_s=0.0)
    arb.submit(cmd("app", Priority.COMMANDED, {"antenna_left": 1.0}, expires=0.5))
    assert arb.step(0.1)["antenna_left"] == pytest.approx(1.0)
    assert arb.step(0.6)["antenna_left"] == pytest.approx(reachy.dofs["antenna_left"].home)


def test_limiter_clamps_position_and_yaw_delta(reachy):
    lim = Limiter(reachy.dofs, reachy.constraints)
    out = lim.clamp({"head.pitch": 2.0, "head.yaw": 2.0, "body_yaw": 0.0})
    assert out["head.pitch"] == pytest.approx(math.radians(40))
    assert out["head.yaw"] == pytest.approx(math.radians(65))


def test_limiter_respects_velocity_and_acceleration(reachy):
    lim = Limiter(reachy.dofs, reachy.constraints)
    dof = reachy.dofs["body_yaw"]
    prev_x, prev_v = lim.pos["body_yaw"], 0.0
    for _ in range(300):
        x = lim.step({"body_yaw": 1.0}, DT)["body_yaw"]
        v = (x - prev_x) / DT
        assert abs(v) <= dof.max_vel + 1e-9
        assert abs(v - prev_v) / DT <= dof.max_acc + 1e-6
        prev_x, prev_v = x, v
    assert lim.pos["body_yaw"] == pytest.approx(1.0, abs=1e-6)


def test_limiter_never_overshoots(reachy):
    lim = Limiter(reachy.dofs, reachy.constraints)
    peak = max(lim.step({"head.roll": 0.5}, DT)["head.roll"] for _ in range(400))
    assert peak <= 0.5 + 1e-9


def test_watchdog_returns_home_slowly(reachy):
    core = MotionCore(
        reachy.dofs,
        reachy.constraints,
        crossfade_s=0.0,
        watchdog_s=0.2,
        watchdog_return_factor=0.3,
    )
    t = 0.0
    for _ in range(200):
        core.feed(Command("app", Priority.COMMANDED, {"body_yaw": 0.8}, t))
        core.tick(t, DT)
        t += DT
    xs = []
    for _ in range(600):  # no more commands
        xs.append(core.tick(t, DT)["body_yaw"])
        t += DT
    assert core.holding
    assert xs[-1] == pytest.approx(0.0, abs=1e-6)
    speed = max(abs(b - a) for a, b in zip(xs, xs[1:], strict=False)) / DT
    assert speed <= reachy.dofs["body_yaw"].max_vel * 0.3 + 1e-9
    core.feed(Command("app", Priority.COMMANDED, {"body_yaw": 0.1}, t))
    core.tick(t, DT)
    assert not core.holding


def test_primitive_validation_and_player(reachy):
    with pytest.raises(ValueError):
        Primitive("bad", (Keyframe(0, {"head.yaw": 0}), Keyframe(1, {"head.pitch": 0})))
    nod = Primitive(
        "nod",
        (
            Keyframe(0.0, {"head.pitch": 0.0}),
            Keyframe(0.3, {"head.pitch": 0.3}),
            Keyframe(0.6, {"head.pitch": 0.0}),
        ),
    )
    p = Player(nod, start=1.0, amplitude=0.5)
    c = p.command(1.3)
    assert c and c.priority == Priority.COMMANDED and c.targets["head.pitch"] == pytest.approx(0.15)
    assert p.command(1.7) is None
    wiggle = Primitive(
        "wiggle",
        (Keyframe(0, {"antenna_left": 0.0}), Keyframe(0.2, {"antenna_left": 0.4})),
        additive=True,
    )
    c2 = Player(wiggle, 0.0).command(0.2 - 1e-9)
    assert c2 and c2.additive and c2.priority == Priority.EXPRESSION


def test_start_outside_limits_is_brought_in_without_a_jump(reachy):
    lim = Limiter(reachy.dofs, reachy.constraints)
    lim.reset({"head.z": -0.08})  # below the -55 mm limit, e.g. a pose left by another app
    z = reachy.dofs["head.z"]
    prev = -0.08
    for _ in range(400):
        x = lim.step({"head.z": 0.0}, DT)["head.z"]
        assert abs(x - prev) <= z.max_vel * DT + 1e-12
        prev = x
    assert prev == pytest.approx(0.0, abs=1e-6)


def test_constraint_correction_is_rate_limited(reachy):
    lim = Limiter(reachy.dofs, reachy.constraints)
    lim.reset({"head.yaw": 1.1, "body_yaw": 0.0})
    yaw_vmax = reachy.dofs["head.yaw"].max_vel
    prev = 1.1
    for _ in range(300):
        out = lim.step({"head.yaw": 1.1, "body_yaw": -1.0}, DT)  # body leaves: head must follow
        assert abs(out["head.yaw"] - prev) <= yaw_vmax * DT * 2 + 1e-12
        prev = out["head.yaw"]
    assert abs(out["head.yaw"] - out["body_yaw"]) <= math.radians(65) + 1e-6


# ----------------------------------------------------------------------------- review fixes


def test_speed_factor_drop_mid_motion_never_exceeds_limits(reachy):
    """Review finding 2: lowering the factor (the watchdog engages) while rushing toward a
    limit."""
    lim = Limiter(reachy.dofs, reachy.constraints)
    hi = reachy.dofs["body_yaw"].max
    for _ in range(400):
        lim.step({"body_yaw": hi}, DT, 1.0)
    lim.reset({"body_yaw": hi - 0.3})
    lim.vel["body_yaw"] = reachy.dofs["body_yaw"].max_vel  # still moving at full speed
    for _ in range(400):
        x = lim.step({"body_yaw": hi * 0.6}, DT, 0.3)["body_yaw"]
        assert x <= hi + 1e-12


@pytest.mark.parametrize(
    "dt,factor", [(0.0, 1.0), (-0.01, 1.0), (DT, 0.0), (DT, 1.5), (math.nan, 1.0)]
)
def test_bad_dt_or_factor_rejected(reachy, dt, factor):
    with pytest.raises(ValueError):
        Limiter(reachy.dofs, reachy.constraints).step({"head.yaw": 0.1}, dt, factor)


def test_non_finite_targets_rejected_and_state_untouched(reachy):
    arb = Arbiter(reachy.dofs, 0.0)
    with pytest.raises(ValueError):
        arb.submit(cmd("llm", Priority.COMMANDED, {"head.yaw": math.nan}))
    lim = Limiter(reachy.dofs, reachy.constraints)
    with pytest.raises(ValueError):
        lim.step({"head.yaw": math.inf}, DT)
    assert all(math.isfinite(v) for v in lim.pos.values())


def test_constraint_coupling_respects_velocity_acceleration_and_limit(reachy):
    """Review findings 6 and N3: body runs away faster than the head can follow."""
    lim = Limiter(reachy.dofs, reachy.constraints, math.radians(3))
    d = reachy.dofs
    prev, pv = dict(lim.pos), {k: 0.0 for k in d}
    for _ in range(400):
        out = lim.step({"head.yaw": 0.0, "body_yaw": 2.5}, DT)
        for k in ("head.yaw", "body_yaw"):
            v = (out[k] - prev[k]) / DT
            assert abs(v) <= d[k].max_vel + 1e-9
            assert abs(v - pv[k]) / DT <= d[k].max_acc + 1e-6
            pv[k] = v
        assert abs(out["head.yaw"] - out["body_yaw"]) <= math.radians(65) + 1e-9
        prev = out


def test_fuzz_limits_hold_with_random_targets_and_factors(reachy):
    """Random targets and speed-factor switches: position, velocity, acceleration and the
    65 degree coupling hold on every tick (second review, N3/N5/N6)."""
    import random

    rng = random.Random(7)
    lim = Limiter(reachy.dofs, reachy.constraints, math.radians(3))
    d = reachy.dofs
    prev, pv = dict(lim.pos), {k: 0.0 for k in d}
    targets = {}
    for t in range(20000):
        if t % 23 == 0:
            targets = {k: rng.uniform(dof.min - 0.5, dof.max + 0.5) for k, dof in d.items()}
        out = lim.step(targets, DT, rng.choice([1.0, 1.0, 0.6, 0.3]))
        for k, dof in d.items():
            v = (out[k] - prev[k]) / DT
            assert dof.min - 1e-12 <= out[k] <= dof.max + 1e-12
            assert abs(v) <= dof.max_vel + 1e-9
            assert abs(v - pv[k]) / DT <= dof.max_acc * (1 + 1e-6)
            pv[k] = v
        assert abs(out["head.yaw"] - out["body_yaw"]) <= math.radians(65) + 1e-9
        prev = out


def test_a_winner_at_its_limit_plus_an_additive_overflow_is_clamped_to_the_range(reachy):
    """SPEC 12: the arbiter clamps the summed target, additive offsets included."""
    dof = reachy.dofs["head.pitch"]
    core = MotionCore(
        reachy.dofs,
        reachy.constraints,
        crossfade_s=0.0,
        watchdog_s=100.0,
        watchdog_return_factor=1.0,
    )
    for i in range(400):
        t = i * DT
        core.feed(Command("gaze", Priority.COMMANDED, {"head.pitch": dof.max}, t))
        core.feed(Command("expr", Priority.EXPRESSION, {"head.pitch": 0.3}, t, additive=True))
        out = core.tick(t, DT)["head.pitch"]
    assert out == pytest.approx(dof.max)


def test_zero_watchdog_return_factor_refused_by_config():
    from roboemot.config import ConfigError, ConfigStack

    with pytest.raises(ConfigError):
        ConfigStack(app={"safety": {"watchdog_return_factor": 0}})
