"""SPEC 10, 12: the arbiter and the limiter give bit-identical results to their reference.

``RefArbiter.step`` and ``RefLimiter.step`` are verbatim copies of the implementations before the
per-tick work was cut (one pass over the commands, settled dofs skip the stopping-speed search).
Long randomised runs drive both and compare every output and the internal state after every
tick bit for bit (``float.hex``, so ``-0.0`` differs from ``0.0``) and key order included.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path

import pytest
from roboemot.manifest import load_manifest
from roboemot.motion import Arbiter, Command, Limiter, MotionCore, Priority
from roboemot.motion.core import LAND_TOLERANCE, _Fade, _require_finite, stop_speed
from roboemot.motion.curves import min_jerk

REACHY = Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
SEEDS = range(4)
TICKS = 2500


class RefArbiter(Arbiter):
    def step(self, now: float) -> dict[str, float]:
        cmds = self.active(now)
        out: dict[str, float] = {}
        goal: dict[str, float] = {}
        reflex = False
        for name, dof in self.dofs.items():
            absolute = [c for c in cmds if not c.additive and name in c.targets]
            if absolute:
                best = min(absolute, key=lambda c: (c.priority, -c.stamp, c.source))
                target, winner, prio = best.targets[name], best.source, best.priority
            else:
                target, winner, prio = dof.home, None, Priority.IDLE
            reflex = reflex or prio == Priority.REFLEX
            if prio >= Priority.COMMANDED:
                target += sum(c.targets[name] for c in cmds if c.additive and name in c.targets)
            if prio != Priority.SAFETY:
                target = min(max(target, dof.min), dof.max)
            goal[name] = target
            if winner != self._winner[name]:
                self._fade[name] = (
                    None if prio == Priority.SAFETY else _Fade(now, self.last[name], winner or "")
                )
                self._winner[name] = winner
            fade = self._fade[name]
            if fade is not None and self.crossfade_s > 0:
                s = (now - fade.start) / self.crossfade_s
                if s >= 1.0:
                    self._fade[name] = None
                else:
                    w = min_jerk(s)
                    target = fade.from_value + (target - fade.from_value) * w
            out[name] = target
        self.last = dict(out)
        self.goal = goal
        self.reflex_active = reflex
        return out


class RefLimiter(Limiter):
    def step(self, targets: dict[str, float], dt: float, factor: float = 1.0) -> dict[str, float]:
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        if not (math.isfinite(factor) and 0 < factor <= 1):
            raise ValueError(f"factor must be in (0, 1], got {factor}")
        _require_finite(targets)
        goal = self.clamp(targets)
        for c in self.constraints:
            if c.type != "yaw_delta" or c.a not in self.pos or c.b not in self.pos:
                continue
            lim = max(0.0, math.radians(c.max_deg) - self.constraint_margin)
            for u, w in ((c.a, c.b), (c.b, c.a)):
                if u in goal:
                    du = self.dofs[u]
                    g = min(max(goal[u], self.pos[w] - lim), self.pos[w] + lim)
                    goal[u] = min(max(g, du.min), du.max)
        prev = dict(self.pos)
        for name, dof in self.dofs.items():
            if name not in goal:
                continue
            vmax, amax = dof.max_vel * factor, dof.max_acc
            x, v = self.pos[name], self.vel[name]
            err = goal[name] - x
            v_cap = min(vmax, stop_speed(abs(err), amax, dt))
            v_des = max(-v_cap, min(v_cap, err / dt))
            dv = max(-amax * dt, min(amax * dt, v_des - v))
            x_new = x + (v + dv) * dt
            if (goal[name] - x_new) * err < 0 and abs(err / dt - v) <= amax * dt * LAND_TOLERANCE:
                x_new = goal[name]
            self.pos[name] = x_new
        for name, dof in self.dofs.items():
            x0, x = prev[name], self.pos[name]
            if dof.min <= x0 <= dof.max:
                x = min(max(x, dof.min), dof.max)
            elif x0 > dof.max:
                x = min(x, x0)
            else:
                x = max(x, x0)
            self.pos[name] = x
        for name in self.dofs:
            self.vel[name] = (self.pos[name] - prev[name]) / dt
        return dict(self.pos)


def bits(values: dict[str, float]) -> list[tuple[str, str, str]]:
    """Key order, type and exact bits of every value (``-0.0`` and ``0.0`` differ)."""
    return [(k, type(v).__name__, float(v).hex()) for k, v in values.items()]


def fade_bits(fade: _Fade | None) -> tuple[str, str, str] | None:
    return None if fade is None else (fade.start.hex(), float(fade.from_value).hex(), fade.winner)


@pytest.fixture()
def reachy():
    return load_manifest(REACHY, allow_unmeasured_dynamics=True)


def _value(rng: random.Random, dof, here: float) -> float:
    """A target that exercises every branch: inside, outside, exact, signed zero, tiny steps."""
    roll = rng.random()
    if roll < 0.25:
        return rng.uniform(dof.min, dof.max)
    if roll < 0.35:
        return rng.uniform(dof.min - 1.0, dof.max + 1.0)  # outside the limits
    if roll < 0.55:
        return here  # exactly where the dof is: error 0
    if roll < 0.65:
        return rng.choice([0.0, -0.0])
    if roll < 0.8:
        return here + rng.choice([1e-15, -1e-15, 1e-9, -1e-9, 5e-324, -5e-324])
    if roll < 0.9:
        return math.nextafter(here, rng.choice([math.inf, -math.inf]))
    return rng.choice([dof.min, dof.max, dof.home])


def _goal(rng: random.Random, lim: Limiter) -> dict[str, float]:
    names = list(lim.dofs)
    pick = rng.sample(names, rng.randint(1, len(names)))  # often some dofs are missing
    return {n: _value(rng, lim.dofs[n], lim.pos[n]) for n in pick}


def _assert_limiters_equal(ref: Limiter, new: Limiter) -> None:
    assert bits(new.pos) == bits(ref.pos)
    assert bits(new.vel) == bits(ref.vel)


@pytest.mark.parametrize("seed", SEEDS)
def test_limiter_step_is_bit_identical_to_the_reference(reachy, seed):
    rng = random.Random(seed)
    dofs = dict(reachy.dofs)  # shared by both: a tightened limit reaches both the same tick
    original = dict(dofs)
    ref = RefLimiter(dofs, reachy.constraints)
    new = Limiter(dofs, reachy.constraints)
    targets = _goal(rng, ref)
    for tick in range(TICKS):
        roll = rng.random()
        if roll < 0.08:
            targets = _goal(rng, ref)  # a new goal; otherwise the goal holds and the dofs settle
        elif roll < 0.1:
            state = {
                n: rng.choice([0.0, -0.0, rng.uniform(d.min - 0.3, d.max + 0.3)])
                for n, d in dofs.items()
            }
            ref.reset(state)  # a pose left by another app, possibly outside the limits
            new.reset(state)
        elif roll < 0.12:
            name = rng.choice(list(dofs))
            d = original[name]
            span = d.max - d.min
            dofs[name] = replace(
                d, min=d.min + span * rng.uniform(0, 0.4), max=d.max - span * rng.uniform(0, 0.4)
            )
        elif roll < 0.13:
            dofs.update(original)
        elif roll < 0.14:
            margin = rng.choice([0.0, 0.01, 0.2, 10.0])
            ref.constraint_margin = new.constraint_margin = margin
        dt = 0.02 if rng.random() < 0.8 else rng.uniform(1e-4, 0.1)
        factor = 1.0 if rng.random() < 0.7 else rng.choice([0.3, 1e-3, rng.uniform(1e-6, 1.0)])
        a = ref.step(dict(targets), dt, factor)
        b = new.step(dict(targets), dt, factor)
        assert bits(b) == bits(a), f"tick {tick}"
        _assert_limiters_equal(ref, new)


def test_limiter_settled_dofs_at_signed_zero_match_the_reference(reachy):
    """Settled at -0.0: the reference turns it into +0.0, the skip must too."""
    for v in (0.0, -0.0):
        ref = RefLimiter(dict(reachy.dofs), reachy.constraints)
        new = Limiter(dict(reachy.dofs), reachy.constraints)
        state = dict.fromkeys(reachy.dofs, -0.0)
        ref.reset(state)
        new.reset(state)
        ref.vel = dict.fromkeys(ref.vel, v)
        new.vel = dict.fromkeys(new.vel, v)
        for goal in (dict.fromkeys(reachy.dofs, -0.0), dict.fromkeys(reachy.dofs, 0.0), {}):
            assert bits(new.step(goal, 0.02)) == bits(ref.step(goal, 0.02))
            _assert_limiters_equal(ref, new)


def test_limiter_errors_leave_the_same_state(reachy):
    ref = RefLimiter(dict(reachy.dofs), reachy.constraints)
    new = Limiter(dict(reachy.dofs), reachy.constraints)
    for lim in (ref, new):
        lim.step({"head.pitch": 0.2}, 0.02)
    for bad in (
        ({"head.pitch": math.nan}, 0.02, 1.0),
        ({"head.pitch": math.inf}, 0.02, 1.0),
        ({"head.pitch": 0.1}, 0.0, 1.0),
        ({"head.pitch": 0.1}, 0.02, 0.0),
        ({"nope": 0.1}, 0.02, 1.0),
    ):
        errors = []
        for lim in (ref, new):
            with pytest.raises((ValueError, KeyError)) as exc:
                lim.step(*bad)
            errors.append((exc.type, str(exc.value)))
        assert errors[0] == errors[1]
        _assert_limiters_equal(ref, new)


SOURCES = [
    ("safety", Priority.SAFETY, False),
    ("reflex", Priority.REFLEX, False),
    ("attention", Priority.COMMANDED, False),
    ("scan", Priority.COMMANDED, False),
    ("mimicry", Priority.MIMICRY, False),
    ("expression", Priority.EXPRESSION, False),
    ("expression-add", Priority.EXPRESSION, True),
    ("speaking-add", Priority.EXPRESSION, True),
    ("idle", Priority.IDLE, False),
    ("idle-additive", Priority.IDLE, True),
    ("reflex-add", Priority.REFLEX, True),
]


def _arb_targets(rng: random.Random, dofs) -> dict[str, float]:
    names = rng.sample(list(dofs), rng.randint(1, len(dofs)))
    out = {}
    for n in names:
        d = dofs[n]
        out[n] = rng.choice(
            [
                rng.uniform(d.min - 0.2, d.max + 0.2),
                0.0,
                -0.0,
                d.home,
                1e-17,
                rng.uniform(-1e-3, 1e-3),
            ]
        )
    return out


def _assert_arbiters_equal(ref: Arbiter, new: Arbiter) -> None:
    assert bits(new.last) == bits(ref.last)
    assert bits(new.goal) == bits(ref.goal)
    assert new.reflex_active is ref.reflex_active
    assert list(new._winner.items()) == list(ref._winner.items())
    assert [(k, fade_bits(f)) for k, f in new._fade.items()] == [
        (k, fade_bits(f)) for k, f in ref._fade.items()
    ]
    assert list(new._cmds) == list(ref._cmds)


@pytest.mark.parametrize("seed", SEEDS)
def test_arbiter_step_is_bit_identical_to_the_reference(reachy, seed):
    rng = random.Random(seed)
    ref, new = RefArbiter(reachy.dofs, 0.15), Arbiter(reachy.dofs, 0.15)
    now = 0.0
    for tick in range(TICKS):
        now += rng.choice([0.02, 0.02, 0.0, 0.005, 0.3])  # a repeated stamp ties on recency
        for _ in range(rng.choice([0, 0, 1, 1, 2, 4])):
            src, prio, additive = rng.choice(SOURCES)
            stamp = rng.choice([now, now, now - 0.01, 0.0])
            expires = rng.choice([None, None, now + 0.05, now + 0.5, now])
            cmd = Command(src, prio, _arb_targets(rng, reachy.dofs), stamp, expires, additive)
            ref.submit(cmd)
            new.submit(cmd)
        if rng.random() < 0.05:
            src = rng.choice(SOURCES)[0]
            ref.clear(src)
            new.clear(src)
        if rng.random() < 0.01:
            ref.crossfade_s = new.crossfade_s = rng.choice([0.0, 0.15, 0.5])
        a, b = ref.step(now), new.step(now)
        assert bits(b) == bits(a), f"tick {tick}"
        _assert_arbiters_equal(ref, new)


@pytest.mark.parametrize("seed", SEEDS)
def test_motion_core_tick_is_bit_identical_to_the_reference(reachy, seed):
    """Arbiter into limiter with the watchdog in between: the path the motion loop runs."""
    rng = random.Random(100 + seed)

    def core(arbiter_cls, limiter_cls):
        c = MotionCore(
            reachy.dofs,
            reachy.constraints,
            crossfade_s=0.15,
            watchdog_s=0.5,
            watchdog_return_factor=0.3,
            constraint_margin=0.01,
        )
        c.arbiter = arbiter_cls(reachy.dofs, 0.15)
        c.limiter = limiter_cls(reachy.dofs, reachy.constraints, 0.01)
        return c

    ref, new = core(RefArbiter, RefLimiter), core(Arbiter, Limiter)
    now = 0.0
    for tick in range(TICKS):
        now += 0.02
        if rng.random() < 0.3:
            src, prio, additive = rng.choice(SOURCES[1:])  # the watchdog owns SAFETY here
            cmd = Command(
                src,
                prio,
                _arb_targets(rng, reachy.dofs),
                now,
                rng.choice([None, now + 0.2]),
                additive,
            )
            liveness = rng.random() < 0.9
            ref.feed(cmd, liveness=liveness)
            new.feed(cmd, liveness=liveness)
        if rng.random() < 0.003:
            now += 1.0  # the app stalls: the watchdog engages
        a, b = ref.tick(now, 0.02), new.tick(now, 0.02)
        assert bits(b) == bits(a), f"tick {tick}"
        assert new.holding is ref.holding
        _assert_arbiters_equal(ref.arbiter, new.arbiter)
        _assert_limiters_equal(ref.limiter, new.limiter)
