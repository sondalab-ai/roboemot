"""Ramps between the app tick's commands (SPEC 10, Motor loop)."""

import math
import statistics

import pytest
from roboemot.manifest import Dof
from roboemot.motion import Command, Priority
from roboemot.motion.core import Arbiter, Limiter
from roboemot.motion.ramps import GAP_TICKS, CommandRamps

RAMP_S = 0.05
CH = "head.pitch"


def cmd(stamp, value, priority=Priority.IDLE, source="s"):
    return Command(source, priority, {CH: value}, stamp, expires=stamp + 0.5)


def test_the_first_command_passes_and_starts_no_ramp():
    ramps = CommandRamps(RAMP_S)
    first = cmd(0.0, 1.0)
    assert ramps.admit(first) is first
    assert ramps.step(0.01) == []


def test_a_command_ramps_from_the_last_target_over_one_app_tick():
    ramps = CommandRamps(RAMP_S)
    ramps.admit(cmd(0.0, 0.0))
    assert ramps.admit(cmd(RAMP_S, 1.0)).targets[CH] == pytest.approx(0.0)
    assert ramps.step(RAMP_S * 1.5)[0].targets[CH] == pytest.approx(0.5)
    assert ramps.step(RAMP_S * 2)[0].targets[CH] == pytest.approx(1.0)
    assert ramps.step(RAMP_S * 2.5) == []  # ended: nothing more to send


def test_a_new_command_starts_from_where_the_ramp_stands():
    ramps = CommandRamps(RAMP_S)
    ramps.admit(cmd(0.0, 0.0))
    ramps.admit(cmd(RAMP_S, 1.0))
    assert ramps.admit(cmd(RAMP_S * 1.5, 2.0)).targets[CH] == pytest.approx(0.5)


@pytest.mark.parametrize("priority", [Priority.SAFETY, Priority.REFLEX])
def test_a_safety_or_reflex_command_is_never_ramped(priority):
    ramps = CommandRamps(RAMP_S)
    ramps.admit(cmd(0.0, 0.0))
    fresh = cmd(RAMP_S, 1.0, priority)
    assert ramps.admit(fresh) is fresh and ramps.step(RAMP_S * 1.5) == []


def test_a_pause_is_a_new_start_and_not_a_ramp():
    ramps = CommandRamps(RAMP_S)
    ramps.admit(cmd(0.0, 0.0))
    late = cmd(RAMP_S * (GAP_TICKS + 1), 1.0)
    assert ramps.admit(late) is late


def test_a_cleared_source_sends_nothing_more():
    ramps = CommandRamps(RAMP_S)
    ramps.admit(cmd(0.0, 0.0))
    ramps.admit(cmd(RAMP_S, 1.0))
    ramps.clear("s")
    assert ramps.step(RAMP_S * 1.5) == []


def test_a_zero_ramp_passes_everything_through():
    ramps = CommandRamps(0.0)
    ramps.admit(cmd(0.0, 0.0))
    fresh = cmd(RAMP_S, 1.0)
    assert ramps.admit(fresh) is fresh


def test_a_slow_move_at_the_app_tick_stays_smooth_under_a_high_acceleration_cap():
    """Regression: with 2000 deg/s^2 the limiter followed each 20 Hz step, stop and go."""

    def median_speed(ramp_s):
        dofs = {CH: Dof(CH, "head", -1.0, 1.0, math.radians(180), math.radians(2000), 0.0, True)}
        arbiter, limiter, ramps = Arbiter(dofs, 0.0), Limiter(dofs, []), CommandRamps(ramp_s)
        dt, speeds, next_tick = 0.02, [], 0.0
        for i in range(400):
            t = i * dt
            if t >= next_tick - 1e-9:
                arbiter.submit(ramps.admit(cmd(t, math.radians(8) * math.sin(math.pi * t))))
                next_tick += 0.05
            for ramped in ramps.step(t):
                arbiter.submit(ramped)
            before = limiter.pos[CH]
            limiter.step(arbiter.step(t), dt)
            if t > 2:
                speeds.append(abs(math.degrees(limiter.pos[CH] - before) / dt))
        return statistics.median(speeds)

    assert median_speed(0.0) < 10.0  # the stop-and-go this fixes
    assert median_speed(0.05) > 15.0


def test_no_ramp_lasts_longer_than_expected():
    ramps = CommandRamps(RAMP_S)
    assert ramps.step(0.0) == []
