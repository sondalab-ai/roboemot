"""The behaviour tick keeps its configured rate (SPEC 11A, ``motion.app_tick_hz``)."""

from __future__ import annotations

import inspect

import pytest
from roboemot.runtime import loop
from roboemot.runtime.loop import TickPacer

PERIOD_S = 0.05


def ticks_in(span_s: float, work_s: float, period_s: float = PERIOD_S) -> int:
    """Ticks a loop starts in ``span_s`` when each tick works for ``work_s`` and then rests for
    what the pacer says (a fake clock: no sleep)."""
    now, count = 0.0, 0
    pacer = TickPacer(now)
    while now < span_s:
        count += 1
        now += work_s
        now += pacer.rest_s(now, period_s)
    return count


def test_the_rate_is_the_configured_one_whatever_the_work_takes():
    # the old loop rested a full period after the work: 4 ms of work gave 18.5 Hz, not 20
    assert ticks_in(10.0, work_s=0.004) == 200
    assert ticks_in(10.0, work_s=0.030) == 200


def test_a_tick_longer_than_the_period_starts_the_next_at_once_with_no_burst_after():
    pacer = TickPacer(0.0)
    assert pacer.rest_s(0.2, PERIOD_S) == 0.0  # 150 ms late: no rest
    # the lost ticks are not run again: the next tick is one period after the late one
    assert pacer.rest_s(0.21, PERIOD_S) == pytest.approx(0.04)


def test_a_new_period_applies_from_the_next_tick():
    pacer = TickPacer(0.0)
    assert pacer.rest_s(0.01, 0.05) == pytest.approx(0.04)
    assert pacer.rest_s(0.06, 0.1) == pytest.approx(0.09)


def test_both_waits_of_the_loop_use_the_pacer():
    source = inspect.getsource(loop._tick_until_stopped)
    assert source.count("stop_event.wait(") == source.count("stop_event.wait(pacer.rest_s(") == 2
