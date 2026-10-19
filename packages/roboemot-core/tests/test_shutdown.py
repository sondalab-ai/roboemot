"""SPEC 11A ``runtime.shutdown_budget_s``: the thread waits of the stop sequence share one
budget instead of adding up."""

from __future__ import annotations

import threading
import time

import pytest
from roboemot import shutdown
from roboemot.config.schema import RoboEmotConfig

STUCK_S = 5.0
OWN_S = 0.3
BUDGET_S = 0.4
COMPONENTS = 5
EXITS_S = 0.01


def stuck_thread(release: threading.Event) -> threading.Thread:
    t = threading.Thread(target=release.wait, args=(STUCK_S,), daemon=True)
    t.start()
    return t


def test_without_a_budget_join_waits_its_own_timeout():
    release = threading.Event()
    t = stuck_thread(release)
    began = time.monotonic()
    assert shutdown.join(t, OWN_S) is False
    assert OWN_S <= time.monotonic() - began < OWN_S + 0.2
    release.set()
    assert shutdown.join(t, OWN_S) is True


def test_joins_that_never_end_together_stay_inside_the_budget():
    release = threading.Event()
    threads = [stuck_thread(release) for _ in range(COMPONENTS)]
    with shutdown.budget(BUDGET_S):
        began = time.monotonic()
        results = [shutdown.join(t, OWN_S) for t in threads]  # alone: COMPONENTS * OWN_S = 1.5 s
        spent = time.monotonic() - began
        assert shutdown.remaining() == 0.0
    release.set()
    assert results == [False] * COMPONENTS
    assert BUDGET_S <= spent < BUDGET_S + COMPONENTS * shutdown.MIN_WAIT_S + 0.2


def test_a_spent_budget_still_joins_a_thread_that_exits_in_milliseconds():
    done = threading.Event()
    t = threading.Thread(target=lambda: done.wait(EXITS_S))
    t.start()
    with shutdown.budget(0.0):
        assert shutdown.remaining() == 0.0
        done.set()  # the component signalled it; it needs a few ms to leave
        assert shutdown.join(t, OWN_S) is True


def test_a_join_without_its_own_limit_is_bounded_by_the_budget():
    release = threading.Event()
    t = stuck_thread(release)
    with shutdown.budget(BUDGET_S):
        began = time.monotonic()
        assert shutdown.join(t, None) is False
    release.set()
    assert time.monotonic() - began < BUDGET_S + 0.2


def test_the_budget_and_its_clock_are_forgotten_at_the_exit_even_on_an_error():
    ticks = iter([100.0, 100.0, 101.0, 102.0])
    with pytest.raises(RuntimeError), shutdown.budget(3.0, clock=lambda: next(ticks)):
        assert shutdown.remaining() == 3.0 - 0.0
        raise RuntimeError("x")
    assert shutdown.remaining() is None
    began = time.monotonic()
    stuck = stuck_thread(threading.Event())
    shutdown.join(stuck, OWN_S)  # real clock again, no budget: the thread's own timeout
    assert OWN_S <= time.monotonic() - began < OWN_S + 0.2


def test_a_nested_budget_is_ignored_and_logged(caplog):
    with shutdown.budget(BUDGET_S):
        with caplog.at_level("WARNING", logger="roboemot.shutdown"), shutdown.budget(60.0):
            assert shutdown.remaining() <= BUDGET_S
        assert shutdown.remaining() is not None  # the inner exit did not end the outer
    assert "already running" in caplog.text
    assert shutdown.remaining() is None


def test_the_budgeted_stack_starts_the_budget_when_it_unwinds_not_when_entered():
    seen = []
    with shutdown.BudgetedExitStack(BUDGET_S) as stack:
        stack.callback(lambda: seen.append(("last", shutdown.remaining())))
        stack.callback(lambda: seen.append(("first", shutdown.remaining())))
        seen.append(("running", shutdown.remaining()))
    assert [(k, v is not None) for k, v in seen] == [
        ("running", False),
        ("first", True),
        ("last", True),
    ]
    assert shutdown.remaining() is None


def test_the_budgeted_stack_ends_the_budget_when_a_callback_raises():
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError), shutdown.BudgetedExitStack(BUDGET_S) as stack:
        stack.callback(boom)
    assert shutdown.remaining() is None


def test_the_budget_is_a_restart_field_with_a_range_under_the_daemons_kill_window():
    cfg = RoboEmotConfig()
    assert cfg.runtime.shutdown_budget_s == 14.0 and cfg.runtime.gc_freeze is True
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"runtime": {"shutdown_budget_s": 1.0}})
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"runtime": {"shutdown_budget_s": 61.0}})


def test_the_margin_is_a_restart_field_with_a_default_unit_and_range():
    cfg = RoboEmotConfig()
    assert cfg.runtime.shutdown_margin_s == 6.0
    for bad in (-1.0, 2.9, 31.0):
        with pytest.raises(ValueError):
            RoboEmotConfig.model_validate({"runtime": {"shutdown_margin_s": bad}})
    field = type(cfg.runtime).model_fields["shutdown_budget_s"]
    assert "reachy" not in (field.description or "").lower()


@pytest.mark.parametrize(
    ("configured", "stop_timeout", "margin", "expected"),
    [
        (14.0, None, 6.0, 14.0),
        (14.0, 20.0, 6.0, 14.0),
        (30.0, 20.0, 6.0, 14.0),
        (10.0, 20.0, 6.0, 10.0),
        (14.0, 5.0, 6.0, 0.0),
    ],
)
def test_the_budget_is_capped_below_the_supervisors_kill_timeout(
    configured, stop_timeout, margin, expected
):
    assert shutdown.capped_budget_s(configured, stop_timeout, margin) == expected


def test_the_margin_may_only_be_raised_at_runtime():
    field = type(RoboEmotConfig().runtime).model_fields["shutdown_margin_s"]
    assert field.json_schema_extra["conservative"] == "higher"
    assert RoboEmotConfig.model_validate({"runtime": {"shutdown_margin_s": 3.0}})


def test_a_cap_that_bites_is_warned(caplog):
    with caplog.at_level("WARNING", logger="roboemot.shutdown"):
        assert shutdown.capped_budget_s(30.0, 20.0, 6.0) == 14.0
    assert "capped" in caplog.text and "30" in caplog.text and "14" in caplog.text


def test_no_warning_when_the_budget_fits(caplog):
    with caplog.at_level("WARNING", logger="roboemot.shutdown"):
        shutdown.capped_budget_s(14.0, 20.0, 6.0)
        shutdown.capped_budget_s(14.0, None, 6.0)
    assert caplog.text == ""


def test_a_kill_timeout_below_the_margin_is_warned_not_silent(caplog):
    with caplog.at_level("WARNING", logger="roboemot.shutdown"):
        assert shutdown.capped_budget_s(14.0, 5.0, 6.0) == 0.0
    assert "no time" in caplog.text or "0 s" in caplog.text
