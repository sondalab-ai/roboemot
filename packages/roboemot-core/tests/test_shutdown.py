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


@pytest.fixture(autouse=True)
def _clean():
    shutdown.end()
    yield
    shutdown.end()


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
    shutdown.begin(BUDGET_S)
    began = time.monotonic()
    results = [shutdown.join(t, OWN_S) for t in threads]  # alone: COMPONENTS * OWN_S = 1.5 s
    spent = time.monotonic() - began
    release.set()
    assert results == [False] * COMPONENTS
    assert BUDGET_S <= spent < BUDGET_S + 0.2
    assert shutdown.remaining() == 0.0


def test_a_spent_budget_still_reaps_a_thread_that_has_already_ended():
    done = threading.Thread(target=lambda: None)
    done.start()
    shutdown.begin(0.0)
    assert shutdown.join(done, OWN_S) is True


def test_a_join_without_its_own_limit_is_bounded_by_the_budget():
    release = threading.Event()
    t = stuck_thread(release)
    shutdown.begin(BUDGET_S)
    began = time.monotonic()
    assert shutdown.join(t, None) is False
    release.set()
    assert time.monotonic() - began < BUDGET_S + 0.2


def test_the_budget_is_a_restart_field_with_a_range_under_the_daemons_kill_window():
    cfg = RoboEmotConfig()
    assert cfg.runtime.shutdown_budget_s == 14.0 and cfg.runtime.gc_freeze is True
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"runtime": {"shutdown_budget_s": 1.0}})
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"runtime": {"shutdown_budget_s": 61.0}})
