"""SPEC 8 stage 8, 11A: ``runtime.gc_freeze`` freezes the loaded objects while the app runs and
undoes it at the stop; the stop sequence starts the shared thread-wait budget."""

from __future__ import annotations

import gc

from roboemot import shutdown
from roboemot_reachymini import daemon
from test_app_perception import data, wait_for  # noqa: F401  (data: fixture)
from test_app_reflex import enable, fake_mini, new_app, run_app


def test_the_app_freezes_the_collector_while_it_runs_and_thaws_it_at_the_stop(data):  # noqa: F811
    app = new_app()
    before = gc.get_freeze_count()
    stop, thread, errors = run_app(app, fake_mini())
    try:
        assert wait_for(lambda: gc.get_freeze_count() > before)
        assert shutdown.remaining() is None  # the budget starts with the stop, not before
    finally:
        stop.set()
        thread.join(timeout=20)
    assert not errors and not thread.is_alive()
    assert gc.get_freeze_count() == 0  # gc.unfreeze thaws everything frozen
    assert shutdown.remaining() is None


def test_gc_freeze_can_be_switched_off(data):  # noqa: F811
    enable(data, "\nruntime:\n  gc_freeze: false\n")
    app = new_app()
    before = gc.get_freeze_count()
    stop, thread, errors = run_app(app, fake_mini())
    try:
        assert wait_for(lambda: app.diagnostics is not None)
        assert gc.get_freeze_count() == before
    finally:
        stop.set()
        thread.join(timeout=20)
    assert not errors


def test_the_stop_budget_is_capped_below_the_daemons_kill_timeout(data, monkeypatch):  # noqa: F811
    enable(data, "\nruntime:\n  shutdown_budget_s: 30.0\n")
    seen = []

    class Recording(shutdown.BudgetedExitStack):
        def __init__(self, total_s):
            seen.append(total_s)
            super().__init__(total_s)

    monkeypatch.setattr(shutdown, "BudgetedExitStack", Recording)
    app = new_app()
    stop, thread, errors = run_app(app, fake_mini())
    try:
        assert wait_for(lambda: bool(seen))
    finally:
        stop.set()
        thread.join(timeout=20)
    assert not errors
    assert seen == [daemon.STOP_TIMEOUT_S - 6.0]
