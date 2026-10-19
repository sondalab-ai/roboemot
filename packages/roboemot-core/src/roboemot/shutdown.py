"""One time budget for the whole stop sequence (SPEC 11A ``runtime.shutdown_budget_s``).

The robot's supervisor kills the app some seconds after SIGINT (the robot plugin declares it
through ``Restartable.stop_timeout_s``; see :func:`capped_budget_s`). The components stop one
after another (order matters: the SDK client goes last) and each waits for its thread with its
own timeout; those timeouts add up to more. Inside :func:`budget` (or a
:class:`BudgetedExitStack`, whose unwinding is the budget) :func:`join` waits for
``min(own timeout, max(time left, MIN_WAIT_S))``.
Once the budget is spent every wait is still ``MIN_WAIT_S`` long, so a thread that was signalled
and exits in milliseconds is joined; a stuck one is left to the process exit (daemon threads).
Outside a budget :func:`join` is a plain ``Thread.join``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager

log = logging.getLogger(__name__)
# Floor of every wait once the budget is spent: enough for a signalled thread to exit.
MIN_WAIT_S = 0.05
_lock = threading.Lock()
_deadline: float | None = None
_clock: Callable[[], float] = time.monotonic


def capped_budget_s(budget_s: float, stop_timeout_s: float | None, margin_s: float) -> float:
    """``budget_s`` limited to ``stop_timeout_s - margin_s`` (never below 0) when the robot's
    supervisor declares a kill timeout; unchanged when it declares none (``None``). Logs a
    warning when the cap lowers the budget, and when the kill timeout leaves no time at all."""
    if stop_timeout_s is None:
        return budget_s
    allowed_s = stop_timeout_s - margin_s
    if allowed_s <= 0:
        log.warning(
            "shutdown: the supervisor kills after %g s, not above the %g s margin: "
            "the stop sequence gets no time to wait for threads",
            stop_timeout_s,
            margin_s,
        )
        return 0.0
    if allowed_s < budget_s:
        log.warning(
            "shutdown: budget capped from %g s to %g s (kill timeout %g s - margin %g s)",
            budget_s,
            allowed_s,
            stop_timeout_s,
            margin_s,
        )
        return allowed_s
    return budget_s


@contextmanager
def budget(total_s: float, clock: Callable[[], float] = time.monotonic) -> Iterator[None]:
    """Share ``total_s`` between the thread waits made inside the block. The deadline and the
    clock are forgotten at the exit. A budget inside another is logged and ignored: the outer
    deadline stays."""
    global _deadline, _clock
    with _lock:
        nested = _deadline is not None
        if not nested:
            _clock, _deadline = clock, clock() + total_s
    if nested:
        log.warning("shutdown: a budget is already running, ignoring the nested one")
        yield
        return
    try:
        yield
    finally:
        with _lock:
            _deadline, _clock = None, time.monotonic


class BudgetedExitStack(ExitStack):
    """An ``ExitStack`` whose unwinding runs inside a :func:`budget` of ``total_s``: the budget
    starts when the stop begins, not when the stack is entered."""

    def __init__(self, total_s: float) -> None:
        super().__init__()
        self._total_s = total_s

    def __exit__(self, *exc_details):  # type: ignore[no-untyped-def]
        with budget(self._total_s):
            return super().__exit__(*exc_details)


def remaining() -> float | None:
    """Seconds left in the budget (0 when spent); None outside one."""
    with _lock:
        return None if _deadline is None else max(0.0, _deadline - _clock())


def join(thread: threading.Thread, timeout_s: float | None) -> bool:
    """Wait for ``thread`` up to ``timeout_s`` (None: no limit of its own) and the budget left
    (at least ``MIN_WAIT_S``). True when the thread has ended. Logs the thread's name when the
    budget is what cut the wait short."""
    left = remaining()
    if left is None:
        thread.join(timeout_s)
        return not thread.is_alive()
    allowed = max(left, MIN_WAIT_S)
    thread.join(allowed if timeout_s is None else min(timeout_s, allowed))
    alive = thread.is_alive()
    if alive and (timeout_s is None or allowed < timeout_s):
        log.warning("shutdown: budget spent, leaving thread %s running", thread.name)
    return not alive
