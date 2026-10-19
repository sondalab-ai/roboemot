"""One time budget for the whole stop sequence (SPEC 11A ``runtime.shutdown_budget_s``).

The Reachy daemon kills the app 20 s after SIGINT. The components stop one after another
(order matters: the SDK client goes last) and each waits for its thread with its own timeout;
those timeouts add up to more. :func:`begin` starts a shared deadline and :func:`join` waits for
``min(own timeout, time left)``, so once the budget is spent every later wait is a poll that
returns at once and the stragglers (daemon threads) are left to the process exit.
Outside a shutdown :func:`join` is a plain ``Thread.join``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)
_lock = threading.Lock()
_deadline: float | None = None
_clock: Callable[[], float] = time.monotonic


def begin(total_s: float, clock: Callable[[], float] = time.monotonic) -> None:
    """Start the shared deadline ``total_s`` from now."""
    global _deadline, _clock
    with _lock:
        _clock = clock
        _deadline = clock() + total_s


def end() -> None:
    """Forget the deadline (the stop sequence is over)."""
    global _deadline
    with _lock:
        _deadline = None


def remaining() -> float | None:
    """Seconds left in the budget (0 when spent); None outside a shutdown."""
    with _lock:
        return None if _deadline is None else max(0.0, _deadline - _clock())


def join(thread: threading.Thread, timeout_s: float | None) -> bool:
    """Wait for ``thread`` up to ``timeout_s`` (None: no limit of its own) and the budget left.
    True when the thread has ended. Gives up with a log line (thread name only) when the budget
    is what ran out."""
    left = remaining()
    limit = timeout_s if left is None else left if timeout_s is None else min(timeout_s, left)
    thread.join(limit)
    if thread.is_alive() and left is not None and (timeout_s is None or left < timeout_s):
        log.warning("shutdown: budget spent, leaving thread %s running", thread.name)
    return not thread.is_alive()
