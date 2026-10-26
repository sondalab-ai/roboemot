"""Lower OS priority for background worker threads (SPEC 13, ``runtime.background_*``).

On the CM4 the vision thread and its ONNX Runtime worker took about 60 % of a core while the
owner talked, and the voice listener fell more than 10 s behind the microphone. The threads named
in ``runtime.background_threads`` get ``runtime.background_nice`` added to their nice value, so the
kernel favours audio, the listener, speech and motion when the CPU is short; they still use every
idle cycle. Linux only (a thread is a task there, with its own nice value); a no-op elsewhere.
An unprivileged process may raise a nice value but never lower it back, so the change lasts for
the life of the thread (the fields need a restart).

- :func:`lower_current` is called by a thread at the start of its loop, before it builds any
  native thread pool (OpenCV, ONNX Runtime), which then inherits the value;
- :func:`run_lowered` runs a step that creates native threads (the vision models: each ONNX
  Runtime session starts its pool when it is built) on a short-lived thread named after a role
  (``vision-models``) that lowers itself first; on Linux a new thread inherits its creator's nice
  value, so the pools do too.

Process-wide state, set once by :func:`configure` at start; the report has counts only.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

NICE_MAX = 19  # the kernel's lowest priority
VISION_MODELS = "vision-models"  # role of the ONNX Runtime pool threads of the vision models

_lock = threading.Lock()
_nice = 0
_names: frozenset[str] = frozenset()
_applied: dict[str, int] = {}
_errors = 0
T = TypeVar("T")


def supported() -> bool:
    """Whether per-thread nice values exist here (Linux)."""
    return sys.platform.startswith("linux")


def _renice(tid: int, increment: int) -> None:
    current = os.getpriority(os.PRIO_PROCESS, tid)
    os.setpriority(os.PRIO_PROCESS, tid, min(NICE_MAX, current + increment))


# Seams for tests (the real calls need Linux).
renice: Callable[[int, int], None] = _renice
native_id: Callable[[], int] = threading.get_native_id
is_supported: Callable[[], bool] = supported


def configure(nice: int, names: Iterable[str]) -> None:
    """Set the increment and the thread names (and roles) it applies to; resets the counts."""
    global _nice, _names, _errors
    with _lock:
        _nice, _names, _errors = nice, frozenset(names), 0
        _applied.clear()


def _wanted(name: str) -> int:
    with _lock:
        return _nice if name in _names and _nice > 0 and is_supported() else 0


def _apply(name: str, tids: Iterable[int], increment: int) -> None:
    global _errors
    for tid in tids:
        try:
            renice(tid, increment)
        except OSError as exc:
            with _lock:
                _errors += 1
            log.warning("thread priority: %s: %s", name, type(exc).__name__)
            continue
        with _lock:
            _applied[name] = _applied.get(name, 0) + 1


def lower_current(name: str | None = None) -> None:
    """Lower the calling thread when its name (or ``name``) is listed."""
    name = threading.current_thread().name if name is None else name
    increment = _wanted(name)
    if increment:
        _apply(name, [native_id()], increment)


def run_lowered(role: str, fn: Callable[..., T], *args: Any) -> T:
    """``fn(*args)`` on a thread named ``role`` that lowers itself first when ``role`` is
    listed (else inline): the native threads ``fn`` starts inherit the value. Returns its result
    or raises its exception; the caller waits for it."""
    if not _wanted(role):
        return fn(*args)
    out: dict[str, Any] = {}

    def body() -> None:
        lower_current(role)
        try:
            out["value"] = fn(*args)
        except BaseException as exc:  # handed to the caller
            out["error"] = exc

    thread = threading.Thread(target=body, name=role, daemon=True)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out["value"]


def snapshot() -> dict[str, Any]:
    """For the debug report: whether it applies here, the increment, threads lowered per name
    (or role) and failures."""
    with _lock:
        return {
            "supported": is_supported(),
            "nice": _nice,
            "names": sorted(_names),
            "lowered": dict(_applied),
            "errors": _errors,
        }
