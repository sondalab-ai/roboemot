"""OS names for the Python threads (SPEC 13): ``top -H`` shows ``roboemot-vision``, not ``python``.

Before Python 3.14 a thread's name stays inside Python: Linux shows every thread of the app under
the process's name, so a CPU or wake-up measure on the robot cannot say which thread it saw.
:func:`install` makes every thread started after it write its name to the kernel (the task's
``comm``) as its first act. Linux only, a no-op elsewhere and on a Python that does it itself. The
kernel keeps 15 bytes of a name. A native thread inherits the name of the thread that creates it,
so the ONNX Runtime pools carry the role of the thread that loaded their model. The main thread
keeps the process's name: tools find the app by it.

Diagnostics only: nothing reads the name back, and a name that cannot be written is ignored.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Callable

COMM_MAX_BYTES = 15  # the kernel's TASK_COMM_LEN less the terminator
COMM_PATH = "/proc/thread-self/comm"
NATIVE_SINCE = (3, 14)  # threading sets the OS name itself from this version

_lock = threading.Lock()
_original_start: Callable[[threading.Thread], None] | None = None


def supported() -> bool:
    """Whether the names must and can be written here (Linux, Python before 3.14)."""
    return sys.platform.startswith("linux") and sys.version_info < NATIVE_SINCE


def _write_comm(name: str) -> None:
    """Give the calling thread the OS name ``name`` (the kernel cuts it to 15 bytes)."""
    with open(COMM_PATH, "w", encoding="utf-8") as comm:
        comm.write(name)


# Seams for tests (the real calls need Linux).
write_comm: Callable[[str], None] = _write_comm
is_supported: Callable[[], bool] = supported


def name_current(name: str) -> None:
    """Write ``name``, cut to what the kernel keeps, as the calling thread's OS name. A failure
    (no /proc, a read-only one) is ignored: the name is for diagnostics only."""
    with contextlib.suppress(OSError):
        write_comm(name.encode("utf-8")[:COMM_MAX_BYTES].decode("utf-8", "ignore"))


def _start_named(thread: threading.Thread) -> None:
    """``Thread.start`` whose thread names itself before it runs. The wrapper sits on the
    instance for the start only: the running thread removes it, so no cycle keeps the thread."""
    run = thread.run

    def named_run() -> None:
        thread.__dict__.pop("run", None)
        name_current(thread.name)
        run()

    thread.run = named_run  # type: ignore[method-assign]
    assert _original_start is not None
    _original_start(thread)


def install() -> bool:
    """Make every thread started from now on carry its Python name as its OS name. Returns
    whether it applies here; a second call changes nothing."""
    global _original_start
    if not is_supported():
        return False
    with _lock:
        if _original_start is None:
            _original_start = threading.Thread.start
            threading.Thread.start = _start_named  # type: ignore[method-assign]
    return True


def uninstall() -> None:
    """Put ``Thread.start`` back (tests)."""
    global _original_start
    with _lock:
        if _original_start is not None:
            threading.Thread.start = _original_start  # type: ignore[method-assign]
            _original_start = None
