"""Atomic file writes shared by the stores that keep state in the data folder (SPEC 11A, 12)."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def replace_file(path: Path, text: str | bytes, mode: int) -> None:
    """Write ``text`` (or raw bytes) over ``path`` atomically: a temp file in the same folder
    with ``mode`` (whatever the old file's mode was), synced, renamed over ``path``, then the
    folder synced as a best effort. A kill at any point leaves the old file or the new one,
    never a truncated one. ``path`` itself is replaced: a caller that means to write through a
    symlink resolves it first."""
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb" if isinstance(text, bytes) else "w") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    with contextlib.suppress(OSError):  # the new file is in place: durability is best effort
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
