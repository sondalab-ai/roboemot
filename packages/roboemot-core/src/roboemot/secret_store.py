"""Secret store: API keys for ``${secret:name}`` route keys (SPEC 8 Routes, 12 Data).

One JSON file ``secrets.json`` (name -> value) in the app data folder, mode 0o600 (owner
decision 2026-10-18; the CM4 has no secure element, so this is protection by file modes, the
same level as the registry key file). Writes are atomic: a temporary file created 0o600 in the
same folder, then renamed over the store. A read refuses the file when it is not a regular file
owned by this user, or when group or others have any access, so a key that was once exposed is
never used silently; the message says what to fix. Values are never part of an exception
message, a log line or a ``repr``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import paths

# Starts with a letter or digit so a name is never read as a command line option.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
NAME = re.compile(rf"^{NAME_PATTERN}$")
SECRET_REF = re.compile(rf"^\$\{{secret:({NAME_PATTERN})\}}$")  # a route's api_key
FILE_MODE = 0o600
MAX_VALUE_CHARS = 4096  # a key is a short token; this only bounds what the file can hold


class SecretUnavailable(RuntimeError):
    """A secret cannot be used. ``reason`` is a short phrase with no value in it."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"secret {name!r} is not available: {reason}")
        self.name, self.reason = name, reason


class SecretStore:
    """Reads and writes the secret file at ``path``; nothing is cached, so every read sees the
    file as it is."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_data_dir(cls, configured: str = "") -> SecretStore:
        """The store of the app data folder (``data_dir`` config value, env var or default)."""
        return cls(paths.secrets_path(configured))

    def __repr__(self) -> str:
        return f"SecretStore({str(self.path)!r})"

    def get(self, name: str) -> str:
        """The value of ``name``; raises :class:`SecretUnavailable` when the store is unsafe or
        unreadable, or the name is not in it."""
        data = self._load(name)
        if name not in data:
            raise SecretUnavailable(name, f"not set (run: roboemot secrets set {name})")
        return data[name]

    def names(self) -> list[str]:
        """Sorted names in the store (an absent file is empty)."""
        return sorted(self._load(""))

    def has(self, name: str) -> bool:
        """Whether ``name`` is set and the store is safe to read."""
        try:
            return name in self._load(name)
        except SecretUnavailable:
            return False

    def set(self, name: str, value: str) -> None:
        """Add or replace ``name``. The data folder is created private when missing."""
        if not NAME.match(name):
            raise ValueError(f"secret name must match {NAME_PATTERN}")
        if not value or len(value) > MAX_VALUE_CHARS or "\x00" in value:
            raise ValueError(f"secret value must be 1 to {MAX_VALUE_CHARS} characters")
        with self._locked():
            data = self._load(name)
            data[name] = value
            self._write(data)

    def remove(self, name: str) -> bool:
        """Delete ``name``; True when it was there. The file goes with its last secret."""
        if not self.path.exists():
            return False
        with self._locked():
            data = self._load(name)
            if name not in data:
                return False
            del data[name]
            if data:
                self._write(data)
            else:
                self.path.unlink()
                self._sync_folder()
        return True

    @contextlib.contextmanager
    def _locked(self):
        """Exclusive lock (flock on the data folder) around a load-modify-write, so two writers
        cannot lose each other's update. The folder is created private when missing."""
        folder = paths.data_dir(str(self.path.parent))
        fd = os.open(folder, os.O_RDONLY | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the lock

    def _sync_folder(self) -> None:
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self, name: str) -> dict[str, str]:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise SecretUnavailable(name, f"cannot open the secret file ({exc.strerror})") from exc
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise SecretUnavailable(name, f"{self.path} is not a regular file")
            if info.st_uid != os.getuid():
                raise SecretUnavailable(name, f"{self.path} is not owned by this user")
            if stat.S_IMODE(info.st_mode) & paths.GROUP_OTHER_BITS:
                raise SecretUnavailable(
                    name,
                    f"{self.path} is readable by group or others (run: chmod 600 {self.path})",
                )
            raw = handle.read()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise SecretUnavailable(name, f"{self.path} is not valid JSON") from exc
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in data.items()
        ):
            raise SecretUnavailable(name, f"{self.path} is not a name-to-text object")
        return data

    def _write(self, data: dict[str, str]) -> None:
        fd, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".")  # 0600
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), FILE_MODE)
                handle.write(json.dumps(data, indent=2, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            self._sync_folder()
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise


def no_secrets(name: str) -> str:
    """Resolver for code that has no store: every secret is unavailable."""
    raise SecretUnavailable(name, "no secret store is configured")


Resolver = Callable[[str], str]

__all__ = [
    "FILE_MODE",
    "NAME",
    "NAME_PATTERN",
    "Resolver",
    "SECRET_REF",
    "SecretStore",
    "SecretUnavailable",
    "no_secrets",
]
