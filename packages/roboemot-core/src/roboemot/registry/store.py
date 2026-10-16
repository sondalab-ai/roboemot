"""Registry storage: an abstract store and a file implementation (SPEC 6, 12 Data).

``FileStore`` keeps JSON metadata and ``.npy`` embeddings in a directory the caller passes in.
Every write goes to a temporary file in the same directory and is renamed into place, so a
crash leaves the old or the new file, never half of one. Encryption at rest is a pluggable
``Cipher`` that defaults to a no-op: the real implementation (key file, optional passphrase)
is deferred to M5, so until then files are protected by their 0o600 / 0o700 modes only.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import numpy as np

FILE_MODE = 0o600
DIR_MODE = 0o700
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class Cipher(Protocol):
    """Transforms the bytes of every stored file; the default leaves them unchanged."""

    def encrypt(self, data: bytes) -> bytes: ...

    def decrypt(self, data: bytes) -> bytes: ...


class NullCipher:
    """No-op cipher used until encryption at rest lands (M5)."""

    def encrypt(self, data: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes) -> bytes:
        return data


class RegistryStore(ABC):
    """Keyed storage of a metadata dict plus named arrays, grouped by collection."""

    @abstractmethod
    def put(
        self, collection: str, key: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> None:
        """Replace the item atomically, dropping arrays it no longer has."""

    @abstractmethod
    def get(self, collection: str, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Load one item; raises ``KeyError`` when absent."""

    @abstractmethod
    def delete(self, collection: str, key: str) -> None:
        """Remove the item and every array of it; absent items are ignored."""

    @abstractmethod
    def keys(self, collection: str) -> list[str]:
        """Keys currently stored in the collection."""


def _check(name: str) -> str:
    if not _SAFE_ID.match(name):
        raise ValueError(f"invalid name {name!r}")
    return name


class FileStore(RegistryStore):
    """Directory-backed store: ``<root>/<collection>/<key>.json`` and ``<key>.<name>.npy``."""

    def __init__(self, root: Path | str, cipher: Cipher | None = None) -> None:
        self.root = Path(root)
        self.cipher: Cipher = cipher or NullCipher()
        self._mkdir(self.root)

    @staticmethod
    def _mkdir(path: Path) -> None:
        path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
        path.chmod(DIR_MODE)

    def _dir(self, collection: str) -> Path:
        path = self.root / _check(collection)
        self._mkdir(path)
        return path

    def _write(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            os.fchmod(fd, FILE_MODE)
            with os.fdopen(fd, "wb") as fh:
                fh.write(self.cipher.encrypt(data))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _array_files(folder: Path, key: str) -> list[Path]:
        return sorted(folder.glob(f"{key}.*.npy"))

    def put(
        self, collection: str, key: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> None:
        folder, key = self._dir(collection), _check(key)
        keep = set()
        for name, array in arrays.items():
            buf = io.BytesIO()
            np.save(buf, array, allow_pickle=False)
            path = folder / f"{key}.{_check(name)}.npy"
            self._write(path, buf.getvalue())
            keep.add(path)
        self._write(folder / f"{key}.json", json.dumps(meta, sort_keys=True).encode())
        for stale in set(self._array_files(folder, key)) - keep:
            stale.unlink(missing_ok=True)

    def get(self, collection: str, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        folder, key = self._dir(collection), _check(key)
        meta_path = folder / f"{key}.json"
        if not meta_path.exists():
            raise KeyError(key)
        meta = json.loads(self.cipher.decrypt(meta_path.read_bytes()))
        arrays = {
            p.name[len(key) + 1 : -len(".npy")]: np.load(
                io.BytesIO(self.cipher.decrypt(p.read_bytes())), allow_pickle=False
            )
            for p in self._array_files(folder, key)
        }
        return meta, arrays

    def delete(self, collection: str, key: str) -> None:
        folder, key = self._dir(collection), _check(key)
        for path in [*self._array_files(folder, key), folder / f"{key}.json"]:
            path.unlink(missing_ok=True)

    def keys(self, collection: str) -> list[str]:
        return sorted(p.stem for p in self._dir(collection).glob("*.json"))
