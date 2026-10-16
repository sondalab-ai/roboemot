"""Registry storage: an abstract store and a file implementation (SPEC 6, 12 Data).

``FileStore`` keeps JSON metadata and ``.npy`` embeddings in a directory the caller passes in.
Every file is written to a temporary name and renamed into place; the metadata JSON is renamed
last, so a crash leaves the old item or the new one, never an array without metadata (orphans
are swept at startup). Each file starts with a format byte followed by the cipher output.

Encryption at rest is a pluggable ``Cipher`` that receives the file's relative path as
associated data. The real implementation (key file, optional passphrase) is deferred to M5.
Until then a store needs ``allow_plaintext=True`` (development and pre-M5 only, logged once)
and files are protected by their 0o600 / 0o700 modes alone.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

FILE_MODE = 0o600
DIR_MODE = 0o700
FORMAT_VERSION = 1
JSON_NAME = "json"
TMP_PREFIX = ".tmp-"
CORRUPT_SUFFIX = ".corrupt"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


class StoreError(Exception):
    """Raised when a stored file has an unknown format."""


class Cipher(Protocol):
    """Transforms the bytes of every stored file; ``aad`` is the file's relative path."""

    def encrypt(self, data: bytes, aad: bytes) -> bytes: ...

    def decrypt(self, data: bytes, aad: bytes) -> bytes: ...


class NullCipher:
    """No-op cipher for development and the time before encryption at rest lands (M5)."""

    def encrypt(self, data: bytes, aad: bytes) -> bytes:
        return data

    def decrypt(self, data: bytes, aad: bytes) -> bytes:
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

    @abstractmethod
    def quarantine(self, collection: str, key: str) -> None:
        """Move an unreadable item aside so it stops failing startup."""


def _check(name: str) -> str:
    if not _SAFE_ID.fullmatch(name):
        raise ValueError(f"invalid name {name!r}")
    return name


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FileStore(RegistryStore):
    """Directory-backed store: ``<root>/<collection>/<key>.json`` and ``<key>.<name>.npy``."""

    def __init__(
        self, root: Path | str, cipher: Cipher | None = None, *, allow_plaintext: bool = False
    ) -> None:
        if cipher is None and not allow_plaintext:
            raise ValueError("FileStore needs a cipher, or allow_plaintext=True (dev, pre-M5)")
        self.cipher: Cipher = cipher or NullCipher()
        if isinstance(self.cipher, NullCipher):
            log.warning("registry store writes plaintext biometric data (no cipher configured)")
        self.root = Path(root)
        self._mkdir(self.root)
        self._sweep()

    @staticmethod
    def _mkdir(path: Path) -> None:
        """Create ``path`` and missing parents, setting 0o700 only on the ones created."""
        missing = [p for p in (path, *path.parents) if not p.exists()]
        for folder in reversed(missing):
            folder.mkdir(mode=DIR_MODE)
            folder.chmod(DIR_MODE)

    def _sweep(self) -> None:
        """Remove leftovers of an interrupted write: temp files and arrays with no metadata."""
        for folder in (p for p in self.root.iterdir() if p.is_dir()):
            for tmp in folder.glob(f"{TMP_PREFIX}*"):
                tmp.unlink(missing_ok=True)
            for npy in folder.glob("*.npy"):
                if not (folder / f"{npy.name.split('.', 1)[0]}.json").exists():
                    npy.unlink(missing_ok=True)

    def _folder(self, collection: str) -> Path:
        return self.root / _check(collection)

    def _aad(self, collection: str, filename: str) -> bytes:
        return f"{collection}/{filename}".encode()

    def _seal(self, collection: str, filename: str, data: bytes) -> bytes:
        return bytes([FORMAT_VERSION]) + self.cipher.encrypt(data, self._aad(collection, filename))

    def _open(self, collection: str, filename: str, raw: bytes) -> bytes:
        if not raw or raw[0] != FORMAT_VERSION:
            raise StoreError("unknown file format")
        return self.cipher.decrypt(raw[1:], self._aad(collection, filename))

    @staticmethod
    def _stage(folder: Path, sealed: bytes) -> Path:
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=TMP_PREFIX)
        try:
            os.fchmod(fd, FILE_MODE)
            with os.fdopen(fd, "wb") as fh:
                fh.write(sealed)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return Path(tmp)

    @staticmethod
    def _array_files(folder: Path, key: str) -> list[Path]:
        return sorted(folder.glob(f"{key}.*.npy"))

    def put(
        self, collection: str, key: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> None:
        folder, key = self._folder(collection), _check(key)
        payloads = {f"{key}.{JSON_NAME}": json.dumps(meta, sort_keys=True).encode()}
        for name, array in arrays.items():
            buf = io.BytesIO()
            np.save(buf, array, allow_pickle=False)
            payloads[f"{key}.{_check(name)}.npy"] = buf.getvalue()
        self._mkdir(folder)
        existed = (folder / f"{key}.json").exists()
        ordered = [*(n for n in payloads if n.endswith(".npy")), f"{key}.json"]
        staged: dict[str, Path] = {}
        moved: list[Path] = []
        try:
            for filename in ordered:
                staged[filename] = self._stage(
                    folder, self._seal(collection, filename, payloads[filename])
                )
            for filename in ordered:
                os.replace(staged.pop(filename), folder / filename)
                moved.append(folder / filename)
        except BaseException:
            for tmp in staged.values():
                tmp.unlink(missing_ok=True)
            if not existed:
                for path in moved:
                    path.unlink(missing_ok=True)
            raise
        keep = {folder / n for n in ordered}
        for stale in set(self._array_files(folder, key)) - keep:
            stale.unlink(missing_ok=True)
        _fsync_dir(folder)

    def get(self, collection: str, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        folder, key = self._folder(collection), _check(key)
        meta_path = folder / f"{key}.json"
        if not meta_path.exists():
            raise KeyError(key)
        meta = json.loads(self._open(collection, meta_path.name, meta_path.read_bytes()))
        arrays = {
            p.name[len(key) + 1 : -len(".npy")]: np.load(
                io.BytesIO(self._open(collection, p.name, p.read_bytes())), allow_pickle=False
            )
            for p in self._array_files(folder, key)
        }
        return meta, arrays

    def delete(self, collection: str, key: str) -> None:
        folder, key = self._folder(collection), _check(key)
        if not folder.is_dir():
            return
        (folder / f"{key}.json").unlink(missing_ok=True)
        for path in self._array_files(folder, key):
            path.unlink(missing_ok=True)
        _fsync_dir(folder)

    def keys(self, collection: str) -> list[str]:
        folder = self._folder(collection)
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.json") if _SAFE_ID.fullmatch(p.stem))

    def quarantine(self, collection: str, key: str) -> None:
        folder, key = self._folder(collection), _check(key)
        for path in [folder / f"{key}.json", *self._array_files(folder, key)]:
            if path.exists():
                os.replace(path, path.with_name(path.name + CORRUPT_SUFFIX))
        _fsync_dir(folder)
