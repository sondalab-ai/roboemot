"""Tombstones for pending records the owner deleted (SPEC 6 Catalogue import, 12 Data).

A catalogue import would bring a deleted name-only record back. The tombstone remembers that
the owner did not want it, without remembering the name: the key is an HMAC-SHA256 of the kind
and the case-folded name under a random salt kept in the registry store. No name, id, embedding
or timestamp is stored. Clearing the tombstones also drops the salt, so older keys cannot be
re-derived. Only pending records are tombstoned: forgetting an enrolled person leaves nothing
behind (``Registry.forget``).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading

import numpy as np

from .store import RegistryStore

log = logging.getLogger(__name__)

TOMBSTONES = "tombstones"
SALT_COLLECTION = "tombstone_salt"
SALT_KEY = "salt"
SALT_BYTES = 32
_NO_ARRAYS: dict[str, np.ndarray] = {}


class TombstoneSet:
    """The set of salted name hashes, in memory and in the registry store."""

    def __init__(self, store: RegistryStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._salt: bytes | None = None
        self._keys: set[str] = set()
        self._load()

    def _load(self) -> None:
        try:
            self._salt = bytes.fromhex(self.store.get(SALT_COLLECTION, SALT_KEY)[0]["salt"])
        except (KeyError, ValueError):
            self._salt = None
        self._keys = set(self.store.keys(TOMBSTONES)) if self._salt is not None else set()

    def _digest(self, kind: str, name: str) -> str:
        assert self._salt is not None
        message = f"{kind}\0{name.strip().casefold()}".encode()
        return hmac.new(self._salt, message, hashlib.sha256).hexdigest()

    def add(self, kind: str, name: str) -> None:
        """Remember that the owner deleted a pending ``kind`` record called ``name``."""
        with self._lock:
            if self._salt is None:
                salt = secrets.token_bytes(SALT_BYTES)
                self.store.put(SALT_COLLECTION, SALT_KEY, {"salt": salt.hex()}, _NO_ARRAYS)
                self._salt = salt
            key = self._digest(kind, name)
            self.store.put(TOMBSTONES, key, {"schema_version": 1}, _NO_ARRAYS)
            self._keys.add(key)

    def has(self, kind: str, name: str) -> bool:
        """True when a pending ``kind`` record called ``name`` was deleted and not cleared."""
        with self._lock:
            return self._salt is not None and self._digest(kind, name) in self._keys

    def count(self) -> int:
        with self._lock:
            return len(self._keys)

    def clear(self) -> int:
        """Drop every tombstone and the salt; returns how many there were."""
        with self._lock:
            n = len(self._keys)
            for key in self.store.keys(TOMBSTONES):
                self.store.delete(TOMBSTONES, key)
            self.store.delete(SALT_COLLECTION, SALT_KEY)
            self._keys.clear()
            self._salt = None
            return n
