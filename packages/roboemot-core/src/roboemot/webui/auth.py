"""Operator PIN, login sessions and the login lockout (SPEC 6 Enrollment, 11A, 12).

The PIN is stored as an scrypt hash with a per-install random salt in the app data dir (0600).
It never reaches a log or a URL. Sessions live in RAM only: a restart logs everybody out.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from roboemot.config.schema import WebUIConfig

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
TOKEN_BYTES = 32
PIN_FILE = "webui_pin.json"
PIN_DIR_MODE = 0o700
PIN_FILE_MODE = 0o600
PIN_FILE_VERSION = 1


class PinError(ValueError):
    """The PIN does not meet the format rules."""


def check_pin_format(pin: object, cfg: WebUIConfig) -> str:
    """Return ``pin`` if it is a digit string of at least ``pin_min_length``, else raise."""
    if not isinstance(pin, str) or not pin.isascii() or not pin.isdigit():
        raise PinError("pin_digits")
    if len(pin) < cfg.pin_min_length:
        raise PinError("pin_too_short")
    return pin


def _digest(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)


class PinStore:
    """The hashed operator PIN in ``<data_dir>/webui_pin.json``."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / PIN_FILE
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self.path.is_file()

    def set(self, pin: str) -> None:
        """Store the hash of ``pin`` with a fresh salt, replacing any previous one atomically."""
        salt = secrets.token_bytes(SALT_BYTES)
        record = {"v": PIN_FILE_VERSION, "salt": salt.hex(), "hash": _digest(pin, salt).hex()}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=PIN_DIR_MODE)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PIN_FILE_MODE)
            with os.fdopen(fd, "w") as fh:
                json.dump(record, fh)
            os.replace(tmp, self.path)

    def verify(self, pin: str) -> bool:
        """Constant-time comparison against the stored hash; False when no PIN is set."""
        try:
            record = json.loads(self.path.read_text())
            salt = bytes.fromhex(record["salt"])
            want = bytes.fromhex(record["hash"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return hmac.compare_digest(_digest(pin, salt), want)


@dataclass
class Session:
    csrf: str
    last_seen: float


class Sessions:
    """Idle-expiring sessions keyed by the hash of an unguessable cookie token."""

    def __init__(self, cfg: WebUIConfig) -> None:
        self.cfg = cfg
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, now: float) -> tuple[str, Session]:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        session = Session(csrf=secrets.token_urlsafe(TOKEN_BYTES), last_seen=now)
        with self._lock:
            self._items[self._key(token)] = session
            while len(self._items) > self.cfg.max_sessions:
                self._items.popitem(last=False)
        return token, session

    def get(self, token: str | None, now: float) -> Session | None:
        """The live session for ``token``, refreshed; an idle one is dropped."""
        if not token:
            return None
        key = self._key(token)
        with self._lock:
            session = self._items.get(key)
            if session is None:
                return None
            if now - session.last_seen > self.cfg.session_ttl_s:
                del self._items[key]
                return None
            session.last_seen = now
            self._items.move_to_end(key)
            return session

    def drop(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._items.pop(self._key(token), None)


class LoginLimiter:
    """Global lockout: ``pin_max_attempts`` wrong PINs lock every login for ``pin_lockout_s``.

    Global rather than per address: the page has one operator, and a per-address counter would
    let an attacker rotate addresses on the LAN.
    """

    def __init__(self, cfg: WebUIConfig) -> None:
        self.cfg = cfg
        self._failures = 0
        self._locked_until = 0.0
        self._lock = threading.Lock()

    def retry_after_s(self, now: float) -> float:
        """Seconds left of a lockout, 0 when logins are allowed."""
        with self._lock:
            return max(0.0, self._locked_until - now)

    def failed(self, now: float) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.cfg.pin_max_attempts:
                self._locked_until = now + self.cfg.pin_lockout_s
                self._failures = 0

    def succeeded(self) -> None:
        with self._lock:
            self._failures = 0
