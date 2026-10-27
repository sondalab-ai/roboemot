"""Operator PIN, login sessions and the login limits (SPEC 6 Enrollment, 11A, 12).

The PIN is stored as an scrypt hash with a per-install random salt in the app data dir (0600).
It never reaches a log or a URL. Sessions live in RAM only: a restart logs everybody out, and
so does a new PIN (sessions older than the PIN file are refused).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from collections import OrderedDict
from collections.abc import Callable
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
MAX_TRACKED_CLIENTS = 256
LOOPBACK = frozenset({"127.0.0.1", "::1"})


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

    def version(self) -> int:
        """Changes whenever the PIN file is replaced (0 without one); ties sessions to a PIN."""
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return 0

    def set(self, pin: str) -> None:
        """Store the hash of ``pin`` with a fresh salt, replacing any previous one atomically.

        The temp file is created exclusively without following links, forced to 0600 and
        synced, then renamed over the old file; the directory is synced afterwards.
        """
        salt = secrets.token_bytes(SALT_BYTES)
        record = {"v": PIN_FILE_VERSION, "salt": salt.hex(), "hash": _digest(pin, salt).hex()}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=PIN_DIR_MODE)
            tmp = self.path.with_suffix(".tmp")
            tmp.unlink(missing_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            fd = os.open(tmp, flags, PIN_FILE_MODE)
            try:
                os.fchmod(fd, PIN_FILE_MODE)
                with os.fdopen(fd, "w") as fh:
                    json.dump(record, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            os.replace(tmp, self.path)
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

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
    created: float
    last_seen: float
    pin_version: int


class Sessions:
    """Sessions keyed by the hash of an unguessable cookie token.

    A session ends after ``session_ttl_s`` idle, after ``session_max_s`` in all, and when the
    PIN file it was created under has been replaced. ``web`` returns the current web UI config,
    read on every call, so a value changed at runtime applies at once (SPEC 11A, 12).
    """

    def __init__(self, web: Callable[[], WebUIConfig]) -> None:
        self._web = web
        self._items: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def cfg(self) -> WebUIConfig:
        return self._web()

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create(self, now: float, pin_version: int) -> tuple[str, Session]:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        session = Session(secrets.token_urlsafe(TOKEN_BYTES), now, now, pin_version)
        with self._lock:
            self._items[self._key(token)] = session
            while len(self._items) > self.cfg.max_sessions:
                self._items.popitem(last=False)
        return token, session

    def get(
        self, token: str | None, now: float, pin_version: int, *, refresh: bool = True
    ) -> Session | None:
        """The live session for ``token``, refreshed unless ``refresh`` is False (a passive poll
        must not keep an idle session alive); an expired or stale one is dropped."""
        if not token:
            return None
        key = self._key(token)
        with self._lock:
            session = self._items.get(key)
            if session is None:
                return None
            cfg = self.cfg  # one read: a reload in between cannot mix two configs
            if (
                now - session.last_seen > cfg.session_ttl_s
                or now - session.created > cfg.session_max_s
                or session.pin_version != pin_version
            ):
                del self._items[key]
                return None
            if refresh:
                session.last_seen = now
                self._items.move_to_end(key)
            return session

    def drop(self, token: str | None) -> None:
        if token:
            with self._lock:
                self._items.pop(self._key(token), None)


@dataclass
class _Client:
    failures: int = 0
    blocked_until: float = 0.0


@dataclass
class _Pool:
    """A guess budget: unrefunded attempts, and when the lockout it triggered ends."""

    failures: int = 0
    locked_until: float = 0.0


class LoginLimiter:
    """PIN guess limits: a global LAN budget plus a per-client back-off, and a loopback budget.

    ``reserve`` checks the limits and counts the attempt in one locked step, before the PIN is
    hashed, so parallel guesses cannot all slip under the budget; ``succeeded`` refunds it.
    ``pin_max_attempts`` unrefunded LAN attempts lock every LAN login for ``pin_lockout_s``.
    Each client also waits ``client_backoff_s`` doubling per failure, so one noisy address is
    slowed long before it exhausts the global budget for the owner.

    Loopback (another process on the robot's own host) has its own finite budget,
    ``loopback_max_attempts``, with its own back-off and lockout. The two budgets are
    independent: loopback failures never lock LAN clients and a LAN lockout never blocks
    loopback. A new PIN file (``sync``) or an app restart clears everything. ``web`` returns the
    current web UI config, read on every attempt, so a tighter budget applies at once.
    """

    def __init__(self, web: Callable[[], WebUIConfig], pin_version: int = 0) -> None:
        self._web = web
        self._lan = _Pool()
        self._local = _Pool()
        self._local_client = _Client()
        self._clients: OrderedDict[str, _Client] = OrderedDict()
        self._version = pin_version
        self._lock = threading.Lock()

    @property
    def cfg(self) -> WebUIConfig:
        return self._web()

    def sync(self, pin_version: int) -> None:
        """Forget all limits when the PIN file changed (the CLI ``set-pin`` clears a lockout)."""
        with self._lock:
            if pin_version != self._version:
                self._version = pin_version
                self._lan, self._local = _Pool(), _Pool()
                self._local_client = _Client()
                self._clients.clear()

    def _slot(self, ip: str, cfg: WebUIConfig | None = None) -> tuple[_Pool, _Client | None, int]:
        """The pool, the client record (None if unknown) and the budget that apply to ``ip``."""
        cfg = cfg or self.cfg
        if ip in LOOPBACK:
            return self._local, self._local_client, cfg.loopback_max_attempts
        return self._lan, self._clients.get(ip), cfg.pin_max_attempts

    def reserve(self, ip: str, now: float) -> float:
        """0 if the attempt may go ahead (and is now counted), else seconds to wait."""
        cfg = self.cfg  # one read: a reload in between cannot mix two configs
        with self._lock:
            pool, client, budget = self._slot(ip, cfg)
            client = client or _Client()
            wait = max(pool.locked_until - now, client.blocked_until - now, 0.0)
            if wait > 0:
                return wait
            client.failures += 1
            step = cfg.client_backoff_s * 2 ** (client.failures - 1)
            client.blocked_until = now + min(step, cfg.client_backoff_max_s)
            if ip not in LOOPBACK:
                self._clients[ip] = client
                self._clients.move_to_end(ip)
                while len(self._clients) > MAX_TRACKED_CLIENTS:
                    self._clients.popitem(last=False)
            pool.failures += 1
            if pool.failures >= budget:
                pool.locked_until = now + cfg.pin_lockout_s
                pool.failures = 0
            return 0.0

    def succeeded(self, ip: str) -> None:
        """Refund the attempts of ``ip``: the PIN was right."""
        with self._lock:
            pool, client, _ = self._slot(ip)
            if client is None:
                return
            pool.failures = max(pool.failures - client.failures, 0)
            if ip in LOOPBACK:
                self._local_client = _Client()
            else:
                self._clients.pop(ip, None)

    def cancel(self, ip: str) -> None:
        """Refund the single attempt just reserved (it was never checked)."""
        with self._lock:
            pool, client, _ = self._slot(ip)
            if client is not None and client.failures > 0:
                client.failures -= 1
                client.blocked_until = 0.0
                pool.failures = max(pool.failures - 1, 0)
