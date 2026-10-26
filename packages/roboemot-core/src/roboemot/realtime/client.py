"""WebSocket client of the realtime speech route (SPEC 8, Realtime speech route; SPEC 12, 12A).

:class:`RealtimeClient` runs on its own thread with its own event loop, like ``SpeechOutput``.
``connect`` allocates a session through the :class:`~roboemot.realtime.session.SessionProvider`,
checks the URL (``connect_target``), opens the socket and sends ``session.update``; it returns
once the server answered ``session.updated``, within ``voice.realtime_route.connect_timeout_s``
for the whole sequence. Server events are parsed (``protocol.parse``) and handed to
``on_event`` on the client thread.

Transport rules:

- the only credential is an ``Authorization`` header, sent only when the URL rules allow it;
- no redirect is followed (the library would re-send the header to the new host), no proxy is
  used (the system's would see the key and the audio), no compression, ``max_size`` is
  ``voice.realtime_route.max_message_bytes`` (a bigger message closes the socket, counted as
  ``oversize``);
- keepalive pings every ``send_timeout_s``, answered within ``send_timeout_s``;
- a send stuck longer than ``send_timeout_s`` aborts the socket (``send_stuck``);
- after ``response.create`` the first server event must come within the route's
  ``first_token_timeout_s``, and every later one, until ``response.done``, within its
  ``stall_timeout_s``; otherwise the socket is closed (``stall``).

Nothing here logs or reports a key, the URL's path or query, audio, text or an identifier: the
library's own logger is silenced, exceptions are logged by type, and ``snapshot`` holds states,
reason codes, counts and the ``scheme://host:port`` endpoint only.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .. import shutdown
from ..config.schema import LLMRoute, RealtimeRouteConfig
from . import protocol
from .session import (
    AllocationRequest,
    ConnectTarget,
    SessionProvider,
    UrlRefused,
    connect_target,
)

log = logging.getLogger(__name__)
# The WebSocket library logs the request line (path and query), every header (the key) and
# frame previews (audio, text) at DEBUG: its logger is held silent whatever the root level is.
WIRE_LOG = logging.getLogger("roboemot.realtime.wire")
WIRE_LOG.setLevel(logging.CRITICAL + 1)
WIRE_LOG.propagate = False
logging.getLogger("websockets").setLevel(logging.WARNING)

AUTH_HEADER = "Authorization"
BEARER = "Bearer "
CLOSE_TOO_BIG = 1009  # RFC 6455: message too big
STATES = ("idle", "connecting", "connected", "closed")


class _NoRedirect(ws_connect):
    """``connect`` that never follows a redirect: a 3xx answer is a failed handshake."""

    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


class RealtimeClient:
    """One realtime session at a time over one WebSocket (see the module docstring).

    ``auth_headers`` returns the route's ``Authorization`` header (``Router.auth_headers``),
    called only when the URL rules allow the key; anything but that header is dropped.
    ``on_event`` runs on the client thread for every server event, ``Malformed`` and
    ``Unknown`` included; an exception it raises is counted and logged by type.
    """

    def __init__(
        self,
        route: LLMRoute,
        settings: RealtimeRouteConfig,
        provider: SessionProvider,
        *,
        auth_headers: Callable[[], dict[str, str]] = dict,
        on_event: Callable[[protocol.ServerEvent], None] = lambda _ev: None,
        platform_id: bool = False,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.route, self.settings, self.provider = route, settings, provider
        self._auth_headers = auth_headers
        self._on_event = on_event
        self._platform_id = platform_id
        self._http_client_factory = http_client_factory
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._ws: ClientConnection | None = None
        self._send_lock: asyncio.Lock | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._awaiting = False  # a response.create was sent and its response.done not seen
        self._watchdog: asyncio.TimerHandle | None = None
        self._dropping: asyncio.Future[None] | None = None
        self._target: ConnectTarget | None = None
        self.state = "idle"
        self.reason: str | None = None
        self._counts: collections.Counter[str] = collections.Counter()
        self._failures: collections.Counter[str] = collections.Counter()
        self._malformed: collections.Counter[str] = collections.Counter()

    def __repr__(self) -> str:
        return f"RealtimeClient(route={self.route.name!r}, state={self.state!r})"

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the client thread and its loop (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="realtime-client", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        loop = self._loop = asyncio.new_event_loop()
        loop.set_exception_handler(_quiet_exception)
        asyncio.set_event_loop(loop)
        self._send_lock = asyncio.Lock()
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _submit(self, coro: Any) -> concurrent.futures.Future[Any]:
        if self._loop is None:
            coro.close()
            raise RuntimeError("realtime client not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def connect(self, session_update: dict[str, Any]) -> bool:
        """Open a session and wait for ``session.updated`` (blocking; never call it from the
        client thread). False on any failure, with ``reason`` set to a code."""
        self.start()
        return bool(self._submit(self._connect(session_update)).result())

    def send(self, event: dict[str, Any]) -> concurrent.futures.Future[bool]:
        """Queue a client event (thread-safe). The future is True once it is written, False when
        there is no open socket or the send failed (a stuck send aborts the socket)."""
        return self._submit(self._send(event))

    def send_audio(self, pcm: bytes) -> concurrent.futures.Future[bool]:
        """Queue ``input_audio_buffer.append`` with PCM16 bytes (thread-safe)."""
        event = protocol.audio_append(pcm)
        return self._submit(self._send(event, audio_bytes=len(pcm)))

    def close(self) -> None:
        """Close the socket and stop the thread (bounded by ``send_timeout_s`` each)."""
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        with contextlib.suppress(RuntimeError, concurrent.futures.TimeoutError):
            self._submit(self._close("closed")).result(self.settings.send_timeout_s * 2)
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(loop.stop)
        shutdown.join(thread, self.settings.send_timeout_s)
        self._thread = self._loop = None
        self._ready.clear()

    # ----------------------------------------------------------------- report

    def snapshot(self) -> dict[str, Any]:
        """States, reason codes and counts for the debug report (SPEC 12A): never a key, a
        path or query, audio, text or an identifier."""
        with self._lock:
            target = self._target
            return {
                "state": self.state,
                "reason": self.reason,
                "endpoint": target.origin if target else None,
                "local": target.local if target else None,
                "key_sent": target.send_key if target else False,
                "key_hosts": list(self.provider.key_hosts),
                "failures": dict(self._failures),
                "malformed": dict(self._malformed),
                **{k: self._counts[k] for k in _COUNTERS},
            }

    # ----------------------------------------------------------------- loop side

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._failures[reason] += 1
            self.reason = reason
            self.state = "closed"

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counts[key] += n

    def _key(self) -> dict[str, str]:
        """Only the ``Authorization`` header of the route's key, or nothing."""
        value = self._auth_headers().get(AUTH_HEADER)
        return {AUTH_HEADER: value} if value else {}

    async def _connect(self, session_update: dict[str, Any]) -> bool:
        await self._close(None)
        with self._lock:
            self.state, self.reason, self._target = "connecting", None, None
            self._counts["connects"] += 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.connect_timeout_s
        try:
            async with asyncio.timeout_at(deadline):
                ws = await self._open(loop, deadline)
                self._ws = ws
                await ws.send(protocol.encode(session_update))
                self._count("sent")
                await self._await_session(ws)
        except UrlRefused as exc:
            log.warning("realtime route %s: connect URL refused (%s)", self.route.name, exc.reason)
            self._fail(f"url_{exc.reason}")
        except TimeoutError:
            log.warning("realtime route %s: no session within the connect timeout", self.route.name)
            self._fail("connect_timeout")
        except _SessionRefused as exc:
            log.warning("realtime route %s: session refused (%s)", self.route.name, exc.reason)
            self._fail(exc.reason)
        except Exception as exc:  # a library error can embed the URL: the type only
            log.warning(
                "realtime route %s: connect failed: %s", self.route.name, type(exc).__name__
            )
            self._fail("connect_failed")
        else:
            with self._lock:
                self.state = "connected"
            self._receiver = asyncio.ensure_future(self._receive(ws))
            log.info("realtime route %s: connected to %s", self.route.name, self._origin())
            return True
        await self._drop_socket()
        return False

    def _origin(self) -> str:
        return self._target.origin if self._target else ""

    async def _open(self, loop: asyncio.AbstractEventLoop, deadline: float) -> ClientConnection:
        headers = self._key()
        key = headers.get(AUTH_HEADER, "")
        request = AllocationRequest(
            self.route,
            key.removeprefix(BEARER) or None,
            self._platform_id,
            max(0.0, deadline - loop.time()),
            self._http_client_factory,
        )
        allocation = await self.provider.allocate(request)
        target = connect_target(self.route, allocation, tuple(self.provider.key_hosts))
        with self._lock:
            self._target = target
        s = self.settings
        return await _NoRedirect(
            target.url,
            additional_headers=headers if target.send_key else None,
            user_agent_header=None,
            compression=None,
            proxy=None,
            open_timeout=max(0.0, deadline - loop.time()),
            ping_interval=s.send_timeout_s,
            ping_timeout=s.send_timeout_s,
            close_timeout=s.send_timeout_s,
            max_size=s.max_message_bytes,
            logger=WIRE_LOG,
        )

    async def _await_session(self, ws: ClientConnection) -> None:
        """Read until ``session.updated``; a server error before it refuses the session."""
        while True:
            event = self._parse(await ws.recv())
            self._deliver(event)
            if event.type == protocol.SESSION_UPDATED:
                return
            if isinstance(event, protocol.ServerError):
                raise _SessionRefused("server_error")

    def _parse(self, frame: str | bytes) -> protocol.ServerEvent:
        event = protocol.parse(frame, self.settings.max_message_bytes)
        with self._lock:
            self._counts["received"] += 1
            if isinstance(event, protocol.Unknown):
                self._counts["unknown"] += 1
            elif isinstance(event, protocol.Malformed):
                self._malformed[event.reason] += 1
            elif isinstance(event, protocol.ServerError):
                self._counts["server_errors"] += 1
            elif isinstance(event, protocol.AudioDelta):
                self._counts["audio_bytes_received"] += len(event.pcm)
        return event

    def _deliver(self, event: protocol.ServerEvent) -> None:
        try:
            self._on_event(event)
        except Exception as exc:
            self._count("callback_errors")
            log.warning("realtime route %s: event handler: %s", self.route.name, type(exc).__name__)

    def _arm(self, timeout_s: float | None) -> None:
        """(Re)start the response watchdog: ``None`` stops it."""
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if timeout_s is not None and self._loop is not None:
            self._watchdog = self._loop.call_later(timeout_s, self._stalled)

    def _stalled(self) -> None:
        """No server event in time while a response was due: close the socket (``stall``)."""
        self._watchdog = None
        self._awaiting = False
        if self._ws is None:
            return
        self._count("stalls")
        log.warning("realtime route %s: response stalled", self.route.name)
        self._fail("stall")
        self._dropping = asyncio.ensure_future(self._drop_socket())  # held: not collected early

    async def _receive(self, ws: ClientConnection) -> None:
        """The receive loop: parse, track the response in progress, hand events out."""
        reason = "closed"
        try:
            while True:
                event = self._parse(await ws.recv())
                if self._awaiting and not isinstance(event, protocol.Malformed):
                    done = event.type == protocol.RESPONSE_DONE
                    self._awaiting = not done
                    self._arm(None if done else self.route.stall_timeout_s)
                self._deliver(event)
        except ConnectionClosed as exc:
            sent = exc.sent.code if exc.sent is not None else None
            if sent == CLOSE_TOO_BIG:
                reason = "oversize"
                self._count("oversize")
                log.warning("realtime route %s: server message over the size cap", self.route.name)
            else:
                reason = "closed_by_server" if exc.rcvd is not None else "connection_lost"
                log.info("realtime route %s: socket closed (%s)", self.route.name, reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = "receive_failed"
            log.warning("realtime route %s: receive: %s", self.route.name, type(exc).__name__)
        self._arm(None)
        self._awaiting = False
        if self._ws is ws:
            self._fail(reason)
            self._receiver = None  # ending on its own: nothing to cancel
            await self._drop_socket()

    async def _send(self, event: dict[str, Any], audio_bytes: int = 0) -> bool:
        assert self._send_lock is not None
        async with self._send_lock:
            ws = self._ws
            if ws is None or self.state != "connected":
                self._count("dropped")
                return False
            if event.get("type") == protocol.RESPONSE_CREATE:
                # armed before the write: a fast server may answer while the send still drains
                self._awaiting = True
                self._arm(self.route.first_token_timeout_s)
            try:
                await asyncio.wait_for(
                    ws.send(protocol.encode(event)), self.settings.send_timeout_s
                )
            except TimeoutError:
                self._count("send_stuck")
                log.warning("realtime route %s: send stuck, socket aborted", self.route.name)
                self._fail("send_stuck")
                await self._drop_socket()
                return False
            except ConnectionClosed:
                self._count("dropped")
                return False
            except Exception as exc:
                log.warning("realtime route %s: send: %s", self.route.name, type(exc).__name__)
                self._fail("send_failed")
                await self._drop_socket()
                return False
        self._count("sent")
        if audio_bytes:
            self._count("audio_bytes_sent", audio_bytes)
        return True

    async def _drop_socket(self) -> None:
        """Abort the socket at once (no close handshake) and forget it."""
        self._arm(None)
        self._awaiting = False
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.transport.abort()
        receiver, self._receiver = self._receiver, None
        current = asyncio.current_task()
        if receiver is not None and receiver is not current and not receiver.done():
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver

    async def _close(self, reason: str | None) -> None:
        """Close the socket with a close handshake bounded by ``send_timeout_s``."""
        ws = self._ws
        receiver, self._receiver = self._receiver, None
        if receiver is not None and not receiver.done():
            receiver.cancel()
            with contextlib.suppress(BaseException):
                await receiver
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), self.settings.send_timeout_s)
            except Exception:
                with contextlib.suppress(Exception):
                    ws.transport.abort()
            self._ws = None
        self._arm(None)
        self._awaiting = False
        if reason is not None:
            with self._lock:
                if self.state != "closed":
                    self.state, self.reason = "closed", reason


class _SessionRefused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_COUNTERS = (
    "connects",
    "sent",
    "received",
    "dropped",
    "unknown",
    "server_errors",
    "oversize",
    "stalls",
    "send_stuck",
    "callback_errors",
    "audio_bytes_sent",
    "audio_bytes_received",
)


def _quiet_exception(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """The loop's exception handler: the exception's type only (a message can carry a URL)."""
    exc = context.get("exception")
    log.warning("realtime client loop: %s", type(exc).__name__ if exc else "error")
