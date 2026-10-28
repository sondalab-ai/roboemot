"""WebSocket client of the realtime speech route (SPEC 8, Realtime speech route; SPEC 12, 12A).

:class:`RealtimeClient` runs on its own thread with its own event loop, like ``SpeechOutput``.
``connect`` allocates a session through the :class:`~roboemot.realtime.session.SessionProvider`,
checks the URL (``connect_target``), opens the socket and sends ``session.update``; it returns
once the server answered ``session.updated``, within ``voice.realtime_route.connect_timeout_s``
for the whole sequence. Server events are parsed (``protocol.parse``) and handed to
``on_event`` on the client thread; unknown ones are only counted.

Transport rules:

- the only credential is an ``Authorization`` header, sent only when the URL rules allow it; a
  route with a key whose header resolves empty never connects (``key_unavailable``);
- a provider that cannot allocate raises ``ProviderRefused``: the client closes with its reason
  code (``provider_http_<code>``, ...). ``platform_id()`` (the opt-in) is read at each
  allocation, and the snapshot keeps whether the last allocation request carried the platform
  identifier (``platform_id_sent``, also when the allocator then failed), never its value;
- no redirect is followed (the library would re-send the header to the new host), no proxy is
  used (websockets 15 would read the system's, which would see the key and the audio; 13 and
  14 have no proxy support), no compression, ``max_size`` is
  ``voice.realtime_route.max_message_bytes`` (a bigger message closes the socket, counted as
  ``oversize``);
- keepalive pings every ``send_timeout_s``, answered within ``send_timeout_s``;
- a send stuck longer than ``send_timeout_s`` aborts the socket (``send_stuck``); a socket lost
  while a write is blocked (the keepalive gave up first) is ``send_stuck`` too;
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
import inspect
import itertools
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
from ..llm.pseudonyms import url_is_local
from . import protocol
from .session import (
    NO_KEY,
    AllocationRequest,
    ConnectTarget,
    ProviderRefused,
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
# websockets 15 uses the system's proxy settings unless told not to; 13 and 14 have no proxy
# support (and would hand an unknown ``proxy`` argument to the socket layer), so there is
# nothing to turn off there. The realtime extra allows >=13,<16 (the robot's version is kept).
NO_PROXY: dict[str, Any] = (
    {"proxy": None} if "proxy" in inspect.signature(ws_connect.__init__).parameters else {}
)

AUTH_HEADER = "Authorization"
# Sent events whose type a server error can name by event_id (the client sends few per turn).
SENT_TYPES_KEPT = 64
BEARER = "Bearer "
CLOSE_TOO_BIG = 1009  # RFC 6455: message too big
BASE64_CHARS = 4  # every 4 base64 characters
BASE64_BYTES = 3  # carry 3 bytes
EVENT_PREFIX = "ev_"  # the client's own event ids (a refusal echoes the one it refuses)
# the server events of a response (they move the response watchdog)
RESPONSE_EVENTS = (
    protocol.ResponseCreated,
    protocol.AudioDelta,
    protocol.AudioDone,
    protocol.ReplyTranscript,
    protocol.FunctionCall,
    protocol.ResponseDone,
)
STATES = ("idle", "connecting", "connected", "closed")


class _NoRedirect(ws_connect):
    """``connect`` that never follows a redirect: a 3xx answer is a failed handshake."""

    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


class RealtimeClient:
    """One realtime session at a time over one WebSocket (see the module docstring).

    ``auth_headers`` (required) returns the route's ``Authorization`` header
    (``Router.auth_headers``); anything but that header is dropped. A route with a key whose
    header resolves empty (or cannot be read) is refused before anything is allocated or sent
    (``key_unavailable``); the header is sent only when the URL rules allow it. ``on_event``
    runs on the client thread for every server event of the subset, ``Malformed`` included;
    ``Unknown`` events are only counted. An exception it raises is counted and logged by type.
    ``on_close(reason)`` runs when a session fails or closes (a connect that fails, a socket
    lost, a stall, a stuck send, ``close``), on whichever thread saw it. ``may_send()`` is the
    owner's gate, asked right before ``session.update`` (the first data) is sent after the
    handshake; False refuses the session (``reconfigured``). While
    ``local_only()`` is true (``privacy.pseudonyms_to_cloud``: real names may not reach a cloud
    host) a connect host that is not local is refused (``pseudonyms``) before the socket opens:
    a provider may allocate a cloud host for a local ``base_url`` (SPEC 12).
    """

    def __init__(
        self,
        route: LLMRoute,
        settings: RealtimeRouteConfig,
        provider: SessionProvider,
        *,
        auth_headers: Callable[[], dict[str, str]],
        on_event: Callable[[protocol.ServerEvent], None] = lambda _ev: None,
        on_close: Callable[[str], None] = lambda _reason: None,
        local_only: Callable[[], bool] = lambda: False,
        may_send: Callable[[], bool] = lambda: True,
        platform_id: Callable[[], bool] = lambda: False,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.route, self.settings, self.provider = route, settings, provider
        self._auth_headers = auth_headers
        self._on_event = on_event
        self._on_close = on_close
        self._local_only = local_only
        self._may_send = may_send
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
        self._awaiting_id: str | None = None  # event_id of that response.create
        self._event_ids = itertools.count(1)
        self._watchdog: asyncio.TimerHandle | None = None
        self._dropping: asyncio.Future[None] | None = None
        self._sending = False  # a write is in flight (a socket lost meanwhile is a stuck send)
        self._target: ConnectTarget | None = None
        self._refused_host: str | None = None  # the connect host the key was refused for
        # the last server error: its code and the type of our event it names (never its text)
        self._server_error: tuple[str | None, str | None] = (None, None)
        self._sent_types: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._platform_id_sent = False  # the last allocation request carried the identifier
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

    def _submit_send(self, coro: Any) -> concurrent.futures.Future[bool]:
        """``_submit`` for a send: without a running loop (never started, or closed) the future
        is already resolved to False and the send is counted as dropped."""
        try:
            return self._submit(coro)
        except RuntimeError:
            self._count("dropped")
            done: concurrent.futures.Future[bool] = concurrent.futures.Future()
            done.set_result(False)
            return done

    def connect(self, session_update: dict[str, Any]) -> bool:
        """Open a session and wait for ``session.updated`` (blocking; never call it from the
        client thread). False on any failure, with ``reason`` set to a code. The wait is bounded
        by ``connect_timeout_s`` (plus ``send_timeout_s`` for the socket's own close): a close
        from another thread stops the loop, and the connect's future would never resolve."""
        self.start()
        future = self._submit(self._connect(session_update))
        s = self.settings
        try:
            return bool(future.result(s.connect_timeout_s + s.send_timeout_s))
        except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            future.cancel()
            with self._lock:
                fire = self.state != "closed"
                if fire:
                    self._failures["connect_timeout"] += 1
                    self.state, self.reason = "closed", "connect_timeout"
            if fire:
                self._closed_cb("connect_timeout")
            return False

    def send(self, event: dict[str, Any]) -> concurrent.futures.Future[bool]:
        """Queue a client event (thread-safe). The future is True once it is written, False when
        there is no open socket or the send failed (a stuck send aborts the socket). The client
        does not judge what it sends: audio, like any data, leaves only through the owner's
        send gate (``RealtimeRoute``, SPEC 12)."""
        return self._submit_send(self._send(event))

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
                "refused_host": self._refused_host,  # url_key_host: a host only, never a path
                "server_error_code": self._server_error[0],
                "server_error_about": self._server_error[1],
                "local": target.local if target else None,
                "key_sent": target.send_key if target else False,
                "platform_id_sent": self._platform_id_sent,
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
        self._closed_cb(reason)

    def _closed_cb(self, reason: str) -> None:
        try:
            self._on_close(reason)
        except Exception as exc:
            self._count("callback_errors")
            log.warning("realtime route %s: close handler: %s", self.route.name, type(exc).__name__)

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counts[key] += n

    def _note_platform_id(self) -> None:
        """The provider is about to send the platform identifier (the report says so)."""
        with self._lock:
            self._platform_id_sent = True

    def _note_sent(self, event: dict[str, Any]) -> None:
        """Remember the type of a client event by its ``event_id``, so a server error that names
        it can say which of our events it is about (the last ``SENT_TYPES_KEPT``)."""
        with self._lock:
            self._sent_types[event["event_id"]] = str(event.get("type"))
            while len(self._sent_types) > SENT_TYPES_KEPT:
                self._sent_types.popitem(last=False)

    def _key(self) -> dict[str, str]:
        """Only the ``Authorization`` header of the route's key, or nothing."""
        value = self._auth_headers().get(AUTH_HEADER)
        return {AUTH_HEADER: value} if value else {}

    async def _connect(self, session_update: dict[str, Any]) -> bool:
        await self._close(None)
        with self._lock:
            self.state, self.reason, self._target = "connecting", None, None
            self._refused_host = None
            self._server_error = (None, None)
            self._sent_types.clear()
            self._counts["connects"] += 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.connect_timeout_s
        try:
            async with asyncio.timeout_at(deadline):
                ws = await self._open(loop, deadline)
                self._ws = ws
                # the owner's gate right before the first data leaves (after the handshake): a
                # reload meanwhile (pseudonyms on, the route changed) sends nothing (SPEC 12)
                if self._local_only() and not ws_target_local(self._target):
                    raise _SessionRefused("pseudonyms")
                if not await asyncio.get_running_loop().run_in_executor(None, self._may_send):
                    raise _SessionRefused("reconfigured")
                update = {"event_id": f"{EVENT_PREFIX}{next(self._event_ids)}", **session_update}
                self._note_sent(update)
                await ws.send(protocol.encode(update))
                self._count("sent")
                await self._await_session(ws)
        except _KeyUnavailable:
            log.warning("realtime route %s: its key is not available", self.route.name)
            self._fail("key_unavailable")
        except UrlRefused as exc:
            with self._lock:
                self._refused_host = exc.host
            log.warning(
                "realtime route %s: connect URL refused (%s)%s",
                self.route.name,
                exc.reason,
                f": host {exc.host} may not get the key" if exc.host else "",
            )
            self._fail(f"url_{exc.reason}")
        except ProviderRefused as exc:
            log.warning("realtime route %s: no session allocated (%s)", self.route.name, exc.reason)
            self._fail(exc.reason)
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
        try:
            headers = self._key()
        except Exception:
            headers = {}
        key = headers.get(AUTH_HEADER, "")
        if self.route.api_key not in NO_KEY and not key.removeprefix(BEARER).strip():
            raise _KeyUnavailable  # never connect a keyed route without its key
        if self._local_only() and not url_is_local(self.route.base_url, self.route.locality):
            raise _SessionRefused("pseudonyms")  # the provider's endpoint gets nothing either
        with self._lock:
            self._platform_id_sent = False
        request = AllocationRequest(
            self.route,
            key.removeprefix(BEARER) or None,
            self._platform_id() is True,  # the opt-in now, not when the client was built
            max(0.0, deadline - loop.time()),
            self._http_client_factory,
            self._note_platform_id,
            self.settings.max_message_bytes,
        )
        allocation = await self.provider.allocate(request)
        target = connect_target(self.route, allocation, tuple(self.provider.key_hosts))
        with self._lock:
            self._target = target
        if self._local_only() and not target.local:
            raise _SessionRefused("pseudonyms")  # nothing is sent to the cloud host
        s = self.settings
        return await _NoRedirect(
            target.url,
            additional_headers=headers if target.send_key else None,
            user_agent_header=None,
            compression=None,
            **NO_PROXY,
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
                log.warning(
                    "realtime route %s: server error %s about %s",
                    self.route.name,
                    event.code,
                    self._server_error[1],
                )
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
                self._server_error = (event.code, self._sent_types.get(event.event_id or ""))
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
                if isinstance(event, protocol.Unknown):
                    continue  # counted only: no progress, not handed out
                if self._awaiting:
                    self._watch(event)
                self._deliver(event)
        except ConnectionClosed as exc:
            sent = exc.sent.code if exc.sent is not None else None
            if sent == CLOSE_TOO_BIG:
                reason = "oversize"
                self._count("oversize")
                log.warning("realtime route %s: server message over the size cap", self.route.name)
            elif self._sending and exc.rcvd is None:
                # Lost while a write was blocked (the keepalive gave up first): a stuck send,
                # whichever timer fired first. Counted here only when the send's own timeout
                # has not already dropped this socket.
                reason = "send_stuck"
                if self._ws is ws:
                    self._count("send_stuck")
                    log.warning("realtime route %s: send stuck, socket lost", self.route.name)
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

    def _watch(self, event: protocol.ServerEvent) -> None:
        """The response watchdog while a response is due: only events of a response move it
        (``response.done`` ends it, others re-arm it with ``stall_timeout_s``); an ``error``
        ends it only when it echoes that ``response.create``'s ``event_id`` (a refusal: no
        response is coming). Other events (``session.updated``, other errors) leave it."""
        if isinstance(event, protocol.ServerError):
            if event.event_id is not None and event.event_id == self._awaiting_id:
                self._awaiting = False
                self._arm(None)
            return
        if isinstance(event, RESPONSE_EVENTS):
            done = event.type == protocol.RESPONSE_DONE
            self._awaiting = not done
            self._arm(None if done else self.route.stall_timeout_s)

    async def _send(self, event: dict[str, Any]) -> bool:
        assert self._send_lock is not None
        async with self._send_lock:
            ws = self._ws
            if ws is None or self.state != "connected":
                self._count("dropped")
                return False
            if "event_id" not in event:  # every client event carries one
                event = {**event, "event_id": f"{EVENT_PREFIX}{next(self._event_ids)}"}
            self._note_sent(event)
            if event.get("type") == protocol.RESPONSE_CREATE:
                # armed before the write: a fast server may answer while the send still drains
                self._awaiting, self._awaiting_id = True, event["event_id"]
                self._arm(self.route.first_token_timeout_s)
            self._sending = True
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
            finally:
                self._sending = False
            if _lost_while_writing(ws):
                # The library wakes a blocked write when the socket is lost (the keepalive gave
                # up while the send waited): the data never left, so this is a stuck send,
                # recorded here unless the receive loop already did.
                if self._ws is ws:
                    self._count("send_stuck")
                    log.warning("realtime route %s: send stuck, socket lost", self.route.name)
                    self._fail("send_stuck")
                    await self._drop_socket()
                return False
        self._count("sent")
        sent_audio = audio_bytes(event)
        if sent_audio:
            self._count("audio_bytes_sent", sent_audio)
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
                closing = self.state != "closed"
                if closing:
                    self.state, self.reason = "closed", reason
            if closing:
                self._closed_cb(reason)


def audio_bytes(event: dict[str, Any]) -> int:
    """The PCM bytes an ``input_audio_buffer.append`` carries (0 for any other event), read
    from the length of its base64 text without decoding it."""
    if event.get("type") != protocol.AUDIO_APPEND:
        return 0
    text = event.get("audio", "")
    padding = len(text) - len(text.rstrip("="))
    return len(text) // BASE64_CHARS * BASE64_BYTES - padding


def ws_target_local(target: ConnectTarget | None) -> bool:
    return target is not None and target.local


def _lost_while_writing(ws: ClientConnection) -> bool:
    """Whether the socket went down without a close frame from the server while a write was
    waiting: an aborted transport (keepalive timeout, network loss), not a server close."""
    return ws.transport.is_closing() and ws.protocol.close_rcvd is None


class _KeyUnavailable(Exception):
    """A route with a key whose header resolved empty (internal)."""


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
