"""The realtime speech route at runtime (SPEC 8, Realtime speech route; SPEC 12A, Links and
Diagnostics).

:class:`RealtimeRoute` owns the client's lifecycle: it connects at start, closes at sleep and at
the end of a conversation session, and reopens on wake and when a turn needs it. It decides
whether the route may answer at all (``blocked``: a reason code), reports the session to the
``realtime`` link of the network monitor (so a connect is retried with the
``connectivity.network`` backoff), charges the route's daily token budget, cancels a response
the robot never asked for, and keeps the numbers of the ``roboemot.realtime_route`` debug
section.

Session state is one small state machine guarded by one lock (``_lock``): the client (or
none), the connect in flight (or none), a revision bumped by every ``apply_config``, a close
counter bumped by every ``close``/``close_soon``, and the requests in flight. Whether a session
may be used (``_fits``: the route unchanged and not blocked, the pseudonym locality, no close
since the connect began) is decided inside that lock on every path that hands a session out or
uses it, with the newest configuration (``_with_config``). The network is never awaited while
holding it, and the configuration stack is never read while holding it (the stack notifies its
subscribers, ``apply_config`` among them, under its own lock).

Lock order: the worker's ``_rt_lock`` (asyncio) -> ``_connect_lock`` (one connect at a time,
held across the network) -> ``_lock`` -> the client's own lock, the router's and the network
monitor's locks (``_fits`` calls the router, whose reachability check reads the monitor, under
``_lock``). None of those calls back into the route; the client never calls the route while
holding its lock, and its callbacks carry a per-client token (an old client's late callback is
dropped).

Privacy is enforced at the send point (SPEC 12): every data-carrying send (``session.update``,
text items, tool results, ``response.create``) passes one gate that reads the committed
configuration and runs ``_fits`` atomically with the send under ``_lock``; refused, nothing is
sent and the session is closed. The stack commits before it notifies ``apply_config``, so the
gate, not the lifecycle, is what closes that window. The client asks the same gate right before
it sends ``session.update`` after the handshake. ``response.cancel`` and item deletes and
truncations carry no data and are not gated.

Responses are correlated explicitly (SPEC 8, Realtime speech route). Each ``response.create``
carries an ``event_id``; an ``error`` that echoes it refuses that request, any other ``error``
is only counted. ``response.created`` binds the oldest request in flight (the server answers in
order, and a new request waits until no abandoned one is in flight, ``busy``); every later
event goes to the run that owns its ``response_id`` and is dropped (counted) otherwise. A run cut
by a barge-in abandons its request: its response is cancelled and its ``response.done`` charged
then. A new request waits while an abandoned one is in flight (``turn.wait_in_order``, before
the turn's first send and before each ``response.create``); past the route's
``stall_timeout_s`` with no event it calls ``expire``, and one that never got its
``response.created`` resets the session. A detached session's requests are settled at the
detach (charged once, their runs ended ``closed``): its close comes later, or never. Nothing
here logs or reports a key, a URL's path or query, text, audio or an identifier (SPEC 12).
"""

from __future__ import annotations

import collections
import contextlib
import datetime as dt
import importlib.util
import itertools
import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig
from ..health.network import route_endpoint
from ..llm.prompt import CHARS_PER_TOKEN
from ..llm.pseudonyms import url_is_local
from ..stats import P50, P95, summarise
from . import protocol
from .session import DirectSession, ProviderUnavailable, SessionProvider, load_session_provider

log = logging.getLogger(__name__)

# "closed": the robot closed the session (conversation over, asleep); the next turn reopens it
STATES = ("off", "blocked", "connecting", "connected", "closed", "fallback_local")
# Reasons a route is blocked; the first ones hold until a restart or a config change.
BLOCKS = ("off", "no_route", "no_opt_in", "no_extra", "provider", "pseudonyms", "budget",
          "unavailable")  # fmt: skip
OWN_CLOSES = (
    "conversation_end",
    "asleep",
    "shutdown",
    "reconfigured",
    "budget",
    "expired",
    "input_reset",
)  # our closes, not failures
LATENCY_LEVELS = {"p50": P50, "p95": P95}
MS_PER_S = 1000.0
LATENCY_WINDOW = 50  # turns whose latencies the report summarises
ITEM_PREFIX = "re_"  # the client's own item ids (user text, tool results)
EVENT_PREFIX = "rq_"  # the event ids of the robot's response.create events
PCM_SAMPLE_BYTES = 2  # PCM16 mono
APPEND_SHARE_OF_CAP = (
    2  # an append carries at most 1/2 of max_message_bytes of PCM (base64 grows it)
)
UPLOAD_FILE_MODE = 0o600  # the day's upload seconds (data folder)
# events that carry no data (send_control): everything else goes through the send gate
CONTROL_EVENTS = (protocol.RESPONSE_CANCEL, protocol.ITEM_DELETE, protocol.ITEM_TRUNCATE)


class _Token:
    """Identifies one client's callbacks: those of a client that is not the current (or the
    connecting) one are dropped, so a late close of an old session never touches a new one.
    ``detached``: the route let go of that client itself, so its close is expected."""

    detached: bool = False


@dataclass
class Request:
    """A ``response.create`` in flight. ``run`` receives its events (``deliver(event)``,
    ``closed(reason)``) until it abandons it (None); ``estimate`` is charged if no usage comes."""

    event_id: str
    run: Any | None
    sent_at: float
    estimate: int = 0  # tokens charged when no usage comes (the run keeps it current)
    response_id: str | None = None
    token: _Token | None = None  # the client it was sent on
    last_event: float = 0.0  # when its last event came (expiry ages from it)
    charged: bool = False  # charged once already (at its done, refusal, expiry or close)


def select_provider(cfg: RoboEmotConfig, robot_id: str, manifest: Any) -> SessionProvider | None:
    """The session provider ``voice.realtime_route.mode`` asks for: direct mode is
    :class:`DirectSession` (even when the robot registers a provider); platform mode needs the
    robot's registered provider, and None (the route is blocked, ``provider``) when there is
    none or it cannot be built. Never direct mode in its place (SPEC 8)."""
    if cfg.voice.realtime_route.mode == "direct":
        return DirectSession()
    try:
        factory = load_session_provider(robot_id)
        if factory is None:
            log.warning("realtime route: no session provider is registered for this robot")
            return None
        provider = factory(manifest)
    except ProviderUnavailable:
        return None
    except Exception as exc:
        log.warning("realtime route: session provider: %s", type(exc).__name__)
        return None
    if not isinstance(provider, SessionProvider):
        log.warning("realtime route: the session provider breaks SessionProvider")
        return None
    return provider


def session_update(cfg: RoboEmotConfig, instructions: str, tools: Any = None) -> dict[str, Any]:
    """``session.update``: the instructions (parts 1 to 4), PCM in at the robot's rate (null at
    the native 16 kHz, which Pollen's server requires) and out at the server's own, the voice,
    server VAD that never answers or interrupts on its own and ends a turn after
    ``server_silence_ms``, with ``input: audio`` the input transcription's language
    (``transcription_language``, else ``conversation.language``) and model (left out when
    empty), and the tools (left as they are when None). The route and the realtime check
    (``realtime.check``) send the same one."""
    rt, vad = cfg.voice.realtime_route, cfg.voice.vad
    audio = rt.input == "audio"
    return protocol.session_update(
        instructions=instructions,
        input_rate_hz=None  # the server's native rate (protocol.NATIVE_INPUT_RATE_HZ)
        if cfg.voice.sample_rate_hz == protocol.NATIVE_INPUT_RATE_HZ
        else cfg.voice.sample_rate_hz,
        output_rate_hz=None,
        voice=rt.voice or None,
        vad_threshold=vad.threshold,
        silence_ms=round(rt.server_silence_ms),
        create_response=False,
        interrupt_response=False,
        tools=tools,
        transcription_language=(rt.transcription_language or cfg.conversation.language)
        if audio
        else None,
        transcription_model=(rt.transcription_model or None) if audio else None,
    )


def realtime_extra() -> bool:
    """Whether the ``realtime`` extra (``websockets``) is installed."""
    return importlib.util.find_spec("websockets") is not None


class RealtimeRoute:
    """See the module docstring. ``router()`` returns the current router (the hub rebuilds it,
    so it is read at each use); ``network`` is the runtime's ``NetworkMonitor``;
    ``client_factory`` builds a :class:`~roboemot.realtime.client.RealtimeClient` (tests)."""

    def __init__(
        self,
        stack: Any,
        router: Callable[[], Any],
        network: Any,
        provider: SessionProvider | None,
        *,
        client_factory: Callable[..., Any] | None = None,
        extra: Callable[[], bool] = realtime_extra,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], dt.date] = dt.date.today,
        upload_path: Path | None = None,
    ) -> None:
        self.stack, self._router, self.network, self.provider = stack, router, network, provider
        self._today = today
        self._upload_path = upload_path  # where the day's upload seconds survive a restart
        self._upload_day: dt.date | None = None
        self._upload_s = 0.0  # microphone audio sent on _upload_day (the closing silence too)
        self._load_upload()
        self._input_holds = 0  # realtime turns in flight: no burst meanwhile (their tool rounds)
        self._audio_blocked: str | None = None  # why the last burst was refused, if it was
        self.input: Any = None  # the audio input (``on_event``) the input events go to
        self._egress = False  # audio may leave now (kept current: the Live tab reads it often)
        self._client_factory = client_factory
        self._extra = extra
        self._extra_ok = False  # the extra is installed (read outside _lock, used under it)
        self.clock = clock
        self._lock = threading.Lock()  # the session state machine (see the module docstring)
        self._connect_lock = threading.Lock()  # one connect at a time, held across the network
        self._client: Any = None
        self._connecting: Any = None  # the client of a connect in flight
        self._token: _Token | None = None  # the current client's callback token
        self._connecting_token: _Token | None = None  # the connecting client's
        self._close_reason = "closed"  # the reason of the last close (the report)
        self._rev = 0  # bumped by apply_config: a check made with an older config is redone
        self._closes = 0  # bumped by close/close_soon: a connect begun before never stores
        self.generation = 0  # one per server session: item ids belong to one generation
        self.output_rate_hz: int | None = None
        self._pending: list[Request] = []  # sent, response.created (or a refusal) not yet seen
        self._live: dict[str, Request] = {}  # created, response.done not yet seen
        # ids of the responses whose response.done was counted (and its usage charged)
        self._charged: collections.deque[str] = collections.deque(maxlen=LATENCY_WINDOW)
        self._ids = itertools.count(1)
        self._events = itertools.count(1)
        self.state = "off"
        self.reason: str | None = None
        self._closed = False
        self._lost = False
        self._resets_in_row = 0  # resets since the server last answered (a response, a transcript)
        self._reopen_at = 0.0  # no reconnect before (clock) after a repeated reset
        self._reopens = 0  # reconnects a reset started that have not returned yet
        self.base_instructions: Callable[[], str] = lambda: ""
        self._counts: collections.Counter[str] = collections.Counter()
        self._statuses: collections.Counter[str] = collections.Counter()
        self._failures: collections.Counter[str] = collections.Counter()
        self._first_audio_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
        self._transcript_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
        self._stages: dict[str, collections.deque[float]] = {
            stage: collections.deque(maxlen=LATENCY_WINDOW) for stage in protocol.LATENCY_STAGES
        }
        self._client_snapshot: dict[str, Any] = {}
        self.apply_config(stack.effective())

    def __repr__(self) -> str:
        return f"RealtimeRoute(state={self.state!r})"

    # ----------------------------------------------------------------- config

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: whether realtime mode counts for the network aggregate follows the
        blocking state (a block makes the chat routes the only ones that answer). A session
        that the new config would not open is closed now, and the local pipeline answers: the
        route changed, the route is blocked, or ``privacy.pseudonyms_to_cloud`` came on while
        the session is on a cloud host (SPEC 12). A connect in flight sees the new revision and
        checks again before it stores its session."""
        route = self.route(cfg)
        self._extra_ok = self._extra()  # find_spec takes the import lock: never under _lock
        endpoint = route_endpoint(route) if route is not None else None
        reason = self.blocked(cfg, self._extra_ok)
        self._note_egress(cfg, reason)
        self.network.set_realtime(reason is None, endpoint)
        with self._lock:
            self._rev += 1
            stale = self._client is not None and not self._fits(cfg, self._client)
            client, orphans = self._detach() if stale else (None, [])
        self._settle(orphans, "reconfigured")
        if client is not None:
            self._close_client(client, "reconfigured")

    def _fits(self, cfg: RoboEmotConfig, client: Any) -> bool:
        """Whether ``cfg`` would open ``client``'s session (``_lock`` held): the route is not
        blocked, it is the same route, and with ``privacy.pseudonyms_to_cloud`` the connected
        host is local."""
        cloud = client.snapshot().get("local") is not True
        return (
            self.blocked(cfg, self._extra_ok) is None
            and client.route == self.route(cfg)
            and not (cfg.privacy.pseudonyms_to_cloud and cloud)
        )

    def _with_config(self, fn: Callable[[RoboEmotConfig], Any]) -> Any:
        """``fn(cfg)`` under ``_lock`` with the newest configuration: read outside the lock (the
        stack notifies under its own), redone when ``apply_config`` ran meanwhile."""
        while True:
            with self._lock:
                rev = self._rev
            cfg = self.stack.effective()
            with self._lock:
                if self._rev == rev:
                    return fn(cfg)

    def settings(self, cfg: RoboEmotConfig | None = None) -> Any:
        return (cfg or self.stack.effective()).voice.realtime_route

    def route(self, cfg: RoboEmotConfig | None = None) -> LLMRoute | None:
        name = self.settings(cfg).route
        return next((r for r in (cfg or self.stack.effective()).llm_routes if r.name == name), None)

    def blocked(self, cfg: RoboEmotConfig, extra: bool | None = None) -> str | None:
        """Why the route cannot answer now (a ``BLOCKS`` code), or None when it may; ``extra``
        is whether the realtime extra is installed (looked up when None)."""
        rt = cfg.voice.realtime_route
        if not rt.enabled:
            return "off"
        route = self.route(cfg)
        if route is None:
            return "no_route"
        if rt.input == "audio" and not cfg.privacy.realtime_audio:
            return "no_opt_in"  # audio input needs its own admin opt-in: fail closed
        if not (self._extra() if extra is None else extra):
            return "no_extra"
        if self.provider is None:
            return "provider"
        if cfg.privacy.pseudonyms_to_cloud and not url_is_local(route.base_url, route.locality):
            return "pseudonyms"  # the server would see real names (owner decision: block)
        router = self._router()
        if router is None or router.realtime_route(route.name) is None:
            if router is not None and router.used(route.name) >= route.daily_token_budget:
                return "budget"
            return "unavailable"
        return None

    @property
    def key_hosts(self) -> list[str]:
        """The provider's declared key hosts (the web UI's key confirmation)."""
        return list(self.provider.key_hosts) if self.provider is not None else []

    # ----------------------------------------------------------------- lifecycle

    @property
    def connected(self) -> bool:
        """A session is open (whether it may be used is ``up``)."""
        with self._lock:
            client = self._client
        return client is not None and client.state == "connected"

    def up(self) -> bool:
        """Connected, and the newest configuration would open this session (not blocked, the
        same route, the pseudonym locality): the session may be used now."""

        def check(cfg: RoboEmotConfig) -> bool:
            client = self._client
            return client is not None and client.state == "connected" and self._fits(cfg, client)

        return bool(self._with_config(check))

    def ready(self, cfg: RoboEmotConfig | None = None, *, closes: int | None = None) -> bool:
        """Whether a turn can be answered through the route, connecting if needed (blocking up
        to ``connect_timeout_s``: never call it on an event loop). A blocked route, one closed
        since this call began (or since ``closes``, the close count a reset saw), or one whose
        last connect failed and whose retry is not due, is not ready. A session the newest
        configuration would not open is closed, not used."""
        with self._lock:
            if closes is None:
                closes = self._closes  # a close from now on means no session for this call
        cfg = cfg or self.stack.effective()
        self._extra_ok = self._extra()
        reason = self.blocked(cfg, self._extra_ok)
        committed = self.stack.effective()  # a turn's cfg may be older than the last reload
        self._note_egress(committed, self.blocked(committed, self._extra_ok))
        route = self.route(cfg)
        self.network.set_realtime(
            reason is None, route_endpoint(route) if route is not None else None
        )
        if reason is not None:
            self._set_state("off" if reason == "off" else "blocked", reason)
            return False

        def check(now: RoboEmotConfig) -> tuple[bool, Any]:
            client = self._client
            if client is None:
                return False, None
            if client.state == "connected" and self._fits(now, client):
                return True, None
            return False, self._detach()  # lost, or a reload this session missed

        ok, stale = self._with_config(check)
        if stale is not None:
            client, orphans = stale
            self._settle(orphans, "reconfigured")
            if client is not None:
                self._close_client(client, "reconfigured")
        if ok:
            return True
        if self._closed:
            return False
        with self._lock:
            backoff = self.clock() < self._reopen_at
        if backoff:
            return False  # the server keeps not answering: the reopen waits (expire)
        retry = self.state == "fallback_local" or self.reason == "pseudonyms"
        if retry and not self.network.realtime_due():
            return False  # the last connect failed: retried with the network backoff
        return self._connect(cfg, closes)

    def start(self) -> None:
        """Connect in the background (start, wake): nothing waits for it."""
        if self.blocked(self.stack.effective()) is None and not self._closed:
            threading.Thread(target=self.ready, name="realtime-connect", daemon=True).start()
        else:
            self.ready()  # records the state

    def close(self, reason: str = "shutdown") -> None:
        """Close the server session (blocking, bounded by the client's own timeouts). A connect
        in flight, or a ``ready`` waiting for one, never stores its session after this."""
        with self._lock:
            self._closes += 1
            self._close_reason = reason
            if reason == "shutdown":
                self._closed = True
            client, orphans = self._detach()
        self._settle(orphans, reason)
        if client is not None:
            client.close()
            self._closed_state(reason)

    def close_soon(self, reason: str) -> None:
        """``close`` without waiting: the session is detached now (no later turn uses it: a new
        person never inherits the server's history), its socket closes on a thread of its own
        (the caller is an event loop)."""
        with self._lock:
            self._closes += 1
            self._close_reason = reason
            client, orphans = self._detach()
        self._settle(orphans, reason)
        if client is not None:
            self._close_client(client, reason)

    def _detach(self) -> tuple[Any, list[Request]]:
        """Take the client out of the state (``_lock`` held) with its requests in flight; the
        caller settles them (``_settle``, outside the lock) and closes the client. Its
        callbacks no longer count (its token is no longer current), its close is expected."""
        client, token = self._client, self._token
        self._client = self._token = None
        if token is None:
            return client, []
        token.detached = True
        return client, self._take(token)

    def _take(self, token: _Token) -> list[Request]:
        """Remove ``token``'s requests from the requests in flight (``_lock`` held)."""
        reqs = [r for r in (*self._pending, *self._live.values()) if r.token is token]
        if reqs:
            self._pending = [r for r in self._pending if r.token is not token]
            self._live = {k: r for k, r in self._live.items() if r.token is not token}
        return reqs

    def _settle(self, reqs: list[Request], reason: str) -> None:
        """Requests whose session is gone (outside ``_lock``): each is charged once (its
        estimate) and its run ends now with ``closed: reason``, not at its stall timeout (the
        client's own close comes later, or never)."""
        self._give_up(reqs)
        for r in reqs:
            if r.run is not None:
                r.run.closed(reason)

    def _close_client(self, client: Any, reason: str) -> None:
        self._closed_state(reason)
        threading.Thread(target=client.close, name="realtime-close", daemon=True).start()

    def _closed_state(self, reason: str) -> None:
        with self._lock:
            if self.state == "connected":
                self.state, self.reason = "closed", reason

    def _connect(self, cfg: RoboEmotConfig, closes: int) -> bool:
        """Open a session (``_connect_lock`` held across the network, ``_lock`` never). The
        final check and the store are one critical section: the newest config must fit, and no
        close may have come since ``closes`` was read."""
        with self._connect_lock:
            route, router = self.route(cfg), self._router()
            if route is None or router is None or self.provider is None:
                return False

            def precheck(now: RoboEmotConfig) -> tuple[str, Any]:
                if self._closes != closes or self._closed:
                    return "closed", None  # closed (asleep, shutdown) while this call waited
                cur = self._client
                if cur is not None and cur.state == "connected" and self._fits(now, cur):
                    return "ok", None  # another call connected meanwhile
                return "connect", self._detach()

            verdict, stale = self._with_config(precheck)
            if stale is not None:
                old, orphans = stale
                self._settle(orphans, "reconfigured")
                if old is not None:
                    old.close()
            if verdict != "connect":
                return verdict == "ok"
            token = _Token()  # the callbacks of this client, and nothing else's
            client = self._make_client(cfg, route, router, token)
            with self._lock:
                if self._closes != closes or self._closed:
                    return False
                self._connecting, self._connecting_token = client, token
                self.state, self.reason = "connecting", None
            update = self.session_update(cfg, self.base_instructions())
            try:
                ok = client.connect(update)
            finally:
                with self._lock:
                    self._connecting = self._connecting_token = None
            snap = client.snapshot()
            if not ok:  # the client's close hook already reported the failure
                with self._lock:
                    self._client_snapshot = snap
                    token.detached = True
                    orphans = self._take(token)  # an unrequested response during the connect
                self._settle(orphans, client.reason or "closed")
                self._count("connect_failures")
                client.close()
                if client.reason == "pseudonyms":  # the allocated host is a cloud one
                    self._set_state("blocked", "pseudonyms")
                elif client.reason == "reconfigured":  # the gate before session.update
                    with self._lock:
                        why = self._close_reason if self._closes != closes else "reconfigured"
                    self._set_state("closed", why)
                else:
                    self._set_state("fallback_local", client.reason)
                return False

            def store(now: RoboEmotConfig) -> bool:
                self._client_snapshot = snap
                if self._closes != closes or self._closed:
                    self.state, self.reason = "closed", self._close_reason
                    return False
                if not self._fits(now, client):
                    self.state, self.reason = "closed", "reconfigured"
                    return False
                self._client, self._token = client, token
                self.generation += 1
                self._counts["opened"] += 1
                if self._lost:  # the last session was lost, not closed by the robot
                    self._counts["reconnects"] += 1
                self._lost = False
                self.state, self.reason = "connected", None
                return True

            if not self._with_config(store):
                with self._lock:
                    token.detached = True
                    orphans = self._take(token)
                self._settle(orphans, "closed")
                client.close()
                return False
            self.network.report_realtime(True)
            return True

    def _may_send_update(self, token: _Token, closes: int, client: Any) -> bool:
        """The client's gate right before it sends ``session.update`` (after the handshake):
        the committed config still fits and nothing closed or reloaded the route meanwhile."""

        def check(cfg: RoboEmotConfig) -> bool:
            if self._closes != closes or self._closed or self._connecting_token is not token:
                return False
            return self._fits(cfg, client)

        return bool(self._with_config(check))

    def _make_client(self, cfg: RoboEmotConfig, route: LLMRoute, router: Any, token: _Token) -> Any:
        if self._client_factory is not None:
            factory = self._client_factory
        else:
            from .client import RealtimeClient  # needs the realtime extra

            factory = RealtimeClient
        with self._lock:
            closes = self._closes
        box: list[Any] = []
        client = factory(
            route,
            cfg.voice.realtime_route,
            self.provider,
            auth_headers=lambda: self._router().auth_headers(route),
            on_event=lambda event: self._on_event(event, token),
            on_close=lambda reason: self._on_close(reason, token),
            local_only=lambda: self.stack.effective().privacy.pseudonyms_to_cloud,
            # read at each allocation (a reconnect after the opt-in went off sends no identifier)
            platform_id=lambda: self.stack.effective().privacy.realtime_platform_id,
            may_send=lambda: bool(box) and self._may_send_update(token, closes, box[0]),
        )
        box.append(client)
        return client

    def session_update(self, cfg: RoboEmotConfig, instructions: str, tools: Any = None) -> Any:
        """``session.update`` for this route (:func:`session_update`)."""
        return session_update(cfg, instructions, tools)

    def _set_state(self, state: str, reason: str | None) -> None:
        with self._lock:
            self.state, self.reason = state, reason

    def timeouts(self) -> tuple[float, float]:
        """The route's ``first_token_timeout_s`` and ``stall_timeout_s`` (a run's waits)."""
        route = self.route()
        if route is None:
            return 0.0, 0.0
        return route.first_token_timeout_s, route.stall_timeout_s

    # ----------------------------------------------------------------- responses

    def request(
        self,
        run: Any,
        instructions: str,
        tool_choice: str | None = None,
        *,
        input_chars: int = 0,
        generation: int | None = None,
    ) -> Request | None:
        """Send ``response.create`` for ``run`` (its ``deliver``/``closed`` get the response's
        events) through the send gate; None when there is no session or the gate refuses it
        (nothing is sent). ``input_chars`` (the utterance) counts in its first estimate.
        ``generation``: the session that holds the turn's words; another one is refused."""
        now = self.clock()
        event_id = f"{EVENT_PREFIX}{next(self._events)}"
        estimate = (len(instructions) + input_chars) // CHARS_PER_TOKEN
        req = Request(event_id, run, now, estimate, last_event=now)
        event = protocol.response_create(instructions, tool_choice, event_id=event_id)  # type: ignore[arg-type]

        def track() -> None:  # under _lock, with the send
            req.token = self._token
            self._pending.append(req)

        return req if self._gated(event, track, generation) else None

    def busy(self) -> bool:
        """An abandoned request is still in flight: a new one waits (correlation by order)."""
        with self._lock:
            return any(r.run is None for r in (*self._pending, *self._live.values()))

    def expire(self) -> None:
        """Abandoned requests with no event for the route's ``stall_timeout_s`` are given up:
        charged once (their estimate), counted, their late events dropped. One that never got
        its ``response.created`` on the current session resets it: its late
        ``response.created`` could otherwise take the next request's place (no order
        guessing). The first reset reopens at once; while the server answers nothing, each
        further one waits the ``connectivity.network`` backoff before the next connect."""
        _, stall = self.timeouts()
        now = self.clock()
        with self._lock:
            never = [r for r in self._pending if r.run is None and now - r.last_event >= stall]
            live = [r for r in self._live.values() if r.run is None and now - r.last_event >= stall]
            for r in (*never, *live):
                self._forget(r)
            reset = any(r.token is not None and r.token is self._token for r in never)
            generation = self.generation
        self._give_up([*never, *live])
        if reset:
            self._reset("expired", "resets", generation)

    def _reset(self, reason: str, counter: str, generation: int) -> bool:
        """Close ``generation``'s session and open a new one in the background, when it is
        still the current one (False otherwise: nothing done). The check, the count and the
        detach are one critical section, and the reconnect is refused if any close came after
        it (``ready(closes=...)``): a session the robot closed meanwhile (asleep, the
        conversation over) is never reopened. The first reset in a row reopens at once; while
        the server answers nothing (no response, no transcript), each further one waits the
        ``connectivity.network`` backoff before the next connect (``ready`` then reopens)."""
        policy = self.stack.effective().connectivity.network
        now = self.clock()
        with self._lock:
            if generation != self.generation or self._client is None:
                return False
            self._counts[counter] += 1
            self._resets_in_row += 1
            n = self._resets_in_row
            delay = 0.0
            if n > 1:
                grown = policy.probe_initial_s * policy.probe_backoff ** (n - 2)
                delay = min(policy.probe_max_s, grown)
            self._reopen_at = now + delay
            self._closes += 1
            self._close_reason = reason
            client, orphans = self._detach()
            closes = self._closes
            if delay == 0.0:
                self._reopens += 1
        self._settle(orphans, reason)
        if client is not None:
            self._close_client(client, reason)
        if delay == 0.0:
            threading.Thread(
                target=self._reopen, args=(closes,), name="realtime-connect", daemon=True
            ).start()
        return True

    def _reopen(self, closes: int) -> None:
        """A reset's reconnect (its own thread): refused if a close came after the reset."""
        try:
            self.ready(closes=closes)
        finally:
            with self._lock:
                self._reopens -= 1

    def opening(self) -> bool:
        """A reset's reconnect, or any connect, is in flight: a session may come soon."""
        with self._lock:
            return self._reopens > 0 or self._connecting is not None

    def abandon(self, req: Request, estimate: int) -> None:
        """``req``'s run stops following it (barge-in, a failure, its own timeout): the
        response is cancelled when it is still in flight, and charged once when its
        ``response.done`` comes, when it expires, or when the session closes."""
        with self._lock:
            req.run, req.estimate = None, max(req.estimate, estimate)
            in_flight = req in self._pending or (
                req.response_id is not None and self._live.get(req.response_id) is req
            )
            # its own client only: a gone one took the response with it
            client = self._client if in_flight and req.token is self._token else None
        if client is not None:
            client.send(protocol.response_cancel())

    def finish(self, req: Request) -> None:
        """``req``'s run is over (its ``response.done`` came, or the session is gone)."""
        with self._lock:
            self._forget(req)

    def _forget(self, req: Request) -> None:
        """(``_lock`` held.)"""
        if req in self._pending:
            self._pending.remove(req)
        if req.response_id is not None and self._live.get(req.response_id) is req:
            del self._live[req.response_id]

    def _give_up(self, reqs: list[Request]) -> None:
        """Requests that will get no ``response.done``: their estimate, once each."""
        for r in reqs:
            if self._first_charge(r, r.response_id):
                self.charge(r.estimate)
                self.note_response(None)

    def _first_charge(self, req: Request | None, response_id: str | None) -> bool:
        """Whether this response is charged now (False when it already was): keyed by its
        request, and by its response id (a late ``response.done`` of a request given up)."""
        with self._lock:
            if req is not None and req.charged:
                return False
            if response_id is not None and response_id in self._charged:
                return False
            if req is not None:
                req.charged = True
            if response_id is not None:
                self._charged.append(response_id)
            return True

    def _current(self, token: _Token | None) -> bool:
        """Whether a callback comes from the current or the connecting client (``_lock`` held);
        None (a direct call) counts as current."""
        return token is None or token is self._token or token is self._connecting_token

    def _on_event(self, event: protocol.ServerEvent, token: _Token | None = None) -> None:
        """Route one server event (client thread) to the run that owns it; an event of a
        client that is no longer current is dropped and counted. Each handler checks the
        token in the same critical section that acts on it (a detach may come in between)."""
        if isinstance(event, protocol.SessionEvent) and event.output_rate_hz:
            with self._lock:
                current = self._current(token)
                if current:
                    self.output_rate_hz = event.output_rate_hz
            if not current:
                self._count("stale_callbacks")
            return
        if isinstance(event, protocol.ResponseCreated):
            self._created(event, token)
        elif isinstance(event, protocol.ServerError):
            self._error(event, token)
        elif isinstance(event, OF_A_RESPONSE):
            self._response_event(event, token)
        elif isinstance(event, tuple(CONFIRMATIONS)):
            with self._lock:
                current = self._current(token)
                if current:  # the server confirmed a delete or a truncate (the HIL reads them)
                    self._counts[CONFIRMATIONS[type(event)]] += 1
            if not current:
                self._count("stale_callbacks")
        elif isinstance(event, OF_THE_INPUT):
            with self._lock:
                current = self._current(token)
                sink = self.input
                # the session the event is of, read with the check (a reset may follow at once)
                connecting = token is not None and token is self._connecting_token
                generation = self.generation + 1 if connecting else self.generation
                if current and isinstance(event, protocol.InputTranscript):
                    self._resets_in_row = 0  # the server answers: the reset streak ends
            if not current:
                self._count("stale_callbacks")
            elif sink is not None:
                sink.on_event(event, generation)
        else:
            with self._lock:
                current = self._current(token)
            if not current:
                self._count("stale_callbacks")

    def _created(self, event: protocol.ResponseCreated, token: _Token | None) -> None:
        rid = event.response_id or f"anon_{next(self._ids)}"
        now = self.clock()
        with self._lock:
            if not self._current(token):
                self._counts["stale_callbacks"] += 1
                return
            tok = token or self._token
            req = next((r for r in self._pending if r.token is tok), None)
            if req is not None:
                self._pending.remove(req)
                unrequested = False
            else:  # the server answered on its own
                req = Request("", None, now, token=tok)
                unrequested = True
            req.response_id, req.last_event = rid, now
            self._live[rid] = req
            run = req.run
            # a cancel goes to the client that created it
            client = self._client if tok is self._token else self._connecting
        if unrequested:
            self._count("unrequested")
        if run is None:  # unrequested, or its run was cut before it began: cancelled
            if client is not None:
                client.send(protocol.response_cancel())
        else:
            run.deliver(event)

    def _error(self, event: protocol.ServerError, token: _Token | None) -> None:
        """An ``error`` refuses the request whose ``event_id`` it echoes; others are only
        counted (by the client)."""
        with self._lock:
            if not self._current(token):
                self._counts["stale_callbacks"] += 1
                return
            tok = token or self._token
            req = next(
                (r for r in self._pending if r.event_id == event.event_id and r.token is tok),
                None,
            )
            if req is not None:
                self._pending.remove(req)
            run = req.run if req is not None else None
        if req is None:
            self._count("errors_unrelated")
            return
        self._count("refused")
        self._give_up([req])
        if run is not None:
            run.deliver(event)

    def _response_event(self, event: Any, token: _Token | None) -> None:
        done = isinstance(event, protocol.ResponseDone)
        with self._lock:
            if not self._current(token):
                self._counts["stale_callbacks"] += 1
                return
            tok = token or self._token
            mine = {k: r for k, r in self._live.items() if r.token is tok}
            rid = event.response_id
            if rid is None and len(mine) == 1:
                rid = next(iter(mine))
            req = mine.get(rid) if rid is not None else None
            if req is not None:
                req.last_event = self.clock()
                if done:
                    del self._live[rid]
                    self._resets_in_row = 0  # the server answers: the reset streak ends
            run = req.run if req is not None else None
        if done:
            self.network.report_realtime(True)  # real traffic: the session works
            self._account(event, req)
        if req is None:
            self._count("foreign_events")
            return
        if run is not None:
            run.deliver(event)

    def _account(self, done: protocol.ResponseDone, req: Request | None) -> None:
        """Every ``response.done`` is counted and charged once (keyed by its response id, or
        its request without one): its usage, or its request's estimate without one. The route
        is the one place that charges a response: the turn's, an abandoned one and an
        unrequested one alike."""
        if not self._first_charge(req, done.response_id):
            return
        self.note_response(done)
        if done.total_tokens is not None:
            self.charge(done.total_tokens)
        elif req is not None:
            self.charge(req.estimate)

    def _on_close(self, reason: str, token: _Token | None = None) -> None:
        with self._lock:
            current = self._current(token)
            tok = token or self._token
            expected = tok is not None and tok.detached
            # the current client's requests; a detached client's were settled at the detach
            # (none should be left: swept anyway)
            reqs = self._take(tok) if tok is not None else []
        if not current:  # an old client's late close: the current session is not touched
            if not expected:
                self._count("stale_callbacks")
            self._settle(reqs, reason)
            return
        self._give_up(reqs)
        if reason not in OWN_CLOSES and reason != "closed":
            self.network.report_realtime(False)
            with self._lock:
                self._failures[reason] += 1
                self._lost = self._lost or self.state == "connected"
            self._set_state("fallback_local", reason)
        for r in reqs:
            if r.run is not None:
                r.run.closed(reason)

    # ----------------------------------------------------------------- sending

    def send(self, event: dict[str, Any]) -> bool:
        """Send a data-carrying event (text items, tool results, ``session.update``) through the
        send gate: the committed configuration must still fit the session, checked atomically
        with the send (SPEC 12). Refused: nothing is sent and the session is closed."""
        return self._gated(event)

    def _gated(
        self,
        event: dict[str, Any],
        track: Callable[[], None] | None = None,
        generation: int | None = None,
    ) -> bool:
        def gate(cfg: RoboEmotConfig) -> tuple[bool, Any]:
            client = self._client
            if client is None or client.state != "connected":
                return False, None
            if generation is not None and generation != self.generation:
                return False, None  # the session that holds the turn's words is gone
            if not self._fits(cfg, client):
                self._counts["sends_refused"] += 1
                return False, self._detach()
            if track is not None:
                track()
            client.send(event)  # only queued on the client's loop: never blocks
            return True, None

        ok, stale = self._with_config(gate)
        if stale is not None:
            client, orphans = stale
            self._settle(orphans, "reconfigured")
            if client is not None:
                self._close_client(client, "reconfigured")
        return bool(ok)

    def hold_input(self) -> None:
        """A realtime turn begins: no audio burst until it ends (``release_input``), so another
        utterance's words never land between its ``response.create`` calls (a tool round)."""
        with self._lock:
            self._input_holds += 1

    def release_input(self) -> None:
        with self._lock:
            self._input_holds = max(0, self._input_holds - 1)

    def input_held(self) -> bool:
        with self._lock:
            return self._input_holds > 0

    def send_audio(
        self,
        pcm: bytes,
        *,
        gate: str,
        instructions: str,
        hold: Callable[[], str | None] = lambda: None,
        track: Callable[[int], None] | None = None,
    ) -> str | None:
        """Send one utterance's audio (PCM16 at ``voice.sample_rate_hz``, the closing silence
        included) as one burst: ``session.update`` with fresh ``instructions`` (a server that
        answers on its own then answers with them), the appends (each under half the message
        cap) and the commit. One gate decides the whole burst atomically with the send, on the
        committed configuration: the session fits (as ``send``), ``input: audio`` with the
        ``privacy.realtime_audio`` opt-in, ``gate`` is the configured ``audio_gate``, and the
        burst fits today's upload budget, which it then uses. None when it was queued, else the
        reason it was refused (nothing was sent): ``no_session``, ``input``, ``no_opt_in``,
        ``gate``, ``upload_budget``, ``turn`` (a realtime turn is in flight), what ``hold()``
        returns (``asleep``, ``speaking``, ``cancelled``: evaluated with the send, it may take
        only leaf locks) or ``refused`` (the session no longer fits: closed). ``track(generation)``
        runs with the send, under this lock: the session the burst goes to (the audio input
        reads where that session's input buffer ends in the same step, under its own lock)."""

        def burst(cfg: RoboEmotConfig) -> tuple[str | None, Any]:
            rt = cfg.voice.realtime_route
            client = self._client
            if client is None or client.state != "connected":
                return "no_session", None
            if not self._fits(cfg, client):
                self._counts["sends_refused"] += 1
                return "refused", self._detach()
            seconds = len(pcm) / PCM_SAMPLE_BYTES / cfg.voice.sample_rate_hz
            if rt.input != "audio":
                why = "input"
            elif (held := hold()) is not None:
                why = held
            elif self._input_holds > 0:
                why = "turn"
            elif not cfg.privacy.realtime_audio:
                why = "no_opt_in"
            elif rt.audio_gate != gate:
                why = "gate"
            elif self._uploaded() + seconds > rt.daily_upload_budget_s:
                why = "upload_budget"
            else:
                why = None
            self._audio_blocked = why
            if why is not None:
                return why, None
            self._upload_s += seconds
            if track is not None:
                track(self.generation)
            client.send(self.session_update(cfg, instructions))  # only queued: never blocks
            step = (
                rt.max_message_bytes // APPEND_SHARE_OF_CAP // PCM_SAMPLE_BYTES * PCM_SAMPLE_BYTES
            )
            for start in range(0, len(pcm), step):
                client.send(protocol.audio_append(pcm[start : start + step]))
            client.send(protocol.audio_commit())
            return None, (self._upload_day, self._upload_s)

        why, stale = self._with_config(burst)
        if why is None:
            self._save_upload(*stale)
            stale = None
        if stale is not None:
            client, orphans = stale
            self._settle(orphans, "reconfigured")
            if client is not None:
                self._close_client(client, "reconfigured")
        if why != "turn":  # a hold is not a refusal: the input counts it once (audio_held)
            self._count("audio_refused" if why is not None else "audio_sent")
        return why

    def _uploaded(self) -> float:
        """Seconds of audio sent today (``_lock`` held). Only a later date starts from zero: a
        clock that jumps back (no RTC, an NTP fix, a time zone) keeps the day's count."""
        today = self._today()
        if self._upload_day is None or today > self._upload_day:
            self._upload_day, self._upload_s = today, 0.0
        return self._upload_s

    def _load_upload(self) -> None:
        """The day's upload seconds from the data folder, so a restart keeps the cap. A file
        that cannot be read counts today as spent (fail closed); none is a fresh day."""
        path = self._upload_path
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            day = dt.date.fromisoformat(data["day"])
            seconds = data["seconds"]
            number = isinstance(seconds, int | float) and not isinstance(seconds, bool)
            if not number or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("upload seconds")  # a value that would open the cap
            # a stored day after today (a clock that jumped forward, then back) is today: its
            # seconds count until the date really moves forward
            self._upload_day, self._upload_s = min(day, self._today()), float(seconds)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("realtime route: upload count unreadable (%s): today counts as spent",
                        type(exc).__name__)  # fmt: skip
            self._upload_day, self._upload_s = self._today(), float("inf")

    def _save_upload(self, day: dt.date | None, seconds: float) -> None:
        """Write ``(day, seconds)`` atomically, mode 0600 (outside ``_lock``: disk, not the
        network). A failure is logged by type; the count in memory still holds."""
        path = self._upload_path
        if path is None or day is None:
            return
        try:
            fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
            tmp = Path(name)
            try:
                with os.fdopen(fd, "w") as handle:
                    os.fchmod(handle.fileno(), UPLOAD_FILE_MODE)
                    record = {"day": day.isoformat(), "seconds": seconds}
                    handle.write(json.dumps(record, allow_nan=False))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
        except (OSError, ValueError) as exc:  # ValueError: a count that is not finite
            log.warning("realtime route: upload count not saved (%s)", type(exc).__name__)

    def upload_left_s(self, cfg: RoboEmotConfig) -> float:
        """Seconds of audio today's upload budget still allows."""
        with self._lock:
            used = self._uploaded()
        return max(0.0, cfg.voice.realtime_route.daily_upload_budget_s - used)

    def send_control(self, event: dict[str, Any]) -> bool:
        """Send an event that carries no data (``response.cancel``, ``conversation.item.delete``
        and ``.truncate``; any other is refused): ungated, also on a client still connecting."""
        if event.get("type") not in CONTROL_EVENTS:
            return False
        with self._lock:
            client = self._client or self._connecting
        if client is None:
            return False
        client.send(event)
        if event["type"] == protocol.ITEM_TRUNCATE:
            self._count("items_truncated")
        return True

    def new_item_id(self) -> str:
        """An id for an item the robot creates (user text, a tool result)."""
        return f"{ITEM_PREFIX}{self.generation}_{next(self._ids)}"

    def delete_items(self, generation: int, item_ids: list[str]) -> int:
        """Delete items of the current server session (older ones died with theirs)."""
        if generation != self.generation or not self.connected:
            return 0
        sent = sum(self.send_control(protocol.item_delete(item)) for item in item_ids)
        self._count("items_deleted", sent)
        return sent

    # ----------------------------------------------------------------- accounting

    def charge(self, tokens: int) -> None:
        """Count ``tokens`` against the route's daily budget (usage, or the chars/4 estimate)."""
        route, router = self.route(), self._router()
        if route is not None and router is not None and tokens > 0:
            router.charge(route.name, tokens)
            if router.used(route.name) >= route.daily_token_budget:
                self._egress = False  # blocked (budget) until a new day's ready()
                self.close_soon("budget")  # spent: nothing more goes to the route today

    def note_response(self, done: protocol.ResponseDone | None) -> None:
        with self._lock:
            self._statuses[done.status if done is not None else "none"] += 1
            if done is not None:
                for stage, value in done.latency_s.items():
                    self._stages[stage].append(value)

    def note_first_audio(self, ms: float) -> None:
        with self._lock:
            self._first_audio_ms.append(ms)

    def note_transcript(self, ms: float) -> None:
        """End of the user's speech to the server's transcript of it (SPEC 13, the HIL)."""
        with self._lock:
            self._transcript_ms.append(ms)

    def reset_session(self, generation: int) -> bool:
        """Reset ``generation``'s session (``_reset``, counted ``input_resets``): an audio burst
        ended without its result while the server may still report on it, and its late items
        must never be taken for the next burst's (their callbacks are stale once the session is
        detached). False when that session is no longer the current one: nothing done."""
        return self._reset("input_reset", "input_resets", generation)

    def count(self, key: str, n: int = 1) -> None:
        self._count(key, n)

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counts[key] += n

    # ----------------------------------------------------------------- report

    @property
    def audio_egress(self) -> bool:
        """Whether microphone audio may leave the robot now: ``input: audio`` with its opt-in
        and the route not blocked (never in text mode). Kept current by ``apply_config``,
        ``ready`` and a spent budget, so reading it never reads the configuration stack (the
        Live tab polls it)."""
        return self._egress

    def _note_egress(self, cfg: RoboEmotConfig, reason: str | None) -> None:
        rt = cfg.voice.realtime_route
        self._egress = rt.input == "audio" and cfg.privacy.realtime_audio and reason is None

    def snapshot(self) -> dict[str, Any]:
        """The ``roboemot.realtime_route`` debug section (SPEC 12A): states, reason codes,
        counts and numbers only; the endpoint as ``scheme://host:port``."""
        cfg = self.stack.effective()
        rt = cfg.voice.realtime_route
        with self._lock:
            client = self._client
        snap = client.snapshot() if client is not None else None
        with self._lock:
            if snap is not None:
                self._client_snapshot = snap
            snap = dict(self._client_snapshot)
            counts = dict(self._counts)
            audio = rt.input == "audio"
            out = {
                "state": self.state,
                "reason": self.reason,
                "mode": rt.mode,
                "input": rt.input,
                "gate": rt.audio_gate if audio else None,  # text input: no audio gate
                "endpoint": snap.get("endpoint"),
                "audio_egress": self._egress,
                "audio_blocked": self._audio_blocked,  # why the last burst was refused
                "platform_id_sent": bool(snap.get("platform_id_sent", False)),
                "provider_hosts": self.key_hosts,  # report keys must avoid "key"
                "generation": self.generation,  # server sessions opened so far
                "responses": dict(self._statuses),
                "failures": dict(self._failures),
                # an unreadable count is "spent": shown as the budget (the report is JSON)
                "upload_s_today": min(self._uploaded(), rt.daily_upload_budget_s),
                "upload_budget_s": rt.daily_upload_budget_s,
                "latency_ms": {
                    "first_audio": summarise(list(self._first_audio_ms), LATENCY_LEVELS),
                    "transcript": summarise(list(self._transcript_ms), LATENCY_LEVELS),
                    "server": {
                        stage: summarise([v * MS_PER_S for v in values], LATENCY_LEVELS)
                        for stage, values in self._stages.items()
                        if values
                    },
                },
            }
        for key in COUNTERS:
            out[key] = counts.get(key, 0)
        out["unknown_events"] = snap.get("unknown", 0)
        out["oversize"] = snap.get("oversize", 0)
        out["client"] = {k: v for k, v in snap.items() if k != "key_hosts"}
        return out


COUNTERS = (
    "opened",
    "reconnects",
    "connect_failures",
    "utterances_sent",
    "audio_sent",
    "audio_refused",
    "audio_held",
    "audio_reopen_wait",
    "audio_items",
    "audio_items_deleted",
    "audio_items_unnamed",
    "audio_items_undeletable",
    "audio_items_ambiguous",
    "items_deleted",
    "items_delete_confirmed",
    "items_truncated",
    "items_truncate_confirmed",
    "transcripts",
    "transcript_timeouts",
    "unrequested",
    "tool_calls",
    "skipped",  # the model called ignore_utterance before any of its audio played
    "skipped_late",  # ... after some played: cut like a barge-in
    "lines",
    "fallbacks_start",
    "fallbacks_mid_turn",
    "cut_mid_turn",
    "cuts",
    "refused",
    "errors_unrelated",
    "foreign_events",
    "stale_callbacks",
    "sends_refused",
    "resets",
    "input_resets",
)

# the server's confirmations of a delete and a truncate, and the counter each one moves
CONFIRMATIONS: dict[type, str] = {
    protocol.ItemDeleted: "items_delete_confirmed",
    protocol.ItemTruncated: "items_truncate_confirmed",
}

# the server events about the input audio (the audio input collects them)
OF_THE_INPUT = (protocol.SpeechBoundary, protocol.InputTranscript)

# the server events that name the response they belong to
OF_A_RESPONSE = (
    protocol.AudioDelta,
    protocol.AudioDone,
    protocol.ReplyTranscript,
    protocol.FunctionCall,
    protocol.ResponseDone,
)
