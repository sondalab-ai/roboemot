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
held across the network) -> ``_lock`` -> the client's own lock. The client never calls the route
while holding its lock.

Responses are correlated explicitly (SPEC 8, Realtime speech route). Each ``response.create``
carries an ``event_id``; an ``error`` that echoes it refuses that request, any other ``error``
is only counted. ``response.created`` binds the oldest request in flight (the server answers in
order, and a new request waits until no abandoned one is in flight, ``busy``); every later
event goes to the run that owns its ``response_id`` and is dropped (counted) otherwise. A run cut
by a barge-in abandons its request: its response is cancelled and its ``response.done`` charged
then; an abandoned request that never gets its ``response.created`` expires on the next
``response.done`` or after the route's ``stall_timeout_s``. Nothing here logs or reports a key, a
URL's path or query, text, audio or an identifier (SPEC 12).
"""

from __future__ import annotations

import collections
import importlib.util
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig
from ..health.network import route_endpoint
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
)  # our closes, not failures
LATENCY_LEVELS = {"p50": P50, "p95": P95}
MS_PER_S = 1000.0
LATENCY_WINDOW = 50  # turns whose latencies the report summarises
ITEM_PREFIX = "re_"  # the client's own item ids (user text, tool results)
EVENT_PREFIX = "rq_"  # the event ids of the robot's response.create events


@dataclass
class Request:
    """A ``response.create`` in flight. ``run`` receives its events (``deliver(event)``,
    ``closed(reason)``) until it abandons it (None); ``estimate`` is charged if no usage comes."""

    event_id: str
    run: Any | None
    sent_at: float
    estimate: int = 0
    response_id: str | None = None


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
    ) -> None:
        self.stack, self._router, self.network, self.provider = stack, router, network, provider
        self._client_factory = client_factory
        self._extra = extra
        self.clock = clock
        self._lock = threading.Lock()  # the session state machine (see the module docstring)
        self._connect_lock = threading.Lock()  # one connect at a time, held across the network
        self._client: Any = None
        self._connecting: Any = None  # the client of a connect in flight
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
        self.base_instructions: Callable[[], str] = lambda: ""
        self._counts: collections.Counter[str] = collections.Counter()
        self._statuses: collections.Counter[str] = collections.Counter()
        self._failures: collections.Counter[str] = collections.Counter()
        self._first_audio_ms: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
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
        self.network.set_realtime(
            self.blocked(cfg) is None, route_endpoint(route) if route is not None else None
        )
        with self._lock:
            self._rev += 1
            stale = self._client is not None and not self._fits(cfg, self._client)
            client = self._detach() if stale else None
        if client is not None:
            self._close_client(client, "reconfigured")

    def _fits(self, cfg: RoboEmotConfig, client: Any) -> bool:
        """Whether ``cfg`` would open ``client``'s session (``_lock`` held): the route is not
        blocked, it is the same route, and with ``privacy.pseudonyms_to_cloud`` the connected
        host is local."""
        cloud = client.snapshot().get("local") is not True
        return (
            self.blocked(cfg) is None
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

    def blocked(self, cfg: RoboEmotConfig) -> str | None:
        """Why the route cannot answer now (a ``BLOCKS`` code), or None when it may."""
        rt = cfg.voice.realtime_route
        if not rt.enabled:
            return "off"
        route = self.route(cfg)
        if route is None:
            return "no_route"
        if rt.input == "audio":
            return "no_opt_in"  # audio input has its own opt-in (not built yet): fail closed
        if not self._extra():
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

    def ready(self, cfg: RoboEmotConfig | None = None) -> bool:
        """Whether a turn can be answered through the route, connecting if needed (blocking up
        to ``connect_timeout_s``: never call it on an event loop). A blocked route, one closed
        since this call began, or one whose last connect failed and whose retry is not due, is
        not ready. A session the newest configuration would not open is closed, not used."""
        with self._lock:
            closes = self._closes  # a close from now on means no session for this call
        cfg = cfg or self.stack.effective()
        reason = self.blocked(cfg)
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
            self._close_client(stale, "reconfigured")
        if ok:
            return True
        if self._closed:
            return False
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
            if reason == "shutdown":
                self._closed = True
            client = self._detach()
        if client is not None:
            client.close()
            self._closed_state(reason)

    def close_soon(self, reason: str) -> None:
        """``close`` without waiting: the session is detached now (no later turn uses it: a new
        person never inherits the server's history), its socket closes on a thread of its own
        (the caller is an event loop)."""
        with self._lock:
            self._closes += 1
            client = self._detach()
        if client is not None:
            self._close_client(client, reason)

    def _detach(self) -> Any:
        """Take the client out of the state (``_lock`` held); the caller closes it."""
        client, self._client = self._client, None
        return client

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
            with self._lock:
                if self._closes != closes or self._closed:
                    return False  # closed (asleep, shutdown) while this call waited
            client = self._make_client(cfg, route, router)
            with self._lock:
                if self._closes != closes or self._closed:
                    return False  # closed (asleep, shutdown) while this call waited
                if self._client is not None and self._client.state == "connected":
                    return True  # another call connected meanwhile (it was checked then)
                old, self._client = self._client, None
                self._connecting = client
                self.state, self.reason = "connecting", None
            if old is not None:
                old.close()
            update = self.session_update(cfg, self.base_instructions())
            try:
                ok = client.connect(update)
            finally:
                with self._lock:
                    self._connecting = None
            snap = client.snapshot()
            if not ok:  # the client's close hook already reported the failure
                with self._lock:
                    self._client_snapshot = snap
                self._count("connect_failures")
                client.close()
                if client.reason == "pseudonyms":  # the allocated host is a cloud one
                    self._set_state("blocked", "pseudonyms")
                else:
                    self._set_state("fallback_local", client.reason)
                return False

            def store(now: RoboEmotConfig) -> bool:
                self._client_snapshot = snap
                if self._closes != closes or self._closed or not self._fits(now, client):
                    self.state, self.reason = "closed", "reconfigured"
                    return False
                self._client = client
                self.generation += 1
                self._counts["opened"] += 1
                if self._lost:  # the last session was lost, not closed by the robot
                    self._counts["reconnects"] += 1
                self._lost = False
                self.state, self.reason = "connected", None
                return True

            if not self._with_config(store):
                client.close()
                return False
            self.network.report_realtime(True)
            return True

    def _make_client(self, cfg: RoboEmotConfig, route: LLMRoute, router: Any) -> Any:
        if self._client_factory is not None:
            factory = self._client_factory
        else:
            from .client import RealtimeClient  # needs the realtime extra

            factory = RealtimeClient
        return factory(
            route,
            cfg.voice.realtime_route,
            self.provider,
            auth_headers=lambda: self._router().auth_headers(route),
            on_event=self._on_event,
            on_close=self._on_close,
            local_only=lambda: self.stack.effective().privacy.pseudonyms_to_cloud,
        )

    def session_update(self, cfg: RoboEmotConfig, instructions: str, tools: Any = None) -> Any:
        """``session.update`` for a text-input session: the instructions (parts 1 to 4), PCM in
        at the robot's rate and out at the server's own, the voice, server VAD that never answers
        or interrupts on its own, and the tools."""
        rt, vad = cfg.voice.realtime_route, cfg.voice.vad
        return protocol.session_update(
            instructions=instructions,
            input_rate_hz=cfg.voice.sample_rate_hz,
            output_rate_hz=None,
            voice=rt.voice or None,
            vad_threshold=vad.threshold,
            silence_ms=round(vad.end_silence_ms),
            create_response=False,
            interrupt_response=False,
            tools=tools,
        )

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
        self, run: Any, instructions: str, tool_choice: str | None = None
    ) -> Request | None:
        """Send ``response.create`` for ``run`` (its ``deliver``/``closed`` get the response's
        events); None when there is no session (nothing is sent)."""
        event_id = f"{EVENT_PREFIX}{next(self._events)}"
        req = Request(event_id, run, self.clock())
        with self._lock:
            self._pending.append(req)
        event = protocol.response_create(instructions, tool_choice, event_id=event_id)  # type: ignore[arg-type]
        if not self.send(event):
            with self._lock:
                if req in self._pending:
                    self._pending.remove(req)
            return None
        return req

    def busy(self) -> bool:
        """An abandoned request is still in flight: a new one waits (correlation by order)."""
        with self._lock:
            return any(r.run is None for r in (*self._pending, *self._live.values()))

    def expire(self) -> None:
        """Abandoned requests older than the route's ``stall_timeout_s`` are given up: charged
        their estimate, counted, and their late events dropped."""
        _, stall = self.timeouts()
        now = self.clock()
        with self._lock:
            old = [r for r in self._pending if r.run is None and now - r.sent_at >= stall]
            old += [r for r in self._live.values() if r.run is None and now - r.sent_at >= stall]
            for r in old:
                self._forget(r)
        self._give_up(old)

    def abandon(self, req: Request, estimate: int) -> None:
        """``req``'s run stops following it (barge-in, a failure, its own timeout): the
        response is cancelled when it is still in flight, and its ``response.done`` is charged
        when it comes (``estimate`` without usage, or if it never comes)."""
        with self._lock:
            req.run, req.estimate = None, estimate
            in_flight = req in self._pending or req.response_id in self._live
        if in_flight:
            self.send(protocol.response_cancel())

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
        for r in reqs:
            self.charge(r.estimate)
            self.note_response(None)

    def _on_event(self, event: protocol.ServerEvent) -> None:
        """Route one server event (client thread) to the run that owns it."""
        if isinstance(event, protocol.SessionEvent) and event.output_rate_hz:
            self.output_rate_hz = event.output_rate_hz
            return
        if isinstance(event, protocol.ResponseCreated):
            self._created(event)
        elif isinstance(event, protocol.ServerError):
            self._error(event)
        elif isinstance(event, OF_A_RESPONSE):
            self._response_event(event)

    def _created(self, event: protocol.ResponseCreated) -> None:
        rid = event.response_id or f"anon_{next(self._ids)}"
        with self._lock:
            req = self._pending.pop(0) if self._pending else None
            if req is None:  # the server answered on its own
                req = Request("", None, self.clock())
                unrequested = True
            else:
                unrequested = False
            req.response_id = rid
            self._live[rid] = req
            run = req.run
        if unrequested:
            self._count("unrequested")
        if run is None:  # unrequested, or its run was cut before it began: cancelled
            self.send(protocol.response_cancel())
        else:
            run.deliver(event)

    def _error(self, event: protocol.ServerError) -> None:
        """An ``error`` refuses the request whose ``event_id`` it echoes; others are only
        counted (by the client)."""
        with self._lock:
            req = next((r for r in self._pending if r.event_id == event.event_id), None)
            if req is not None:
                self._pending.remove(req)
            run = req.run if req is not None else None
        if req is None:
            self._count("errors_unrelated")
            return
        self._count("refused")
        if run is not None:
            run.deliver(event)
        else:
            self._give_up([req])

    def _response_event(self, event: Any) -> None:
        done = isinstance(event, protocol.ResponseDone)
        with self._lock:
            rid = event.response_id
            if rid is None and len(self._live) == 1:
                rid = next(iter(self._live))
            req = self._live.get(rid) if rid is not None else None
            if done and req is not None:
                del self._live[rid]
            expired: list[Request] = []
            if done:  # an abandoned request never created by now never will be
                expired = [r for r in self._pending if r.run is None]
                for r in expired:
                    self._pending.remove(r)
            run = req.run if req is not None else None
        if done:
            self.network.report_realtime(True)  # real traffic: the session works
            self._account(event)
            self._give_up(expired)
            if req is not None and run is None and event.total_tokens is None:
                self.charge(req.estimate)  # abandoned, no usage: its estimate
        if req is None:
            self._count("foreign_events")
            return
        if run is not None:
            run.deliver(event)

    def _account(self, done: protocol.ResponseDone) -> None:
        """Every ``response.done`` is counted, and its usage charged, once (keyed by response
        id): the turn's, an abandoned one and an unrequested one alike."""
        with self._lock:
            seen = done.response_id is not None and done.response_id in self._charged
            if done.response_id is not None and not seen:
                self._charged.append(done.response_id)
        if seen:
            return
        self.note_response(done)
        if done.total_tokens is not None:
            self.charge(done.total_tokens)

    def _on_close(self, reason: str) -> None:
        with self._lock:
            reqs = [*self._pending, *self._live.values()]
            self._pending, self._live = [], {}
        self._give_up([r for r in reqs if r.run is None])
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
        """Queue a client event; False when there is no session (nothing is sent). While a
        connect completes, its client (a cancel of an unrequested response)."""
        with self._lock:
            client = self._client or self._connecting
        if client is None:
            return False
        client.send(event)
        return True

    def new_item_id(self) -> str:
        """An id for an item the robot creates (user text, a tool result)."""
        return f"{ITEM_PREFIX}{self.generation}_{next(self._ids)}"

    def delete_items(self, generation: int, item_ids: list[str]) -> int:
        """Delete items of the current server session (older ones died with theirs)."""
        if generation != self.generation or not self.connected:
            return 0
        sent = sum(self.send(protocol.item_delete(item)) for item in item_ids)
        self._count("items_deleted", sent)
        return sent

    # ----------------------------------------------------------------- accounting

    def charge(self, tokens: int) -> None:
        """Count ``tokens`` against the route's daily budget (usage, or the chars/4 estimate)."""
        route, router = self.route(), self._router()
        if route is not None and router is not None and tokens > 0:
            router.charge(route.name, tokens)
            if router.used(route.name) >= route.daily_token_budget:
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

    def count(self, key: str, n: int = 1) -> None:
        self._count(key, n)

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._counts[key] += n

    # ----------------------------------------------------------------- report

    @property
    def audio_egress(self) -> bool:
        """Whether microphone audio may leave the robot now: never in text mode."""
        return False

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
            out = {
                "state": self.state,
                "reason": self.reason,
                "mode": rt.mode,
                "input": rt.input,
                "gate": None,  # text input: no audio gate
                "endpoint": snap.get("endpoint"),
                "audio_egress": self.audio_egress,
                "platform_id_sent": False,
                "provider_hosts": self.key_hosts,  # report keys must avoid "key"
                "generation": self.generation,  # server sessions opened so far
                "responses": dict(self._statuses),
                "failures": dict(self._failures),
                "upload_s_today": 0.0,
                "latency_ms": {
                    "first_audio": summarise(list(self._first_audio_ms), LATENCY_LEVELS),
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
    "items_deleted",
    "transcripts",
    "transcript_timeouts",
    "unrequested",
    "tool_calls",
    "lines",
    "fallbacks_start",
    "fallbacks_mid_turn",
    "cut_mid_turn",
    "cuts",
    "refused",
    "errors_unrelated",
    "foreign_events",
)

# the server events that name the response they belong to
OF_A_RESPONSE = (
    protocol.AudioDelta,
    protocol.AudioDone,
    protocol.ReplyTranscript,
    protocol.FunctionCall,
    protocol.ResponseDone,
)
