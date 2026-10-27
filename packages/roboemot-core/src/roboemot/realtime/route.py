"""The realtime speech route at runtime (SPEC 8, Realtime speech route; SPEC 12A, Links and
Diagnostics).

:class:`RealtimeRoute` owns the client's lifecycle: it connects at start, closes at sleep and at
the end of a conversation session, and reopens on wake and when a turn needs it. It decides
whether the route may answer at all (``blocked``: a reason code), reports the session to the
``realtime`` link of the network monitor (so a connect is retried with the
``connectivity.network`` backoff), charges the route's daily token budget, cancels a response
the robot never asked for, and keeps the numbers of the ``roboemot.realtime_route`` debug
section.

One response at a time: the conversation worker serialises turns and fixed lines, and a turn
listens to the server's events through ``listen``. Nothing here logs or reports a key, a URL's
path or query, text, audio or an identifier (SPEC 12).
"""

from __future__ import annotations

import collections
import contextlib
import importlib.util
import itertools
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig
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
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._client: Any = None
        self._connecting: Any = None  # the client of a connect in progress
        self._abort = False  # a close came while a connect was in flight
        self.generation = 0  # one per server session: item ids belong to one generation
        self.output_rate_hz: int | None = None
        self._listener: Callable[[protocol.ServerEvent], None] | None = None
        self._close_listener: Callable[[str], None] | None = None
        self._expected = 0  # response.create sent, response.created not yet seen
        # responses the robot cancelled whose response.done no turn waits for any more:
        # (response id or None, the estimate charged if it never comes)
        self._owed: collections.deque[tuple[str | None, int]] = collections.deque()
        # ids of the responses whose response.done was counted (and its usage charged)
        self._charged: collections.deque[str] = collections.deque(maxlen=LATENCY_WINDOW)
        self.current_response: str | None = None  # the id of the response a turn follows
        self._ids = itertools.count(1)
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
        the session is on a cloud host (SPEC 12)."""
        self.network.set_realtime(self.blocked(cfg) is None)
        client = self._client
        if client is not None and not self._fits(cfg, client):
            self.close_soon("reconfigured")
        # a connect in flight checks the config again before its session is used (_connect)

    def _fits(self, cfg: RoboEmotConfig, client: Any) -> bool:
        """Whether ``cfg`` would open ``client``'s session: the route is not blocked, it is the
        same route, and with ``privacy.pseudonyms_to_cloud`` the connected host is local."""
        cloud = client.snapshot().get("local") is not True
        return (
            self.blocked(cfg) is None
            and client.route == self.route(cfg)
            and not (cfg.privacy.pseudonyms_to_cloud and cloud)
        )

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
        client = self._client
        return client is not None and client.state == "connected"

    def ready(self, cfg: RoboEmotConfig | None = None) -> bool:
        """Whether a turn can be answered through the route, connecting if needed (blocking up
        to ``connect_timeout_s``: never call it on an event loop). A blocked route, or one whose
        last connect failed and whose retry is not due, is not ready."""
        cfg = cfg or self.stack.effective()
        reason = self.blocked(cfg)
        self.network.set_realtime(reason is None)
        if reason is not None:
            self._set_state("off" if reason == "off" else "blocked", reason)
            return False
        client = self._client
        if client is not None and client.state == "connected":
            if self._fits(cfg, client):
                return True
            self.close_soon("reconfigured")  # a reload this session missed
        if self._closed:
            return False
        retry = self.state == "fallback_local" or self.reason == "pseudonyms"
        if retry and not self.network.realtime_due():
            return False  # the last connect failed: retried with the network backoff
        return self._connect(cfg)

    def start(self) -> None:
        """Connect in the background (start, wake): nothing waits for it."""
        if self.blocked(self.stack.effective()) is None and not self._closed:
            threading.Thread(target=self.ready, name="realtime-connect", daemon=True).start()
        else:
            self.ready()  # records the state

    def close(self, reason: str = "shutdown") -> None:
        """Close the server session (blocking, bounded by the client's own timeouts)."""
        with self._lock:
            self._abort = True
            client, self._client = self._client, None
            if reason == "shutdown":
                self._closed = True
        if client is not None:
            client.close()
        if self.state == "connected":
            self._set_state("closed", reason)

    def close_soon(self, reason: str) -> None:
        """Detach the session now (no later turn uses it: a new person never inherits the
        server's history) and close its socket on a thread of its own (the caller is an event
        loop)."""
        with self._lock:
            self._abort = True  # a connect in flight does not keep its session
            client, self._client = self._client, None
        if client is None:
            return
        if self.state == "connected":
            self._set_state("closed", reason)
        threading.Thread(target=client.close, name="realtime-close", daemon=True).start()

    def _connect(self, cfg: RoboEmotConfig) -> bool:
        with self._connect_lock:
            if self.connected:
                return True
            route = self.route(cfg)
            router = self._router()
            if route is None or router is None or self.provider is None:
                return False
            old, self._client = self._client, None
            if old is not None:
                old.close()
            self._set_state("connecting", None)
            self._abort = False
            client = self._connecting = self._make_client(cfg, route, router)
            update = self.session_update(cfg, self.base_instructions())
            try:
                ok = client.connect(update)
            finally:
                self._connecting = None
            snap = client.snapshot()
            with self._lock:
                self._client_snapshot = snap
            if not ok:  # the client's close hook already reported the failure
                self._count("connect_failures")
                client.close()
                if client.reason == "pseudonyms":  # the allocated host is a cloud one
                    self._set_state("blocked", "pseudonyms")
                else:
                    self._set_state("fallback_local", client.reason)
                return False
            now = self.stack.effective()  # a reload, sleep or shutdown during the connect
            if self._abort or self._closed or not self._fits(now, client):
                client.close()
                self._set_state("closed", "reconfigured")
                return False
            with self._lock:
                self._client = client
                self.generation += 1
                self._counts["opened"] += 1
                if self._lost:  # the last session was lost, not closed by the robot
                    self._counts["reconnects"] += 1
                self._lost = False
            self.network.report_realtime(True)
            self._set_state("connected", None)
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

    # ----------------------------------------------------------------- events

    @contextlib.contextmanager
    def listen(
        self, on_event: Callable[[protocol.ServerEvent], None], on_close: Callable[[str], None]
    ) -> Iterator[None]:
        """Hand the server's events (client thread) and a lost session to one turn."""
        with self._lock:
            self._listener, self._close_listener = on_event, on_close
        try:
            yield
        finally:
            with self._lock:
                self._listener = self._close_listener = None

    def _on_event(self, event: protocol.ServerEvent) -> None:
        if isinstance(event, protocol.SessionEvent) and event.output_rate_hz:
            self.output_rate_hz = event.output_rate_hz
        if isinstance(event, protocol.ResponseCreated):
            with self._lock:
                requested = self._expected > 0
                self._expected = max(0, self._expected - 1)
                listener = self._listener
            if not requested:  # the server answered on its own: cancelled, counted
                self._count("unrequested")
                self.send(protocol.response_cancel())
                return
            with self._lock:  # the oldest abandoned request answers first (in order)
                index = next((i for i, e in enumerate(self._owed) if e[0] is None), None)
                if index is not None:
                    self._owed[index] = (event.response_id, self._owed[index][1])
            if listener is None or index is not None:
                # its turn was cut before it began: cancel it again, never hand it to a turn
                self.send(protocol.response_cancel())
                return
        if isinstance(event, protocol.ResponseDone):
            self.network.report_realtime(True)  # real traffic: the session works
            self._account(event)
            if self._settle_owed(event):
                return
        with self._lock:
            listener = self._listener
        if listener is not None:
            listener(event)

    def owe(self, response_id: str | None, estimate: int) -> None:
        """A cancelled response whose ``response.done`` will come after its turn stopped
        listening: its usage (else ``estimate``) is charged and its status counted then."""
        with self._lock:
            self._owed.append((response_id, estimate))

    def _account(self, done: protocol.ResponseDone) -> None:
        """Every ``response.done`` is counted, and its usage charged, once (keyed by response
        id): the turn's, an abandoned one and an unrequested one alike. Without usage the turn
        (or the owed entry) charges its estimate."""
        with self._lock:
            seen = done.response_id is not None and done.response_id in self._charged
            if done.response_id is not None and not seen:
                self._charged.append(done.response_id)
        if seen:
            return
        self.note_response(done)
        if done.total_tokens is not None:
            self.charge(done.total_tokens)

    def _settle_owed(self, done: protocol.ResponseDone) -> bool:
        """Charge and count a ``response.done`` owed by a cut turn; False when it is not one."""
        with self._lock:
            entry = next((e for e in self._owed if e[0] and e[0] == done.response_id), None)
            if entry is None and done.response_id != self.current_response:
                entry = next((e for e in self._owed if e[0] is None), None)
            if entry is None:
                return False
            self._owed.remove(entry)
        if done.total_tokens is None:  # usage was charged by _account when given
            self.charge(entry[1])
        return True

    def _settle_lost(self) -> None:
        """The session is gone: the owed responses are charged their estimates."""
        with self._lock:
            owed, self._owed = list(self._owed), collections.deque()
        for _, estimate in owed:
            self.charge(estimate)
            self.note_response(None)

    def _on_close(self, reason: str) -> None:
        self._settle_lost()
        if reason not in OWN_CLOSES and reason != "closed":
            self.network.report_realtime(False)
            with self._lock:
                self._failures[reason] += 1
                self._lost = self._lost or self.state == "connected"
            self._set_state("fallback_local", reason)
        with self._lock:
            listener = self._close_listener
        if listener is not None:
            listener(reason)

    # ----------------------------------------------------------------- sending

    def send(self, event: dict[str, Any]) -> bool:
        """Queue a client event; False when there is no session (nothing is sent). While a
        connect completes, its client (a cancel of an unrequested response)."""
        client = self._client or self._connecting
        if client is None:
            return False
        client.send(event)
        return True

    def request_response(self, instructions: str, tool_choice: str | None = None) -> bool:
        """``response.create`` with this response's instructions and ``tool_choice`` (left out
        without tools)."""
        with self._lock:
            self._expected += 1
        ok = self.send(protocol.response_create(instructions, tool_choice))  # type: ignore[arg-type]
        if not ok:
            with self._lock:
                self._expected = max(0, self._expected - 1)
        return ok

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

    def up(self) -> bool:
        """Connected and allowed to answer now (not blocked: e.g. a budget spent meanwhile)."""
        return self.connected and self.blocked(self.stack.effective()) is None

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
)
