"""Reachability of the LLM routes (SPEC 12A; routes in SPEC 8).

Each route gets a :class:`~roboemot.health.Link`. Real requests report their outcome
(``report``), or ask for a prompt check after a timeout (``recheck``). A background thread
checks due routes with a TCP connect to the route's host and port, which needs no API feature
beyond the OpenAI-compatible subset and no credentials.

The aggregate ``network`` link is online while any route that can answer a turn is: the chat
routes, plus the ``realtime`` link (the realtime speech route's session, reported by
``RealtimeRoute``) while realtime mode is on (``set_realtime``). A ``kind: realtime`` route's own
``route.<name>`` link never counts, so a live realtime route cannot mask dead chat routes in local
mode. It emits ``link.network.lost`` when the last counted link goes down and
``link.network.restored`` when one comes back.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .. import shutdown
from ..config.schema import LinkPolicy, LLMRoute
from . import Link, OnChange

DEFAULT_PORTS = {"https": 443, "http": 80}
REALTIME_LINK = "realtime"  # the realtime speech route's session (SPEC 12A, Links)

Probe = Callable[[str, int, float], bool]  # (host, port, timeout_s) -> reachable


def tcp_probe(host: str, port: int, timeout_s: float) -> bool:
    """True if a TCP connection to ``host:port`` opens within ``timeout_s``."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def route_endpoint(route: LLMRoute) -> tuple[str, int]:
    url = urlsplit(route.base_url)
    return url.hostname or "", url.port or DEFAULT_PORTS.get(url.scheme, DEFAULT_PORTS["https"])


@dataclass
class NetworkMonitor:
    routes: list[LLMRoute]
    policy: LinkPolicy
    on_change: OnChange | None = None
    probe: Probe = tcp_probe
    clock: Callable[[], float] = time.monotonic
    links: dict[str, Link] = field(default_factory=dict)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _wake: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _net_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _unconfirmed: set[str] = field(default_factory=set, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.links = {
            r.name: Link(f"route.{r.name}", self.policy, self._route_changed) for r in self.routes
        }
        self.network = Link("network", self.policy, self.on_change)
        self.realtime = Link(REALTIME_LINK, self.policy, self._route_changed)
        self._realtime_active = False
        self._realtime_confirmed = False  # counts once a session opened, or once offline
        self._realtime_endpoint: tuple[str, int] | None = None

    def set_realtime(self, active: bool, endpoint: tuple[str, int] | None = None) -> None:
        """Realtime mode is on (the realtime route may answer a turn) or off: the ``realtime``
        link counts for the ``network`` aggregate only while it is on. A new ``endpoint``
        (host, port) starts the link afresh, unconfirmed, as ``set_routes`` does for a chat
        route. Any thread."""
        with self._lock:
            changed, self._realtime_active = self._realtime_active != active, active
            if endpoint != self._realtime_endpoint:  # also through None: a fresh link
                self.realtime = Link(REALTIME_LINK, self.policy, self._route_changed)
                self._realtime_confirmed = False
                self._realtime_endpoint = endpoint
                changed = True
        if changed:
            self._sync_network(self.clock())

    def report_realtime(self, ok: bool, now: float | None = None) -> None:
        """Outcome of a realtime session (connected, or failed to connect or dropped)."""
        now = self.clock() if now is None else now
        confirms = False
        if ok:
            with self._lock:
                confirms, self._realtime_confirmed = not self._realtime_confirmed, True
        self.realtime.report(ok, now)
        if confirms:
            self._sync_network(now)

    def realtime_due(self, now: float | None = None) -> bool:
        """Whether a new realtime connect may be tried (``connectivity.network`` backoff)."""
        return self.realtime.due(self.clock() if now is None else now)

    def set_routes(self, routes: list[LLMRoute]) -> None:
        """Hot reload (SPEC 11A): follow a changed route list. Removed names lose their link;
        a name whose endpoint (host and port) is unchanged keeps its link state; a new name, or
        one whose endpoint changed, gets a fresh link that is probed at once. Such a link counts
        for the ``network`` aggregate only after its first successful probe (or once it has gone
        offline), so adding a route while every other one is down sends no false
        ``link.network.restored``; removing the last down route may restore it. Safe from any
        thread: the aggregate is recomputed in one place, under its own lock."""
        with self._lock:
            if routes == self.routes:
                return
            old, old_routes = self.links, {r.name: r for r in self.routes}
            links: dict[str, Link] = {}
            fresh: set[str] = set()
            for r in routes:
                prev = old_routes.get(r.name)
                if prev is not None and route_endpoint(prev) == route_endpoint(r):
                    links[r.name] = old[r.name]
                else:
                    links[r.name] = Link(f"route.{r.name}", self.policy, self._route_changed)
                    fresh.add(r.name)
            self._unconfirmed = (self._unconfirmed & set(links)) | fresh
            self.links = links  # replaced wholesale, never mutated (see _links_map)
            self.routes = list(routes)
        self._sync_network(self.clock())
        self._wake.set()

    def set_policy(self, policy: LinkPolicy) -> None:
        """Hot reload (SPEC 11A): new thresholds apply from the next report or check."""
        self.policy = policy
        for link in (*self._links(), self.network, self.realtime):
            link.policy = policy

    @property
    def online(self) -> bool:
        return self.network.online

    def is_online(self, route: str) -> bool:
        link = self._links_map().get(route)
        return link is None or link.online

    def report(self, route: str, ok: bool, now: float | None = None) -> None:
        """Outcome of a real request on ``route`` (passive detection)."""
        link = self._links_map().get(route)
        if link is not None:
            self._record(route, link, ok, self.clock() if now is None else now)

    def recheck(self, route: str) -> None:
        """Check ``route`` at the next opportunity (after a timeout: slow, or unreachable?)."""
        link = self._links_map().get(route)
        if link is not None:
            link.next_check = -math.inf

    def snapshot(self) -> dict[str, object]:
        """Aggregate and per-route state for diagnostics; endpoints only, never credentials."""
        now = self.clock()
        routes = {}
        for route, link in self._pairs():
            host, port = route_endpoint(route)
            routes[route.name] = {"endpoint": f"{host}:{port}"} | link.snapshot(now)
        with self._lock:
            active = self._realtime_active
        realtime = {"counted": active} | self.realtime.snapshot(now)
        return {"online": self.online, "routes": routes, "realtime": realtime}

    def check_due(self, now: float | None = None) -> None:
        """Run the checks that are due (blocking up to ``probe_timeout_s`` each)."""
        now = self.clock() if now is None else now
        for route, link in self._pairs():
            if link.due(now):
                host, port = route_endpoint(route)
                ok = self.probe(host, port, self.policy.probe_timeout_s)
                if self._links_map().get(route.name) is link:  # else removed or replaced meanwhile
                    self._record(route.name, link, ok, self.clock())

    def _record(self, name: str, link: Link, ok: bool, now: float) -> None:
        """Feed one outcome to ``link``; a first success makes a new link count for the
        aggregate."""
        confirms = False
        if ok:
            with self._lock:
                if name in self._unconfirmed and self.links.get(name) is link:
                    self._unconfirmed.discard(name)
                    confirms = True
        link.report(ok, now)
        if confirms:
            self._sync_network(now)

    def _route_changed(self, event: str, now: float) -> None:
        if self.on_change is not None:
            self.on_change(event, now)
        self._sync_network(now)

    def _sync_network(self, now: float) -> None:
        """The one place the ``network`` aggregate is computed: online while a counted link is
        (none that could count counts as online). The candidates are the chat routes' links and,
        in realtime mode, the ``realtime`` link. Links not yet confirmed count only once
        offline; if none counts, the aggregate keeps its state."""
        with self._net_lock:
            with self._lock:
                kinds = {r.name: r.kind for r in self.routes}
                chat = [
                    (name, link)
                    for name, link in self.links.items()
                    if kinds.get(name, "chat") != "realtime"
                ]
                counted = [
                    link for name, link in chat if name not in self._unconfirmed or not link.online
                ]
                if self._realtime_active and (self._realtime_confirmed or not self.realtime.online):
                    counted.append(self.realtime)  # the chat links' unconfirmed rule
                candidates = bool(chat) or self._realtime_active
                keep = candidates and not counted
            if not keep:
                self.network.set(not candidates or any(link.online for link in counted), now)

    def _links_map(self) -> dict[str, Link]:
        # The dict is replaced wholesale by set_routes and never mutated, so the caller may use
        # the returned reference without the lock.
        with self._lock:
            return self.links

    def _links(self) -> list[Link]:
        return list(self._links_map().values())

    def _pairs(self) -> list[tuple[LLMRoute, Link]]:
        """Routes with their links, consistent with each other even while ``set_routes`` runs."""
        with self._lock:
            return [(r, self.links[r.name]) for r in self.routes]

    # ---- background checks

    def start(self) -> None:
        thread = threading.Thread(target=self._run, name="network-monitor", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            shutdown.join(self._thread, self.policy.probe_timeout_s * len(self.routes) + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            if self._stop.is_set():  # stop() may have set _wake just before this clear
                break
            self.check_due()
            soonest = min((link.next_check for link in self._links()), default=math.inf)
            self._wake.wait(None if math.isinf(soonest) else max(0.0, soonest - self.clock()))


__all__ = ["NetworkMonitor", "Probe", "route_endpoint", "tcp_probe"]
