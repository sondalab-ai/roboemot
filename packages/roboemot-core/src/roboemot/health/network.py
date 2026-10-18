"""Reachability of the LLM routes (SPEC 12A; routes in SPEC 8).

Each route gets a :class:`~roboemot.health.Link`. Real requests report their outcome
(``report``), or ask for a prompt check after a timeout (``recheck``). A background thread
checks due routes with a TCP connect to the route's host and port, which needs no API feature
beyond the OpenAI-compatible subset and no credentials. The aggregate ``network`` link is
online while any route is: it emits ``link.network.lost`` when the last route goes down and
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

from ..config.schema import LinkPolicy, LLMRoute
from . import Link, OnChange

DEFAULT_PORTS = {"https": 443, "http": 80}

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
    _thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.links = {
            r.name: Link(f"route.{r.name}", self.policy, self._route_changed) for r in self.routes
        }
        self.network = Link("network", self.policy, self.on_change)

    def set_routes(self, routes: list[LLMRoute]) -> None:
        """Hot reload (SPEC 11A): follow a changed route list. New names get a link (online until
        a check says otherwise), removed names lose theirs, unchanged names keep their link state
        (a changed endpoint is probed from the next check). Safe from any thread; wakes the
        background thread so a new route is probed without waiting for the next due check."""
        with self._lock:
            if routes == self.routes:
                return
            old = self.links
            self.links = {
                r.name: old.get(r.name) or Link(f"route.{r.name}", self.policy, self._route_changed)
                for r in routes
            }
            self.routes = list(routes)
            online = not self.links or any(link.online for link in self.links.values())
        self.network.set(online, self.clock())
        self._wake.set()

    def set_policy(self, policy: LinkPolicy) -> None:
        """Hot reload (SPEC 11A): new thresholds apply from the next report or check."""
        self.policy = policy
        for link in (*self._links(), self.network):
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
            link.report(ok, self.clock() if now is None else now)

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
        return {"online": self.online, "routes": routes}

    def check_due(self, now: float | None = None) -> None:
        """Run the checks that are due (blocking up to ``probe_timeout_s`` each)."""
        now = self.clock() if now is None else now
        for route, link in self._pairs():
            if link.due(now):
                host, port = route_endpoint(route)
                link.report(self.probe(host, port, self.policy.probe_timeout_s), self.clock())

    def _route_changed(self, event: str, now: float) -> None:
        if self.on_change is not None:
            self.on_change(event, now)
        self.network.set(any(link.online for link in self._links()), now)

    def _links_map(self) -> dict[str, Link]:
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
        self._thread = threading.Thread(target=self._run, name="network-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=self.policy.probe_timeout_s * len(self.routes) + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            self.check_due()
            soonest = min((link.next_check for link in self._links()), default=math.inf)
            self._wake.wait(None if math.isinf(soonest) else max(0.0, soonest - self.clock()))


__all__ = ["NetworkMonitor", "Probe", "route_endpoint", "tcp_probe"]
