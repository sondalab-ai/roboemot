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
    _thread: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.links = {
            r.name: Link(f"route.{r.name}", self.policy, self._route_changed) for r in self.routes
        }
        self.network = Link("network", self.policy, self.on_change)

    def set_policy(self, policy: LinkPolicy) -> None:
        """Hot reload (SPEC 11A): new thresholds apply from the next report or check."""
        self.policy = policy
        for link in (*self.links.values(), self.network):
            link.policy = policy

    @property
    def online(self) -> bool:
        return self.network.online

    def is_online(self, route: str) -> bool:
        link = self.links.get(route)
        return link is None or link.online

    def report(self, route: str, ok: bool, now: float | None = None) -> None:
        """Outcome of a real request on ``route`` (passive detection)."""
        if route in self.links:
            self.links[route].report(ok, self.clock() if now is None else now)

    def recheck(self, route: str) -> None:
        """Check ``route`` at the next opportunity (after a timeout: slow, or unreachable?)."""
        if route in self.links:
            self.links[route].next_check = -math.inf

    def snapshot(self) -> dict[str, object]:
        """Aggregate and per-route state for diagnostics; endpoints only, never credentials."""
        now = self.clock()
        routes = {}
        for route in self.routes:
            host, port = route_endpoint(route)
            routes[route.name] = {"endpoint": f"{host}:{port}"} | self.links[route.name].snapshot(
                now
            )
        return {"online": self.online, "routes": routes}

    def check_due(self, now: float | None = None) -> None:
        """Run the checks that are due (blocking up to ``probe_timeout_s`` each)."""
        now = self.clock() if now is None else now
        for route in self.routes:
            link = self.links[route.name]
            if link.due(now):
                host, port = route_endpoint(route)
                link.report(self.probe(host, port, self.policy.probe_timeout_s), self.clock())

    def _route_changed(self, event: str, now: float) -> None:
        if self.on_change is not None:
            self.on_change(event, now)
        self.network.set(any(link.online for link in self.links.values()), now)

    # ---- background checks

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="network-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.policy.probe_timeout_s * len(self.routes) + 1)

    def _run(self) -> None:
        if not self.links:
            return
        while not self._stop.is_set():
            self.check_due()
            soonest = min(link.next_check for link in self.links.values())
            self._stop.wait(max(0.0, soonest - self.clock()))


__all__ = ["NetworkMonitor", "Probe", "route_endpoint", "tcp_probe"]
