"""Find robots on the local network by mDNS (SPEC 4, Discovery).

A manifest declares how its robot announces itself (``robot.discovery``): an mDNS service type
and, optionally, TXT values that must match (e.g. ``model`` to tell Wireless from Lite). This
module browses those service types and returns each answering robot with the manifest it
matches. The mDNS layer is the optional ``zeroconf`` dependency (``roboemot-core[discovery]``),
behind a ``Browse`` callable so callers and tests can substitute it.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..manifest import RobotInfo

DEFAULT_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class ServiceRecord:
    """One resolved mDNS service instance."""

    service_type: str  # "_reachy-mini._tcp", without ".local."
    name: str  # instance name, e.g. "Neve"
    host: str  # advertised host, e.g. "reachy-mini.local"
    addresses: tuple[str, ...]
    port: int
    txt: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FoundRobot:
    """A robot that answered, with the manifest it matches."""

    manifest_id: str
    record: ServiceRecord

    @property
    def address(self) -> str:
        """Best address to connect to: IPv4 first (mDNS hosts often resolve to link-local v6)."""
        v4 = [a for a in self.record.addresses if ipaddress.ip_address(a).version == 4]
        return (v4 or list(self.record.addresses) or [self.record.host])[0]


Browse = Callable[[list[str], float], list[ServiceRecord]]


def load_hints(paths: Iterable[Path]) -> list[RobotInfo]:
    """Read ``robot`` from each manifest that declares ``discovery``; limits are not checked."""
    hints = []
    for path in paths:
        info = RobotInfo.model_validate(yaml.safe_load(path.read_text())["robot"])
        if info.discovery is not None:
            hints.append(info)
    return hints


def match(hints: list[RobotInfo], records: Iterable[ServiceRecord]) -> list[FoundRobot]:
    """Pair each record with the first manifest whose service type and TXT values match."""
    found = []
    for rec in records:
        for info in hints:
            d = info.discovery
            assert d is not None
            if d.mdns_service == rec.service_type and all(
                rec.txt.get(k) == v for k, v in d.match_txt.items()
            ):
                found.append(FoundRobot(info.id, rec))
                break
    return sorted(found, key=lambda f: (f.manifest_id, f.record.name))


def discover(
    hints: list[RobotInfo], timeout_s: float = DEFAULT_TIMEOUT_S, browse: Browse | None = None
) -> list[FoundRobot]:
    """Browse the service types the manifests declare for ``timeout_s`` and match the answers."""
    types = sorted({h.discovery.mdns_service for h in hints if h.discovery})
    if not types:
        return []
    return match(hints, (browse or zeroconf_browse)(types, timeout_s))


def resolve_host(host: str, port: int) -> tuple[str, ...]:
    """Addresses of ``host`` from the OS resolver, IPv4 first; empty when unknown."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return ()
    addrs = dict.fromkeys(str(i[4][0]).split("%")[0] for i in infos)
    return tuple(sorted(addrs, key=lambda a: ipaddress.ip_address(a).version))


def zeroconf_browse(service_types: list[str], timeout_s: float) -> list[ServiceRecord]:
    """Browse with python-zeroconf, then resolve every instance seen."""
    try:
        from zeroconf import ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf
    except ImportError as e:  # pragma: no cover
        hint = 'pip install "roboemot-core[discovery]"'
        raise RuntimeError(f"discovery needs zeroconf: {hint}") from e
    import threading

    seen: set[tuple[str, str]] = set()
    lock = threading.Lock()

    def on_change(zeroconf, service_type, name, state_change) -> None:  # noqa: ANN001
        if state_change is ServiceStateChange.Added:
            with lock:
                seen.add((service_type, name))

    zc = Zeroconf()
    try:
        browsers = [ServiceBrowser(zc, f"{t}.local.", handlers=[on_change]) for t in service_types]
        threading.Event().wait(timeout_s)
        for b in browsers:
            b.cancel()
        records = []
        for full_type, full_name in sorted(seen):
            info = ServiceInfo(full_type, full_name)
            info.request(zc, int(timeout_s * 1000))
            if not info.server or not info.port:
                continue
            host = info.server.rstrip(".")
            # Reachy Mini 1.11 answers SRV/TXT reliably but its A/AAAA records only some of the
            # time; the OS resolver (mDNSResponder, avahi) still knows the host.
            addresses = tuple(info.parsed_addresses()) or resolve_host(host, info.port)
            records.append(
                ServiceRecord(
                    service_type=full_type.removesuffix(".local."),
                    name=full_name.removesuffix(f".{full_type}"),
                    host=host,
                    addresses=addresses,
                    port=info.port,
                    txt={
                        k.decode(): (v or b"").decode(errors="replace")
                        for k, v in info.properties.items()
                    },
                )
            )
        return records
    finally:
        zc.close()


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "Browse",
    "FoundRobot",
    "ServiceRecord",
    "discover",
    "load_hints",
    "match",
    "resolve_host",
    "zeroconf_browse",
]
