"""Host and Origin checks against DNS rebinding and cross-site requests (SPEC 12).

Same rule as a robot daemon's trusted-host middleware: a Host header must name an IP literal,
``localhost``, a ``.local`` mDNS name, or a name in ``webui.allowed_hosts``. A rebinding
attack arrives under the attacker's public DNS name, which is none of these.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from urllib.parse import urlsplit

LOCAL_SUFFIX = ".local"
LOCALHOST = "localhost"


def hostname_of(netloc: str) -> str | None:
    """Host name without port or brackets, lower case, no trailing dot; None if malformed."""
    try:
        host = urlsplit(f"//{netloc}").hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def is_trusted_host(host: str | None, allowed: Iterable[str] = ()) -> bool:
    """True for an IP literal, ``localhost``, ``*.local`` or an allowed name."""
    if not host:
        return False
    host = host.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host == LOCALHOST or host.endswith(LOCAL_SUFFIX) or host in set(allowed)
    return True


def same_origin(origin: str, host_header: str) -> bool:
    """The Origin of a browser request is this very server (scheme http, same host:port)."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme == "http" and parts.netloc.lower() == host_header.lower()
