"""Where a realtime session connects, and the rules a connect URL must pass (SPEC 4A, 8, 12).

A :class:`SessionProvider` turns a realtime route into an :class:`Allocation`: the WebSocket URL
to connect to and the host suffixes the provider trusts with the route's key. Core ships
:class:`DirectSession` (the URL derived from the route's ``base_url``); a robot package may
register a provider in the ``roboemot.realtime`` entry-point group under the manifest
``robot.id`` (a factory taking the manifest), for a service that allocates sessions.

``connect_target`` applies the URL rules to whatever URL a provider returns, in both modes:

- ``ws`` or ``wss`` only, and ``wss`` unless the route and the connect host are both local
  (``url_is_local``, the route's effective locality);
- no user info, no fragment, no whitespace, control character or backslash, an ASCII host, and
  at most ``URL_MAX_CHARS`` characters; the host and port read by httpx and by the WebSocket
  library must agree;
- not this machine or a link-local address (``this_machine``, the web UI's
  ``url_local_refused`` rule) unless local;
- the query is kept for the connection, never logged or reported (``ConnectTarget`` hides the
  URL from ``repr``; ``origin`` is ``scheme://host:port`` only);
- the key is sent only to the route's own host, or to a host matching one of the provider's
  ``key_hosts`` suffixes, declared in code (shown in the report). A keyed route whose connect
  host is neither is refused, nothing is sent.

A refusal raises :class:`UrlRefused` whose message is a reason code only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..config.schema import LLMRoute
from ..llm.pseudonyms import this_machine, url_is_local

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roboemot.realtime"
URL_MAX_CHARS = 2048
REALTIME_PATH = "/realtime"
WS_SCHEMES = {"http": "ws", "https": "wss"}
DEFAULT_PORTS = {"ws": 80, "wss": 443}
NO_KEY = ("", "none")


class UrlRefused(ValueError):
    """A connect URL broke a rule; ``reason`` (also the message) is a code, never the URL."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ProviderUnavailable(RuntimeError):
    """The provider registered for this robot cannot be loaded or built. The realtime route is
    then unavailable: there is no fallback to direct mode (it would send the key to the
    provider's allocator as if it were a realtime server)."""


@dataclass(frozen=True)
class AllocationRequest:
    """What a provider gets to allocate a session: the route (its ``base_url`` is the provider's
    own endpoint), the route's key (None for none; never logged, not in ``repr``), whether the
    platform identifier may be sent (an opt-in), the time it may take, and an optional httpx
    client factory (tests)."""

    route: LLMRoute
    key: str | None = field(default=None, repr=False)
    platform_id: bool = False
    timeout_s: float = 0.0
    http_client_factory: Callable[[], httpx.AsyncClient] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class Allocation:
    """Where to connect (not in ``repr``: its query can carry a token) and the key-host suffixes
    that apply to it (a subset of the provider's ``key_hosts``)."""

    connect_url: str = field(repr=False)
    key_hosts: tuple[str, ...] = ()


@runtime_checkable
class SessionProvider(Protocol):
    """Allocates realtime sessions. ``key_hosts`` is static: host suffixes (``.example.org``)
    or exact hosts that may receive the route's key, declared in code, shown in the report."""

    key_hosts: tuple[str, ...]

    async def allocate(self, request: AllocationRequest) -> Allocation: ...


@dataclass(frozen=True)
class ConnectTarget:
    """A URL that passed the rules: ``url`` (with its query; not in ``repr``), ``origin``
    (``scheme://host:port``, safe to report), whether the route counts as local for it, and
    whether the key is sent."""

    url: str = field(repr=False)
    origin: str
    local: bool
    send_key: bool


def realtime_url(base_url: str) -> str:
    """The WebSocket URL of a route ``base_url``: ``http`` becomes ``ws`` and ``https`` ``wss``,
    ``/realtime`` is appended to the path, the query is kept and the fragment dropped."""
    parts = urlsplit(base_url)
    scheme = WS_SCHEMES.get(parts.scheme.lower())
    if scheme is None:
        raise UrlRefused("scheme")
    path = parts.path.rstrip("/") + REALTIME_PATH
    return urlunsplit((scheme, parts.netloc, path, parts.query, ""))


class DirectSession:
    """Direct mode: connect to the route's own ``base_url`` + ``/realtime``. The key, if any, goes
    to that host only."""

    key_hosts: tuple[str, ...] = ()

    async def allocate(self, request: AllocationRequest) -> Allocation:
        return Allocation(realtime_url(request.route.base_url))


def _host_port(url: str) -> tuple[str, str, int]:
    """Scheme, host and port as httpx reads them, cross-checked with the WebSocket library's own
    parser (the one that connects); any disagreement is refused."""
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError, TypeError):
        raise UrlRefused("unparsable") from None
    scheme = parsed.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise UrlRefused("scheme")
    host = parsed.host.lower().rstrip(".")
    if not host:
        raise UrlRefused("unparsable")
    if not host.isascii():
        raise UrlRefused("host")
    port = parsed.port or DEFAULT_PORTS[scheme]
    try:
        split = urlsplit(url)
        other_host = (split.hostname or "").lower().rstrip(".")
        other_port = split.port or DEFAULT_PORTS[scheme]
    except ValueError:
        raise UrlRefused("unparsable") from None
    if (other_host, other_port) != (host, port):
        raise UrlRefused("unparsable")
    return scheme, host, port


def _suffix_ok(suffix: str) -> str | None:
    """A key-host entry normalised (lower case, no trailing dot), or None when it is unusable:
    empty, a bare dot, or a single label (it would match a whole top-level domain)."""
    s = suffix.strip().lower().rstrip(".")
    bare = s.lstrip(".")
    if not bare or "." not in bare or not bare.isascii():
        return None
    return s


def host_matches(host: str, key_hosts: tuple[str, ...]) -> bool:
    """Whether ``host`` is one of ``key_hosts``: an entry that starts with a dot matches any
    host below it (on a label boundary, never the bare domain's look-alikes); another entry
    matches that exact host."""
    host = host.lower().rstrip(".")
    for entry in key_hosts:
        s = _suffix_ok(entry)
        if s is None:
            continue
        if s.startswith("."):
            if host.endswith(s) and len(host) > len(s):
                return True
        elif host == s:
            return True
    return False


def connect_target(
    route: LLMRoute, allocation: Allocation, provider_key_hosts: tuple[str, ...] = ()
) -> ConnectTarget:
    """Check ``allocation.connect_url`` against the URL rules (module docstring) for ``route``.
    ``provider_key_hosts`` is the provider's static declaration; an allocation that names a
    suffix outside it is refused (``key_hosts``)."""
    url = allocation.connect_url
    if not isinstance(url, str) or not url:
        raise UrlRefused("unparsable")
    if len(url) > URL_MAX_CHARS:
        raise UrlRefused("too_long")
    if any(c.isspace() or ord(c) < ord(" ") or c in "\\\x7f" for c in url):
        raise UrlRefused("characters")
    if not set(allocation.key_hosts) <= set(provider_key_hosts):
        raise UrlRefused("key_hosts")
    scheme, host, port = _host_port(url)
    split = urlsplit(url)
    if "@" in split.netloc:
        raise UrlRefused("userinfo")
    if split.fragment or url.endswith("#"):
        raise UrlRefused("fragment")
    local = url_is_local(route.base_url, route.locality) and url_is_local(url, route.locality)
    if scheme != "wss" and not local:
        raise UrlRefused("insecure")
    if this_machine(url) and not local:
        raise UrlRefused("this_machine")
    keyed = route.api_key not in NO_KEY
    own = _host_port_of_route(route)
    send_key = keyed and (host == own or host_matches(host, allocation.key_hosts))
    if keyed and not send_key:
        raise UrlRefused("key_host")
    bracket = f"[{host}]" if ":" in host else host
    return ConnectTarget(url, f"{scheme}://{bracket}:{port}", local, send_key)


def _host_port_of_route(route: LLMRoute) -> str:
    """The host of the route's ``base_url`` (httpx), lower case, no trailing dot."""
    try:
        return httpx.URL(route.base_url).host.lower().rstrip(".")
    except (httpx.InvalidURL, ValueError, TypeError):
        return ""


def load_session_provider(robot_id: str) -> Callable[[Any], SessionProvider] | None:
    """The provider factory registered for ``robot_id`` in ``roboemot.realtime`` (call it with the
    manifest), or None when no package registers one (direct mode). Fails closed: an entry point
    that cannot be loaded, or two of them for one robot, raise :class:`ProviderUnavailable`
    (logged by type only)."""
    found = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == robot_id]
    if not found:
        return None
    if len(found) > 1:
        raise ProviderUnavailable(f"{len(found)} realtime providers registered for {robot_id}")
    try:
        factory = found[0].load()
    except Exception as exc:
        log.warning("realtime provider %s: %s", robot_id, type(exc).__name__)
        raise ProviderUnavailable(f"realtime provider of {robot_id} cannot be loaded") from None
    if not callable(factory):
        raise ProviderUnavailable(f"realtime provider of {robot_id} is not a factory")
    return factory  # type: ignore[no-any-return]


def build_session_provider(robot_id: str, manifest: Any) -> SessionProvider:
    """The registered provider built with ``manifest``, or :class:`DirectSession` when none is
    registered. A provider that fails to build or breaks the protocol raises
    :class:`ProviderUnavailable` (never direct mode in its place)."""
    factory = load_session_provider(robot_id)
    if factory is None:
        return DirectSession()
    try:
        provider = factory(manifest)
    except Exception as exc:
        log.warning("realtime provider %s: %s", robot_id, type(exc).__name__)
        raise ProviderUnavailable(f"realtime provider of {robot_id} cannot be built") from None
    if not isinstance(provider, SessionProvider):
        raise ProviderUnavailable(f"realtime provider of {robot_id} breaks SessionProvider")
    return provider
