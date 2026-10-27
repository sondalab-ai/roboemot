"""Where a realtime session connects, and the rules a connect URL must pass (SPEC 4A, 8, 12).

A :class:`SessionProvider` turns a realtime route into an :class:`Allocation`: the WebSocket URL
to connect to and the host suffixes the provider trusts with the route's key. Core ships
:class:`DirectSession` (the URL derived from the route's ``base_url``); a robot package may
register a provider in the ``roboemot.realtime`` entry-point group under the manifest
``robot.id`` (a factory taking the manifest), for a service that allocates sessions.

``connect_target`` applies the URL rules to whatever URL a provider returns, in both modes:

- ``ws`` or ``wss`` only, and ``wss`` unless the route and the connect host are both local
  (``url_is_local``: the route's effective locality; a connect host other than the route's own
  is judged by its address, whatever ``locality`` says);
- no user info, no fragment, no whitespace, control character or backslash, an ASCII host, and
  at most ``URL_MAX_CHARS`` characters; the host and port read by httpx and by the WebSocket
  library must agree;
- not this machine or a link-local address (``this_machine``, the web UI's
  ``url_local_refused`` rule) unless local;
- the query is kept for the connection, never logged or reported (``ConnectTarget`` hides the
  URL from ``repr``; ``origin`` is ``scheme://host:port`` only);
- the key is sent only to the route's own origin (scheme, host and port of its ``base_url``,
  ``https`` read as ``wss``), or over ``wss`` to a host matching one of the provider's
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
IDENTITY = "identity"
# headers that describe the body as received; the rebuilt reply holds it decoded and whole
BODY_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


class UrlRefused(ValueError):
    """A connect URL broke a rule; ``reason`` (also the message) is a code, never the URL.
    ``host`` is the refused host for ``key_host`` (a host only, never a path or query), so the
    host a provider allocated can be recorded and declared."""

    def __init__(self, reason: str, host: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.host = host


class ProviderRefused(RuntimeError):
    """A provider could not allocate a session: its endpoint answered with an error status
    (``provider_http_<code>``), an unusable reply, or the route does not fit it. ``reason`` (also
    the message) is a code only, never a URL, body, key or identifier; the client closes with it
    and the local pipeline answers."""

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
    platform identifier may be sent (the opt-in, read for this allocation), the time it may take,
    an optional httpx client factory (tests), and ``on_platform_id``, which the provider calls
    right before it sends a request that carries the identifier (the report's
    ``platform_id_sent``; never the value)."""

    route: LLMRoute
    key: str | None = field(default=None, repr=False)
    platform_id: bool = False
    timeout_s: float = 0.0
    http_client_factory: Callable[[], httpx.AsyncClient] | None = field(
        default=None, repr=False, compare=False
    )
    on_platform_id: Callable[[], None] = field(default=lambda: None, repr=False, compare=False)
    max_reply_bytes: int = 0  # the reply's size cap (0: none); the client's max_message_bytes


@dataclass(frozen=True)
class Allocation:
    """Where to connect (not in ``repr``: its query can carry a token) and the key-host suffixes
    that apply to it (a subset of the provider's ``key_hosts``)."""

    connect_url: str = field(repr=False)
    key_hosts: tuple[str, ...] = ()


@runtime_checkable
class SessionProvider(Protocol):
    """Allocates realtime sessions. ``key_hosts`` is static: host suffixes (``.example.org``)
    or exact hosts that may receive the route's key, declared in code, shown in the report.
    A provider makes its allocation request with :func:`post_allocation` (never its own HTTP
    client): it holds the request to the route's own origin, without redirects or proxies."""

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


def _http_origin(url: str) -> tuple[str, str, int] | None:
    """Scheme, host and port of an http(s) URL as httpx reads it (the default port when none is
    written), or None when it is not one, holds user info, a fragment or odd characters."""
    if any(c.isspace() or ord(c) < ord(" ") or c in "\\\x7f" for c in url):
        return None
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, ValueError, TypeError):
        return None
    scheme = parsed.scheme.lower()
    if scheme not in WS_SCHEMES or not parsed.host or parsed.userinfo or parsed.fragment:
        return None
    port = parsed.port or (DEFAULT_PORTS["wss"] if scheme == "https" else DEFAULT_PORTS["ws"])
    return scheme, parsed.host.lower().rstrip("."), port


def allocation_client(request: AllocationRequest) -> httpx.AsyncClient:
    """The HTTP client of an allocation request: the request's factory (tests), else one that
    reads no proxy or certificate settings from the environment and follows no redirect."""
    if request.http_client_factory is not None:
        return request.http_client_factory()
    return httpx.AsyncClient(
        trust_env=False, follow_redirects=False, timeout=max(request.timeout_s, 0.0) or None
    )


async def post_allocation(
    request: AllocationRequest,
    url: str,
    *,
    json: Any = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """POST an allocation request to ``url``, which must be at the route's own origin (scheme,
    host and port of its ``base_url``: the origin the key was approved for, SPEC 8, 12). A
    redirect is returned as it is, never followed. Raises :class:`UrlRefused`
    (``allocator_origin``) before anything is sent when ``url`` is elsewhere. A reply body over
    ``request.max_reply_bytes`` is not read further: :class:`ProviderRefused`
    (``provider_reply``). The request asks for ``identity`` and a compressed reply is refused
    (it would expand before the cap counts it), as is a ``Content-Length`` over the cap, before
    the body is read; bytes are counted as received."""
    target = _http_origin(url)
    if target is None or target != _http_origin(request.route.base_url):
        raise UrlRefused("allocator_origin")
    cap = request.max_reply_bytes
    sent = {**(headers or {}), "Accept-Encoding": IDENTITY}  # the cap counts what is sent
    async with allocation_client(request) as client:
        req = client.build_request("POST", url, json=json, headers=sent)
        resp = await client.send(req, stream=True, follow_redirects=False)
        try:
            if resp.headers.get("content-encoding", IDENTITY).strip().lower() != IDENTITY:
                raise ProviderRefused("provider_reply")  # it would expand before the cap counts
            length = resp.headers.get("content-length")
            if length is not None and (not length.isdigit() or (cap > 0 and int(length) > cap)):
                raise ProviderRefused("provider_reply")
            body = await _raw_body(resp, cap)
        finally:
            await resp.aclose()
        kept = {k: v for k, v in resp.headers.items() if k.lower() not in BODY_HEADERS}
        return httpx.Response(resp.status_code, headers=kept, content=bytes(body), request=req)


async def _raw_body(resp: httpx.Response, cap: int) -> bytes:
    """The reply's body as received, refused (``provider_reply``) once it passes ``cap`` (0:
    no cap). A reply whose body was already loaded (a test transport) is checked whole; its
    encoding is identity, so it is the same bytes."""
    body = bytearray()
    try:
        async for chunk in resp.aiter_raw():
            body += chunk
            if cap > 0 and len(body) > cap:
                raise ProviderRefused("provider_reply")
    except httpx.StreamConsumed:
        body = bytearray(resp.content)
        if cap > 0 and len(body) > cap:
            raise ProviderRefused("provider_reply") from None
    return bytes(body)


def provider_endpoint(route: LLMRoute, path: str) -> str:
    """The route's ``base_url`` when it can be a provider's endpoint at ``path``: exactly that
    path, no query, fragment or user info (``provider_url``), ``https`` unless the route is local
    (``provider_insecure``), and not this machine unless local (``provider_this_machine``).
    Otherwise :class:`ProviderRefused` with the reason, before anything is sent."""
    url = route.base_url
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 - raises on a bad port
    except ValueError:
        raise ProviderRefused("provider_url") from None
    host = parts.hostname or ""
    if (
        parts.path != path
        or parts.query
        or parts.fragment
        or "@" in parts.netloc
        or url.endswith(("?", "#"))
        or not _plain_host(host)
    ):
        raise ProviderRefused("provider_url")
    local = url_is_local(url, route.locality)
    if parts.scheme.lower() != "https" and not local:
        raise ProviderRefused("provider_insecure")
    if this_machine(url) and not local:
        raise ProviderRefused("provider_this_machine")
    return url


def allocated_url(url: object) -> str:
    """A connect URL a provider was handed: a non-empty string (else ``provider_reply``) whose
    path ends in ``/realtime`` (else :class:`UrlRefused` ``path``); ``http`` and ``https`` are
    read as ``ws`` and ``wss``, the rest is kept. :func:`connect_target` applies the other rules."""
    if not isinstance(url, str) or not url:
        raise ProviderRefused("provider_reply")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise UrlRefused("unparsable") from None
    if not parts.path.endswith(REALTIME_PATH):
        raise UrlRefused("path")
    scheme = WS_SCHEMES.get(parts.scheme.lower())
    return url if scheme is None else urlunsplit((scheme, *parts[1:]))


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
    if not _plain_host(host):
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


def _plain_host(host: str) -> bool:
    """A non-empty ASCII host with no escape (``%``) and no internationalised label (``xn--``):
    URL parsers can disagree on the others, and they allow look-alikes."""
    return (
        bool(host)
        and host.isascii()
        and "%" not in host
        and not any(label.startswith("xn--") for label in host.lower().split("."))
    )


def _suffix_ok(suffix: str) -> str | None:
    """A key-host entry normalised (lower case, no trailing dot), or None when it is unusable:
    empty, a bare dot, a single label (it would match a whole top-level domain), or an
    internationalised name, in Unicode or punycode (``xn--`` labels): such entries are dropped,
    so a look-alike host can never be declared."""
    s = suffix.strip().lower().rstrip(".")
    bare = s.lstrip(".")
    if not bare or "." not in bare or not bare.isascii():
        return None
    if any(label.startswith("xn--") for label in bare.split(".")):
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
    own = route_origin(route)
    # The route's locality speaks for its own host only: another host (one a provider chose) is
    # judged by its address (auto), so an explicit ``local`` cannot cover a cloud host.
    there = route.locality if own is not None and host == own[1] else "auto"
    local = url_is_local(route.base_url, route.locality) and url_is_local(url, there)
    if scheme != "wss" and not local:
        raise UrlRefused("insecure")
    if this_machine(url) and not local:
        raise UrlRefused("this_machine")
    keyed = route.api_key not in NO_KEY
    # The approval trusts the route's scheme, host and port (``secret_violations``): the key goes
    # to that exact origin, or over wss to a host the provider declared.
    at_route = (scheme, host, port) == own
    # a declared host gets the key on the wss port only, never on a port the allocator names
    declared = (
        scheme == "wss"
        and port == DEFAULT_PORTS["wss"]
        and host_matches(host, allocation.key_hosts)
    )
    send_key = keyed and (at_route or declared)
    if keyed and not send_key:
        raise UrlRefused("key_host", host)
    bracket = f"[{host}]" if ":" in host else host
    return ConnectTarget(url, f"{scheme}://{bracket}:{port}", local, send_key)


def route_origin(route: LLMRoute) -> tuple[str, str, int] | None:
    """Scheme, host and port of the route's own WebSocket origin (``realtime_url`` of its
    ``base_url``: https gives wss, http ws, the default port when none is written), or None
    when the base URL cannot give one."""
    try:
        return _host_port(realtime_url(route.base_url))
    except UrlRefused:
        return None


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
