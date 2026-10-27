"""Reachy Mini realtime session provider: a session from the platform's allocator (SPEC 4A,
Session providers; SPEC 8, Realtime speech route; SPEC 12, Data).

Registered in the ``roboemot.realtime`` entry-point group under ``reachy-mini-wireless`` and used
in platform mode (``voice.realtime_route.mode: platform``). The route's ``base_url`` **is** the
allocator's URL (``configs/companion.yaml`` shows Pollen's), so the secret guard pins it, the
reachability probe checks it and locality reads it; nothing here names that URL.

``PlatformSession.allocate``:

- refuses a ``base_url`` whose path is not exactly ``/session``, or that has a query, fragment or
  user info (``provider_url``), that is ``http`` while the route is not local
  (``provider_insecure``), or that is this machine while not local (``provider_this_machine``):
  core's ``provider_endpoint``; nothing is sent then;
- POSTs a JSON object to the ``base_url`` through core's ``post_allocation`` (the route's own
  origin, no redirect, no proxy): ``{}``, or ``{"hardware_id": ...}`` only with the opt-in
  ``privacy.realtime_platform_id`` and when the daemon reports a well-formed one (read through
  ``DaemonClient``; with the opt-in off the daemon is not asked). The route's key goes in the
  ``X-Reachy-Mini-Authorization: Bearer`` header, only when the route has one;
- refuses any answer but 2xx (a redirect included) as ``provider_http_<code>``, a reply without
  a string ``connect_url`` as ``provider_reply``, an allocator it cannot reach as
  ``provider_unreachable``, and a ``connect_url`` whose path does not end in ``/realtime``
  (``url_path``, core's ``allocated_url``, which reads ``http``/``https`` as ``ws``/``wss``); core's
  ``connect_target`` then applies the URL and key rules, on both the allocator and the connect
  host.

The key may reach only the allocator's own origin (``KEY_HOSTS`` is empty): a keyed connect to
any other host is refused (``url_key_host``) and the local pipeline answers. Nothing here
logs the key, the identifier, the allocator's reply or a URL's query.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

from roboemot.realtime.session import (
    Allocation,
    AllocationRequest,
    ProviderRefused,
    UrlRefused,
    allocated_url,
    post_allocation,
    provider_endpoint,
)

from .daemon import REQUEST_TIMEOUT_S, DaemonClient, DaemonError

log = logging.getLogger(__name__)

# Hosts besides the allocator's own origin that may receive the route's key over wss (SPEC 8,
# URL and key rules): none. The allocator hands out a host of its choice, and anyone can create
# a Space under hf.space, so a suffix would let the allocator alone pick who gets the token. The
# real connect host is recorded on the robot (SR-5) and then allowed exactly.
KEY_HOSTS: tuple[str, ...] = ()
SESSION_PATH = "/session"
AUTH_HEADER = "X-Reachy-Mini-Authorization"
BEARER = "Bearer "
ID_FIELD = "hardware_id"
ID_SHAPE = re.compile(r"[A-Za-z0-9_-]{1,64}")  # the daemon's is 16 hex characters
CONNECT_URL = "connect_url"


def _daemon_client(timeout_s: float) -> DaemonClient:
    """The client for this machine's daemon, with ``timeout_s`` per request."""
    return DaemonClient(timeout_s=timeout_s)


class PlatformSession:
    """Allocates realtime sessions from the platform's allocator (module docstring).
    ``daemon_client(timeout_s)`` builds the daemon client (tests pass one with a mock
    transport)."""

    key_hosts: tuple[str, ...] = KEY_HOSTS

    def __init__(self, daemon_client: Callable[[float], DaemonClient] = _daemon_client) -> None:
        self._daemon_client = daemon_client

    async def allocate(self, request: AllocationRequest) -> Allocation:
        url = provider_endpoint(request.route, SESSION_PATH)
        body: dict[str, str] = {}
        if request.platform_id:
            hardware_id = await self._hardware_id(request.timeout_s)
            if hardware_id is not None:
                body[ID_FIELD] = hardware_id
        headers = {AUTH_HEADER: BEARER + request.key} if request.key else {}
        if ID_FIELD in body:
            request.on_platform_id()  # the report's platform_id_sent, even if the POST fails
        try:
            resp = await post_allocation(request, url, json=body, headers=headers)
        except (UrlRefused, ProviderRefused):
            raise
        except Exception:  # an httpx error can carry the URL: the reason code only
            raise ProviderRefused("provider_unreachable") from None
        if not resp.is_success:
            raise ProviderRefused(f"provider_http_{resp.status_code}")
        return Allocation(_connect_url(resp), KEY_HOSTS)

    async def _hardware_id(self, timeout_s: float) -> str | None:
        """The daemon's identifier when it is well formed, else None (logged by type only);
        the daemon gets at most the connect's time left."""
        wait_s = min(REQUEST_TIMEOUT_S, timeout_s) if timeout_s > 0 else REQUEST_TIMEOUT_S
        try:
            value = await asyncio.to_thread(self._daemon_client(wait_s).hardware_id)
        except DaemonError as exc:
            log.info("platform identifier not read (%s)", type(exc.__cause__ or exc).__name__)
            return None
        if value is None or ID_SHAPE.fullmatch(value) is None:
            return None
        return value


def _connect_url(resp: Any) -> str:
    """The reply's ``connect_url`` through core's ``allocated_url`` (``provider_reply`` when the
    reply is not a JSON object with one)."""
    try:
        data = resp.json()
    except ValueError:
        raise ProviderRefused("provider_reply") from None
    return allocated_url(data.get(CONNECT_URL) if isinstance(data, dict) else None)


def provider(manifest: Any) -> PlatformSession:
    """The ``roboemot.realtime`` factory: the manifest is not needed."""
    return PlatformSession()


__all__ = ["AUTH_HEADER", "KEY_HOSTS", "PlatformSession", "provider"]
