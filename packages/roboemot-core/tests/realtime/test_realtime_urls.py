"""URL and key rules of the realtime route, in direct and provider mode (SPEC 8, 12)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from fake_server import FakeServer
from realtime_helpers import KEY, UPDATE, bearer, route, settings
from roboemot.plugins import contract_violation
from roboemot.realtime import session as sess
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.session import (
    Allocation,
    AllocationRequest,
    ConnectTarget,
    DirectSession,
    UrlRefused,
    connect_target,
    host_matches,
    realtime_url,
)

QUERY_SECRET = "QSENTINEL42"


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://box.local:8765/v1", "ws://box.local:8765/v1/realtime"),
        ("https://api.example.org/v1/", "wss://api.example.org/v1/realtime"),
        ("https://api.example.org", "wss://api.example.org/realtime"),
        ("https://api.example.org/v1?tok=1#frag", "wss://api.example.org/v1/realtime?tok=1"),
    ],
)
def test_the_direct_url_is_derived_from_the_base_url(base, expected):
    assert realtime_url(base) == expected


def test_only_http_and_https_base_urls_derive():
    with pytest.raises(UrlRefused, match="scheme"):
        realtime_url("ftp://example.org/v1")


def target(url: str, *, locality: str = "cloud", keyed: bool = False, **kw) -> ConnectTarget:
    base = kw.pop("base", "https://api.example.org/v1")
    hosts = kw.pop("key_hosts", ())
    declared = kw.pop("declared", hosts)
    return connect_target(
        route(base, locality=locality, keyed=keyed), Allocation(url, hosts), declared
    )


@pytest.mark.parametrize(
    ("url", "reason", "kw"),
    [
        ("https://api.example.org/realtime", "scheme", {}),
        ("ws://api.example.org/realtime", "insecure", {}),
        ("wss://user:pw@api.example.org/realtime", "userinfo", {}),
        ("wss://api.example.org/" + "a" * 2048, "too_long", {}),
        ("wss://api.example.org/real time", "characters", {}),
        ("wss://api.example.org\\@evil.example/realtime", "characters", {}),
        ("wss://api.example.org/realtime#x", "fragment", {}),
        ("wss://127.0.0.1/realtime", "this_machine", {}),
        ("wss://[::ffff:127.0.0.1]/realtime", "this_machine", {}),
        ("wss://169.254.169.254/realtime", "this_machine", {}),
        ("wss://localhost/realtime", "this_machine", {}),
        ("wss://exämple.org/realtime", "host", {}),
        ("wss://other.example.net/realtime", "key_host", {"keyed": True}),
        (
            "wss://a.example.net/realtime",
            "key_hosts",
            {"key_hosts": (".example.net",), "declared": ()},
        ),
        ("", "unparsable", {}),
    ],
)
def test_every_refusal(url, reason, kw):
    with pytest.raises(UrlRefused) as exc:
        target(url, **kw)
    assert exc.value.reason == reason
    assert str(exc.value) == reason  # the message is the code, never the URL


def test_ws_is_allowed_only_when_the_route_and_the_host_are_local():
    lan = "http://192.168.1.20:8765/v1"
    assert target("ws://192.168.1.20:8765/v1/realtime", locality="auto", base=lan).local
    # an auto route on a LAN allocator cannot hand out a plain ws:// cloud host
    with pytest.raises(UrlRefused, match="insecure"):
        target("ws://rt.example.org/realtime", locality="auto", base=lan)
    # an explicit local route speaks for its own host only, not for one a provider chose
    with pytest.raises(UrlRefused, match="insecure"):
        target("ws://rt.example.org/realtime", locality="local", base=lan)
    assert target(
        "ws://box.example.org/v1/realtime", locality="local", base="http://box.example.org/v1"
    ).local
    # loopback only when local
    assert target(
        "ws://127.0.0.1:8765/realtime", locality="local", base="http://127.0.0.1/v1"
    ).local


def test_the_key_goes_to_the_route_host_or_a_declared_suffix():
    own = target("wss://api.example.org/realtime", keyed=True)
    assert own.send_key and own.origin == "wss://api.example.org:443"
    alloc = target(
        "wss://node7.rt.example.net/realtime?session=s",
        keyed=True,
        key_hosts=(".rt.example.net",),
    )
    assert alloc.send_key and alloc.origin == "wss://node7.rt.example.net:443"
    keyless = target("wss://other.example.net/realtime")
    assert keyless.send_key is False


@pytest.mark.parametrize(
    ("url", "locality"),
    [
        ("wss://api.example.org:9999/realtime", "cloud"),  # same host, another port
        ("ws://api.example.org/realtime", "local"),  # ws for an https base
        ("ws://api.example.org:443/realtime", "local"),  # ws on the https port
    ],
)
def test_the_key_needs_the_route_origin_not_only_its_host(url, locality):
    """Fix round 1, item 3: the approval trusts scheme, host and port (SPEC 8, 12)."""
    with pytest.raises(UrlRefused, match="key_host"):
        target(url, locality=locality, keyed=True)
    assert target(url, locality=locality).send_key is False  # keyless: allowed, no key


def test_the_route_origin_uses_default_and_explicit_ports():
    assert target("wss://api.example.org:443/v1/realtime", keyed=True).send_key
    base = "http://192.168.1.20:8765/v1"
    assert target("ws://192.168.1.20:8765/r", locality="local", keyed=True, base=base).send_key
    with pytest.raises(UrlRefused, match="key_host"):
        target("ws://192.168.1.20:8766/r", locality="local", keyed=True, base=base)


def test_key_hosts_receive_the_key_over_wss_only():
    with pytest.raises(UrlRefused, match="key_host"):
        target(
            "ws://node.rt.local/realtime",  # local by address, so only the key rule refuses
            locality="local",
            keyed=True,
            base="http://box.local/v1",
            key_hosts=(".rt.local",),
        )


@pytest.mark.parametrize(
    ("host", "hosts", "ok"),
    [
        ("a.hf.example", (".hf.example",), True),
        ("A.HF.EXAMPLE.", (".hf.example",), True),
        ("hf.example", (".hf.example",), False),
        ("evilhf.example", (".hf.example",), False),
        ("hf.example", ("hf.example",), True),
        ("x.hf.example", ("hf.example",), False),
        ("anything.example", (".",), False),
        ("anything.example", ("",), False),
        ("xn--e1afmkfd.example", (".xn--e1afmkfd.example",), False),  # punycode entries drop
        ("a.xn--e1afmkfd.example", (".xn--e1afmkfd.example",), False),
        ("a.example", (".example",), False),  # a single label would cover a whole TLD
    ],
)
def test_key_host_suffixes_match_on_a_label_boundary(host, hosts, ok):
    assert host_matches(host, hosts) is ok


def test_reprs_hide_the_url_and_the_key():
    t = target("wss://api.example.org/realtime?tok=" + QUERY_SECRET)
    req = AllocationRequest(route("https://api.example.org/v1"), key=KEY)
    for obj in (t, req, Allocation("wss://x.example/realtime?tok=" + QUERY_SECRET)):
        assert QUERY_SECRET not in repr(obj) and KEY not in repr(obj)


# ------------------------------------------------------------------- against the fake


class OtherHost:
    """A provider that allocates the fake on ``localhost`` while the route names 127.0.0.1."""

    key_hosts: tuple[str, ...] = ()

    def __init__(self, url: str) -> None:
        self.url = url
        self.requests: list[AllocationRequest] = []

    async def allocate(self, request: AllocationRequest) -> Allocation:
        self.requests.append(request)
        return Allocation(self.url)


def test_direct_mode_sends_the_key_to_the_route_host_only(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base(query=f"token={QUERY_SECRET}"), keyed=True),
            settings(),
            DirectSession(),
            auth_headers=lambda: {**bearer(), "X-Other": "no"},
        )
        try:
            assert c.connect(UPDATE)
            sent = server.requests[0]
            assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
            assert "X-Other" not in sent["headers"]
            assert sent["path"] == f"/v1/realtime?token={QUERY_SECRET}"  # kept for the connection
            snap = c.snapshot()
            assert snap["key_sent"] is True
        finally:
            c.close()
    for text in (caplog.text, repr(snap), repr(c)):
        assert KEY not in text and QUERY_SECRET not in text


def test_a_keyed_route_never_sends_its_key_to_another_host(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer(host="127.0.0.1") as server:
        other = f"ws://localhost:{server.port}/realtime?token={QUERY_SECRET}"
        provider = OtherHost(other)
        c = RealtimeClient(
            route(server.http_base(), keyed=True), settings(), provider, auth_headers=bearer
        )
        try:
            assert c.connect(UPDATE) is False
            assert c.reason == "url_key_host"
            assert server.requests == []  # nothing was sent anywhere
            assert provider.requests[0].key == KEY  # the allocator is the route's own host
        finally:
            c.close()
    assert KEY not in caplog.text and QUERY_SECRET not in caplog.text


def test_a_keyless_route_connects_to_an_allocated_host_without_a_key():
    with FakeServer() as server:
        provider = OtherHost(f"ws://localhost:{server.port}/realtime")
        c = RealtimeClient(route(server.http_base()), settings(), provider, auth_headers=bearer)
        try:
            assert c.connect(UPDATE)
            assert "Authorization" not in server.requests[0]["headers"]
            assert c.snapshot()["endpoint"] == f"ws://localhost:{server.port}"
        finally:
            c.close()


def test_a_redirect_is_never_followed(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as target_server:
        to = f"ws://localhost:{target_server.port}/realtime?token={QUERY_SECRET}"
        with FakeServer("redirect", redirect_to=to) as server:
            c = RealtimeClient(
                route(server.http_base(), keyed=True),
                settings(),
                DirectSession(),
                auth_headers=bearer,
            )
            try:
                assert c.connect(UPDATE) is False
                assert c.reason == "connect_failed"
                assert len(server.requests) == 1
                assert target_server.requests == []
            finally:
                c.close()
    assert KEY not in caplog.text and QUERY_SECRET not in caplog.text


def test_failures_never_log_the_query_or_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        base = server.http_base(query=f"token={QUERY_SECRET}")
    c = RealtimeClient(route(base, keyed=True), settings(), DirectSession(), auth_headers=bearer)
    try:
        assert c.connect(UPDATE) is False
    finally:
        c.close()
    cloud = RealtimeClient(
        route(f"http://rt.example.org/v1?token={QUERY_SECRET}", locality="cloud"),
        settings(),
        DirectSession(),
        auth_headers=dict,
    )
    try:
        assert cloud.connect(UPDATE) is False and cloud.reason == "url_insecure"
    finally:
        cloud.close()
    for text in (caplog.text, repr(c.snapshot()), repr(cloud.snapshot())):
        assert KEY not in text and QUERY_SECRET not in text


# ------------------------------------------------------------------- the provider seam


def fake_eps(monkeypatch, *eps):
    monkeypatch.setattr(sess, "entry_points", lambda group: [e for e in eps if group == e.group])


def ep(name: str, loader):
    return SimpleNamespace(name=name, group=sess.ENTRY_POINT_GROUP, load=loader)


def test_without_a_registered_provider_direct_mode_is_used(monkeypatch):
    fake_eps(monkeypatch)
    assert sess.load_session_provider("desk-head-v1") is None
    assert isinstance(sess.build_session_provider("desk-head-v1", object()), DirectSession)


def test_a_registered_provider_is_built_with_the_manifest(monkeypatch):
    seen = []

    def factory(manifest):
        seen.append(manifest)
        return OtherHost("wss://x.example/realtime")

    fake_eps(monkeypatch, ep("bot", lambda: factory), ep("other", lambda: None))
    manifest = object()
    provider = sess.build_session_provider("bot", manifest)
    assert isinstance(provider, OtherHost) and seen == [manifest]


def boom():
    raise ImportError("missing sdk")


@pytest.mark.parametrize(
    "eps",
    [
        [ep("bot", boom)],
        [ep("bot", lambda: OtherHost), ep("bot", lambda: OtherHost)],
        [ep("bot", lambda: lambda manifest: object())],
        [ep("bot", lambda: lambda manifest: 1 / 0)],
        [ep("bot", lambda: "not callable")],
    ],
)
def test_a_broken_provider_fails_closed(monkeypatch, eps):
    fake_eps(monkeypatch, *eps)
    with pytest.raises(sess.ProviderUnavailable):
        sess.build_session_provider("bot", object())


def test_the_group_contract_is_a_factory_of_the_manifest():
    assert contract_violation(sess.ENTRY_POINT_GROUP, lambda manifest: DirectSession()) is None
    assert contract_violation(sess.ENTRY_POINT_GROUP, lambda: DirectSession()) is not None


# ------------------------------------------------------------------- the allocation helper


def recording_request(base: str, responses: dict[str, httpx.Response]):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return responses.get(str(req.url), httpx.Response(404))

    request = AllocationRequest(
        route(base, keyed=True),
        key=KEY,
        timeout_s=2.0,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return request, seen


async def test_post_allocation_goes_to_the_route_origin_only():
    """Fix round 1, item 5: providers allocate through a core helper (SPEC 8, 12)."""
    ok = "https://alloc.example.org/session"
    request, seen = recording_request(
        "https://alloc.example.org", {ok: httpx.Response(200, json={"connect_url": "x"})}
    )
    resp = await sess.post_allocation(request, ok, json={}, headers={"X-Auth": KEY})
    assert resp.status_code == 200 and len(seen) == 1
    for elsewhere in (
        "https://alloc.example.org:8443/session",  # another port
        "http://alloc.example.org/session",  # another scheme
        "https://evil.example.org/session",  # another host
        "https://u:p@alloc.example.org/session",  # user info
    ):
        with pytest.raises(UrlRefused, match="allocator_origin"):
            await sess.post_allocation(request, elsewhere, json={})
    assert len(seen) == 1  # nothing else was sent


async def test_post_allocation_never_follows_a_redirect():
    url = "https://alloc.example.org/session"
    moved = httpx.Response(307, headers={"Location": "https://evil.example.org/session"})
    request, seen = recording_request("https://alloc.example.org/v1", {url: moved})
    resp = await sess.post_allocation(request, url, json={})
    assert resp.status_code == 307
    assert [str(r.url) for r in seen] == [url]


async def test_the_default_allocation_client_ignores_the_environment():
    request = AllocationRequest(route("https://alloc.example.org"), timeout_s=1.0)
    client = sess.allocation_client(request)
    try:
        assert client.trust_env is False and client.follow_redirects is False
    finally:
        await client.aclose()


# ------------------------------------------------------------------ SR-4: provider refusals


class Refusing:
    """A provider whose allocator answered with an error status."""

    key_hosts: tuple[str, ...] = ()

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.requests: list[AllocationRequest] = []

    async def allocate(self, request: AllocationRequest) -> Allocation:
        self.requests.append(request)
        raise sess.ProviderRefused(self.reason)


def test_a_provider_refusal_is_the_close_reason_and_nothing_connects(caplog):
    """SR-4: a provider names why it cannot allocate (``provider_http_<code>`` and the like);
    the client fails with that code, never ``connect_failed`` (SPEC 8, Session providers)."""
    caplog.set_level(logging.DEBUG)
    with FakeServer() as server:
        provider = Refusing("provider_http_503")
        closes: list[str] = []
        c = RealtimeClient(
            route(server.http_base(), keyed=True),
            settings(),
            provider,
            auth_headers=bearer,
            on_close=closes.append,
        )
        try:
            assert c.connect(UPDATE) is False
            assert c.reason == "provider_http_503" and closes == ["provider_http_503"]
            assert c.snapshot()["failures"] == {"provider_http_503": 1}
            assert server.requests == []
        finally:
            c.close()
    assert KEY not in caplog.text


def test_provider_refused_carries_a_reason_code_only():
    exc = sess.ProviderRefused("provider_url")
    assert exc.reason == "provider_url" and str(exc) == "provider_url"


def test_the_client_reports_whether_the_platform_identifier_was_sent():
    """SR-4: the opt-in reaches the provider, and the client's snapshot says whether the last
    allocation request carried the identifier (SPEC 12A, ``platform_id_sent``)."""

    class Sending(OtherHost):
        async def allocate(self, request: AllocationRequest) -> Allocation:
            self.requests.append(request)
            if request.platform_id:
                request.on_platform_id()  # right before the request with it is sent
            return Allocation(self.url)

    with FakeServer() as server:
        url = f"ws://localhost:{server.port}/realtime"
        for opted in (False, True):
            provider = Sending(url)
            c = RealtimeClient(
                route(server.http_base()), settings(), provider, auth_headers=bearer,
                platform_id=lambda opted=opted: opted,
            )  # fmt: skip
            try:
                assert c.snapshot()["platform_id_sent"] is False  # nothing allocated yet
                assert c.connect(UPDATE)
                assert provider.requests[0].platform_id is opted
                assert c.snapshot()["platform_id_sent"] is opted
            finally:
                c.close()


def test_the_identifier_opt_in_is_read_at_each_allocation():
    """Security review of SR-4: an opt-in turned off after the client was built is honoured at
    its next allocation (a reconnect), and the report follows (SPEC 12, Data)."""
    opted = [True]

    class Sending(Refusing):
        async def allocate(self, request: AllocationRequest) -> Allocation:
            if request.platform_id:
                request.on_platform_id()
            return await super().allocate(request)

    provider = Sending("provider_http_503")
    c = RealtimeClient(
        route("http://127.0.0.1:9/v1"), settings(), provider, auth_headers=bearer,
        platform_id=lambda: opted[0],
    )  # fmt: skip
    try:
        assert c.connect(UPDATE) is False
        assert provider.requests[-1].platform_id is True
        assert c.snapshot()["platform_id_sent"] is True  # sent, even though the allocator failed
        opted[0] = False
        assert c.connect(UPDATE) is False
        assert provider.requests[-1].platform_id is False
        assert c.snapshot()["platform_id_sent"] is False
    finally:
        c.close()


@pytest.mark.parametrize(
    ("base", "locality", "reason"),
    [
        ("https://alloc.example.org/session", "cloud", None),
        ("http://box.local:8000/session", "auto", None),  # local: plain http allowed
        ("https://alloc.example.org/session/", "cloud", "provider_url"),
        ("https://alloc.example.org/session?x=1", "cloud", "provider_url"),
        ("https://u@alloc.example.org/session", "cloud", "provider_url"),
        ("https://alloc.example.org:99999/session", "cloud", "provider_url"),
        ("http://alloc.example.org/session", "cloud", "provider_insecure"),
        ("https://localhost/session", "cloud", "provider_this_machine"),
    ],
)
def test_a_provider_endpoint_is_checked_before_anything_is_sent(base, locality, reason):
    r = route(base, locality=locality)
    if reason is None:
        assert sess.provider_endpoint(r, "/session") == base
    else:
        with pytest.raises(sess.ProviderRefused) as refused:
            sess.provider_endpoint(r, "/session")
        assert refused.value.reason == reason


def test_an_allocated_url_needs_the_realtime_path_and_is_read_as_ws():
    assert sess.allocated_url("https://h.example.org/v1/realtime?q=1") == (
        "wss://h.example.org/v1/realtime?q=1"
    )
    with pytest.raises(UrlRefused, match="path"):  # fix round 1, M7: no trailing slash
        sess.allocated_url("ws://h.local/realtime/")
    with pytest.raises(UrlRefused, match="path"):
        sess.allocated_url("wss://h.example.org/v1?x=/realtime")
    for bad in (None, "", 42):
        with pytest.raises(sess.ProviderRefused, match="provider_reply"):
            sess.allocated_url(bad)


def test_a_key_host_refusal_logs_the_host_never_the_query(caplog):
    """Fix round 1 of SR-4: the refused connect host is logged so it can be recorded and
    declared; its path and query never are (SPEC 8, 12)."""
    caplog.set_level(logging.DEBUG)
    provider = OtherHost(f"wss://rt.example.net/secret-path/realtime?t={QUERY_SECRET}")
    c = RealtimeClient(
        route("https://alloc.example.org/v1", locality="cloud", keyed=True),
        settings(),
        provider,
        auth_headers=bearer,
    )
    try:
        assert c.connect(UPDATE) is False and c.reason == "url_key_host"
    finally:
        c.close()
    assert "rt.example.net" in caplog.text
    for secret in (QUERY_SECRET, "secret-path", KEY):
        assert secret not in caplog.text


# ------------------------------------------------------------------ SR-4 fix round 1 (minors)


def test_a_declared_key_host_gets_the_key_on_the_wss_port_only():
    """M2: a declared host matches on 443 only, never on another port the allocator names."""
    hosts = (".example.net",)
    for url in ("wss://a.example.net/realtime", "wss://a.example.net:443/realtime"):
        assert target(url, keyed=True, key_hosts=hosts).send_key is True
    with pytest.raises(UrlRefused, match="key_host"):
        target("wss://a.example.net:22/realtime", keyed=True, key_hosts=hosts)


def test_a_percent_in_the_host_is_refused():
    """M3: parsers can disagree on an escaped host."""
    with pytest.raises(UrlRefused, match="host"):
        target("wss://rt%2eevil.com.example.net/realtime", key_hosts=(".example.net",))


@pytest.mark.parametrize(
    "base",
    [
        "https://xn--80ak6aa92e.com/session",
        "https://exämple.org/session",
        "https://a%2eb.org/session",
    ],
)
def test_a_provider_endpoint_needs_a_plain_ascii_host(base):
    """M4: the SR-1 rule against look-alike hosts also holds for a provider's endpoint."""
    with pytest.raises(sess.ProviderRefused, match="provider_url"):
        sess.provider_endpoint(route(base, locality="cloud"), "/session")


def test_with_pseudonyms_a_cloud_allocator_is_not_even_asked():
    """M6: the pseudonym gate runs before the allocation, so a cloud allocator gets neither
    the key nor the identifier while real names may not reach a cloud host."""
    provider = OtherHost("wss://rt.example.org/realtime")
    c = RealtimeClient(
        route("https://alloc.example.org/v1", locality="cloud", keyed=True),
        settings(),
        provider,
        auth_headers=bearer,
        local_only=lambda: True,
    )
    try:
        assert c.connect(UPDATE) is False and c.reason == "pseudonyms"
        assert provider.requests == []
    finally:
        c.close()


async def test_an_allocation_reply_over_the_cap_is_refused():
    """M10: a reply larger than ``max_reply_bytes`` (the route's ``max_message_bytes``) is not
    read whole: ``provider_reply``."""
    url = "https://alloc.example.org/session"
    big = httpx.Response(200, content=b"x" * 100)
    request, _ = recording_request("https://alloc.example.org", {url: big})
    capped = AllocationRequest(
        request.route, None, False, 1.0, request.http_client_factory, max_reply_bytes=10
    )
    with pytest.raises(sess.ProviderRefused, match="provider_reply"):
        await sess.post_allocation(capped, url, json={})
    small = AllocationRequest(
        request.route, None, False, 1.0, request.http_client_factory, max_reply_bytes=100
    )
    resp = await sess.post_allocation(small, url, json={})
    assert resp.status_code == 200 and resp.content == b"x" * 100


def test_the_client_caps_the_allocation_reply_at_its_message_size():
    provider = OtherHost("ws://localhost:9/realtime")
    c = RealtimeClient(
        route("http://127.0.0.1:9/v1"), settings(), provider, auth_headers=bearer,
        local_only=lambda: False,
    )  # fmt: skip
    try:
        c.connect(UPDATE)
        assert provider.requests[0].max_reply_bytes == settings().max_message_bytes
    finally:
        c.close()


def _capped(handler, cap: int = 64) -> AllocationRequest:
    return AllocationRequest(
        route("https://alloc.example.org"),
        None,
        False,
        1.0,
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_reply_bytes=cap,
    )


ALLOC_URL = "https://alloc.example.org/session"


async def test_a_compressed_allocation_reply_is_refused_and_identity_is_asked():
    """Security review of M10: a compressed body would be expanded before the cap counts it."""
    import gzip

    seen = []

    def handler(req):
        seen.append(req.headers.get("accept-encoding"))
        body = gzip.compress(b'{"connect_url": "' + b"x" * 10_000 + b'"}')
        return httpx.Response(200, content=body, headers={"Content-Encoding": "gzip"})

    with pytest.raises(sess.ProviderRefused, match="provider_reply"):
        await sess.post_allocation(_capped(handler, cap=100_000), ALLOC_URL, json={})
    assert seen == ["identity"]


async def test_a_content_length_over_the_cap_is_refused_before_the_body_is_read():
    read = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            read.append(True)
            yield b"x" * 10

    def handler(req):
        return httpx.Response(200, headers={"Content-Length": "1000"}, stream=Body())

    with pytest.raises(sess.ProviderRefused, match="provider_reply"):
        await sess.post_allocation(_capped(handler, cap=64), ALLOC_URL, json={})
    assert read == []


async def test_an_identity_reply_under_the_cap_parses_once():
    def handler(req):
        return httpx.Response(
            200, json={"connect_url": "wss://h.example.org/realtime"},
            headers={"Content-Encoding": "identity"},
        )  # fmt: skip

    resp = await sess.post_allocation(_capped(handler), ALLOC_URL, json={})
    assert resp.json() == {"connect_url": "wss://h.example.org/realtime"}
    assert "content-encoding" not in resp.headers


async def test_a_streamed_reply_stops_being_read_at_the_cap():
    sent = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                sent.append(40)
                yield b"x" * 40

    def handler(req):
        return httpx.Response(200, stream=Body())  # no Content-Length

    with pytest.raises(sess.ProviderRefused, match="provider_reply"):
        await sess.post_allocation(_capped(handler, cap=64), ALLOC_URL, json={})
    assert len(sent) == 2  # stopped once past the cap, not read whole
