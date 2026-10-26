"""URL and key rules of the realtime route, in direct and provider mode (SPEC 8, 12)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

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
        "wss://node7.rt.example.net:8443/realtime?session=s",
        keyed=True,
        key_hosts=(".rt.example.net",),
    )
    assert alloc.send_key and alloc.origin == "wss://node7.rt.example.net:8443"
    keyless = target("wss://other.example.net/realtime")
    assert keyless.send_key is False


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
