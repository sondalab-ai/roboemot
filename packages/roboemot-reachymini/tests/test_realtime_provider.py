"""The Reachy Mini realtime session provider: a session from the platform's allocator
(SPEC 4A, Session providers; SPEC 8, Realtime speech route; SPEC 12, Data; SPEC 12A).

The allocator and the daemon are ``httpx.MockTransport`` fakes: no network. A connect that the
URL rules accept would open a socket, so every client test here ends before the socket opens
(a refusal or an allocator error); the accepted allocations are checked with ``connect_target``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import entry_points

import httpx
import pytest
from roboemot.config import ConfigStack
from roboemot.config.schema import LinkPolicy, LLMRoute, RealtimeRouteConfig
from roboemot.diagnostics import Diagnostics, FnSource, Target
from roboemot.health.network import NetworkMonitor
from roboemot.plugins import contract_violation
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.route import RealtimeRoute
from roboemot.realtime.session import (
    ENTRY_POINT_GROUP,
    Allocation,
    AllocationRequest,
    ProviderRefused,
    SessionProvider,
    UrlRefused,
    connect_target,
)
from roboemot_reachymini import realtime as rm_realtime
from roboemot_reachymini.daemon import DaemonClient
from roboemot_reachymini.diagnostics import sources as reachy_sources
from roboemot_reachymini.realtime import KEY_HOSTS, PlatformSession

KEY = "hf_SENTINELKEY0123456789"
HW_ID = "c0ffee5e171e1d42"  # the daemon's hashed id has this shape (16 hex)
QUERY_SECRET = "QSENTINEL42"
ALLOCATOR = "https://allocator.hf.space/session"
CLOUD_RT = f"wss://rt-7.hf.space/v1/realtime?session_token={QUERY_SECRET}"
SENTINELS = (KEY, HW_ID, QUERY_SECRET)
UPDATE = protocol.session_update(
    instructions="be kind",
    input_rate_hz=16000,
    output_rate_hz=None,
    voice="v",
    vad_threshold=0.5,
    silence_ms=500,
    create_response=False,
    interrupt_response=False,
)
PLATFORM_MODE = {"realtime_route": {"enabled": True, "route": "platform", "mode": "platform"}}
POLICY = LinkPolicy(failures_to_offline=1, successes_to_online=1)


def route(base_url: str = ALLOCATOR, *, keyed: bool = True, locality: str = "auto") -> LLMRoute:
    return LLMRoute(
        name="platform",
        base_url=base_url,
        model="s2s",
        kind="realtime",
        locality=locality,
        api_key="${secret:hf_token}" if keyed else "none",
    )


def settings() -> RealtimeRouteConfig:
    """Below the schema's floors, so the tests run fast."""
    return RealtimeRouteConfig.model_construct(
        connect_timeout_s=2.0, send_timeout_s=0.5, max_message_bytes=65_536
    )


def bearer() -> dict[str, str]:
    return {"Authorization": f"Bearer {KEY}"}


class Allocator:
    """The allocator fake: records every request, answers ``reply`` (a response, or a
    callable of the request)."""

    def __init__(self, reply: httpx.Response | None = None) -> None:
        self.reply = reply or httpx.Response(200, json={"connect_url": CLOUD_RT, "session_id": "s"})
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.reply

    def factory(self) -> httpx.AsyncClient:
        # follow_redirects on: only post_allocation's per-request override keeps a redirect
        # from being followed (fix round 1, M1)
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def bodies(self) -> list[object]:
        return [json.loads(r.content) for r in self.requests]


class Daemon:
    """The daemon fake: ``GET /api/daemon/hardware-id``."""

    def __init__(self, answer: httpx.Response | None = None) -> None:
        self.answer = answer or httpx.Response(200, json={"hardware_id": HW_ID})
        self.requests: list[httpx.Request] = []
        self.timeouts: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/daemon/status":  # the debug report's robot.daemon
            return httpx.Response(200, json={"version": "1.12.0", "hardware_id": HW_ID})
        if request.url.path != "/api/daemon/hardware-id":
            return httpx.Response(200, json={})
        return self.answer

    def client(self, timeout_s: float) -> DaemonClient:
        self.timeouts.append(timeout_s)
        return DaemonClient(timeout_s=timeout_s, transport=httpx.MockTransport(self.handler))


def provider(daemon: Daemon | None = None) -> PlatformSession:
    return PlatformSession(daemon_client=(daemon or Daemon()).client)


def allocate(
    allocator: Allocator,
    daemon: Daemon | None = None,
    *,
    r: LLMRoute | None = None,
    key: str | None = KEY,
    platform_id: bool = False,
    noted: list[bool] | None = None,
    timeout_s: float = 2.0,
) -> Allocation:
    notes = noted if noted is not None else []
    request = AllocationRequest(
        r or route(), key, platform_id, timeout_s, allocator.factory, lambda: notes.append(True)
    )
    return asyncio.run(provider(daemon).allocate(request))


def client(allocator: Allocator, daemon: Daemon | None = None, **kw) -> RealtimeClient:
    r = kw.pop("route", None) or route()
    return RealtimeClient(
        r,
        settings(),
        provider(daemon),
        auth_headers=bearer,
        http_client_factory=allocator.factory,
        **kw,
    )


def connect_fails(c: RealtimeClient) -> str | None:
    try:
        assert c.connect(UPDATE) is False
        return c.reason
    finally:
        c.close()


# ------------------------------------------------------------------ the entry point


def test_the_provider_is_registered_under_the_robot_id_and_meets_the_contract():
    eps = entry_points(group=ENTRY_POINT_GROUP)
    (found,) = [e for e in eps if e.name == "reachy-mini-wireless"]
    factory = found.load()
    assert contract_violation(ENTRY_POINT_GROUP, factory) is None
    built = factory(object())
    assert isinstance(built, PlatformSession) and isinstance(built, SessionProvider)
    # the allocator's own origin plus the connect host recorded on the robot, exactly (SPEC 13)
    assert built.key_hosts == KEY_HOSTS == ("llgpsjdk3p4ayeud.endpoints.huggingface.cloud",)


# ------------------------------------------------------------------ the identifier (opt-in)


def test_the_identifier_is_absent_unless_opted_in_and_the_daemon_is_not_asked():
    allocator, daemon, noted = Allocator(), Daemon(), []
    allocate(allocator, daemon, noted=noted)
    assert allocator.bodies() == [{}]  # an empty object, like the platform's own app
    assert daemon.requests == [] and noted == []


def test_with_the_opt_in_the_identifier_goes_in_the_body():
    allocator, daemon, noted = Allocator(), Daemon(), []
    allocate(allocator, daemon, platform_id=True, noted=noted)
    assert allocator.bodies() == [{"hardware_id": HW_ID}]
    assert [r.url.path for r in daemon.requests] == ["/api/daemon/hardware-id"]
    assert noted == [True]
    assert 0 < daemon.timeouts[0] <= 2.0  # bounded by the connect's time left


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"hardware_id": None}),  # no robot attached (a dev machine)
        httpx.Response(200, json={"hardware_id": "not an id!"}),  # not the expected shape
        httpx.Response(200, json={"hardware_id": "a" * 65}),  # too long
        httpx.Response(200, json={"hardware_id": 42}),
        httpx.Response(200, json=["x"]),
        httpx.Response(404, json={"detail": "Not Found"}),  # an older daemon
        httpx.Response(500, json={"detail": f"boom {HW_ID}"}),
    ],
)
def test_without_a_usable_identifier_nothing_is_sent_in_its_place(answer, caplog):
    caplog.set_level(logging.DEBUG)
    allocator, noted = Allocator(), []
    allocate(allocator, Daemon(answer), platform_id=True, noted=noted)
    assert allocator.bodies() == [{}] and noted == []
    assert HW_ID not in caplog.text


def test_the_identifier_and_the_key_are_never_logged_or_reported(caplog):
    """The identifier goes to the allocator, which then fails; the key is used. Neither, nor the
    query of an allocated URL, shows up in any log, the client's snapshot, the route's report
    or a repr (SPEC 12, 12A)."""
    caplog.set_level(logging.DEBUG)
    for reply in (
        httpx.Response(503, text=f"overloaded {HW_ID} {KEY}"),
        httpx.Response(200, json={"connect_url": "wss://evil.example.com/realtime?t=" + KEY}),
    ):
        allocator = Allocator(reply)
        stack = ConfigStack(
            app={
                "llm_routes": [route().model_dump(mode="json")],
                "voice": PLATFORM_MODE,
                "privacy": {"realtime_platform_id": True},
            }
        )
        made: list[RealtimeClient] = []

        def factory(*a, made=made, allocator=allocator, **kw):
            made.append(RealtimeClient(*a, http_client_factory=allocator.factory, **kw))
            return made[-1]

        rt = RealtimeRoute(
            stack,
            lambda: Router(),
            NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
            provider(),
            extra=lambda: True,
            client_factory=factory,
        )
        try:
            assert rt.ready() is False
            assert allocator.bodies() == [{"hardware_id": HW_ID}]
            assert allocator.requests[0].headers[rm_realtime.AUTH_HEADER] == f"Bearer {KEY}"
            diag = Diagnostics().add(FnSource("roboemot.realtime_route", rt.snapshot))
            for source in reachy_sources(Target("robot"), Daemon().client(2.0)):
                diag.add(source)  # a real Reachy report: robot.daemon carries the id (I2)
            report = diag.report()
            texts = (caplog.text, json.dumps(report), repr(made[0].snapshot()), repr(rt))
            assert made[0].snapshot()["platform_id_sent"] is True
            section = report["sections"]["roboemot.realtime_route"]
            assert section.get("data", section)["platform_id_sent"] is True
            for text in texts:
                for secret in SENTINELS:
                    assert secret not in text
        finally:
            rt.close()


class Router:
    def realtime_route(self, name):
        return object()

    def used(self, name):
        return 0

    def auth_headers(self, route):
        return bearer()

    def charge(self, name, tokens):
        pass


# ------------------------------------------------------------------ where the key goes


def test_the_key_goes_to_the_allocator_in_its_own_header_once():
    allocator = Allocator()
    allocate(allocator)
    (sent,) = allocator.requests
    assert str(sent.url) == ALLOCATOR and sent.method == "POST"
    assert sent.headers[rm_realtime.AUTH_HEADER] == f"Bearer {KEY}"
    assert "authorization" not in sent.headers  # only the platform's header carries it


def test_a_route_without_a_key_sends_no_key_header():
    allocator = Allocator()
    allocate(allocator, r=route(keyed=False), key=None)
    assert rm_realtime.AUTH_HEADER.lower() not in allocator.requests[0].headers


@pytest.mark.parametrize(
    ("connect_url", "reason"),
    [
        ("wss://allocator.hf.space/v1/realtime?t=1", None),  # the allocator's own origin
        ("wss://allocator.hf.space:8443/realtime", "key_host"),  # another port
        (CLOUD_RT, "key_host"),  # another Space: anyone can create one (fix round 1, I1)
        ("wss://hf.space/realtime", "key_host"),
        ("wss://rt.hf.space.evil.example/realtime", "key_host"),
        ("wss://evil.example.com/realtime", "key_host"),
        ("ws://allocator.hf.space/realtime", "insecure"),  # never over plain ws
    ],
)
def test_the_key_reaches_only_the_allocators_own_origin_over_wss(connect_url, reason):
    got = allocate(Allocator(httpx.Response(200, json={"connect_url": connect_url})))
    assert got.key_hosts == KEY_HOSTS
    if reason is None:
        target = connect_target(route(), got, KEY_HOSTS)
        assert target.send_key is True and target.origin == "wss://allocator.hf.space:443"
    else:
        with pytest.raises(UrlRefused) as refused:
            connect_target(route(), got, KEY_HOSTS)
        assert refused.value.reason == reason


RECORDED = "llgpsjdk3p4ayeud.endpoints.huggingface.cloud"


@pytest.mark.parametrize(
    ("connect_url", "reason"),
    [
        (f"wss://{RECORDED}/v1/realtime?t=1", None),
        (f"wss://{RECORDED}:443/realtime", None),
        (f"wss://{RECORDED.upper()}/realtime", None),
        (f"wss://{RECORDED}:8443/realtime", "key_host"),  # only the wss port
        (f"ws://{RECORDED}/realtime", "insecure"),
        ("wss://other.endpoints.huggingface.cloud/realtime", "key_host"),  # no suffix match
        (f"wss://x.{RECORDED}/realtime", "key_host"),
        (f"wss://{RECORDED}.evil.example/realtime", "key_host"),
    ],
)
def test_the_recorded_connect_host_gets_the_key_exactly(connect_url, reason):
    """SPEC 8, 13: the connect host recorded on the robot is declared exactly; a look-alike,
    another endpoint or another port gets no key."""
    got = allocate(Allocator(httpx.Response(200, json={"connect_url": connect_url})))
    if reason is None:
        target = connect_target(route(), got, KEY_HOSTS)
        assert target.send_key is True and target.origin == f"wss://{RECORDED}:443"
    else:
        with pytest.raises(UrlRefused) as refused:
            connect_target(route(), got, KEY_HOSTS)
        assert refused.value.reason == reason


def test_another_space_gets_no_key_and_the_route_falls_back(caplog):
    """Fix round 1, I1: an allocated host other than the allocator's origin never gets the
    token; the local pipeline answers, and the refused host (never its query) is logged so the
    real connect host can be recorded on the robot (SPEC 8, 12, 13)."""
    caplog.set_level(logging.DEBUG)
    allocator = Allocator()  # answers CLOUD_RT, on rt-7.hf.space
    stack = ConfigStack(
        app={"llm_routes": [route().model_dump(mode="json")], "voice": PLATFORM_MODE}
    )
    rt = RealtimeRoute(
        stack,
        lambda: Router(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        provider(),
        extra=lambda: True,
        client_factory=lambda *a, **kw: RealtimeClient(
            *a, http_client_factory=allocator.factory, **kw
        ),
    )
    try:
        assert rt.ready() is False
        snap = rt.snapshot()
        assert snap["state"] == "fallback_local" and snap["reason"] == "url_key_host"
        assert snap["client"]["key_sent"] is False
    finally:
        rt.close()
    assert "rt-7.hf.space" in caplog.text
    assert QUERY_SECRET not in caplog.text and KEY not in caplog.text


def test_a_keyed_route_sends_nothing_to_an_undeclared_host(caplog):
    caplog.set_level(logging.DEBUG)
    reply = httpx.Response(200, json={"connect_url": "wss://evil.example.com/realtime"})
    allocator = Allocator(reply)
    assert connect_fails(client(allocator)) == "url_key_host"
    assert len(allocator.requests) == 1
    assert KEY not in caplog.text


def test_an_https_connect_url_is_read_as_wss():
    url = "https://allocator.hf.space/v1/realtime?x=1"
    got = allocate(Allocator(httpx.Response(200, json={"connect_url": url})))
    target = connect_target(route(), got, KEY_HOSTS)
    assert target.url == "wss://allocator.hf.space/v1/realtime?x=1" and target.send_key


# ------------------------------------------------------------------ allocator answers


def test_a_redirect_is_refused_and_never_followed():
    moved = httpx.Response(307, headers={"Location": "https://evil.example.com/session"})
    allocator = Allocator(moved)
    assert connect_fails(client(allocator)) == "provider_http_307"
    assert [str(r.url) for r in allocator.requests] == [ALLOCATOR]


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_an_allocator_error_is_a_provider_http_reason(status):
    allocator = Allocator(httpx.Response(status, json={"error": {"message": KEY}}))
    with pytest.raises(ProviderRefused) as refused:
        allocate(allocator)
    assert refused.value.reason == f"provider_http_{status}"
    assert KEY not in str(refused.value)


def test_an_allocator_error_falls_back_to_the_local_pipeline_with_its_reason():
    allocator = Allocator(httpx.Response(503))
    stack = ConfigStack(
        app={
            "llm_routes": [route().model_dump(mode="json")],
            "voice": PLATFORM_MODE,
        }
    )
    rt = RealtimeRoute(
        stack,
        lambda: Router(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        provider(),
        extra=lambda: True,
        client_factory=lambda *a, **kw: RealtimeClient(
            *a, http_client_factory=allocator.factory, **kw
        ),
    )
    try:
        assert rt.ready() is False  # the turn goes the local way
        snap = rt.snapshot()
        assert snap["state"] == "fallback_local" and snap["reason"] == "provider_http_503"
        assert snap["platform_id_sent"] is False  # off by default
    finally:
        rt.close()


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=["wss://rt-7.hf.space/realtime"]),
        httpx.Response(200, json={"session_id": "s"}),
        httpx.Response(200, json={"connect_url": ""}),
        httpx.Response(200, json={"connect_url": 42}),
    ],
)
def test_an_unusable_reply_is_refused(reply):
    with pytest.raises(ProviderRefused) as refused:
        allocate(Allocator(reply))
    assert refused.value.reason == "provider_reply"


def test_an_unreachable_allocator_is_its_own_reason():
    class Down(Allocator):
        def handler(self, request):
            self.requests.append(request)
            raise httpx.ConnectError(f"cannot reach {ALLOCATOR}?k={KEY}")

    with pytest.raises(ProviderRefused) as refused:
        allocate(Down())
    assert refused.value.reason == "provider_unreachable"
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


# ------------------------------------------------------------------ the connect URL


@pytest.mark.parametrize(
    ("connect_url", "reason"),
    [
        ("wss://rt-7.hf.space/v1/chat", "url_path"),  # not /realtime
        ("wss://rt-7.hf.space/realtime/x", "url_path"),
        ("wss://allocator.hf.space/realtime/", "url_path"),  # exactly /realtime (M7)
        ("wss://rt-7.hf.space/v1?next=/realtime", "url_path"),  # the query does not count
        ("ws://rt-7.hf.space/realtime", "url_insecure"),  # ws to a cloud host
        ("wss://user:pw@rt-7.hf.space/realtime", "url_userinfo"),
        ("wss://127.0.0.1/realtime", "url_this_machine"),  # loopback
        ("wss://localhost:8765/v1/realtime", "url_this_machine"),
        ("ftp://rt-7.hf.space/realtime", "url_scheme"),
    ],
)
def test_a_bad_connect_url_is_refused_before_anything_is_sent(connect_url, reason):
    allocator = Allocator(httpx.Response(200, json={"connect_url": connect_url}))
    assert connect_fails(client(allocator)) == reason
    assert len(allocator.requests) == 1


# ------------------------------------------------------------------ the route's base URL


@pytest.mark.parametrize(
    ("base_url", "locality", "reason"),
    [
        ("https://allocator.hf.space/", "auto", "provider_url"),  # path not /session
        ("https://allocator.hf.space/v1/session", "auto", "provider_url"),
        ("https://allocator.hf.space/session/", "auto", "provider_url"),
        ("https://allocator.hf.space/session?k=1", "auto", "provider_url"),
        ("http://allocator.hf.space/session", "auto", "provider_insecure"),  # cloud over http
        ("http://localhost:8000/session", "cloud", "provider_insecure"),
        ("https://127.0.0.1:8443/session", "cloud", "provider_this_machine"),
    ],
)
def test_a_route_that_is_not_an_allocator_is_refused_and_nothing_is_sent(
    base_url, locality, reason
):
    allocator = Allocator()
    with pytest.raises(ProviderRefused) as refused:
        allocate(allocator, r=route(base_url, locality=locality))
    assert refused.value.reason == reason and allocator.requests == []


# ------------------------------------------------------------------ locality on both hosts


def test_a_local_allocator_may_hand_out_a_local_ws_host_but_not_a_cloud_one():
    lan = route("http://192.168.1.20:8000/session", keyed=False)
    local = allocate(
        Allocator(httpx.Response(200, json={"connect_url": "ws://192.168.1.21:8765/v1/realtime"})),
        r=lan,
        key=None,
    )
    assert connect_target(lan, local, KEY_HOSTS).local is True
    cloud_ws = allocate(
        Allocator(httpx.Response(200, json={"connect_url": "ws://rt-7.hf.space/realtime"})),
        r=lan,
        key=None,
    )
    with pytest.raises(UrlRefused, match="insecure"):
        connect_target(lan, cloud_ws, KEY_HOSTS)
    cloud = allocate(Allocator(), r=lan, key=None)
    assert connect_target(lan, cloud, KEY_HOSTS).local is False  # both must be local


def test_with_pseudonyms_a_cloud_host_from_a_local_allocator_is_refused():
    lan = route("http://192.168.1.20:8000/session", keyed=False)
    allocator = Allocator()
    c = client(allocator, route=lan, local_only=lambda: True)
    assert connect_fails(c) == "pseudonyms"


def test_with_pseudonyms_a_cloud_allocator_blocks_the_route():
    stack = ConfigStack(
        app={
            "llm_routes": [route().model_dump(mode="json")],
            "voice": PLATFORM_MODE,
            "privacy": {"pseudonyms_to_cloud": True},
        }
    )
    rt = RealtimeRoute(
        stack,
        lambda: Router(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        provider(),
        extra=lambda: True,
    )
    assert rt.blocked(stack.effective()) == "pseudonyms"


def test_turning_the_opt_in_off_keeps_the_identifier_out_of_the_next_allocation():
    """Security review of SR-4: the client reads the opt-in from the committed config at each
    allocation, so a reconnect after the admin turned it off sends ``{}`` (SPEC 12, Data)."""
    stack = ConfigStack(app={"privacy": {"realtime_platform_id": True}})
    allocator, daemon = Allocator(httpx.Response(503)), Daemon()
    c = client(
        allocator, daemon, platform_id=lambda: stack.effective().privacy.realtime_platform_id
    )
    try:
        assert c.connect(UPDATE) is False and c.snapshot()["platform_id_sent"] is True
        stack.set("privacy.realtime_platform_id", False, source="test", admin=True)
        assert c.connect(UPDATE) is False
        assert allocator.bodies() == [{"hardware_id": HW_ID}, {}]
        assert len(daemon.requests) == 1  # the daemon is not asked again
        assert c.snapshot()["platform_id_sent"] is False
    finally:
        c.close()


def test_with_no_time_left_the_daemon_is_not_asked():
    """Fix round 1, M9: a daemon call that could not finish before the connect deadline is not
    started (its thread would outlive the connect)."""
    allocator, daemon, noted = Allocator(), Daemon(), []
    allocate(allocator, daemon, platform_id=True, noted=noted, timeout_s=0.0)
    assert daemon.requests == [] and allocator.bodies() == [{}] and noted == []
