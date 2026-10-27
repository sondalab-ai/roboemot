"""The realtime route at runtime outside a turn: the network aggregate, what blocks it, the
session provider choice, the config checks, the instructions, sleep and the report (SPEC 8,
Realtime speech route; SPEC 12A, Links and Diagnostics)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from roboemot.config import ConfigStack
from roboemot.config.schema import LinkPolicy, LLMRoute, RegisterParams, RoboEmotConfig
from roboemot.diagnostics import Diagnostics, FnSource
from roboemot.health.network import NetworkMonitor
from roboemot.llm.prompt import realtime_instructions, system_prompt
from roboemot.realtime import route as route_module
from roboemot.realtime.route import RealtimeRoute, select_provider
from roboemot.realtime.session import DirectSession
from roboemot.runtime.sleep import SleepGate

POLICY = LinkPolicy(failures_to_offline=1, successes_to_online=1)
CHAT = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m"}
RT = {"name": "rt", "base_url": "http://s2s.local:8765/v1", "model": "s2s", "kind": "realtime"}
CLOUD_RT = {**RT, "base_url": "https://rt.example.com/v1", "locality": "cloud"}


def monitor(*routes):
    return NetworkMonitor([LLMRoute(**r) for r in routes], POLICY, probe=lambda *a: True)


# ------------------------------------------------------------------ network aggregate


def test_a_live_realtime_route_never_masks_dead_chat_routes_in_local_mode():
    mon = monitor(CHAT, RT)
    mon.report("rt", True, 0.0)
    mon.report("lan", False, 0.0)
    assert not mon.online  # the realtime route's own link does not count


def test_in_realtime_mode_the_realtime_link_keeps_the_network_online():
    mon = monitor(CHAT, RT)
    mon.set_realtime(True)
    mon.report("lan", False, 0.0)
    assert mon.online  # the realtime session can answer
    mon.report_realtime(False, 1.0)
    assert not mon.online  # nothing can answer a turn now
    mon.report_realtime(True, 2.0)
    assert mon.online
    mon.set_realtime(False)
    assert not mon.online
    snap = mon.snapshot()
    assert snap["realtime"]["counted"] is False and snap["realtime"]["online"] is True


def test_only_realtime_routes_and_local_mode_count_as_no_route_at_all():
    mon = monitor(RT)
    mon.report("rt", False, 0.0)
    assert mon.online  # no route that can answer: the aggregate keeps its state


# ------------------------------------------------------------------ blocks


def rt_stack(rt=None, routes=(CHAT, RT), **app):
    cfg = {
        "llm_routes": list(routes),
        "voice": {"realtime_route": {"enabled": True, "route": "rt", **(rt or {})}},
        **app,
    }
    return ConfigStack(app=cfg)


class FakeRouter:
    def __init__(self, ok=True, used=0):
        self.ok, self._used = ok, used

    def realtime_route(self, name):
        return object() if self.ok else None

    def used(self, name):
        return self._used

    def auth_headers(self, route):
        return {}

    def charge(self, name, tokens):
        self._used += tokens


DIRECT = DirectSession()


def make_route(stack, router=None, provider=DIRECT, extra=True):
    return RealtimeRoute(
        stack,
        lambda: router or FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        provider,
        extra=lambda: extra,
    )


@pytest.mark.parametrize(
    ("stack", "kw", "reason"),
    [
        (lambda: rt_stack({"enabled": False}), {}, "off"),
        (lambda: rt_stack({"route": ""}), {}, "no_route"),
        (lambda: rt_stack({"input": "audio"}), {}, "no_opt_in"),
        (lambda: rt_stack(), {"extra": False}, "no_extra"),
        (lambda: rt_stack(), {"provider": None}, "provider"),
        (
            lambda: rt_stack(routes=(CHAT, CLOUD_RT), privacy={"pseudonyms_to_cloud": True}),
            {},
            "pseudonyms",
        ),
        (lambda: rt_stack(), {"router": FakeRouter(ok=False, used=10**9)}, "budget"),
        (lambda: rt_stack(), {"router": FakeRouter(ok=False)}, "unavailable"),
    ],
)
def test_what_blocks_the_route(stack, kw, reason):
    s = stack()
    r = make_route(s, **kw)
    assert r.blocked(s.effective()) == reason
    assert r.ready() is False  # nothing is connected
    assert r.snapshot()["reason"] == reason
    assert r.network.snapshot()["realtime"]["counted"] is False


def test_pseudonyms_do_not_block_a_local_route():
    s = rt_stack(privacy={"pseudonyms_to_cloud": True})
    assert make_route(s).blocked(s.effective()) is None


# ------------------------------------------------------------------ providers


class Provider:
    key_hosts = (".allocated.example",)

    async def allocate(self, request):
        raise AssertionError("not called here")


def test_direct_mode_ignores_a_registered_provider(monkeypatch):
    monkeypatch.setattr(route_module, "load_session_provider", lambda robot: lambda m: Provider())
    cfg = rt_stack().effective()
    assert isinstance(select_provider(cfg, "robot", None), DirectSession)


def test_platform_mode_needs_the_robots_provider(monkeypatch):
    cfg = rt_stack({"mode": "platform"}).effective()
    monkeypatch.setattr(route_module, "load_session_provider", lambda robot: None)
    assert select_provider(cfg, "robot", None) is None  # never direct mode in its place
    monkeypatch.setattr(route_module, "load_session_provider", lambda robot: lambda m: Provider())
    provider = select_provider(cfg, "robot", None)
    assert isinstance(provider, Provider)
    route = make_route(rt_stack({"mode": "platform"}), provider=provider)
    assert route.key_hosts == [".allocated.example"]

    def broken(manifest):
        raise RuntimeError("boom")

    monkeypatch.setattr(route_module, "load_session_provider", lambda robot: broken)
    assert select_provider(cfg, "robot", None) is None


# ------------------------------------------------------------------ config checks


def test_the_route_must_be_a_realtime_route():
    with pytest.raises(ValidationError, match="kind: realtime"):
        RoboEmotConfig(
            llm_routes=[CHAT], voice={"realtime_route": {"enabled": True, "route": "lan"}}
        )
    with pytest.raises(ValidationError, match="kind: realtime"):
        RoboEmotConfig(llm_routes=[CHAT], voice={"realtime_route": {"route": "missing"}})


def test_the_speech_backend_refuses_a_realtime_route():
    with pytest.raises(ValidationError, match="openai_speech.route"):
        RoboEmotConfig(llm_routes=[CHAT, RT], voice={"openai_speech": {"route": "rt"}})
    RoboEmotConfig(llm_routes=[CHAT, RT], voice={"openai_speech": {"route": "lan"}})


def test_the_new_fields_and_their_defaults():
    rt = RoboEmotConfig().voice.realtime_route
    assert (rt.enabled, rt.route, rt.mode, rt.input, rt.voice) == (False, "", "direct", "text", "")
    assert rt.prebuffer_ms == 200.0 and rt.mood_instructions is True
    with pytest.raises(ValidationError):
        RoboEmotConfig(voice={"realtime_route": {"prebuffer_ms": 2001}})


# ------------------------------------------------------------------ instructions


def test_the_realtime_instructions_drop_the_mood_tag_rule():
    cfg = RoboEmotConfig(voice={"mood_source": "llm_tagged"})
    assert "[mood=" in system_prompt(cfg, RegisterParams())
    text = realtime_instructions(cfg, RegisterParams())
    assert "[mood=" not in text
    assert "Reply in at most" in text and "never claim to be human" in text


# ------------------------------------------------------------------ sleep


def test_sleep_closes_the_server_session_and_wake_reopens_it():
    calls = []
    realtime = SimpleNamespace(
        close_soon=lambda reason: calls.append(("close", reason)),
        start=lambda: calls.append(("start",)),
    )
    loop = SimpleNamespace(motors_off=True, link=SimpleNamespace(online=True), clear=lambda s: None)
    gate = SleepGate(
        SimpleNamespace(conversation=None, speech_output=None),
        SimpleNamespace(publish=lambda *a, **k: None),
        loop,
        SimpleNamespace(service=None),
        SimpleNamespace(pause=lambda: None, resume=lambda: None, paused=False),
        None,
        SimpleNamespace(release=lambda: None),
        None,
        lambda: True,
        lambda: realtime,
    )
    gate.tick(0.0)
    loop.motors_off = False
    gate.tick(1.0)
    assert calls == [("close", "asleep"), ("start",)]


# ------------------------------------------------------------------ report


def test_the_report_section_holds_numbers_and_states_only():
    s = rt_stack()
    r = make_route(s)
    report = Diagnostics().add(FnSource("roboemot.realtime_route", r.snapshot)).report()
    section = report["sections"]["roboemot.realtime_route"]
    body = section.get("data", section)
    assert body["state"] == "off" or body["state"] in route_module.STATES
    for key in ("mode", "input", "audio_egress", "platform_id_sent", "opened", "reconnects",
                "fallbacks_start", "fallbacks_mid_turn", "unrequested", "tool_calls",
                "items_deleted", "provider_hosts", "latency_ms", "upload_s_today"):  # fmt: skip
        assert key in body, key
    assert body["audio_egress"] is False
    assert "***" not in json.dumps(body["provider_hosts"])


# ------------------------------------------------------------------ client close hook


def test_the_client_reports_a_lost_session_and_its_own_close():
    from fake_server import FakeServer
    from realtime_helpers import UPDATE, route, settings, wait_until
    from roboemot.realtime import protocol
    from roboemot.realtime.client import RealtimeClient

    closes = []
    with FakeServer("close_mid") as server:
        client = RealtimeClient(
            route(server.http_base()),
            settings(),
            DirectSession(),
            auth_headers=dict,
            on_close=closes.append,
        )
        try:
            assert client.connect(UPDATE)
            client.send(protocol.response_create())
            assert wait_until(lambda: closes == ["closed_by_server"])
        finally:
            client.close()
    assert closes == ["closed_by_server"]  # a close after a failure is not reported twice


# ------------------------------------------------------------------ hot reload (SPEC 12)


class StubClient:
    """A connected client on a host that is not local (a provider's cloud host)."""

    def __init__(self, route, settings, provider, **kw):
        self.route, self.state, self.closed = route, "idle", 0

    def connect(self, update):
        self.state = "connected"
        return True

    def snapshot(self):
        return {"local": False, "endpoint": "wss://rt.example.com:443"}

    def send(self, event):
        return None

    def close(self):
        self.closed += 1
        self.state = "closed"


def stub_route(stack):
    made = []

    def factory(*a, **kw):
        made.append(StubClient(*a, **kw))
        return made[-1]

    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
        client_factory=factory,
    )
    return r, made


def test_turning_pseudonyms_on_closes_a_cloud_session():
    from rt_rig import wait_for

    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready() and r.connected
    stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)
    r.apply_config(stack.effective())
    assert not r.connected  # detached at once: no later turn uses it
    assert wait_for(lambda: made[0].closed == 1)
    assert r.snapshot()["state"] == "closed" and r.snapshot()["reason"] == "reconfigured"


def test_changing_the_route_closes_the_session():
    from rt_rig import wait_for

    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    routes = [CHAT, {**RT, "base_url": "http://other.local:8765/v1"}]
    stack.set("llm_routes", routes, source="test", admin=True)
    r.apply_config(stack.effective())
    assert not r.connected
    assert wait_for(lambda: made[0].closed == 1)


def test_an_unchanged_config_keeps_the_session():
    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    r.apply_config(stack.effective())
    assert r.connected and made[0].closed == 0
