"""The realtime route at runtime outside a turn: the network aggregate, what blocks it, the
session provider choice, the config checks, the instructions, sleep and the report (SPEC 8,
Realtime speech route; SPEC 12A, Links and Diagnostics)."""

from __future__ import annotations

import asyncio
import json
import threading
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
    mon.report_realtime(True, 0.0)  # a session opened: the link is confirmed
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
        self.route, self.state, self.closed, self.reason = route, "idle", 0, None
        self.local_only = kw.get("local_only", lambda: False)
        self.may_send = kw.get("may_send", lambda: True)
        self.on_close = kw.get("on_close", lambda reason: None)
        self.on_event = kw.get("on_event", lambda event: None)
        self.updates_sent = 0
        self.started = False

    def connect(self, update):
        self.started = True
        if self.local_only():  # the real client refuses a cloud host here
            self.state, self.reason = "closed", "pseudonyms"
            return False
        if not self.may_send():  # and asks its owner right before session.update
            self.state, self.reason = "closed", "reconfigured"
            return False
        self.updates_sent += 1
        self.state = "connected"
        return True

    def snapshot(self):
        return {"local": False, "endpoint": "wss://rt.example.com:443"}

    def send(self, event):
        return None

    def close(self):
        """Like the real client: ``on_close("closed")`` once, for a client that was started
        and is not closed already."""
        self.closed += 1
        fire = self.started and self.state != "closed"
        self.state = "closed"
        if fire:
            self.on_close("closed")


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


# ------------------------------------------------------------------ fix round 1: B2


class SlowClient(StubClient):
    """A connect that waits for the test (a connect in flight)."""

    gate: threading.Event

    def connect(self, update):
        SlowClient.entered.set()
        assert SlowClient.gate.wait(5)
        return super().connect(update)


def slow_route(stack):
    SlowClient.gate, SlowClient.entered = threading.Event(), threading.Event()
    made = []

    def factory(*a, **kw):
        made.append(SlowClient(*a, **kw))
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


def connect_while(stack, change):
    """Start a connect, run ``change`` while it is in flight, let it finish."""
    r, made = slow_route(stack)
    result = []
    thread = threading.Thread(target=lambda: result.append(r.ready()))
    thread.start()
    assert SlowClient.entered.wait(5)
    change(r)
    SlowClient.gate.set()
    thread.join(5)
    return r, made, result[0]


def test_pseudonyms_switched_on_during_a_connect_close_the_new_session():
    stack = rt_stack()

    def change(r):
        stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)
        r.apply_config(stack.effective())

    r, made, ok = connect_while(stack, change)
    assert ok is False and not r.connected and made[0].closed == 1


def test_a_route_edit_during_a_connect_closes_the_new_session():
    stack = rt_stack()

    def change(r):
        routes = [CHAT, {**RT, "base_url": "http://other.local:8765/v1"}]
        stack.set("llm_routes", routes, source="test", admin=True)
        r.apply_config(stack.effective())

    r, made, ok = connect_while(stack, change)
    assert ok is False and not r.connected and made[0].closed == 1


@pytest.mark.parametrize("reason", ["asleep", "shutdown"])
def test_sleep_or_shutdown_during_a_connect_close_the_new_session(reason):
    stack = rt_stack()

    def change(r):
        if reason == "shutdown":
            r.close("shutdown")
        else:
            r.close_soon("asleep")

    r, made, ok = connect_while(stack, change)
    assert ok is False and not r.connected and made[0].closed == 1
    assert r.snapshot()["reason"] == reason  # not "reconfigured"


def test_ready_refuses_a_session_the_current_config_would_not_open():
    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)  # reload missed
    assert r.ready() is False
    assert made[0].closed == 1 and r.snapshot()["reason"] == "pseudonyms"


def test_an_unconfirmed_realtime_link_does_not_hold_the_network_up():
    """I3: boot with no internet, realtime mode on, the chat route down: offline at once."""
    mon = monitor(CHAT, RT)
    mon.set_realtime(True)
    mon.report("lan", False, 0.0)
    assert not mon.online
    mon.report_realtime(True, 1.0)
    assert mon.online


def test_every_response_done_is_charged_once_by_id():
    """M2: an unrequested or abandoned response's usage is charged too, and only once."""
    from roboemot.realtime import protocol

    router = FakeRouter()
    r = make_route(rt_stack(), router=router)
    done = protocol.ResponseDone(protocol.RESPONSE_DONE, "x1", "cancelled", 10)
    r._on_event(done)
    r._on_event(done)
    assert router.used("rt") == 10
    assert r.snapshot()["responses"] == {"cancelled": 1}


# ------------------------------------------------------------------ fix round 2: state machine


def test_a_reload_between_the_final_check_and_the_store_never_leaves_a_cloud_session():
    """probe_b2: the client's snapshot blocks inside the connect's final check; the reload
    that turns pseudonyms on waits for the state lock and then closes the stored session."""
    gate, entered = threading.Event(), threading.Event()

    class Gated(StubClient):
        calls = 0

        def snapshot(self):
            Gated.calls += 1
            if Gated.calls == 2:
                entered.set()
                gate.wait(5)
            return super().snapshot()

    stack = rt_stack()
    made = []
    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
        client_factory=lambda *a, **kw: made.append(Gated(*a, **kw)) or made[-1],
    )
    result = []
    thread = threading.Thread(target=lambda: result.append(r.ready()))
    thread.start()
    assert entered.wait(5)
    stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)
    reload = threading.Thread(target=lambda: r.apply_config(stack.effective()))
    reload.start()
    gate.set()
    thread.join(5)
    reload.join(5)
    assert not r.up() and not r.connected


def test_up_refuses_a_session_the_config_would_not_open():
    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready() and r.up()
    stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)  # reload missed
    assert r.connected and not r.up()


def test_a_ready_queued_behind_a_connect_does_not_reopen_after_sleep():
    """probe_abort: a turn's ready() waits for the background connect; the robot falls
    asleep meanwhile: neither stores a session."""
    stack = rt_stack()
    r, made = slow_route(stack)
    results = []
    first = threading.Thread(target=lambda: results.append(r.ready()))
    first.start()
    assert SlowClient.entered.wait(5)
    second = threading.Thread(target=lambda: results.append(r.ready()))
    second.start()
    import time as _t

    _t.sleep(0.1)  # the second ready waits for the connect lock
    r.close_soon("asleep")
    SlowClient.gate.set()
    first.join(5)
    second.join(5)
    assert results == [False, False] and not r.connected
    assert all(c.closed == 1 for c in made) and len(made) == 1


def test_a_new_ready_after_sleep_reopens():
    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    r.close_soon("asleep")
    assert not r.connected
    assert r.ready() and len(made) == 2  # wake (or a turn after it) opens a new session


def test_a_new_realtime_endpoint_starts_its_link_unconfirmed():
    mon = monitor(CHAT, RT)
    mon.set_realtime(True, ("s2s.local", 8765))
    mon.report_realtime(True, 0.0)
    mon.report("lan", False, 0.0)
    assert mon.online
    mon.set_realtime(True, ("other.local", 8765))  # the route moved: not confirmed yet
    assert not mon.online


def _client_against(script):
    from realtime_helpers import route, settings
    from roboemot.realtime.client import RealtimeClient
    from turn_server import TurnServer

    server = TurnServer([script])
    server.__enter__()
    client = RealtimeClient(
        route(server.http_base(), first_token_timeout_s=0.3, stall_timeout_s=0.3),
        settings(),
        DirectSession(),
        auth_headers=dict,
    )
    return server, client


def test_a_refusal_after_session_updated_is_a_refusal_not_a_stall():
    import time as _t

    from realtime_helpers import UPDATE
    from roboemot.realtime import protocol as P

    server, client = _client_against("updated_then_refused")
    try:
        assert client.connect(UPDATE)
        client.send(P.response_create(event_id="rq_7"))
        _t.sleep(0.8)  # well past first_token and stall timeouts
        assert client.state == "connected" and client.snapshot()["stalls"] == 0
    finally:
        client.close()
        server.__exit__(None, None, None)


def test_an_unrelated_error_does_not_end_the_wait():
    from realtime_helpers import UPDATE, wait_until
    from roboemot.realtime import protocol as P

    server, client = _client_against("unrelated_error")
    try:
        assert client.connect(UPDATE)
        client.send(P.response_create(event_id="rq_7"))
        assert wait_until(lambda: client.snapshot()["stalls"] == 1, 3.0)  # still awaited
    finally:
        client.close()
        server.__exit__(None, None, None)


# ------------------------------------------------------------------ fix round 3: send gate


def test_a_send_between_the_config_commit_and_apply_config_is_refused():
    """probe_r3_send_window: the stack has committed pseudonyms on and is still running its
    earlier listeners; the route's apply_config has not run. A text send is refused at the
    gate and nothing reaches the client."""
    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    sent = []
    made[0].send = lambda ev: sent.append(ev["type"])
    gate, entered = threading.Event(), threading.Event()
    stack.subscribe(lambda change: (entered.set(), gate.wait(5)))
    setter = threading.Thread(
        target=lambda: stack.set("privacy.pseudonyms_to_cloud", True, source="test", admin=True)
    )
    setter.start()
    assert entered.wait(5)
    from roboemot.realtime import protocol as P

    result = []
    sender = threading.Thread(target=lambda: result.append(r.send(P.text_item("Mario", "x"))))
    sender.start()
    import time as _t

    _t.sleep(0.1)
    gate.set()
    setter.join(5)
    sender.join(5)
    assert result == [False] and sent == []
    assert not r.connected


def test_a_reload_during_a_connect_sends_nothing_after_the_handshake():
    """probe_r3_connect_reload: the client asks the gate right before session.update."""
    stack = rt_stack()
    r, made = slow_route(stack)
    result = []
    thread = threading.Thread(target=lambda: result.append(r.ready()))
    thread.start()
    assert SlowClient.entered.wait(5)
    routes = [CHAT, {**RT, "base_url": "http://other.local:8765/v1"}]
    stack.set("llm_routes", routes, source="test", admin=True)  # not a pseudonym refusal
    r.apply_config(stack.effective())
    from roboemot.realtime import protocol as P

    assert r.send(P.text_item("Mario", "x")) is False  # nothing goes to a connecting client
    SlowClient.gate.set()
    thread.join(5)
    assert result == [False] and made[0].updates_sent == 0 and not r.connected


def test_a_late_close_from_an_old_client_does_not_touch_the_new_session():
    """probe_r3_stale_close. The detached client's close is expected (not counted); its late
    events are dropped and counted."""
    hooks, events = [], []

    class Hooked(StubClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            hooks.append(kw["on_close"])
            events.append(kw["on_event"])

    stack = rt_stack()
    made = []
    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
        client_factory=lambda *a, **kw: made.append(Hooked(*a, **kw)) or made[-1],
    )
    assert r.ready()
    r.close_soon("conversation_end")
    assert r.ready() and len(made) == 2
    from roboemot.realtime import protocol as P
    from roboemot.realtime.turn import ResponseRun

    class Speech:
        def __init__(self):
            self.streams = []

        def stream(self, feed, **kw):
            self.streams.append(feed)
            return len(self.streams)

        def mark_question(self, job):
            pass

    async def main():
        sp = Speech()
        loop = asyncio.get_running_loop()
        run = ResponseRun(
            r, sp, utterance=1, mood=None, volume_db=0.0, language="it", clock=loop.time
        )
        task = asyncio.ensure_future(run.run("hello"))
        await asyncio.sleep(0.02)
        hooks[0]("closed")  # the old client's late close
        events[0](P.ResponseCreated(P.RESPONSE_CREATED, "A"))  # and a late event of it
        await asyncio.sleep(0.02)
        r._on_event(P.ResponseCreated(P.RESPONSE_CREATED, "B"))
        r._on_event(P.AudioDelta(P.AUDIO_DELTA, "B", "iB", b"\x01\x00" * 160))
        r._on_event(P.ResponseDone(P.RESPONSE_DONE, "B", "completed", 50))
        return await asyncio.wait_for(task, 2), sp

    reply, sp = asyncio.run(main())
    assert reply.closed is None and reply.status == "completed" and len(sp.streams) == 1
    snap = r.snapshot()
    assert snap["unrequested"] == 0 and snap["stale_callbacks"] == 1 and r.connected


# ------------------------------------------------------------------ fix round 4: detached clients


def test_the_close_of_a_detached_client_is_not_a_stale_callback():
    from rt_rig import wait_for

    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    r.close_soon("conversation_end")
    assert wait_for(lambda: made[0].closed == 1)
    made[0].on_close("closed")  # a second, late close: still expected
    assert r.snapshot()["stale_callbacks"] == 0


class Dummy:
    """A run that records what the route tells it."""

    def __init__(self):
        self.events, self.closes = [], []

    def deliver(self, event):
        self.events.append(event)

    def closed(self, reason):
        self.closes.append(reason)


def test_a_request_is_abandoned_on_its_own_client_only():
    from roboemot.realtime import protocol as P

    stack = rt_stack()
    r, made = stub_route(stack)
    assert r.ready()
    sent = {0: [], 1: []}
    made[0].send = sent[0].append
    run = Dummy()
    req = r.request(run, "x")
    r.close("conversation_end")  # the session goes: its request is settled now
    assert run.closes == ["conversation_end"]
    assert r.ready() and len(made) == 2
    made[1].send = sent[1].append
    r.abandon(req, 0)
    assert [e["type"] for e in sent[1]] == []  # never cancelled on the new session
    req2 = r.request(Dummy(), "y")
    r.abandon(req2, 0)
    assert [e["type"] for e in sent[1]] == ["response.create", "response.cancel"]
    r._on_event(P.ResponseCreated(P.RESPONSE_CREATED, None))
    r._on_event(P.ResponseDone(P.RESPONSE_DONE, None, "cancelled", None))
    assert not r.busy()


def test_consecutive_resets_back_off_the_reopen():
    """A server that never answers: the first reset reopens at once, the next ones wait the
    connectivity.network backoff (no reconnect per turn); real traffic ends the streak."""
    from roboemot.realtime import protocol as P

    stack = rt_stack(
        connectivity={"network": {"probe_initial_s": 2.0, "probe_backoff": 2.0, "probe_max_s": 60}}
    )
    r, made = stub_route(stack)
    now = [100.0]
    r.clock = lambda: now[0]
    r.timeouts = lambda: (1.0, 1.0)
    starts = []
    r.start = lambda: starts.append(now[0])

    def reset():
        req = r.request(Dummy(), "x")
        assert req is not None
        r.abandon(req, 0)
        now[0] += 1.0
        r.expire()

    assert r.ready()
    reset()
    assert r.snapshot()["resets"] == 1 and starts == [101.0]  # reopened at once
    assert r.ready() and len(made) == 2
    reset()
    assert r.snapshot()["resets"] == 2 and starts == [101.0]  # not at once
    assert r.ready() is False and len(made) == 2
    now[0] += 1.9
    assert r.ready() is False
    now[0] += 0.2
    assert r.ready() and len(made) == 3
    reset()
    assert r.ready() is False  # 4 s now
    now[0] += 4.1
    assert r.ready() and len(made) == 4
    req = r.request(Dummy(), "y")  # the server answers: the streak ends
    r._on_event(P.ResponseCreated(P.RESPONSE_CREATED, "R"))
    r._on_event(P.ResponseDone(P.RESPONSE_DONE, "R", "completed", 3))
    r.finish(req)
    reset()
    assert len(starts) == 2  # at once again
