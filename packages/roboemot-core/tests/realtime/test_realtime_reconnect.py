"""The realtime route reconnects in the background after a lost session or a failed connect,
paced by the ``connectivity.network`` backoff, and never after a close of the robot's own,
shutdown, a block or sleep (SPEC 8, Realtime speech route; CV-12 robot QA 2026-11-03: the Live
tab stayed on ``connection_lost`` until the user spoke)."""

from __future__ import annotations

import threading

import pytest
from realtime_helpers import wait_until
from roboemot.config import ConfigStack
from roboemot.config.schema import LinkPolicy
from roboemot.health.network import NetworkMonitor
from roboemot.realtime import protocol
from roboemot.realtime.route import RealtimeRoute
from roboemot.realtime.session import DirectSession

INITIAL_S = 2.0  # connectivity.network.probe_initial_s in these tests
POLICY = LinkPolicy(
    failures_to_offline=1,
    successes_to_online=1,
    probe_initial_s=INITIAL_S,
    probe_backoff=2.0,
    probe_max_s=60.0,
    probe_jitter=0.0,
)
RT = {"name": "rt", "base_url": "http://s2s.local:8765/v1", "model": "s2s", "kind": "realtime"}


def rt_stack(**app):
    cfg = {
        "llm_routes": [RT],
        "voice": {"realtime_route": {"enabled": True, "route": "rt"}},
        "connectivity": {"network": POLICY.model_dump()},
        **app,
    }
    return ConfigStack(app=cfg)


class Router:
    """``realtime_route`` follows the route's own link (``unavailable`` while it is offline),
    like the real router."""

    def __init__(self, monitor):
        self.monitor, self.ok = monitor, True

    def realtime_route(self, name):
        return object() if self.ok and self.monitor.is_online(name) else None

    def used(self, name):
        return 0

    def auth_headers(self, route):
        return {}

    def charge(self, name, tokens):
        return None


class Client:
    """A client whose connects succeed or fail as the test says (``Client.fail``); ``gate``,
    when set, holds the connect until the test releases it."""

    fail = False
    gate: threading.Event | None = None
    entered = threading.Event()

    def __init__(self, route, settings, provider, **kw):
        self.route, self.state, self.reason = route, "idle", None
        self.on_close = kw["on_close"]
        self.sent, self.updates, self.closed = [], [], 0

    def connect(self, update):
        Client.entered.set()
        if Client.gate is not None:
            assert Client.gate.wait(5)
        self.updates.append(update)
        if Client.fail:
            self.state, self.reason = "closed", "connect_failed"
            self.on_close("connect_failed")
            return False
        self.state = "connected"
        return True

    def drop(self, reason="connection_lost"):
        """The server or the network ends the session (no turn in flight)."""
        self.state, self.reason = "closed", reason
        self.on_close(reason)

    def snapshot(self):
        return {"local": True, "endpoint": "ws://s2s.local:8765"}

    def send(self, event):
        self.sent.append(event)

    def close(self):
        self.closed += 1
        self.state = "closed"


@pytest.fixture
def rig(monkeypatch):
    """A route on a fake clock (route and monitor), its clients and its decision events. The
    reconnect worker's thread is not started: the tests drive ``_reconnect_step`` (``spawned``
    counts the starts)."""
    Client.fail, Client.gate, Client.entered = False, None, threading.Event()
    now = [100.0]
    stack = rt_stack()
    monitor = NetworkMonitor(
        stack.effective().llm_routes, POLICY, probe=lambda *a: True, clock=lambda: now[0]
    )
    router = Router(monitor)
    made, events, spawned = [], [], []

    def factory(*a, **kw):
        made.append(Client(*a, **kw))
        return made[-1]

    r = RealtimeRoute(
        stack,
        lambda: router,
        monitor,
        DirectSession(),
        extra=lambda: True,
        client_factory=factory,
        clock=lambda: now[0],
    )
    r.events = lambda kind, code, **kw: events.append((kind, code, kw))
    r._keep_connecting = lambda: spawned.append(1)
    yield r, now, made, events, spawned, router
    r.close()


def test_a_lost_session_reconnects_by_itself_once_the_backoff_allows_it(rig):
    """Regression (robot QA 2026-11-03): no turn calls ready(), yet the route comes back."""
    r, now, made, events, spawned, _ = rig
    assert r.ready() and len(made) == 1
    made[0].drop()
    assert (r.state, r.reason) == ("fallback_local", "connection_lost")
    assert spawned == [1]  # the worker was started by the loss itself
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # the backoff holds it
    assert len(made) == 1
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0
    assert len(made) == 2 and r.connected and (r.state, r.reason) == ("connected", None)
    assert r._reconnect_step() is None and not r._retrying  # over: the session is up
    assert ("session", "reconnected", {"after": "connection_lost"}) in events
    assert r.snapshot()["reconnects"] == 1


def test_the_worker_thread_reconnects_and_ends_without_a_leak(rig):
    r, now, made, _, _, _ = rig
    del r._keep_connecting  # the real worker this time
    assert r.ready()
    made[0].drop()
    thread = r._retry_thread
    assert thread is not None and thread.is_alive()
    now[0] += INITIAL_S
    r._retry_wake.set()  # the fake clock moved: cut the real wait short
    assert wait_until(lambda: r.connected)
    thread.join(5)
    assert not thread.is_alive() and not r._retrying and len(made) == 2


def test_a_failed_start_keeps_retrying_with_the_growing_backoff(rig):
    r, now, made, _, spawned, _ = rig
    Client.fail = True
    r.start()
    assert spawned == [1]
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # attempted, failed: 2 s
    assert len(made) == 1 and r.state == "fallback_local"
    now[0] += INITIAL_S - 0.1
    assert r._reconnect_step() == pytest.approx(0.1)  # not due: no attempt
    assert len(made) == 1
    now[0] += 0.1
    assert r._reconnect_step() == pytest.approx(2 * INITIAL_S)  # failed again: 4 s
    assert len(made) == 2
    Client.fail = False
    now[0] += 2 * INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected and len(made) == 3
    assert r._reconnect_step() is None
    assert spawned == [1]  # one worker throughout (each failure only told it again)


def test_a_turns_failed_connect_starts_the_background_reconnect(rig):
    r, now, made, _, spawned, _ = rig
    Client.fail = True
    assert r.ready() is False
    assert r.state == "fallback_local" and spawned == [1]


@pytest.mark.parametrize("reason", ["asleep", "budget"])
def test_no_background_reconnect_after_sleep_or_a_spent_budget(rig, reason):
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    r.close_soon(reason)
    now[0] += 10 * INITIAL_S
    assert r._reconnect_step() is None
    assert len(made) == 1 and not r._retrying


def test_the_conversations_end_after_a_loss_keeps_the_background_reconnect(rig):
    """The Live tab must not stay on connection_lost because the conversation ended: the
    worker goes on (a new session is a fresh one anyway)."""
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    r.close_soon("conversation_end")
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # still paced by the backoff
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected and len(made) == 2
    assert r._reconnect_step() is None and spawned == [1]


def test_the_conversations_end_with_no_loss_starts_nothing(rig):
    r, _, made, _, spawned, _ = rig
    assert r.ready()
    r.close_soon("conversation_end")
    assert spawned == [] and not r._retrying and r.state == "closed"


def in_flight(r, close):
    """Run one worker step whose connect is held, call ``close`` while it is in flight, let
    the connect succeed; returns the step's result."""
    Client.gate, Client.entered = threading.Event(), threading.Event()
    results = []
    worker = threading.Thread(target=lambda: results.append(r._reconnect_step()))
    worker.start()
    assert Client.entered.wait(5)
    close()
    Client.gate.set()
    worker.join(5)
    return results[0]


def test_the_conversations_end_during_a_background_connect_stores_no_session(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S
    in_flight(r, lambda: r.close_soon("conversation_end"))
    assert not r.connected and (r.state, r.reason) == ("closed", "conversation_end")
    assert made[1].closed == 1  # the connected client was closed, not kept
    assert r._reconnect_step() is None and not r._retrying


@pytest.mark.parametrize("reason", ["asleep", "shutdown"])
def test_sleep_or_shutdown_during_a_background_connect_stores_no_session(rig, reason):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S
    close = (
        (lambda: r.close("shutdown")) if reason == "shutdown" else (lambda: r.close_soon(reason))
    )
    in_flight(r, close)
    assert not r.connected and made[1].closed == 1
    assert r._reconnect_step() is None and not r._retrying
    now[0] += 10 * INITIAL_S
    assert r._reconnect_step() is None and len(made) == 2


def test_wake_after_sleep_reconnects_in_the_background(rig):
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    r.close_soon("asleep")
    assert r._reconnect_step() is None
    r.start()  # wake
    assert spawned == [1, 1]
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # the loss's backoff still holds
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected and len(made) == 2


def test_a_reload_that_unblocks_a_lost_route_starts_the_worker(rig):
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    r.stack.set("voice.realtime_route.enabled", False, source="test", admin=True)
    r.apply_config(r.stack.effective())
    assert r._reconnect_step() is None  # the block ends it at once (recorded: off)
    assert (r.state, r.reason) == ("off", "off") and spawned == [1]
    r._set_state("fallback_local", "connection_lost")  # as a turn's failed connect leaves it
    r.stack.set("voice.realtime_route.enabled", True, source="test", admin=True)
    r.apply_config(r.stack.effective())
    assert spawned == [1, 1] and r._retrying
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected


def test_a_reload_that_blocks_the_route_ends_the_worker_at_once(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # waiting the backoff
    r.stack.set("voice.realtime_route.enabled", False, source="test", admin=True)
    r.apply_config(r.stack.effective())
    assert r._retry_wake.is_set()  # the wait is cut short
    assert r._reconnect_step() is None and len(made) == 1  # no wait for the backoff


def test_a_failing_worker_ends_and_the_next_loss_starts_a_new_one(rig, monkeypatch):
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S

    def boom(*a, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr(r, "ready", boom)
    RealtimeRoute._keep_connecting(r)  # the real loop, on this thread: it logs and ends
    assert not r._retrying
    monkeypatch.undo()
    r._keep_connecting = lambda: spawned.append(1)
    assert r.ready() and len(made) == 2
    made[1].drop()
    assert spawned == [1, 1] and r._retrying


def test_a_flapping_server_is_reconnected_ever_more_slowly(rig):
    """A server that accepts each session and drops it at once, never answering: the
    reset streak's backoff bounds the reconnect rate (up to probe_max_s)."""
    r, now, made, _, _, _ = rig
    assert r.ready()
    waits = []
    for _ in range(8):
        made[-1].drop()
        n = len(made)
        while len(made) == n:
            wait = r._reconnect_step()
            assert wait is not None
            if wait > 0:
                waits.append(wait)
                now[0] += wait
    assert waits[:5] == pytest.approx([2.0, 2.0, 4.0, 8.0, 16.0])
    assert waits == sorted(waits) and waits[-1] == pytest.approx(POLICY.probe_max_s)


def test_a_session_that_held_starts_a_new_streak(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    for _ in range(3):
        made[-1].drop()
        now[0] += 100.0
        assert r._reconnect_step() == 0.0
    now[0] += POLICY.probe_max_s  # held long enough
    made[-1].drop()
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # the link's first step only


def test_no_router_or_no_provider_ends_the_worker(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    r._router = lambda: None
    assert r._reconnect_step() is None and not r._retrying


def test_a_reload_that_keeps_the_route_open_lets_the_worker_go_on(rig):
    """A reload closes nothing of the robot's: the worker connects with the new config."""
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    r.stack.set("voice.realtime_route.voice", "alloy", source="test", admin=True)
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected and len(made) == 2
    assert made[1].updates[0]["session"]["audio"]["output"]["voice"] == "alloy"


@pytest.mark.parametrize("reason", ["conversation_end", "asleep", "expired"])
def test_an_own_close_reported_by_the_client_starts_no_reconnect(rig, reason):
    r, _, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop(reason)
    assert spawned == [] and r.state != "fallback_local"


def test_no_background_reconnect_after_shutdown(rig):
    r, now, made, _, spawned, _ = rig
    assert r.ready()
    made[0].drop()
    r.close("shutdown")
    now[0] += 10 * INITIAL_S
    assert r._reconnect_step() is None and len(made) == 1
    r.start()
    assert spawned == [1]  # a start after shutdown starts nothing


def test_sleep_during_a_connect_that_then_fails_starts_no_reconnect(rig):
    r, now, made, _, spawned, _ = rig
    Client.fail, Client.gate = True, threading.Event()
    result = []
    turn = threading.Thread(target=lambda: result.append(r.ready()))
    turn.start()
    assert Client.entered.wait(5)
    r.close_soon("asleep")
    Client.gate.set()
    turn.join(5)
    assert result == [False] and spawned == [] and not r._retrying


def test_a_blocked_route_ends_the_worker_without_a_connect(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    r.stack.set("voice.realtime_route.enabled", False, source="test", admin=True)
    now[0] += INITIAL_S
    assert r._reconnect_step() is None
    assert len(made) == 1 and (r.state, r.reason) == ("off", "off")


def test_a_route_known_unreachable_is_waited_for_not_connected_through(rig):
    """``unavailable`` while the route's own link is offline lifts when its check succeeds:
    the worker waits for that check and never connects meanwhile."""
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    r.network.report("rt", False)  # the route's link check failed: offline, next in 2 s
    now[0] += INITIAL_S
    assert r._reconnect_step() == pytest.approx(INITIAL_S)
    assert (r.state, r.reason) == ("blocked", "unavailable") and len(made) == 1
    assert r._reconnect_step() == pytest.approx(INITIAL_S)  # still waiting, still no connect
    assert len(made) == 1
    now[0] += INITIAL_S
    r.network.report("rt", True)  # the check succeeds
    assert r._reconnect_step() == 0.0 and r.connected and len(made) == 2


def test_an_unavailable_route_whose_link_is_online_ends_the_worker(rig):
    r, now, made, _, _, router = rig
    assert r.ready()
    made[0].drop()
    router.ok = False  # the key is missing: not the link
    now[0] += INITIAL_S
    assert r._reconnect_step() is None and len(made) == 1


def test_a_turn_racing_the_background_reconnect_opens_one_session(rig):
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S
    Client.gate = threading.Event()
    results = []
    worker = threading.Thread(target=lambda: results.append(r._reconnect_step()))
    worker.start()
    assert Client.entered.wait(5)  # the worker's connect is in flight
    turn = threading.Thread(target=lambda: results.append(r.ready()))
    turn.start()
    assert not wait_until(lambda: len(made) > 2, timeout_s=0.2)  # queued on the connect lock
    Client.gate.set()
    worker.join(5)
    turn.join(5)
    assert len(results) == 2 and len(made) == 2 and r.connected
    assert 0.0 in results  # the worker's step connected
    assert True in results  # the turn found the worker's session


def test_a_turn_queued_behind_a_failed_background_connect_does_not_connect_again(rig):
    """The backoff is checked again under the connect lock: the worker's attempt failed, so
    the queued turn falls back at once instead of a second connect."""
    r, now, made, _, _, _ = rig
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S
    Client.fail, Client.gate = True, threading.Event()
    step, turned = [], []
    worker = threading.Thread(target=lambda: step.append(r._reconnect_step()))
    worker.start()
    assert Client.entered.wait(5)
    turn = threading.Thread(target=lambda: turned.append(r.ready()))
    turn.start()
    assert not wait_until(lambda: len(made) > 2, timeout_s=0.2)
    Client.gate.set()
    worker.join(5)
    turn.join(5)
    assert step == [pytest.approx(2 * INITIAL_S)]  # failed: the grown backoff
    assert turned == [False] and len(made) == 2  # one attempt only


def test_a_background_connect_sends_no_microphone_audio(rig):
    r, now, made, _, _, _ = rig
    r.stack.set("privacy.realtime_audio", True, source="test", admin=True)
    r.stack.set("voice.realtime_route.input", "audio", source="test", admin=True)
    assert r.ready()
    made[0].drop()
    now[0] += INITIAL_S
    assert r._reconnect_step() == 0.0 and r.connected
    sent = [e.get("type") for c in made for e in c.sent]
    assert protocol.AUDIO_APPEND not in sent and protocol.AUDIO_COMMIT not in sent
    assert all(len(c.updates) <= 1 for c in made)  # only the session.update of the connect
