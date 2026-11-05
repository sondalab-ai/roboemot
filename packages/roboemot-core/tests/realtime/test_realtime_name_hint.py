"""The robot's name as the server's transcription hint (SPEC 8, Realtime speech route):
``session.update`` ``audio.input.transcription.prompt`` carries the name the wake word resolves.
The hint is decided at connect only: a server that refuses it gets the connect's update again
without it, and that endpoint is remembered; a session's later updates carry it only when the
connect's update did and the server confirmed it."""

from __future__ import annotations

import pytest
from fake_server import FakeServer
from realtime_helpers import UPDATE, settings, wait_until
from realtime_helpers import route as client_route
from roboemot.config import ConfigStack
from roboemot.health.network import NetworkMonitor
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.route import RealtimeRoute, without_hint
from rt_rig import build_rt
from test_realtime_audio_route import SECOND, audio_route, audio_stack, transcription
from test_realtime_route import CHAT, DIRECT, POLICY, RT, FakeRouter
from turn_server import TurnServer


def hinted_route(stack, name="Neve"):
    r, made = audio_route(stack)
    r.transcription_name = lambda: name
    return r, made


def prompt_of(update):
    return update["session"]["audio"]["input"].get("transcription", {}).get("prompt")


# ------------------------------------------------------------------ what is sent


def test_the_update_carries_the_robots_name_as_the_transcription_prompt():
    r, made = hinted_route(audio_stack())
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None
    assert transcription(made[0]) == {"language": "it", "prompt": "Neve"}
    assert r.snapshot()["transcription_hint"] == {"sent": 1, "refused": 0}


@pytest.mark.parametrize(
    ("rt", "name"),
    [
        ({}, ""),  # no name: only the generic wake phrases apply
        ({"transcription_name_hint": False}, "Neve"),  # turned off
    ],
)
def test_no_hint_without_a_name_or_with_the_field_off(rt, name):
    r, made = hinted_route(audio_stack(rt), name)
    assert r.ready()
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None
    assert transcription(made[0]) == {"language": "it"}
    assert r.snapshot()["transcription_hint"] == {"sent": 0, "refused": 0}


def test_text_input_sends_no_hint():
    stack = audio_stack({"input": "text"})
    r, _ = hinted_route(stack)
    assert r.hint(stack.effective()) is None
    update = r.session_update(stack.effective(), "x")
    assert "transcription" not in update["session"]["audio"]["input"]


def test_the_hint_is_the_name_the_worker_resolves_for_the_wake_word():
    with TurnServer() as server:
        rig = build_rt(server, app={"voice": {"wake_word": "neve"}})
        try:
            assert rig.route.transcription_name() == "Neve"  # display_name, sanitised
        finally:
            rig.close()


# ------------------------------------------------------------------ the refusal's lifecycle


class HintStub:
    """A connected client on a local host: ``refuse`` makes its server refuse the hinted
    update (the session comes up on the fallback), ``origin`` is its connect endpoint."""

    origin = "ws://s2s.local:8765"
    refuse = False

    def __init__(self, route, settings, provider, **kw):
        self.route, self.state, self.reason = route, "idle", None
        self.on_event = kw.get("on_event", lambda event: None)
        self.on_close = kw.get("on_close", lambda reason: None)
        self.sent: list[dict] = []
        self.how = None
        self.origin = HintStub.origin

    def connect(self, update, fallback=None, prefer_fallback=lambda origin: False):
        if fallback is not None and prefer_fallback(self.origin):
            self.sent.append(fallback)
            self.how = "preferred"
        elif fallback is not None and HintStub.refuse:
            self.sent += [update, fallback]
            self.how = "resent"
        else:
            self.sent.append(update)
        self.state = "connected"
        return True

    def snapshot(self):
        return {"local": True, "endpoint": self.origin, "sent": 1, "update_fallback": self.how}

    def send(self, event):
        self.sent.append(event)

    def close(self):
        self.state = "closed"


@pytest.fixture()
def stub_route():
    HintStub.origin, HintStub.refuse = "ws://s2s.local:8765", False
    names = ["Neve"]
    made: list[HintStub] = []
    stack = audio_stack()

    def factory(*a, **kw):
        made.append(HintStub(*a, **kw))
        return made[-1]

    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
        client_factory=factory,
    )
    r.transcription_name = lambda: names[0]
    return r, made, stack, names


def reconnect(r):
    r.close_soon("conversation_end")
    assert r.ready()


def burst_prompt(r, client):
    assert r.send_audio(SECOND, gate="addressed", instructions="") is None
    return prompt_of([e for e in client.sent if e["type"] == "session.update"][-1])


def refuse_once(r, made):
    HintStub.refuse = True
    assert r.ready()
    HintStub.refuse = False
    assert [prompt_of(u) for u in made[-1].sent] == ["Neve", None]
    assert r.snapshot()["transcription_hint"] == {"sent": 1, "refused": 1}
    assert burst_prompt(r, made[-1]) is None  # the session holds no hint


def test_a_refused_endpoint_gets_the_update_without_the_hint_at_the_next_connect(stub_route):
    r, made, _, _ = stub_route
    refuse_once(r, made)
    reconnect(r)
    assert [prompt_of(u) for u in made[-1].sent] == [None]  # the fallback, sent first
    assert r.snapshot()["transcription_hint"] == {"sent": 1, "refused": 1}
    assert burst_prompt(r, made[-1]) is None


@pytest.mark.parametrize(
    "changes",
    [
        [("transcription_model", "m2")],
        [("transcription_name_hint", False), ("transcription_name_hint", True)],
    ],
    ids=["model", "switch"],
)
def test_a_new_model_or_a_toggled_switch_clears_a_refusal(stub_route, changes):
    r, made, stack, _ = stub_route
    refuse_once(r, made)
    for key, value in changes:  # each reload reaches the route (the stack notifies it)
        stack.set(f"voice.realtime_route.{key}", value, source="test")
        r.apply_config(stack.effective())
    reconnect(r)
    assert prompt_of(made[-1].sent[0]) == "Neve"  # tried again
    assert burst_prompt(r, made[-1]) == "Neve"


def test_another_endpoint_is_not_held_by_one_endpoints_refusal(stub_route):
    """In platform mode each allocation can name another endpoint."""
    r, made, _, _ = stub_route
    refuse_once(r, made)
    HintStub.origin = "ws://other.local:8765"
    reconnect(r)
    assert [prompt_of(u) for u in made[-1].sent] == ["Neve"]
    assert burst_prompt(r, made[-1]) == "Neve"


def test_a_name_that_arrives_mid_session_applies_from_the_next_connect(stub_route):
    r, made, _, names = stub_route
    names[0] = ""  # the platform has not reported its name yet
    assert r.ready()
    names[0] = "Neve"
    assert burst_prompt(r, made[-1]) is None  # not added mid-session
    reconnect(r)
    assert prompt_of(made[-1].sent[0]) == "Neve"
    assert burst_prompt(r, made[-1]) == "Neve"


def test_an_old_sessions_late_error_changes_nothing(stub_route):
    r, made, _, _ = stub_route
    assert r.ready()
    reconnect(r)
    old, new = made
    old.on_event(protocol.ServerError(protocol.ERROR, "unknown_parameter", "rq_1"))
    snap = r.snapshot()
    assert snap["transcription_hint"]["refused"] == 0 and snap["stale_callbacks"] == 1
    assert burst_prompt(r, new) == "Neve"


def test_a_mid_session_error_is_not_a_refusal_and_resends_nothing(stub_route):
    r, made, _, _ = stub_route
    assert r.ready()
    sent = len(made[0].sent)
    r._on_event(protocol.ServerError(protocol.ERROR, "unknown_parameter", "ev_9"))
    assert len(made[0].sent) == sent
    assert r.snapshot()["transcription_hint"]["refused"] == 0
    assert burst_prompt(r, made[0]) == "Neve"


# ------------------------------------------------------------------ against the fake server


def live_route(server, name):
    rt_route = {**RT, "base_url": server.http_base(), "locality": "local", "api_key": "none"}
    stack = ConfigStack(
        app={
            "llm_routes": [CHAT, rt_route],
            "voice": {
                "sample_rate_hz": 16000,
                "realtime_route": {
                    "enabled": True,
                    "route": "rt",
                    "input": "audio",
                    "connect_timeout_s": 2,
                    "send_timeout_s": 1,
                },
            },
            "privacy": {"realtime_audio": True},
        }
    )
    r = RealtimeRoute(
        stack,
        lambda: FakeRouter(),
        NetworkMonitor(stack.effective().llm_routes, POLICY, probe=lambda *a: True),
        DIRECT,
        extra=lambda: True,
    )
    r.transcription_name = name
    return r


def updates(server):
    return [e for e in server.received if e["type"] == "session.update"]


@pytest.mark.parametrize("echo", [True, False])
def test_a_refused_hint_at_connect_is_resent_once_without_it(echo):
    """The update is the only event before ``session.updated``: an error then is about it,
    whether or not the server echoes its event_id."""
    with FakeServer(refuse_prompt=True, echo_error_id=echo) as server:
        r = live_route(server, lambda: "Neve")
        try:
            assert r.ready()
            assert r.state == "connected"
            assert [prompt_of(u) for u in updates(server)] == ["Neve", None]
            snap = r.snapshot()
            assert snap["transcription_hint"] == {"sent": 1, "refused": 1}
            assert snap["client"]["update_fallback"] == "resent"
            assert snap["errors_unrelated"] == 0  # the resent error is not handed on
            assert r.send_audio(SECOND, gate="addressed", instructions="") is None
            assert wait_until(lambda: len(updates(server)) == 3)
            assert prompt_of(updates(server)[-1]) is None
            assert r.state == "connected"
        finally:
            r.close()


def test_any_non_account_error_at_connect_counts_as_a_refusal_of_the_hint():
    """Pinned: the server gives no cause, so an error on the hinted update that the update
    without the hint then passes is taken as the hint's refusal, whatever its code."""
    with FakeServer("error_once") as server:
        r = live_route(server, lambda: "Neve")
        try:
            assert r.ready()
            assert [prompt_of(u) for u in updates(server)] == ["Neve", None]
            assert r.snapshot()["transcription_hint"] == {"sent": 1, "refused": 1}
        finally:
            r.close()


@pytest.mark.parametrize("code", ["invalid_api_key", "insufficient_quota"])
def test_an_account_error_at_connect_is_not_resent_and_keeps_the_hint(code):
    with FakeServer("error_first", error_code=code) as server:
        r = live_route(server, lambda: "Neve")
        try:
            assert not r.ready()
            # every connect (a background reconnect may follow) sent the hint and no resend
            assert {prompt_of(u) for u in updates(server)} == {"Neve"}
            assert r.snapshot()["transcription_hint"]["refused"] == 0
            assert r.hint(r.stack.effective()) == "Neve"
        finally:
            r.close()


def test_a_server_that_refuses_every_update_gets_one_resend_and_the_hint_stays():
    """Refused again without the hint: the hint was not the cause, so it is not dropped."""
    with FakeServer("error_first") as server:
        r = live_route(server, lambda: "Neve")
        try:
            assert not r.ready()
            # the first connect (a background reconnect may follow: it starts the same way)
            assert [prompt_of(u) for u in updates(server)][:2] == ["Neve", None]
            snap = r.snapshot()
            assert snap["transcription_hint"]["sent"] >= 1
            assert snap["transcription_hint"]["refused"] == 0
        finally:
            r.close()


def test_the_send_gate_can_refuse_the_resend():
    """The resend is data too: the owner's gate is asked again before it leaves (SPEC 12)."""
    hinted = {**UPDATE, "session": {**UPDATE["session"]}}
    hinted["session"]["audio"] = {
        **UPDATE["session"]["audio"],
        "input": {**UPDATE["session"]["audio"]["input"], "transcription": {"prompt": "Neve"}},
    }
    asked = []

    def may_send():
        asked.append(1)
        return len(asked) == 1  # the first update may go, the resend may not

    with FakeServer(refuse_prompt=True) as server:
        client = RealtimeClient(
            client_route(server.http_base()),
            settings(),
            DIRECT,
            auth_headers=dict,
            may_send=may_send,
        )
        try:
            assert not client.connect(hinted, fallback=without_hint(hinted))
            assert client.reason == "reconfigured"
            assert len(updates(server)) == 1 and len(asked) == 2
        finally:
            client.close()
