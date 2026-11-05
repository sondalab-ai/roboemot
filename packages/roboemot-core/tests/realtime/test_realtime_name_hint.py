"""The robot's name as the server's transcription hint (SPEC 8, Realtime speech route):
``session.update`` ``audio.input.transcription.prompt`` carries the name the wake word resolves,
and a server that refuses the field gets the update again without it, once."""

from __future__ import annotations

import pytest
from fake_server import FakeServer
from realtime_helpers import wait_until
from roboemot.config import ConfigStack
from roboemot.health.network import NetworkMonitor
from roboemot.realtime.route import RealtimeRoute
from rt_rig import build_rt
from test_realtime_audio_route import SECOND, audio_route, audio_stack, transcription
from test_realtime_route import CHAT, DIRECT, POLICY, RT, FakeRouter
from turn_server import TurnServer


def hinted_route(stack, name="Neve"):
    r, made = audio_route(stack)
    r.transcription_name = lambda: name
    return r, made


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
    update = r.session_update(stack.effective(), "x")
    assert "transcription" not in update["session"]["audio"]["input"]


def test_the_hint_is_the_name_the_worker_resolves_for_the_wake_word():
    with TurnServer() as server:
        rig = build_rt(server, app={"voice": {"wake_word": "neve"}})
        try:
            assert rig.route.transcription_name() == "Neve"  # display_name, sanitised
        finally:
            rig.close()


# ------------------------------------------------------------------ a server that refuses it


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


def prompt_of(update):
    return update["session"]["audio"]["input"].get("transcription", {}).get("prompt")


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
            assert snap["client"]["update_fallback"] is True
            assert r.send_audio(SECOND, gate="addressed", instructions="") is None
            assert wait_until(lambda: len(updates(server)) == 3)
            assert prompt_of(updates(server)[-1]) is None  # not sent to this route again
            assert r.state == "connected"
        finally:
            r.close()


def test_a_hint_refused_mid_session_is_resent_once_without_it():
    """The platform's name can arrive after the connect: the first hinted update is then a
    burst's, and the server's error names its event_id."""
    names = [""]
    with FakeServer(refuse_prompt=True) as server:
        r = live_route(server, lambda: names[0])
        try:
            assert r.ready()
            assert [prompt_of(u) for u in updates(server)] == [None]
            names[0] = "Neve"
            assert r.send_audio(SECOND, gate="addressed", instructions="") is None
            assert wait_until(lambda: len(updates(server)) == 3)
            assert [prompt_of(u) for u in updates(server)] == [None, "Neve", None]
            assert wait_until(lambda: r.snapshot()["transcription_hint"]["refused"] == 1)
            snap = r.snapshot()
            assert snap["transcription_hint"] == {"sent": 1, "refused": 1}
            assert snap["errors_unrelated"] == 0  # the refusal is the hint's, not unexplained
            assert r.state == "connected"
            assert r.send_audio(SECOND, gate="addressed", instructions="") is None
            assert wait_until(lambda: len(updates(server)) == 4)
            assert prompt_of(updates(server)[-1]) is None
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
            assert r.hint(r.stack.effective()) == "Neve"
        finally:
            r.close()
