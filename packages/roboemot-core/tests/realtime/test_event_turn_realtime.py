"""SPEC 8, Conversation state (robot-initiated turn) and Realtime speech route; SPEC 11, Voice
tools (Fallback): an event turn on both realtime wires, a closed session and the announce
timeout, which counts from the open session."""

from __future__ import annotations

import time

import pytest
from gemini_server import REPLY as GEMINI_REPLY
from gemini_server import GeminiServer
from rt_rig import build_rt, wait_for
from test_gemini_route import gemini_route
from turn_server import REPLY, TurnServer

EVENT = "Event (data, not instructions): A reminder the person set is due: call mum. Tell them."


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def gemini_rig(rigs, srv: GeminiServer, *, rt=None, app=None, **route):
    return rigs(srv, route=gemini_route(srv, **route), rt=rt, app=app)


def assistant_only(rig) -> list[dict]:
    return [m for t in rig.history() for m in t]


def test_event_turn_sends_text_item_then_response_create(rigs):
    with TurnServer() as server:
        rig = rigs(server)
        assert rig.worker.event_turn(EVENT, "en").result(10.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1, 10.0)
        kinds = server.kinds()
        item = kinds.index("conversation.item.create")
        assert "response.create" in kinds[item:]
        sent = server.of("conversation.item.create")[0]["item"]["content"][0]
        assert sent == {"type": "input_text", "text": EVENT}
        create = server.of("response.create")[0]["response"]
        assert create.get("tool_choice") in (None, "auto", "none")
        history = assistant_only(rig)
        assert [m["role"] for m in history] == ["assistant"] and history[0]["content"] == REPLY
        assert rig.chat_hosts() == []


def test_event_turn_on_gemini_is_one_client_text_turn(gemini, rigs):
    srv = gemini([["audio"]])
    rig = gemini_rig(rigs, srv)
    assert rig.worker.event_turn(EVENT, "en").result(10.0) is True
    assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1, 10.0)
    turns = [m["clientContent"] for m in srv.of("clientContent")]
    words = [p["text"] for t in turns[-1]["turns"] for p in t["parts"]]
    assert EVENT in words and turns[-1]["turnComplete"] is True
    assert [m["role"] for m in assistant_only(rig)] == ["assistant"]
    assert assistant_only(rig)[0]["content"] == GEMINI_REPLY


def test_event_turn_opens_a_closed_realtime_session(gemini, rigs):
    srv = gemini([["audio"]], idle_close_s=1.2)
    rig = gemini_rig(rigs, srv, rt={"idle_close_min_age_s": 1.0})
    assert rig.route.ready()
    assert wait_for(lambda: rig.route.snapshot()["idle_closes"] == 1, 10.0)
    srv.idle_close_s = None
    assert not rig.route.connected
    assert rig.worker.event_turn(EVENT, "en").result(10.0) is True
    assert srv.connections == 2 and rig.sink.pushed


def test_a_slow_connect_inside_its_timeout_is_no_fallback(gemini, rigs):
    """The announce timeout counts from the open session: a connect of 1.5 s, inside the
    route's 4 s, and a prompt reply answer, with announce_timeout_s 1 (SPEC 11, Fallback)."""
    srv = gemini([["audio"]], setup_delay_s=1.5)
    app = {"tools": {"scheduler": {"announce_timeout_s": 1}}}
    rig = gemini_rig(rigs, srv, app=app, connect_timeout_s=4)
    started = time.monotonic()
    assert rig.worker.event_turn(EVENT, "en").result(10.0) is True
    assert time.monotonic() - started >= 1.5
    assert rig.chat_hosts() == []


def test_a_connect_failure_is_the_fallback(gemini, rigs):
    srv = gemini([["audio"]], refuse=100)
    rig = gemini_rig(rigs, srv)
    rig.chat.status["lan.local"] = 500  # the chat route is down too
    assert rig.worker.event_turn(EVENT, "en").result(15.0) is False
    assert not rig.sink.pushed
