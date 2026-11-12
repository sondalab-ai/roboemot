"""The client and the URL rules for a ``gemini_live`` route (SPEC 8, Gemini Live wire; SPEC 12):
the URL as written, the key in ``x-goog-api-key`` only and only to the route's own origin, and
a codec per session whose ids never repeat (the route keys its charges by response id)."""

from __future__ import annotations

import logging

import pytest
from realtime_helpers import KEY, UPDATE, bearer, settings, wait_until
from roboemot.config.schema import LLMRoute
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.gemini_wire import GeminiWire
from roboemot.realtime.session import (
    Allocation,
    DirectSession,
    UrlRefused,
    connect_target,
    session_url,
)

BASE = "wss://generativelanguage.googleapis.com/ws/google.ai.v1beta.BidiGenerateContent"


def gemini_route(base_url: str = BASE, *, keyed: bool = True, **kw: object) -> LLMRoute:
    return LLMRoute(
        name="gemini",
        base_url=base_url,
        model="gemini-3.8-live",
        kind="realtime",
        protocol="gemini_live",
        api_key="${secret:gemini_key}" if keyed else "none",
        **kw,
    )


def test_the_session_url_is_the_base_url_as_written():
    assert session_url(gemini_route()) == BASE
    pollen = LLMRoute(name="p", base_url="https://h.example/v1", model="m", kind="realtime")
    assert session_url(pollen) == "wss://h.example/v1/realtime"  # unchanged


def test_the_key_goes_to_the_routes_own_origin_only():
    route = gemini_route()
    own = connect_target(route, Allocation(BASE))
    assert own.send_key and own.origin == "wss://generativelanguage.googleapis.com:443"
    with pytest.raises(UrlRefused) as refused:
        connect_target(route, Allocation("wss://elsewhere.example/ws"))
    assert refused.value.reason == "key_host"


def test_the_key_is_one_header_and_never_in_the_url(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini()
    route = gemini_route(srv.base_url(), locality="local")
    c = RealtimeClient(route, settings(), DirectSession(), auth_headers=bearer)
    try:
        assert c.connect(UPDATE)
        snap = c.snapshot()
    finally:
        c.close()
    request = srv.requests[0]
    assert request["path"] == "/ws/BidiGenerateContent"  # no /realtime, no query
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["x-goog-api-key"] == KEY
    assert "authorization" not in headers
    assert snap["key_sent"] is True and snap["wire"] is not None
    assert KEY not in caplog.text


def test_a_keyed_route_without_its_key_never_connects(gemini):
    srv = gemini()
    route = gemini_route(srv.base_url(), locality="local")
    c = RealtimeClient(route, settings(), DirectSession(), auth_headers=lambda: {})
    try:
        assert c.connect(UPDATE) is False
        assert c.reason == "key_unavailable" and srv.requests == []
    finally:
        c.close()


def test_each_session_gets_a_new_codec_whose_ids_never_repeat(gemini):
    srv = gemini([["audio"]])
    events: list = []
    route = gemini_route(srv.base_url(), keyed=False, locality="local")
    c = RealtimeClient(
        route, settings(), DirectSession(), auth_headers=dict, on_event=events.append
    )
    try:
        ids, wires = [], []
        for _ in range(2):
            assert c.connect(UPDATE)
            wires.append(c._wire)
            assert isinstance(c._wire, GeminiWire)
            assert c.send(protocol.text_item("ciao")).result(2)
            assert c.send(protocol.response_create()).result(2)
            assert wait_until(lambda: sum(isinstance(e, protocol.ResponseDone) for e in events)
                              > len(ids))  # fmt: skip
            ids.append(next(e for e in reversed(events) if isinstance(e, protocol.ResponseDone)))
        assert wires[0] is not wires[1]
        assert ids[0].response_id != ids[1].response_id
        assert len(srv.setups) == 2  # every connect sends its own setup
    finally:
        c.close()
