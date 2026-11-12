"""The client and the URL rules for a ``gemini_live`` route (SPEC 8, Gemini Live wire; SPEC 12):
the URL as written, the key in ``x-goog-api-key`` only and only to the route's own origin, and
a codec per session whose ids never repeat (the route keys its charges by response id)."""

from __future__ import annotations

import base64
import json
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


def test_a_gemini_key_never_goes_to_a_providers_declared_host():
    """A platform allocation whose key hosts match still gets no Gemini key: it has no
    allocator, so its key goes to the route's own origin only."""
    allocation = Allocation("wss://sessions.example.com/ws", (".example.com",))
    with pytest.raises(UrlRefused) as refused:
        connect_target(gemini_route(), allocation, (".example.com",))
    assert refused.value.reason == "key_host"
    pollen = LLMRoute(
        name="p",
        base_url="https://h.example/v1",
        model="m",
        kind="realtime",
        api_key="${secret:k}",
    )
    assert connect_target(pollen, allocation, (".example.com",)).send_key  # unchanged


def test_the_real_skill_schemas_go_whole_into_the_setup():
    from roboemot.realtime.turn import skip_tool, tone_tool
    from roboemot.skills.registry import SkillRegistry
    from test_conversation_tools import DEFAULTS, Env

    registry = Env().registry
    chat_tools = SkillRegistry.tools(registry.offered(DEFAULTS))
    tools = [*protocol.function_tools(chat_tools), skip_tool(), tone_tool(["happy", "sad"])]
    update = protocol.session_update(
        instructions="be kind",
        input_rate_hz=None,
        output_rate_hz=None,
        voice="Kore",
        vad_threshold=0.5,
        silence_ms=500,
        create_response=False,
        interrupt_response=False,
        tools=tools,
    )
    frames, local = GeminiWire("gemini-3.8-live", 1 << 20).encode(update)
    declared = frames[0]["setup"]["tools"][0]["functionDeclarations"]
    assert len(chat_tools) == len(DEFAULTS) and local == []
    assert [d["name"] for d in declared] == [t["name"] for t in tools]
    for d, t in zip(declared, tools, strict=True):
        assert d["parametersJsonSchema"] == t["parameters"] and d["behavior"] == "BLOCKING"
        assert "parameters" not in d  # the OpenAPI subset field is never used
    json.dumps(frames)  # JSON as sent


def test_the_key_is_in_no_repr_snapshot_or_failure(gemini, caplog):
    caplog.set_level(logging.DEBUG)
    srv = gemini(refuse=1)
    route = gemini_route(srv.base_url(), locality="local")
    c = RealtimeClient(route, settings(), DirectSession(), auth_headers=bearer)
    try:
        assert c.connect(UPDATE) is False  # closed with 1008 during the connect
        assert c.close_code == 1008 and c.reason == "connect_failed"
        assert KEY not in repr(c) and KEY not in json.dumps(c.snapshot())
        assert c.connect(UPDATE)
        assert KEY not in repr(c) and KEY not in json.dumps(c.snapshot())
    finally:
        c.close()
    assert KEY not in caplog.text
    for record in caplog.records:
        assert record.exc_info is None or KEY not in str(record.exc_info[1])


def test_usage_reports_never_name_a_response_twice():
    """The codec's guarantee the route relies on (it skips only a report naming responses all
    reported before): across a tool round and a late turnComplete, every ended response is
    named by exactly one report."""
    wire = GeminiWire("gemini-3.8-live", 1 << 20)
    wire.encode(UPDATE)

    def server(message: dict) -> list:
        return wire.decode(json.dumps(message).encode())

    pcm = base64.b64encode(b"\x00\x00" * 240).decode()
    part = {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": pcm}}
    audio = {"serverContent": {"modelTurn": {"parts": [part]}}}
    done = {"serverContent": {"generationComplete": True}}
    usage = {"serverContent": {"turnComplete": True}, "usageMetadata": {"totalTokenCount": 9}}
    events = server({"setupComplete": {}})
    for turn in range(3):
        wire.encode(protocol.text_item(f"t{turn}"))
        events += wire.encode(protocol.response_create())[1]
        if turn == 1:  # a tool round: the call ends one response, Gemini continues in another
            events += server({"toolCall": {"functionCalls": [{"id": "c1", "name": "play"}]}})
            wire.encode(protocol.function_output("c1", "{}"))
            events += wire.encode(protocol.response_create())[1]
        events += server(audio) + server(done)
        if turn:  # the first turn's usage comes after the second turn ended
            events += server(usage)
    events += server(usage)
    reports = [e for e in events if isinstance(e, protocol.UsageReport)]
    named = [i for r in reports for i in r.response_ids]
    assert len(reports) == 3 and len(named) == len(set(named))
    ended = {e.response_id for e in events if isinstance(e, protocol.ResponseDone)}
    assert set(named) == ended and len(ended) == 4
