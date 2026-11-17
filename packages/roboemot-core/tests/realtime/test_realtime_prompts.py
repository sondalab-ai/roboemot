"""SPEC 8, 11A, 12: the realtime client's tap into the Live tab's LLM prompts ring. Each frame is
recorded as it left the wire codec (Gemini's ``setup``, OpenAI's ``session.update``), audio is
summarized, never kept; no key, header or URL; a local connect host records nothing.

The fake servers listen on this machine, so their connect target is local: the tests that need
a cloud target say so by patching ``ws_target_local``."""

from __future__ import annotations

import json

import pytest
from fake_server import PCM, FakeServer
from realtime_helpers import KEY, UPDATE, bearer, route, settings, wait_until
from roboemot.config.schema import LiveConfig
from roboemot.llm.prompt_log import PromptRing
from roboemot.realtime import client as client_mod
from roboemot.realtime import protocol
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.session import DirectSession
from test_gemini_client import gemini_route


def ring() -> PromptRing:
    return PromptRing(LiveConfig(enabled=True, prompts=True))


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setattr(client_mod, "ws_target_local", lambda target: False)


def dumped(prompts: PromptRing) -> str:
    return json.dumps(prompts.snapshot(0), ensure_ascii=False)


def test_openai_frames_are_recorded_and_audio_is_summarized(cloud):
    prompts = ring()
    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base()), settings(), DirectSession(), auth_headers=dict,
            prompts=prompts,
        )  # fmt: skip
        try:
            assert c.connect(UPDATE)
            for _ in range(3):
                assert c.send(protocol.audio_append(PCM)).result(2)
            assert c.send(protocol.text_item("ciao")).result(2)
            assert c.send(protocol.function_output("c1", '{"ok": true}')).result(2)
        finally:
            c.close()
    entries = prompts.snapshot(0)["entries"]
    labels = [e["label"] for e in entries]
    item = "conversation.item.create"
    assert labels == ["session.update", "audio", item, item]
    assert {e["route"] for e in entries} == {"rt"} and entries[0]["via"] == "openai_realtime"
    assert json.loads(entries[0]["text"])["session"]["instructions"] == "be kind"
    assert entries[1]["text"] == f"[audio {3 * len(PCM)} bytes in 3 frames]"
    assert "ciao" in entries[2]["text"] and "function_call_output" in entries[3]["text"]
    assert protocol.audio_append(PCM)["audio"] not in dumped(prompts)
    assert str(server.port) not in dumped(prompts)


def test_a_gemini_session_shows_its_setup_and_no_key_or_url(gemini, cloud):
    prompts = ring()
    srv = gemini([["audio"]])
    r = gemini_route(srv.base_url(), locality="local")
    c = RealtimeClient(r, settings(), DirectSession(), auth_headers=bearer, prompts=prompts)
    try:
        assert c.connect(UPDATE)
        for _ in range(2):
            assert c.send(protocol.audio_append(PCM)).result(2)
        assert c.send(protocol.text_item("ciao")).result(2)
        assert c.send(protocol.response_create()).result(2)
        assert wait_until(
            lambda: any(e["label"] == "clientContent" for e in prompts.snapshot(0)["entries"])
        )
    finally:
        c.close()
    entries = prompts.snapshot(0)["entries"]
    setup = entries[0]
    assert (setup["route"], setup["via"], setup["label"]) == ("gemini", "gemini_live", "setup")
    assert "be kind" in json.dumps(json.loads(setup["text"])["setup"]["systemInstruction"])
    audio = [e for e in entries if e["label"] == "audio"]
    assert audio and audio[0]["text"] == f"[audio {2 * len(PCM)} bytes in 2 frames]"
    text = dumped(prompts)
    assert KEY not in text and "x-goog-api-key" not in text.lower()
    assert "127.0.0.1" not in text and "localhost" not in text and "wss://" not in text
    assert protocol.audio_append(PCM)["audio"] not in text


def test_a_local_connect_host_records_nothing():
    prompts = ring()
    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base()), settings(), DirectSession(), auth_headers=dict,
            prompts=prompts,
        )  # fmt: skip
        try:
            assert c.connect(UPDATE)
            assert c.send(protocol.text_item("ciao")).result(2)
        finally:
            c.close()
    assert prompts.size == 0


def test_with_the_flag_off_nothing_is_recorded(cloud):
    prompts = PromptRing(LiveConfig(enabled=True, prompts=False))
    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base()), settings(), DirectSession(), auth_headers=dict,
            prompts=prompts,
        )  # fmt: skip
        try:
            assert c.connect(UPDATE)
            assert c.send(protocol.text_item("ciao")).result(2)
        finally:
            c.close()
    assert prompts.size == 0 and prompts.snapshot(0)["rev"] == 0


def test_a_ring_that_fails_never_fails_a_send(cloud):
    class Broken(PromptRing):
        def frame(self, *a, **kw):
            raise RuntimeError("boom")

    with FakeServer() as server:
        c = RealtimeClient(
            route(server.http_base()), settings(), DirectSession(), auth_headers=dict,
            prompts=Broken(LiveConfig(enabled=True, prompts=True)),
        )  # fmt: skip
        try:
            assert c.connect(UPDATE)
            assert c.send(protocol.text_item("ciao")).result(2)
        finally:
            c.close()
