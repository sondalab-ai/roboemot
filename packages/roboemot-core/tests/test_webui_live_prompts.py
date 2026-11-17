"""SPEC 11A, 12: ``GET /api/live/prompts`` (the cloud LLM requests behind a since-cursor) and
``POST /api/live/prompts/clear``. A PIN session like ``/api/live/audio``, CSRF on the POST, the
same disabled codes, an off block while ``webui.live.prompts`` is off, nothing logged."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from roboemot.conversation import ConversationPhase
from roboemot.diagnostics import Diagnostics
from roboemot.llm.prompt_log import PromptRing
from roboemot.runtime.live import NO_PROMPTS, LiveState
from roboemot.webui import make_server
from roboemot.webui.api import PASSIVE_POLLS
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, Client, FakeService
from test_webui_live import NOW, pipeline, switch
from test_webui_live_audio import setting, stack  # noqa: F401  (fixture)

SECRET = "promptSECRETsystem"


@pytest.fixture
def served(tmp_path, stack):  # noqa: F811
    ring = PromptRing(stack.effective().webui.live)
    ring.attach(stack)
    host = SimpleNamespace(
        circadian=SimpleNamespace(state="awake", reason=None), phase=ConversationPhase(),
        activity=None, realtime_route=None, voice_listener=None, dance=None, levels=None,
        transcript=None, prompts=ring,
    )  # fmt: skip
    live = LiveState(host, pipeline(), stack.effective(), clock=lambda: NOW)
    stack.subscribe(live.apply_config)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, live=live
    )
    srv.set_pin(PIN)
    srv.start()
    yield SimpleNamespace(client=Client(srv.port), server=srv, ring=ring)
    srv.stop()


def prompts(client, query=""):
    return client.request("GET", "/api/live/prompts" + query)


def record(ring, text=SECRET):
    ring.chat("cloud", "gpt-x", [{"role": "system", "content": text}], None)


def on(c):
    c.login()
    switch(c, True)
    setting(c, "webui.live.prompts", True)


def test_it_needs_a_session_like_the_audio_poll(served):
    assert prompts(served.client).status == 401
    assert prompts(served.client).json == {"error": "unauthorized"}
    assert served.client.request("POST", "/api/live/prompts/clear").status == 401
    assert "/api/live/prompts" in PASSIVE_POLLS


def test_it_is_off_with_the_live_tab_and_the_clear_too(served):
    served.client.login()
    assert prompts(served.client).json == {"error": "live_disabled"}
    r = served.client.request("POST", "/api/live/prompts/clear", {})
    assert r.status == 404 and r.json == {"error": "live_disabled"}


def test_the_host_without_a_source_answers_unavailable(tmp_path, stack):  # noqa: F811
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert prompts(client).status == 503
        assert prompts(client).json == {"error": "live_unavailable"}
        assert client.request("POST", "/api/live/prompts/clear", {}).status == 503
    finally:
        srv.stop()


def test_the_flag_off_is_an_off_block_and_nothing_is_kept(served):
    c = served.client
    c.login()
    switch(c, True)
    record(served.ring)
    body = prompts(c).json
    assert body == {"poll_ms": 500, "prompts": NO_PROMPTS}
    assert served.ring.size == 0


def test_the_cursor_returns_only_new_entries(served):
    c = served.client
    on(c)
    record(served.ring, "one")
    first = prompts(c).json["prompts"]
    assert first["enabled"] is True and [e["text"] for e in first["entries"]] == ["[system] one"]
    record(served.ring, "two")
    after = prompts(c, f"?after={first['rev']}&epoch={first['epoch']}").json["prompts"]
    assert [e["text"] for e in after["entries"]] == ["[system] two"]
    other = prompts(c, f"?after={after['rev']}&epoch=0123abcd.9").json["prompts"]
    assert len(other["entries"]) == 2


@pytest.mark.parametrize("bad", ["after=abc", "after=-1", "epoch=zz", "after=1&after=2"])
def test_a_bad_cursor_is_a_clean_400(served, bad):
    c = served.client
    on(c)
    r = prompts(c, "?" + bad)
    assert r.status == 400 and r.json == {"error": "bad_request"}


def test_clear_needs_csrf_and_empties_the_ring(served):
    c = served.client
    on(c)
    record(served.ring)
    epoch = prompts(c).json["prompts"]["epoch"]
    no_csrf = c.request("POST", "/api/live/prompts/clear", {}, csrf=False)
    assert no_csrf.status == 403 and served.ring.size == 1
    ok = c.request("POST", "/api/live/prompts/clear", {})
    assert ok.status == 200 and ok.json == {"ok": True}
    body = prompts(c).json["prompts"]
    assert body["entries"] == [] and body["epoch"] != epoch


def test_turning_the_flag_off_forgets_what_was_kept(served):
    c = served.client
    on(c)
    record(served.ring)
    setting(c, "webui.live.prompts", False)
    assert served.ring.size == 0
    setting(c, "webui.live.prompts", True)
    assert prompts(c).json["prompts"]["entries"] == []


def test_the_prompts_are_not_in_the_state_or_the_debug_report_or_the_log(served, caplog):
    caplog.set_level(logging.DEBUG)
    c = served.client
    on(c)
    record(served.ring)
    assert SECRET in prompts(c).body.decode()
    assert SECRET not in c.request("GET", "/api/live/state").body.decode()
    assert SECRET not in c.request("GET", "/api/debug").body.decode()
    c.request("POST", "/api/live/prompts/clear", {})
    assert SECRET not in caplog.text


def test_a_frame_with_long_number_lists_passes_the_vector_guard(served):
    c = served.client
    on(c)
    served.ring.frame("gem", "gemini_live", {"setup": {"x": list(range(200))}})
    r = prompts(c)
    assert r.status == 200 and r.json["prompts"]["entries"][0]["label"] == "setup"
    assert '"x": "[200 numbers]"' in r.json["prompts"]["entries"][0]["text"]


def test_the_response_is_not_cacheable(served):
    c = served.client
    on(c)
    assert prompts(c).headers["Cache-Control"] == "no-store"


def test_the_settings_page_lists_the_flag_as_an_opt_in(served):
    c = served.client
    c.login()
    fields = {f["path"]: f for f in c.request("GET", "/api/settings").json["fields"]}
    assert fields["webui.live.prompts"]["opt_in"] is True
    assert fields["webui.live.prompts"]["default"] is False
    for name in ("prompts_max_entries", "prompts_max_chars", "prompts_response_max_chars"):
        assert f"webui.live.{name}" in fields
