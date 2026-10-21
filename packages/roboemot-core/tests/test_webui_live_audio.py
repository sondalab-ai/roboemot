"""SPEC 11A, 12: ``GET /api/live/audio`` (levels and the live transcript behind a since-cursor) and
``POST /api/live/transcript/clear``. PIN session like every page API, CSRF on the POST, the same
disabled codes as the state endpoint, nothing logged, nothing stored."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from roboemot.config.stack import ConfigStack
from roboemot.conversation import ConversationPhase, TranscriptRing
from roboemot.diagnostics import Diagnostics
from roboemot.runtime.levels import AudioLevels
from roboemot.runtime.live import LiveState
from roboemot.webui import make_server
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, Client, FakeService
from test_webui_live import NOW, pipeline, switch

SECRET = "tokenSECRETapi"


@pytest.fixture
def stack(tmp_path):
    cfg = {"data_dir": str(tmp_path / "data"), "webui": {"bind": "127.0.0.1", "port": 0}}
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def served(tmp_path, stack):
    live_cfg = stack.effective().webui.live
    levels = AudioLevels(lambda rms, stamp: None, clock=lambda: NOW)
    levels.noise_floor = lambda: 0.01
    ring = TranscriptRing(live_cfg)
    host = SimpleNamespace(
        circadian=object(), minor_present=False, phase=ConversationPhase(),
        levels=levels, transcript=ring,
    )  # fmt: skip
    parts = pipeline()
    live = LiveState(host, parts, stack.effective(), clock=lambda: NOW)

    def on_change(change):
        live.apply_config(change)
        ring.apply_config(change.config.webui.live)

    stack.subscribe(on_change)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, live=live
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    yield SimpleNamespace(
        client=client, server=srv, host=host, stack=stack, levels=levels, parts=parts
    )
    srv.stop()


def setting(client, path, value):
    r = client.request("POST", "/api/settings", {"path": path, "value": value, "pin": PIN})
    assert r.status == 200, r.json
    return r


def audio(client, after=None):
    q = "" if after is None else f"?after={after}"
    return client.request("GET", "/api/live/audio" + q)


def test_it_needs_a_session_like_the_other_page_apis(served):
    anonymous = audio(served.client)
    assert anonymous.status == 401 and anonymous.json == {"error": "unauthorized"}
    post = served.client.request("POST", "/api/live/transcript/clear")
    assert post.status == 401


def test_it_is_off_with_the_live_tab_and_the_clear_too(served):
    served.client.login()
    assert audio(served.client).json == {"error": "live_disabled"}
    r = served.client.request("POST", "/api/live/transcript/clear", {})
    assert r.status == 404 and r.json == {"error": "live_disabled"}


def test_the_host_without_a_source_answers_unavailable(tmp_path, stack):
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert audio(client).json == {"error": "live_unavailable"}
        assert client.request("POST", "/api/live/transcript/clear", {}).status == 503
    finally:
        srv.stop()


def test_the_levels_and_the_idle_transcript_block(served):
    c = served.client
    c.login()
    switch(c, True)
    served.levels.offer(0.1, NOW - 0.1)
    body = audio(c).json
    assert set(body) == {"audio_poll_ms", "levels", "transcript"}
    assert body["audio_poll_ms"] == 500
    assert body["levels"] == {
        "mic": {"state": "ok", "db": -20.0},
        "out": {"state": "unavailable", "db": None},
        "floor_db": -40.0,
    }
    assert body["transcript"]["enabled"] is False and body["transcript"]["turns"] == []


def test_asleep_reports_paused_and_reads_no_level(served):
    c = served.client
    c.login()
    switch(c, True)
    served.levels.offer(0.1, NOW)
    served.parts.sleep.asleep = True
    try:
        body = audio(c).json["levels"]
    finally:
        served.parts.sleep.asleep = False
    assert body["mic"] == {"state": "paused", "db": None}
    assert body["out"]["state"] == "paused" and body["floor_db"] is None


def test_a_paused_listener_reports_paused(served):
    c = served.client
    c.login()
    switch(c, True)
    served.parts.sleep.paused_parts = lambda: ["listener"]
    assert audio(c).json["levels"]["mic"]["state"] == "paused"


def test_transcript_is_off_until_its_own_flag_and_keeps_nothing_before(served):
    c = served.client
    c.login()
    switch(c, True)
    assert served.host.transcript.add("user", SECRET, "en") is None
    setting(c, "webui.live.transcript", True)
    seq = served.host.transcript.add("user", SECRET, "en")
    body = audio(c).json["transcript"]
    assert body["enabled"] is True and [t["text"] for t in body["turns"]] == [SECRET]
    assert body["seq"] == seq and body["max_turns"] == 50


def test_the_since_cursor_returns_only_new_turns(served):
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    ring = served.host.transcript
    first = ring.add("user", "one", "en")
    ring.add("robot", "two", "en")
    assert [t["text"] for t in audio(c).json["transcript"]["turns"]] == ["one", "two"]
    after = audio(c, first).json["transcript"]
    assert [t["text"] for t in after["turns"]] == ["two"]
    assert audio(c, after["seq"]).json["transcript"]["turns"] == []


@pytest.mark.parametrize("bad", ["abc", "-1", "1.5", "", "99999999999999999999999", "1&after=2"])
def test_a_bad_cursor_is_a_clean_400(served, bad):
    c = served.client
    c.login()
    switch(c, True)
    r = c.request("GET", "/api/live/audio?after=" + bad)
    assert r.status == 400 and r.json == {"error": "bad_request"}


def test_clear_needs_csrf_and_empties_the_ring_and_changes_the_epoch(served):
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    served.host.transcript.add("user", SECRET, "en")
    epoch = audio(c).json["transcript"]["epoch"]
    no_csrf = c.request("POST", "/api/live/transcript/clear", {}, csrf=False)
    assert no_csrf.status == 403 and served.host.transcript.size == 1
    ok = c.request("POST", "/api/live/transcript/clear", {})
    assert ok.status == 200 and ok.json == {"ok": True}
    body = audio(c).json["transcript"]
    assert body["turns"] == [] and body["epoch"] != epoch


def test_get_on_the_clear_route_and_post_on_audio_are_not_served(served):
    c = served.client
    c.login()
    switch(c, True)
    assert c.request("GET", "/api/live/transcript/clear").status == 404
    assert c.request("POST", "/api/live/audio", {}).status == 404


def test_logging_out_does_not_clear_the_transcript(served):
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    served.host.transcript.add("user", SECRET, "en")
    c.request("POST", "/api/logout", {})
    again = Client(served.server.port)
    again.login()
    assert [t["text"] for t in audio(again).json["transcript"]["turns"]] == [SECRET]


def test_turning_the_flag_off_forgets_what_was_kept(served):
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    served.host.transcript.add("user", SECRET, "en")
    setting(c, "webui.live.transcript", False)
    setting(c, "webui.live.transcript", True)
    assert audio(c).json["transcript"]["turns"] == []


def test_the_transcript_is_not_in_the_state_endpoint_or_the_debug_report(served):
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    served.host.transcript.add("user", SECRET, "en")
    assert SECRET not in c.request("GET", "/api/live/state").body.decode()
    assert SECRET not in c.request("GET", "/api/debug").body.decode()


def test_no_transcript_text_is_logged_by_the_server(served, caplog):
    caplog.set_level(logging.DEBUG)
    c = served.client
    c.login()
    switch(c, True)
    setting(c, "webui.live.transcript", True)
    served.host.transcript.add("user", SECRET, "en")
    audio(c)
    c.request("POST", "/api/live/transcript/clear", {})
    assert SECRET not in caplog.text


def test_the_responses_are_not_cacheable(served):
    c = served.client
    c.login()
    switch(c, True)
    assert audio(c).headers["Cache-Control"] == "no-store"


def test_the_settings_page_lists_the_new_fields_with_the_flag_as_an_opt_in(served):
    c = served.client
    c.login()
    fields = {f["path"]: f for f in c.request("GET", "/api/settings").json["fields"]}
    assert fields["webui.live.transcript"]["opt_in"] is True
    assert fields["webui.live.transcript"]["default"] is False
    for name in ("audio_poll_ms", "transcript_max_turns", "transcript_max_age_s"):
        assert f"webui.live.{name}" in fields
