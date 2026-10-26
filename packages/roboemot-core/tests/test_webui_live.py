"""SPEC 11A, 12: ``GET /api/live/state``. PIN session like every page API, off by default
(``live_disabled``), switched on and off from the settings page without a restart, nothing
private in the answer, nothing logged."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.affect import AffectState
from roboemot.config.stack import ConfigStack
from roboemot.conversation import ConversationPhase
from roboemot.diagnostics import Diagnostics
from roboemot.runtime.live import LiveState
from roboemot.webui import make_server
from roboemot.webui.settings import SettingsEditor
from roboemot.world.types import Entry
from test_webui import PIN, Client, FakeService

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
SECRET_ID = "9d41-registry-private-id"
NOW = 100.0


def pipeline():
    people = {SECRET_ID: SimpleNamespace(display_name="Mia", status="active")}
    face = Entry(
        entry_id="entry-5", kind="person", category="face", identity=SECRET_ID,
        azimuth_rad=0.31, elevation_rad=0.0, distance_m=1.4, uncertainty_rad=0.1,
        confidence=0.9, first_seen=0.0, last_seen=NOW, embedding=np.full(64, 0.5),
    )  # fmt: skip
    return SimpleNamespace(
        runner=SimpleNamespace(
            engine=SimpleNamespace(state=lambda now: AffectState(0.1, 0.2, "joy", 0.5))
        ),
        circ=SimpleNamespace(snapshot=lambda: {"state": "awake", "sleepiness": 0.1}),
        sleep=SimpleNamespace(asleep=False, paused_parts=lambda: []),
        network=SimpleNamespace(online=True),
        tracker=SimpleNamespace(active=SimpleNamespace(people=1)),
        selector=SimpleNamespace(target=SimpleNamespace(entry_id="entry-5")),
        perception=SimpleNamespace(registry=SimpleNamespace(get_person=people.__getitem__)),
        world=SimpleNamespace(entries=[face]),
        reflexes=SimpleNamespace(last_fired=lambda now: None),
        loop=SimpleNamespace(motors_off=False, external=False, clock=lambda: NOW),
    )  # fmt: skip


@pytest.fixture
def stack(tmp_path):
    cfg = {"data_dir": str(tmp_path / "data"), "webui": {"bind": "127.0.0.1", "port": 0}}
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def served(tmp_path, stack):
    host = SimpleNamespace(circadian=object(), phase=ConversationPhase())
    live = LiveState(host, pipeline(), stack.effective(), clock=lambda: NOW)
    stack.subscribe(live.apply_config)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, live=live
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    yield SimpleNamespace(client=client, server=srv, host=host, stack=stack)
    srv.stop()


def switch(client, value):
    return client.request(
        "POST", "/api/settings", {"path": "webui.live.enabled", "value": value, "pin": PIN}
    )


def test_it_needs_a_session_like_the_other_page_apis(served):
    anonymous = served.client.request("GET", "/api/live/state")
    assert anonymous.status == 401 and anonymous.json == {"error": "unauthorized"}
    other = served.client.request("GET", "/api/registry")
    assert (other.status, other.json) == (anonymous.status, anonymous.json)


def test_it_is_off_by_default_with_a_stable_code(served):
    assert served.client.login().status == 200
    resp = served.client.request("GET", "/api/live/state")
    assert resp.status == 404 and resp.json == {"error": "live_disabled"}


def test_the_host_without_a_source_answers_unavailable(tmp_path, stack):
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        resp = client.request("GET", "/api/live/state")
        assert resp.status == 503 and resp.json == {"error": "live_unavailable"}
    finally:
        srv.stop()


def test_switching_it_on_from_the_settings_page_needs_no_restart_and_off_again(served):
    c = served.client
    c.login()
    assert c.request("GET", "/api/live/state").status == 404
    on = switch(c, True)
    assert on.status == 200 and on.json["restart_required"] is False
    resp = c.request("GET", "/api/live/state")
    assert resp.status == 200
    assert resp.json["poll_ms"] == 1000
    assert set(resp.json) == {
        "poll_ms", "mood", "modes", "conversation", "attention", "reflexes", "motion", "system",
        "video",
    }  # fmt: skip
    assert resp.json["mood"]["label"] == "joy"
    assert resp.json["attention"]["faces"] == [{"name": "Mia", "target": True}]
    assert switch(c, False).status == 200
    assert c.request("GET", "/api/live/state").json == {"error": "live_disabled"}


def test_the_poll_interval_follows_a_hot_reload(served):
    c = served.client
    c.login()
    switch(c, True)
    r = c.request("POST", "/api/settings", {"path": "webui.live.poll_ms", "value": 400, "pin": PIN})
    assert r.status == 200
    assert c.request("GET", "/api/live/state").json["poll_ms"] == 400


def test_the_phase_transitions_show_in_the_answer(served):
    c = served.client
    c.login()
    switch(c, True)
    phase = served.host.phase

    def conversation():
        return c.request("GET", "/api/live/state").json["conversation"]

    assert conversation()["phase"] == "idle"
    phase.set_listening(True)
    assert conversation()["phase"] == "listening"
    phase.set_listening(False)
    phase.begin_thinking()
    phase.set_route("lan")
    snap = conversation()
    assert snap["phase"] == "thinking" and snap["route"] == "lan" and snap["elapsed_s"] >= 0
    phase.end_thinking()
    assert conversation()["phase"] == "idle"


def test_nothing_private_is_in_the_answer_and_nothing_is_logged(served, caplog):
    caplog.set_level(logging.DEBUG)
    c = served.client
    c.login()
    switch(c, True)
    resp = c.request("GET", "/api/live/state")
    for secret in (SECRET_ID.encode(), b"entry-5", b"0.31", b"1.4", b"embedding", b"azimuth"):
        assert secret not in resp.body, secret
    assert "Mia" not in caplog.text and SECRET_ID not in caplog.text


def test_only_get_is_served(served):
    c = served.client
    c.login()
    switch(c, True)
    assert c.request("POST", "/api/live/state", {}).status == 404


def test_the_response_is_never_cached(served):
    c = served.client
    c.login()
    switch(c, True)
    assert c.request("GET", "/api/live/state").headers["Cache-Control"] == "no-store"


def test_the_tab_strings_exist_in_both_languages_for_every_value_the_state_can_hold():
    from roboemot.conversation.phase import IDLE, LISTENING, SPEAKING, THINKING
    from roboemot.webui.strings import EN, IT

    live = {k for k in EN if k.startswith(("live_", "err_live_"))}
    assert live and live == {k for k in IT if k.startswith(("live_", "err_live_"))}
    needed = [f"live_phase_{p}" for p in (IDLE, LISTENING, THINKING, SPEAKING)]
    needed += [f"live_motion_{s}" for s in ("writing", "external", "motors_off")]
    needed += [f"live_circ_{s}" for s in ("awake", "drowsy", "resting")]
    needed += [f"live_part_{p}" for p in ("app_tick", "vision", "light_reader", "listener")]
    needed += ["err_live_disabled", "err_live_unavailable", "tab_live"]
    for lang in (EN, IT):
        assert [k for k in needed if not lang.get(k)] == []


def test_the_settings_are_in_the_catalogue_hot_reloadable_and_opt_in(stack):
    from roboemot.config.catalogue import build_catalogue

    fields = {f["path"]: f for f in build_catalogue(stack)}
    assert fields["webui.live.enabled"]["opt_in"] and not fields["webui.live.enabled"]["restart"]
    assert fields["webui.live.enabled"]["value"] is False
    for name in ("poll_ms", "max_tracked", "system_cache_s"):
        assert not fields[f"webui.live.{name}"]["restart"]
    poll = fields["webui.live.poll_ms"]
    assert (poll["unit"], poll["ge"], poll["le"]) == ("ms", 250, 10000)


def test_start_webui_keeps_its_live_port_in_step_with_the_stack(tmp_path, stack):
    from contextlib import ExitStack

    from roboemot.manifest import load_manifest
    from roboemot.runtime.webui import start_webui

    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    host = SimpleNamespace(
        perception=SimpleNamespace(enrollment=FakeService()),
        diagnostics=Diagnostics(),
        catalogue=None,
        name_source=None,
        route_check_router=lambda: None,
        circadian=None,
        phase=ConversationPhase(),
        webui=None,
        bus=None,
    )
    with ExitStack() as cleanup:
        server = start_webui(host, stack, stack.effective(), manifest, cleanup, pipeline())
        assert server is not None
        live = server._httpd.api.live
        assert live.state() is None
        stack.set("webui.live.enabled", True, source="test", admin=True)
        assert live.state() is not None
        stack.set("webui.live.enabled", False, source="test", admin=True)
        assert live.state() is None


def test_a_failed_start_leaves_no_listener_behind(tmp_path, stack, monkeypatch):
    from contextlib import ExitStack

    from roboemot.runtime import webui as webui_module

    def boom(*a, **kw):
        raise OSError("port taken")

    monkeypatch.setattr(webui_module, "make_server", boom)
    before = len(stack._listeners)
    host = SimpleNamespace(
        perception=SimpleNamespace(enrollment=None), diagnostics=None, catalogue=None,
        name_source=None, route_check_router=lambda: None, circadian=None,
        phase=ConversationPhase(), webui=None,
    )  # fmt: skip
    manifest = SimpleNamespace(robot=SimpleNamespace(profile=None, id="x"))
    with ExitStack() as cleanup:
        assert (
            webui_module.start_webui(host, stack, stack.effective(), manifest, cleanup, pipeline())
            is None
        )
    assert len(stack._listeners) == before


def test_a_listener_that_raises_does_not_starve_the_next_one(stack, caplog):
    seen = []

    def bad(change):
        raise RuntimeError("secret detail")

    stack.subscribe(bad)
    stack.subscribe(lambda change: seen.append(change.config.webui.live.enabled))
    stack.set("webui.live.enabled", True, source="test", admin=True)
    assert seen == [True]
    assert "RuntimeError" in caplog.text and "secret detail" not in caplog.text


def test_subscribing_twice_keeps_one_entry_and_unsubscribe_removes_it(stack):
    calls = []

    def fn(change):
        calls.append(1)

    off = stack.subscribe(fn)
    stack.subscribe(fn)
    stack.set("webui.live.poll_ms", 400, source="test")
    assert calls == [1]
    off()
    off()
    stack.set("webui.live.poll_ms", 500, source="test")
    assert calls == [1]


def test_start_webui_drops_the_preview_when_the_robot_falls_asleep_and_at_shutdown(tmp_path, stack):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    from contextlib import ExitStack

    from roboemot.bus import Bus
    from roboemot.manifest import load_manifest
    from roboemot.runtime.webui import start_webui
    from roboemot.vision.runner import LatestFrameRunner
    from roboemot.vision.types import Frame

    class Module:
        name = "dummy"

        def process(self, frame):
            pass

    stack.set("webui.live.enabled", True, source="test", admin=True)
    stack.set("webui.live.video", True, source="test", admin=True)
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    runner = LatestFrameRunner(Module())
    service = SimpleNamespace(runner=runner, paused=False)
    faces = SimpleNamespace(
        face_overlay=lambda: [], clear_overlay=lambda: None, matcher=SimpleNamespace(registry=None)
    )
    bus = Bus()
    host = SimpleNamespace(
        perception=SimpleNamespace(enrollment=FakeService(), service=service, pipeline=faces),
        diagnostics=Diagnostics(),
        catalogue=None,
        name_source=None,
        route_check_router=lambda: None,
        circadian=None,
        phase=ConversationPhase(),
        webui=None,
        bus=bus,
        live_video=None,
    )
    with ExitStack() as cleanup:
        parts = pipeline()
        parts.world.get = lambda _id: None
        server = start_webui(host, stack, stack.effective(), manifest, cleanup, parts)
        assert server is not None and host.live_video is server._httpd.api.live.video
        video = host.live_video

        def serve():
            video.frame("10.0.0.1")
            runner.submit(Frame(np.zeros((48, 64, 3), np.uint8), 1.0))
            runner.run_pending()
            return video.frame("10.0.0.1")

        assert serve()[1] is None and video._jpeg is not None and runner.latest() is not None
        bus.publish("robot.asleep", None, 1.0)
        assert video._jpeg is None and runner.latest() is None
        assert video.viewers()["video_viewers_active"] is False
        assert serve()[1] is None
        stack.set("webui.live.video", False, source="test", admin=True)
        assert video._jpeg is None and runner.latest() is None
        stack.set("webui.live.video", True, source="test", admin=True)
        assert serve()[1] is None
    assert video._jpeg is None and runner.latest() is None  # shutdown
