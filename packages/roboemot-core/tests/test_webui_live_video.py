"""SPEC 11A, 12: ``GET /api/live/frame.jpg`` and ``GET /api/live/viewers``. PIN session like every
page API; one JPEG with the security headers; a stable JSON code and no image for every state the
picture cannot be served; the banner flag; img-src allows blob: and nothing more; no image bytes
in the logs."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.stack import ConfigStack
from roboemot.conversation import ConversationPhase
from roboemot.diagnostics import Diagnostics
from roboemot.runtime.live import LiveState
from roboemot.vision.runner import LatestFrameRunner
from roboemot.vision.types import Frame
from roboemot.webui import make_server
from roboemot.webui.settings import SettingsEditor
from test_webui import PIN, Client, FakeService

NOW = 100.0


class Module:
    name = "dummy"

    def process(self, frame):
        pass


def pipeline():
    return SimpleNamespace(
        runner=SimpleNamespace(),
        sleep=SimpleNamespace(asleep=False, paused_parts=lambda: []),
        selector=SimpleNamespace(target=None),
        world=SimpleNamespace(entries=[], get=lambda _id: None),
        loop=SimpleNamespace(motors_off=False, external=False, clock=lambda: NOW),
    )


@pytest.fixture
def stack(tmp_path):
    cfg = {"data_dir": str(tmp_path / "data"), "webui": {"bind": "127.0.0.1", "port": 0}}
    return ConfigStack(app=cfg, audit_path=tmp_path / "audit.jsonl")


@pytest.fixture
def served(tmp_path, stack):
    runner = LatestFrameRunner(Module())
    service = SimpleNamespace(runner=runner, paused=False)
    faces = SimpleNamespace(face_overlay=lambda: [], matcher=SimpleNamespace(registry=None))
    host = SimpleNamespace(
        circadian=None,
        minor_present=False,
        phase=ConversationPhase(),
        perception=SimpleNamespace(service=service, pipeline=faces),
    )
    p = pipeline()
    live = LiveState(host, p, stack.effective(), clock=lambda: NOW)
    stack.subscribe(live.apply_config)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, live=live
    )
    srv.set_pin(PIN)
    srv.start()
    client = Client(srv.port)
    yield SimpleNamespace(client=client, server=srv, runner=runner, service=service, p=p)
    srv.stop()


def set_flag(client, name, value):
    r = client.request(
        "POST", "/api/settings", {"path": f"webui.live.{name}", "value": value, "pin": PIN}
    )
    assert r.status == 200, r.body
    return r


def enable(c):
    c.login()
    set_flag(c, "enabled", True)
    set_flag(c, "video", True)


def feed(served):
    served.runner.submit(Frame(np.full((480, 640, 3), 90, np.uint8), 1.0))


def test_it_needs_a_session_like_the_other_page_apis(served):
    anonymous = served.client.request("GET", "/api/live/frame.jpg")
    assert anonymous.status == 401 and anonymous.json == {"error": "unauthorized"}
    viewers = served.client.request("GET", "/api/live/viewers")
    assert viewers.status == 401 and viewers.json == {"error": "unauthorized"}


def test_each_unavailable_state_answers_a_stable_code_and_no_image(served):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    c = served.client
    c.login()
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (404, {"error": "live_disabled"})
    set_flag(c, "enabled", True)
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (404, {"error": "video_off"})
    set_flag(c, "video", True)
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (503, {"error": "video_no_frame"})
    served.p.sleep.asleep = True
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (503, {"error": "video_asleep"})
    served.p.sleep.asleep = False
    served.service.paused = True
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (503, {"error": "video_asleep"})
    served.service.paused = False
    assert b"\xff\xd8" not in c.request("GET", "/api/live/frame.jpg").body


def test_no_vision_answers_its_own_code(served):
    c = served.client
    enable(c)
    served.server._httpd.api.live._host.perception.service = None
    r = c.request("GET", "/api/live/frame.jpg")
    assert (r.status, r.json) == (503, {"error": "video_no_vision"})


def test_it_serves_one_jpeg_with_the_security_headers(served):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    c = served.client
    enable(c)
    assert c.request("GET", "/api/live/frame.jpg").status == 503  # starts watching
    feed(served)
    r = c.request("GET", "/api/live/frame.jpg")
    assert r.status == 200 and r.body[:2] == b"\xff\xd8" and r.body[-2:] == b"\xff\xd9"
    assert r.headers["Content-Type"] == "image/jpeg"
    assert r.headers["Cache-Control"] == "no-store"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert int(r.headers["Content-Length"]) == len(r.body)
    csp = r.headers["Content-Security-Policy"]
    assert "img-src 'self' blob:;" in csp and "data:" not in csp
    assert "script-src 'self';" in csp and "blob:" not in csp.replace("img-src 'self' blob:", "")


def test_post_is_not_served(served):
    c = served.client
    enable(c)
    assert c.request("POST", "/api/live/frame.jpg", {}).status in (404, 405)


def test_the_banner_flag_follows_who_was_sent_a_frame(served):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    c = served.client
    enable(c)
    assert c.request("GET", "/api/live/viewers").json == {
        "video_viewers_active": False, "count": 0, "poll_ms": 3000,
    }  # fmt: skip
    c.request("GET", "/api/live/frame.jpg")
    feed(served)
    assert c.request("GET", "/api/live/frame.jpg").status == 200
    assert c.request("GET", "/api/live/viewers").json == {
        "video_viewers_active": True,
        "count": 1,
        "poll_ms": 3000,
    }
    set_flag(c, "video", False)
    assert c.request("GET", "/api/live/viewers").json["video_viewers_active"] is False


def test_viewers_is_inactive_by_default_and_the_banner_poll_follows_its_setting(served):
    c = served.client
    c.login()
    assert c.request("GET", "/api/live/viewers").json == {
        "video_viewers_active": False,
        "count": 0,
        "poll_ms": 3000,
    }
    set_flag(c, "banner_poll_ms", 1500)
    assert c.request("GET", "/api/live/viewers").json["poll_ms"] == 1500


def test_without_a_live_source_the_banner_poll_still_answers(tmp_path, stack):
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        assert client.request("GET", "/api/live/viewers").json["video_viewers_active"] is False
        r = client.request("GET", "/api/live/frame.jpg")
        assert (r.status, r.json) == (503, {"error": "live_unavailable"})
    finally:
        srv.stop()


def test_the_video_settings_are_in_the_catalogue_with_units_and_ranges(served):
    from roboemot.config.schema import LiveConfig

    fields = LiveConfig.model_fields
    extra = {n: fields[n].json_schema_extra for n in fields}
    assert extra["video"]["opt_in"] is True and fields["video"].default is False
    assert extra["video_max_width"]["unit"] == "px" and fields["video_max_width"].default == 320
    assert extra["video_fps"]["unit"] == "hz" and 2 <= fields["video_fps"].default <= 5
    assert fields["video_jpeg_quality"].default == 60
    assert extra["viewer_window_s"]["unit"] == "s"
    for name in ("video_fps", "video_max_width", "video_jpeg_quality", "viewer_window_s"):
        metadata = {type(m).__name__ for m in fields[name].metadata}
        assert {"Ge", "Le"} <= metadata, name


def test_logs_hold_the_client_address_and_no_image(served, caplog):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    caplog.set_level(logging.DEBUG)
    c = served.client
    enable(c)
    c.request("GET", "/api/live/frame.jpg")
    feed(served)
    jpeg = c.request("GET", "/api/live/frame.jpg").body
    text = "\n".join(f"{r.getMessage()} {r.args!r}" for r in caplog.records)
    assert "live video view started (127.0.0.1)" in text
    assert "\\xff\\xd8" not in text and jpeg[:8].decode("latin-1") not in text


def test_the_page_polls_the_frame_and_the_banner_with_the_strings_in_both_languages():
    from roboemot.webui.strings import EN, IT

    keys = [
        "live_video", "live_video_alt", "live_video_pause", "live_video_resume",
        "live_video_paused", "live_video_note", "live_video_off", "live_video_banner",
        "err_video_off", "err_video_asleep", "err_video_no_vision", "err_video_no_frame",
    ]  # fmt: skip
    for key in keys:
        assert EN[key] and IT[key], key
    assert EN["live_video_banner"] == "Live video view active"
    static = Path(__file__).resolve().parents[1] / "src/roboemot/webui/static"
    html = (static / "index.html").read_text()
    assert 'id="live-video-banner"' in html and "sl-callout" in html


def test_the_banner_poll_does_not_extend_an_idle_session(tmp_path, stack):
    class Clock:
        now = 1000.0

        def __call__(self):
            return self.now

    clock = Clock()
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path, clock=clock)
    srv.set_pin(PIN)
    srv.start()
    try:
        client = Client(srv.port)
        client.login()
        ttl = stack.effective().webui.session_ttl_s
        clock.now += ttl - 10
        assert (
            client.request("GET", "/api/live/viewers").status == 200
        )  # still valid, not refreshed
        clock.now += 20
        assert client.request("GET", "/api/live/viewers").status == 401  # idle since the login
        client.login()
        clock.now += ttl - 10
        assert client.request("GET", "/api/registry").status != 401  # a real call refreshes
        clock.now += ttl - 10
        assert client.request("GET", "/api/live/viewers").status == 200
    finally:
        srv.stop()


def test_the_viewer_window_must_cover_two_polls():
    from pydantic import ValidationError
    from roboemot.config.schema import LiveConfig

    LiveConfig(video_fps=0.5, viewer_window_s=4, banner_poll_ms=2000)
    with pytest.raises(ValidationError):
        LiveConfig(video_fps=0.5, viewer_window_s=3, banner_poll_ms=1000)
    with pytest.raises(ValidationError):
        LiveConfig(video_fps=0.5, viewer_window_s=1, banner_poll_ms=1000)


def test_a_banner_poll_slower_than_half_the_window_is_refused_and_capped():
    from pydantic import ValidationError
    from roboemot.config.schema import LiveConfig

    LiveConfig(viewer_window_s=6, banner_poll_ms=3000)
    for window, banner in ((1, 1000), (6, 3500), (2, 1500)):
        with pytest.raises(ValidationError):
            LiveConfig(viewer_window_s=window, banner_poll_ms=banner)
    with pytest.raises(ValidationError):
        LiveConfig(viewer_window_s=60, banner_poll_ms=30000)  # the field is capped at 5 s
    probe = dict(viewer_window_s=1, video_fps=2.0, banner_poll_ms=30000)
    with pytest.raises(ValidationError):  # the reviewer's combination
        LiveConfig(**probe)


def test_a_cross_site_request_is_refused_before_the_handler(served):
    pytest.importorskip("cv2")  # encodes a JPEG: the vision extra (python-vision CI job)
    c = served.client
    enable(c)
    c.request("GET", "/api/live/frame.jpg")
    feed(served)
    for site in ("cross-site", "same-site"):
        r = c.request("GET", "/api/live/frame.jpg", headers={"Sec-Fetch-Site": site})
        assert (r.status, r.json) == (403, {"error": "cross_site"})
    assert c.request("GET", "/api/live/viewers").json["video_viewers_active"] is False
    for site in ("same-origin", "none"):
        r = c.request("GET", "/api/live/frame.jpg", headers={"Sec-Fetch-Site": site})
        assert r.status == 200
    assert r.headers["Cross-Origin-Resource-Policy"] == "same-origin"


def test_the_new_video_settings_have_units_and_ranges():
    from roboemot.config.schema import LiveConfig

    f = LiveConfig.model_fields
    assert (
        f["video_stale_s"].json_schema_extra["unit"] == "s" and f["video_stale_s"].default == 10.0
    )
    assert f["video_idle_stop_s"].json_schema_extra["unit"] == "s"
    assert f["video_idle_stop_s"].default == 600.0
    assert f["banner_poll_ms"].json_schema_extra["unit"] == "ms"
    assert max(m.le for m in f["banner_poll_ms"].metadata if hasattr(m, "le")) == 5000
