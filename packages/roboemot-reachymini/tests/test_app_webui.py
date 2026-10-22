"""The web UI served by the app (SPEC 4A, 6, 12, 12A): real EnrollmentService, fake vision."""

from __future__ import annotations

import http.client
import io
import json
import socket
import threading

import pytest
import yaml
from roboemot.webui.__main__ import main as set_pin
from roboemot_reachymini import app as app_module
from test_app_perception import (
    AppWithFakes,
    data,  # noqa: F401  (fixture)
    new_app,
    start,
    wait_for,
)

PIN = "246810"
FORM = {
    "display_name": "Ada",
    "consent_who": "Ada",
    "scope": ["face", "refresh"],
    "age_group": "adult",
    "gender": "female",
    "agreed": True,
}


@pytest.fixture()
def webui_data(data, tmp_path):  # noqa: F811
    """App config with relaxed (untuned) enrollment checks, an ephemeral port and a PIN."""
    cfg = yaml.safe_load((data / "app.yaml").read_text())
    cfg["webui"] = {"port": 0}
    cfg["enrollment"] = {"min_sharpness": 0, "min_pose_spread": 0, "min_gap_s": 0}
    (data / "app.yaml").write_text(yaml.safe_dump(cfg))
    folder = tmp_path / "roboemot-data"
    folder.mkdir(mode=0o700)
    assert set_pin(["set-pin", "--data-dir", str(folder)], stdin=io.StringIO(PIN + "\n")) == 0
    return folder


class Browser:
    def __init__(self, port: int) -> None:
        self.port, self.cookie, self.csrf = port, None, None

    def call(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        if self.cookie:
            headers["Cookie"] = self.cookie
        if method == "POST":
            headers["Content-Type"] = "application/json"
            if self.csrf:
                headers["X-CSRF-Token"] = self.csrf
        conn.request(method, path, json.dumps(body).encode() if body is not None else None, headers)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        parsed = json.loads(raw) if "json" in resp.getheader("Content-Type", "") else raw
        return resp.status, parsed, resp

    def login(self):
        status, body, resp = self.call("POST", "/api/login", {"pin": PIN})
        if status == 200:
            self.cookie = resp.getheader("Set-Cookie").split(";")[0]
            self.csrf = body["csrf"]
        return status


def test_the_app_declares_its_page_and_leaves_the_server_to_itself():
    cls = app_module.RoboEmotApp
    assert cls.custom_app_url == "http://0.0.0.0:8042"
    assert cls.dont_start_webserver is True


def test_enroll_through_the_web_ui_against_the_real_service(webui_data):
    app = new_app(AppWithFakes)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.webui is not None and app.perception is not None)
        port = app.webui.port
        assert app.webui.host == "0.0.0.0"  # the cm4 profile: reachable on the LAN
        page = Browser(port)
        assert page.call("GET", "/")[0] == 200
        assert page.call("GET", "/api/registry")[0] == 401
        assert page.login() == 200

        assert page.call("POST", "/api/enrollment/start", {})[0] == 200
        ready = []

        def can_complete():
            status, body, _ = page.call("GET", "/api/enrollment/status")
            ready.append(body)
            return status == 200 and body["can_complete"]

        assert wait_for(can_complete), ready[-1]
        body = ready[-1]
        assert body["frames_collected"] >= body["frames_needed"] or body["can_complete"]
        assert set(body) == {"state", "frames_collected", "frames_needed", "can_complete", "hints"}

        status, data, _ = page.call("POST", "/api/enrollment/complete", FORM)
        assert status == 200, data
        status, data, _ = page.call("GET", "/api/registry")
        assert [(r["display_name"], r["kind"]) for r in data["records"]] == [("Ada", "person")]
        status, report, _ = page.call("GET", "/api/debug")
        assert status == 200 and report["sections"]["roboemot.registry"]["ok"]
        assert "Ada" not in json.dumps(report)

        person = data["records"][0]["id"]
        forget = f"/api/registry/{person}/forget"
        assert page.call("POST", forget, {"confirm": True})[0] == 400  # PIN again
        assert page.call("POST", forget, {"confirm": True, "pin": PIN})[0] == 200
        assert page.call("GET", "/api/registry")[1]["records"] == []
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
    with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=2):
        pass
    assert not [t for t in threading.enumerate() if t.name == "roboemot-webui"]


def test_app_runs_without_the_page_when_the_port_is_taken(webui_data, data):  # noqa: F811
    cfg = yaml.safe_load((data / "app.yaml").read_text())
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        taken.listen()
        cfg["webui"] = {"port": taken.getsockname()[1]}
        (data / "app.yaml").write_text(yaml.safe_dump(cfg))
        app = new_app(AppWithFakes)
        stop, thread, errors = start(app)
        try:
            assert wait_for(lambda: app.perception is not None and app.perception.pipeline)
            assert app.webui is None
        finally:
            stop.set()
            thread.join(15)
    assert not errors and not thread.is_alive()


def test_exceptions_in_server_stop_do_not_skip_the_rest_of_the_shutdown(webui_data, monkeypatch):
    from roboemot.webui.server import Server

    def broken(self):
        raise RuntimeError("stop failed")

    monkeypatch.setattr(Server, "stop", broken)
    app = new_app(AppWithFakes)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.webui is not None and app.perception.pipeline is not None)
    finally:
        stop.set()
        thread.join(15)
    assert not thread.is_alive()
    assert errors and "stop failed" in str(errors[0])
    assert not [
        t
        for t in threading.enumerate()
        if t.name.startswith("roboemot-") and t.name != "roboemot-webui"
    ]
    monkeypatch.undo()
    app.webui.stop()


def test_the_declared_page_follows_a_port_set_in_the_app_config(tmp_path):
    """R4 S4 port point: an owner's webui.port in the data-folder app.yaml reaches the URL."""
    path = tmp_path / "app.yaml"
    path.write_text("schema_version: 1\nwebui:\n  port: 8080\n")
    assert app_module.webui_port(path) == 8080
    for text in ("webui:\n  port: 0\n", "webui: [1\n", "- x\n"):
        path.write_text(text)
        assert app_module.webui_port(path) == 8042
    assert app_module.webui_port(tmp_path / "missing.yaml") == 8042
