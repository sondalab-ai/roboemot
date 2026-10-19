"""SPEC 4A, 12A: the one Reachy Mini daemon client (``roboemot_reachymini.daemon``)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
import pytest
import roboemot_reachymini
from roboemot.diagnostics import Target
from roboemot_reachymini import daemon
from roboemot_reachymini.daemon import APP_NAME, DaemonClient, DaemonError

PACKAGE = Path(roboemot_reachymini.__file__).parent
HTTP_MODULES = {"urllib", "http", "httpx", "requests", "aiohttp"}


def client(handler, target: Target | None = None, **kw) -> DaemonClient:
    return DaemonClient(target or Target("robot"), transport=httpx.MockTransport(handler), **kw)


def test_base_url_comes_from_the_target_with_the_default_port():
    assert DaemonClient(Target()).base_url == "http://localhost:8000"
    assert DaemonClient(Target("robot")).base_url == "http://robot:8000"
    assert DaemonClient(Target("robot", 9000)).base_url == "http://robot:9000"
    assert DaemonClient().base_url == "http://localhost:8000"


def test_reads_hit_the_documented_paths():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, dict(req.url.params)))
        return httpx.Response(200, json={"path": req.url.path})

    c = client(handler)
    assert c.status() == {"path": "/api/daemon/status"}
    c.motors()
    c.state()
    c.current_app()
    assert [p for _, p, _ in seen] == [
        "/api/daemon/status",
        "/api/motors/status",
        "/api/state/full",
        "/api/apps/current-app-status",
    ]
    assert seen[2][2]["with_imu"] == "true"


def test_wifi_has_no_api_prefix_and_none_means_not_a_wireless():
    def handler(req):
        assert req.url.path == "/wifi/status"
        return httpx.Response(404)

    assert client(handler).wifi() is None
    assert client(lambda r: httpx.Response(200, json={"mode": "wlan"})).wifi() == {"mode": "wlan"}


def test_http_error_carries_status_and_daemon_detail():
    def handler(req):
        return httpx.Response(400, json={"detail": "An app is already running"})

    with pytest.raises(DaemonError, match="POST /apps/start-app/x -> HTTP 400: An app is already"):
        client(handler).start_app("x")


def test_http_error_with_a_plain_body_keeps_the_body():
    with pytest.raises(DaemonError, match="HTTP 500: boom"):
        client(lambda r: httpx.Response(500, text="boom")).status()


def test_unreachable_daemon_is_a_daemon_error_naming_the_call():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(DaemonError, match="GET /daemon/status"):
        client(handler).status()


def test_empty_body_is_none_and_bad_json_is_an_error():
    assert client(lambda r: httpx.Response(200)).current_app() is None
    with pytest.raises(DaemonError, match="current-app-status"):
        client(lambda r: httpx.Response(200, text="<html>")).current_app()


def test_write_calls_send_the_right_verbs_and_body():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, json.loads(req.content) if req.content else None))
        return httpx.Response(200, json={"startup_app": "roboemot"})

    c = client(handler)
    c.stop_current_app()
    c.wake_up()
    c.start_app("roboemot")
    c.set_startup_app("roboemot")
    assert c.startup_app() == "roboemot"
    assert seen[:4] == [
        ("POST", "/api/apps/stop-current-app", None),
        ("POST", "/api/move/play/wake_up", None),
        ("POST", "/api/apps/start-app/roboemot", None),
        ("PUT", "/api/apps/startup-app", {"startup_app": "roboemot"}),
    ]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"info": {"name": APP_NAME}, "state": "running"}, True),
        ({"info": {"name": APP_NAME}, "state": "starting"}, False),
        ({"info": {"name": "conversation"}, "state": "running"}, False),
        (None, False),
        ({"info": None, "state": "running"}, False),
        ([], False),
    ],
)
def test_app_running_is_current_app_status_for_our_app(answer, expected):
    assert client(lambda r: httpx.Response(200, json=answer)).app_running() is expected


def test_app_running_is_false_when_the_daemon_does_not_answer():
    def handler(req):
        raise httpx.ConnectError("refused")

    assert client(handler).app_running() is False


def test_restart_uses_the_long_timeout_and_the_restart_path():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, req.extensions["timeout"]["read"]))
        return httpx.Response(200)

    client(handler).restart_current_app()
    assert seen == [("POST", "/api/apps/restart-current-app", daemon.RESTART_TIMEOUT_S)]
    assert daemon.RESTART_TIMEOUT_S > daemon.STOP_TIMEOUT_S


def test_wait_app_returns_the_first_final_state_and_fails_when_the_app_is_replaced():
    states = iter(["starting", "running"])

    def handler(req):
        return httpx.Response(200, json={"info": {"name": "roboemot"}, "state": next(states)})

    assert client(handler).wait_app("roboemot", poll_s=0)["state"] == "running"
    other = client(lambda r: httpx.Response(200, json={"info": {"name": "x"}, "state": "running"}))
    with pytest.raises(DaemonError, match="no longer the current app"):
        other.wait_app("roboemot", poll_s=0)


def test_wait_app_gives_back_the_last_status_on_timeout():
    ticks = iter([0.0, 0.5, 1.0, 2.0])
    c = client(lambda r: httpx.Response(200, json={"info": {"name": "a"}, "state": "starting"}))
    status = c.wait_app(
        "a", timeout_s=1.5, poll_s=0, clock=lambda: next(ticks), sleep=lambda s: None
    )
    assert status["state"] == "starting"


def test_single_source_of_truth_for_the_daemon_constants():
    assert (daemon.APP_NAME, daemon.DEFAULT_PORT, daemon.STOP_TIMEOUT_S) == ("roboemot", 8000, 20.0)


def _module_asts():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path, ast.parse(path.read_text())


def _imports(tree: ast.AST) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_only_daemon_py_opens_http_to_the_daemon():
    offenders = {
        p.name: sorted(_imports(t) & HTTP_MODULES)
        for p, t in _module_asts()
        if p.name != "daemon.py" and _imports(t) & HTTP_MODULES
    }
    assert offenders == {}


def _assigned_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_port_app_name_and_stop_timeout_are_defined_once_in_daemon_py():
    owned = {"APP_NAME", "DEFAULT_PORT", "DAEMON_PORT", "STOP_TIMEOUT_S", "STATUS_PATH"}
    defined = {p.name: _assigned_names(t) & owned for p, t in _module_asts()}
    assert {k: v for k, v in defined.items() if v} == {
        "daemon.py": {"APP_NAME", "DEFAULT_PORT", "STOP_TIMEOUT_S"}
    }


def test_current_app_status_path_appears_only_in_daemon_py():
    hits = []
    for path, tree in _module_asts():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and "current-app-status" in node.value
                and path.name != "daemon.py"
            ):
                hits.append(path.name)
    assert hits == []
