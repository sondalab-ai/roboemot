"""Security-audit regressions for the web UI: rebinding, limits, connections, PIN handling."""

from __future__ import annotations

import io
import json
import socket
import stat
import threading
import time

import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.webui import make_server
from roboemot.webui.__main__ import main as cli_main
from roboemot.webui.api import Request, WebApi
from roboemot.webui.auth import PinStore
from roboemot.webui.guard import is_trusted_host
from test_webui import PIN, Client, Clock, FakeService

REMOTE = "192.168.1.50"


def serve(tmp_path, **webui):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0} | webui)
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path)
    srv.set_pin(PIN)
    return srv.start()


def raw_request(port: int, data: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(data)
        chunks = []
        while chunk := sock.recv(4096):
            chunks.append(chunk)
    return b"".join(chunks)


# ------------------------------------------------------------------ 1. DNS rebinding


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("127.0.0.1", True),
        ("192.168.1.5", True),
        ("::1", True),
        ("localhost", True),
        ("reachy-mini.local", True),
        ("Reachy-Mini.LOCAL.", True),
        ("evil.example", False),
        ("127.0.0.1.evil.example", False),
        ("local", False),
        ("evil.local.example", False),
        ("", False),
        (None, False),
    ],
)
def test_trusted_host_rule(host, ok):
    assert is_trusted_host(host) is ok


def test_allowed_hosts_extends_the_rule():
    assert is_trusted_host("robot.lan", ["robot.lan"])
    assert not is_trusted_host("other.lan", ["robot.lan"])


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8042", "127.0.0.1.evil.example"])
def test_rebinding_host_refused_everywhere(tmp_path, host):
    srv = serve(tmp_path)
    try:
        c = Client(srv.port)
        for method, path in [("GET", "/"), ("GET", "/static/app.js"), ("GET", "/api/strings")]:
            resp = c.request(method, path, headers={"Host": host})
            assert resp.status == 400
        login = c.request("POST", "/api/login", {"pin": PIN}, headers={"Host": host})
        assert login.status == 400
        assert "Set-Cookie" not in login.headers
    finally:
        srv.stop()


def test_rebinding_origin_and_host_both_evil(tmp_path):
    srv = serve(tmp_path)
    try:
        c = Client(srv.port)
        assert c.login().status == 200
        evil = {"Host": "evil.example", "Origin": "http://evil.example"}
        assert c.request("POST", "/api/enrollment/cancel", {}, headers=evil).status == 400
        only_origin = {"Origin": "http://evil.example"}
        assert c.request("POST", "/api/enrollment/cancel", {}, headers=only_origin).status == 403
    finally:
        srv.stop()


def test_configured_host_name_is_accepted(tmp_path):
    srv = serve(tmp_path, allowed_hosts=["robot.lan"])
    try:
        c = Client(srv.port)
        assert c.request("GET", "/api/strings", headers={"Host": "robot.lan:8042"}).status == 200
        assert c.request("GET", "/api/strings", headers={"Host": "evil.example"}).status == 400
    finally:
        srv.stop()


def test_duplicate_host_or_origin_refused(tmp_path):
    srv = serve(tmp_path)
    try:
        two_hosts = b"GET /api/strings HTTP/1.1\r\nHost: evil.example\r\nHost: 127.0.0.1\r\n\r\n"
        assert raw_request(srv.port, two_hosts).startswith(b"HTTP/1.0 400")
        none = b"GET /api/strings HTTP/1.0\r\n\r\n"
        assert raw_request(srv.port, none).startswith(b"HTTP/1.0 400")
    finally:
        srv.stop()


def test_allowed_hosts_needs_bare_names():
    with pytest.raises(ValidationError):
        RoboEmotConfig(webui={"allowed_hosts": ["http://robot.lan"]})


# ------------------------------------------------------------------ 2. PIN set by CLI only


def test_cli_reads_the_pin_from_stdin(tmp_path):
    assert cli_main(["set-pin", "--data-dir", str(tmp_path)], stdin=io.StringIO(PIN + "\n")) == 0
    assert PinStore(tmp_path).verify(PIN)
    assert not PinStore(tmp_path).verify("000000")


def test_cli_rejects_a_weak_pin_and_takes_no_pin_argument(tmp_path, capsys):
    assert cli_main(["set-pin", "--data-dir", str(tmp_path)], stdin=io.StringIO("12\n")) == 1
    assert not (tmp_path / "webui_pin.json").exists()
    with pytest.raises(SystemExit):
        cli_main(["set-pin", "--data-dir", str(tmp_path), PIN])
    with pytest.raises(SystemExit):
        cli_main(["set-pin", "--data-dir", str(tmp_path), "--pin", PIN])


# ------------------------------------------------------------------ 3, 4. guess limits


@pytest.fixture
def api(tmp_path):
    cfg = RoboEmotConfig(
        webui={
            "pin_max_attempts": 3,
            "loopback_max_attempts": 3,
            "pin_lockout_s": 60,
            "client_backoff_s": 1.0,
        }
    )
    pins = PinStore(tmp_path)
    pins.set(PIN)
    clock = Clock()
    return WebApi(FakeService(), Diagnostics(), cfg, pins, clock=clock), clock, pins


def login(api, pin, ip=REMOTE):
    body = json.dumps({"pin": pin}).encode()
    headers = {"content-type": "application/json", "host": "192.168.1.2:8042"}
    return api.handle(Request("POST", "/api/login", headers, body, ip))


def test_client_backoff_doubles(api):
    app, clock, _ = api
    assert login(app, "000000").status == 401
    again = login(app, "000000")  # no time passed: still backing off
    assert again.status == 429
    assert again.headers["Retry-After"] == "2"
    clock.now += 1.01
    assert login(app, "000000").status == 401
    clock.now += 1.5  # needs 2 s now
    assert login(app, "000000").status == 429
    clock.now += 1.0
    assert login(app, "000000").status == 401


def test_global_budget_locks_every_remote_client_but_not_loopback(api):
    app, clock, _ = api
    for n in range(3):
        assert login(app, "000000", ip=f"10.0.0.{n}").status == 401
    other = login(app, PIN, ip="10.0.0.99")
    assert other.status == 429 and other.data == {"error": "locked"}
    assert login(app, PIN, ip="127.0.0.1").status == 200
    clock.now += 61
    assert login(app, PIN, ip="10.0.0.99").status == 200


def test_loopback_has_its_own_finite_budget_and_backs_off(api):
    app, clock, _ = api
    assert login(app, "000000", ip="127.0.0.1").status == 401
    again = login(app, "000000", ip="127.0.0.1")  # backing off: doubles per failure
    assert again.status == 429 and again.headers["Retry-After"] == "2"
    for _ in range(2):
        clock.now += 100
        assert login(app, "000000", ip="127.0.0.1").status == 401
    locked = login(app, PIN, ip="::1")  # three wrong PINs used the loopback budget
    assert locked.status == 429 and locked.data == {"error": "locked"}
    clock.now += 61 + 100
    assert login(app, PIN, ip="127.0.0.1").status == 200


def test_loopback_failures_do_not_lock_lan_clients(api):
    app, clock, _ = api
    for _ in range(3):
        clock.now += 100
        assert login(app, "000000", ip="127.0.0.1").status == 401
    assert login(app, PIN, ip="127.0.0.1").status == 429  # loopback is locked
    for n in range(2):  # the LAN budget (3) is untouched
        assert login(app, PIN, ip=f"10.3.0.{n}").status == 200


def test_lan_lockout_does_not_block_loopback(api):
    app, clock, _ = api
    for n in range(3):
        assert login(app, "000000", ip=f"10.4.0.{n}").status == 401
    assert login(app, PIN, ip="10.4.0.99").status == 429
    assert login(app, PIN, ip="127.0.0.1").status == 200


def test_new_pin_clears_a_loopback_lockout_too(api):
    app, clock, pins = api
    for _ in range(3):
        clock.now += 100
        login(app, "000000", ip="127.0.0.1")
    assert login(app, PIN, ip="127.0.0.1").status == 429
    time.sleep(0.01)
    pins.set("135790")
    assert login(app, "135790", ip="127.0.0.1").status == 200


def test_new_pin_clears_a_lockout(api, tmp_path):
    app, _, pins = api
    for n in range(3):
        login(app, "000000", ip=f"10.0.0.{n}")
    assert login(app, PIN, ip="10.0.0.99").status == 429
    time.sleep(0.01)
    pins.set("135790")
    assert login(app, "135790", ip="10.0.0.99").status == 200


def test_success_refunds_the_attempt(api):
    app, clock, _ = api
    for _ in range(5):
        clock.now += 100
        assert login(app, PIN).status == 200


def test_parallel_guesses_cannot_exceed_the_budget(api, monkeypatch):
    app, _, pins = api
    calls = []
    real = pins.verify

    def slow(pin):
        calls.append(pin)
        time.sleep(0.05)
        return real(pin)

    monkeypatch.setattr(pins, "verify", slow)
    results = []

    def attempt(n):
        results.append(login(app, "000000", ip=f"10.1.0.{n}").status)

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) <= 3
    assert set(results) <= {401, 429}


def test_one_hash_at_a_time(api, monkeypatch):
    app, _, pins = api
    running, peak, lock = 0, 0, threading.Lock()
    real = pins.verify

    def slow(pin):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return real(pin)

    monkeypatch.setattr(pins, "verify", slow)
    threads = [
        threading.Thread(target=login, args=(app, PIN), kwargs={"ip": f"10.2.0.{n}"})
        for n in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_busy_hashing_gets_429_and_refunds(api):
    app, _, _ = api
    assert app._hashing.acquire(blocking=False)
    try:
        resp = login(app, PIN, ip="10.0.0.7")
        assert resp.status == 429 and resp.data == {"error": "busy"}
    finally:
        app._hashing.release()
    assert login(app, PIN, ip="10.0.0.7").status == 200


# ------------------------------------------------------------------ 5. connections, deadline


def test_connection_cap_refuses_extra_clients(tmp_path):
    srv = serve(tmp_path, max_connections=1)
    holder = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
    try:
        time.sleep(0.2)  # the server thread now owns the only slot, waiting for a request
        refused = raw_request(srv.port, b"GET /api/strings HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert refused.startswith(b"HTTP/1.0 503")
        holder.sendall(b"GET /api/strings HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        assert holder.recv(64).startswith(b"HTTP/1.0 200")
    finally:
        holder.close()
        time.sleep(0.2)
        try:
            assert Client(srv.port).request("GET", "/api/strings").status == 200
        finally:
            srv.stop()


def test_slow_drip_is_cut_off_by_the_overall_deadline(tmp_path):
    srv = serve(tmp_path, request_deadline_s=1, request_timeout_s=5)
    try:
        sock = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        start = time.monotonic()
        closed = False
        for byte in b"GET /api/strings HTTP/1.0\r\nHost: 127.0.0.1\r\nX-A: " + b"a" * 40:
            try:
                sock.sendall(bytes([byte]))
                time.sleep(0.2)
            except OSError:
                closed = True
                break
            if time.monotonic() - start > 4:
                break
        if not closed:
            sock.settimeout(2)
            try:
                closed = sock.recv(1) == b""
            except (TimeoutError, OSError):
                closed = True
        sock.close()
        assert closed
        assert time.monotonic() - start < 4
    finally:
        srv.stop()


# ------------------------------------------------------------------ 7. bind rule, 13. frame


def test_bind_defaults_to_loopback_unless_robot_hosted(tmp_path):
    cfg = RoboEmotConfig(webui={"port": 0})
    sim = make_server(FakeService(), Diagnostics(), cfg, tmp_path)
    robot = make_server(FakeService(), Diagnostics(), cfg, tmp_path, robot_hosted=True)
    explicit = make_server(
        FakeService(),
        Diagnostics(),
        RoboEmotConfig(webui={"port": 0, "bind": "127.0.0.1"}),
        tmp_path,
        robot_hosted=True,
    )
    try:
        assert sim.host == "127.0.0.1"
        assert robot.host == "0.0.0.0"
        assert explicit.host == "127.0.0.1"
    finally:
        for s in (sim, robot, explicit):
            s.stop()


def test_frame_ancestors_is_configurable_and_validated(tmp_path):
    srv = serve(tmp_path, frame_ancestors="'self' http://reachy-mini.local")
    try:
        csp = Client(srv.port).request("GET", "/").headers["Content-Security-Policy"]
        assert "frame-ancestors 'self' http://reachy-mini.local" in csp
    finally:
        srv.stop()
    default = serve(tmp_path)
    try:
        csp = Client(default.port).request("GET", "/").headers["Content-Security-Policy"]
        assert csp.endswith("frame-ancestors 'none'")
    finally:
        default.stop()
    with pytest.raises(ValidationError):
        RoboEmotConfig(webui={"frame_ancestors": "'none'; script-src *"})


# ------------------------------------------------------------------ 12. PIN file


def test_pin_file_survives_a_planted_stale_temp_and_symlink(tmp_path):
    target = tmp_path / "victim.txt"
    target.write_text("keep")
    (tmp_path / "webui_pin.tmp").symlink_to(target)
    store = PinStore(tmp_path)
    store.set(PIN)
    assert target.read_text() == "keep"
    assert store.verify(PIN)
    assert stat.S_IMODE((tmp_path / "webui_pin.json").stat().st_mode) == 0o600
    assert not (tmp_path / "webui_pin.tmp").exists()


def test_pin_file_mode_is_forced_even_if_umask_is_open(tmp_path):
    import os

    old = os.umask(0)
    try:
        PinStore(tmp_path / "data").set(PIN)
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "data" / "webui_pin.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700


def test_stop_between_thread_creation_and_start_does_not_join_an_unstarted_thread(
    tmp_path, monkeypatch
):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path)
    real_start = threading.Thread.start

    def start_after_stop(self):
        if self.name == "roboemot-webui":
            srv.stop()
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start_after_stop)
    srv.start()
    srv.stop()


# ------------------------------------------------------------- R3 SF2: security fields apply live


def live_api(tmp_path, **webui):
    """A WebApi over a config stack, as the runtime builds it (the config is a callable)."""
    from roboemot.config.stack import ConfigStack

    stack = ConfigStack(app={"webui": {"pin_lockout_s": 60, "client_backoff_s": 0.1} | webui})
    pins = PinStore(tmp_path)
    pins.set(PIN)
    clock = Clock()
    return WebApi(FakeService(), Diagnostics(), stack.effective, pins, clock=clock), clock, stack


def get(api, path, cookie):
    headers = {"host": "192.168.1.2:8042", "cookie": cookie}
    return api.handle(Request("GET", path, headers, b"", REMOTE))


def cookie_of(resp):
    return resp.cookies[0].split(";")[0]


def test_a_tighter_session_ttl_applies_without_a_restart(tmp_path):
    api, clock, stack = live_api(tmp_path, session_ttl_s=900)
    cookie = cookie_of(login(api, PIN))
    stack.set("webui.session_ttl_s", 120)
    clock.now += 121
    assert get(api, "/api/registry", cookie).status == 401


def test_a_tighter_session_max_and_max_sessions_apply_without_a_restart(tmp_path):
    api, clock, stack = live_api(tmp_path, max_sessions=8)
    first = cookie_of(login(api, PIN))
    second = cookie_of(login(api, PIN))
    stack.set("webui.max_sessions", 1)
    third = cookie_of(login(api, PIN))
    assert get(api, "/api/registry", first).status == 401
    assert get(api, "/api/registry", second).status == 401
    assert get(api, "/api/registry", third).status == 200
    stack.set("webui.session_max_s", 600)
    clock.now += 601
    assert get(api, "/api/registry", third).status == 401


def test_a_tighter_pin_budget_applies_without_a_restart(tmp_path):
    api, clock, stack = live_api(tmp_path, pin_max_attempts=10)
    stack.set("webui.pin_max_attempts", 2)
    assert login(api, "000000").status == 401
    clock.now += 1
    assert login(api, "000000").status == 401
    clock.now += 1
    assert login(api, PIN, ip="192.168.1.77").status == 429  # the LAN pool is locked


def test_server_limits_follow_the_config_or_need_a_restart(tmp_path):
    """frame_ancestors and the request deadlines are read per connection; the connection cap is a
    semaphore built once, so it is a restart field and the web UI says so."""
    from roboemot.config import field_meta
    from roboemot.config.stack import ConfigStack

    assert field_meta("webui.max_connections").get("restart") is True
    for path in ("webui.frame_ancestors", "webui.request_deadline_s", "webui.request_timeout_s"):
        assert not field_meta(path).get("restart"), path
    stack = ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0, "frame_ancestors": "'self'"}}
    )
    srv = make_server(FakeService(), Diagnostics(), stack.effective, tmp_path).start()
    try:
        stack.set("webui.frame_ancestors", "'none'", admin=True)
        csp = Client(srv.port).request("GET", "/").headers["Content-Security-Policy"]
        assert csp.endswith("frame-ancestors 'none'")
        stack.set("webui.request_deadline_s", 1)
        sock = socket.create_connection(("127.0.0.1", srv.port), timeout=5)
        start = time.monotonic()
        sock.sendall(b"GET /api/strings HTTP/1.0\r\n")  # never finished
        sock.settimeout(4)
        try:
            closed = sock.recv(1) == b""
        except OSError:
            closed = True
        sock.close()
        assert closed and time.monotonic() - start < 3
    finally:
        srv.stop()
