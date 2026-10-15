"""SPEC 4A, Packaging: the install wizard (scripts/install_on_robot.py) against a fake daemon."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("sondalab_palette")

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "install_on_robot", ROOT / "scripts/install_on_robot.py"
)
wiz = importlib.util.module_from_spec(_spec)
sys.modules["install_on_robot"] = wiz
_spec.loader.exec_module(wiz)


class FakeDaemon:
    """Minimal Reachy Mini daemon: app slot, installed list, startup app, request log."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, object]] = []
        self.installed = ["roboemot", "conversation"]
        self.current: dict | None = None
        self.startup: str | None = None
        self.start_states = ["starting", "running"]

    def handle(self, method: str, path: str, body: object) -> tuple[int, object]:
        self.requests.append((method, path, body))
        match method, path:
            case "GET", "/api/daemon/status":
                return 200, {"robot_name": "mini", "state": "running", "wireless_version": True}
            case "GET", "/api/apps/list-available/installed":
                return 200, [{"name": n, "source_kind": "installed"} for n in self.installed]
            case "GET", "/api/apps/current-app-status":
                if self.current and self.current["state"] == "starting" and self.start_states:
                    self.current["state"] = self.start_states.pop(0)
                return 200, self.current
            case "POST", "/api/apps/stop-current-app":
                self.current = None
                return 200, None
            case "POST", "/api/move/play/wake_up":
                return 200, {"uuid": "x"}
            case "GET", "/api/apps/startup-app":
                return 200, {"startup_app": self.startup}
            case "PUT", "/api/apps/startup-app":
                self.startup = body["startup_app"]
                return 200, body
        if method == "POST" and path.startswith("/api/apps/start-app/"):
            if self.current:
                return 400, {"detail": "An app is already running"}
            name = path.rsplit("/", 1)[1]
            self.current = {"info": {"name": name, "source_kind": "installed"}, "state": "starting"}
            return 200, self.current
        return 404, {"detail": "Not Found"}

    @property
    def mutations(self) -> list[tuple[str, str, object]]:
        return [r for r in self.requests if r[0] != "GET"]


@pytest.fixture()
def daemon():
    fake = FakeDaemon()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n)) if n else None
            code, payload = fake.handle(self.command, self.path, body)
            raw = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = do_PUT = _serve

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_port}"
    yield fake
    server.shutdown()


@pytest.fixture()
def api(daemon):
    return wiz.DaemonApi(daemon.url)


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(wiz.time, "sleep", lambda s: None)


def quiet_ui(yes: bool = False) -> wiz.Ui:
    return wiz.Ui(interactive=False, assume_yes=yes)


def test_api_reads(api):
    assert api.daemon_status()["state"] == "running"
    assert api.installed_apps() == ["roboemot", "conversation"]
    assert api.current_app() is None
    assert api.startup_app() is None


def test_api_error_carries_daemon_detail(api, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    with pytest.raises(wiz.WizardError, match="HTTP 400: An app is already running"):
        api.start_app("roboemot")


def test_api_unreachable_is_a_wizard_error():
    with pytest.raises(wiz.WizardError, match="daemon/status"):
        wiz.DaemonApi("http://127.0.0.1:9", timeout_s=0.5).daemon_status()


def test_set_startup_app_sends_put_body(api, daemon):
    api.set_startup_app("roboemot")
    assert daemon.mutations == [("PUT", "/api/apps/startup-app", {"startup_app": "roboemot"})]
    assert api.startup_app() == "roboemot"


def test_wait_app_returns_error_state(api, daemon):
    daemon.start_states = ["error"]
    api.start_app("roboemot")
    assert api.wait_app("roboemot", poll_s=0)["state"] == "error"


def test_wait_app_times_out_with_last_status(api, daemon):
    daemon.start_states = []
    api.start_app("roboemot")
    ticks = iter([0.0, 0.0, 1.0, 2.0])
    status = api.wait_app("roboemot", timeout_s=1.5, poll_s=0, clock=lambda: next(ticks))
    assert status["state"] == "starting"


def test_start_stops_other_app_then_wakes_and_starts(api, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    wiz.start(quiet_ui(yes=True), api)
    assert [(m, p) for m, p, _ in daemon.mutations] == [
        ("POST", "/api/apps/stop-current-app"),
        ("POST", "/api/move/play/wake_up"),
        ("POST", "/api/apps/start-app/roboemot"),
    ]
    assert daemon.current["state"] == "running"


def test_start_declined_leaves_running_app_alone(api, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    ui = quiet_ui()
    ui.confirm = lambda question, default: False
    with pytest.raises(wiz.WizardError, match="conversation is still running"):
        wiz.start(ui, api)
    assert daemon.mutations == []


def test_start_reports_app_error(api, daemon):
    daemon.start_states = ["error"]
    with pytest.raises(wiz.WizardError, match="roboemot state error"):
        wiz.start(quiet_ui(), api)


@pytest.mark.parametrize("dev", [True, False])
def test_stage_packages_leaves_checkout_untouched(tmp_path, dev):
    repo_app_yaml = ROOT / "packages" / wiz.PKG_DATA / "app.yaml"
    existed = repo_app_yaml.exists()
    data = wiz.stage_packages(tmp_path, dev) / wiz.PKG_DATA
    assert (data / "app.yaml").exists() is dev
    assert (data / "reachy-mini-wireless.yaml").exists()
    assert {p.name for p in (data / "presets").glob("*.yaml")} >= {"default.yaml"}
    assert repo_app_yaml.exists() is existed


def test_third_party_requirements_exclude_own_packages():
    reqs = wiz.third_party_requirements()
    assert "roboemot-core" not in reqs
    assert any(r.startswith("pydantic") for r in reqs)
    assert any(r.startswith("scipy") for r in reqs)


def test_non_interactive_ui_takes_defaults():
    ui = quiet_ui()
    assert ui.ask("host?", "reachy-mini.local") == "reachy-mini.local"
    assert ui.confirm("stop?", False) is False
    assert quiet_ui(yes=True).confirm("stop?", False) is True


class FakeRemote:
    """Records robot commands; answers the probes a healthy Wireless gives."""

    commands: list[str] = []
    copies: list[list[Path]] = []

    def __init__(self, user, host, control_dir) -> None:
        self.target = f"{user}@{host}"

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def copy(self, files, dest) -> None:
        FakeRemote.copies.append(files)

    def run(self, command: str, check: bool = True) -> str:
        FakeRemote.commands.append(command)
        if command.startswith("test -x"):
            return "yes"
        if "platform.machine" in command:
            return "aarch64\n3.12\n1.11.0"
        if command.startswith("df "):
            return "/dev/root 30000000 1000000 20000000 5% /"
        if "--dry-run" in command:
            return "Would install scipy-1.14.1"
        if "entry_points" in command:
            return "roboemot_reachymini.app.RoboEmotApp\nTrue"
        return ""


@pytest.fixture()
def offline(monkeypatch, daemon):
    FakeRemote.commands, FakeRemote.copies = [], []
    port = int(daemon.url.rsplit(":", 1)[1])
    monkeypatch.setattr(wiz, "DAEMON_PORT", port)
    monkeypatch.setattr(wiz, "Remote", FakeRemote)
    monkeypatch.setattr(wiz, "build_wheels", lambda staged, out: [out / "a.whl", out / "b.whl"])


def options(**kw) -> wiz.Options:
    base = dict(
        host="127.0.0.1", user="pollen", dev=True, start=None, startup=None, dry_run=False, yes=True
    )
    return wiz.Options(**{**base, **kw})


def test_dry_run_touches_nothing(offline, daemon, capsys):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    wiz.run(options(dry_run=True), quiet_ui(yes=True))
    assert daemon.mutations == []
    assert FakeRemote.copies == []
    writes = ("rm ", "mkdir", "pip install --disable-pip-version-check -q", "--force-reinstall")
    assert not [c for c in FakeRemote.commands if any(w in c for w in writes)]
    assert any("pip install --dry-run" in c for c in FakeRemote.commands)
    out = capsys.readouterr().out
    assert "would stop conversation" in out and "dry run passed" in out


def test_full_run_installs_verifies_starts_and_sets_startup(offline, daemon):
    wiz.run(options(start=True, startup=True), quiet_ui(yes=True))
    assert len(FakeRemote.copies) == 1
    assert any("--force-reinstall" in c for c in FakeRemote.commands)
    assert daemon.current["state"] == "running"
    assert daemon.startup == "roboemot"


def test_full_run_fails_when_dev_config_mismatches(offline, daemon):
    with pytest.raises(wiz.WizardError, match="expected dev=False"):
        wiz.run(options(dev=False, start=False, startup=False), quiet_ui())


def test_full_run_fails_when_daemon_does_not_list_app(offline, daemon):
    daemon.installed = ["conversation"]
    with pytest.raises(wiz.WizardError, match="does not list roboemot"):
        wiz.run(options(start=False, startup=False), quiet_ui())


def test_main_returns_1_and_prints_reason(offline, daemon, capsys):
    daemon.installed = ["conversation"]
    rc = wiz.main(
        ["--host", "127.0.0.1", "--user", "pollen", "--dev", "--no-start", "--no-startup-app"]
    )
    assert rc == 1
    assert "does not list roboemot" in capsys.readouterr().out


@pytest.mark.sim
def test_daemon_api_against_simulator():
    """The REST calls the wizard relies on exist and answer as modelled in FakeDaemon."""
    real = wiz.DaemonApi("http://localhost:8000")
    status = real.daemon_status()
    assert status["state"] == "running" and status["simulation_enabled"]
    assert isinstance(real.installed_apps(), list)
    assert real.current_app() is None
    real.wake_up()
    assert real.startup_app() is None
    with pytest.raises(wiz.WizardError, match="HTTP 400"):
        real.set_startup_app("roboemot-not-installed")
