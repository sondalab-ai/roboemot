"""SPEC 4A, Packaging: the robot installer (roboemot.install) with the Reachy Mini Wireless
platform, against a fake daemon and a fake robot that emulates SSH, pip and the filesystem."""

from __future__ import annotations

import copy
import json
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import roboemot.install as inst
from roboemot.install import Options, Ui, WizardError
from roboemot_reachymini import install as reachy
from roboemot_reachymini.install import DaemonApi, ReachyMiniWireless

ROOT = Path(__file__).resolve().parents[3]
RECORD = ReachyMiniWireless.record_path
SITE = "/venvs/apps_venv/lib/python3.12/site-packages"
OWN = {"roboemot-core": "0.1.0.dev0", "roboemot-reachymini": "0.1.0.dev0"}


# ---------------------------------------------------------------- fakes


class FakeRobot:
    """Emulates what the installer runs over SSH: pip on a shared environment, the record file,
    temporary directories, and the Python probes."""

    def __init__(self) -> None:
        self.env = {"pip": "26.2.1", "numpy": "2.0.0", "httpx": "0.28.1", "reachy-mini": "1.11.0"}
        self.requires: dict[str, list[str]] = {"reachy-mini": ["numpy", "httpx"]}
        self.deps = {"scipy": "1.18.1"}  # what installing the app adds
        self.conflict: dict[str, str] = {}  # upgrades the app needs (fails while pinned)
        self.files: dict[str, str] = {}
        self.tmp_dirs: set[str] = set()
        self.stray: dict[str, list[str]] = {}  # module -> leftover paths
        self.dev_cfg = True
        self.commands: list[str] = []
        self.copies: list[list[Path]] = []
        self.fail_install = False

    # Remote interface
    target = "pollen@robot"

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def copy(self, files, dest) -> None:
        assert dest in self.tmp_dirs
        self.copies.append(files)

    def run(self, command: str, check: bool = True) -> str:
        self.commands.append(command)
        if " -m pip " in command:
            out, rc = self._pip(command)
            return f"{out}\n__rc={rc}"
        plain = " ".join(shlex.split(command)) if command.startswith("/venvs") else command
        if command.startswith("test -x"):
            return "yes"
        if command.startswith("test -e"):
            return "no"
        if "platform.machine" in plain:
            return "aarch64\n3.12"
        if "version('reachy_mini')" in plain:
            return "1.11.0"
        if "reachy_mini_apps" in plain:
            return f"roboemot_reachymini.main\n{self.dev_cfg}"
        if "distributions()" in plain:
            return json.dumps({n: self.requires.get(n, []) for n in self.env})
        if "find_spec" in plain:
            mods = shlex.split(command)[3:]
            found = {m: [f"{SITE}/{m}"] for m in mods if "roboemot-core" in self.env}
            found.update({m: p for m, p in self.stray.items() if m in mods})
            return json.dumps(found)
        if "purelib" in plain:
            return SITE
        if command.startswith("df "):
            return "/dev/root 30000000 1000000 20000000 5% /"
        if command.startswith("cat "):
            return self.files.get(shlex.split(command)[1], "")
        if command.startswith("printf %s") and "&& mv" in command:
            t = shlex.split(command)
            self.files[t[8]] = t[2]
            return ""
        if command.startswith("mktemp -d"):
            d = f"/tmp/roboemot-install.{len(self.tmp_dirs)}"
            self.tmp_dirs.add(d)
            return d
        if command.startswith("ls -d /tmp/roboemot-install"):
            return "\n".join(sorted(self.tmp_dirs))
        if command.startswith("rm -f "):
            self.files.pop(shlex.split(command)[2], None)
            return ""
        if command.startswith("rm -rf "):
            path = shlex.split(command)[2]
            self.tmp_dirs.discard(path)
            self.stray = {m: [p for p in ps if p != path] for m, ps in self.stray.items()}
            self.stray = {m: ps for m, ps in self.stray.items() if ps}
            return ""
        raise AssertionError(f"unexpected robot command: {command}")

    def _pip(self, command: str) -> tuple[str, int]:
        assert "--no-cache-dir" in command, command
        pinned = "-c /dev/stdin" in command
        args = command.split(" -m pip ", 1)[1].split(" 2>&1", 1)[0]
        targets = [t for t in shlex.split(args) if not t.startswith("-")][1:]
        if args.startswith("list"):
            return json.dumps([{"name": n, "version": v} for n, v in self.env.items()]), 0
        if args.startswith("uninstall"):
            for t in targets:
                self.env.pop(t, None)
            return "", 0
        if not args.startswith("install"):
            raise AssertionError(args)
        if "--force-reinstall" in args:
            return "", 0
        if "--no-deps" in args:  # install_exact
            for t in targets:
                name, version = t.split("==")
                self.env[name] = version
            return "", 0
        if pinned and self.conflict:
            return "ERROR: ResolutionImpossible", 1
        changes = {**self.deps, **(self.conflict if not pinned else {})}
        if "--dry-run" in args:
            items = [{"metadata": {"name": n, "version": v}} for n, v in changes.items()]
            if any(t.endswith(".whl") for t in targets):
                items += [{"metadata": {"name": n, "version": v}} for n, v in OWN.items()]
            return "WARNING: something\n" + json.dumps({"install": items}), 0
        if self.fail_install:
            raise KeyboardInterrupt  # the connection drops mid-install
        self.env.update(changes)
        self.env.update(OWN)
        self.requires.update(
            {"roboemot-core": ["numpy", "scipy"], "roboemot-reachymini": ["roboemot-core"]}
        )
        return "", 0


class FakeDaemon:
    """Minimal Reachy Mini daemon: app slot, installed list (from the robot), startup app."""

    def __init__(self, robot: FakeRobot) -> None:
        self.robot = robot
        self.requests: list[tuple[str, str, object]] = []
        self.current: dict | None = None
        self.startup: str | None = None
        self.start_states = ["starting", "running"]
        self.others = ["conversation"]
        self.hide_own = False  # a daemon that does not pick the app up

    @property
    def installed(self) -> list[str]:
        own = ["roboemot"] if "roboemot-reachymini" in self.robot.env and not self.hide_own else []
        return own + self.others

    def handle(self, method: str, path: str, body: object) -> tuple[int, object]:
        self.requests.append((method, path, body))
        match method, path:
            case "GET", "/api/daemon/status":
                return 200, {
                    "robot_name": "mini",
                    "state": "running",
                    "wireless_version": True,
                    "version": "1.11.0",
                }
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
def robot():
    return FakeRobot()


@pytest.fixture()
def daemon(robot):
    fake = FakeDaemon(robot)

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
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_port}"
    fake.port = server.server_port
    yield fake
    server.shutdown()


@pytest.fixture()
def api(daemon):
    return DaemonApi(daemon.url)


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(reachy.time, "sleep", lambda s: None)


@pytest.fixture()
def offline(monkeypatch, robot, daemon):
    monkeypatch.setattr(inst, "load_platform", lambda rid: ReachyMiniWireless(port=daemon.port))
    monkeypatch.setattr(inst, "Remote", lambda user, host, control_dir: robot)
    monkeypatch.setattr(
        inst,
        "build_wheels",
        lambda dirs, out: [out / f"{d.name.replace('-', '_')}-0.1.whl" for d in dirs],
    )


def quiet_ui(yes: bool = False) -> Ui:
    return Ui(interactive=False, assume_yes=yes)


def options(**kw) -> Options:
    base = dict(host="127.0.0.1", user="pollen", robot="reachy-mini-wireless", dev=True, yes=True)
    return Options(**{**base, **kw})


def install(**kw) -> None:
    inst.run_install(options(start=False, startup=False, **kw), quiet_ui(yes=True), ROOT)


def uninstall(**kw) -> None:
    inst.run_uninstall(options(uninstall=True, **kw), quiet_ui(yes=True), ROOT)


# ---------------------------------------------------------------- daemon API


def test_api_reads(api):
    assert api.daemon_status()["state"] == "running"
    assert api.installed_apps() == ["conversation"]
    assert api.current_app() is None
    assert api.startup_app() is None


def test_api_error_carries_daemon_detail(api, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    with pytest.raises(WizardError, match="HTTP 400: An app is already running"):
        api.start_app("roboemot")


def test_api_unreachable_is_a_wizard_error():
    with pytest.raises(WizardError, match="daemon/status"):
        DaemonApi("http://127.0.0.1:9", timeout_s=0.5).daemon_status()


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


def platform_for(daemon) -> ReachyMiniWireless:
    p = ReachyMiniWireless(port=daemon.port)
    p.connect(quiet_ui(), "127.0.0.1")
    return p


def test_start_stops_other_app_then_wakes_and_starts(daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    platform_for(daemon).start(quiet_ui(yes=True))
    assert [(m, p) for m, p, _ in daemon.mutations] == [
        ("POST", "/api/apps/stop-current-app"),
        ("POST", "/api/move/play/wake_up"),
        ("POST", "/api/apps/start-app/roboemot"),
    ]
    assert daemon.current["state"] == "running"


def test_start_declined_leaves_running_app_alone(daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    ui = quiet_ui()
    ui.confirm = lambda question, default: False
    with pytest.raises(WizardError, match="conversation is still running"):
        platform_for(daemon).start(ui)
    assert daemon.mutations == []


def test_start_reports_app_error(daemon):
    daemon.start_states = ["error"]
    with pytest.raises(WizardError, match="roboemot state error"):
        platform_for(daemon).start(quiet_ui())


@pytest.mark.parametrize("dev", [True, False])
def test_stage_packages_leaves_checkout_untouched(tmp_path, dev):
    repo_app_yaml = ROOT / "packages" / reachy.PKG_DATA / "app.yaml"
    existed = repo_app_yaml.exists()
    dirs = reachy.stage_packages(ROOT, tmp_path, dev)
    data = tmp_path / reachy.PKG_DATA
    assert [d.name for d in dirs] == ["roboemot-core", "roboemot-reachymini"]
    assert (data / "app.yaml").exists() is dev
    assert (data / "reachy-mini-wireless.yaml").exists()
    assert {p.name for p in (data / "presets").glob("*.yaml")} >= {"default.yaml"}
    assert repo_app_yaml.exists() is existed


def test_non_interactive_ui_takes_defaults():
    ui = quiet_ui()
    assert ui.ask("host?", "reachy-mini.local") == "reachy-mini.local"
    assert ui.confirm("stop?", False) is False
    assert quiet_ui(yes=True).confirm("stop?", False) is True


def test_installer_registered_under_the_manifest_id():
    assert "reachy-mini-wireless" in inst.installed_platforms()
    assert isinstance(inst.load_platform("reachy-mini-wireless"), ReachyMiniWireless)
    with pytest.raises(WizardError, match="no installer for desk-head-v1"):
        inst.load_platform("desk-head-v1")


# ---------------------------------------------------------------- install


def test_install_dry_run_touches_nothing(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    env = dict(robot.env)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert daemon.mutations == [] and robot.copies == [] and robot.files == {}
    assert robot.env == env and robot.tmp_dirs == set()
    writes = ("rm ", "mkdir", "mktemp", "mv ")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]
    out = capsys.readouterr().out
    assert "would add scipy 1.18.1" in out
    assert "would stop conversation" in out and "dry run passed" in out


def test_install_records_changes_and_cleans_tmp(offline, robot, daemon):
    install()
    assert len(robot.copies) == 1 and robot.tmp_dirs == set()
    record = json.loads(robot.files[RECORD])
    assert record["added"] == ["scipy"] and record["replaced"] == {}
    assert record["pending_before"] is None
    pip = [c for c in robot.commands if " -m pip install" in c and "--dry-run" not in c]
    assert pip and all("-c /dev/stdin" in c for c in pip if "--no-deps" not in c)


def test_install_starts_and_sets_autostart_recording_the_previous_one(offline, robot, daemon):
    daemon.startup = "conversation"
    inst.run_install(options(start=True, startup=True), quiet_ui(yes=True), ROOT)
    assert daemon.current["state"] == "running" and daemon.startup == "roboemot"
    record = json.loads(robot.files[RECORD])
    assert record["autostart_before"] == "conversation" and record["autostart_recorded"]


def test_install_upgrades_shared_packages_only_when_allowed(offline, robot, daemon):
    robot.conflict = {"numpy": "2.1.0"}
    ui = quiet_ui()
    ui.confirm = lambda question, default: "shared packages" not in question
    with pytest.raises(WizardError, match="shared packages left as they were"):
        inst.run_install(options(start=False, startup=False, yes=False), ui, ROOT)
    assert robot.env["numpy"] == "2.0.0" and "roboemot-core" not in robot.env
    install()
    assert robot.env["numpy"] == "2.1.0"
    assert json.loads(robot.files[RECORD])["replaced"] == {
        "numpy": {"before": "2.0.0", "after": "2.1.0"}
    }


def test_install_fails_when_dev_config_mismatches(offline, robot):
    robot.dev_cfg = False
    with pytest.raises(WizardError, match="expected dev=True"):
        install()


def test_install_fails_when_daemon_does_not_list_app(offline, daemon):
    daemon.hide_own = True
    with pytest.raises(WizardError, match="does not list roboemot"):
        install()


def test_main_returns_1_and_prints_reason(offline, robot, capsys):
    robot.dev_cfg = False
    rc = inst.main(
        ["--host", "127.0.0.1", "--user", "pollen", "--robot", "reachy-mini-wireless", "--dev",
         "--no-start", "--no-startup-app"],
        root=ROOT,
    )  # fmt: skip
    assert rc == 1
    assert "expected dev=True" in capsys.readouterr().out


# ---------------------------------------------------------------- uninstall


def snapshot(robot, daemon) -> tuple:
    return (
        dict(robot.env),
        dict(robot.files),
        set(robot.tmp_dirs),
        daemon.startup,
        daemon.installed,
    )


def test_install_then_uninstall_leaves_the_robot_as_it_was(offline, robot, daemon):
    robot.conflict = {"numpy": "2.1.0"}
    daemon.startup = "conversation"
    before = snapshot(robot, daemon)
    inst.run_install(options(start=True, startup=True), quiet_ui(yes=True), ROOT)
    assert snapshot(robot, daemon) != before
    uninstall()
    assert snapshot(robot, daemon) == before
    assert daemon.current is None  # the app was stopped


def test_uninstall_dry_run_touches_nothing(offline, robot, daemon, capsys):
    inst.run_install(options(start=True, startup=True), quiet_ui(yes=True), ROOT)
    state = copy.deepcopy(snapshot(robot, daemon))
    n = len(daemon.mutations)
    uninstall(dry_run=True)
    assert snapshot(robot, daemon) == state and len(daemon.mutations) == n
    out = capsys.readouterr().out
    assert "would stop roboemot" in out and "would set the autostart app back to none" in out
    assert (
        "would remove roboemot-core 0.1.0.dev0, roboemot-reachymini 0.1.0.dev0, scipy 1.18.1" in out
    )


def test_uninstall_twice_is_harmless(offline, robot, daemon):
    before = snapshot(robot, daemon)
    install()
    uninstall()
    uninstall()
    assert snapshot(robot, daemon) == before


def test_uninstall_keeps_a_dependency_another_app_now_needs(offline, robot, daemon):
    install()
    robot.env["other-app"] = "1.0"
    robot.requires["other-app"] = ["scipy"]
    uninstall()
    assert "scipy" in robot.env and "roboemot-core" not in robot.env


def test_uninstall_leaves_autostart_the_owner_changed(offline, robot, daemon):
    inst.run_install(options(start=False, startup=True), quiet_ui(yes=True), ROOT)
    daemon.startup = "conversation"
    uninstall()
    assert daemon.startup == "conversation"


def test_uninstall_after_an_interrupted_install(offline, robot, daemon):
    before = snapshot(robot, daemon)
    robot.fail_install = True
    with pytest.raises(KeyboardInterrupt):
        install()
    assert json.loads(robot.files[RECORD])["pending_before"] is not None
    robot.fail_install = False
    robot.env.update({"scipy": "1.18.1", **OWN})  # pip got that far before the drop
    uninstall()
    assert snapshot(robot, daemon)[:3] == (before[0], before[1], before[2])


def test_uninstall_removes_leftover_modules(offline, robot, daemon):
    robot.stray = {"roboemot": [f"{SITE}/roboemot"]}
    uninstall()
    assert robot.stray == {}
    assert f"rm -rf {SITE}/roboemot" in robot.commands


def test_uninstall_keeps_the_record_when_verification_fails(offline, robot, daemon, monkeypatch):
    install()
    monkeypatch.setattr(robot, "_pip", _pip_ignoring_uninstall(robot._pip))
    with pytest.raises(WizardError, match="uninstall incomplete"):
        uninstall()
    assert RECORD in robot.files


def _pip_ignoring_uninstall(real):
    def pip(command):
        if " uninstall " in command:
            return "", 0
        return real(command)

    return pip


def test_every_pip_call_avoids_the_cache(offline, robot, daemon):
    install()
    uninstall()
    pip = [c for c in robot.commands if " -m pip " in c]
    assert pip and all("--no-cache-dir" in c for c in pip)


# ---------------------------------------------------------------- robot selection


def test_select_robot_takes_first_robot_found(monkeypatch):
    found = [("Neve  192.168.88.198", "192.168.88.198", "reachy-mini-wireless")]
    monkeypatch.setattr(inst, "find_robots", lambda root: found)
    host, platform = inst.select_robot(quiet_ui(), Options(), ROOT)
    assert host == "192.168.88.198" and platform.manifest_id == "reachy-mini-wireless"


def test_select_robot_other_asks_for_host(monkeypatch):
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve", "1.2.3.4", "reachy-mini-wireless")]
    )
    ui = quiet_ui()
    ui.choose = lambda question, options: len(options) - 1
    host, _ = inst.select_robot(ui, Options(), ROOT)
    assert host == ReachyMiniWireless.default_host


@pytest.mark.parametrize("found", [[], RuntimeError("discovery needs zeroconf")])
def test_select_robot_falls_back_to_default(monkeypatch, found):
    def fake(root):
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(inst, "find_robots", fake)
    host, _ = inst.select_robot(quiet_ui(), Options(), ROOT)
    assert host == ReachyMiniWireless.default_host


def test_choose_reprompts_until_valid(monkeypatch):
    ui = Ui(interactive=True, assume_yes=False)
    answers = iter(["9", "x", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert ui.choose("Which robot?", ["a", "b"]) == 1


@pytest.mark.sim
def test_daemon_api_against_simulator():
    """The REST calls the installer relies on exist and answer as modelled in FakeDaemon."""
    real = DaemonApi("http://localhost:8000")
    status = real.daemon_status()
    assert status["state"] == "running" and status["simulation_enabled"]
    assert isinstance(real.installed_apps(), list)
    assert real.current_app() is None
    real.wake_up()
    assert real.startup_app() is None
    with pytest.raises(WizardError, match="HTTP 400"):
        real.set_startup_app("roboemot-not-installed")
