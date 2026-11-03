"""SPEC 4A, Packaging: the robot installer (roboemot.install) with the Reachy Mini Wireless
platform, against a fake daemon and a fake robot that emulates SSH, pip and the filesystem."""

from __future__ import annotations

import copy
import io
import json
import re
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import roboemot.install as inst
import yaml
from roboemot import paths
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Target
from roboemot.install import Options, Ui, WizardError
from roboemot.install import run as run_mod
from roboemot.install.build import with_extras
from roboemot.install.caches import cache_path, clear_caches
from roboemot.install.piper_lock import lock_name
from roboemot_reachymini import daemon as daemon_mod
from roboemot_reachymini import install as reachy
from roboemot_reachymini.daemon import INSTALL_TIMEOUT_S, DaemonClient, DaemonError
from roboemot_reachymini.install import ReachyMiniWireless
from sondalab_palette.term import MissingFlag, NoValidAnswer

ROOT = Path(__file__).resolve().parents[3]
RECORD = ReachyMiniWireless.record_path
APPS_PY = f"{reachy.APPS_VENV}/bin/python"
DAEMON_PY = "/venvs/mini_daemon/bin/python"
SITE = "/venvs/apps_venv/lib/python3.12/site-packages"
OWN = {"roboemot-core": "0.1.0.dev0", "roboemot-reachymini": "0.1.0.dev0"}
HOME = "/home/pollen"
DATA = f"{HOME}/.local/share/roboemot"
PIN_PATH = f"{DATA}/webui_pin.json"
APP_YAML = f"{DATA}/app.yaml"
PIPER_VENV = f"{DATA}/piper-venv"
PIPER_PYTHON = f"{PIPER_VENV}/bin/python"
PIN = "246810"
RID = "reachy-mini-wireless"


# ---------------------------------------------------------------- fakes


class FakeRobot:
    """Emulates what the installer runs over SSH: pip on a shared environment, the record file,
    temporary directories, and the Python probes."""

    def __init__(self) -> None:
        self.env = {
            "pip": "26.2.1",
            "numpy": "2.0.0",
            "httpx": "0.28.1",
            "websockets": "14.2",
            "reachy-mini": "1.11.0",
        }
        self.requires: dict[str, list[str]] = {"reachy-mini": ["numpy", "httpx", "websockets"]}
        self.deps = {"scipy": "1.18.1"}  # what installing the app adds
        self.conflict: dict[str, str] = {}  # upgrades the app needs (fails while pinned)
        self.files: dict[str, str] = {}
        self.tmp_dirs: set[str] = set()
        self.stray: dict[str, list[str]] = {}  # module -> leftover paths
        self.commands: list[str] = []
        self.streamed: list[str] = []
        self.copies: list[list[Path]] = []
        self.fail_install = False
        self.stdins: list[str | None] = []
        self.piper_pips: list[str] = []
        self.piper_importable = False
        self.machine = "aarch64\n3.12"  # architecture and Python minor the probe reports
        self.piper_pip_fails = False
        self.dirs: set[str] = {HOME}  # folders that exist (data folder emulation)
        self.symlinks: dict[str, str] = {}  # path -> where realpath says it really is
        self.env_data_dir = ""  # ROBOEMOT_DATA_DIR in the SSH user's environment
        self.secrets: dict[str, str] = {}  # the robot's secret store (name -> value)
        self.apps_venv = True  # a fresh robot has only the daemon venv
        self.apps_in_env: list[str] = []  # other apps' entry points in the apps venv
        self.record_at_venv: str | None = None  # the record file when the venv was created
        self.daemon_version = "1.11.0"
        self.sizes: dict[str, str] = {}  # `du -sh` answers
        self.daemon_venv_kb = 900_000  # `du -sk` of /venvs/mini_daemon
        self.fetches: list[str] = []  # `roboemot models fetch` argument lists
        self.fetch_fails = False  # True: every fetch fails; an int: that many fail first
        self.verified_models: list[str] = []  # "voice:<path>" files with a valid marker
        self.models_on_disk: dict[str, int] = {}  # bytes unverified files already hold
        self.retired_bytes = 0  # retired model files still in the models folder
        self.free_kb = 20_000_000
        self.free_kb_by_cache = self.free_kb  # free space while the uv cache is still there
        self.links_into_cache: list[str] = []  # symlinks under /venvs into ~/.cache/uv
        self.home_free_kb: int | None = None  # `df` below HOME (the models), else free_kb
        self.df_unreadable: set[str] = set()  # paths whose `df` answers nothing

    def _exists(self, path: str) -> bool:
        return (
            path in self.dirs or path in self.files or (path == reachy.APPS_VENV and self.apps_venv)
        )

    @property
    def app_yaml(self) -> str | None:
        """The app config in the data folder, as written."""
        return self.files.get(APP_YAML)

    # Remote interface
    target = "pollen@robot"

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def copy(self, files, dest) -> None:
        assert dest in self.tmp_dirs
        self.copies.append(files)

    def stream(self, command, on_line, check: bool = True, **kw) -> str:
        self.streamed.append(command)
        out = self.run(command, check=check, **kw)
        for line in out.splitlines():
            on_line(line)
        return out

    def run(self, command: str, check: bool = True, stdin: str | None = None) -> str:
        self.commands.append(command)
        if command.startswith(APPS_PY) and not self.apps_venv:
            raise AssertionError(f"the apps venv does not exist: {command}")
        if command.startswith(f"{DAEMON_PY} -m venv "):
            assert shlex.split(command)[-1] == reachy.APPS_VENV, command
            self.record_at_venv = self.files.get(RECORD)
            self.apps_venv, self.env = True, {"pip": "25.0.1"}
            return ""
        if command.startswith(f"{DAEMON_PY} -c") and "version('reachy_mini')" in " ".join(
            shlex.split(command)
        ):
            return self.daemon_version
        if " -m venv " in command:
            self.dirs |= {shlex.split(command)[-1], PIPER_PYTHON}
            return ""
        if command.startswith(f"{PIPER_PYTHON} -I -c"):
            return "yes" if self.piper_importable else "no"
        if command.startswith(f"{PIPER_PYTHON} -m pip "):
            assert "--no-cache-dir" in command, command
            self.piper_pips.append(command)
            if self.piper_pip_fails:
                return "ERROR: no network\n__rc=1"
            self.piper_importable = True
            return "\n__rc=0"
        if " -m roboemot.cli models fetch " in command:
            assert command.startswith(APPS_PY), command
            self.fetches.append(command.split(" -m roboemot.cli ")[1].split(" 2>&1")[0])
            if self.fetch_fails:
                if self.fetch_fails is not True:
                    self.fetch_fails -= 1
                return "ModelError: no network\n__rc=1"
            self.dirs |= (
                {f"{DATA}/voice-models", f"{DATA}/models"}
                if "--all" in command
                else {f"{DATA}/voice-models"}
            )
            return "downloaded silero\n__rc=0"
        if " -m roboemot.cli secrets " in command:
            words = shlex.split(command)
            action, names = words[4], words[8:]
            assert words[5:7] == ["--data-dir", DATA], command
            assert not names or words[7] == "--", command  # names are never read as options
            if action == "list":
                return "\n".join(sorted(self.secrets))
            if action == "set":
                self.stdins.append(stdin)
                self.secrets[names[0]] = (stdin or "").rstrip("\n")
                self.dirs.add(f"{DATA}/secrets.json")
            else:
                for n in names:
                    self.secrets.pop(n, None)
            return ""
        if "-m roboemot.webui set-pin" in command:
            self.stdins.append(stdin)
            self.dirs.add(PIN_PATH)
            return ""
        if " -m pip " in command:
            out, rc = self._pip(command)
            return f"{out}\n__rc={rc}"
        plain = " ".join(shlex.split(command)) if command.startswith("/venvs") else command
        if command.startswith("test -x"):
            return "yes" if self.apps_venv or APPS_PY not in command else "no"
        if command == 'printf %s "$HOME"':
            return HOME
        if command == 'printf %s "${ROBOEMOT_DATA_DIR:-}"':
            return self.env_data_dir
        if command.startswith("find /venvs -lname "):
            assert shlex.split(command)[3] == f"{UV_CACHE}/*", command
            return self.links_into_cache[0] if self.links_into_cache else ""
        if command.startswith("du -sk "):
            return str(self.daemon_venv_kb)
        if command.startswith("du -sh "):
            return self.sizes.get(shlex.split(command)[2], "4.0K")
        if command.startswith("realpath "):
            path = shlex.split(command)[1]
            return self.symlinks.get(path, path) if self._exists(path) else ""
        if command.startswith("test -e"):
            return "yes" if self._exists(shlex.split(command)[2]) else "no"
        if command.startswith("mkdir -p -m 700 "):
            path = shlex.split(command)[4]
            while path != HOME:
                self.dirs.add(path)
                path = path.rsplit("/", 1)[0]
            return ""
        if command.startswith("rmdir "):
            path = shlex.split(command)[1]
            if not any(d.startswith(path + "/") for d in self.dirs):
                self.dirs.discard(path)
            return ""
        if "check_file" in plain:
            return json.dumps(
                {
                    "done": self.verified_models,
                    "on_disk": self.models_on_disk,
                    "retired": self.retired_bytes,
                }
            )
        if "platform.machine" in plain:
            return self.machine
        if "version('reachy_mini')" in plain:
            if "reachy-mini" in self.env:
                return self.env["reachy-mini"]
            if command.endswith("|| true"):
                return ""
            raise WizardError(f"on robot: {command}\nPackageNotFoundError: reachy_mini")
        if "e.name !=" in plain:
            return "\n".join(self.apps_in_env)
        if "reachy_mini_apps" in plain:
            return "roboemot_reachymini.main"
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
            path = shlex.split(command)[2]
            if path in self.df_unreadable:
                return ""
            kb = self.free_kb_by_cache if UV_CACHE in self.dirs else self.free_kb
            if path.startswith(HOME) and self.home_free_kb is not None:
                kb = self.home_free_kb
            return f"/dev/root 30000000 1000000 {kb} 5% /"
        if command.startswith("cat "):
            return self.files.get(shlex.split(command)[1], "")
        if command.startswith("umask 077 && mkdir -p ") and "&& mv" in command:
            t = shlex.split(command)
            assert t[4] == "-p" and t[-1] == APP_YAML and "printf" not in t, command
            self.files[t[-1]] = stdin or ""
            return ""
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
            if path == reachy.APPS_VENV:
                self.apps_venv, self.env = False, {}
            self.dirs = {d for d in self.dirs if d != path and not d.startswith(path + "/")}
            self.files = {f: v for f, v in self.files.items() if not f.startswith(path + "/")}
            self.tmp_dirs.discard(path)
            self.stray = {m: [p for p in ps if p != path] for m, ps in self.stray.items()}
            self.stray = {m: ps for m, ps in self.stray.items() if ps}
            return ""
        raise AssertionError(f"unexpected robot command: {command}")

    def _pip(self, command: str) -> tuple[str, int]:
        assert "--no-cache-dir" in command, command
        pinned = "-c /dev/stdin" in command
        args = command.split(" -m pip ", 1)[1].split(" 2>&1", 1)[0]
        args = args.replace("--progress-bar off ", "")
        targets = [t for t in shlex.split(args) if not t.startswith("-")][1:]
        if args.startswith("list"):
            return json.dumps([{"name": n, "version": v} for n, v in self.env.items()]), 0
        if args.startswith("uninstall"):
            for t in targets:
                self.env.pop(t, None)
            return "", 0
        if not args.startswith("install"):
            raise AssertionError(args)
        if any(t.startswith("reachy-mini==") for t in targets):
            self.env["reachy-mini"] = targets[0].split("==")[1]
            return "", 0
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
        if any("[voice" in t for t in targets):
            changes["vosk"] = "0.3.45"
        if any("vision" in t.partition("[")[2] for t in targets):
            changes["opencv-python-headless"] = "4.10.0"
        realtime = any(
            "realtime" in t.partition("[")[2] or t.startswith("websockets") for t in targets
        )  # the core wheel with the extra, or the dry run's requirement list
        if realtime and "websockets" not in self.env:
            changes["websockets"] = "15.0.1"
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
        self.autostart_on_wake = False  # the startup app begins as soon as the robot wakes up
        self.lock: dict | None = {"state": "free", "holder_name": None}  # None: no endpoint
        self.restart_start_fails = False  # the stop runs, then the start raises (HTTP 400)
        self.restart_answer_s = 0.0  # the restart answers this late (the new start is under way)
        self.stopping_reads = 0  # the app reads stopping this many times, then is gone

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
                if self.current and self.current["state"] == "stopping" and self.stopping_reads:
                    self.stopping_reads -= 1
                    if not self.stopping_reads:
                        self.current = None
                        return 200, None
                if self.current and self.current["state"] == "starting" and self.start_states:
                    self.current["state"] = self.start_states.pop(0)
                return 200, self.current
            case "POST", "/api/apps/stop-current-app":
                if self.current and self.current["state"] == "stopping":  # daemon 1.13 stuck
                    return 400, {"detail": "No app is currently running"}
                self.current = None
                return 200, None
            case "POST", "/api/apps/restart-current-app":
                if not self.current:
                    return 400, {"detail": "No app is running"}
                if self.restart_start_fails:
                    self.current = None
                    return 400, {"detail": "Another app holds the robot"}
                self.current = {"info": self.current["info"], "state": "starting"}
                self.start_states = ["running"]
                threading.Event().wait(self.restart_answer_s)  # no_settle stubs time.sleep
                return 200, self.current
            case "POST", "/api/move/play/wake_up":
                if self.autostart_on_wake and not self.current:
                    self.current = {"info": {"name": "roboemot"}, "state": "starting"}
                return 200, {"uuid": "x"}
            case "GET", "/api/daemon/robot-app-lock-status":
                return (200, self.lock) if self.lock else (404, {"detail": "Not Found"})
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
    return DaemonClient(Target("127.0.0.1", daemon.port))


@pytest.fixture(autouse=True)
def no_settle(monkeypatch):
    monkeypatch.setattr(reachy.time, "sleep", lambda s: None)


@pytest.fixture(autouse=True)
def plain_terminal(monkeypatch):
    """Unicode glyphs, no colour, 100 columns, whatever the shell running the tests says."""
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "100")
    for name in ("TERM", "SONDALAB_ASCII", "SONDALAB_ACCESSIBLE", "ACCESSIBLE", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)


def flat(text: str) -> str:
    """Output with every whitespace run folded to one space: a step line reads
    ``label detail`` and a wrapped detail reads as one line."""
    return " ".join(text.split())


@pytest.fixture()
def offline(monkeypatch, robot, daemon):
    monkeypatch.setattr(inst, "load_platform", lambda rid: ReachyMiniWireless(port=daemon.port))
    monkeypatch.setattr(inst, "Remote", lambda user, host, control_dir: robot)

    def build(dirs, out):
        assert not (dirs[0].parent / reachy.PKG_DATA / "app.yaml").exists()  # not package data
        return [out / f"{d.name.replace('-', '_')}-0.1.whl" for d in dirs]

    monkeypatch.setattr(inst, "build_wheels", build)


def quiet_ui(yes: bool = False) -> Ui:
    return Ui(interactive=False, assume_yes=yes)


def options(**kw) -> Options:
    base = dict(
        host="127.0.0.1",
        user="pollen",
        robot="reachy-mini-wireless",
        dev=True,
        yes=True,
        vision=False,
        cloud_voice=False,
    )
    return Options(**{**base, **kw})


def install(**kw) -> None:
    inst.run_install(options(start=False, startup=False, **kw), quiet_ui(yes=True), ROOT)


def uninstall(**kw) -> None:
    inst.run_uninstall(options(uninstall=True, **kw), quiet_ui(yes=True), ROOT)


# ---------------------------------------------------------------- daemon API


def test_api_reads(api):
    assert api.status()["state"] == "running"
    assert api.installed_apps() == ["conversation"]
    assert api.current_app() is None
    assert api.startup_app() is None


def test_api_error_carries_daemon_detail(api, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    with pytest.raises(DaemonError, match="HTTP 400: An app is already running"):
        api.start_app("roboemot")


def test_api_unreachable_is_a_wizard_error_at_the_platform_boundary():
    wireless = ReachyMiniWireless(port=9)
    wireless.api = DaemonClient(Target("127.0.0.1", 9), timeout_s=0.5)
    with pytest.raises(WizardError, match="apps/startup-app"):
        wireless.autostart()
    with pytest.raises(WizardError, match="apps/list-available/installed"):
        wireless.verify_removed(quiet_ui())


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
    ui.confirm = lambda question, default, **kw: False
    with pytest.raises(WizardError, match="conversation is still running"):
        platform_for(daemon).start(ui)
    assert daemon.mutations == []


def test_start_waits_for_the_app_the_daemon_autostarts_on_wake_up(daemon, capsys):
    daemon.autostart_on_wake = True
    platform_for(daemon).start(quiet_ui(yes=True))
    assert [(m, p) for m, p, _ in daemon.mutations] == [("POST", "/api/move/play/wake_up")]
    assert daemon.current["state"] == "running"
    assert "started by the daemon on wake-up" in capsys.readouterr().out


def test_start_skips_to_the_url_when_the_app_already_runs(daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    platform_for(daemon).start(quiet_ui())
    assert daemon.mutations == []
    assert "roboemot is already running" in capsys.readouterr().out


VOICE_MODELS = f"{DATA}/voice-models"
UV_CACHE = f"{HOME}/.cache/uv"


def core_targets(robot) -> list[str]:
    """The roboemot-core wheel argument(s) of the main pip install, extras included."""
    for command in robot.commands:
        if " -m pip install" in command and "--no-deps" not in command and ".whl" in command:
            return [t for t in shlex.split(command) if "roboemot_core-" in t]
    return []


def test_install_adds_the_voice_extra_and_records_what_it_brings(offline, robot, daemon):
    install()
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice,realtime]"]
    assert {"vosk", "scipy"} <= set(json.loads(robot.files[RECORD])["added"])


def test_vision_is_on_by_default_and_no_vision_turns_it_off():
    assert inst.parse_args(["--yes"]).vision is True and Options().vision is True
    assert inst.parse_args(["--yes", "--no-vision"]).vision is False
    assert inst.parse_args(["--yes", "--vision"]).vision is True


def test_the_default_install_adds_the_vision_extra_and_the_vision_weights(offline, robot, daemon):
    inst.run_install(
        Options(
            host="127.0.0.1",
            user="pollen",
            robot="reachy-mini-wireless",
            dev=True,
            yes=True,
            start=False,
            startup=False,
            cloud_voice=False,
        ),
        quiet_ui(yes=True),
        ROOT,
    )
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice,realtime,vision]"]
    assert robot.fetches == ["models fetch --all"]


def test_the_dry_run_says_vision_will_be_installed(offline, robot, daemon, capsys):
    inst.run_install(
        Options(
            host="127.0.0.1",
            user="pollen",
            robot="reachy-mini-wireless",
            dev=True,
            yes=True,
            dry_run=True,
            cloud_voice=False,
        ),
        quiet_ui(yes=True),
        ROOT,
    )
    out = flat(capsys.readouterr().out)
    assert "○ Install the vision extra" in out and "voice and vision models" in out


def test_the_dry_run_with_no_vision_does_not_mention_vision(offline, robot, daemon, capsys):
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    # the Next: line repeats --no-vision; no step mentions vision
    assert not [ln for ln in out.splitlines() if "vision" in ln and "Next:" not in ln]


def test_not_enough_space_for_the_vision_weights_suggests_no_vision(offline, robot, daemon, capsys):
    robot.home_free_kb = needed_mib() * 1024
    install(vision=True)
    out = capsys.readouterr().out
    assert robot.fetches == [] and "--no-vision" in out


def test_vision_flag_adds_the_vision_extra_and_the_vision_weights(offline, robot, daemon):
    install(vision=True)
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice,realtime,vision]"]
    assert robot.fetches == ["models fetch --all"]
    assert "opencv-python-headless" in json.loads(robot.files[RECORD])["added"]


# ---------------------------------------------------------------- voice on by default (SPEC 4A)


def merged_config(dev, voice, existing=None):
    wanted, owned = inst.owned_app_config(ROOT, dev, voice)
    return inst.merge_app_config(existing or {}, wanted, owned)


def test_voice_is_on_by_default_and_no_voice_turns_it_off():
    assert inst.parse_args(["--yes"]).voice is True and Options().voice is True
    assert inst.parse_args(["--yes", "--no-voice"]).voice is False
    assert inst.parse_args(["--yes", "--voice"]).voice is True


def test_the_app_config_carries_voice_and_the_dev_content():
    cfg = merged_config(dev=True, voice=True)
    assert cfg["voice"] == {"enabled": True}
    assert cfg["safety"] == {"allow_unmeasured_dynamics": True}


def test_without_dev_the_app_config_is_voice_only():
    assert merged_config(dev=False, voice=True) == {"schema_version": 1, "voice": {"enabled": True}}


def test_no_voice_with_dev_keeps_the_dev_content_only():
    cfg = merged_config(dev=True, voice=False)
    assert "voice" not in cfg and cfg["safety"] == {"allow_unmeasured_dynamics": True}


def test_no_voice_without_dev_has_no_app_config():
    assert merged_config(dev=False, voice=False) == {}


OWNER_CONFIG = {
    "schema_version": 1,
    "webui": {"port": 8080},
    "voice": {"enabled": True, "language": "en"},
    "safety": {"allow_unmeasured_dynamics": True, "max_speed_scale": 0.5},
}


def test_the_merge_keeps_every_key_the_installer_does_not_own():
    """R4 S4: owner edits survive; only voice.enabled and the dev flag follow the flags."""
    cfg = merged_config(dev=False, voice=False, existing=OWNER_CONFIG)
    assert cfg == {
        "schema_version": 1,
        "webui": {"port": 8080},
        "voice": {"language": "en"},
        "safety": {"max_speed_scale": 0.5},
    }
    assert OWNER_CONFIG["voice"]["enabled"] is True  # the input is not changed


def test_the_written_app_config_has_a_header_naming_what_the_installer_owns():
    text = inst.app_config_text(merged_config(dev=True, voice=True))
    assert text.startswith("# RoboEmot app config") and "voice.enabled" in text.splitlines()[0]
    assert yaml.safe_load(text)["voice"] == {"enabled": True}


def test_a_reinstall_keeps_the_owners_edits_in_the_data_folder(offline, robot, daemon):
    """R4 S4: the app config lives in the data folder and a reinstall changes only the keys
    the installer owns."""
    install()
    cfg = yaml.safe_load(robot.app_yaml)
    cfg["webui"] = {"port": 8080}
    robot.files[APP_YAML] = yaml.safe_dump(cfg)
    install(dev=False)
    cfg = yaml.safe_load(robot.app_yaml)
    assert cfg["webui"] == {"port": 8080} and cfg["voice"] == {"enabled": True}
    assert "safety" not in cfg


def test_an_app_config_created_in_a_foreign_data_folder_is_recorded_and_removed(
    offline, robot, daemon
):
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}  # predates the install
    install()
    record = json.loads(robot.files[RECORD])
    assert record["data_dir"] is None and record["app_config"] == APP_YAML
    uninstall()
    assert APP_YAML not in robot.files and DATA in robot.dirs


def test_an_app_config_the_owner_had_is_recorded_and_uninstall_removes_it_whole(
    offline, robot, daemon, capsys
):
    """Review SF2, owner decision 2026-10-22: the file the installer wrote goes whole, said both
    at install and at uninstall; no per-key restore."""
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    robot.files[APP_YAML] = yaml.safe_dump({"schema_version": 1, "webui": {"port": 8080}})
    install()
    assert json.loads(robot.files[RECORD])["app_config"] == APP_YAML
    assert "uninstall removes it whole" in flat(capsys.readouterr().out)
    uninstall()
    assert APP_YAML not in robot.files
    assert f"✓ Removed the app config {APP_YAML}, settings saved from the web UI included" in (
        flat(capsys.readouterr().out)
    )


def test_the_uninstall_dry_run_names_the_app_config_it_would_remove(offline, robot, daemon, capsys):
    """Review MT3."""
    install()
    capsys.readouterr()
    uninstall(dry_run=True)
    assert f"○ Remove the app config {APP_YAML}, settings saved from the web UI included" in (
        flat(capsys.readouterr().out)
    )
    assert APP_YAML in robot.files


@pytest.mark.parametrize("case", ["symlink", "elsewhere"])
def test_uninstall_refuses_an_app_config_it_cannot_trust(offline, robot, daemon, capsys, case):
    """Review MT3: a symlinked file, or a recorded path that is not the computed one, stays."""
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    install()
    record = json.loads(robot.files[RECORD])
    if case == "symlink":
        robot.symlinks[APP_YAML] = "/etc/shadow-copy"
    else:
        record["app_config"] = "/etc/app.yaml"
        robot.files[RECORD] = json.dumps(record)
    if case == "symlink":  # verification then keeps the record, as for the data folder
        with pytest.raises(WizardError, match="still present"):
            uninstall()
    else:
        uninstall()
    assert APP_YAML in robot.files
    assert ("symlink?" if case == "symlink" else "left as it is") in capsys.readouterr().out


def test_an_install_dry_run_over_an_owner_app_config_names_the_keys_and_writes_nothing(
    offline, robot, daemon, capsys
):
    """Review MT2."""
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    owner = yaml.safe_dump({"schema_version": 1, "webui": {"port": 8080}})
    robot.files[APP_YAML] = owner
    install(dry_run=True)
    out = flat(capsys.readouterr().out)
    assert (
        f"○ Write the app config {APP_YAML} (safety.allow_unmeasured_dynamics=true, "
        "voice.enabled=true" in out
    )
    assert robot.files[APP_YAML] == owner and RECORD not in robot.files


def test_an_app_config_that_is_not_yaml_stops_the_install(offline, robot, daemon):
    """Review SF1/MT4: a broken app.yaml is refused before the auto-stop, the caches and pip."""
    robot.files[APP_YAML] = "voice: [unclosed\n"
    robot.dirs.add(UV_CACHE)
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    with pytest.raises(WizardError, match="not valid YAML"):
        install(clear_caches=True)
    assert "roboemot-core" not in robot.env and RECORD not in robot.files
    assert UV_CACHE in robot.dirs and daemon.current["state"] == "running"
    assert not any(" -m pip install" in c for c in robot.commands)


def test_an_app_config_that_is_not_a_mapping_stops_the_install(offline, robot, daemon):
    robot.files[APP_YAML] = "- a list\n"
    with pytest.raises(WizardError, match="not a mapping"):
        install()
    assert "roboemot-core" not in robot.env


def test_the_default_install_enables_voice_in_the_app_config(offline, robot, daemon):
    install()
    cfg = yaml.safe_load(robot.app_yaml)
    assert cfg["voice"] == {"enabled": True}
    assert cfg["safety"]["allow_unmeasured_dynamics"] is True


def test_install_without_dev_still_writes_the_voice_config(offline, robot, daemon):
    install(dev=False)
    assert yaml.safe_load(robot.app_yaml) == {"schema_version": 1, "voice": {"enabled": True}}


def test_no_voice_writes_no_voice_key_and_no_file_without_dev(offline, robot, daemon):
    install(dev=False, voice=False)
    assert robot.app_yaml is None


def test_no_voice_defaults_piper_and_the_models_to_off(offline, robot, daemon):
    install(voice=False, vision=False)
    assert robot.piper_pips == [] and PIPER_PYTHON not in robot.dirs
    assert robot.fetches == [] and VOICE_MODELS not in robot.dirs
    assert "voice" not in yaml.safe_load(robot.app_yaml)


def test_no_voice_does_not_even_ask_about_piper_or_the_models(offline, robot, daemon, monkeypatch):
    asked = []
    monkeypatch.setattr(Ui, "confirm", lambda self, q, default: asked.append(q) or True)
    inst.run_install(options(voice=False, start=False, startup=False, yes=False), quiet_ui(), ROOT)
    assert not [q for q in asked if q.startswith(("Install Piper", "Download the"))]


def test_no_voice_with_vision_still_fetches_the_vision_weights(offline, robot, daemon):
    install(voice=False, vision=True)
    assert robot.fetches == ["models fetch --all"] and robot.piper_pips == []


def test_no_voice_installs_piper_and_the_models_when_asked_explicitly(offline, robot, daemon):
    install(voice=False, piper=True, models=True)
    assert len(robot.piper_pips) == 1 and robot.fetches == ["models fetch --voice"]


def test_voice_on_with_piper_and_models_declined_warns_the_robot_will_not_speak_or_listen(
    offline, robot, daemon, capsys
):
    install(piper=False, models=False)
    out = capsys.readouterr().out
    assert "the robot will not speak" in out and "the robot will not listen" in out


def test_voice_on_with_everything_installed_has_no_voice_warning(offline, robot, daemon, capsys):
    install()
    out = capsys.readouterr().out
    assert "will not speak" not in out and "will not listen" not in out


def test_no_voice_warns_about_nothing_declined(offline, robot, daemon, capsys):
    install(voice=False)
    out = capsys.readouterr().out
    assert "will not speak" not in out and "will not listen" not in out


def test_the_dry_run_prints_what_the_app_config_will_contain_and_writes_nothing(
    offline, robot, daemon, capsys
):
    before = dict(robot.files), set(robot.dirs)
    inst.run_install(options(dry_run=True, start=False, startup=False), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert "voice.enabled=true" in out and "safety.allow_unmeasured_dynamics=true (dev)" in out
    assert (dict(robot.files), set(robot.dirs)) == before


def test_the_dry_run_with_no_voice_and_no_dev_says_there_are_no_installer_keys(
    offline, robot, daemon, capsys
):
    inst.run_install(
        options(dry_run=True, dev=False, voice=False, start=False, startup=False),
        quiet_ui(yes=True),
        ROOT,
    )
    out = capsys.readouterr().out
    assert "no installer keys" in out and "voice.enabled" not in out
    assert APP_YAML not in robot.files


def test_verification_fails_when_the_app_config_lacks_voice(offline, robot, monkeypatch):
    monkeypatch.setattr(inst, "app_config_text", lambda config: "schema_version: 1\n")
    with pytest.raises(WizardError, match="has voice.enabled=None, expected voice.enabled=True"):
        install(dev=False)


def test_verification_fails_when_a_no_voice_install_finds_voice_in_the_app_config(
    offline, robot, monkeypatch
):
    real = inst.app_config_text
    monkeypatch.setattr(
        inst, "app_config_text", lambda config: real({**config, "voice": {"enabled": True}})
    )
    with pytest.raises(WizardError, match="expected voice.enabled=None"):
        install(voice=False)


def test_verification_accepts_the_app_config_as_asked_in_every_combination(offline, robot, daemon):
    for dev in (True, False):
        for voice in (True, False):
            install(dev=dev, voice=voice, piper=False, models=False)


def test_uninstall_after_a_default_install_restores_the_robot_and_removes_the_app_config(
    offline, robot, daemon
):
    before = snapshot(robot, daemon)
    install()
    assert json.loads(robot.files[RECORD])["app_config"] == APP_YAML
    uninstall()
    assert snapshot(robot, daemon) == before


def test_models_are_fetched_on_the_robot_and_uninstall_removes_them(offline, robot, daemon):
    before = snapshot(robot, daemon)
    install()
    assert robot.fetches == ["models fetch --voice"]
    assert VOICE_MODELS in robot.dirs
    uninstall()
    assert snapshot(robot, daemon) == before


def test_models_can_be_declined_with_the_flag(offline, robot, daemon):
    install(models=False)
    assert robot.fetches == [] and VOICE_MODELS not in robot.dirs


def test_the_models_question_defaults_to_no(offline, robot, daemon, monkeypatch):
    # main's Term rule: a confirmation that changes the robot defaults to no (it was yes)
    asked = []
    monkeypatch.setattr(
        Ui,
        "confirm",
        lambda self, q, default, **kw: asked.append((q, default, kw)) or q.startswith("Download"),
    )
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    # interactive: off a terminal the question is not asked at all (--models is required)
    ui = Ui(interactive=True, assume_yes=False)
    inst.run_install(options(start=False, startup=False, yes=False), ui, ROOT)
    models = [(d, kw) for q, d, kw in asked if q.startswith("Download the voice models")]
    assert models == [(False, {"flag": "--models"})]


def test_not_enough_space_warns_and_skips_the_models_without_failing(
    offline, robot, daemon, capsys
):
    robot.home_free_kb = (needed_mib() - 100) * 1024
    install()
    out = flat(capsys.readouterr().out)
    assert robot.fetches == [] and "Skipped the models" in out and "MiB free" in out
    assert reachy.APPS_VENV and "roboemot-core" in robot.env


def app_need_mb(extras: tuple[str, ...] = ReachyMiniWireless.extras) -> int:
    """MB the app install needs on the apps venv's filesystem (downloads and unpacked), with the
    venv already there."""
    return sum(inst.install_need_mb(ReachyMiniWireless(), None, extras, creating=False))


def test_the_space_check_counts_the_extras(offline, robot, capsys):
    """R4 S3: the free-space floor comes from what is installed: vision adds its share."""
    robot.free_kb = (app_need_mb() + 1) * 1024
    with pytest.raises(
        WizardError, match=f"{app_need_mb((*ReachyMiniWireless.extras, 'vision'))} MB needed"
    ):
        install(vision=True)
    assert "roboemot-core" not in robot.env


def test_a_new_apps_venv_counts_the_daemon_venv_tree(offline, robot, capsys):
    """R4 S3: creating the apps venv installs reachy-mini's tree; the daemon venv's size
    (`du`) stands in for it, plus its downloads."""
    fresh_robot(robot)
    robot.daemon_venv_kb = 1_000 * 1024
    tree = 1_000 + int(1_000 * inst.ENV_DOWNLOAD_SHARE)
    robot.free_kb = (app_need_mb() + tree - 1) * 1024
    with pytest.raises(WizardError, match=f"{app_need_mb() + tree} MB needed"):
        install()
    assert robot.files == {} and not robot.apps_venv


def test_a_too_old_python_is_refused_before_the_running_app_is_stopped(offline, robot, daemon):
    """R4 S5: the interpreter, architecture and Python checks come before the auto-stop."""
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.machine = "aarch64\n3.9"
    with pytest.raises(WizardError, match="Target Python too old: Python 3.9"):
        install()
    assert STOP not in mutation_paths(daemon) and daemon.current["state"] == "running"


def test_the_auto_stop_and_the_cache_step_come_before_the_space_check(offline, robot, daemon):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.dirs.add(UV_CACHE)
    install(clear_caches=True)
    cmds = robot.commands
    probe = next(i for i, c in enumerate(cmds) if "platform.machine" in c)
    clear = cmds.index(f"rm -rf {UV_CACHE}")
    space = next(i for i, c in enumerate(cmds) if c.startswith("df -Pk /tmp"))
    assert probe < clear < space and STOP in mutation_paths(daemon)


def needed_mib(vision: bool = False) -> int:
    return -(-inst.models_bytes(vision) // inst.MIB)


def test_the_models_fit_exactly_when_free_mib_equals_the_need(offline, robot, daemon):
    robot.home_free_kb = needed_mib() * 1024
    install()
    assert robot.fetches == ["models fetch --voice"]


def test_one_mib_short_skips_the_models(offline, robot, daemon):
    robot.home_free_kb = (needed_mib() - 1) * 1024
    install()
    assert robot.fetches == []


def test_unreadable_free_space_skips_the_models_and_the_install_goes_on(
    offline, robot, daemon, capsys
):
    robot.df_unreadable = {DATA}
    install()
    assert robot.fetches == [] and "roboemot-core" in robot.env
    assert "cannot read the free space" in capsys.readouterr().out


def test_a_failed_download_warns_and_the_install_goes_on(offline, robot, daemon, capsys):
    robot.fetch_fails = True
    install()
    assert "Skipped the models voice models: the download failed" in flat(capsys.readouterr().out)
    assert "roboemot-core" in robot.env


def voice_files(robot, share: float) -> list[str]:
    """The first ``share`` of the voice model files, as the robot's verified list names them."""
    from roboemot.voice import models as voice_models

    files = [f"voice:{f.path}" for m in voice_models.load_manifest().values() for f in m.files]
    return files[: int(len(files) * share)]


def test_a_rerun_after_a_partial_fetch_needs_only_the_rest(offline, robot, daemon, capsys):
    """R4 S2/T2: files already verified on the robot are not counted again, so a disk that
    cannot hold the full peak still fits the rest of an interrupted fetch."""
    robot.verified_models = voice_files(robot, 0.6)
    rest = -(-inst.models_bytes(False, done=frozenset(robot.verified_models)) // inst.MIB)
    assert 0 < rest < needed_mib()
    robot.home_free_kb = rest * 1024
    install()
    assert robot.fetches == ["models fetch --voice"]
    assert f"need {rest} MiB" in capsys.readouterr().out


def test_bytes_already_on_disk_are_taken_off_the_need(offline, robot, daemon, capsys):
    """Review (S2 caveat): a stale partial download is freed before the new one, and a complete
    file not yet marked is kept, so neither is counted again."""
    from roboemot.voice import models as voice_models

    big = max(
        (f for m in voice_models.load_manifest().values() for f in m.files),
        key=lambda f: f.size_bytes or 0,
    )
    robot.models_on_disk = {f"voice:{big.path}": big.size_bytes}
    rest = -(-inst.models_bytes(False, on_disk=robot.models_on_disk) // inst.MIB)
    assert rest < needed_mib()
    robot.home_free_kb = rest * 1024
    install()
    assert robot.fetches == ["models fetch --voice"]


def test_with_every_model_verified_there_is_nothing_to_download(offline, robot, daemon, capsys):
    robot.verified_models = voice_files(robot, 1.0)
    robot.home_free_kb = 1024
    install()
    assert robot.fetches == []
    assert "nothing to download" in capsys.readouterr().out


def test_retired_model_files_are_reported_never_deleted(offline, robot, daemon, capsys):
    """Review ZN N3: the age model (10.5 MB) left on a robot is pointed at `models prune`; the
    installer itself deletes nothing it did not create."""
    robot.verified_models = voice_files(robot, 1.0)
    robot.retired_bytes = 10_513_214
    install()
    out = flat(capsys.readouterr().out)
    assert "Retired model files 11 MiB still on the robot" in out and "roboemot models prune" in out
    assert not any("prune" in f for f in robot.fetches)


def test_a_failed_download_is_tried_once_more(offline, robot, daemon, capsys):
    robot.fetch_fails = 1
    install()
    out = capsys.readouterr().out
    assert robot.fetches == ["models fetch --voice"] * 2
    assert "trying once more" in out and "were not downloaded" not in out


def test_dry_run_reports_the_model_sizes_and_downloads_nothing(offline, robot, daemon, capsys):
    install(dry_run=True)
    assert robot.fetches == [] and VOICE_MODELS not in robot.dirs
    assert "○ Download the models voice models:" in flat(capsys.readouterr().out)
    assert robot.files == {}


def test_models_in_a_foreign_data_folder_are_recorded_and_removed_alone(offline, robot, daemon):
    robot.dirs.add(DATA)
    install()
    assert json.loads(robot.files[RECORD])["models_dirs"] == [VOICE_MODELS]
    uninstall()
    assert DATA in robot.dirs and VOICE_MODELS not in robot.dirs


def test_models_folder_that_was_already_there_is_not_recorded(offline, robot, daemon):
    robot.dirs |= {DATA, VOICE_MODELS}
    install()
    assert json.loads(robot.files[RECORD])["models_dirs"] == []
    uninstall()
    assert VOICE_MODELS in robot.dirs


def test_model_sizes_come_from_the_pinned_manifests():
    assert 0 < inst.models_bytes(False) < inst.models_bytes(True)
    from roboemot.vision import models as vision_models

    vision = sum(f.size_bytes for s in vision_models.load_manifest().values() for f in s.files)
    assert inst.models_bytes(True) - inst.models_bytes(False) == vision


def test_with_extras_marks_only_the_core_wheel():
    files = ["/tmp/x/roboemot_core-0.1.0-py3-none-any.whl", "/tmp/x/roboemot_reachymini-0.1.0.whl"]
    assert with_extras(files, ("voice", "vision")) == [
        files[0] + "[voice,vision]",
        files[1],
    ]
    assert with_extras(files, ()) == files


def test_install_clears_the_uv_cache_before_installing(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.sizes[UV_CACHE] = "5.6G"
    install(clear_caches=True)
    assert UV_CACHE not in robot.dirs
    assert f"rm -rf {UV_CACHE}" in robot.commands
    assert robot.commands.index(f"rm -rf {UV_CACHE}") < robot.commands.index(
        "mktemp -d /tmp/roboemot-install.XXXXXX"
    )
    out = flat(capsys.readouterr().out)
    assert f"✓ Cleared a cache {UV_CACHE} (5.6G)" in out
    assert UV_CACHE not in json.loads(robot.files[RECORD]).values()


def test_yes_alone_does_not_clear_the_cache(offline, robot, capsys):
    """R4 S6: the cache is the one change uninstall cannot undo; -y does not consent to it."""
    robot.dirs.add(UV_CACHE)
    install()
    assert UV_CACHE in robot.dirs and "--clear-caches does" in capsys.readouterr().out


def test_a_cache_an_environment_links_into_is_kept(offline, robot, capsys):
    """R4 S6: a venv made by uv with symlinks into its cache would break if it went."""
    robot.dirs.add(UV_CACHE)
    robot.links_into_cache = ["/venvs/mini_daemon/lib/python3.12/site-packages/numpy"]
    install(clear_caches=True)
    assert UV_CACHE in robot.dirs and f"rm -rf {UV_CACHE}" not in robot.commands
    assert "links into it" in capsys.readouterr().out


def test_install_keeps_the_uv_cache_when_declined(offline, robot):
    robot.dirs.add(UV_CACHE)
    install(clear_caches=False)
    assert UV_CACHE in robot.dirs


@pytest.mark.parametrize("interactive", [True, False])
def test_the_cache_prompt_is_asked_only_on_a_terminal_and_defaults_to_no(
    robot, monkeypatch, interactive
):
    # main's Term rule: a confirmation that changes the robot defaults to no (it was yes)
    robot.dirs.add(UV_CACHE)
    asked = []
    monkeypatch.setattr(
        Ui, "confirm", lambda self, q, default, **kw: asked.append((q, default, kw)) or False
    )
    ui = Ui(interactive=interactive, assume_yes=False)
    clear_caches(ui, robot, HOME, ReachyMiniWireless.clearable_caches, None, False)
    assert UV_CACHE in robot.dirs
    expected = [(False, {"flag": "--clear-caches"})] if interactive else []
    assert [(d, kw) for q, d, kw in asked if UV_CACHE in q] == expected


def test_an_unattended_run_without_consent_keeps_the_cache(offline, robot):
    robot.dirs.add(UV_CACHE)
    opts = options(start=False, startup=False, yes=False, piper=False, models=False)
    inst.run_install(opts, quiet_ui(), ROOT)
    assert UV_CACHE in robot.dirs


def test_the_flag_clears_the_cache_without_yes(offline, robot):
    robot.dirs.add(UV_CACHE)
    opts = options(start=False, startup=False, yes=False, clear_caches=True)
    inst.run_install(
        Options(**{**opts.__dict__, "piper": False, "models": False}), quiet_ui(), ROOT
    )
    assert UV_CACHE not in robot.dirs


def test_clearing_the_cache_can_make_a_full_robot_pass_the_space_check(offline, robot):
    robot.dirs.add(UV_CACHE)
    robot.free_kb_by_cache = (app_need_mb() - 100) * 1024
    install(clear_caches=True)
    assert UV_CACHE not in robot.dirs and "roboemot-core" in robot.env


def test_a_declined_clear_leaves_the_free_space_error(offline, robot):
    robot.dirs.add(UV_CACHE)
    robot.free_kb_by_cache = (app_need_mb() - 100) * 1024
    with pytest.raises(WizardError, match="MB free"):
        install(clear_caches=False)
    assert UV_CACHE in robot.dirs and "roboemot-core" not in robot.env


def test_install_dry_run_reports_the_uv_cache_and_clears_nothing(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.sizes[UV_CACHE] = "5.6G"
    install(dry_run=True, clear_caches=True)
    assert UV_CACHE in robot.dirs
    assert not any(c.startswith("rm -rf " + UV_CACHE) for c in robot.commands)
    assert f"○ Clear a cache {UV_CACHE} (5.6G" in flat(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("kw", "said"),
    [
        ({}, "-y does not clear it"),
        ({"clear_caches": False}, "kept"),
    ],
)
def test_the_dry_run_says_what_the_real_run_does_with_the_cache(offline, robot, capsys, kw, said):
    """Review SF3/MT5: `--dry-run -y` reports the cache as kept, as the real `-y` run keeps it."""
    robot.dirs.add(UV_CACHE)
    install(dry_run=True, **kw)
    out = capsys.readouterr().out
    assert said in out and "would clear" not in out


def test_the_dry_run_reports_a_linked_cache_as_kept(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.links_into_cache = ["/venvs/mini_daemon/lib/x"]
    install(dry_run=True, clear_caches=True)
    out = capsys.readouterr().out
    assert "links into it" in out and "would clear" not in out


def test_the_free_space_error_names_the_kept_cache_and_the_flag(offline, robot):
    """Review S-4: a full robot whose cache was kept (-y alone) is told how to free it."""
    robot.dirs.add(UV_CACHE)
    robot.sizes[UV_CACHE] = "5.6G"
    robot.free_kb_by_cache = (app_need_mb() - 100) * 1024
    with pytest.raises(WizardError, match=r"kept .*\.cache/uv \(5\.6G\): --clear-caches frees it"):
        install()


def test_install_does_not_delete_a_cache_that_is_a_symlink(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.symlinks[UV_CACHE] = "/data/elsewhere"
    install(clear_caches=True)
    assert UV_CACHE in robot.dirs
    assert "symlink" in capsys.readouterr().out


@pytest.mark.parametrize("spec", ["/etc", "~/", "~/..", "~/../x", "cache"])
def test_cache_paths_stay_strictly_below_home(spec):
    with pytest.raises(WizardError):
        cache_path(HOME, spec)


@pytest.mark.parametrize("home", ["/", ""])
def test_cache_paths_refuse_a_root_or_empty_home(home):
    with pytest.raises(WizardError):
        cache_path(home, "~/.cache/uv")


def test_a_symlinked_home_skips_the_cache_with_a_warning(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.symlinks[UV_CACHE] = "/mnt/real-home/.cache/uv"
    install(clear_caches=True)
    assert UV_CACHE in robot.dirs
    assert f"rm -rf {UV_CACHE}" not in robot.commands
    assert "symlink" in capsys.readouterr().out


def test_start_reports_an_error_of_the_app_the_daemon_autostarted(daemon):
    daemon.autostart_on_wake = True
    daemon.start_states = ["error"]
    with pytest.raises(WizardError, match="roboemot state error"):
        platform_for(daemon).start(quiet_ui(yes=True))


def test_start_clears_an_earlier_error_then_starts(daemon, capsys):
    # the daemon counts an errored app as running and refuses every start until it is stopped
    daemon.current = {"info": {"name": "roboemot"}, "state": "error", "error": "boom"}
    platform_for(daemon).start(quiet_ui())
    assert [(m, p) for m, p, _ in daemon.mutations] == [
        ("POST", "/api/apps/stop-current-app"),
        ("POST", "/api/move/play/wake_up"),
        ("POST", "/api/apps/start-app/roboemot"),
    ]
    assert daemon.current["state"] == "running"
    assert "roboemot had an error boom" in flat(capsys.readouterr().out)


def test_start_reports_app_error(daemon):
    daemon.start_states = ["error"]
    with pytest.raises(WizardError, match="roboemot state error"):
        platform_for(daemon).start(quiet_ui())


def test_stage_packages_leaves_checkout_untouched(tmp_path):
    repo_app_yaml = ROOT / "packages" / reachy.PKG_DATA / "app.yaml"
    existed = repo_app_yaml.exists()
    dirs = reachy.stage_packages(ROOT, tmp_path)
    data = tmp_path / reachy.PKG_DATA
    assert [d.name for d in dirs] == ["roboemot-core", "roboemot-reachymini"]
    assert not (data / "app.yaml").exists()  # the app config is not package data
    assert (data / "reachy-mini-wireless.yaml").exists()
    assert {p.name for p in (data / "presets").glob("*.yaml")} >= {"default.yaml"}
    assert repo_app_yaml.exists() is existed


def test_installer_registered_under_the_manifest_id():
    assert "reachy-mini-wireless" in inst.installed_platforms()
    assert isinstance(inst.load_platform("reachy-mini-wireless"), ReachyMiniWireless)
    with pytest.raises(WizardError, match="No installer for desk-head-v1"):
        inst.load_platform("desk-head-v1")


# ---------------------------------------------------------------- install


def test_install_dry_run_touches_nothing(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    env = dict(robot.env)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert daemon.mutations == [] and robot.copies == [] and robot.files == {}
    assert robot.env == env and robot.tmp_dirs == set()
    writes = ("rm ", "rmdir", "mkdir", "mktemp", "mv ")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]
    assert robot.dirs == {HOME}
    out = capsys.readouterr().out
    assert "○ Add dependencies scipy 1.18.1" in flat(out)
    assert f"○ Create the data folder {DATA}" in flat(out)
    assert "○ Stop conversation" in out and "○ Start roboemot" in out
    assert "✓ Dry run passed" in out and "Summary  " in out
    assert (
        "Next: python scripts/install_on_robot.py --host 127.0.0.1 --user pollen "
        "--robot reachy-mini-wireless --dev --piper --no-cloud-voice --no-vision --models --yes"
    ) in out
    assert not [ln for ln in out.splitlines() if "✓" in ln and ("Create" in ln or "Add " in ln)]


def test_install_records_changes_and_cleans_tmp(offline, robot, daemon):
    install()
    assert len(robot.copies) == 2 and robot.tmp_dirs == set()  # the wheels, then the Piper lock
    record = json.loads(robot.files[RECORD])
    assert record["added"] == ["scipy", "vosk"] and record["replaced"] == {}
    assert record["pending_before"] is None
    pip = [
        c
        for c in robot.commands
        if " -m pip install" in c and "--dry-run" not in c and PIPER_VENV not in c
    ]
    assert pip and all("-c /dev/stdin" in c for c in pip if "--no-deps" not in c)


def test_install_starts_and_sets_autostart_recording_the_previous_one(offline, robot, daemon):
    daemon.startup = "conversation"
    inst.run_install(options(start=True, startup=True), quiet_ui(yes=True), ROOT)
    assert daemon.current["state"] == "running" and daemon.startup == "roboemot"
    record = json.loads(robot.files[RECORD])
    assert record["autostart_before"] == "conversation" and record["autostart_recorded"]


def test_install_upgrades_shared_packages_only_when_allowed(offline, robot, daemon):
    robot.conflict = {"numpy": "2.1.0"}
    ui = typed()  # a terminal: the question is asked, after the wheels are copied
    ui.confirm = lambda question, default, **kw: "shared packages" not in question
    with pytest.raises(WizardError, match="shared packages left as they were"):
        inst.run_install(options(start=False, startup=False, piper=False, yes=False), ui, ROOT)
    assert robot.env["numpy"] == "2.0.0" and "roboemot-core" not in robot.env
    install()
    assert robot.env["numpy"] == "2.1.0"
    assert json.loads(robot.files[RECORD])["replaced"] == {
        "numpy": {"before": "2.0.0", "after": "2.1.0"}
    }


def test_install_fails_when_dev_config_mismatches(offline, robot, monkeypatch):
    monkeypatch.setattr(inst, "app_config_text", lambda config: "voice: {enabled: true}\n")
    with pytest.raises(WizardError, match="expected safety.allow_unmeasured_dynamics=True"):
        install()


def test_install_fails_when_daemon_does_not_list_app(offline, daemon):
    daemon.hide_own = True
    with pytest.raises(WizardError, match="does not list roboemot"):
        install()


def test_main_returns_1_and_prints_reason(offline, robot, capsys, monkeypatch):
    monkeypatch.setattr(inst, "app_config_text", lambda config: "voice: {enabled: true}\n")
    rc = inst.main(
        ["--host", "127.0.0.1", "--user", "pollen", "--robot", "reachy-mini-wireless", "--dev",
         "--no-start", "--no-startup-app", "--no-piper", "--no-models", "--no-cloud-voice"],
        root=ROOT,
    )  # fmt: skip
    assert rc == 1
    # the error block is on stderr; the epic's app-config check replaces main's data/app.yaml
    # check (expected dev=True), which went with the package-data config
    err = flat(capsys.readouterr().err)
    assert "App config not as asked" in err
    assert "expected safety.allow_unmeasured_dynamics=True" in err


# ---------------------------------------------------------------- uninstall


def snapshot(robot, daemon) -> tuple:
    return (
        dict(robot.env),
        dict(robot.files),
        set(robot.tmp_dirs),
        set(robot.dirs),
        daemon.startup,
        daemon.installed,
        dict(robot.secrets),
        robot.apps_venv,
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
    assert "○ Stop roboemot" in out and "○ Restore the autostart app none" in flat(out)
    assert (
        "○ Remove packages roboemot-core 0.1.0.dev0, roboemot-reachymini 0.1.0.dev0, scipy 1.18.1"
        in flat(out)
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
    assert snapshot(robot, daemon)[:4] == before[:4]


def test_uninstall_removes_leftover_modules(offline, robot, daemon):
    robot.stray = {"roboemot": [f"{SITE}/roboemot"]}
    uninstall()
    assert robot.stray == {}
    assert f"rm -rf {SITE}/roboemot" in robot.commands


def test_uninstall_never_deletes_outside_site_packages_or_tmp(offline, robot, daemon):
    """Paths come from the robot; a crafted one must not escape (security review finding)."""
    robot.stray = {"roboemot": [f"{SITE}/../../../../etc", SITE, "/home/pollen"]}
    robot.tmp_dirs = {"/tmp/roboemot-install.x/../../home", "/tmp/roboemot-install.ok1"}
    uninstall()
    removed = [c for c in robot.commands if c.startswith("rm -rf")]
    assert removed == ["rm -rf /tmp/roboemot-install.ok1"]


def test_uninstall_keeps_the_record_when_verification_fails(offline, robot, daemon, monkeypatch):
    install()
    monkeypatch.setattr(robot, "_pip", _pip_ignoring_uninstall(robot._pip))
    with pytest.raises(WizardError, match="Uninstall incomplete"):
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


# ---------------------------------------------------------------- data folder (SPEC 4A, 12)


def test_install_creates_and_records_the_data_folder(offline, robot, daemon):
    install()
    record = json.loads(robot.files[RECORD])
    assert record["data_dir"] == DATA
    assert record["data_dir_parents"] == [f"{HOME}/.local", f"{HOME}/.local/share"]
    assert DATA in robot.dirs
    assert f"mkdir -p -m 700 {DATA}" in robot.commands


def test_reinstall_with_vision_over_an_install_without_it_adds_it_and_uninstall_restores(
    offline, robot, daemon
):
    before = snapshot(robot, daemon)
    install(vision=False)
    assert robot.fetches == ["models fetch --voice"]
    install(vision=True)
    assert robot.fetches[-1] == "models fetch --all"
    assert "opencv-python-headless" in json.loads(robot.files[RECORD])["added"]
    uninstall()
    assert snapshot(robot, daemon) == before


def test_reinstall_keeps_the_recorded_data_folder(offline, robot, daemon):
    install()
    first = json.loads(robot.files[RECORD])
    install()
    assert json.loads(robot.files[RECORD])["data_dir_parents"] == first["data_dir_parents"]


def test_uninstall_removes_the_data_folder_and_the_parents_it_created(offline, robot, daemon):
    before = snapshot(robot, daemon)
    install()
    robot.dirs.add(f"{DATA}/registry")  # the app wrote a registry
    uninstall()
    assert snapshot(robot, daemon) == before
    assert f"rm -rf {DATA}" in robot.commands


def robot_cache_path(monkeypatch) -> str:
    """Where the app code puts the trait cache for the robot user (default data folder)."""
    monkeypatch.delenv(paths.ENV_DIR, raising=False)
    monkeypatch.setattr(Path, "home", lambda: Path(HOME))
    return str(paths.traits_cache_path())


def test_uninstall_removes_the_persona_trait_cache_with_the_data_folder(
    offline, robot, daemon, monkeypatch
):
    before = snapshot(robot, daemon)
    install()
    cache = robot_cache_path(monkeypatch)
    assert cache.startswith(DATA + "/")  # the app writes inside the folder the installer records
    robot.dirs.add(cache)
    uninstall()
    assert cache not in robot.dirs
    assert snapshot(robot, daemon) == before


def test_a_trait_cache_in_a_preexisting_data_folder_is_left(offline, robot, daemon, monkeypatch):
    robot.dirs.add(DATA)
    install()
    cache = robot_cache_path(monkeypatch)
    robot.dirs.add(cache)
    uninstall()
    assert cache in robot.dirs


def test_uninstall_keeps_parents_that_are_not_empty(offline, robot, daemon):
    install()
    robot.dirs.add(f"{HOME}/.local/share/other-app")
    uninstall()
    assert DATA not in robot.dirs and f"{HOME}/.local/share" in robot.dirs


def test_a_preexisting_data_folder_is_not_ours_to_remove(offline, robot, daemon, capsys):
    robot.dirs.add(DATA)
    install()
    assert json.loads(robot.files[RECORD])["data_dir"] is None
    assert "not recorded" in capsys.readouterr().out
    uninstall()
    assert DATA in robot.dirs
    assert f"rm -rf {DATA}" not in robot.commands


def test_uninstall_without_a_record_leaves_the_data_folder_and_says_so(
    offline, robot, daemon, capsys
):
    robot.dirs.add(DATA)
    uninstall()
    assert DATA in robot.dirs
    assert "! Data folder not recorded" in capsys.readouterr().out


def test_uninstall_dry_run_shows_the_data_folder_and_keeps_it(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall(dry_run=True)
    assert f"○ Delete the data folder {DATA}" in flat(capsys.readouterr().out)
    assert DATA in robot.dirs
    assert not [c for c in robot.commands[-30:] if c.startswith(("rmdir", "rm -rf /home"))]


def test_uninstall_verification_fails_while_the_data_folder_remains(
    offline, robot, daemon, monkeypatch
):
    install()
    real = robot.run

    def run(command, check=True):
        if command == f"rm -rf {DATA}":
            return ""
        return real(command, check)

    monkeypatch.setattr(robot, "run", run)
    with pytest.raises(WizardError, match="data folder .* still present"):
        uninstall()
    assert RECORD in robot.files


def tamper(robot, **fields) -> None:
    record = json.loads(robot.files[RECORD])
    record.update(fields)
    robot.files[RECORD] = json.dumps(record)


def deleted(robot) -> list[str]:
    return [
        c
        for c in robot.commands
        if c.startswith("rm -rf") and "roboemot-install" not in c and "piper-venv" not in c
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "/",
        HOME,
        f"{DATA}/../x",
        "relative",
        "/home/other/.local/share/roboemot",  # right suffix, someone else's home
        "/tmp/.local/share/roboemot",
    ],
)
def test_uninstall_refuses_a_data_folder_that_is_not_the_one_it_would_choose(
    offline, robot, daemon, bad
):
    install()
    tamper(robot, data_dir=bad)
    robot.dirs.add(bad)
    with pytest.raises(WizardError, match="is not /home/pollen/.local/share/roboemot"):
        uninstall()
    assert deleted(robot) == [] and RECORD in robot.files and bad in robot.dirs


def test_uninstall_refuses_a_symlinked_data_folder(offline, robot, daemon):
    install()
    robot.symlinks[DATA] = "/etc"
    with pytest.raises(WizardError, match="resolves to /etc"):
        uninstall()
    assert deleted(robot) == [] and RECORD in robot.files


def test_tampered_parent_folders_are_never_removed(offline, robot, daemon):
    install()
    tamper(
        robot, data_dir_parents=["/etc", HOME, "/home", f"{HOME}/.local", f"{HOME}/.local/share"]
    )
    uninstall()
    rmdirs = [c for c in robot.commands if c.startswith("rmdir")]
    assert rmdirs == [f"rmdir {HOME}/.local/share 2>/dev/null || true",
                      f"rmdir {HOME}/.local 2>/dev/null || true"]  # fmt: skip


def test_interrupted_between_the_record_write_and_mkdir_can_be_undone(
    offline, robot, daemon, monkeypatch
):
    before = snapshot(robot, daemon)
    real = robot.run

    def run(command, check=True):
        if command.startswith("mkdir -p -m 700"):
            raise KeyboardInterrupt
        return real(command, check)

    monkeypatch.setattr(robot, "run", run)
    with pytest.raises(KeyboardInterrupt):
        install()
    monkeypatch.setattr(robot, "run", real)
    assert json.loads(robot.files[RECORD])["data_dir"] == DATA and DATA not in robot.dirs
    uninstall()
    assert snapshot(robot, daemon) == before


def test_reinstall_after_the_folder_vanished_keeps_the_record_and_uninstalls_clean(
    offline, robot, daemon
):
    before = snapshot(robot, daemon)
    install()
    robot.dirs = {d for d in robot.dirs if d != DATA and not d.startswith(DATA + "/")}
    install()
    assert json.loads(robot.files[RECORD])["data_dir"] == DATA
    uninstall()
    assert snapshot(robot, daemon) == before


def test_dry_run_with_a_preexisting_folder_says_it_would_not_be_recorded(
    offline, robot, daemon, capsys
):
    robot.dirs.add(DATA)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert "not recorded" in capsys.readouterr().out
    assert robot.files == {} and DATA in robot.dirs


def test_install_warns_when_the_data_folder_may_land_elsewhere(offline, robot, daemon, capsys):
    robot.env_data_dir = "/data/elsewhere"
    install()
    assert "ROBOEMOT_DATA_DIR=/data/elsewhere" in capsys.readouterr().out


def test_install_warns_about_a_data_dir_in_the_app_config(offline, robot, daemon, capsys):
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    robot.files[APP_YAML] = "schema_version: 1\ndata_dir: /mnt/big\n"
    install()
    assert "data_dir=/mnt/big" in capsys.readouterr().out


# ---------------------------------------------------------------- robot selection


def test_select_robot_takes_first_robot_found(monkeypatch):
    found = [("Neve  192.168.88.198", "192.168.88.198", "reachy-mini-wireless")]
    monkeypatch.setattr(inst, "find_robots", lambda root: found)
    host, platform = inst.select_robot(typed("1\n"), Options(), ROOT)
    assert host == "192.168.88.198" and platform.manifest_id == "reachy-mini-wireless"


def test_select_robot_other_asks_for_host(monkeypatch):
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve", "1.2.3.4", "reachy-mini-wireless")]
    )
    host, _ = inst.select_robot(typed("2\n", "\n"), Options(), ROOT)  # another host, Enter
    assert host == ReachyMiniWireless.default_host


@pytest.mark.parametrize("found", [[], RuntimeError("discovery needs zeroconf")])
def test_select_robot_falls_back_to_default(monkeypatch, found):
    def fake(root):
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(inst, "find_robots", fake)
    host, _ = inst.select_robot(typed("\n"), Options(), ROOT)  # Enter takes the shown default
    assert host == ReachyMiniWireless.default_host


def test_choose_reprompts_until_valid():
    assert typed("9\n", "x\n", "2\n").choose("Which robot?", ["a", "b"], flag="--host") == 1
    with pytest.raises(NoValidAnswer):
        typed("9\n", "x\n", "0\n").choose("Which robot?", ["a", "b"], flag="--host")


def test_select_robot_without_a_terminal_needs_host_before_mdns(monkeypatch):
    looked = []
    monkeypatch.setattr(inst, "find_robots", lambda root: looked.append(root) or [])
    with pytest.raises(MissingFlag, match="--host"):
        inst.select_robot(quiet_ui(), Options(), ROOT)
    assert looked == []
    host, _ = inst.select_robot(quiet_ui(), Options(host="10.0.0.9"), ROOT)
    assert host == "10.0.0.9" and looked == []  # a given host skips discovery


# ---------------------------------------------------------------- no terminal, no defaults (TP:253)


def answers(*lines: str):
    """A ``read`` for an interactive Ui: each call returns the next typed line."""
    it = iter(lines)
    return lambda: next(it)


def typed(*lines: str) -> Ui:
    return Ui(interactive=True, assume_yes=False, read=answers(*lines))


FULL = dict(
    host="127.0.0.1",
    user="pollen",
    robot="reachy-mini-wireless",
    dev=True,
    piper=False,
    models=False,  # the epic's models step: its own flag, like --piper
    cloud_voice=False,
)


@pytest.mark.parametrize(
    ("missing", "flag"),
    [
        ("host", "--host"),
        ("user", "--user"),
        ("dev", "--dev"),
        ("piper", "--piper"),
        ("models", "--models"),
        ("start", "--start"),
        ("startup", "--startup-app"),
    ],
)
def test_without_a_terminal_a_missing_flag_stops_before_the_robot_is_touched(
    offline, robot, daemon, monkeypatch, missing, flag
):
    looked = []
    monkeypatch.setattr(inst, "find_robots", lambda root: looked.append(root) or [])
    given = {**FULL, "start": False, "startup": False}
    given[missing] = None
    with pytest.raises(MissingFlag) as e:
        inst.run_install(Options(**given), quiet_ui(), ROOT)
    assert e.value.flag == flag
    assert robot.commands == [] and robot.copies == [] and daemon.mutations == []
    assert looked == []  # not even mDNS


def test_without_a_terminal_main_names_the_missing_flag_and_installs_nothing(
    offline, robot, daemon, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr(inst, "find_robots", lambda root: [("Neve", "1.2.3.4", RID)])
    assert inst.main([], root=ROOT) == 1
    err = capsys.readouterr().err
    assert "Missing --host" in err and "Try: python scripts/install_on_robot.py --host" in err
    assert robot.commands == [] and daemon.mutations == []


def test_yes_answers_the_confirmations_but_not_host_or_user(offline, robot, daemon):
    with pytest.raises(MissingFlag, match="--user"):
        inst.run_install(Options(host="127.0.0.1", robot=RID, yes=True), quiet_ui(yes=True), ROOT)
    install(dev=None, piper=None)  # yes=True answers dev, piper, start, startup
    assert json.loads(robot.files[RECORD])["pending_before"] is None


def test_with_several_platforms_a_missing_robot_flag_stops_first(offline, robot, monkeypatch):
    monkeypatch.setattr(inst, "installed_platforms", lambda: [RID, "desk-head-v1"])
    given = {**FULL, "robot": None, "start": False, "startup": False}
    with pytest.raises(MissingFlag, match="--robot"):
        inst.run_install(Options(**given), quiet_ui(), ROOT)
    assert robot.commands == []


def test_an_existing_pin_and_secret_are_kept_without_a_terminal(offline, robot, daemon, capsys):
    robot.dirs |= {DATA, PIN_PATH}
    robot.secrets = {"cloud_key": "theirs"}
    opts = Options(**FULL, start=False, startup=False, secrets=["cloud_key"])
    inst.run_install(opts, quiet_ui(), ROOT)  # no --yes: nothing to answer, nothing raised
    out = flat(capsys.readouterr().out)
    assert "✓ Kept the operator PIN" in out and "✓ Kept the secret cloud_key" in out
    assert robot.secrets == {"cloud_key": "theirs"} and robot.stdins == []


def test_start_over_a_running_app_without_a_terminal_needs_yes_before_installing(
    offline, robot, daemon
):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    with pytest.raises(MissingFlag, match="--yes"):
        inst.run_install(Options(**FULL, start=True, startup=False), quiet_ui(), ROOT)
    assert robot.commands == [] and daemon.mutations == []


@pytest.mark.parametrize(("missing", "flag"), [("dev", "--dev"), ("piper", "--piper")])
def test_a_dry_run_without_a_terminal_still_needs_dev_and_piper(
    offline, robot, daemon, missing, flag
):
    with pytest.raises(MissingFlag) as e:
        inst.run_install(Options(**{**FULL, missing: None}, dry_run=True), quiet_ui(), ROOT)
    assert e.value.flag == flag and robot.commands == []


class _TtyStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_prompt_with_no_answer_fails_with_a_try_line(offline, robot, daemon, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _TtyStdin(""))  # a terminal whose input ends at once
    monkeypatch.delenv("CI", raising=False)
    argv = ["--dry-run", "--host", "127.0.0.1", "--user", "pollen", "--robot", RID]
    assert inst.main(argv, root=ROOT) == 1
    err = capsys.readouterr().err
    assert "✗ No valid answer to: Install in dev mode?" in err
    assert f"Try: python scripts/install_on_robot.py {' '.join(argv)} --dev" in err


def test_the_dry_run_next_line_repeats_the_answered_dev_choice(offline, robot, daemon, capsys):
    opts = Options(host="127.0.0.1", user="pollen", robot=RID, piper=False, dry_run=True)
    inst.run_install(opts, typed("y\n", "\n", "\n", "\n"), ROOT)  # dev; cloud, models, key: no
    assert f"--robot {RID} --dev --no-piper --no-cloud-voice" in capsys.readouterr().out


def test_the_installer_modules_import_without_the_palette():
    code = (
        "import sys; sys.modules['sondalab_palette'] = None; "
        "import roboemot.install, roboemot_reachymini.settings, roboemot_reachymini.install"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_a_dry_run_without_a_terminal_needs_no_start_flags(offline, robot, daemon):
    inst.run_install(Options(**FULL, dry_run=True), quiet_ui(), ROOT)
    assert daemon.mutations == []


def test_uninstall_without_a_terminal_needs_only_host_and_user(offline, robot, daemon):
    with pytest.raises(MissingFlag, match="--user"):
        inst.run_uninstall(Options(host="127.0.0.1", robot=RID, uninstall=True), quiet_ui(), ROOT)
    assert robot.commands == []
    inst.run_uninstall(
        Options(host="127.0.0.1", user="pollen", robot=RID, uninstall=True), quiet_ui(), ROOT
    )


def test_a_shared_package_change_without_a_terminal_stops_before_any_write(offline, robot, daemon):
    robot.conflict = {"numpy": "2.1.0"}
    with pytest.raises(MissingFlag, match="--yes"):
        inst.run_install(Options(**FULL, start=False, startup=False), quiet_ui(), ROOT)
    assert robot.env["numpy"] == "2.0.0"
    assert robot.copies == [] and robot.files == {} and robot.tmp_dirs == set()
    writes = ("rm ", "rmdir", "mkdir", "mktemp", "mv ")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]


def test_a_shared_package_change_stops_before_the_auto_stop_and_the_cache_step(
    offline, robot, daemon
):
    """Merge of main into the epic: main's "shared packages checked before any write" also holds
    for the epic's writes before pip (stopping its own running app, clearing a cache)."""
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.dirs.add(UV_CACHE)
    robot.conflict = {"numpy": "2.1.0"}
    opts = Options(**FULL, start=False, startup=False, clear_caches=True)
    with pytest.raises(MissingFlag, match="--yes"):
        inst.run_install(opts, quiet_ui(), ROOT)
    assert ("POST", "/api/apps/stop-current-app") not in mutation_paths(daemon)
    assert daemon.current["state"] == "running" and UV_CACHE in robot.dirs
    assert not [c for c in robot.commands if c.startswith("rm ")]


@pytest.mark.parametrize("empty", ["host", "user"])
def test_an_empty_host_or_user_counts_as_missing(offline, robot, daemon, empty):
    given = {**FULL, "start": False, "startup": False, empty: ""}
    with pytest.raises(MissingFlag, match=f"--{empty}"):
        inst.run_install(Options(**given), quiet_ui(), ROOT)
    assert robot.commands == [] and daemon.requests == []


def test_non_interactive_ui_never_takes_a_default():
    ui = quiet_ui()
    with pytest.raises(MissingFlag, match="--host"):
        ui.ask("host?", "reachy-mini.local", flag="--host")
    with pytest.raises(MissingFlag, match="--yes"):
        ui.confirm("stop?", False)
    with pytest.raises(MissingFlag, match="--robot"):
        ui.choose("Which?", ["a", "b"], flag="--robot")
    assert quiet_ui(yes=True).confirm("stop?", False) is True


# ---------------------------------------------------------------- defaults and answers (TP:181)


def test_actions_that_change_the_robot_default_to_no(offline, robot, daemon, monkeypatch, capsys):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    robot.dev_cfg = False  # Enter at the dev question: production config
    ui = typed(*["\n"] * 7)  # Enter at every question
    inst.run_install(Options(host="127.0.0.1", user="pollen", robot=RID), ui, ROOT)
    err = capsys.readouterr().err
    for question in (
        "Install in dev mode? [y/N]",
        "Turn on cloud voice (audio leaves the robot)? [y/N]",
        "Install Piper for on-board speech (own virtualenv)? [y/N]",
        "Start roboemot now? [y/N]",
    ):
        assert question in err
    assert daemon.current is None and robot.piper_pips == []


def test_stopping_another_app_defaults_to_no(daemon, capsys):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    with pytest.raises(WizardError, match="conversation is still running"):
        platform_for(daemon).start(typed("\n"))
    assert "[y/N]" in capsys.readouterr().err and daemon.mutations == []


def test_confirm_asks_again_then_gives_up(capsys):
    with pytest.raises(NoValidAnswer):
        typed("maybe\n", "sure\n", "ok\n").confirm("Start?", False)
    assert capsys.readouterr().err.count("? Start? [y/N]") == 3


def test_end_of_input_is_not_an_answer():
    with pytest.raises(NoValidAnswer, match="input ended"):
        typed("").confirm("Start?", True)
    with pytest.raises(NoValidAnswer, match="input ended"):
        typed("").ask("SSH user", "pollen", flag="--user")


@pytest.mark.sim
def test_daemon_api_against_simulator():
    """The REST calls the installer relies on exist and answer as modelled in FakeDaemon."""
    real = DaemonClient(Target("localhost", 8000), timeout_s=INSTALL_TIMEOUT_S)
    status = real.status()
    assert status["state"] == "running" and status["simulation_enabled"]
    assert isinstance(real.installed_apps(), list)
    assert real.current_app() is None
    lock = real.app_lock_status()  # R4 T3: the shape start_blocker reads, not only the fake's
    assert lock is not None and lock["state"] in ("free", "local_app", "remote_session")
    real.wake_up()
    assert real.startup_app() is None
    with pytest.raises(DaemonError, match="HTTP 400"):
        real.set_startup_app("roboemot-not-installed")


# ---------------------------------------------------------------- operator PIN


def test_install_pipes_the_pin_over_stdin_never_in_a_command(offline, robot, monkeypatch):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: PIN)
    install()
    assert robot.stdins == [PIN + "\n"]
    assert PIN_PATH in robot.dirs
    assert not any(PIN in c for c in robot.commands)
    assert any(f"--data-dir {DATA}" in c for c in robot.commands)


def test_pin_is_set_after_the_data_folder_exists(offline, robot, monkeypatch):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: PIN)
    seen = []
    real = robot.run

    def spy(command, check=True, stdin=None):
        if "set-pin" in command:
            seen.append(DATA in robot.dirs)
        return real(command, check, stdin)

    monkeypatch.setattr(robot, "run", spy)
    install()
    assert seen == [True]


def test_install_without_a_terminal_skips_the_pin_and_says_how(offline, robot, capsys):
    install()
    assert robot.stdins == [] and PIN_PATH not in robot.dirs
    assert "roboemot.webui set-pin" in capsys.readouterr().out


def test_install_dry_run_only_says_it_would_set_the_pin(offline, robot, monkeypatch, capsys):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: PIN)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert robot.stdins == [] and PIN_PATH not in robot.dirs
    assert "○ Ask for the operator PIN" in capsys.readouterr().out


def test_install_rejects_a_weak_pin_then_accepts_a_good_one(offline, robot, monkeypatch):
    answers = iter(["12", PIN, PIN])
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: next(answers))
    install()
    assert robot.stdins == [PIN + "\n"]


def test_uninstall_leaves_no_pin_file_with_the_data_folder_gone(
    offline, robot, daemon, monkeypatch
):
    before = snapshot(robot, daemon)
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: PIN)
    install()
    assert PIN_PATH in robot.dirs
    uninstall()
    assert PIN_PATH not in robot.dirs and DATA not in robot.dirs
    assert snapshot(robot, daemon) == before


# ---------------------------------------------------------------- secrets (SPEC 8, 12 Data)

API_KEY = "sk-installer-5d2e"


def key_prompts(monkeypatch, value=API_KEY) -> None:
    def answer(self, question):
        return value if "secret" in question or question == "Repeat the value" else PIN

    monkeypatch.setattr(Ui, "ask_secret", answer)


def test_install_stores_a_secret_over_stdin_and_records_its_name(offline, robot, monkeypatch):
    key_prompts(monkeypatch)
    install(secrets=["cloud_key"])
    assert robot.secrets == {"cloud_key": API_KEY}
    assert API_KEY + "\n" in robot.stdins
    assert not any(API_KEY in c for c in robot.commands)
    record = json.loads(robot.files[RECORD])
    assert record["secrets"] == ["cloud_key"] and API_KEY not in json.dumps(record)


def test_the_secret_is_stored_after_the_data_folder_exists(offline, robot, monkeypatch):
    key_prompts(monkeypatch)
    seen = []
    real = robot.run

    def spy(command, check=True, stdin=None):
        if "secrets set" in command:
            seen.append(DATA in robot.dirs)
        return real(command, check, stdin)

    monkeypatch.setattr(robot, "run", spy)
    install(secrets=["cloud_key"])
    assert seen == [True]


def test_install_dry_run_never_touches_the_secret_store(offline, robot, monkeypatch, capsys):
    key_prompts(monkeypatch)
    inst.run_install(options(dry_run=True, secrets=["cloud_key"]), quiet_ui(yes=True), ROOT)
    assert not any("roboemot.cli secrets" in c for c in robot.commands)
    assert robot.secrets == {} and robot.stdins == []
    assert f"{DATA}/secrets.json" not in robot.dirs
    assert "○ Ask for a secret 'cloud_key'" in flat(capsys.readouterr().out)


def test_install_without_a_terminal_sets_no_secret_unless_asked(offline, robot):
    install()
    assert not any("roboemot.cli secrets" in c for c in robot.commands)


def test_a_secret_that_was_not_typed_is_not_set_and_says_how(offline, robot, capsys):
    install(secrets=["cloud_key"])  # not interactive: nothing can be typed
    out = capsys.readouterr().out
    assert robot.secrets == {} and json.loads(robot.files[RECORD])["secrets"] == []
    assert "roboemot.cli secrets set --data-dir" in out and "-- cloud_key" in out


@pytest.mark.parametrize("bad", ["a b", "x;rm -rf ~", "$(id)", ""])
def test_a_bad_secret_name_is_rejected_before_anything_runs(offline, robot, bad):
    with pytest.raises(WizardError, match="secret name"):
        install(secrets=[bad])
    assert robot.commands == []


def test_uninstall_removes_only_the_secrets_the_installer_added(
    offline, robot, daemon, monkeypatch
):
    robot.dirs.add(DATA)  # a data folder that was there before: the installer leaves it
    robot.secrets = {"mine": "keep-me", "cloud_key": "theirs"}
    key_prompts(monkeypatch)
    monkeypatch.setattr(Ui, "confirm", lambda self, question, default, **kw: False)
    install(secrets=["cloud_key", "extra"])
    assert robot.secrets == {"mine": "keep-me", "cloud_key": "theirs", "extra": API_KEY}
    assert json.loads(robot.files[RECORD])["secrets"] == ["extra"]  # cloud_key was not ours
    uninstall()
    assert robot.secrets == {"mine": "keep-me", "cloud_key": "theirs"}
    assert RECORD not in robot.files


def test_a_failing_secret_list_stops_before_anything_is_recorded(offline, robot, monkeypatch):
    key_prompts(monkeypatch)
    real = robot.run

    def run(command, check=True, stdin=None):
        if "roboemot.cli secrets list" in command:
            raise WizardError("on robot: store is readable by group or others")
        return real(command, check, stdin)

    monkeypatch.setattr(robot, "run", run)
    with pytest.raises(WizardError, match="group or others"):
        install(secrets=["cloud_key"])
    assert json.loads(robot.files[RECORD])["secrets"] == [] and robot.secrets == {}


def test_a_failing_secret_removal_keeps_the_record_so_a_rerun_finishes(offline, robot, monkeypatch):
    key_prompts(monkeypatch)
    install(secrets=["cloud_key"])
    real = robot.run
    failing = {"on": True}

    def run(command, check=True, stdin=None):
        if "roboemot.cli secrets remove" in command and failing["on"]:
            raise WizardError("on robot: boom")
        return real(command, check, stdin)

    monkeypatch.setattr(robot, "run", run)
    with pytest.raises(WizardError, match="boom"):
        uninstall()
    assert json.loads(robot.files[RECORD])["secrets"] == ["cloud_key"]
    failing["on"] = False
    uninstall()
    assert robot.secrets == {} and RECORD not in robot.files


def test_uninstall_dry_run_keeps_the_secrets_and_the_record(offline, robot, monkeypatch, capsys):
    key_prompts(monkeypatch)
    install(secrets=["cloud_key"])
    capsys.readouterr()
    uninstall(dry_run=True)
    assert robot.secrets == {"cloud_key": API_KEY}
    assert json.loads(robot.files[RECORD])["secrets"] == ["cloud_key"]
    assert "○ Remove the secrets cloud_key" in flat(capsys.readouterr().out)


def test_uninstall_ignores_a_tampered_secret_name_in_the_record(offline, robot, monkeypatch):
    key_prompts(monkeypatch)
    install(secrets=["cloud_key"])
    record = json.loads(robot.files[RECORD])
    record["secrets"] = ["cloud_key", "x; rm -rf ~"]
    robot.files[RECORD] = json.dumps(record)
    uninstall()
    assert not any("rm -rf ~" in c for c in robot.commands)


# ---------------------------------------------------------------- Piper virtualenv (SPEC 8)


def test_install_creates_a_recorded_piper_venv_with_a_pinned_no_cache_pip(offline, robot, daemon):
    install()
    assert PIPER_PYTHON in robot.dirs
    assert json.loads(robot.files[RECORD])["piper_venv"] == PIPER_VENV
    assert len(robot.piper_pips) == 1
    pip = robot.piper_pips[0]
    assert "--require-hashes" in pip and "--only-binary=:all:" in pip
    assert f"-r /tmp/roboemot-install.0/{lock_name('aarch64', '3.12')}" in pip
    assert [c[0].name for c in robot.copies[1:]] == [lock_name("aarch64", "3.12")]
    assert robot.tmp_dirs == set()
    assert any(
        c == f"{reachy.ReachyMiniWireless.python} -m venv {PIPER_VENV}" for c in robot.commands
    )


def test_install_without_a_lock_for_the_robot_python_fails_before_touching_it(
    offline, robot, daemon
):
    robot.machine = "aarch64\n3.9"
    with pytest.raises(WizardError, match="No hash-pinned Piper lock"):
        inst.install_piper(
            Ui(False, True),
            ReachyMiniWireless(),
            robot,
            inst.InstallRecord("roboemot", "reachy", []),
            dry_run=False,
        )
    assert robot.piper_pips == [] and not any(" -m venv " in c for c in robot.commands)


def test_piper_venv_default_matches_the_core_config_default():
    from roboemot import paths

    assert paths.piper_python(f"{DATA}").as_posix() == PIPER_PYTHON


def test_install_dry_run_only_says_it_would_create_the_piper_venv(offline, robot, daemon, capsys):
    before = dict(robot.files), set(robot.dirs)
    inst.run_install(options(dry_run=True, start=False, startup=False), quiet_ui(yes=True), ROOT)
    assert (dict(robot.files), set(robot.dirs)) == before
    assert robot.piper_pips == [] and not any(" -m venv " in c for c in robot.commands)
    assert f"○ Create the Piper venv {PIPER_VENV}" in flat(capsys.readouterr().out)


def piper_args(robot, dry_run):
    return Ui(False, True), ReachyMiniWireless(), robot, inst.InstallRecord("r", "p", []), dry_run


def test_piper_dry_run_runs_no_mktemp_copy_or_venv_and_still_rejects_an_unlocked_target(
    offline, robot, daemon
):
    inst.install_piper(*piper_args(robot, True))
    assert not any(c.startswith("mktemp") or " -m venv " in c for c in robot.commands)
    assert robot.copies == [] and robot.piper_pips == []
    robot.machine = "aarch64\n3.9"
    with pytest.raises(WizardError, match="No hash-pinned Piper lock"):
        inst.install_piper(*piper_args(robot, True))


def test_failed_piper_pip_removes_the_remote_tmp_dir(offline, robot, daemon):
    robot.piper_pip_fails = True
    with pytest.raises(WizardError, match="piper-tts failed"):
        inst.install_piper(*piper_args(robot, False))
    assert robot.tmp_dirs == set() and len(robot.copies) == 1


def test_failed_lock_copy_removes_the_tmp_dir_and_creates_no_venv(offline, robot, daemon):
    def boom(files, dest):
        raise WizardError("scp failed")

    robot.copy = boom
    with pytest.raises(WizardError, match="scp failed"):
        inst.install_piper(*piper_args(robot, False))
    assert robot.tmp_dirs == set() and not any(" -m venv " in c for c in robot.commands)


@pytest.mark.parametrize(
    "answer", ["", "aarch64", "aarch64\n3", "aarch64 extra\n3.12", "x\n3.12\ny"]
)
def test_malformed_platform_probe_is_a_wizard_error(robot, answer):
    robot.machine = answer
    with pytest.raises(WizardError, match="unexpected answer"):
        inst.probe_platform(robot, "/venvs/apps_venv/bin/python")


def test_platform_probe_returns_arch_and_python_minor(robot):
    assert inst.probe_platform(robot, "/venvs/apps_venv/bin/python") == ("aarch64", "3.12")


def test_install_without_piper_installs_no_venv(offline, robot, daemon):
    install(piper=False)
    assert PIPER_VENV not in robot.dirs and robot.piper_pips == []
    assert json.loads(robot.files[RECORD])["piper_venv"] is None


def test_uninstall_removes_the_piper_venv_even_when_the_data_folder_predates_the_install(
    offline, robot, daemon
):
    robot.dirs.add(DATA)
    install()
    assert PIPER_PYTHON in robot.dirs
    uninstall()
    assert PIPER_VENV not in robot.dirs and DATA in robot.dirs


def test_uninstall_dry_run_keeps_the_piper_venv(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall(dry_run=True)
    assert PIPER_VENV in robot.dirs
    assert "○ Delete the Piper venv" in capsys.readouterr().out


def test_uninstall_refuses_a_piper_venv_outside_the_data_folder(offline, robot, daemon):
    install()
    tamper(robot, piper_venv="/etc")
    robot.dirs.add("/etc")
    uninstall()
    assert "rm -rf /etc" not in robot.commands and "/etc" in robot.dirs


def test_a_foreign_piper_folder_is_left_alone(offline, robot, daemon):
    robot.dirs |= {DATA, PIPER_VENV}
    install()
    assert robot.piper_pips == [] and json.loads(robot.files[RECORD])["piper_venv"] is None


def test_rerun_after_a_failed_piper_pip_installs_it_again_instead_of_trusting_the_folder(
    offline, robot, daemon
):
    robot.piper_pip_fails = True
    with pytest.raises(WizardError, match="piper-tts failed"):
        install()
    assert PIPER_PYTHON in robot.dirs and len(robot.piper_pips) == 1  # folder exists, unusable
    robot.piper_pip_fails = False
    install()
    assert len(robot.piper_pips) == 2 and robot.piper_importable
    venvs = [c for c in robot.commands if " -m venv " in c]
    assert len(venvs) == 1  # the venv itself was not recreated


def test_rerun_with_a_working_piper_venv_does_not_touch_pip_again(offline, robot, daemon):
    install()
    install()
    assert len(robot.piper_pips) == 1


def test_piper_flags_with_yes_prompt_nothing_and_decide(offline, robot, daemon):
    ns = inst.parse_args(["--yes", "--no-piper"])
    assert ns.piper is False and inst.parse_args(["--yes", "--piper"]).piper is True
    assert inst.parse_args(["--yes"]).piper is None  # --yes answers the question: yes
    install()  # piper=None with yes=True installs it
    assert robot.piper_pips


# ---------------------------------------------------------------- missing apps venv (fresh robot)


def fresh_robot(robot) -> None:
    robot.apps_venv, robot.env = False, {}
    robot.dirs.discard(reachy.APPS_VENV)


def test_install_creates_the_missing_apps_venv_like_the_daemon_and_records_it_first(
    offline, robot, daemon
):
    fresh_robot(robot)
    install()
    assert robot.apps_venv and robot.env["reachy-mini"] == "1.11.0"
    assert json.loads(robot.record_at_venv)["env_created"] == reachy.APPS_VENV
    assert json.loads(robot.files[RECORD])["env_created"] == reachy.APPS_VENV
    venv = f"{DAEMON_PY} -m venv {reachy.APPS_VENV}"
    pin = f"{APPS_PY} -m pip install --progress-bar off"
    cmds = robot.commands
    assert venv in cmds
    created = cmds.index(venv)
    pip = next(i for i, c in enumerate(cmds) if c.startswith(pin) and "reachy-mini==1.11.0" in c)
    assert pip > created
    assert not any(c.startswith(APPS_PY) for c in cmds[:created])
    assert "df -Pk /venvs | tail -1" in cmds
    assert f"df -Pk {reachy.APPS_VENV} | tail -1" not in cmds
    probe = next(c for c in cmds if "platform.machine" in c)
    assert probe.startswith(DAEMON_PY)
    assert "roboemot" in daemon.installed


def test_install_into_an_existing_apps_venv_creates_nothing(offline, robot, daemon):
    install()
    assert not any(" -m venv /venvs" in c for c in robot.commands)
    assert json.loads(robot.files[RECORD])["env_created"] is None


def test_install_dry_run_with_a_missing_apps_venv_reports_and_writes_nothing(
    offline, robot, daemon, capsys
):
    fresh_robot(robot)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = flat(capsys.readouterr().out)
    assert f"○ Create the environment {reachy.APPS_VENV} (reachy-mini==1.11.0)" in out
    assert "Dependency check skipped" in out
    assert "Dry run passed" in out
    assert not robot.apps_venv and robot.files == {} and robot.tmp_dirs == set()
    writes = ("rm ", "rmdir", "mkdir", "mktemp", "mv ")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]
    assert not any(" -m venv " in c or " -m pip " in c for c in robot.commands)


def test_install_without_a_way_to_create_the_venv_is_a_clear_error(offline, robot, monkeypatch):
    monkeypatch.setattr(ReachyMiniWireless, "base_python", None)
    fresh_robot(robot)
    with pytest.raises(WizardError, match="cannot create it") as exc:
        install()
    assert "is this a" not in str(exc.value)
    assert robot.files == {} and not robot.apps_venv


def test_install_stops_when_the_daemon_interpreter_is_missing_too(offline, robot, monkeypatch):
    fresh_robot(robot)
    run = robot.run
    monkeypatch.setattr(
        robot,
        "run",
        lambda c, **kw: "no" if c.startswith(f"test -x {DAEMON_PY}") else run(c, **kw),
    )
    with pytest.raises(WizardError, match=f"neither does {DAEMON_PY}"):
        install()
    assert robot.files == {}


def test_failed_pip_in_the_new_venv_keeps_the_record_so_uninstall_can_clean_up(
    offline, robot, daemon, monkeypatch
):
    fresh_robot(robot)
    run, offline_pip = robot.run, [False]
    monkeypatch.setattr(
        robot,
        "run",
        lambda c, **kw: (
            "ERROR: no network\n__rc=1"
            if offline_pip[0] and "reachy-mini==" in c and " -m pip " in c
            else run(c, **kw)
        ),
    )
    offline_pip[0] = True
    with pytest.raises(WizardError, match="pip install failed"):
        install()
    assert json.loads(robot.files[RECORD])["env_created"] == reachy.APPS_VENV
    offline_pip[0] = False
    uninstall()
    assert not robot.apps_venv and RECORD not in robot.files


def test_a_rerun_after_a_failed_pip_in_the_new_venv_finishes_it(offline, robot, daemon, capsys):
    """R4 B1/T1: the venv was created but `pip install reachy-mini==` failed (network); the
    rerun with the network back finishes the venv instead of failing on the reachy_mini probe,
    and uninstall then removes the venv (its baseline covers what the finish installed)."""
    fresh_robot(robot)
    before = snapshot(robot, daemon)
    run, offline_pip = robot.run, [True]
    robot.run = lambda c, **kw: (
        "ERROR: no network\n__rc=1"
        if offline_pip[0] and "reachy-mini==" in c and " -m pip " in c
        else run(c, **kw)
    )
    with pytest.raises(WizardError, match="pip install failed"):
        install()
    assert robot.apps_venv and "reachy-mini" not in robot.env
    offline_pip[0] = False
    install()
    assert robot.env["reachy-mini"] == "1.11.0" and "roboemot-core" in robot.env
    assert "Finishing the environment" in capsys.readouterr().out
    assert "reachy-mini" in json.loads(robot.files[RECORD])["env_baseline"]
    uninstall()
    assert snapshot(robot, daemon) == before and not robot.apps_venv


def test_a_venv_without_its_base_package_that_the_installer_did_not_create_is_refused(
    offline, robot, daemon
):
    """Review S-6: refused before the auto-stop and the cache step."""
    del robot.env["reachy-mini"]
    robot.dirs.add(UV_CACHE)
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    with pytest.raises(WizardError, match="not created by this installer"):
        install(clear_caches=True)
    assert robot.files == {} and UV_CACHE in robot.dirs
    assert daemon.current["state"] == "running"


def half_created(robot) -> None:
    """A venv the installer created whose `pip install reachy-mini` failed."""
    fresh_robot(robot)
    run, failing = robot.run, [True]
    robot.run = lambda c, **kw: (
        "ERROR: no network\n__rc=1"
        if failing[0] and "reachy-mini==" in c and " -m pip " in c
        else run(c, **kw)
    )
    with pytest.raises(WizardError, match="pip install failed"):
        install()
    failing[0] = False


def test_an_incomplete_created_venv_another_package_moved_into_is_not_finished(
    offline, robot, daemon
):
    """Review S-1: the finish step installs unpinned, so it refuses a venv with new packages."""
    half_created(robot)
    robot.env["someone-else"] = "1.0"
    with pytest.raises(WizardError, match="holds packages added since it was created"):
        install()
    assert "reachy-mini" not in robot.env


def test_the_dry_run_says_it_would_finish_the_venv_and_writes_nothing(
    offline, robot, daemon, capsys
):
    """Review MT6."""
    half_created(robot)
    files, env = dict(robot.files), dict(robot.env)
    capsys.readouterr()
    install(dry_run=True)
    out = capsys.readouterr().out
    assert f"○ Finish the environment {reachy.APPS_VENV}" in flat(out)
    assert robot.files == files and robot.env == env


def test_uninstall_removes_the_venv_the_installer_created(offline, robot, daemon):
    fresh_robot(robot)
    before = snapshot(robot, daemon)
    install()
    uninstall()
    assert snapshot(robot, daemon) == before
    assert f"rm -rf {reachy.APPS_VENV}" in robot.commands


def test_uninstall_keeps_a_created_venv_another_app_lives_in(offline, robot, daemon, capsys):
    fresh_robot(robot)
    install()
    robot.apps_in_env = ["conversation"]
    uninstall()
    out = capsys.readouterr().out
    assert robot.apps_venv and "roboemot-core" not in robot.env
    assert f"keeping {reachy.APPS_VENV}" in out and "conversation" in out
    assert RECORD not in robot.files


def test_uninstall_dry_run_says_it_would_remove_the_created_venv(offline, robot, daemon, capsys):
    fresh_robot(robot)
    install()
    state = copy.deepcopy(snapshot(robot, daemon))
    uninstall(dry_run=True)
    assert snapshot(robot, daemon) == state
    assert f"○ Delete the environment {reachy.APPS_VENV}" in flat(capsys.readouterr().out)


def test_uninstall_leaves_a_venv_it_did_not_create(offline, robot, daemon):
    install()
    uninstall()
    assert robot.apps_venv and "pip" in robot.env


def test_uninstall_with_the_venv_already_gone_still_cleans_up(offline, robot, daemon, capsys):
    install()
    assert DATA in robot.dirs
    fresh_robot(robot)
    seen = len(robot.commands)
    uninstall()
    assert RECORD not in robot.files and DATA not in robot.dirs
    assert not any(c.startswith(APPS_PY) for c in robot.commands[seen:])
    assert "✓ Uninstalled" in capsys.readouterr().out


def test_uninstall_twice_after_removing_a_created_venv_is_harmless(offline, robot, daemon):
    fresh_robot(robot)
    install()
    uninstall()
    after = snapshot(robot, daemon)
    uninstall()
    assert snapshot(robot, daemon) == after


def test_uninstall_ignores_a_recorded_venv_path_that_is_not_the_apps_venv(offline, robot, daemon):
    fresh_robot(robot)
    install()
    record = json.loads(robot.files[RECORD])
    record["env_created"] = "/venvs/mini_daemon"
    robot.files[RECORD] = json.dumps(record)
    uninstall()
    assert "rm -rf /venvs/mini_daemon" not in robot.commands
    assert robot.apps_venv


def test_create_env_pins_the_daemons_own_reachy_mini_version(robot):
    robot.daemon_version = "1.12.0"
    robot.apps_venv = False
    wireless = ReachyMiniWireless()
    wireless.create_env(quiet_ui(), robot, wireless.plan_env(robot))
    pip = [c for c in robot.commands if " -m pip install" in c]
    assert len(pip) == 1 and "reachy-mini==1.12.0" in pip[0] and "--no-cache-dir" in pip[0]


def test_env_users_lists_other_apps_only(robot):
    robot.apps_in_env = ["conversation", "marionette"]
    py = inst.RemotePython(robot, APPS_PY)
    assert ReachyMiniWireless().env_users(py) == ["conversation", "marionette"]
    robot.apps_in_env = []
    assert ReachyMiniWireless().env_users(py) == []


def test_uninstall_with_the_venv_gone_needs_no_creating_interpreter(
    offline, robot, daemon, monkeypatch
):
    install()
    fresh_robot(robot)
    monkeypatch.setattr(ReachyMiniWireless, "base_python", None)
    uninstall()
    assert RECORD not in robot.files and DATA not in robot.dirs


def test_uninstall_with_the_venv_and_the_daemon_interpreter_both_gone(
    offline, robot, daemon, monkeypatch
):
    install()
    fresh_robot(robot)
    run = robot.run
    monkeypatch.setattr(
        robot,
        "run",
        lambda c, **kw: "no" if c.startswith(f"test -x {DAEMON_PY}") else run(c, **kw),
    )
    uninstall()
    assert RECORD not in robot.files and DATA not in robot.dirs


def test_uninstall_of_a_created_venv_with_replaced_packages_needs_no_internet(
    offline, robot, daemon, capsys
):
    """R4 S7/T5: the venv the installer created goes whole, so the replaced versions inside it
    are not restored first (that needs the package index): offline, uninstall still ends clean."""
    fresh_robot(robot)
    before = snapshot(robot, daemon)
    robot.conflict = {"pip": "26.0"}
    install()
    assert json.loads(robot.files[RECORD])["replaced"] == {
        "pip": {"before": "25.0.1", "after": "26.0"}
    }
    run = robot.run
    robot.run = lambda c, **kw: (
        "ERROR: no network\n__rc=1" if " -m pip install" in c else run(c, **kw)
    )
    start = len(robot.commands)
    uninstall()
    assert snapshot(robot, daemon) == before
    assert not any(" -m pip uninstall" in c for c in robot.commands[start:])
    out = flat(capsys.readouterr().out)
    assert f"✓ Removed the app's packages with {reachy.APPS_VENV}" in out


def test_uninstall_with_the_venv_gone_reports_no_drift(offline, robot, daemon, capsys):
    """R4 N10: with the environment already gone, the replaced packages are not "changed"."""
    fresh_robot(robot)
    robot.conflict = {"pip": "26.0"}
    install()
    fresh_robot(robot)
    capsys.readouterr()
    uninstall()
    assert "changed since the install" not in capsys.readouterr().out


def half_made(robot) -> None:
    fresh_robot(robot)
    robot.dirs.add(reachy.APPS_VENV)


def test_install_refuses_a_venv_folder_without_an_interpreter(offline, robot):
    half_made(robot)
    with pytest.raises(WizardError, match="Environment without an interpreter"):
        install()
    assert robot.files == {} and not any(" -m venv " in c for c in robot.commands)


def test_uninstall_keeps_a_venv_folder_without_an_interpreter(offline, robot, daemon, capsys):
    fresh_robot(robot)
    install()
    half_made(robot)
    uninstall()
    assert f"rm -rf {reachy.APPS_VENV}" not in robot.commands
    assert reachy.APPS_VENV in robot.dirs and RECORD not in robot.files
    assert "no interpreter" in capsys.readouterr().out


def test_a_daemon_without_reachy_mini_leaves_no_record_and_no_venv(offline, robot):
    fresh_robot(robot)
    robot.daemon_version = ""
    with pytest.raises(WizardError, match="reachy_mini"):
        install()
    assert robot.files == {} and not robot.apps_venv


def test_uninstall_keeps_a_created_venv_holding_packages_it_does_not_account_for(
    offline, robot, daemon, capsys
):
    fresh_robot(robot)
    install()
    robot.env["foreign-lib"] = "1.0"
    uninstall()
    out = capsys.readouterr().out
    assert robot.apps_venv and f"keeping {reachy.APPS_VENV}" in out and "foreign-lib" in out
    assert RECORD not in robot.files


# ---------------------------------------------------------------- progress (SPEC 4A, Installer)


def test_long_steps_are_announced_and_pip_is_streamed(offline, robot, daemon, capsys):
    robot.apps_venv = False
    install()
    out = capsys.readouterr().out
    for step in (
        "Creating the apps venv",
        "Creating the Piper venv",
        "Installing Python packages",
        "Reinstalling the app wheels",
    ):
        assert f"{step}" in out
    assert out.count("can take minutes on the robot") >= 5
    streamed = [c for c in robot.streamed if " -m pip " in c]
    assert streamed and all("--no-cache-dir" in c and "--progress-bar off" in c for c in streamed)
    assert not [c for c in streamed if "--quiet" in c]


def test_uninstall_announces_the_pip_steps_it_streams(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall()
    out = capsys.readouterr().out
    assert "Uninstalling packages\n" in out
    assert re.search(r"✓ Uninstalling packages +\d+s\n", out)


def test_the_dry_run_streams_nothing(offline, robot, daemon):
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert robot.streamed == []


# ---------------------------------------------------------------- run script (SPEC 4A, Installer)


def test_app_lock_status_reads_the_endpoint_and_tolerates_its_absence(api, daemon):
    daemon.lock = {"state": "remote_session", "holder_name": "Mobile"}
    assert api.app_lock_status() == {"state": "remote_session", "holder_name": "Mobile"}
    daemon.lock = None
    assert api.app_lock_status() is None


def test_start_refuses_while_the_mobile_app_holds_the_robot(daemon):
    daemon.lock = {"state": "remote_session", "holder_name": "Reachy phone"}
    with pytest.raises(WizardError, match="Reachy mobile app holds the robot.*close it first"):
        platform_for(daemon).start(quiet_ui(yes=True))
    assert daemon.mutations == []


def test_start_goes_on_when_the_daemon_has_no_lock_endpoint(daemon):
    daemon.lock = None
    platform_for(daemon).start(quiet_ui(yes=True))
    assert daemon.current["state"] == "running"


def test_app_url_uses_the_configured_web_ui_port():
    port = RoboEmotConfig().webui.port
    assert ReachyMiniWireless().app_url("reachy-mini.local") == f"http://reachy-mini.local:{port}"


def test_install_start_prints_the_web_ui_address(offline, robot, daemon, capsys):
    inst.run_install(options(start=True, startup=False), quiet_ui(yes=True), ROOT)
    out = flat(capsys.readouterr().out)
    assert f"✓ Web UI http://127.0.0.1:{RoboEmotConfig().webui.port}" in out
    assert "python scripts/run_on_robot.py" in out


def run_args(*extra: str) -> list[str]:
    return ["--host", "127.0.0.1", "--robot", "reachy-mini-wireless", "-y", *extra]


def test_run_starts_the_installed_app_without_ssh(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    robot.commands.clear()
    robot.streamed.clear()
    assert run_mod.main(run_args()) == 0
    out = flat(capsys.readouterr().out)
    assert "RoboEmot run" in out and "RoboEmot installer" not in out
    assert daemon.current["state"] == "running"
    assert f"✓ Web UI http://127.0.0.1:{RoboEmotConfig().webui.port}" in out
    assert robot.commands == [] and robot.streamed == []


def test_run_says_to_install_first_when_the_app_is_missing(offline, robot, daemon, capsys):
    assert run_mod.main(run_args()) == 1
    err = flat(capsys.readouterr().err)  # the Term error block
    assert "roboemot is not installed run the installer first" in err
    assert daemon.mutations == []


def test_run_dry_run_changes_nothing(offline, robot, daemon, capsys):
    install()
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    before = list(daemon.mutations)
    assert run_mod.main(run_args("--dry-run")) == 0
    out = flat(capsys.readouterr().out)
    assert "○ Stop conversation" in out and "✓ Dry run passed" in out
    assert daemon.mutations == before


def test_run_reports_the_mobile_app_holding_the_robot(offline, robot, daemon, capsys):
    install()
    daemon.lock = {"state": "remote_session", "holder_name": "phone"}
    assert run_mod.main(run_args()) == 1
    assert "mobile app holds the robot (phone); close it first" in flat(capsys.readouterr().err)


# ---------------------------------------------------------------- fix round 1


def hold_slot(daemon) -> None:
    daemon.lock = {"state": "remote_session", "holder_name": "phone"}


def test_start_blocker_is_read_only_and_names_the_mobile_app(daemon):
    platform = platform_for(daemon)
    assert platform.start_blocker() is None
    hold_slot(daemon)
    assert "Reachy mobile app holds the robot" in platform.start_blocker()
    assert daemon.mutations == []


def test_a_local_app_holding_the_slot_is_not_a_blocker_and_asks_to_stop_it(daemon):
    daemon.lock = {"state": "local_app", "holder_name": "conversation"}
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    ui = quiet_ui()
    asked: list[str] = []
    ui.confirm = lambda question, default: asked.append(question) or True
    platform_for(daemon).start(ui)
    assert asked and "conversation is running" in asked[0]
    assert daemon.current["info"]["name"] == "roboemot"


def test_run_dry_run_reports_that_it_would_refuse(offline, robot, daemon, capsys):
    install()
    hold_slot(daemon)
    capsys.readouterr()
    assert run_mod.main(run_args("--dry-run")) == 0
    out = flat(capsys.readouterr().out)
    assert "Would refuse the Reachy mobile app holds the robot" in out
    assert daemon.current is None


def test_installer_dry_run_reports_that_it_would_refuse(offline, robot, daemon, capsys):
    hold_slot(daemon)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = flat(capsys.readouterr().out)
    assert "Start would be refused the Reachy mobile app holds the robot" in out


def test_install_with_a_held_slot_finishes_warns_and_points_to_the_run_script(
    offline, robot, daemon, capsys
):
    hold_slot(daemon)
    inst.run_install(options(start=True, startup=False), quiet_ui(yes=True), ROOT)
    out = flat(capsys.readouterr().out)
    assert "Did not start roboemot the Reachy mobile app holds the robot" in out
    assert "python scripts/run_on_robot.py" in out
    assert "✓ Installed roboemot" in out and "RoboEmot" in out
    assert daemon.current is None and RECORD in robot.files


def test_a_new_apps_venv_prints_one_check_line(offline, robot, daemon, capsys):
    fresh_robot(robot)
    install()
    out = capsys.readouterr().out
    assert f"✓ Created the apps venv {reachy.APPS_VENV} with reachy-mini==1.11.0" in flat(out)
    assert "✓ Creating the apps venv" not in out
    first = out.index("Installing Python packages")
    assert "✓ Installing Python packages" not in out[first : out.index("Created the apps venv")]


def test_uninstall_pip_is_announced_without_the_minutes_note(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall()
    out = capsys.readouterr().out
    assert "Uninstalling packages\n" in out
    assert "Uninstalling packages (can take" not in out


# ------------------------------------------------------- auto-stop of our own running app


STOP = ("POST", "/api/apps/stop-current-app")


def mutation_paths(daemon) -> list[tuple[str, str]]:
    return [(m, p) for m, p, _ in daemon.mutations]


def test_install_stops_its_own_running_app_before_changing_packages(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    install()
    assert STOP in mutation_paths(daemon)
    assert "✓ Stopped roboemot before installing: a running process keeps the old code" in (
        flat(capsys.readouterr().out)
    )
    pip_at = next(i for i, c in enumerate(robot.commands) if " -m pip install" in c)
    assert pip_at > 0 and daemon.current is None


def test_install_leaves_another_running_app_alone(offline, robot, daemon):
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    install()
    assert STOP not in mutation_paths(daemon)
    assert daemon.current["info"]["name"] == "conversation"


def test_install_with_no_app_running_stops_nothing(offline, robot, daemon):
    install()
    assert STOP not in mutation_paths(daemon)


def test_dry_run_reports_that_it_would_stop_its_own_app(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert "○ Stop roboemot before installing" in flat(capsys.readouterr().out)
    assert daemon.mutations == [] and daemon.current["info"]["name"] == "roboemot"


def test_start_prompt_keeps_its_default_after_the_auto_stop(offline, robot, daemon):
    # the auto-stop does not answer the start question; main's Term rule makes its default no
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    asked: list[tuple[str, bool]] = []
    ui = quiet_ui(yes=True)
    ui.confirm = lambda question, default, **kw: asked.append((question, default)) or True
    inst.run_install(options(startup=False, yes=True), ui, ROOT)
    assert ("Start roboemot now?", False) in asked
    assert daemon.current["info"]["name"] == "roboemot"


def test_a_failure_after_the_auto_stop_says_how_to_start_it_again(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.conflict = {"numpy": "2.1.0"}
    # interactive: off a terminal a shared-package change stops before the auto-stop (main)
    ui = Ui(interactive=True, assume_yes=False)
    ui.confirm = lambda question, default, **kw: "shared packages" not in question
    with pytest.raises(WizardError, match="shared packages left as they were"):
        inst.run_install(options(start=False, startup=False, yes=False), ui, ROOT)
    out = flat(capsys.readouterr().out)
    assert "roboemot is stopped it was stopped before installing" in out
    assert "python scripts/run_on_robot.py" in out


def test_a_failure_without_an_auto_stop_prints_no_start_hint(offline, robot, daemon, capsys):
    robot.conflict = {"numpy": "2.1.0"}
    ui = Ui(interactive=True, assume_yes=False)
    ui.confirm = lambda question, default, **kw: "shared packages" not in question
    with pytest.raises(WizardError):
        inst.run_install(options(start=False, startup=False, yes=False), ui, ROOT)
    assert "was stopped before installing" not in capsys.readouterr().out


def test_a_failed_stop_aborts_with_the_daemon_message_and_installs_nothing(
    offline, robot, daemon, monkeypatch
):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    real = daemon.handle

    def handle(method, path, body):
        if (method, path) == STOP:
            return 500, {"detail": "motors busy"}
        return real(method, path, body)

    monkeypatch.setattr(daemon, "handle", handle)
    with pytest.raises(WizardError, match="motors busy"):
        install()
    assert "roboemot-core" not in robot.env and robot.copies == []
    assert not any(" -m pip install" in c for c in robot.commands)


def test_dry_run_counts_the_stop_once(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert out.count("○ Stop roboemot") == 1


# ---------------------------------------------------------------- run --restart

RESTART = ("POST", "/api/apps/restart-current-app", None)


def own_running(daemon) -> None:
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}


def test_run_restart_restarts_a_running_roboemot_and_prints_the_web_ui(
    offline, robot, daemon, capsys
):
    install()
    own_running(daemon)
    capsys.readouterr()
    assert run_mod.main(run_args("--restart")) == 0
    out = capsys.readouterr().out
    assert RESTART in daemon.mutations
    assert daemon.current == {"info": {"name": "roboemot"}, "state": "running"}
    assert f"✓ Web UI http://127.0.0.1:{RoboEmotConfig().webui.port}" in flat(out)
    assert not any(m[1].startswith("/api/apps/start-app/") for m in daemon.mutations)


def test_restart_asks_first_says_the_head_moves_and_a_no_changes_nothing(daemon):
    own_running(daemon)
    ui = quiet_ui()
    asked: list[str] = []
    ui.confirm = lambda question, default: asked.append(question) or False
    with pytest.raises(WizardError, match="not restarted"):
        platform_for(daemon).restart(ui)
    assert asked and "head" in asked[0]
    assert daemon.mutations == []


def test_run_restart_without_yes_does_not_restart_unattended(offline, robot, daemon, capsys):
    install()
    own_running(daemon)
    before = list(daemon.mutations)
    args = ["--host", "127.0.0.1", "--robot", "reachy-mini-wireless", "--restart"]
    assert run_mod.main(args) == 1
    assert daemon.mutations == before


def test_run_restart_starts_roboemot_when_it_is_not_running(offline, robot, daemon, capsys):
    install()
    assert run_mod.main(run_args("--restart")) == 0
    assert RESTART not in daemon.mutations
    assert ("POST", "/api/apps/start-app/roboemot", None) in daemon.mutations
    assert daemon.current["state"] == "running"


def test_run_restart_with_another_app_running_takes_the_start_flow(offline, robot, daemon):
    install()
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    assert run_mod.main(run_args("--restart")) == 0
    assert RESTART not in daemon.mutations
    assert ("POST", "/api/apps/stop-current-app", None) in daemon.mutations
    assert daemon.current["info"]["name"] == "roboemot"


def test_run_dry_run_restart_reports_and_changes_nothing(offline, robot, daemon, capsys):
    install()
    own_running(daemon)
    before = list(daemon.mutations)
    capsys.readouterr()
    assert run_mod.main(run_args("--dry-run", "--restart")) == 0
    out = flat(capsys.readouterr().out)
    assert "○ Restart roboemot" in out and "✓ Dry run passed" in out
    assert daemon.mutations == before


def test_run_restart_whose_start_fails_after_the_stop_says_roboemot_is_stopped(
    offline, robot, daemon, capsys
):
    install()
    own_running(daemon)
    daemon.restart_start_fails = True
    capsys.readouterr()
    assert run_mod.main(run_args("--restart")) == 1
    err = flat(capsys.readouterr().err)  # the Term error block
    assert "roboemot is stopped" in err and "without --restart" in err
    assert "Another app holds the robot" in err


def test_run_restart_that_times_out_waits_for_the_new_start(
    offline, robot, daemon, capsys, monkeypatch
):
    install()
    own_running(daemon)
    monkeypatch.setattr(daemon_mod, "RESTART_TIMEOUT_S", 0.2)
    daemon.restart_answer_s = 1.0
    capsys.readouterr()
    assert run_mod.main(run_args("--restart")) == 0
    out = flat(capsys.readouterr().out)
    assert "still restarting" in out and "✓ Started roboemot running" in out
    assert daemon.current == {"info": {"name": "roboemot"}, "state": "running"}


def test_run_restart_refuses_while_the_mobile_app_holds_the_robot(offline, robot, daemon, capsys):
    install()
    own_running(daemon)
    hold_slot(daemon)
    before = list(daemon.mutations)
    assert run_mod.main(run_args("--restart")) == 1
    assert "mobile app holds the robot (phone)" in flat(capsys.readouterr().err)
    assert daemon.mutations == before


def test_the_installer_prints_the_web_ui_address_with_the_configured_port(
    offline, robot, daemon, capsys
):
    """R4 S4 port point: a webui.port the owner set in app.yaml is the one printed."""
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    robot.files[APP_YAML] = yaml.safe_dump({"schema_version": 1, "webui": {"port": 8080}})
    inst.run_install(options(start=True, startup=False), quiet_ui(yes=True), ROOT)
    assert "✓ Web UI http://127.0.0.1:8080" in flat(capsys.readouterr().out)


def test_without_a_port_in_the_app_config_the_default_is_printed(offline, robot, daemon, capsys):
    inst.run_install(options(start=True, startup=False), quiet_ui(yes=True), ROOT)
    port = RoboEmotConfig().webui.port
    assert f"✓ Web UI http://127.0.0.1:{port}" in flat(capsys.readouterr().out)


# ------------------------------------------------------------------ realtime extra (SR-4)


def test_the_realtime_extra_is_installed_with_voice():
    """SPEC 4A, Installer: the realtime route's extra (websockets) comes with the voice one."""
    assert ReachyMiniWireless.extras == ("voice", "realtime")
    assert inst.EXTRA_DISK_MB["realtime"] == (1, 2)


def test_the_dry_run_adds_no_package_when_websockets_is_already_there(
    offline, robot, daemon, capsys
):
    """reachy-mini requires websockets, so the robot's apps venv has it: it stays pinned."""
    assert robot.env["websockets"] == "14.2"  # reachy-mini requires it
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert "○ Install the realtime extra websockets 14.2, already present" in flat(out)
    added = [line for line in out.splitlines() if "○ Add dependencies" in line]
    assert added and not any("websockets" in line for line in added)
    assert robot.env["websockets"] == "14.2" and "roboemot-core" not in robot.env


def test_the_dry_run_names_websockets_when_it_would_be_added(offline, robot, daemon, capsys):
    del robot.env["websockets"]
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert "○ Install the realtime extra websockets, not present yet" in flat(out)
    assert "websockets 15.0.1" in flat(out).split("○ Add dependencies", 1)[1]


def test_the_dry_run_flags_a_websockets_too_old_for_the_extra(offline, robot, daemon, capsys):
    """Fix round 1, M12: reachy-mini allows websockets 12, the extra needs 13 or later, so an
    old version is not reported as enough."""
    robot.env["websockets"] = "12.0"
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert "already present" not in out
    assert "○ Install the realtime extra websockets 12.0 present, needs >=13,<16" in flat(out)


@pytest.mark.parametrize(
    ("version", "spec", "ok"),
    [
        ("15.0.1", ">=13,<16", True),
        ("13", ">=13,<16", True),
        ("12.0", ">=13,<16", False),
        ("16.0", ">=13,<16", False),
        ("15.0rc1", ">=13,<16", False),  # unreadable counts as no
        ("14.2", "", True),
        ("14.2", "~=14.0", False),  # an operator it does not read counts as no
    ],
)
def test_version_meets_reads_plain_specifiers(version, spec, ok):
    assert inst.version_meets(version, spec) is ok


STOPPING = {"info": {"name": "roboemot", "source_kind": "installed"}, "state": "stopping"}


def test_start_waits_out_an_app_that_is_still_stopping(daemon, capsys):
    daemon.current = dict(STOPPING)
    daemon.stopping_reads = 3
    platform_for(daemon).start(quiet_ui())
    assert [(m, p) for m, p, _ in daemon.mutations] == [
        ("POST", "/api/move/play/wake_up"),
        ("POST", "/api/apps/start-app/roboemot"),
    ]
    assert daemon.current["state"] == "running"
    assert "roboemot is stopping" in flat(capsys.readouterr().out)


def test_start_names_the_stuck_stopping_state_and_the_reboot(daemon):
    """Robot run 2026-10-28 (daemon 1.13.0): an app that exited stays stopping; start says an
    app runs, stop says none runs. The start says so and sends nothing the daemon would refuse."""
    daemon.current = dict(STOPPING)
    with pytest.raises(WizardError, match="stuck") as err:
        platform_for(daemon).start(quiet_ui())
    assert "reboot" in str(err.value)
    assert daemon.mutations == []


# ---------------------------------------------------------------- cloud voice (SPEC 4A, 8, 12)

HF_TOKEN = "hf_installer_7c1a"
CLOUD_KEYS = [
    "llm_routes[pollen]",
    "voice.realtime_route.enabled=true",
    "voice.realtime_route.route=pollen",
    "voice.realtime_route.mode=platform",
    "voice.realtime_route.input=audio",
    "voice.realtime_route.skip_tool=true",
    "voice.realtime_route.transcript_timeout_s=10.0",
    "privacy.realtime_audio=true",
    "voice.backend=none",
    "voice.fallback_backend=none",
    "voice.wake_word_policy=never",
    "voice.direction.enabled=true",
    "voice.facing.enabled=true",
]
OWNER_ROUTE = {
    "name": "cloud",
    "base_url": "https://llm.example/v1",
    "api_key": "none",
    "model": "chat-small",
}


def cloud_install(**kw) -> None:
    install(cloud_voice=True, piper=None, **kw)


def written_route(cfg: dict) -> dict:
    return next(r for r in cfg["llm_routes"] if r["name"] == "pollen")


def test_the_cloud_voice_flag_parses_both_ways_and_defaults_to_unanswered():
    assert inst.parse_args(["--yes"]).cloud_voice is None
    assert inst.parse_args(["--cloud-voice"]).cloud_voice is True
    assert inst.parse_args(["--no-cloud-voice"]).cloud_voice is False


def test_the_reachy_platform_supplies_the_pollen_route():
    route = ReachyMiniWireless.cloud_voice
    assert route.entry() == {
        "name": "pollen",
        "kind": "realtime",
        "base_url": "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session",
        "api_key": "${secret:hf_token}",
        "model": "s2s",
    }
    assert route.mode == "platform" and route.secret == "hf_token"


def test_the_profile_keys_are_listed_in_full():
    keys = inst.cloud_voice_config(ReachyMiniWireless.cloud_voice)
    assert sorted(inst.key_text(k, v) for k, v in keys.items()) == sorted(CLOUD_KEYS)
    assert keys[("voice", "realtime_route", "transcript_timeout_s")] == 10.0  # the owner's


def cloud_app_config(existing: dict | None = None) -> dict:
    wanted, owned = inst.owned_app_config(ROOT, dev=False, voice=True)
    cloud = inst.cloud_voice_config(ReachyMiniWireless.cloud_voice)
    return inst.merge_app_config(existing or {}, {**wanted, **cloud}, owned | set(cloud))


def test_the_merged_cloud_voice_app_config_validates_against_the_schema():
    from roboemot.config import ConfigStack

    cfg = ConfigStack(app=cloud_app_config({"llm_routes": [OWNER_ROUTE]})).effective()
    assert cfg.voice.realtime_route.enabled and cfg.voice.realtime_route.route == "pollen"
    assert cfg.voice.realtime_route.input == "audio" and cfg.privacy.realtime_audio
    assert cfg.voice.backend == "none" and cfg.voice.fallback_backend == "none"
    assert cfg.voice.wake_word_policy == "never"
    assert cfg.voice.facing.enabled and cfg.voice.direction.enabled
    assert [r.name for r in cfg.llm_routes] == ["cloud", "pollen"]
    broken = cloud_app_config()
    broken["voice"]["direction"]["enabled"] = False  # facing needs direction
    with pytest.raises(Exception, match="voice.facing.enabled needs voice.direction.enabled"):
        ConfigStack(app=broken).effective()


def test_a_cloud_voice_install_writes_its_keys_and_keeps_the_owners(
    offline, robot, daemon, monkeypatch
):
    key_prompts(monkeypatch, HF_TOKEN)
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    owner = {
        "schema_version": 1,
        "webui": {"port": 8080},
        "voice": {"language": "en", "realtime_route": {"voice": "Serena"}},
        "llm_routes": [OWNER_ROUTE],
    }
    robot.files[APP_YAML] = yaml.safe_dump(owner)
    cloud_install()
    cfg = yaml.safe_load(robot.app_yaml)
    assert cfg["webui"] == {"port": 8080} and cfg["voice"]["language"] == "en"
    assert cfg["voice"]["realtime_route"]["voice"] == "Serena"
    assert cfg["llm_routes"][0] == OWNER_ROUTE
    assert written_route(cfg) == ReachyMiniWireless.cloud_voice.entry()
    assert cfg["privacy"] == {"realtime_audio": True}
    assert cfg["voice"]["wake_word_policy"] == "never" and cfg["voice"]["enabled"] is True
    keys = json.loads(robot.files[RECORD])["app_config_keys"]
    assert ["llm_routes", "pollen"] in keys and ["privacy", "realtime_audio"] in keys
    assert len(keys) == len(CLOUD_KEYS)


def test_a_reinstall_without_cloud_voice_removes_only_the_keys_it_wrote(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    cloud_install()
    cfg = yaml.safe_load(robot.app_yaml)
    cfg["webui"] = {"port": 8080}
    cfg["llm_routes"].append(OWNER_ROUTE)
    cfg["voice"]["language"] = "en"
    robot.files[APP_YAML] = yaml.safe_dump(cfg)
    capsys.readouterr()
    install(cloud_voice=False, piper=False)
    cfg = yaml.safe_load(robot.app_yaml)
    assert cfg == {
        "schema_version": 1,
        "safety": {"allow_unmeasured_dynamics": True},
        "voice": {"enabled": True, "language": "en"},
        "webui": {"port": 8080},
        "llm_routes": [OWNER_ROUTE],
    }
    assert json.loads(robot.files[RECORD])["app_config_keys"] == []
    assert "✓ Removed app config keys llm_routes[pollen]" in flat(capsys.readouterr().out)


def test_a_plain_install_leaves_hand_set_cloud_settings_alone(offline, robot, daemon):
    """Keys the owner set by hand (docs/realtime-voice-setup.md) are not the installer's."""
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    hand = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
    robot.files[APP_YAML] = yaml.safe_dump(hand)
    install()
    cfg = yaml.safe_load(robot.app_yaml)
    assert cfg["privacy"] == {"realtime_audio": True}
    assert cfg["voice"] == {"enabled": True, "wake_word_policy": "never"}


def test_uninstall_after_a_cloud_voice_install_leaves_the_robot_as_it_was(
    offline, robot, daemon, monkeypatch
):
    key_prompts(monkeypatch, HF_TOKEN)
    before = snapshot(robot, daemon)
    cloud_install()
    assert robot.secrets == {"hf_token": HF_TOKEN}
    uninstall()
    assert snapshot(robot, daemon) == before


def test_the_cloud_voice_dry_run_lists_every_key_and_the_piper_removal_and_writes_nothing(
    offline, robot, daemon, capsys
):
    install()  # local voice: Piper venv recorded
    files, dirs, record = dict(robot.files), set(robot.dirs), robot.files[RECORD]
    robot.commands.clear()
    capsys.readouterr()
    inst.run_install(options(dry_run=True, cloud_voice=True, piper=None), quiet_ui(yes=True), ROOT)
    out = flat(capsys.readouterr().out)
    for key in CLOUD_KEYS:
        assert key in out, key
    assert f"○ Delete the Piper venv {PIPER_VENV}" in out
    assert f"○ Ask for a secret 'hf_token', stored in {DATA}/secrets.json" in out
    assert "! Cloud voice needs its key without a terminal the real run stops" in out
    assert "--cloud-voice" in out and "--piper" not in out.split("Next:")[1]
    assert robot.files == files and robot.dirs == dirs and robot.files[RECORD] == record
    writes = ("rm ", "rmdir", "mkdir", "mktemp", "mv ", "umask")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]
    assert not any("secrets set" in c for c in robot.commands)


def test_cloud_voice_removes_the_recorded_piper_venv_and_skips_piper_and_its_voices(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    install()
    assert PIPER_PYTHON in robot.dirs and robot.fetches == ["models fetch --voice"]
    pips = len(robot.piper_pips)
    capsys.readouterr()
    cloud_install()
    assert PIPER_VENV not in robot.dirs and len(robot.piper_pips) == pips
    assert json.loads(robot.files[RECORD])["piper_venv"] is None
    out = flat(capsys.readouterr().out)
    assert f"✓ Deleted the Piper venv {PIPER_VENV}; the server speaks" in out
    assert "Voice without Piper" not in out
    fetch = robot.fetches[-1].split()
    assert "silero-vad" in fetch and "vosk-small-it" in fetch
    assert not [n for n in fetch if n.startswith("piper")] and "--voice" not in fetch
    uninstall()  # the record agrees with the robot: nothing left to do for Piper
    assert PIPER_VENV not in robot.dirs


def test_cloud_voice_counts_no_tts_voices_in_the_model_sizes():
    assert inst.models_bytes(False, tts=False) < inst.models_bytes(False)
    assert inst.models_bytes(True, tts=False) < inst.models_bytes(True)


def test_yes_without_the_flag_on_a_terminal_leaves_the_opt_in_off(
    offline, robot, daemon, monkeypatch, capsys
):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    ui = Ui(interactive=True, assume_yes=True, read=answers())
    opts = options(cloud_voice=None, start=False, startup=False, secrets=[])
    inst.run_install(opts, ui, ROOT)
    captured = capsys.readouterr()
    assert "Turn on cloud voice" not in captured.err
    assert "✓ Cloud voice off -y does not turn it on, --cloud-voice does" in flat(captured.out)
    cfg = yaml.safe_load(robot.app_yaml)
    assert "privacy" not in cfg and "llm_routes" not in cfg
    assert json.loads(robot.files[RECORD])["app_config_keys"] == []


@pytest.mark.parametrize("yes", [False, True])
def test_a_run_without_a_terminal_needs_a_cloud_voice_flag(offline, robot, daemon, yes):
    opts = options(cloud_voice=None, start=False, startup=False, yes=yes, piper=False, models=False)
    with pytest.raises(MissingFlag, match="--cloud-voice or --no-cloud-voice"):
        inst.run_install(opts, quiet_ui(yes=yes), ROOT)
    assert robot.commands == [] and daemon.requests == []


def test_the_terminal_question_shows_the_privacy_text_and_defaults_to_no(
    offline, robot, daemon, monkeypatch, capsys
):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    opts = Options(**{**FULL, "cloud_voice": None}, start=False, startup=False, secrets=[])
    inst.run_install(opts, typed("\n"), ROOT)
    captured = capsys.readouterr()
    text = flat(captured.out)
    assert (
        "the audio of every utterance spoken while a person faces the robot and looks at it "
        "leaves the robot for Pollen's server (Hugging Face), phone calls included" in text
    )
    assert (
        "The skip tool only decides whether the robot answers: the audio is already at the server"
        in text
    )  # noqa: E501
    assert "Turn on cloud voice (audio leaves the robot)? [y/N]" in captured.err
    assert "privacy" not in yaml.safe_load(robot.app_yaml)


def test_a_yes_on_the_terminal_turns_cloud_voice_on_and_asks_for_the_token(
    offline, robot, daemon, monkeypatch, capsys
):
    asked: list[str] = []

    def answer(self, question):
        asked.append(question)
        return HF_TOKEN if "secret" in question or question == "Repeat the value" else None

    monkeypatch.setattr(Ui, "ask_secret", answer)
    opts = Options(**{**FULL, "cloud_voice": None, "piper": None}, start=False, startup=False)
    inst.run_install(opts, typed("y\n", "\n"), ROOT)  # cloud voice: yes; other API key: no
    assert "Value of the secret 'hf_token'" in asked
    assert robot.secrets == {"hf_token": HF_TOKEN} and HF_TOKEN + "\n" in robot.stdins
    captured = capsys.readouterr()
    assert HF_TOKEN not in captured.out + captured.err
    assert not any(HF_TOKEN in c for c in robot.commands)
    assert HF_TOKEN not in robot.files[RECORD] and HF_TOKEN not in robot.app_yaml
    assert json.loads(robot.files[RECORD])["secrets"] == ["hf_token"]
    assert yaml.safe_load(robot.app_yaml)["privacy"] == {"realtime_audio": True}
    assert robot.piper_pips == []


def test_a_token_already_in_the_store_is_kept_without_asking(
    offline, robot, daemon, monkeypatch, capsys
):
    def answer(self, question):
        assert "secret" not in question, question

    monkeypatch.setattr(Ui, "ask_secret", answer)
    robot.secrets = {"hf_token": "theirs"}
    cloud_install(secrets=[])
    assert robot.secrets == {"hf_token": "theirs"} and robot.stdins == []
    assert "✓ Kept the secret hf_token: already in the store" in flat(capsys.readouterr().out)
    assert json.loads(robot.files[RECORD])["secrets"] == []


def test_no_token_at_the_end_of_the_checks_stops_with_nothing_changed(
    offline, robot, daemon, monkeypatch
):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    install()  # local voice: Piper venv and app.yaml
    before = snapshot(robot, daemon)
    record = robot.files[RECORD]
    with pytest.raises(WizardError, match="Cloud voice needs the 'hf_token' key"):
        cloud_install()
    assert snapshot(robot, daemon) == before and robot.files[RECORD] == record
    assert PIPER_VENV in robot.dirs and "privacy" not in yaml.safe_load(robot.app_yaml)


@pytest.mark.parametrize(
    ("kw", "what"),
    [
        (dict(piper=True), "--cloud-voice and --piper together"),
        (dict(voice=False), "--cloud-voice needs voice"),
    ],
)
def test_cloud_voice_refuses_flags_it_cannot_honour(offline, robot, daemon, kw, what):
    with pytest.raises(WizardError, match=what):
        install(cloud_voice=True, **kw)
    assert robot.commands == []


def test_a_platform_without_a_cloud_voice_route_refuses_the_flag_and_needs_no_answer(
    offline, robot, daemon, monkeypatch
):
    bare = ReachyMiniWireless(port=daemon.port)
    bare.cloud_voice = None
    monkeypatch.setattr(inst, "load_platform", lambda rid: bare)
    with pytest.raises(WizardError, match="No cloud voice for Reachy Mini Wireless"):
        install(cloud_voice=True)
    assert robot.commands == []
    inst.run_install(options(cloud_voice=None, start=False, startup=False), quiet_ui(True), ROOT)
    assert "privacy" not in yaml.safe_load(robot.app_yaml)


def test_fetch_args_without_tts_name_the_listener_and_vision_models_only():
    args = inst.fetch_args(vision=True, tts=False).split()
    assert args[:2] == ["models", "fetch"] and "--all" not in args and "--voice" not in args
    assert "silero-vad" in args and "vosk-small-it" in args and "vosk-small-en" in args
    assert not [a for a in args if a.startswith("piper")]
    from roboemot.vision import models as vision_models

    assert set(vision_models.load_manifest()) <= set(args)


def test_no_voice_on_a_terminal_is_never_asked_about_cloud_voice(
    offline, robot, daemon, monkeypatch, capsys
):
    monkeypatch.setattr(Ui, "ask_secret", lambda self, question: None)
    opts = Options(**{**FULL, "cloud_voice": None}, voice=False, start=False, startup=False)
    inst.run_install(opts, typed("\n"), ROOT)  # the API key question only
    assert "cloud voice" not in capsys.readouterr().err


@pytest.mark.parametrize("kw", [dict(voice=False), dict(piper=True)])
def test_an_unattended_run_that_rules_cloud_voice_out_needs_no_cloud_flag(
    offline, robot, daemon, kw
):
    install(cloud_voice=None, **kw)
    assert "privacy" not in (yaml.safe_load(robot.app_yaml or "{}") or {})


def test_web_ui_settings_that_override_the_profile_are_named_without_values(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    saved = {
        "voice.wake_word_policy": "unless_enrolled_in_view",
        "voice.realtime_route.transcript_timeout_s": 10,
        "webui.port": 8080,
    }
    robot.files[f"{DATA}/webui_settings.json"] = json.dumps({"v": 1, "overrides": saved})
    install(cloud_voice=True, piper=None, dry_run=True)
    out = flat(capsys.readouterr().out)
    assert (
        "! Web UI overrides "
        f"{DATA}/webui_settings.json sets voice.realtime_route.transcript_timeout_s, "
        "voice.wake_word_policy, which win over app.yaml"
    ) in out
    assert "unless_enrolled_in_view" not in out and "webui.port" not in out


SETTINGS = f"{DATA}/webui_settings.json"


def saved_settings(robot, **overrides) -> None:
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    robot.files[SETTINGS] = json.dumps({"v": 1, "overrides": overrides})


@pytest.mark.parametrize(
    ("saved", "cloud", "key"),
    [
        ({"privacy.realtime_audio": True}, False, "privacy.realtime_audio"),
        ({"voice.realtime_route.input": "audio"}, False, "voice.realtime_route.input"),
        ({"voice.facing.enabled": False}, True, "voice.facing.enabled"),
        ({"voice.direction.enabled": False}, True, "voice.direction.enabled"),
        ({"voice.realtime_route.route": "other"}, True, "voice.realtime_route.route"),
        ({"llm_routes.pollen.base_url": "https://x.example"}, True, "llm_routes.pollen.base_url"),
    ],
)
def test_saved_web_ui_settings_against_the_privacy_choice_stop_the_install_unchanged(
    offline, robot, daemon, monkeypatch, capsys, saved, cloud, key
):
    key_prompts(monkeypatch, HF_TOKEN)
    saved_settings(robot, **saved)
    files = dict(robot.files)
    with pytest.raises(WizardError, match="Web UI settings") as err:
        install(cloud_voice=cloud, piper=None if cloud else False)
    assert key in str(err.value) and "nothing was changed" in str(err.value)
    for value in saved.values():
        if isinstance(value, str) and "." in value:  # a distinctive value, not a common word
            assert value not in str(err.value)
    assert robot.files == files and robot.env.get("roboemot-core") is None


def test_saved_settings_that_agree_with_the_choice_pass(offline, robot, daemon, monkeypatch):
    key_prompts(monkeypatch, HF_TOKEN)
    saved_settings(robot, **{"privacy.realtime_audio": False})
    install(cloud_voice=False, piper=False)
    saved_settings(robot, **{"voice.facing.enabled": True, "llm_routes.cloud.model": "m"})
    install(cloud_voice=True, piper=None)
    assert yaml.safe_load(robot.app_yaml)["privacy"] == {"realtime_audio": True}


def test_yes_without_the_flag_over_a_recorded_profile_turns_it_off_and_says_so(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    cloud_install()
    capsys.readouterr()
    ui = Ui(interactive=True, assume_yes=True, read=answers())
    inst.run_install(options(cloud_voice=None, start=False, startup=False, secrets=[]), ui, ROOT)
    out = flat(capsys.readouterr().out)
    assert "✓ Turned cloud voice off (it was on): removes llm_routes[pollen], privacy." in out
    assert "local voice needs Piper (--piper)" in out
    cfg = yaml.safe_load(robot.app_yaml)
    assert "privacy" not in cfg and "llm_routes" not in cfg
    assert json.loads(robot.files[RECORD])["app_config_keys"] == []


def test_an_install_interrupted_before_the_app_config_move_still_removes_the_keys_later(
    offline, robot, daemon, monkeypatch
):
    key_prompts(monkeypatch, HF_TOKEN)
    install()
    real = robot.run

    def broken(command, check=True, stdin=None):
        if command.startswith("umask 077 && mkdir -p ") and "&& mv" in command:
            raise WizardError("connection lost")
        return real(command, check=check, stdin=stdin)

    monkeypatch.setattr(robot, "run", broken)
    with pytest.raises(WizardError, match="connection lost"):
        cloud_install()
    keys = json.loads(robot.files[RECORD])["app_config_keys"]
    assert ["privacy", "realtime_audio"] in keys and len(keys) == len(CLOUD_KEYS)
    monkeypatch.setattr(robot, "run", real)
    cfg = yaml.safe_load(robot.app_yaml)
    cfg["privacy"] = {"realtime_audio": True}  # as if the write had landed after all
    robot.files[APP_YAML] = yaml.safe_dump(cfg)
    install(cloud_voice=False, piper=False)
    assert "privacy" not in yaml.safe_load(robot.app_yaml)
    assert json.loads(robot.files[RECORD])["app_config_keys"] == []


def test_secret_hf_token_with_cloud_voice_asks_before_replacing_the_stored_token(
    offline, robot, daemon, monkeypatch, capsys
):
    asked: list[str] = []

    def answer(self, question):
        asked.append(question)
        return "hf_new_token_2" if "secret" in question or question == "Repeat the value" else None

    monkeypatch.setattr(Ui, "ask_secret", answer)
    robot.secrets = {"hf_token": "hf_old_token_1"}
    opts = Options(
        **{**FULL, "cloud_voice": True, "piper": None},
        start=False,
        startup=False,
        secrets=["hf_token"],
    )
    inst.run_install(opts, typed("y\n"), ROOT)  # replace it: yes
    err = capsys.readouterr().err
    assert "Secret 'hf_token' is already set. Replace it? [y/N]" in err
    assert asked.count("Value of the secret 'hf_token'") == 1
    assert robot.secrets == {"hf_token": "hf_new_token_2"}
    assert json.loads(robot.files[RECORD])["secrets"] == []  # it was the owner's before


def test_a_hand_written_pollen_route_is_replaced_with_a_warning_naming_the_fields(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    robot.dirs |= {f"{HOME}/.local", f"{HOME}/.local/share", DATA}
    hand = {**ReachyMiniWireless.cloud_voice.entry(), "model": "s2s-beta", "timeout_s": 30}
    robot.files[APP_YAML] = yaml.safe_dump({"llm_routes": [hand]})
    cloud_install()
    out = flat(capsys.readouterr().out)
    assert (
        "! Replaced a route llm_routes[pollen] as written by hand: changed model; lost timeout_s"
        in out
    )
    assert "s2s-beta" not in out
    assert written_route(yaml.safe_load(robot.app_yaml)) == ReachyMiniWireless.cloud_voice.entry()


def test_cloud_voice_deletes_the_tts_voice_files_and_the_dry_run_lists_them(
    offline, robot, daemon, monkeypatch, capsys
):
    key_prompts(monkeypatch, HF_TOKEN)
    install()
    voices = f"{DATA}/voice-models"
    tts = [
        f"{voices}/piper/it_IT-paola-medium.onnx",
        f"{voices}/piper/it_IT-paola-medium.onnx.verified",
        f"{voices}/piper/en_US-amy-medium.onnx.json",
    ]
    vad = f"{voices}/silero/silero_vad_v5.1.2.onnx"
    for f in [*tts, vad]:
        robot.files[f] = "x"
    robot.dirs.add(f"{voices}/piper")
    capsys.readouterr()
    install(cloud_voice=True, piper=None, dry_run=True)
    out = flat(capsys.readouterr().out)
    assert f"○ Delete the TTS voices {voices}: piper/it_IT-paola-medium.onnx," in out
    assert all(f in robot.files for f in tts)
    cloud_install()
    assert not [f for f in tts if f in robot.files] and vad in robot.files
    assert f"✓ Deleted the TTS voices {voices}" in flat(capsys.readouterr().out)


def test_a_record_with_unreadable_profile_keys_is_refused():
    from roboemot.install.record import InstallRecord

    base = {"app": "a", "platform": "p", "packages": ["x"]}
    for bad in ("voice", [["voice", 1]], [[]], [["voice", ""]]):
        with pytest.raises(WizardError, match="app_config_keys"):
            InstallRecord.from_json({**base, "app_config_keys": bad})
