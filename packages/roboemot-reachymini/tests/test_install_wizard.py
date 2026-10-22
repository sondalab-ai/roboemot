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
import yaml
from roboemot import paths
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Target
from roboemot.install import Options, Ui, WizardError
from roboemot.install import run as run_mod
from roboemot.install.build import with_extras
from roboemot.install.caches import cache_path, clear_caches
from roboemot.install.piper_lock import lock_name
from roboemot_reachymini import install as reachy
from roboemot_reachymini.daemon import INSTALL_TIMEOUT_S, DaemonClient, DaemonError
from roboemot_reachymini.install import ReachyMiniWireless

ROOT = Path(__file__).resolve().parents[3]
RECORD = ReachyMiniWireless.record_path
APPS_PY = f"{reachy.APPS_VENV}/bin/python"
DAEMON_PY = "/venvs/mini_daemon/bin/python"
SITE = "/venvs/apps_venv/lib/python3.12/site-packages"
OWN = {"roboemot-core": "0.1.0.dev0", "roboemot-reachymini": "0.1.0.dev0"}
HOME = "/home/pollen"
DATA = f"{HOME}/.local/share/roboemot"
PIN_PATH = f"{DATA}/webui_pin.json"
PIPER_VENV = f"{DATA}/piper-venv"
PIPER_PYTHON = f"{PIPER_VENV}/bin/python"
PIN = "246810"


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
        self.app_yaml: str | None = None  # data/app.yaml as staged (set by the offline fixture)
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
        self.fetch_fails = False
        self.free_kb = 20_000_000
        self.free_kb_by_cache = self.free_kb  # free space while the uv cache is still there
        self.home_free_kb: int | None = None  # `df` below HOME (the models), else free_kb
        self.df_unreadable: set[str] = set()  # paths whose `df` answers nothing

    def _exists(self, path: str) -> bool:
        return path in self.dirs or (path == reachy.APPS_VENV and self.apps_venv)

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
            cfg = yaml.safe_load(self.app_yaml) if self.app_yaml is not None else None
            return f"roboemot_reachymini.main\n{json.dumps(cfg)}"
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


@pytest.fixture()
def offline(monkeypatch, robot, daemon):
    monkeypatch.setattr(inst, "load_platform", lambda rid: ReachyMiniWireless(port=daemon.port))
    monkeypatch.setattr(inst, "Remote", lambda user, host, control_dir: robot)

    def build(dirs, out):
        staged = dirs[0].parent / reachy.PKG_DATA / "app.yaml"
        robot.app_yaml = staged.read_text() if staged.exists() else None
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
    ui.confirm = lambda question, default: False
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
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice]"]
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
        ),
        quiet_ui(yes=True),
        ROOT,
    )
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice,vision]"]
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
        ),
        quiet_ui(yes=True),
        ROOT,
    )
    out = capsys.readouterr().out
    assert "would install the vision extra" in out and "voice and vision models" in out


def test_the_dry_run_with_no_vision_does_not_mention_vision(offline, robot, daemon, capsys):
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert "vision" not in capsys.readouterr().out


def test_not_enough_space_for_the_vision_weights_suggests_no_vision(offline, robot, daemon, capsys):
    robot.home_free_kb = needed_mib() * 1024
    install(vision=True)
    out = capsys.readouterr().out
    assert robot.fetches == [] and "--no-vision" in out


def test_vision_flag_adds_the_vision_extra_and_the_vision_weights(offline, robot, daemon):
    install(vision=True)
    assert [t.partition("[")[2] for t in core_targets(robot)] == ["voice,vision]"]
    assert robot.fetches == ["models fetch --all"]
    assert "opencv-python-headless" in json.loads(robot.files[RECORD])["added"]


# ---------------------------------------------------------------- voice on by default (SPEC 4A)


def staged_config(tmp_path, dev, voice):
    reachy.stage_packages(ROOT, tmp_path, dev, inst.app_config_overrides(voice))
    path = tmp_path / reachy.PKG_DATA / "app.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else None


def test_voice_is_on_by_default_and_no_voice_turns_it_off():
    assert inst.parse_args(["--yes"]).voice is True and Options().voice is True
    assert inst.parse_args(["--yes", "--no-voice"]).voice is False
    assert inst.parse_args(["--yes", "--voice"]).voice is True


def test_the_staged_app_config_carries_voice_and_the_dev_content(tmp_path):
    cfg = staged_config(tmp_path, dev=True, voice=True)
    assert cfg["voice"] == {"enabled": True}
    assert cfg["safety"] == {"allow_unmeasured_dynamics": True}


def test_without_dev_the_staged_app_config_is_voice_only(tmp_path):
    cfg = staged_config(tmp_path, dev=False, voice=True)
    assert cfg == {"schema_version": 1, "voice": {"enabled": True}}


def test_no_voice_with_dev_keeps_the_dev_content_only(tmp_path):
    cfg = staged_config(tmp_path, dev=True, voice=False)
    assert "voice" not in cfg and cfg["safety"] == {"allow_unmeasured_dynamics": True}


def test_no_voice_without_dev_writes_no_app_config(tmp_path):
    assert staged_config(tmp_path, dev=False, voice=False) is None


def test_the_staged_app_config_is_generated_with_a_header_and_a_note_per_part(tmp_path):
    reachy.stage_packages(ROOT, tmp_path, True, inst.app_config_overrides(True))
    text = (tmp_path / reachy.PKG_DATA / "app.yaml").read_text()
    assert text.startswith("# Written by the RoboEmot installer")
    assert "# voice: " in text and "# dev: " in text


def test_app_config_parts_that_share_a_top_level_key_are_refused():
    with pytest.raises(ValueError, match="share a top-level key"):
        reachy.app_yaml_text({"voice": {"x": 1}}, {"voice": {"enabled": True}})


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


def test_the_dry_run_with_no_voice_and_no_dev_says_there_is_no_app_config(
    offline, robot, daemon, capsys
):
    inst.run_install(
        options(dry_run=True, dev=False, voice=False, start=False, startup=False),
        quiet_ui(yes=True),
        ROOT,
    )
    out = capsys.readouterr().out
    assert "no app config file" in out and "voice.enabled" not in out


def test_verification_fails_when_the_installed_app_config_lacks_voice(offline, robot, monkeypatch):
    real = reachy.app_yaml_text
    monkeypatch.setattr(
        reachy, "app_yaml_text", lambda dev_content, app_config: real(dev_content, {})
    )
    with pytest.raises(WizardError, match="expected voice="):
        install()


def test_verification_fails_when_a_no_voice_install_finds_voice_in_the_app_config(
    offline, robot, monkeypatch
):
    real = reachy.app_yaml_text
    monkeypatch.setattr(
        reachy,
        "app_yaml_text",
        lambda dev_content, app_config: real(dev_content, {"voice": {"enabled": True}}),
    )
    with pytest.raises(WizardError, match="expected no voice"):
        install(voice=False)


def test_verification_accepts_the_app_config_as_asked_in_every_combination(offline, robot, daemon):
    for dev in (True, False):
        for voice in (True, False):
            install(dev=dev, voice=voice, piper=False, models=False)


def test_uninstall_after_a_default_install_restores_the_robot_and_records_no_app_config(
    offline, robot, daemon
):
    before = snapshot(robot, daemon)
    install()
    assert "app.yaml" not in robot.files[RECORD]
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


def test_the_models_question_defaults_to_yes(offline, robot, daemon, monkeypatch):
    asked = []
    monkeypatch.setattr(Ui, "confirm", lambda self, q, default: asked.append((q, default)) or True)
    inst.run_install(options(start=False, startup=False, yes=False), quiet_ui(), ROOT)
    assert [d for q, d in asked if q.startswith("Download the voice models")] == [True]


def test_not_enough_space_warns_and_skips_the_models_without_failing(
    offline, robot, daemon, capsys
):
    robot.home_free_kb = (needed_mib() - 100) * 1024
    install()
    out = capsys.readouterr().out
    assert robot.fetches == [] and "skipped" in out and "MB free" in out
    assert reachy.APPS_VENV and "roboemot-core" in robot.env


def app_need_mb(extras: tuple[str, ...] = ("voice",)) -> int:
    """MB the app install needs on the apps venv's filesystem (downloads and unpacked), with the
    venv already there."""
    return sum(inst.install_need_mb(ReachyMiniWireless(), None, extras, creating=False))


def test_the_space_check_counts_the_extras(offline, robot, capsys):
    """R4 S3: the free-space floor comes from what is installed: vision adds its share."""
    robot.free_kb = (app_need_mb() + 1) * 1024
    with pytest.raises(WizardError, match=f"need {app_need_mb(('voice', 'vision'))} MB"):
        install(vision=True)
    assert "roboemot-core" not in robot.env


def test_a_new_apps_venv_counts_the_daemon_venv_tree(offline, robot, capsys):
    """R4 S3: creating the apps venv installs reachy-mini's tree; the daemon venv's size
    (`du`) stands in for it, plus its downloads."""
    fresh_robot(robot)
    robot.daemon_venv_kb = 1_000 * 1024
    tree = 1_000 + int(1_000 * inst.ENV_DOWNLOAD_SHARE)
    robot.free_kb = (app_need_mb() + tree - 1) * 1024
    with pytest.raises(WizardError, match=f"need {app_need_mb() + tree} MB"):
        install()
    assert robot.files == {} and not robot.apps_venv


def test_a_too_old_python_is_refused_before_the_running_app_is_stopped(offline, robot, daemon):
    """R4 S5: the interpreter, architecture and Python checks come before the auto-stop."""
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.machine = "aarch64\n3.9"
    with pytest.raises(WizardError, match="target Python is 3.9"):
        install()
    assert STOP not in mutation_paths(daemon) and daemon.current["state"] == "running"


def test_the_auto_stop_and_the_cache_step_come_before_the_space_check(offline, robot, daemon):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.dirs.add(UV_CACHE)
    install()
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
    assert "were not downloaded" in capsys.readouterr().out
    assert "roboemot-core" in robot.env


def test_dry_run_reports_the_model_sizes_and_downloads_nothing(offline, robot, daemon, capsys):
    install(dry_run=True)
    assert robot.fetches == [] and VOICE_MODELS not in robot.dirs
    assert "would download the voice models" in capsys.readouterr().out
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
    assert inst.models_bytes(True) - inst.models_bytes(False) > inst.VISION_DERIVED_BYTES


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
    install()
    assert UV_CACHE not in robot.dirs
    assert f"rm -rf {UV_CACHE}" in robot.commands
    assert robot.commands.index(f"rm -rf {UV_CACHE}") < robot.commands.index(
        "mktemp -d /tmp/roboemot-install.XXXXXX"
    )
    out = capsys.readouterr().out
    assert "5.6G" in out and "cleared" in out
    assert UV_CACHE not in json.loads(robot.files[RECORD]).values()


def test_install_keeps_the_uv_cache_when_declined(offline, robot):
    robot.dirs.add(UV_CACHE)
    install(clear_caches=False)
    assert UV_CACHE in robot.dirs


@pytest.mark.parametrize("interactive", [True, False])
def test_the_cache_prompt_defaults_to_yes_only_on_a_terminal(robot, monkeypatch, interactive):
    robot.dirs.add(UV_CACHE)
    asked = []
    monkeypatch.setattr(Ui, "confirm", lambda self, q, default: asked.append((q, default)) or False)
    ui = Ui(interactive=interactive, assume_yes=False)
    clear_caches(ui, robot, HOME, ReachyMiniWireless.clearable_caches, None, False)
    assert UV_CACHE in robot.dirs
    assert [d for q, d in asked if UV_CACHE in q] == [interactive]


def test_an_unattended_run_without_consent_keeps_the_cache(offline, robot):
    robot.dirs.add(UV_CACHE)
    inst.run_install(options(start=False, startup=False, yes=False), quiet_ui(), ROOT)
    assert UV_CACHE in robot.dirs


def test_the_flag_clears_the_cache_without_yes(offline, robot):
    robot.dirs.add(UV_CACHE)
    inst.run_install(
        options(start=False, startup=False, yes=False, clear_caches=True), quiet_ui(), ROOT
    )
    assert UV_CACHE not in robot.dirs


def test_clearing_the_cache_can_make_a_full_robot_pass_the_space_check(offline, robot):
    robot.dirs.add(UV_CACHE)
    robot.free_kb_by_cache = (app_need_mb() - 100) * 1024
    install()
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
    install(dry_run=True)
    assert UV_CACHE in robot.dirs
    assert not any(c.startswith("rm -rf " + UV_CACHE) for c in robot.commands)
    assert f"would clear {UV_CACHE} (5.6G" in capsys.readouterr().out


def test_install_does_not_delete_a_cache_that_is_a_symlink(offline, robot, capsys):
    robot.dirs.add(UV_CACHE)
    robot.symlinks[UV_CACHE] = "/data/elsewhere"
    install()
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
    install()
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
    assert "ended with an error earlier (boom)" in capsys.readouterr().out


def test_start_reports_app_error(daemon):
    daemon.start_states = ["error"]
    with pytest.raises(WizardError, match="roboemot state error"):
        platform_for(daemon).start(quiet_ui())


@pytest.mark.parametrize("dev", [True, False])
def test_stage_packages_leaves_checkout_untouched(tmp_path, dev):
    repo_app_yaml = ROOT / "packages" / reachy.PKG_DATA / "app.yaml"
    existed = repo_app_yaml.exists()
    dirs = reachy.stage_packages(ROOT, tmp_path, dev, {})
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
    writes = ("rm ", "rmdir", "mkdir", "mktemp", "mv ")
    assert not [c for c in robot.commands if c.startswith(writes) or "&& mv" in c]
    assert robot.dirs == {HOME}
    out = capsys.readouterr().out
    assert "would add scipy 1.18.1" in out
    assert f"would create the data folder {DATA}" in out
    assert "would stop conversation" in out and "dry run passed" in out


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


def test_install_fails_when_dev_config_mismatches(offline, robot, monkeypatch):
    monkeypatch.setattr(reachy, "app_yaml_text", lambda dev_content, app_config: "voice: {}\n")
    with pytest.raises(WizardError, match="expected dev=True"):
        install()


def test_install_fails_when_daemon_does_not_list_app(offline, daemon):
    daemon.hide_own = True
    with pytest.raises(WizardError, match="does not list roboemot"):
        install()


def test_main_returns_1_and_prints_reason(offline, robot, capsys, monkeypatch):
    monkeypatch.setattr(reachy, "app_yaml_text", lambda dev_content, app_config: "voice: {}\n")
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
    assert "no record of the data folder" in capsys.readouterr().out


def test_uninstall_dry_run_shows_the_data_folder_and_keeps_it(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall(dry_run=True)
    assert f"would delete the data folder {DATA}" in capsys.readouterr().out
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


def test_install_warns_about_a_data_dir_in_the_staged_config(tmp_path, capsys):
    (tmp_path / "app.yaml").write_text("schema_version: 1\ndata_dir: /mnt/big\n")
    (tmp_path / "other.yaml").write_text("data_dir: /ignored\n")
    robot = FakeRobot()
    inst.warn_data_dir_overrides(quiet_ui(), robot, [tmp_path])
    out = capsys.readouterr().out
    assert "data_dir=/mnt/big" in out and "/ignored" not in out


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
    real = DaemonClient(Target("localhost", 8000), timeout_s=INSTALL_TIMEOUT_S)
    status = real.status()
    assert status["state"] == "running" and status["simulation_enabled"]
    assert isinstance(real.installed_apps(), list)
    assert real.current_app() is None
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
    assert "would ask for the operator PIN" in capsys.readouterr().out


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
    assert "would ask for the secret 'cloud_key'" in capsys.readouterr().out


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
    monkeypatch.setattr(Ui, "confirm", lambda self, question, default: False)
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
    assert "would remove the secrets cloud_key" in capsys.readouterr().out


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
    with pytest.raises(WizardError, match="no hash-pinned Piper lock"):
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
    assert f"would create the Piper virtualenv {PIPER_VENV}" in capsys.readouterr().out


def piper_args(robot, dry_run):
    return Ui(False, True), ReachyMiniWireless(), robot, inst.InstallRecord("r", "p", []), dry_run


def test_piper_dry_run_runs_no_mktemp_copy_or_venv_and_still_rejects_an_unlocked_target(
    offline, robot, daemon
):
    inst.install_piper(*piper_args(robot, True))
    assert not any(c.startswith("mktemp") or " -m venv " in c for c in robot.commands)
    assert robot.copies == [] and robot.piper_pips == []
    robot.machine = "aarch64\n3.9"
    with pytest.raises(WizardError, match="no hash-pinned Piper lock"):
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
    assert "would delete the Piper virtualenv" in capsys.readouterr().out


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
    out = capsys.readouterr().out
    assert (
        f"would create the environment {reachy.APPS_VENV}" in out and "reachy-mini==1.11.0" in out
    )
    assert "dependency check skipped" in out
    assert "dry run passed" in out
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
    with pytest.raises(WizardError, match=f"neither is {DAEMON_PY}"):
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
    assert "finishing it" in capsys.readouterr().out
    assert "reachy-mini" in json.loads(robot.files[RECORD])["env_baseline"]
    uninstall()
    assert snapshot(robot, daemon) == before and not robot.apps_venv


def test_a_venv_without_its_base_package_that_the_installer_did_not_create_is_refused(
    offline, robot, daemon
):
    del robot.env["reachy-mini"]
    with pytest.raises(WizardError, match="not created by this installer"):
        install()
    assert robot.files == {}


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
    assert f"would delete the environment {reachy.APPS_VENV}" in capsys.readouterr().out


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
    assert "uninstalled" in capsys.readouterr().out


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
    assert "removed with the environment" in capsys.readouterr().out


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
    with pytest.raises(WizardError, match="exists without an interpreter"):
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
        "creating the apps venv",
        "creating the Piper virtualenv",
        "installing Python packages",
        "reinstalling the app wheels",
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
    assert "uninstalling Python packages\n" in out
    assert "✓ uninstalling Python packages (" in out


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
    out = capsys.readouterr().out
    assert f"web UI: http://127.0.0.1:{RoboEmotConfig().webui.port}" in out
    assert "python scripts/run_on_robot.py" in out


def run_args(*extra: str) -> list[str]:
    return ["--host", "127.0.0.1", "--robot", "reachy-mini-wireless", "-y", *extra]


def test_run_starts_the_installed_app_without_ssh(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    robot.commands.clear()
    robot.streamed.clear()
    assert run_mod.main(run_args()) == 0
    out = capsys.readouterr().out
    assert "RoboEmot run" in out and "RoboEmot installer" not in out
    assert daemon.current["state"] == "running"
    assert f"web UI: http://127.0.0.1:{RoboEmotConfig().webui.port}" in out
    assert robot.commands == [] and robot.streamed == []


def test_run_says_to_install_first_when_the_app_is_missing(offline, robot, daemon, capsys):
    assert run_mod.main(run_args()) == 1
    assert "not installed: run the installer first" in capsys.readouterr().out
    assert daemon.mutations == []


def test_run_dry_run_changes_nothing(offline, robot, daemon, capsys):
    install()
    daemon.current = {"info": {"name": "conversation"}, "state": "running"}
    before = list(daemon.mutations)
    assert run_mod.main(run_args("--dry-run")) == 0
    out = capsys.readouterr().out
    assert "would stop conversation" in out and "dry run passed" in out
    assert daemon.mutations == before


def test_run_reports_the_mobile_app_holding_the_robot(offline, robot, daemon, capsys):
    install()
    daemon.lock = {"state": "remote_session", "holder_name": "phone"}
    assert run_mod.main(run_args()) == 1
    assert "mobile app holds the robot (phone); close it first" in capsys.readouterr().out


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
    out = capsys.readouterr().out
    assert "would refuse: the Reachy mobile app holds the robot" in out
    assert daemon.current is None


def test_installer_dry_run_reports_that_it_would_refuse(offline, robot, daemon, capsys):
    hold_slot(daemon)
    inst.run_install(options(dry_run=True), quiet_ui(yes=True), ROOT)
    assert "would refuse: the Reachy mobile app holds the robot" in capsys.readouterr().out


def test_install_with_a_held_slot_finishes_warns_and_points_to_the_run_script(
    offline, robot, daemon, capsys
):
    hold_slot(daemon)
    inst.run_install(options(start=True, startup=False), quiet_ui(yes=True), ROOT)
    out = capsys.readouterr().out
    assert "roboemot not started: the Reachy mobile app holds the robot" in out
    assert "python scripts/run_on_robot.py" in out
    assert "installed" in out and "RoboEmot" in out
    assert daemon.current is None and RECORD in robot.files


def test_a_new_apps_venv_prints_one_check_line(offline, robot, daemon, capsys):
    fresh_robot(robot)
    install()
    out = capsys.readouterr().out
    assert f"created {reachy.APPS_VENV} with reachy-mini==1.11.0" in out
    assert "✓ creating the apps venv" not in out
    first = out.index("installing Python packages")
    assert "✓ installing Python packages" not in out[first : out.index("created /venvs")]


def test_uninstall_pip_is_announced_without_the_minutes_note(offline, robot, daemon, capsys):
    install()
    capsys.readouterr()
    uninstall()
    out = capsys.readouterr().out
    assert "uninstalling Python packages\n" in out
    assert "uninstalling Python packages (can take" not in out


# ------------------------------------------------------- auto-stop of our own running app


STOP = ("POST", "/api/apps/stop-current-app")


def mutation_paths(daemon) -> list[tuple[str, str]]:
    return [(m, p) for m, p, _ in daemon.mutations]


def test_install_stops_its_own_running_app_before_changing_packages(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    install()
    assert STOP in mutation_paths(daemon)
    assert "stopped roboemot before installing: a running process keeps the old code" in (
        capsys.readouterr().out
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
    assert "would stop roboemot before installing" in capsys.readouterr().out
    assert daemon.mutations == [] and daemon.current["info"]["name"] == "roboemot"


def test_start_prompt_defaults_to_yes_after_the_auto_stop(offline, robot, daemon):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    asked: list[tuple[str, bool]] = []
    ui = quiet_ui(yes=True)
    ui.confirm = lambda question, default: asked.append((question, default)) or True
    inst.run_install(options(startup=False, yes=True), ui, ROOT)
    assert ("Start roboemot now?", True) in asked
    assert daemon.current["info"]["name"] == "roboemot"


def test_a_failure_after_the_auto_stop_says_how_to_start_it_again(offline, robot, daemon, capsys):
    daemon.current = {"info": {"name": "roboemot"}, "state": "running"}
    robot.conflict = {"numpy": "2.1.0"}
    ui = quiet_ui()
    ui.confirm = lambda question, default: "shared packages" not in question
    with pytest.raises(WizardError, match="shared packages left as they were"):
        inst.run_install(options(start=False, startup=False, yes=False), ui, ROOT)
    out = capsys.readouterr().out
    assert "roboemot was stopped before installing" in out
    assert "python scripts/run_on_robot.py" in out


def test_a_failure_without_an_auto_stop_prints_no_start_hint(offline, robot, daemon, capsys):
    robot.conflict = {"numpy": "2.1.0"}
    ui = quiet_ui()
    ui.confirm = lambda question, default: "shared packages" not in question
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
    assert out.count("would stop roboemot") == 1
