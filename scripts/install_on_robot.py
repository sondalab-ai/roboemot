#!/usr/bin/env python3
"""Install wizard for the RoboEmot app on Reachy Mini Wireless (SPEC 4A, Packaging).

Runs on a PC on the same LAN as the robot, never on the robot. It finds the robot by mDNS
(roboemot.discovery, SPEC 4) or takes a host, builds the two wheels
locally, copies them over SSH, installs them into the robot's apps venv, checks that the
daemon sees the app, and optionally stops the running app, wakes the robot, starts RoboEmot
and sets it as the startup app. Every prompt has a flag, so it also runs unattended.

    python scripts/install_on_robot.py              # interactive wizard
    python scripts/install_on_robot.py --dry-run    # read-only checks, robot untouched
    python scripts/install_on_robot.py --host 192.168.1.20 --dev --start --yes

--dev installs configs/dev-unmeasured.yaml as the app config, so the app runs before the
manifest dynamics are measured (SPEC 12). Without it the app refuses to start.

Needs the dev extra (`pip install -e "packages/roboemot-reachymini[dev]"`) for
`sondalab-palette`, the Sondalab terminal colour roles.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import sondalab_palette as sl
except ImportError:  # pragma: no cover
    sys.exit(
        "install_on_robot: missing sondalab-palette; "
        'pip install -e "packages/roboemot-reachymini[dev]"'
    )

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "roboemot"
DEFAULT_HOST = "reachy-mini.local"
APPS_VENV = "/venvs/apps_venv"
DAEMON_PORT = 8000
REMOTE_TMP = "/tmp/roboemot-install"
MIN_PYTHON = (3, 11)  # requires-python of both packages
MIN_FREE_MB = 300  # wheels plus numpy/scipy/openai if the apps venv lacks them
HTTP_TIMEOUT_S = 5.0
APP_START_TIMEOUT_S = 60.0
WAKE_SETTLE_S = 3.0
PACKAGES = ("roboemot-core", "roboemot-reachymini")
PKG_DATA = "roboemot-reachymini/src/roboemot_reachymini/data"


class WizardError(Exception):
    """A step failed; the message says what to do about it."""


# ---------------------------------------------------------------- terminal UI


def paint(text: str, role: str) -> str:
    """Colour ``text`` with a Sondalab role, only when stdout is a terminal."""
    return sl.paint(text, role) if sys.stdout.isatty() else text


class Ui:
    """Wizard output and prompts. Non-interactive runs take defaults; ``--yes`` accepts all."""

    def __init__(self, interactive: bool, assume_yes: bool) -> None:
        self.interactive = interactive
        self.assume_yes = assume_yes
        self.step_no = 0
        self.steps = 0

    def title(self, text: str, note: str = "") -> None:
        print(
            f"\n{paint('RoboEmot', 'accent')} {text}" + (f"  {paint(note, 'warn')}" if note else "")
        )

    def step(self, name: str) -> None:
        self.step_no += 1
        print(f"\n{paint(f'[{self.step_no}/{self.steps}]', 'dim')} {paint(name, 'prompt')}")

    def ok(self, msg: str) -> None:
        print(f"  {paint('✓', 'ok')} {msg}")

    def info(self, msg: str) -> None:
        print(f"  {paint('·', 'dim')} {paint(msg, 'dim')}")

    def warn(self, msg: str) -> None:
        print(f"  {paint('!', 'warn')} {msg}")

    def fail(self, msg: str) -> None:
        print(f"  {paint('✗', 'err')} {msg}")

    def ask(self, question: str, default: str) -> str:
        if not self.interactive:
            return default
        try:
            answer = input(f"  {paint('?', 'prompt')} {question} {paint(f'[{default}]', 'dim')} ")
        except EOFError:
            return default
        return answer.strip() or default

    def choose(self, question: str, options: list[str]) -> int:
        """Numbered pick list; returns the chosen index (the first one by default)."""
        for n, label in enumerate(options, 1):
            print(f"    {paint(str(n), 'accent')}  {label}")
        while True:
            answer = self.ask(question, "1")
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.warn(f"pick a number from 1 to {len(options)}")

    def confirm(self, question: str, default: bool) -> bool:
        if self.assume_yes:
            return True
        if not self.interactive:
            return default
        hint = "Y/n" if default else "y/N"
        try:
            answer = input(f"  {paint('?', 'prompt')} {question} {paint(f'[{hint}]', 'dim')} ")
        except EOFError:
            return default
        answer = answer.strip().lower()
        return default if not answer else answer in ("y", "yes")


# ---------------------------------------------------------------- daemon REST API


class DaemonApi:
    """The Reachy Mini daemon REST API (FastAPI, prefix ``/api``), reachy_mini 1.11+."""

    def __init__(self, base_url: str, timeout_s: float = HTTP_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        """Send one request and return the decoded JSON (None for an empty body)."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base_url}/api{path}", data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if detail.startswith("{"):
                detail = json.loads(detail).get("detail", detail)
            raise WizardError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise WizardError(f"{method} {path} -> {getattr(e, 'reason', e)}") from e
        return json.loads(raw) if raw.strip() else None

    def daemon_status(self) -> dict[str, Any]:
        return self._call("GET", "/daemon/status")

    def installed_apps(self) -> list[str]:
        return [a["name"] for a in self._call("GET", "/apps/list-available/installed")]

    def current_app(self) -> dict[str, Any] | None:
        return self._call("GET", "/apps/current-app-status")

    def stop_current_app(self) -> None:
        self._call("POST", "/apps/stop-current-app")

    def wake_up(self) -> None:
        self._call("POST", "/move/play/wake_up")

    def start_app(self, name: str) -> dict[str, Any]:
        return self._call("POST", f"/apps/start-app/{name}")

    def startup_app(self) -> str | None:
        return self._call("GET", "/apps/startup-app").get("startup_app")

    def set_startup_app(self, name: str | None) -> None:
        self._call("PUT", "/apps/startup-app", {"startup_app": name})

    def wait_app(
        self,
        name: str,
        timeout_s: float = APP_START_TIMEOUT_S,
        poll_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        """Poll until ``name`` is running, or return its last status (error, done, timeout)."""
        deadline = clock() + timeout_s
        status: dict[str, Any] | None = None
        while clock() < deadline:
            status = self.current_app()
            if status is None or status["info"]["name"] != name:
                raise WizardError(f"{name} is no longer the current app (status: {status})")
            if status["state"] in ("running", "error", "done"):
                return status
            sleep(poll_s)
        assert status is not None
        return status


# ---------------------------------------------------------------- SSH


@dataclass
class Remote:
    """SSH/scp to the robot through one multiplexed connection, so ssh asks the password once."""

    user: str
    host: str
    control_dir: Path

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    def _opts(self) -> list[str]:
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.control_dir}/%C",
            "-o", "ControlPersist=120",
            "-o", "ConnectTimeout=10",
        ]  # fmt: skip

    def run(self, command: str, check: bool = True) -> str:
        """Run ``command`` on the robot and return its stdout. ssh prompts on the tty."""
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, command], capture_output=True, text=True
        )
        if check and proc.returncode != 0:
            raise WizardError(f"on robot: {command}\n{proc.stderr.strip() or proc.stdout.strip()}")
        return proc.stdout.strip()

    def connect(self) -> None:
        """Open the master connection interactively, so a password prompt reaches the user."""
        proc = subprocess.run(
            ["ssh", *self._opts(), self.target, "true"], stdout=subprocess.DEVNULL
        )
        if proc.returncode != 0:
            raise WizardError(
                f"ssh {self.target} failed. Check host and user. On a first connection run "
                f"`ssh {self.target}` once to accept the host key; "
                f"`ssh-copy-id {self.target}` then avoids the password."
            )

    def copy(self, files: list[Path], dest: str) -> None:
        proc = subprocess.run(
            ["scp", "-q", *self._opts(), *map(str, files), f"{self.target}:{dest}/"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise WizardError(f"scp failed: {proc.stderr.strip()}")

    def close(self) -> None:
        subprocess.run(
            ["ssh", *self._opts(), "-O", "exit", self.target], capture_output=True, check=False
        )


# ---------------------------------------------------------------- local build


def stage_packages(dest: Path, dev: bool) -> Path:
    """Copy both packages into ``dest`` with the app's package data, leaving the checkout alone.

    Mirrors scripts/sync-package-data.sh, plus data/app.yaml from dev-unmeasured.yaml in dev.
    """
    for pkg in PACKAGES:
        shutil.copytree(
            ROOT / "packages" / pkg,
            dest / pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist", ".*"),
        )
    data = dest / PKG_DATA
    (data / "presets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs/manifests/reachy-mini-wireless.yaml", data)
    for preset in (ROOT / "configs/presets").glob("*.yaml"):
        shutil.copy2(preset, data / "presets")
    (data / "app.yaml").unlink(missing_ok=True)
    if dev:
        shutil.copy2(ROOT / "configs/dev-unmeasured.yaml", data / "app.yaml")
    return dest


def build_wheels(staged: Path, out: Path) -> list[Path]:
    """Build both wheels with pip (no dependencies) and return their paths, core first."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(out),
         *(str(staged / p) for p in PACKAGES)],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if proc.returncode != 0:
        raise WizardError(f"wheel build failed:\n{proc.stderr.strip()}")
    wheels = [next(out.glob(f"{p.replace('-', '_')}-*.whl")) for p in PACKAGES]
    return wheels


def third_party_requirements() -> list[str]:
    """Runtime dependencies of both packages, minus the packages themselves."""
    reqs: list[str] = []
    for pkg in PACKAGES:
        meta = tomllib.loads((ROOT / "packages" / pkg / "pyproject.toml").read_text())
        reqs += [r for r in meta["project"]["dependencies"] if not r.startswith(PACKAGES)]
    return sorted(set(reqs))


# ---------------------------------------------------------------- wizard steps

PY = f"{APPS_VENV}/bin/python"
PROBE = (
    "import sys, platform; from importlib.metadata import version; "
    "print(platform.machine()); print('%d.%d' % sys.version_info[:2]); "
    "print(version('reachy_mini'))"
)
VERIFY = (
    "from importlib.metadata import entry_points; "
    "from reachy_mini import ReachyMiniApp; "
    f"ep = [e for e in entry_points(group='reachy_mini_apps') if e.name == '{APP_NAME}']; "
    "assert ep, 'no reachy_mini_apps entry point named roboemot'; "
    "cls = ep[0].load(); assert issubclass(cls, ReachyMiniApp), cls; "
    "import roboemot_reachymini, pathlib; "
    "print(cls.__module__ + '.' + cls.__name__); "
    "print((pathlib.Path(roboemot_reachymini.__file__).parent / 'data' / 'app.yaml').exists())"
)


@dataclass
class Options:
    host: str | None
    user: str | None
    dev: bool | None
    start: bool | None
    startup: bool | None
    dry_run: bool
    yes: bool


def find_robots() -> list[tuple[str, str]]:
    """Robots announcing themselves on the LAN as (label, address), via roboemot.discovery."""
    from roboemot.discovery import discover, load_hints

    found = discover(load_hints([ROOT / "configs/manifests/reachy-mini-wireless.yaml"]))
    return [(f"{f.record.name}  {f.address}  ({f.record.host})", f.address) for f in found]


def pick_host(ui: Ui) -> str:
    """Offer the robots found by mDNS, or ask for a host when none answers."""
    ui.info("looking for robots on the network (mDNS)")
    try:
        found = find_robots()
    except (ImportError, RuntimeError, OSError) as e:
        ui.warn(f"discovery unavailable: {e}")
        found = []
    if not found:
        return ui.ask("No robot found. Robot hostname or IP", DEFAULT_HOST)
    other = "another host or IP"
    i = ui.choose("Which robot?", [label for label, _ in found] + [other])
    return found[i][1] if i < len(found) else ui.ask("Robot hostname or IP", DEFAULT_HOST)


def check_robot(ui: Ui, host: str) -> DaemonApi:
    """Resolve the host and reach the daemon."""
    try:
        addr = socket.getaddrinfo(host, DAEMON_PORT, proto=socket.IPPROTO_TCP)[0][4][0]
    except socket.gaierror as e:
        raise WizardError(f"cannot resolve {host} ({e}); try the robot's IP address") from e
    ui.ok(f"{host} -> {addr}")
    api = DaemonApi(f"http://{host}:{DAEMON_PORT}")
    status = api.daemon_status()
    version, name = status.get("version") or "?", status.get("robot_name")
    ui.ok(f"daemon {version} on {name}, state {status['state']}")
    if status["state"] != "running":
        raise WizardError(f"daemon state is {status['state']}; start it from Reachy Mini Control")
    if not status.get("wireless_version"):
        ui.warn("this daemon is not a Wireless one: the apps venv layout may differ")
    return api


def check_ssh(ui: Ui, remote: Remote) -> None:
    """Log in once and check architecture, Python, the SDK and free space."""
    ui.info(f"connecting to {remote.target} (ssh may ask for the password)")
    remote.connect()
    ui.ok(f"ssh {remote.target}")
    if remote.run(f"test -x {PY} && echo yes || echo no") != "yes":
        raise WizardError(f"{PY} not found: is this a Reachy Mini Wireless?")
    arch, py, sdk = remote.run(f"{PY} -c {shlex.quote(PROBE)}").splitlines()
    if arch != "aarch64":
        raise WizardError(f"robot architecture is {arch}, expected aarch64")
    if tuple(map(int, py.split("."))) < MIN_PYTHON:
        raise WizardError(
            f"apps venv has Python {py}; RoboEmot needs {'.'.join(map(str, MIN_PYTHON))}+"
        )
    ui.ok(f"{arch}, apps venv Python {py}, reachy_mini {sdk}")
    for path in ("/tmp", APPS_VENV):
        free_kb = int(remote.run(f"df -Pk {path} | tail -1").split()[3])
        if free_kb < MIN_FREE_MB * 1024:
            raise WizardError(f"only {free_kb // 1024} MB free on {path}, need {MIN_FREE_MB} MB")
        ui.ok(f"{free_kb // 1024} MB free on {path}")


def install(ui: Ui, remote: Remote, wheels: list[Path], dry_run: bool) -> None:
    """Copy and install the wheels, or in a dry run resolve only the dependencies on the robot."""
    if dry_run:
        reqs = " ".join(shlex.quote(r) for r in third_party_requirements())
        out = remote.run(f"{PY} -m pip install --dry-run --disable-pip-version-check {reqs}")
        would = [line for line in out.splitlines() if line.startswith("Would install")]
        ui.ok(would[0] if would else "dependencies already satisfied in the apps venv")
        ui.info("dry run: nothing copied or installed")
        return
    remote.run(f"rm -rf {REMOTE_TMP} && mkdir -p {REMOTE_TMP}")
    remote.copy(wheels, REMOTE_TMP)
    ui.ok(f"copied {', '.join(w.name for w in wheels)}")
    files = " ".join(f"{REMOTE_TMP}/{w.name}" for w in wheels)
    ui.info("installing into the apps venv (first time: numpy/scipy may take minutes)")
    pip = f"{PY} -m pip install --disable-pip-version-check -q"
    remote.run(f"{pip} {files} && {pip} --no-deps --force-reinstall {files}")
    remote.run(f"rm -rf {REMOTE_TMP}", check=False)
    ui.ok("installed")


def verify(ui: Ui, api: DaemonApi, remote: Remote, dev: bool) -> None:
    """The app loads in the apps venv and the daemon lists it."""
    cls, has_dev_cfg = remote.run(f"{PY} -c {shlex.quote(VERIFY)}").splitlines()
    ui.ok(f"entry point {APP_NAME} -> {cls}")
    if (has_dev_cfg == "True") != dev:
        raise WizardError(f"data/app.yaml present={has_dev_cfg}, expected dev={dev}")
    if APP_NAME not in api.installed_apps():
        raise WizardError(f"the daemon does not list {APP_NAME} among installed apps")
    ui.ok(f"daemon lists {APP_NAME} as installed")


def start(ui: Ui, api: DaemonApi) -> None:
    """Stop the running app if any, wake the robot, start RoboEmot and wait for it."""
    current = api.current_app()
    if current and current["state"] in ("starting", "running"):
        other = current["info"]["name"]
        if not ui.confirm(f"{other} is running; stop it (only one app runs at a time)?", True):
            raise WizardError(f"{other} is still running; RoboEmot not started")
        api.stop_current_app()
        ui.ok(f"stopped {other}")
    api.wake_up()
    ui.ok("wake-up requested")
    time.sleep(WAKE_SETTLE_S)
    api.start_app(APP_NAME)
    status = api.wait_app(APP_NAME)
    if status["state"] != "running":
        raise WizardError(
            f"{APP_NAME} state {status['state']}: {status.get('error') or 'no detail'}"
        )
    ui.ok(f"{APP_NAME} is running")


def dry_run_start(ui: Ui, api: DaemonApi) -> None:
    """Report what the start step would do, without doing it."""
    current = api.current_app()
    if current and current["state"] in ("starting", "running"):
        ui.info(f"would stop {current['info']['name']}, wake the robot and start {APP_NAME}")
    else:
        ui.info(f"no app running; would wake the robot and start {APP_NAME}")
    ui.info(f"startup app now: {api.startup_app() or 'none'}")


def run(opts: Options, ui: Ui) -> None:
    ui.steps = 6
    ui.title(
        "installer for Reachy Mini Wireless",
        "dry run: the robot is not modified" if opts.dry_run else "",
    )

    ui.step("Robot")
    host = opts.host or pick_host(ui)
    api = check_robot(ui, host)

    ui.step("SSH access")
    user = opts.user or ui.ask("SSH user", "pollen")
    # The ssh control socket path must stay under 104 bytes: macOS $TMPDIR is too long.
    with (
        tempfile.TemporaryDirectory(prefix="roboemot-") as tmp,
        tempfile.TemporaryDirectory(prefix="re-ssh-", dir="/tmp") as sock_dir,
    ):
        remote = Remote(user, host, Path(sock_dir))
        try:
            check_ssh(ui, remote)

            ui.step("Configuration")
            dev = opts.dev
            if dev is None:
                ui.info("manifest dynamics are not measured yet: dev mode lets the app run")
                ui.info(
                    "with conservative fallback limits (SPEC 12); without it, it refuses to start"
                )
                dev = ui.confirm("Install in dev mode?", True)
            ui.ok("dev mode (configs/dev-unmeasured.yaml)" if dev else "production config")

            ui.step("Build")
            wheels = build_wheels(stage_packages(Path(tmp) / "src", dev), Path(tmp) / "wheels")
            ui.ok(f"built {', '.join(w.name for w in wheels)}")

            ui.step("Install" if not opts.dry_run else "Install (dependency check only)")
            install(ui, remote, wheels, opts.dry_run)
            if not opts.dry_run:
                verify(ui, api, remote, dev)
            elif APP_NAME in api.installed_apps():
                ui.info(f"{APP_NAME} is already installed from an earlier run")
        finally:
            remote.close()

    ui.step("Start")
    if opts.dry_run:
        dry_run_start(ui, api)
        ui.title("dry run passed", "")
        return
    if opts.start if opts.start is not None else ui.confirm(f"Start {APP_NAME} now?", True):
        start(ui, api)
    startup = opts.startup
    if startup is None:
        startup = ui.confirm(f"Start {APP_NAME} automatically when the robot wakes up?", False)
    if startup:
        api.set_startup_app(APP_NAME)
        ui.ok(f"{APP_NAME} is the startup app (starts on the robot's first wake-up)")
    ui.title("installed", "")
    ui.info(f"stop:   curl -X POST http://{host}:{DAEMON_PORT}/api/apps/stop-current-app")
    ui.info(f"start:  curl -X POST http://{host}:{DAEMON_PORT}/api/apps/start-app/{APP_NAME}")


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", help="robot hostname or IP (default: pick from robots found by mDNS)")
    p.add_argument("--user", help="SSH user (default: ask, pollen)")
    p.add_argument("--dev", action=argparse.BooleanOptionalAction, help="dev config (SPEC 12)")
    p.add_argument(
        "--start", action=argparse.BooleanOptionalAction, help="start the app after install"
    )
    p.add_argument(
        "--startup-app", dest="startup", action=argparse.BooleanOptionalAction,
        help="make RoboEmot the app the daemon starts on wake-up",
    )  # fmt: skip
    p.add_argument(
        "--dry-run", action="store_true", help="read-only checks; nothing on the robot changes"
    )
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to every confirmation")
    a = p.parse_args(argv)
    return Options(a.host, a.user, a.dev, a.start, a.startup, a.dry_run, a.yes)


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    ui = Ui(interactive=sys.stdin.isatty() and not os.environ.get("CI"), assume_yes=opts.yes)
    try:
        run(opts, ui)
    except WizardError as e:
        ui.fail(str(e))
        return 1
    except KeyboardInterrupt:
        ui.fail("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
