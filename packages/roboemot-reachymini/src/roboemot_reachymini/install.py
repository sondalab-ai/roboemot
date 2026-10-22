"""Reachy Mini Wireless installer platform (SPEC 4A, Packaging; flow in ``roboemot.install``).

Facts from the daemon source (reachy_mini 1.12, checked on the robot):

- Every app shares one environment, ``/venvs/apps_venv``. A freshly flashed robot has only
  ``/venvs/mini_daemon``: the daemon creates the apps venv lazily on the first app install
  (``<daemon python> -m venv``, then ``pip install reachy-mini``, unpinned, then the app). The
  installer does the same when it is missing, pinned to the daemon's own ``reachy_mini``
  version, and records it so uninstall can remove it. The daemon runs an app as
  ``<apps_venv>/bin/python -u -m <entry point module>`` and lists apps from the
  ``reachy_mini_apps`` entry points found there.
- ``/venvs/.app_metadata/<app>.json`` exists only for apps the daemon installed itself (from
  Hugging Face); the uninstaller removes ours if one ever appears.
- REST API under ``/api`` (client: ``roboemot_reachymini.daemon``): daemon status, installed
  apps, current app, stop, wake-up, start, and the startup app (``PUT /api/apps/startup-app``),
  which the daemon starts on wake-up.
- The change record lives at ``/venvs/.roboemot-install.json``: writable by ``pollen``,
  outside the shared environment (pip would not touch it) and outside ``/tmp``.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Target
from roboemot.install import Ui, WizardError, describe_app_config
from roboemot.install.pyenv import RemotePython
from roboemot.install.remote import Remote

from .daemon import APP_NAME, DEFAULT_PORT, INSTALL_TIMEOUT_S, DaemonClient, DaemonError

APPS_VENV = "/venvs/apps_venv"
DAEMON_PYTHON = "/venvs/mini_daemon/bin/python"
VERSION_PATTERN = re.compile(r"[0-9][0-9A-Za-z.+!-]*")
WAKE_SETTLE_S = 3.0
PACKAGES = ("roboemot-core", "roboemot-reachymini")
PKG_DATA = "roboemot-reachymini/src/roboemot_reachymini/data"
DAEMON_METADATA = "/venvs/.app_metadata"

PROBE = "from importlib.metadata import version; print(version('reachy_mini'))"
OTHER_APPS = (
    "from importlib.metadata import entry_points; "
    f"print(*[e.name for e in entry_points(group='reachy_mini_apps') if e.name != '{APP_NAME}'],"
    " sep='\\n')"
)
VERIFY = (
    "from importlib.metadata import entry_points; "
    "from reachy_mini import ReachyMiniApp; "
    f"ep = [e for e in entry_points(group='reachy_mini_apps') if e.name == '{APP_NAME}']; "
    f"assert ep, 'no reachy_mini_apps entry point named {APP_NAME}'; "
    "cls = ep[0].load(); assert issubclass(cls, ReachyMiniApp), cls; "
    "import roboemot_reachymini, pathlib; "
    "print(ep[0].module); "
    "import json, yaml; "
    "f = pathlib.Path(roboemot_reachymini.__file__).parent / 'data' / 'app.yaml'; "
    "print(json.dumps(yaml.safe_load(f.read_text()) or {} if f.exists() else None))"
)
DEV_NOTE = (
    "dev: the manifest dynamics are not measured yet (SPEC 12), so the app runs with "
    "conservative fallback limits (install with --no-dev once they are)"
)
VOICE_NOTE = "voice: the installer turns the voice pipeline on for this robot (--no-voice skips it)"


@contextmanager
def _wizard_errors() -> Iterator[None]:
    """Daemon failures as the wizard's own error type, so the flow reports them and stops."""
    try:
        yield
    except DaemonError as exc:
        raise WizardError(str(exc)) from exc


def app_yaml_text(dev_content: dict[str, object], app_config: dict[str, object]) -> str:
    """The installer's ``data/app.yaml``: a header, then each part under a comment saying what it
    is for, generated from the dev file's content and the core's ``app_config`` (voice).
    Parts must not share a top-level key (that would be a duplicate YAML key)."""
    dev_content = {k: v for k, v in dev_content.items() if k != "schema_version"}
    if set(dev_content) & set(app_config):
        raise ValueError("dev content and installer app config share a top-level key")
    lines = [
        "# Written by the RoboEmot installer (scripts/install_on_robot.py); every install rewrites",
        "# it and uninstall removes it with the package. Settings changed on the robot at runtime",
        "# live in the app data folder, not here.",
        "schema_version: 1",
    ]
    for note, part in ((VOICE_NOTE, app_config), (DEV_NOTE, dev_content)):
        if part:
            lines += [f"# {note}", yaml.safe_dump(part, sort_keys=False).rstrip()]
    return "\n".join(lines) + "\n"


def stage_packages(
    root: Path, dest: Path, dev: bool, app_config: dict[str, object] | None = None
) -> list[Path]:
    """Copy both packages into ``dest`` with the app's package data, leaving the checkout alone.

    Mirrors scripts/sync-package-data.sh, plus data/app.yaml (``app_yaml_text``) when there is
    something to put in it: the dev content from dev-unmeasured.yaml with ``dev``, and
    ``app_config`` (core's installer settings, e.g. voice.enabled).
    """
    for pkg in PACKAGES:
        shutil.copytree(
            root / "packages" / pkg,
            dest / pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist", ".*"),
        )
    data = dest / PKG_DATA
    (data / "presets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "configs/manifests/reachy-mini-wireless.yaml", data)
    for preset in (root / "configs/presets").glob("*.yaml"):
        shutil.copy2(preset, data / "presets")
    (data / "app.yaml").unlink(missing_ok=True)
    dev_content = yaml.safe_load((root / "configs/dev-unmeasured.yaml").read_text()) if dev else {}
    if dev_content or app_config:
        (data / "app.yaml").write_text(app_yaml_text(dev_content, app_config or {}))
    return [dest / pkg for pkg in PACKAGES]


class ReachyMiniWireless:
    manifest_id = "reachy-mini-wireless"
    display_name = "Reachy Mini Wireless"
    app_name = APP_NAME
    default_host = "reachy-mini.local"
    default_user = "pollen"
    arch = "aarch64"
    python = f"{APPS_VENV}/bin/python"
    env_dir = APPS_VENV
    base_python = DAEMON_PYTHON
    record_path = "/venvs/.roboemot-install.json"
    space_paths = ("/tmp", APPS_VENV)
    distributions = PACKAGES
    modules = ("roboemot", "roboemot_reachymini")
    extras = ("voice",)
    clearable_caches = ("~/.cache/uv",)  # the uv download cache left by the robot image's setup

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self.api: DaemonClient | None = None

    def _api(self) -> DaemonClient:
        assert self.api is not None, "connect() first"
        return self.api

    def connect(self, ui: Ui, host: str) -> None:
        """Resolve the host and reach the daemon."""
        try:
            addr = socket.getaddrinfo(host, self.port, proto=socket.IPPROTO_TCP)[0][4][0]
        except socket.gaierror as e:
            raise WizardError(f"cannot resolve {host} ({e}); try the robot's IP address") from e
        ui.ok(f"{host} -> {addr}")
        self.api = DaemonClient(Target(host, self.port), timeout_s=INSTALL_TIMEOUT_S)
        with _wizard_errors():
            status = self.api.status()
        version, name = status.get("version") or "?", status.get("robot_name")
        ui.ok(f"daemon {version} on {name}, state {status['state']}")
        if status["state"] != "running":
            raise WizardError(
                f"daemon state is {status['state']}; start it from Reachy Mini Control"
            )
        if not status.get("wireless_version"):
            ui.warn("this daemon is not a Wireless one: the apps venv layout may differ")

    def plan_env(self, remote: Remote) -> str:
        """The requirement the new apps venv gets: ``reachy-mini`` at the daemon's own version.
        Read-only, so it runs before anything is recorded."""
        version = RemotePython(remote, DAEMON_PYTHON).py(PROBE).strip()
        if not VERSION_PATTERN.fullmatch(version):
            raise WizardError(
                f"cannot read the reachy_mini version of {DAEMON_PYTHON}: {version!r}"
            )
        return f"reachy-mini=={version}"

    def create_env(self, ui: Ui, remote: Remote, plan: str) -> None:
        """Create the apps venv the way the daemon does and install ``plan`` (from
        ``plan_env``) into it."""
        with ui.progress(f"creating the apps venv {APPS_VENV}", report=False):
            remote.run(f"{shlex.quote(DAEMON_PYTHON)} -m venv {shlex.quote(APPS_VENV)}")
        result = RemotePython(remote, self.python, ui).install([plan], None, report=False)
        if not result.ok:
            raise WizardError(f"pip install failed in the new {APPS_VENV}:\n{result.output}")
        ui.ok(f"created {APPS_VENV} with {plan}")

    def env_users(self, py: RemotePython) -> list[str]:
        """Apps other than ours that the daemon would list from this environment."""
        return py.py(OTHER_APPS).split()

    def probe(self, ui: Ui, py: RemotePython) -> None:
        ui.ok(f"reachy_mini {py.py(PROBE)} in the apps venv")

    def stage(self, root: Path, dest: Path, dev: bool, app_config: dict[str, object]) -> list[Path]:
        return stage_packages(root, dest, dev, app_config)

    def verify(self, ui: Ui, py: RemotePython, dev: bool, app_config: dict[str, object]) -> None:
        """The app loads in the apps venv and runs as a module, the installed ``data/app.yaml``
        has the dev flag and the installer's settings (voice) as asked, and the daemon lists it."""
        module, raw = py.py(VERIFY).splitlines()
        ui.ok(f"entry point {APP_NAME} -> {module}")
        found = json.loads(raw) or {}
        actual_dev = (found.get("safety") or {}).get("allow_unmeasured_dynamics") is True
        if actual_dev != dev:
            raise WizardError(f"data/app.yaml has dev={actual_dev}, expected dev={dev}")
        for key, group in app_config.items():
            got = found.get(key)
            if got != group:
                raise WizardError(f"data/app.yaml has {key}={got}, expected {key}={group}")
        if not app_config and "voice" in found:
            raise WizardError(f"data/app.yaml has voice={found['voice']}, expected no voice")
        ui.ok(f"data/app.yaml: {describe_app_config(dev, app_config) or 'absent'}")
        with _wizard_errors():
            installed = self._api().installed_apps()
        if APP_NAME not in installed:
            raise WizardError(f"the daemon does not list {APP_NAME} among installed apps")
        ui.ok(f"daemon lists {APP_NAME} as installed")

    def running_app(self) -> str | None:
        with _wizard_errors():
            current = self._api().current_app()
        if current and current["state"] in ("starting", "running"):
            return current["info"]["name"]
        return None

    def stop_app(self) -> None:
        with _wizard_errors():
            self._api().stop_current_app()

    def app_installed(self) -> bool:
        """Whether the daemon lists RoboEmot among its installed apps."""
        with _wizard_errors():
            return APP_NAME in self._api().installed_apps()

    def app_url(self, host: str) -> str | None:
        """The app's web UI address from the default ``webui.port`` (as ``custom_app_url`` in
        the app); a ``webui.port`` override on the robot is not seen."""
        return f"http://{host}:{RoboEmotConfig().webui.port}"

    def start_blocker(self) -> str | None:
        """Why the app cannot start now, read-only: the Reachy mobile app holds the robot's
        app slot through a remote session, and the daemon then refuses to start local apps."""
        with _wizard_errors():
            lock = self._api().app_lock_status()
        if lock and lock.get("state") == "remote_session":
            holder = lock.get("holder_name") or "a remote session"
            return f"the Reachy mobile app holds the robot ({holder}); close it first"
        return None

    def start(self, ui: Ui) -> None:
        """Stop the running app if any, wake the robot, start RoboEmot and wait for it."""
        api = self._api()
        blocker = self.start_blocker()
        if blocker:
            raise WizardError(blocker)
        self._clear_errored_app(ui)
        other = self.running_app()
        if other == APP_NAME:
            self._await_own(ui, f"{APP_NAME} is already running")
            return
        if other:
            if not ui.confirm(f"{other} is running; stop it (only one app runs at a time)?", True):
                raise WizardError(f"{other} is still running; RoboEmot not started")
            with _wizard_errors():
                api.stop_current_app()
            ui.ok(f"stopped {other}")
        with _wizard_errors():
            api.wake_up()
        ui.ok("wake-up requested")
        time.sleep(WAKE_SETTLE_S)
        if self.running_app() == APP_NAME:
            self._await_own(ui, f"{APP_NAME} started by the daemon on wake-up")
            return
        try:
            with _wizard_errors():
                api.start_app(APP_NAME)
        except WizardError:
            if not self._current_is_own():
                raise
            self._await_own(ui, f"{APP_NAME} started by the daemon on wake-up")
            return
        self._await_own(ui)

    def restart(self, ui: Ui) -> None:
        """Restart the running RoboEmot through the daemon and wait until it runs again. It asks
        first, and a no is the default (scripts pass ``-y``): the daemon's stop sends the head to
        its zero pose before the app starts again."""
        blocker = self.start_blocker()
        if blocker:
            raise WizardError(blocker)
        question = (
            f"restart {APP_NAME}? The head moves: it goes to its zero pose while the app stops, "
            "then the app starts again"
        )
        if not ui.confirm(question, False):
            raise WizardError(f"{APP_NAME} not restarted")
        with _wizard_errors():
            self._api().restart_current_app()
        ui.ok("restart requested")
        self._await_own(ui)

    def _clear_errored_app(self, ui: Ui) -> None:
        """The daemon refuses every start while the current app is in ``error`` (it counts as
        running); clear such a leftover with a stop so a new start can go through."""
        with _wizard_errors():
            current = self._api().current_app()
        if not current or current["state"] != "error":
            return
        name = current["info"]["name"]
        ui.warn(f"{name} ended with an error earlier ({current.get('error') or 'no detail'})")
        with _wizard_errors():
            self._api().stop_current_app()
        ui.ok(f"cleared {name}")

    def _current_is_own(self) -> bool:
        """Whether RoboEmot is the daemon's current app, in any state (an errored start counts:
        it is reported, not started again)."""
        with _wizard_errors():
            current = self._api().current_app()
        return bool(current) and current["info"]["name"] == APP_NAME

    def _await_own(self, ui: Ui, note: str | None = None) -> None:
        """Wait until RoboEmot is the current app and running, or raise with its error."""
        if note:
            ui.ok(note)
        with _wizard_errors():
            status = self._api().wait_app(APP_NAME)
        if status["state"] != "running":
            raise WizardError(
                f"{APP_NAME} state {status['state']}: {status.get('error') or 'no detail'}"
            )
        ui.ok(f"{APP_NAME} is running")

    def autostart(self) -> str | None:
        with _wizard_errors():
            return self._api().startup_app()

    def set_autostart(self, name: str | None) -> None:
        with _wizard_errors():
            self._api().set_startup_app(name)

    def cleanup(self, ui: Ui, py: RemotePython, dry_run: bool) -> None:
        """Daemon-side leftovers: the app metadata file, if the daemon ever wrote one."""
        path = f"{DAEMON_METADATA}/{APP_NAME}.json"
        if py.remote.run(f"test -e {shlex.quote(path)} && echo yes || echo no") == "yes":
            if not dry_run:
                py.remote.run(f"rm -f {shlex.quote(path)}")
            ui.ok(f"{'would delete' if dry_run else 'deleted'} {path}")

    def verify_removed(self, ui: Ui) -> None:
        with _wizard_errors():
            installed = self._api().installed_apps()
        if APP_NAME in installed:
            raise WizardError(f"the daemon still lists {APP_NAME}")
        ui.ok(f"daemon no longer lists {APP_NAME}")


__all__ = ["ReachyMiniWireless", "stage_packages"]
