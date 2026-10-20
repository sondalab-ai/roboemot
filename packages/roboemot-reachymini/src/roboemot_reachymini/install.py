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

import re
import shlex
import shutil
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Target
from roboemot.install import Ui, WizardError
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
    "print((pathlib.Path(roboemot_reachymini.__file__).parent / 'data' / 'app.yaml').exists())"
)


@contextmanager
def _wizard_errors() -> Iterator[None]:
    """Daemon failures as the wizard's own error type, so the flow reports them and stops."""
    try:
        yield
    except DaemonError as exc:
        raise WizardError(str(exc)) from exc


def stage_packages(root: Path, dest: Path, dev: bool) -> list[Path]:
    """Copy both packages into ``dest`` with the app's package data, leaving the checkout alone.

    Mirrors scripts/sync-package-data.sh, plus data/app.yaml from dev-unmeasured.yaml in dev.
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
    if dev:
        shutil.copy2(root / "configs/dev-unmeasured.yaml", data / "app.yaml")
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
        requirement = plan
        with ui.progress(f"creating the apps venv {APPS_VENV}", report=False):
            remote.run(f"{shlex.quote(DAEMON_PYTHON)} -m venv {shlex.quote(APPS_VENV)}")
        result = RemotePython(remote, self.python, ui).install([requirement], None, report=False)
        if not result.ok:
            raise WizardError(f"pip install failed in the new {APPS_VENV}:\n{result.output}")
        ui.ok(f"created {APPS_VENV} with {requirement}")

    def env_users(self, py: RemotePython) -> list[str]:
        """Apps other than ours that the daemon would list from this environment."""
        return py.py(OTHER_APPS).split()

    def probe(self, ui: Ui, py: RemotePython) -> None:
        ui.ok(f"reachy_mini {py.py(PROBE)} in the apps venv")

    def stage(self, root: Path, dest: Path, dev: bool) -> list[Path]:
        return stage_packages(root, dest, dev)

    def verify(self, ui: Ui, py: RemotePython, dev: bool) -> None:
        """The app loads in the apps venv, runs as a module, and the daemon lists it."""
        module, has_dev_cfg = py.py(VERIFY).splitlines()
        ui.ok(f"entry point {APP_NAME} -> {module}")
        if (has_dev_cfg == "True") != dev:
            raise WizardError(f"data/app.yaml present={has_dev_cfg}, expected dev={dev}")
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
        other = self.running_app()
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
        with _wizard_errors():
            api.start_app(APP_NAME)
            status = api.wait_app(APP_NAME)
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
