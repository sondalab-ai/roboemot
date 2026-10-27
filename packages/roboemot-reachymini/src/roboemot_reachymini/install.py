"""Reachy Mini Wireless installer platform (SPEC 4A, Packaging; flow in ``roboemot.install``).

Facts from the daemon source (reachy_mini 1.11, checked on the robot):

- Every app shares one environment, ``/venvs/apps_venv``; the daemon runs an app as
  ``<apps_venv>/bin/python -u -m <entry point module>`` and lists apps from the
  ``reachy_mini_apps`` entry points found there.
- ``/venvs/.app_metadata/<app>.json`` exists only for apps the daemon installed itself (from
  Hugging Face); the uninstaller removes ours if one ever appears.
- REST API under ``/api``: daemon status, installed apps, current app, stop, wake-up, start,
  and the startup app (``PUT /api/apps/startup-app``), which the daemon starts on wake-up.
- The change record lives at ``/venvs/.roboemot-install.json``: writable by ``pollen``,
  outside the shared environment (pip would not touch it) and outside ``/tmp``.
"""

from __future__ import annotations

import json
import shlex
import shutil
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roboemot.install import Ui, WizardError
from roboemot.install.pyenv import RemotePython

APP_NAME = "roboemot"
APPS_VENV = "/venvs/apps_venv"
DAEMON_PORT = 8000
HTTP_TIMEOUT_S = 5.0
APP_START_TIMEOUT_S = 60.0
WAKE_SETTLE_S = 3.0
PACKAGES = ("roboemot-core", "roboemot-reachymini")
PKG_DATA = "roboemot-reachymini/src/roboemot_reachymini/data"
DAEMON_METADATA = "/venvs/.app_metadata"

PROBE = "from importlib.metadata import version; print(version('reachy_mini'))"
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
    record_path = "/venvs/.roboemot-install.json"
    space_paths = ("/tmp", APPS_VENV)
    distributions = PACKAGES
    modules = ("roboemot", "roboemot_reachymini")

    def __init__(self, port: int = DAEMON_PORT) -> None:
        self.port = port
        self.api: DaemonApi | None = None

    def _api(self) -> DaemonApi:
        assert self.api is not None, "connect() first"
        return self.api

    def connect(self, ui: Ui, host: str) -> None:
        """Resolve the host and reach the daemon."""
        try:
            addr = socket.getaddrinfo(host, self.port, proto=socket.IPPROTO_TCP)[0][4][0]
        except socket.gaierror as e:
            raise WizardError(f"cannot resolve {host} ({e}); try the robot's IP address") from e
        ui.ok(f"{host} -> {addr}")
        self.api = DaemonApi(f"http://{host}:{self.port}")
        status = self.api.daemon_status()
        version, name = status.get("version") or "?", status.get("robot_name")
        ui.ok(f"daemon {version} on {name}, state {status['state']}")
        if status["state"] != "running":
            raise WizardError(
                f"daemon state is {status['state']}; start it from Reachy Mini Control"
            )
        if not status.get("wireless_version"):
            ui.warn("this daemon is not a Wireless one: the apps venv layout may differ")

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
        if APP_NAME not in self._api().installed_apps():
            raise WizardError(f"the daemon does not list {APP_NAME} among installed apps")
        ui.ok(f"daemon lists {APP_NAME} as installed")

    def running_app(self) -> str | None:
        current = self._api().current_app()
        if current and current["state"] in ("starting", "running"):
            return current["info"]["name"]
        return None

    def stop_app(self) -> None:
        self._api().stop_current_app()

    def start(self, ui: Ui) -> None:
        """Stop the running app if any, wake the robot, start RoboEmot and wait for it."""
        api = self._api()
        other = self.running_app()
        if other:
            if not ui.confirm(f"{other} is running; stop it (only one app runs at a time)?", False):
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

    def autostart(self) -> str | None:
        return self._api().startup_app()

    def set_autostart(self, name: str | None) -> None:
        self._api().set_startup_app(name)

    def cleanup(self, ui: Ui, py: RemotePython, dry_run: bool) -> None:
        """Daemon-side leftovers: the app metadata file, if the daemon ever wrote one."""
        path = f"{DAEMON_METADATA}/{APP_NAME}.json"
        if py.remote.run(f"test -e {shlex.quote(path)} && echo yes || echo no") == "yes":
            if not dry_run:
                py.remote.run(f"rm -f {shlex.quote(path)}")
            ui.ok(f"{'would delete' if dry_run else 'deleted'} {path}")

    def verify_removed(self, ui: Ui) -> None:
        if APP_NAME in self._api().installed_apps():
            raise WizardError(f"the daemon still lists {APP_NAME}")
        ui.ok(f"daemon no longer lists {APP_NAME}")


__all__ = ["DaemonApi", "ReachyMiniWireless", "stage_packages"]
