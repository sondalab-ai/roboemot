"""The Reachy Mini daemon's REST API, the one HTTP client RoboEmot has for it (SPEC 4A, 11A, 12A).

Facts from the daemon (reachy_mini 1.11, Wi-Fi router 1.12): FastAPI on port 8000, routes under
``/api`` except ``/wifi/status``, which exists only on the Wireless daemon. The daemon runs RoboEmot
as its current app: ``stop-current-app`` sends SIGINT and kills the process after
``STOP_TIMEOUT_S``; ``restart-current-app`` does that and starts the app again. Diagnostics, the
web UI's Restart action and the installer all go through ``DaemonClient``; nothing else in this
package opens a connection to the daemon.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from roboemot.diagnostics import Target

APP_NAME = "roboemot"
DEFAULT_PORT = 8000
STOP_TIMEOUT_S = 20  # the daemon's stop_current_app default: SIGINT, then kill
RESTART_MARGIN_S = 10.0  # for the daemon to start the app again after the stop
RESTART_TIMEOUT_S = STOP_TIMEOUT_S + RESTART_MARGIN_S
REQUEST_TIMEOUT_S = 2.0
INSTALL_TIMEOUT_S = 5.0
APP_START_TIMEOUT_S = 60.0
API_PREFIX = "/api"
WIFI_STATUS_PATH = "/wifi/status"
RUNNING = "running"
STATE_QUERY = {
    "with_head_joints": "true",
    "with_passive_joints": "true",
    "with_doa": "true",
    "with_imu": "true",
}


class DaemonError(Exception):
    """A daemon call failed: unreachable, an HTTP error status or an undecodable answer. The
    message names the call and carries the daemon's ``detail`` when there is one."""


class DaemonClient:
    """Calls to the daemon of ``target`` (``host=None``: this machine). ``transport`` replaces
    the HTTP transport (tests). Each call opens its own short-lived connection."""

    def __init__(
        self,
        target: Target | None = None,
        *,
        timeout_s: float = REQUEST_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        target = target or Target()
        self.base_url = f"http://{target.host or 'localhost'}:{target.port or DEFAULT_PORT}"
        self._timeout_s = timeout_s
        self._transport = transport

    def _send(
        self,
        method: str,
        path: str,
        *,
        prefix: str = API_PREFIX,
        params: dict[str, str] | None = None,
        body: Any = None,
        timeout_s: float | None = None,
        missing_ok: bool = False,
    ) -> Any:
        """One request; the decoded JSON (None for an empty body, or for 404 when ``missing_ok``).
        Failures raise ``DaemonError``."""
        call = f"{method} {path}"
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self._timeout_s if timeout_s is None else timeout_s,
                transport=self._transport,
            ) as http:
                resp = http.request(method, prefix + path, params=params, json=body)
        except httpx.HTTPError as exc:
            raise DaemonError(f"{call} -> {str(exc) or type(exc).__name__}") from exc
        if missing_ok and resp.status_code == httpx.codes.NOT_FOUND:
            return None
        if resp.is_error:
            raise DaemonError(f"{call} -> HTTP {resp.status_code}: {_detail(resp)}")
        if not resp.content.strip():
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise DaemonError(f"{call} -> answer is not JSON") from exc

    def status(self) -> dict[str, Any]:
        """``GET /api/daemon/status``: name, version, backend state, errors."""
        return self._send("GET", "/daemon/status")

    def motors(self) -> dict[str, Any]:
        """``GET /api/motors/status``: enabled, disabled, gravity compensation."""
        return self._send("GET", "/motors/status")

    def state(self) -> dict[str, Any]:
        """``GET /api/state/full`` with joints, direction of arrival and the IMU."""
        return self._send("GET", "/state/full", params=STATE_QUERY)

    def wifi(self) -> dict[str, Any] | None:
        """``GET /wifi/status``; ``None`` when the daemon has no Wi-Fi router (not a Wireless)."""
        return self._send("GET", WIFI_STATUS_PATH, prefix="", missing_ok=True)

    def current_app(self) -> dict[str, Any] | None:
        """``GET /api/apps/current-app-status``: the app the daemon runs, or ``None``."""
        return self._send("GET", "/apps/current-app-status")

    def app_running(self, name: str = APP_NAME) -> bool:
        """Whether ``name`` is the daemon's current app in state running. A daemon that does
        not answer, or answers something else, counts as no."""
        try:
            status = self.current_app()
        except DaemonError:
            return False
        info = status.get("info") if isinstance(status, dict) else None
        current = info.get("name") if isinstance(info, dict) else None
        return current == name and status.get("state") == RUNNING

    def restart_current_app(self) -> None:
        """``POST /api/apps/restart-current-app``. The daemon stops this process (SIGINT) before
        it answers, so a dropped connection is the normal outcome and callers expect
        ``DaemonError``."""
        self._send("POST", "/apps/restart-current-app", timeout_s=RESTART_TIMEOUT_S)

    def installed_apps(self) -> list[str]:
        """Names of the apps the daemon lists as installed."""
        return [a["name"] for a in self._send("GET", "/apps/list-available/installed")]

    def stop_current_app(self) -> None:
        """``POST /api/apps/stop-current-app``."""
        self._send("POST", "/apps/stop-current-app")

    def wake_up(self) -> None:
        """``POST /api/move/play/wake_up``."""
        self._send("POST", "/move/play/wake_up")

    def start_app(self, name: str) -> dict[str, Any]:
        """``POST /api/apps/start-app/<name>``."""
        return self._send("POST", f"/apps/start-app/{name}")

    def startup_app(self) -> str | None:
        """The app the daemon starts on wake-up."""
        return self._send("GET", "/apps/startup-app").get("startup_app")

    def set_startup_app(self, name: str | None) -> None:
        """``PUT /api/apps/startup-app``."""
        self._send("PUT", "/apps/startup-app", body={"startup_app": name})

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
                raise DaemonError(f"{name} is no longer the current app (status: {status})")
            if status["state"] in (RUNNING, "error", "done"):
                return status
            sleep(poll_s)
        assert status is not None
        return status


def _detail(resp: httpx.Response) -> str:
    """The daemon's ``detail`` field when the body is a JSON object with one, else the text."""
    text = resp.text
    if text.startswith("{"):
        try:
            return str(json.loads(text).get("detail", text))
        except (ValueError, AttributeError):
            return text
    return text


__all__ = ["APP_NAME", "DEFAULT_PORT", "STOP_TIMEOUT_S", "DaemonClient", "DaemonError"]
