"""Reachy Mini sources for the debug report (SPEC 12A, Diagnostics; daemon API in SPEC 4A).

Registered in the ``roboemot.diagnostics`` entry-point group under the manifest id. Each
section is one read-only call to the daemon's REST API (reachy_mini 1.11, Wi-Fi 1.12):

- ``robot.daemon``: ``GET /api/daemon/status`` (name, version, hardware id, backend state and
  control-loop stats, errors, Wi-Fi address);
- ``robot.motors``: ``GET /api/motors/status`` (enabled, disabled, gravity_compensation);
- ``robot.state``: ``GET /api/state/full`` with joints, direction of arrival and the IMU
  (accelerometer, gyroscope, orientation, board temperature; ``null`` on Lite and simulation);
- ``robot.app``: ``GET /api/apps/current-app-status``;
- ``robot.wifi``: ``GET /wifi/status`` (reachy_mini 1.12; mode ``hotspot``, ``wlan``,
  ``disconnected`` or ``busy``, the connected network, and the number of saved networks, never
  their names). Unlike the others this router has no ``/api`` prefix and exists only on the
  Wireless daemon: elsewhere the section says it is not available.

The daemon reports no battery, CPU or memory figures, and no Wi-Fi signal strength; those come
from the generic ``local`` sources (``network.wifi``) when the report is built on the robot
(``scripts/robot_ssh.py --debug``).
"""

from __future__ import annotations

from typing import Any

import httpx
from roboemot.diagnostics import FnSource, Source, Target

DEFAULT_PORT = 8000
REQUEST_TIMEOUT_S = 2.0
WIFI_STATUS_PATH = "/wifi/status"
STATE_QUERY = {
    "with_head_joints": "true",
    "with_passive_joints": "true",
    "with_doa": "true",
    "with_imu": "true",
}


def sources(target: Target, client: httpx.Client | None = None) -> list[Source]:
    """Daemon sections for ``target`` (``host=None``: the daemon on this machine)."""
    base = f"http://{target.host or 'localhost'}:{target.port or DEFAULT_PORT}"
    http = client or httpx.Client(base_url=base, timeout=REQUEST_TIMEOUT_S)

    def get(path: str, params: dict[str, str] | None = None) -> Any:
        resp = http.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def wifi() -> dict[str, Any]:
        resp = http.get(WIFI_STATUS_PATH)
        if resp.status_code == 404:
            return {"available": False, "reason": "no Wi-Fi router on this daemon (not a Wireless)"}
        resp.raise_for_status()
        return wifi_summary(resp.json())

    return [
        FnSource("robot.daemon", lambda: get("/api/daemon/status"), "robot"),
        FnSource("robot.motors", lambda: get("/api/motors/status"), "robot"),
        FnSource("robot.state", lambda: get("/api/state/full", STATE_QUERY), "robot"),
        FnSource("robot.app", lambda: get("/api/apps/current-app-status"), "robot"),
        FnSource("robot.wifi", wifi, "robot"),
    ]


def wifi_summary(status: dict[str, Any]) -> dict[str, Any]:
    """Mode and connected network of ``GET /wifi/status``; saved networks only as a count."""
    return {
        "available": True,
        "mode": status.get("mode"),
        "connected_network": status.get("connected_network"),
        "known_networks": len(status.get("known_networks") or []),
    }


__all__ = ["DEFAULT_PORT", "sources"]
