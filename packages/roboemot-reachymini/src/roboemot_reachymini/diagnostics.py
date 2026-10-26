"""Reachy Mini sources for the debug report (SPEC 12A, Diagnostics; daemon API in SPEC 4A).

Registered in the ``roboemot.diagnostics`` entry-point group under the manifest id. Each
section is one read-only call to the daemon's REST API through ``DaemonClient`` (reachy_mini
1.12):

- ``robot.daemon``: ``GET /api/daemon/status`` (name, version, hardware id, backend state and
  control-loop stats, errors, Wi-Fi address);
- ``robot.motors``: ``GET /api/motors/status`` (enabled, disabled, gravity_compensation);
- ``robot.state``: ``GET /api/state/full`` with joints, direction of arrival and the IMU
  (accelerometer, gyroscope, orientation, board temperature; ``null`` on Lite and simulation);
- ``robot.app``: the daemon's current app status;
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

from roboemot.diagnostics import FnSource, Source, Target

from .daemon import DaemonClient


def sources(target: Target, daemon: DaemonClient | None = None) -> list[Source]:
    """Daemon sections for ``target`` (``host=None``: the daemon on this machine)."""
    api = daemon or DaemonClient(target)

    def wifi() -> dict[str, Any]:
        status = api.wifi()
        if status is None:
            return {"available": False, "reason": "no Wi-Fi router on this daemon (not a Wireless)"}
        return wifi_summary(status)

    return [
        FnSource("robot.daemon", api.status, "robot"),
        FnSource("robot.motors", api.motors, "robot"),
        FnSource("robot.state", api.state, "robot"),
        FnSource("robot.app", api.current_app, "robot"),
        FnSource("robot.wifi", wifi, "robot"),
    ]


def wifi_summary(status: Any) -> dict[str, Any]:
    """Mode and connected network of ``GET /wifi/status``; saved networks only as a count."""
    if not isinstance(status, dict):
        status = {}
    mode = status.get("mode")
    connected = status.get("connected_network")
    known = status.get("known_networks")
    return {
        "available": True,
        "mode": mode if isinstance(mode, str) else None,
        "connected_network": connected if isinstance(connected, str) else None,
        "known_networks": len(known) if isinstance(known, list) else None,
    }


__all__ = ["sources"]
