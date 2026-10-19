"""Reachy Mini section of the web UI settings page (SPEC 4A, 11A, 12).

Registered in the ``roboemot.settings`` entry-point group under ``reachy-mini-wireless``. It
claims the config paths the manifest ``behaviour`` block fills (expression map, gesture map,
circadian motion), so they are edited as ordinary settings but listed under the robot, and adds
the manifest's hardware limits as read-only rows: pose limits, dynamics, yaw-delta constraints and
safety. Angles are radians inside ``Manifest.dofs`` and shown in degrees; translations in mm.

It also implements ``Restartable``: RoboEmot is restarted by the daemon
(``POST /api/apps/restart-current-app``: SIGINT to the app, kill after ``STOP_TIMEOUT_S``, then
start again),
and the action is offered only while ``GET /api/apps/current-app-status`` shows RoboEmot running.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
from roboemot.config.catalogue import FieldDescriptor, descriptor
from roboemot.diagnostics import Target
from roboemot.manifest import Manifest
from roboemot.webui.robot_section import PATH_ROOT, RobotSection
from roboemot.webui.settings import StackPort

from .diagnostics import DEFAULT_PORT, REQUEST_TIMEOUT_S
from .install import APP_NAME

log = logging.getLogger(__name__)

TITLE = "Reachy Mini"
CLAIMS = ("expression_map", "gesture_map", "circadian_motion")
MM_PER_M = 1000.0
SOURCE = "manifest"
WATCHDOG_CEILING = "safety.watchdog_ms"
STATUS_PATH = "/api/apps/current-app-status"
RESTART_PATH = "/api/apps/restart-current-app"
RUNNING = "running"
STOP_TIMEOUT_S = 20  # the daemon's stop_current_app default: SIGINT, then kill
RESTART_MARGIN_S = 10.0  # for the daemon to start the app again after the stop
RESTART_TIMEOUT_S = STOP_TIMEOUT_S + RESTART_MARGIN_S
AVAILABLE_TTL_S = 3.0  # how long a status answer is reused; listings call it on every request
RESTART_EFFECTS = (
    ("zero_pose", {}),
    ("webrtc_drop", {}),
    ("kill_on_timeout", {"timeout_s": STOP_TIMEOUT_S}),
)


def _row(path: str, kind: str, value: Any, description: str, **fields: Any) -> FieldDescriptor:
    """A read-only manifest row: ``source`` manifest, ``default`` equal to the value."""
    return descriptor(
        path, kind, value, default=value, source=SOURCE, description=description, **fields
    )


def _to_mm(metres: float) -> float:
    """Metres to millimetres, for the translation limits."""
    return metres * MM_PER_M


class ReachySettings:
    """The ``roboemot.settings`` plugin of the Reachy Mini. ``transport`` replaces the HTTP
    transport towards the daemon (tests)."""

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        target: Target | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        target = target or Target()
        self._base = f"http://{target.host or 'localhost'}:{target.port or DEFAULT_PORT}"
        self._transport = transport
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: tuple[float, bool] | None = None

    def _client(self, timeout_s: float) -> httpx.Client:
        """A client for the daemon (this machine unless ``target`` says otherwise)."""
        return httpx.Client(base_url=self._base, timeout=timeout_s, transport=self._transport)

    def restart_available(self) -> bool:
        """Whether the daemon reports RoboEmot as the running app. In the simulator or a dev run
        the daemon did not start it, so the daemon could not restart it either. A daemon that
        does not answer counts as not available. The answer is reused for ``AVAILABLE_TTL_S``."""
        with self._lock:
            now = self._clock()
            if self._seen is not None and now - self._seen[0] < AVAILABLE_TTL_S:
                return self._seen[1]
            up = self._running()
            self._seen = (self._clock(), up)
            return up

    def _running(self) -> bool:
        """One ``current-app-status`` call: is RoboEmot the app in state running."""
        try:
            with self._client(REQUEST_TIMEOUT_S) as http:
                resp = http.get(STATUS_PATH)
                resp.raise_for_status()
                status = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.info("restart check: daemon not reachable (%s)", type(exc).__name__)
            return False
        info = status.get("info") if isinstance(status, dict) else None
        name = info.get("name") if isinstance(info, dict) else None
        return name == APP_NAME and status.get("state") == RUNNING

    def restart_effects(self) -> tuple[tuple[str, dict[str, float]], ...]:
        """What the daemon's restart does to the robot and the session."""
        return RESTART_EFFECTS

    def restart(self) -> None:
        """Ask the daemon to restart the app. The daemon stops this very process (SIGINT) before
        it answers, so a dropped connection is the normal outcome and nothing is read from it."""
        try:
            with self._client(RESTART_TIMEOUT_S) as http:
                resp = http.post(RESTART_PATH)
            if resp.is_error:
                log.info("restart request: the daemon answered HTTP %d", resp.status_code)
        except httpx.HTTPError as exc:
            log.info("restart request ended: %s", type(exc).__name__)

    def section(self, stack: StackPort, manifest: Manifest) -> RobotSection:
        """The title, the claimed behaviour paths and the read-only hardware rows of
        ``manifest``; ``stack`` is unused (the hardware does not change at runtime)."""
        base = f"{PATH_ROOT}.{manifest.robot.id}"
        rows = [
            *self._limits(base, manifest),
            *self._dynamics(base, manifest),
            *self._constraints(base, manifest),
            *self._safety(base, manifest),
        ]
        return RobotSection(TITLE, rows, CLAIMS)

    @staticmethod
    def _limits(base: str, manifest: Manifest) -> list[FieldDescriptor]:
        """Min and max of every dof: angles from radians to degrees, translations to mm."""
        rows = []
        for name, dof in manifest.dofs.items():
            if dof.kind == "angle":
                unit, scale = "deg", math.degrees
            else:
                unit, scale = "mm", _to_mm
            for edge, text, value in (
                ("min", "Hardware minimum", dof.min),
                ("max", "Hardware maximum", dof.max),
            ):
                rows.append(
                    _row(
                        f"{base}.limits.{name}.{edge}_{unit}",
                        "number",
                        round(scale(value), 6),
                        f"{text} of {name}; a ceiling no setting can exceed.",
                        unit=unit,
                    )
                )
        return rows

    @staticmethod
    def _dynamics(base: str, manifest: Manifest) -> list[FieldDescriptor]:
        """Velocity and acceleration caps per pose group, as in the manifest (null: not measured,
        the limiter then uses its conservative fallbacks)."""
        units = {
            "max_vel_dps": "deg/s",
            "max_acc_dps2": "deg/s2",
            "max_lin_vel_mps": "m/s",
            "max_lin_acc_mps2": "m/s2",
        }
        rows = []
        for group, dyn in manifest.model.dynamics.items():
            pose = manifest.model.pose_groups.get(group)
            pose_type = pose.type if pose else None
            for key, unit in units.items():
                value = getattr(dyn, key)
                if key.startswith("max_lin") and pose_type != "pose6d":
                    continue
                rows.append(
                    _row(
                        f"{base}.dynamics.{group}.{key}",
                        "number",
                        value,
                        f"Measured {key} of {group}; empty until measured on the robot.",
                        nullable=True,
                        unit=unit,
                    )
                )
        return rows

    @staticmethod
    def _constraints(base: str, manifest: Manifest) -> list[FieldDescriptor]:
        """Yaw-delta limits between two dofs, in degrees."""
        rows = []
        for c in manifest.constraints:
            name = f"{c.a}_{c.b}".replace(".", "_")
            rows.append(
                _row(
                    f"{base}.constraints.{name}.max_deg",
                    "number",
                    c.max_deg,
                    f"Largest allowed yaw difference between {c.a} and {c.b}.",
                    unit="deg",
                )
            )
        return rows

    @staticmethod
    def _safety(base: str, manifest: Manifest) -> list[FieldDescriptor]:
        """Stop behaviour and the watchdog ceiling declared by the manifest."""
        s = manifest.model.safety
        estop = "none" if s.estop == "none" else "configured"
        return [
            _row(f"{base}.safety.estop", "text", estop, "Hardware emergency-stop input."),
            _row(f"{base}.safety.soft_stop", "text", s.soft_stop, "What a soft stop does."),
            _row(f"{base}.safety.hard_stop", "text", s.hard_stop, "What a hard stop does."),
            _row(
                f"{base}.safety.watchdog_ms",
                "number",
                s.watchdog_ms,
                f"Ceiling for {WATCHDOG_CEILING}: the config value can only be lower.",
                unit="ms",
            ),
        ]


plugin = ReachySettings()
