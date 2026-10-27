"""Reachy Mini section of the web UI settings page (SPEC 4A, 11A, 12).

Registered in the ``roboemot.settings`` entry-point group under ``reachy-mini-wireless``. It
claims the config paths the manifest ``behaviour`` block fills (expression map, gesture map,
circadian motion), so they are edited as ordinary settings but listed under the robot, and adds
the manifest's hardware limits as read-only rows: pose limits, dynamics, yaw-delta constraints and
safety. Angles are radians inside ``Manifest.dofs`` and shown in degrees; translations in mm.
With a stack it also shows what the realtime session provider (``realtime.py``) uses, read from
the configuration: ``realtime_session.allocator_host`` (the route's host in platform mode, else
empty) and ``realtime_session.sends_platform_id`` (whether the platform identifier goes with each
session request). The rows are leaves below ``realtime_session``: a text field named after a
redacted word (``session``) would show as a secret.

It also implements ``Restartable``: RoboEmot is restarted by the daemon
(``DaemonClient.restart_current_app``: SIGINT to the app, kill after ``STOP_TIMEOUT_S``, then
start again), and the action is offered only while the daemon shows RoboEmot running.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

from roboemot.config.catalogue import FieldDescriptor, descriptor
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import Manifest
from roboemot.realtime.session import route_origin
from roboemot.webui.robot_section import PATH_ROOT, RobotSection
from roboemot.webui.settings import StackPort

from .daemon import STOP_TIMEOUT_S, DaemonClient, DaemonError

log = logging.getLogger(__name__)

TITLE = "Reachy Mini"
CLAIMS = ("expression_map", "gesture_map", "circadian_motion")
MM_PER_M = 1000.0
SOURCE = "manifest"
CONFIG_SOURCE = "config"
PLATFORM_MODE = "platform"
WATCHDOG_CEILING = "safety.watchdog_ms"
AVAILABLE_TTL_S = 3.0  # how long a status answer is reused; listings call it on every request
RESTART_EFFECTS = (
    ("zero_pose", {}),
    ("remote_session_drop", {}),
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
    """The ``roboemot.settings`` plugin of the Reachy Mini. ``daemon`` is the client for the
    daemon (this machine unless it says otherwise; tests pass one with a mock transport)."""

    def __init__(
        self,
        daemon: DaemonClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._daemon = daemon or DaemonClient()
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: tuple[float, bool] | None = None

    def restart_available(self) -> bool:
        """Whether the daemon reports RoboEmot as the running app. In the simulator or a dev run
        the daemon did not start it, so the daemon could not restart it either. A daemon that
        does not answer counts as not available. The answer is reused for ``AVAILABLE_TTL_S``."""
        with self._lock:
            now = self._clock()
            if self._seen is not None and now - self._seen[0] < AVAILABLE_TTL_S:
                return self._seen[1]
            up = self._daemon.app_running()
            self._seen = (self._clock(), up)
            return up

    def restart_effects(self) -> tuple[tuple[str, dict[str, float]], ...]:
        """What the daemon's restart does to the robot and the session."""
        return RESTART_EFFECTS

    def stop_timeout_s(self) -> float:
        """The daemon's grace between SIGINT and kill, which bounds the stop sequence."""
        return STOP_TIMEOUT_S

    def restart(self) -> None:
        """Ask the daemon to restart the app. The daemon stops this very process (SIGINT) before
        it answers, so a dropped connection is the normal outcome and nothing is read from it."""
        try:
            self._daemon.restart_current_app()
        except DaemonError as exc:
            cause = type(exc.__cause__ or exc).__name__
            log.info(
                "restart request ended (%s%s)",
                cause,
                "" if exc.status is None else f", HTTP {exc.status}",
            )

    def section(self, stack: StackPort, manifest: Manifest) -> RobotSection:
        """The title, the claimed behaviour paths and the read-only hardware rows of
        ``manifest``, plus the realtime session rows read from ``stack`` when there is one."""
        base = f"{PATH_ROOT}.{manifest.robot.id}"
        rows = [
            *self._limits(base, manifest),
            *self._dynamics(base, manifest),
            *self._constraints(base, manifest),
            *self._safety(base, manifest),
        ]
        if stack is not None:
            rows += self._realtime_session(base, stack.effective())
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
    def _realtime_session(base: str, cfg: RoboEmotConfig) -> list[FieldDescriptor]:
        """The allocator host and whether the identifier is sent, in platform mode (SPEC 12,
        Data); in direct mode the provider is not used: no host, nothing sent."""
        rt = cfg.voice.realtime_route
        platform = rt.mode == PLATFORM_MODE
        route = next((r for r in cfg.llm_routes if r.name == rt.route), None) if platform else None
        origin = route_origin(route) if route is not None else None
        prefix = f"{base}.realtime_session"
        return [
            descriptor(
                f"{prefix}.allocator_host",
                "text",
                origin[1] if origin else None,
                nullable=True,
                source=CONFIG_SOURCE,
                description="Host of the realtime route's session allocator (platform mode).",
            ),
            descriptor(
                f"{prefix}.sends_platform_id",
                "bool",
                platform and cfg.privacy.realtime_platform_id,
                source=CONFIG_SOURCE,
                description="Whether the robot's platform identifier goes with each session "
                "request (privacy.realtime_platform_id, platform mode).",
            ),
        ]

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
