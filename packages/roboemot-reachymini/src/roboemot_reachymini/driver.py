"""Reachy Mini PoseGroupDriver on the official SDK (SPEC 4A).

Facts from the SDK source (reachy_mini 1.12.0, the robot's version) that shape this driver:

- ``goto_target`` blocks the caller until the move ends, so it is never used here; every
  motion is streamed with ``set_target``.
- ``automatic_body_yaw`` defaults to True in ``ReachyMini.__init__``; RoboEmot's arbiter
  commands body yaw itself, so the driver turns it off.
- ``enable_motors()`` pins all targets to the present pose; ``set_target`` must come after it.
- Antenna order is ``[right, left]``; head pose is a 4x4 matrix from ``create_head_pose``
  with Euler order ``xyz`` (roll, pitch, yaw).
- The SDK talks to the daemon over one websocket (``mini.client``, a ``WSClient``). When the
  daemon's backend stops, every call raises ``ConnectionError("Lost connection with the
  server.")`` and the client never reconnects by itself, so :meth:`reconnect` swaps in a new
  ``WSClient`` to the same host and port (checked in the MuJoCo simulation, SPEC 12A).
- The state getters return the last cached values even on a dead connection; only
  ``WSClient._is_alive`` (refreshed every second from the state stream) tells, so
  :meth:`read_state` checks it.
- The motor mode is in the daemon status the client receives
  (``backend_status.motor_control_mode``: enabled, disabled, gravity_compensation). The daemon
  publishes that status once a second (``io/ws_server.py``), so a motor change is seen up to
  ``DAEMON_STATUS_PERIOD_S`` late. ``get_status(wait=False)`` clears ``status_received``, which
  the SDK's own waits use, so the driver reads ``_last_status`` directly.
- The state messages (``HeadPoseMsg``, ``JointPositionsMsg``, 50 Hz each) carry no timestamp;
  the receive thread stores each one as a new object (``WSClient._dispatch``). :meth:`state_seq`
  counts those replacements: it tells a new sample from a repeated one, not how old a sample is.
- Checked against the 1.12.0 source (the robot's version, enforced at app start by
  ``sdk.require_sdk``) and run against 1.12.0 in MuJoCo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from roboemot.drivers import MotorMode, PoseGroupDriver

from .sdk import ClientPort

HEAD_ANGLES = ("head.roll", "head.pitch", "head.yaw")
HEAD_TRANSLATIONS = ("head.x", "head.y", "head.z")
ANTENNAS = ("antenna_right", "antenna_left")  # SDK order
# SDK 1.12 facts, to re-check on an SDK bump: the status thread builds the status, then sleeps
# 1 s (daemon/daemon.py:872-882), so its period is a little over this; goto_sleep disables the
# motors after a 2 s asyncio.sleep that follows its last goto (daemon/backend/abstract.py:1370).
DAEMON_STATUS_PERIOD_S = 1.0
SLEEP_MOTORS_OFF_DELAY_S = 2.0


def head_matrix(values: dict[str, float]) -> np.ndarray:
    """4x4 head pose from roll, pitch, yaw (extrinsic ``xyz``, as scipy's ``from_euler`` in the
    SDK's ``create_head_pose``: R = Rz(yaw) Ry(pitch) Rx(roll)) and the translations. Closed form:
    scipy's ``Rotation`` costs tens of microseconds per call on the 50 Hz path."""
    roll, pitch, yaw = (values.get(k, 0.0) for k in HEAD_ANGLES)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    x, y, z = (values.get(k, 0.0) for k in HEAD_TRANSLATIONS)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
            [-sp, cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def head_values(matrix: Any) -> dict[str, float]:
    """Inverse of :func:`head_matrix` (pitch in [-pi/2, pi/2], as scipy's ``as_euler``)."""
    m = np.asarray(matrix, dtype=float).reshape(4, 4).tolist()
    return {
        "head.roll": math.atan2(m[2][1], m[2][2]),
        "head.pitch": math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2])),
        "head.yaw": math.atan2(m[1][0], m[0][0]),
        "head.x": m[0][3],
        "head.y": m[1][3],
        "head.z": m[2][3],
    }


class ReachyMiniDriver(PoseGroupDriver):
    """Wraps a connected ``reachy_mini.ReachyMini`` (owned by the bridge / app process)."""

    status_period_s = DAEMON_STATUS_PERIOD_S

    def __init__(self, mini: Any) -> None:
        self.mini = mini
        self.mode: MotorMode | None = None
        self._seq = 0
        self._seen: tuple[Any, Any] = (None, None)  # the messages counted last (held, not ids)
        mini.set_automatic_body_yaw(False)

    def write_pose_targets(self, t: float, values: dict[str, float]) -> None:
        if self.mode != "active":
            return  # limp or teach: position targets are ignored by the robot anyway
        kwargs: dict[str, Any] = {}
        if any(k in values for k in HEAD_ANGLES + HEAD_TRANSLATIONS):
            kwargs["head"] = head_matrix(values)
        if all(k in values for k in ANTENNAS):
            kwargs["antennas"] = [values[k] for k in ANTENNAS]
        if "body_yaw" in values:
            kwargs["body_yaw"] = float(values["body_yaw"])
        if kwargs:
            self.mini.set_target(**kwargs)

    def read_state(self) -> dict[str, float]:
        client: ClientPort = self.mini.client
        if not client._is_alive:
            raise ConnectionError("Reachy Mini daemon: no state received (connection lost)")
        state = head_values(self.mini.get_current_head_pose())
        head_joints, antennas = self.mini.get_current_joint_positions()
        state["body_yaw"] = float(head_joints[0])  # first head joint is the body rotation
        # The pose's yaw wraps at +-pi, but a head behind the robot is commanded past it (body
        # +-160 deg, head within 65 deg of the body): read it within pi of the body, so command
        # and measure agree there (a 2*pi jump read as a hand on the head, robot run 2026-10-28).
        delta = state["head.yaw"] - state["body_yaw"]
        if abs(delta) > math.pi:
            state["head.yaw"] -= math.copysign(math.tau, delta)
        state.update(zip(ANTENNAS, (float(a) for a in antennas), strict=True))
        return state

    def state_seq(self) -> int:
        """Grows once both state messages (head pose, joints) have been replaced since the last
        count: a sample made of a new head pose and old joints is not new yet."""
        client: ClientPort = self.mini.client
        if not client._is_alive:
            raise ConnectionError("Reachy Mini daemon: no state received (connection lost)")
        head, joints = client._last_head_pose, client._last_joint_positions
        if head is not self._seen[0] and joints is not self._seen[1]:
            self._seq += 1
            self._seen = (head, joints)
        return self._seq

    def motors_enabled(self) -> bool | None:
        """From the last daemon status; None until one has arrived."""
        client: ClientPort = self.mini.client
        status = client._last_status
        backend = None if status is None else status.backend_status
        return None if backend is None else backend.motor_control_mode == "enabled"

    def robot_name(self) -> str | None:
        """The daemon's ``robot_name`` (the owner's name for the robot) from its last status;
        None until one has arrived, or when it is not a non-empty string."""
        # Verified against reachy-mini 1.12: get_status(wait=False) clears status_received and
        # asserts _last_status, hence the guard before the call.
        try:
            client: ClientPort = self.mini.client
            if client._last_status is None:
                return None
            name = client.get_status(wait=False).robot_name
        except Exception:
            return None
        return name.strip() or None if isinstance(name, str) else None

    def reconnect(self, timeout_s: float) -> None:
        """New SDK websocket to the same daemon; body yaw is the arbiter's again afterwards."""
        from reachy_mini.io.ws_client import WSClient

        old = self.mini.client
        client = WSClient(old.host, old.port)
        try:
            client.wait_for_connection(timeout=timeout_s)
        except Exception:
            client.disconnect()
            raise
        old.disconnect()
        self.mini.client = client
        self.mini.set_automatic_body_yaw(False)  # the restarted daemon is back to its default

    def set_mode(self, mode: MotorMode) -> None:
        if mode == "active":
            if self.mode == "teach":
                self.mini.disable_gravity_compensation()
            self.mini.enable_motors()  # pins targets to the present pose (SDK 1.12)
        elif mode == "limp":
            self.mini.disable_motors()
        elif mode == "teach":
            self.mini.enable_gravity_compensation()
        else:
            raise ValueError(mode)
        self.mode = mode
