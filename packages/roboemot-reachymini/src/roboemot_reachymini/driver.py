"""Reachy Mini PoseGroupDriver on the official SDK (SPEC 4A).

Facts from the SDK source (reachy_mini 1.11) that shape this driver:

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
"""

from __future__ import annotations

from typing import Any

import numpy as np
from roboemot.drivers import MotorMode, PoseGroupDriver
from scipy.spatial.transform import Rotation

HEAD_ANGLES = ("head.roll", "head.pitch", "head.yaw")
HEAD_TRANSLATIONS = ("head.x", "head.y", "head.z")
ANTENNAS = ("antenna_right", "antenna_left")  # SDK order


def head_matrix(values: dict[str, float]) -> np.ndarray:
    pose = np.eye(4)
    rpy = [values.get(k, 0.0) for k in HEAD_ANGLES]
    pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = [values.get(k, 0.0) for k in HEAD_TRANSLATIONS]
    return pose


def head_values(matrix: Any) -> dict[str, float]:
    m = np.asarray(matrix, dtype=float).reshape(4, 4)
    roll, pitch, yaw = Rotation.from_matrix(m[:3, :3]).as_euler("xyz")
    out = dict(zip(HEAD_ANGLES, (float(roll), float(pitch), float(yaw)), strict=True))
    out.update(
        zip(HEAD_TRANSLATIONS, (float(m[0, 3]), float(m[1, 3]), float(m[2, 3])), strict=True)
    )
    return out


class ReachyMiniDriver(PoseGroupDriver):
    """Wraps a connected ``reachy_mini.ReachyMini`` (owned by the bridge / app process)."""

    def __init__(self, mini: Any) -> None:
        self.mini = mini
        self.mode: MotorMode | None = None
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
        if not getattr(self.mini.client, "_is_alive", True):
            raise ConnectionError("Reachy Mini daemon: no state received (connection lost)")
        state = head_values(self.mini.get_current_head_pose())
        head_joints, antennas = self.mini.get_current_joint_positions()
        state["body_yaw"] = float(head_joints[0])  # first head joint is the body rotation
        state.update(zip(ANTENNAS, (float(a) for a in antennas), strict=True))
        return state

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
            self.mini.enable_motors()  # pins targets to the present pose (SDK 1.11)
        elif mode == "limp":
            self.mini.disable_motors()
        elif mode == "teach":
            self.mini.enable_gravity_compensation()
        else:
            raise ValueError(mode)
        self.mode = mode
