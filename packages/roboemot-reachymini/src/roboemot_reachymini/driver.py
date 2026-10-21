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
  (``backend_status.motor_control_mode``: enabled, disabled, gravity_compensation).
- Platform moves (the owner's go-to-sleep or wake-up from the mobile app, the daemon's idle
  reset) run in ``Backend.play_move`` and raise ``Backend._active_move_depth``
  (``daemon/backend/abstract.py:485-500``, ``is_move_running``); while it is above 0 the daemon
  drops every ``set_target``/``set_full_target`` from every transport (``abstract.py:1662``) and
  accepts them again as soon as it ends. That flag is in the WebRTC ``StateSnapshot``
  (``io/protocol.py:145``) but not in anything the SDK client (``/ws/sdk``) receives, and
  ``GET /api/move/running`` and ``WS /api/move/ws/updates`` list only the moves the REST router
  created itself (``daemon/app/routers/move.py``, ``move_tasks``): a go-to-sleep sent over the
  WebRTC data channel (``GotoSleepCmd``, ``abstract.py:1714``) or run by the idle reset is in
  neither. The one REST route that reads the flag itself is ``POST /api/move/set_target``
  (``move.py:219-225``), which answers ``move_running`` and, with an empty target, sets
  nothing: ``DaemonClient.move_running`` sends that. A background thread polls it every
  ``MOVE_POLL_S`` so ``platform_move_running`` is a cached read for the 50 Hz loop; an answer older
  than ``MOVE_STALE_S``, or a failed probe, reads as None (unknown). Not verified on a real robot
  with a control-only session; the MuJoCo daemon answers it as the real one.
- Checked against the 1.12.0 source (the robot's version, enforced at app start by
  ``sdk.require_sdk``) and run against 1.12.0 in MuJoCo.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from roboemot import shutdown
from roboemot.drivers import MotorMode, PoseGroupDriver
from scipy.spatial.transform import Rotation

from .sdk import ClientPort

HEAD_ANGLES = ("head.roll", "head.pitch", "head.yaw")
HEAD_TRANSLATIONS = ("head.x", "head.y", "head.z")
ANTENNAS = ("antenna_right", "antenna_left")  # SDK order
MOVE_POLL_S = 0.2  # between probes of the daemon's move flag
MOVE_STALE_S = 1.0  # a probe answer older than this is unknown
MOVE_REQUEST_TIMEOUT_S = 0.5  # one probe's HTTP timeout: the poller joins within the budget
MOVE_JOIN_S = 2.0

log = logging.getLogger(__name__)


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

    def __init__(
        self,
        mini: Any,
        move_probe: Callable[[], bool] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = MOVE_POLL_S,
    ) -> None:
        self.mini = mini
        self.mode: MotorMode | None = None
        self._move_probe, self._clock, self._poll_s = move_probe, clock, poll_s
        self._move: tuple[bool, float] | None = None  # (running, when probed)
        self._move_stop = threading.Event()
        self._move_thread: threading.Thread | None = None
        self._move_error: str | None = None
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
        state.update(zip(ANTENNAS, (float(a) for a in antennas), strict=True))
        return state

    def motors_enabled(self) -> bool | None:
        """From the last daemon status; None until one has arrived."""
        client: ClientPort = self.mini.client
        if client._last_status is None:
            return None
        backend = client.get_status(wait=False).backend_status
        return None if backend is None else backend.motor_control_mode == "enabled"

    def platform_move_running(self) -> bool | None:
        """The daemon's move flag from the last probe (see the module docstring); None without a
        probe, before the first answer, when the last probe failed or when it is stale. The first
        call starts the poller; never blocks."""
        if self._move_probe is None:
            return None
        if self._move_thread is None:
            self._start_move_poller()
        move = self._move
        if move is None or self._clock() - move[1] > MOVE_STALE_S:
            return None
        return move[0]

    def _start_move_poller(self) -> None:
        self._move_stop.clear()
        thread = threading.Thread(target=self._poll_move, name="reachy-move-flag", daemon=True)
        thread.start()
        self._move_thread = thread

    def _poll_move(self) -> None:
        """Probe the move flag until closed; a failure is logged once per type and reads as
        unknown."""
        assert self._move_probe is not None
        while not self._move_stop.is_set():
            try:
                self._move = (self._move_probe(), self._clock())
                self._move_error = None
            except Exception as exc:
                self._move = None
                name = type(exc).__name__
                if name != self._move_error:
                    self._move_error = name
                    log.warning("platform move probe: %s", name)
            self._move_stop.wait(self._poll_s)

    def close(self) -> None:
        """Stop the move poller (joined within ``MOVE_JOIN_S``)."""
        self._move_stop.set()
        if self._move_thread is not None:
            shutdown.join(self._move_thread, MOVE_JOIN_S)
            self._move_thread = None

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
