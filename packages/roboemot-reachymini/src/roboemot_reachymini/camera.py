"""Camera frames from the Reachy Mini SDK media (SPEC 4A, 5).

The app owns the SDK client and its media (one process, SPEC 4A); this source only reads
from the ``ReachyMini`` instance the daemon passed to ``run``, never creating a client.
Coded against reachy-mini 1.12's documented interface, not exercised against a real robot or
the simulator here: ``mini.media.get_frame()`` returns the newest BGR ``(H, W, 3)`` uint8
frame or ``None``, and ``mini.media.camera.K`` is the 3x3 intrinsic matrix rescaled to the
streamed size. Frames carry no capture timestamp, so the stamp is the monotonic time of the
read, a small overestimate of the capture time (the stream latency is not known).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import numpy as np
from roboemot.vision import CameraModel, Frame

from .sdk import MiniPort

JointStateFn = Callable[[], Mapping[str, float]]


class ReachyCamera:
    """``FrameSource`` over ``reachy_mini.media``; pairs each frame with the joint state."""

    def __init__(
        self,
        mini: MiniPort,
        joint_state: JointStateFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mini = mini
        self._joint_state = joint_state
        self._clock = clock

    def read(self) -> Frame | None:
        """The newest frame stamped now with the current joint state, or None without one.

        Stamp and joint state are taken right after the frame is read, so the joint state can
        differ slightly from the one at capture while the head moves.
        """
        image = self._mini.media.get_frame()
        if image is None:
            return None
        stamp = self._clock()
        state = dict(self._joint_state()) if self._joint_state else None
        return Frame(np.asarray(image), stamp, state)

    def camera_model(self, width: int, height: int) -> CameraModel | None:
        """Pinhole model from the SDK intrinsics ``K`` for a ``width`` x ``height`` frame.

        ``media.camera.K`` is already rescaled to the streamed size, so it is used only when
        its principal point lies inside the frame; otherwise None and vision falls back to
        ``world.camera_fov_h_deg`` and ``camera_fov_v_deg``. Lens distortion is ignored by the
        pinhole model, so bearings are approximate towards the edges of this wide-angle camera.
        """
        camera = self._mini.media.camera
        k = None if camera is None else camera.K
        if k is None:
            return None
        fx, fy, cx, cy = (float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2]))
        if fx <= 0 or fy <= 0 or not (0 < cx < width and 0 < cy < height):
            return None
        return CameraModel.from_intrinsics(width, height, fx, fy, cx, cy)
