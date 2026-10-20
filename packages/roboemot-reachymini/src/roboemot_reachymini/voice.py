"""The Reachy Mini sound direction for the voice pipeline (SPEC 8 stage 2, 5A attention hint).

``LatestBearing`` keeps the newest base-frame azimuth of the DoA source's observations; the
platform hands it to core's listener as its ``BearingSource``.
"""

from __future__ import annotations

import threading

from roboemot.bus import Bus, Message
from roboemot.world import FrameSpec, Observation, to_base_frame

from .world import SOUND_SOURCE


class LatestBearing:
    """The newest sound direction in the base frame, from the DoA source's observations.

    The source publishes sensor-frame bearings on ``percept.entity`` with the joint state at
    capture; this rotates them with the same frame as the world model and keeps the latest.
    """

    def __init__(self, bus: Bus, frame: FrameSpec) -> None:
        self._frame = frame
        self._lock = threading.Lock()
        self._latest: tuple[float, float] | None = None
        self._unsubscribe = bus.subscribe("percept.entity", self._on_entity)

    def close(self) -> None:
        """Stop listening."""
        self._unsubscribe()

    def _on_entity(self, msg: Message) -> None:
        obs = msg.payload
        if not isinstance(obs, Observation) or obs.source != SOUND_SOURCE:
            return
        az, _ = to_base_frame(
            obs.azimuth_rad, obs.elevation_rad, msg.meta.get("joint_state"), self._frame
        )
        with self._lock:
            self._latest = (obs.stamp, az)

    def get(self, now: float, max_age_s: float) -> float | None:
        """The base-frame azimuth if the newest reading is at most ``max_age_s`` old."""
        with self._lock:
            latest = self._latest
        return latest[1] if latest is not None and now - latest[0] <= max_age_s else None
