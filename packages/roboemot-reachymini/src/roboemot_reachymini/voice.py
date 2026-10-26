"""The Reachy Mini sound direction for the voice pipeline (SPEC 8 stage 2, 5A attention hint).

``LatestBearing`` keeps the newest base-frame azimuth of the DoA source's observations; the
platform hands it to core's listener as its ``BearingSource``.
"""

from __future__ import annotations

import threading
from collections import deque

from roboemot.bus import Bus, Message
from roboemot.world import FrameSpec, Observation, to_base_frame

from .world import SOUND_SOURCE

HISTORY = 64  # readings kept (about 13 s at the default 5 Hz poll) for a listener that lags


class LatestBearing:
    """The newest sound direction in the base frame, from the DoA source's observations.

    The source publishes sensor-frame bearings on ``percept.entity`` with the joint state at
    capture; this rotates them with the same frame as the world model and keeps the last
    ``HISTORY``, so a listener behind real time gets the reading of its own audio, not a newer one.
    """

    def __init__(self, bus: Bus, frame: FrameSpec) -> None:
        self._frame = frame
        self._lock = threading.Lock()
        self._readings: deque[tuple[float, float]] = deque(maxlen=HISTORY)
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
            self._readings.append((obs.stamp, az))

    def get(self, now: float, max_age_s: float) -> float | None:
        """The base-frame azimuth of the reading nearest to ``now``, if it is at most
        ``max_age_s`` away on either side (a reading after ``now`` is a later sound)."""
        with self._lock:
            readings = list(self._readings)
        if not readings:
            return None
        stamp, az = min(reversed(readings), key=lambda r: abs(now - r[0]))  # ties: newest
        return az if abs(now - stamp) <= max_age_s else None
