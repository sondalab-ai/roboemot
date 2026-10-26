"""The two audio levels the Live tab shows, in dBFS (SPEC 11A, 12). Levels only: no sample leaves
the audio path and none is kept.

Microphone: the hub already computes one RMS per chunk for the reflex engine; ``offer`` is that
sink with one tuple store added (no lock, no allocation beyond the tuple), then it forwards to the
reflex engine unchanged. So the level exists only while ``reflexes.audio`` feeds the hub; without it
the hub computes no RMS and the page is told ``unavailable`` (starting extra work on the microphone
thread for a display was ruled out). The noise floor is the reflex engine's own running floor.

Robot output: the speaker's playback reference (the barge-in echo reference) is read when asked,
over the last ``webui.live.level_window_ms``: zero cost on the playback path, a short lock and one
RMS per poll.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

AMPLITUDE_DB_PER_DECADE = 20.0
MS_PER_S = 1000.0
SILENCE_DBFS = -120.0  # reported for digital silence, so the number stays finite

OK, PAUSED, NO_DATA, UNAVAILABLE = "ok", "paused", "no_data", "unavailable"


def dbfs(rms: float) -> float:
    """RMS (full scale 1.0) in dBFS, never below ``SILENCE_DBFS``."""
    if not (math.isfinite(rms) and rms > 0):
        return SILENCE_DBFS
    return max(SILENCE_DBFS, AMPLITUDE_DB_PER_DECADE * math.log10(rms))


class AudioLevels:
    """``offer`` is installed as the hub's level sink; ``snapshot`` builds the page's block."""

    def __init__(
        self,
        forward: Callable[[float, float], None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._forward = forward
        self._clock = clock
        self._mic: tuple[float, float] | None = None  # (rms, stamp) of the newest chunk
        self.reference: Any = None  # PlaybackReference of the speaker, when there is one
        self.noise_floor: Callable[[], float | None] = lambda: None

    @property
    def mic_fed(self) -> bool:
        """Whether the hub was given this sink (it computes no RMS otherwise)."""
        return self._forward is not None

    def offer(self, rms: float, stamp: float) -> None:
        """Hub thread, once per chunk: remember the level, then the reflex engine's own call."""
        self._mic = (rms, stamp)
        if self._forward is not None:
            self._forward(rms, stamp)

    def snapshot(self, *, paused: bool, max_age_s: float, window_ms: int) -> dict[str, Any]:
        """``mic`` and ``out`` as ``{state, db}`` (``db`` None unless ``ok``), and ``floor_db``
        (the reflex noise floor, None when unknown). ``paused``: the robot sleeps or the listener
        is paused, so nothing is read."""
        if paused:
            return {
                "mic": {"state": PAUSED, "db": None},
                "out": {"state": PAUSED, "db": None},
                "floor_db": None,
            }
        return {"mic": self._mic_block(max_age_s), "out": self._out_block(window_ms),
                "floor_db": self._floor_db()}  # fmt: skip

    def _mic_block(self, max_age_s: float) -> dict[str, Any]:
        if not self.mic_fed:
            return {"state": UNAVAILABLE, "db": None}
        mic = self._mic
        if mic is None or self._clock() - mic[1] > max_age_s:
            return {"state": NO_DATA, "db": None}
        return {"state": OK, "db": round(dbfs(mic[0]), 1)}

    def _out_block(self, window_ms: int) -> dict[str, Any]:
        reference = self.reference
        if reference is None:
            return {"state": UNAVAILABLE, "db": None}
        now = self._clock()
        piece = reference.window(now - window_ms / MS_PER_S, now)
        if piece.size == 0:
            return {"state": OK, "db": SILENCE_DBFS}
        rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64))))
        return {"state": OK, "db": round(dbfs(rms), 1)}

    def _floor_db(self) -> float | None:
        floor = self.noise_floor()
        return None if floor is None else round(dbfs(floor), 1)
