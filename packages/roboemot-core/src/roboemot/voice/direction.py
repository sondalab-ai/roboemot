"""Voice-at-face check: is the voice coming from a visible face? (SPEC 8 stage 2, 5A)

Music and television are heard like a person. Without the wake word an utterance is addressed
only when, during it, the robot's sound direction pointed at a visible face. The listener calls
``begin`` on speech start and ``sample`` on each piece of utterance audio; the wake gate calls
``check`` only for an utterance it would accept without the wake word, so a wake word (heard in
the free text or rescued by the grammar pass) never reaches it.

Each sample holds the sound direction and the visible faces' azimuths, both in the robot base
frame: the platform's ``BearingSource`` and the world model rotate them with the joint state at
capture, so the head turn and the microphone array geometry are the platform's business. A
reading matches when it is within ``voice.direction.tolerance_deg`` of the nearest face; the
check passes when at least ``min_match_share`` of the readings match. When fewer than
``min_doa_share`` of the samples have a reading (no direction on this robot, or stale), the
check runs degraded on ``degraded_rule`` and is counted as such.
"""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config.schema import VoiceDirectionConfig
from ..world.frame import wrap_angle

OUTCOMES = ("passed", "rejected", "degraded_passed", "degraded_rejected")
SUMMARY_SIZE = 64  # utterances kept for the angle-difference summary of the debug report


@dataclass(frozen=True)
class DirectionSample:
    """What the robot hears and sees at one instant: the sound direction (None when no fresh
    reading), the azimuths of the visible faces, and whether attention is on one of them."""

    bearing_rad: float | None
    faces_rad: tuple[float, ...]
    engaged: bool


EMPTY = DirectionSample(None, (), False)
DirectionLookup = Callable[[float], DirectionSample]


def face_offset(sample: DirectionSample) -> float | None:
    """Angle (radians) from the sound direction to the nearest visible face; None without a
    reading or without a face."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    return min(abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad)


class VoiceDirection:
    """See the module docstring. ``begin``/``sample``/``check`` run on the listener thread;
    ``snapshot`` from any thread."""

    def __init__(self, cfg: VoiceDirectionConfig, lookup: DirectionLookup) -> None:
        self._cfg = cfg
        self._lookup = lookup
        self._samples: list[DirectionSample] = []
        self._lock = threading.Lock()
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._no_face = 0
        self._offsets_deg: deque[float] = deque(maxlen=SUMMARY_SIZE)

    def apply_config(self, cfg: VoiceDirectionConfig) -> None:
        """Hot reload: tolerance, shares, ages and the degraded rule."""
        self._cfg = cfg

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def begin(self) -> None:
        """A new utterance: forget the previous one's samples."""
        self._samples = []

    def sample(self, stamp: float) -> None:
        """Record the direction and the faces at ``stamp`` (a lookup that raises counts as
        nothing heard and nothing seen)."""
        try:
            self._samples.append(self._lookup(stamp))
        except Exception:
            self._samples.append(EMPTY)

    def check(self) -> bool:
        """Whether the utterance's voice came from a visible face; counts the outcome."""
        cfg = self._cfg
        samples = self._samples
        heard = [s for s in samples if s.bearing_rad is not None]
        if heard and len(heard) >= cfg.min_doa_share * len(samples):
            offsets = [face_offset(s) for s in heard]
            tol = math.radians(cfg.tolerance_deg)
            matched = sum(1 for d in offsets if d is not None and d <= tol)
            passed = matched >= cfg.min_match_share * len(heard)
            seen = [d for d in offsets if d is not None]
            self._record("passed" if passed else "rejected", seen)
            return passed
        if cfg.degraded_rule == "engaged" and samples:
            engaged = sum(1 for s in samples if s.engaged and s.faces_rad)
            passed = engaged >= cfg.min_match_share * len(samples)
        else:
            passed = False
        self._record("degraded_passed" if passed else "degraded_rejected", None)
        return passed

    def _record(self, outcome: str, offsets: list[float] | None) -> None:
        with self._lock:
            self._outcomes[outcome] += 1
            if offsets is None:
                return
            if offsets:
                self._offsets_deg.append(math.degrees(statistics.median(offsets)))
            else:
                self._no_face += 1

    def snapshot(self) -> dict[str, Any]:
        """Check counts, degraded count and the summary of each checked utterance's median angle
        from the voice to the nearest face (degrees): numbers only."""
        with self._lock:
            offsets = list(self._offsets_deg)
            outcomes = dict(self._outcomes)
            no_face = self._no_face
        summary: dict[str, Any] = {"count": len(offsets)}
        if offsets:
            summary |= {
                "min": round(min(offsets), 1),
                "median": round(statistics.median(offsets), 1),
                "max": round(max(offsets), 1),
                "last": round(offsets[-1], 1),
            }
        return {
            "enabled": self._cfg.enabled,
            "checks": outcomes,
            "degraded": outcomes["degraded_passed"] + outcomes["degraded_rejected"],
            "no_face": no_face,
            "face_offset_deg": summary,
        }
