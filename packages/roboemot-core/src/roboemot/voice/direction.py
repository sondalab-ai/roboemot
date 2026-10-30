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
check passes when at least ``min_match_share`` of the readings match. Every reading counts,
however few: a sparse reading off every face is evidence, not absence.

With fewer than ``min_doa_readings`` readings the check runs degraded, for one of three reasons:
``no_source`` (the robot has no sound direction), ``muted`` (the robot muted it itself while it
spoke: its own voice and echo tail, as in a barge-in) or ``silent`` (it has one and gave none:
its detector did not take the sound for speech). ``degraded_rule`` decides; ``auto`` rejects
``silent`` and falls back to an engaged face for the other two.

An utterance that passes then runs the facing check (``roboemot.voice.facing``, when on): the
face matched at the voice in each reading, or, degraded, the face attention is on, must look at
the robot. ``check`` returns the verdict the wake gate counts: ``AT_FACE``, ``NO_FACE_AT_VOICE``
or ``NOT_FACING``.
"""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..config.schema import VoiceDirectionConfig, VoiceFacingConfig
from ..world.frame import wrap_angle
from .facing import FacingCheck
from .wake import AT_FACE, NO_FACE_AT_VOICE, NOT_FACING

OUTCOMES = ("passed", "rejected", "degraded_passed", "degraded_rejected")
DEGRADED_REASONS = ("no_source", "silent", "muted")
SUMMARY_SIZE = 64  # utterances kept for the angle-difference summary of the debug report
P90 = 0.9


@dataclass(frozen=True)
class DirectionSample:
    """What the robot hears and sees at one instant: the sound direction (None when no fresh
    reading), the azimuths of the visible faces, whether attention is on one of them, and whether
    the robot had muted its own sound direction (it was speaking). For the facing check: each
    visible face's head yaw (in ``faces_rad`` order) and the attended face's, None when not
    measured recently enough."""

    bearing_rad: float | None
    faces_rad: tuple[float, ...]
    engaged: bool
    muted: bool = False
    faces_yaw_rad: tuple[float | None, ...] = ()
    engaged_yaw_rad: float | None = None


EMPTY = DirectionSample(None, (), False)
DirectionLookup = Callable[[float], DirectionSample]


def face_offset(sample: DirectionSample) -> float | None:
    """Angle (radians) from the sound direction to the nearest visible face; None without a
    reading or without a face."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    return min(abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad)


def matched_yaw(sample: DirectionSample, tolerance_rad: float) -> float | None:
    """Head yaw of the face nearest the sound direction when it is within ``tolerance_rad``
    (the face the reading matched); None when none matched or its yaw is unknown."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    offsets = [abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad]
    nearest = min(range(len(offsets)), key=offsets.__getitem__)
    if offsets[nearest] > tolerance_rad or nearest >= len(sample.faces_yaw_rad):
        return None
    return sample.faces_yaw_rad[nearest]


class VoiceDirection:
    """See the module docstring. ``doa_available``: the robot declares a sound direction (its
    manifest's ``doa`` capability). ``begin``/``sample``/``check`` run on the listener thread;
    ``snapshot`` from any thread. ``_lock`` is a leaf lock: ``check`` takes it inside the wake
    gate's lock, and nothing else is called while it is held."""

    def __init__(
        self,
        cfg: VoiceDirectionConfig,
        lookup: DirectionLookup,
        *,
        doa_available: bool,
        facing: FacingCheck | None = None,
    ) -> None:
        self._cfg = cfg
        self._lookup = lookup
        self._doa = doa_available
        self._facing = facing
        self._samples: list[DirectionSample] = []
        self._lock = threading.Lock()
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._reasons = dict.fromkeys(DEGRADED_REASONS, 0)
        self._no_face = 0
        self._offsets_deg: deque[float] = deque(maxlen=SUMMARY_SIZE)

    def apply_config(
        self, cfg: VoiceDirectionConfig, facing: VoiceFacingConfig | None = None
    ) -> None:
        """Hot reload: tolerance, shares, ages and the degraded rule; the facing check's
        settings with ``facing``."""
        self._cfg = cfg
        if facing is not None and self._facing is not None:
            self._facing.apply_config(facing)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def begin(self) -> None:
        """A new utterance: forget the previous one's samples."""
        self._samples = []

    def sample(self, stamp: float, *, muted: bool = False) -> None:
        """Record the direction and the faces at ``stamp``; ``muted`` when the robot was speaking
        then (its sound direction is off). A lookup that raises counts as nothing heard and
        nothing seen."""
        try:
            found = self._lookup(stamp)
        except Exception:
            found = EMPTY
        self._samples.append(replace(found, muted=True) if muted else found)
        if self._facing is not None and self._facing.enabled:
            self._facing.poll()

    def check(self) -> str:
        """Whether the utterance's voice came from a visible face (and, with the facing check
        on, that face looked at the robot): ``AT_FACE``, ``NO_FACE_AT_VOICE`` or ``NOT_FACING``.
        Counts the outcome; call it once per utterance."""
        cfg = self._cfg
        heard = [s for s in self._samples if s.bearing_rad is not None]
        if len(heard) >= cfg.min_doa_readings:
            offsets = [face_offset(s) for s in heard]
            tol = math.radians(cfg.tolerance_deg)
            matched = sum(1 for d in offsets if d is not None and d <= tol)
            passed = matched >= cfg.min_match_share * len(heard)
            self._record("passed" if passed else "rejected", [d for d in offsets if d is not None])
            if not passed:
                return NO_FACE_AT_VOICE
            return self._facing_verdict([matched_yaw(s, tol) for s in heard])
        reason = self._reason()
        passed = self._rule(reason) == "engaged" and self._engaged()
        self._record("degraded_passed" if passed else "degraded_rejected", None, reason)
        if not passed:
            return NO_FACE_AT_VOICE
        return self._facing_verdict([s.engaged_yaw_rad for s in self._samples if s.engaged])

    def _facing_verdict(self, yaws: list[float | None]) -> str:
        """The facing check over the checked face's yaw per sample, when it is on."""
        facing = self._facing
        if facing is None or not facing.enabled:
            return AT_FACE
        return AT_FACE if facing.decide(yaws) else NOT_FACING

    def _reason(self) -> str:
        """Why the utterance has too few readings (see the module docstring)."""
        if not self._doa:
            return "no_source"
        return "muted" if any(s.muted for s in self._samples) else "silent"

    def _rule(self, reason: str) -> str:
        """``degraded_rule`` with ``auto`` resolved for ``reason``."""
        rule = self._cfg.degraded_rule
        if rule == "auto":
            return "reject" if reason == "silent" else "engaged"
        return rule

    def _engaged(self) -> bool:
        """Attention was on a visible person for ``degraded_engaged_share`` of the samples."""
        samples = self._samples
        engaged = sum(1 for s in samples if s.engaged)
        return bool(samples) and engaged >= self._cfg.degraded_engaged_share * len(samples)

    def _record(self, outcome: str, offsets: list[float] | None, reason: str = "") -> None:
        with self._lock:
            self._outcomes[outcome] += 1
            if reason:
                self._reasons[reason] += 1
            if offsets is None:
                return
            if offsets:
                self._offsets_deg.append(math.degrees(statistics.median(offsets)))
            else:
                self._no_face += 1

    def snapshot(self) -> dict[str, Any]:
        """Check counts, degraded counts by reason and the summary of each checked utterance's
        median angle from the voice to the nearest face (degrees): numbers only."""
        with self._lock:
            offsets = list(self._offsets_deg)
            outcomes = dict(self._outcomes)
            reasons = dict(self._reasons)
            no_face = self._no_face
        summary: dict[str, Any] = {"count": len(offsets)}
        if offsets:
            ordered = sorted(offsets)
            summary |= {
                "min": round(ordered[0], 1),
                "median": round(statistics.median(ordered), 1),
                "p90": round(ordered[min(len(ordered) - 1, int(P90 * len(ordered)))], 1),
                "max": round(ordered[-1], 1),
                "last": round(offsets[-1], 1),
            }
        snap = {
            "enabled": self._cfg.enabled,
            "doa_available": self._doa,
            "checks": outcomes,
            "degraded": reasons,
            "no_face": no_face,
            "face_offset_deg": summary,
        }
        if self._facing is not None:
            snap["facing"] = self._facing.snapshot()
        return snap
