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
the robot. ``check`` returns the verdict the wake gate counts: ``AT_FACE``, ``NO_FACE_AT_VOICE``,
``NOT_FACING`` or ``NO_FACING_DATA``; ``FACING_VERDICTS`` maps the facing outcome to it per
path. The snapshot's ``facing.checks`` break down ``checks.passed`` plus
``checks.degraded_passed`` (each utterance that passed here runs the facing check once).
``facing.recent`` keeps the last ``voice.facing.report_checks`` facing checks with the face
used, its yaw and eye span, the sound bearing and the robot's head and body yaw
(``facing_record``, SPEC 12A); recording never changes a verdict.
"""

from __future__ import annotations

import logging
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
from .wake import AT_FACE, NO_FACE_AT_VOICE, NO_FACING_DATA, NOT_FACING

log = logging.getLogger(__name__)

OUTCOMES = ("passed", "rejected", "degraded_passed", "degraded_rejected")
DEGRADED_REASONS = ("no_source", "silent", "muted")
SUMMARY_SIZE = 64  # utterances kept for the angle-difference summary of the debug report
P90 = 0.9
RECORD_FACES = 3  # world entries listed per facing check record, most samples first
RECORD_DIGITS = 1  # decimals of the angles, ages and pixel sizes in a record
HEARD = "heard"  # judged on sound direction readings; else on a degraded reason
# The gate's verdict for each facing outcome, per path (SPEC 8, facing check). Heard at a face,
# a busy CPU leaves the voice-at-face check alone to decide; degraded (muted, no_source, an
# explicit engaged rule on silent) there is no other evidence, so any fallback rejects, as
# degraded_rule reject does. A face with no yaw is not one looking at the robot.
FACING_VERDICTS = {
    HEARD: {
        "passed": AT_FACE,
        "not_facing": NOT_FACING,
        "cpu_high": AT_FACE,
        "no_data": NO_FACING_DATA,
    },
    **{
        reason: {
            "passed": AT_FACE,
            "not_facing": NOT_FACING,
            "cpu_high": NOT_FACING,
            "no_data": NOT_FACING,
        }
        for reason in DEGRADED_REASONS
    },
}


@dataclass(frozen=True)
class FaceRef:
    """Which world entry a face of a sample is, for the facing check records of the debug
    report only (SPEC 12A): its ID and kind, how long ago it was last seen (s) and the eye
    span of its latest landmarks in face-detector pixels (None without one)."""

    entry_id: str
    kind: str
    age_s: float
    eye_px: float | None = None


@dataclass(frozen=True)
class DirectionSample:
    """What the robot hears and sees at one instant: the sound direction (None when no fresh
    reading), the azimuths of the visible faces, whether attention is on one of them, and whether
    the robot had muted its own sound direction (it was speaking). For the facing check: each
    visible face's head yaw (in ``faces_rad`` order) and the attended face's, None when not
    measured recently enough. For its debug records only (never a decision): the faces' world
    entries (``faces_ref``, in ``faces_rad`` order), the attended face's (``engaged_ref``) and its
    azimuth, and the robot's measured head and body yaw (base frame, None when unknown)."""

    bearing_rad: float | None
    faces_rad: tuple[float, ...]
    engaged: bool
    muted: bool = False
    faces_yaw_rad: tuple[float | None, ...] = ()
    engaged_yaw_rad: float | None = None
    faces_ref: tuple[FaceRef, ...] = ()
    engaged_ref: FaceRef | None = None
    engaged_rad: float | None = None
    head_yaw_rad: float | None = None
    body_yaw_rad: float | None = None


EMPTY = DirectionSample(None, (), False)
DirectionLookup = Callable[[float], DirectionSample]


def face_offset(sample: DirectionSample) -> float | None:
    """Angle (radians) from the sound direction to the nearest visible face; None without a
    reading or without a face."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    return min(abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad)


def matched_index(sample: DirectionSample, tolerance_rad: float) -> int | None:
    """Index in ``faces_rad`` of the face nearest the sound direction when it is within
    ``tolerance_rad`` (the face the reading matched); None without a reading, a face or a match."""
    if sample.bearing_rad is None or not sample.faces_rad:
        return None
    offsets = [abs(wrap_angle(sample.bearing_rad - f)) for f in sample.faces_rad]
    nearest = min(range(len(offsets)), key=offsets.__getitem__)
    return nearest if offsets[nearest] <= tolerance_rad else None


def matched_yaw(sample: DirectionSample, tolerance_rad: float) -> float | None:
    """Head yaw of the face nearest the sound direction when it is within ``tolerance_rad``
    (the face the reading matched); None when none matched or its yaw is unknown."""
    nearest = matched_index(sample, tolerance_rad)
    if nearest is None or nearest >= len(sample.faces_yaw_rad):
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
        size = facing.config.report_checks if facing is not None else 1
        self._recent: deque[dict[str, Any]] = deque(maxlen=size)  # facing check records
        self.last: dict[str, Any] = {}  # the last check's numbers (``details``)

    def apply_config(
        self, cfg: VoiceDirectionConfig, facing: VoiceFacingConfig | None = None
    ) -> None:
        """Hot reload: tolerance, shares, ages and the degraded rule; the facing check's
        settings with ``facing`` (a new ``report_checks`` keeps the newest records)."""
        self._cfg = cfg
        if facing is not None and self._facing is not None:
            self._facing.apply_config(facing)
            with self._lock:
                if self._recent.maxlen != facing.report_checks:
                    self._recent = deque(self._recent, maxlen=facing.report_checks)

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
        Counts the outcome; call it once per utterance. ``last`` then holds its numbers."""
        self.last = {}
        verdict = self._check()
        facing = self._facing
        if facing is not None and facing.enabled and "path" in self.last:
            outcome, yaw = facing.last
            self.last = {
                **self.last,
                "facing": outcome,
                "yaw_deg": yaw,
                "limit_deg": facing.config.max_yaw_deg,
                "cpu_percent": facing.cpu_percent,
                "cpu_max_percent": facing.config.cpu_max_percent,
            }
        return verdict

    def _check(self) -> str:
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
            return self._facing_verdict(HEARD, heard, tol)
        reason = self._reason()
        passed = self._rule(reason) == "engaged" and self._engaged()
        self._record("degraded_passed" if passed else "degraded_rejected", None, reason)
        if not passed:
            return NO_FACE_AT_VOICE
        return self._facing_verdict(reason, [s for s in self._samples if s.engaged])

    def _facing_verdict(
        self, path: str, samples: list[DirectionSample], tol: float | None = None
    ) -> str:
        """The facing check over the checked face's yaw in each of ``samples``, when it is on,
        as the gate's verdict for ``path``: on ``HEARD`` the face each reading matched within
        ``tol`` (``matched_yaw``), degraded the face attention is on. Keeps a record of it for
        the debug report (``_remember``)."""
        self.last = {"path": path}
        facing = self._facing
        if facing is None or not facing.enabled:
            return AT_FACE
        if path == HEARD and tol is not None:
            yaws = [matched_yaw(s, tol) for s in samples]
        else:
            yaws = [s.engaged_yaw_rad for s in samples]
        outcome = facing.decide(yaws)
        verdict = FACING_VERDICTS[path][outcome]
        try:
            self._remember(path, outcome, verdict, samples, tol)
        except Exception as exc:  # a record never changes a decision
            log.warning("facing check record failed: %s", type(exc).__name__)
        return verdict

    def _remember(
        self,
        path: str,
        outcome: str,
        verdict: str,
        samples: list[DirectionSample],
        tol: float | None,
    ) -> None:
        """Keep one facing check record (``facing_record``) in the bounded ``recent`` list."""
        record = facing_record(path, outcome, verdict, samples, tol)
        with self._lock:
            self._recent.append(record)

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
            recent = list(self._recent)
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
            snap["facing"] = {**self._facing.snapshot(), "recent": recent}
        return snap


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, RECORD_DIGITS)


def _deg(rad: float | None) -> float | None:
    """Radians to degrees rounded for a record; None stays None."""
    return None if rad is None else _rounded(math.degrees(rad))


def facing_record(
    path: str,
    outcome: str,
    verdict: str,
    samples: list[DirectionSample],
    tol: float | None,
) -> dict[str, Any]:
    """One facing check for the debug report (SPEC 12A), from the samples it judged.

    ``face_source`` is ``matched`` (on ``HEARD``: the face nearest each sound reading within
    ``tol``, ``matched_yaw``) or ``engaged`` (degraded: attention's face, ``engaged_yaw_rad``).
    ``faces`` lists the world entries used, most samples first (at most
    ``RECORD_FACES``), each with its median azimuth, age, signed yaw and eye span in detector
    pixels. ``bearing_deg`` and ``offset_deg`` are the medians of the sound bearing and of its
    angle to the face used; ``head_yaw_deg``, ``body_yaw_deg`` the robot's measured yaw then,
    ``bearing_head_deg`` the bearing relative to the head (a voice behind the robot reads near
    +-180). ``yaw_deg`` is the signed median over ``yaw_samples``. Numbers and entry IDs only.
    """
    used: dict[str, dict[str, list[float]]] = {}
    kinds: dict[str, str] = {}
    yaws: list[float] = []
    offsets: list[float] = []
    for s in samples:
        if path == HEARD and tol is not None:
            i = matched_index(s, tol)
            ref = s.faces_ref[i] if i is not None and i < len(s.faces_ref) else None
            az = s.faces_rad[i] if i is not None else None
            yaw = s.faces_yaw_rad[i] if i is not None and i < len(s.faces_yaw_rad) else None
        else:
            ref, az, yaw = s.engaged_ref, s.engaged_rad, s.engaged_yaw_rad
        if yaw is not None:
            yaws.append(yaw)
        if az is not None and s.bearing_rad is not None:
            offsets.append(abs(wrap_angle(s.bearing_rad - az)))
        if ref is None:
            continue
        kinds[ref.entry_id] = ref.kind
        face = used.setdefault(ref.entry_id, {"az": [], "age": [], "yaw": [], "eye": []})
        face["age"].append(ref.age_s)
        if az is not None:
            face["az"].append(az)
        if yaw is not None:
            face["yaw"].append(yaw)
        if ref.eye_px is not None:
            face["eye"].append(ref.eye_px)
    ranked = sorted(used.items(), key=lambda kv: -len(kv[1]["age"]))[:RECORD_FACES]
    bearings = [s.bearing_rad for s in samples if s.bearing_rad is not None]
    heads = [s.head_yaw_rad for s in samples if s.head_yaw_rad is not None]
    bodies = [s.body_yaw_rad for s in samples if s.body_yaw_rad is not None]
    relative = [
        wrap_angle(s.bearing_rad - s.head_yaw_rad)
        for s in samples
        if s.bearing_rad is not None and s.head_yaw_rad is not None
    ]
    return {
        "path": path,
        "face_source": "matched" if path == HEARD else "engaged",
        "outcome": outcome,
        "verdict": verdict,
        "samples": len(samples),
        "yaw_samples": len(yaws),
        "yaw_deg": _deg(_median(yaws)),
        "faces": [
            {
                "entry_id": eid,
                "kind": kinds[eid],
                "samples": len(f["age"]),
                "azimuth_deg": _deg(_median(f["az"])),
                "age_s": round(statistics.median(f["age"]), RECORD_DIGITS),
                "yaw_deg": _deg(_median(f["yaw"])),
                "eye_px": _rounded(_median(f["eye"])),
            }
            for eid, f in ranked
        ],
        "bearing_deg": _deg(_median(bearings)),
        "offset_deg": _deg(_median(offsets)),
        "head_yaw_deg": _deg(_median(heads)),
        "body_yaw_deg": _deg(_median(bodies)),
        "bearing_head_deg": _deg(_median(relative)),
    }
