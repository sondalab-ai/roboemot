"""Face enrollment from the live vision pipeline (SPEC 6, Enrollment; SPEC 12 Data).

The web UI (or any caller) drives one session at a time: ``start`` begins collecting,
``status`` reports progress, ``complete`` enrolls with a consent record, ``cancel`` discards.
While a session runs, the pipeline calls ``on_frame`` for every processed frame. The service
picks the largest, most central face track and takes a sample only from a face that is fully
inside the frame, large enough, confidently detected and sharp, at least ``min_gap_s`` after the
previous sample. It embeds the face itself on the vision thread (so the ONNX session stays
single-threaded). Samples must resemble each other: the first one is only trusted once a second
one agrees with it, so two people never mix in one session. SPEC 6 asks for varied angles: the
session is not ready until the nose offset from the eye midline (a yaw proxy from the
landmarks) spans ``min_pose_spread``, or the per-person cap is reached.

Every quality threshold (``enrollment.*``) is untuned: set from reasoning, not yet measured on
real camera frames.

Privacy (SPEC 12): face crops are used for the sharpness check and dropped at once; embeddings
stay in this object's memory until ``complete`` hands them to ``Registry.enroll`` and are
zeroed on completion, cancellation or timeout. Zeroing is best effort: NumPy or the registry may
hold copies that are only freed by the garbage collector. Neither embeddings nor names are
logged.

Not exported from ``roboemot.registry``: it needs the vision types, and ``roboemot.vision``
imports the registry. Import it as ``roboemot.registry.enrollment``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config.schema import EnrollmentConfig, IdentityConfig
from .records import Consent, RecordSummary
from .registry import Registry

if TYPE_CHECKING:
    from ..vision.pipeline import VisionPipeline
    from ..vision.stages import FaceEmbedder
    from ..vision.tracker import Track
    from ..vision.types import Frame

log = logging.getLogger(__name__)

IDLE = "idle"
COLLECTING = "collecting"
READY = "ready"
COMPLETING = "completing"
TIMED_OUT = "timed_out"
COMPLETED = "completed"
CANCELLED = "cancelled"
UNAVAILABLE = "unavailable"
ACTIVE_STATES = (COLLECTING, READY)
REJECT_REASONS = ("score", "small", "cropped", "blurred", "inconsistent")
LUMA = (0.114, 0.587, 0.299)  # BGR weights of the Rec. 601 luma
EPSILON = 1e-9
LAPLACIAN_CENTRE = -4.0
RIGHT_EYE, LEFT_EYE, NOSE = 0, 1, 2  # YuNet landmark order: eyes, nose tip, mouth corners
LANDMARK_SHAPE = (5, 2)


class EnrollmentError(RuntimeError):
    """The request does not fit the session state (busy, nothing to complete, no vision)."""


def _area_resize(gray: np.ndarray, size: int) -> np.ndarray:
    """Mean-pool ``gray`` to ``size`` x ``size`` (nearest sampling when it is smaller)."""
    rows, cols = gray.shape
    if rows < size or cols < size:
        picked = gray[np.linspace(0, rows - 1, size).astype(int)]
        return picked[:, np.linspace(0, cols - 1, size).astype(int)]
    ys = np.linspace(0, rows, size + 1).astype(int)
    xs = np.linspace(0, cols, size + 1).astype(int)
    pooled = np.add.reduceat(gray, ys[:-1], axis=0) / np.diff(ys)[:, None]
    return np.add.reduceat(pooled, xs[:-1], axis=1) / np.diff(xs)[None, :]


def sharpness(crop: np.ndarray, size: int) -> float:
    """Variance of the 4-neighbour Laplacian of a BGR crop, mean-pooled to ``size`` square.

    Area pooling (not point sampling) keeps the value comparable between a near and a far
    face and stops sensor noise from passing as sharpness. A blurred face has little edge
    energy. Returns 0.0 for an empty crop.
    """
    if crop.size == 0:
        return 0.0
    gray = _area_resize(crop.astype(np.float32) @ np.array(LUMA, np.float32), size)
    lap = (
        LAPLACIAN_CENTRE * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def yaw_proxy(landmarks: np.ndarray | None) -> float | None:
    """Nose offset from the eye midpoint in inter-eye distances (0 when frontal), or None.

    None when the landmarks are missing or degenerate.
    """
    if landmarks is None or np.shape(landmarks) != LANDMARK_SHAPE:
        return None
    pts = np.asarray(landmarks, np.float32)
    eye_dx = float(pts[LEFT_EYE, 0] - pts[RIGHT_EYE, 0])
    if abs(eye_dx) < EPSILON:
        return None
    return float((pts[NOSE, 0] - (pts[LEFT_EYE, 0] + pts[RIGHT_EYE, 0]) / 2) / abs(eye_dx))


class EnrollmentService:
    """One enrollment session at a time over a registry and a live pipeline.

    ``registry`` is None when the data folder could not be opened: every call that needs it
    raises ``EnrollmentError``.
    """

    def __init__(
        self,
        registry: Registry | None,
        cfg: EnrollmentConfig,
        identity: IdentityConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.cfg = cfg
        self.identity = identity
        self._clock = clock
        self._lock = threading.Lock()
        self._embedder: FaceEmbedder | None = None
        self._unavailable: str | None = "vision is not running"
        self._state = IDLE
        self._session: str | None = None
        self._token = 0
        self._deadline = 0.0
        self._samples: list[np.ndarray] = []
        self._poses: list[float | None] = []
        self._validated = False
        self._mean: np.ndarray | None = None
        self._last_stamp: float | None = None
        self._track_id: int | None = None
        self._face_visible = False
        self._rejected = dict.fromkeys(REJECT_REASONS, 0)

    # ------------------------------------------------------------ wiring

    def attach(self, pipeline: VisionPipeline) -> None:
        """Take embeddings with the pipeline's face embedder and listen to its frames."""
        self._embedder = pipeline.face_embedder
        self._unavailable = None
        pipeline.face_tap = self.on_frame

    def set_unavailable(self, reason: str) -> None:
        """Vision cannot run (models missing, ...): ``start`` raises with this reason."""
        self._embedder = None
        self._unavailable = reason

    # ------------------------------------------------------------ session

    def start(self, session_id: str) -> None:
        """Begin collecting face embeddings for ``session_id``.

        Raises ``EnrollmentError`` while vision is unavailable or another session is active.
        A finished, cancelled or timed-out session is replaced.
        """
        with self._lock:
            self._expire_locked()
            if self._embedder is None:
                raise EnrollmentError(f"vision unavailable: {self._unavailable}")
            if self._state in (*ACTIVE_STATES, COMPLETING):
                raise EnrollmentError("an enrollment session is already running")
            self._reset_locked()
            self._state, self._session = COLLECTING, session_id
            self._deadline = self._clock() + self.cfg.timeout_s
        log.info("enrollment session started")

    def expire(self) -> None:
        """End a session that ran past ``enrollment.timeout_s``, even with no frames or calls.

        The app calls it from its main loop, so abandoned samples do not outlive the timeout.
        """
        with self._lock:
            self._expire_locked()

    def status(self) -> dict[str, Any]:
        """Progress for the UI: counts and flags only, never embeddings or names."""
        with self._lock:
            self._expire_locked()
            hint = None
            if self._state == COLLECTING and len(self._samples) >= self.cfg.target_samples:
                hint = "turn the head slightly left and right"
            return {
                "session_id": self._session,
                "state": self._state if self._embedder is not None else UNAVAILABLE,
                "reason": self._unavailable if self._embedder is None else None,
                "collected": len(self._samples),
                "target": self.cfg.target_samples,
                "minimum": self.identity.enroll_min_embeddings,
                "can_complete": self._can_complete_locked(),
                "pose_varied": self._varied_locked(),
                "hint": hint,
                "face_visible": self._face_visible,
                "rejected": dict(self._rejected),
            }

    def complete(
        self,
        name: str,
        consent: Consent,
        pending_id: str | None = None,
    ) -> str:
        """Enroll the collected samples as ``name`` and return the person id.

        With ``pending_id`` the owner has chosen a name-only record from a catalogue import
        (SPEC 6) and it is completed in place (``Registry.complete_enrollment``); without it a
        new person is made and no pending record is touched, whatever its name.

        Needs an active session with at least ``identity.enroll_min_embeddings`` samples that
        show varied angles. The registry (disk I/O) runs outside the service lock, in the
        ``completing`` state: a second ``complete``, a ``start`` or a new sample meanwhile is
        refused or ignored, so a session completes once. When the registry refuses
        (``ConsentError``, ``ValueError``) the samples are kept and the session returns to its
        earlier state, so the caller can correct the input and call again; ``cancel`` discards
        them. On success the samples are zeroed and the session ends as ``completed``.
        """
        with self._lock:
            self._expire_locked()
            if self._state not in ACTIVE_STATES or self._embedder is None:
                raise EnrollmentError("no enrollment session is running")
            have, need = len(self._samples), self.identity.enroll_min_embeddings
            if have < need:
                raise EnrollmentError(f"only {have} of {need} samples collected")
            if not self._can_complete_locked():
                raise EnrollmentError("the samples show too little change of head angle")
            previous, token, model_id = self._state, self._token, self._embedder.model_id
            matrix = np.vstack(self._samples)
            self._state = COMPLETING
        try:
            if pending_id is None:
                record = self._reg().enroll(name, matrix, model_id, consent=consent)
            else:
                record = self._reg().complete_enrollment(
                    pending_id,
                    matrix,
                    model_id,
                    consent,
                    display_name=name,
                )
        except BaseException:
            with self._lock:
                if self._token == token and self._state == COMPLETING:
                    self._state = previous
            raise
        with self._lock:
            self._reset_locked()
            self._state = COMPLETED
        log.info("enrollment session completed")
        return record.person_id

    def cancel(self) -> None:
        """Discard the session and its samples; harmless when none is running.

        A session that is just being stored (``completing``) is not cancelled: the person is
        enrolled once the registry call returns.
        """
        with self._lock:
            if self._state == COMPLETING:
                return
            was_active = self._state in ACTIVE_STATES
            self._reset_locked()
            self._state = CANCELLED if was_active else self._state
        if was_active:
            log.info("enrollment session cancelled")

    # ------------------------------------------------------------ pipeline hook

    def on_frame(self, frame: Frame, tracks: list[Track]) -> None:
        """Pipeline tap (vision thread): maybe take one embedding from this frame."""
        with self._lock:
            self._expire_locked()
            if self._state != COLLECTING or self._embedder is None:
                return
            self._face_visible = bool(tracks)
            track = self._pick(frame, tracks)
            if track is None or track.detection is None:
                return
            if self._last_stamp is not None and frame.stamp - self._last_stamp < self.cfg.min_gap_s:
                return
            reason = self._reject_reason(frame, track)
            if reason is not None:
                self._rejected[reason] += 1
                return
            token, embedder = self._token, self._embedder
        embedding = np.asarray(embedder.embed(frame.image, track.detection), np.float32).ravel()
        norm = float(np.linalg.norm(embedding))
        if norm < EPSILON or not np.isfinite(norm):
            return
        embedding /= norm
        pose = yaw_proxy(track.detection.landmarks)
        with self._lock:
            if self._state != COLLECTING or self._token != token:
                return  # cancelled, completed or restarted while embedding: discard
            if not self._accept_locked(embedding, pose):
                return
            self._last_stamp = frame.stamp
            n = len(self._samples)
            cap = self.identity.max_embeddings_per_person
            if n >= cap or (n >= self.cfg.target_samples and self._varied_locked()):
                self._state = READY

    # ------------------------------------------------------------ registry pass-throughs

    def list_records(self) -> list[RecordSummary]:
        """People, animals and objects without biometrics (``Registry.list``)."""
        return self._reg().list()

    def quarantined(self) -> list[str]:
        """Keys of people items that could not be read at startup (``Registry.quarantined``)."""
        return self._reg().quarantined()

    def rename(self, record_id: str, display_name: str) -> None:
        """Rename a person, animal or object; ``KeyError`` when the id is unknown."""
        if self._is_person(record_id):
            self._reg().rename_person(record_id, display_name)
        else:
            self._reg().rename_entity(record_id, display_name)

    def merge(self, keep_id: str, drop_id: str) -> None:
        """Merge two animals or objects (people cannot be merged); ``ValueError`` otherwise."""
        if self._is_person(keep_id) or self._is_person(drop_id):
            raise ValueError("people cannot be merged")
        self._reg().merge_entities(keep_id, drop_id)

    def delete(self, record_id: str) -> None:
        """Erase a person (``forget``: embeddings, fields, history) or an animal or object.

        An id the registry does not know raises ``KeyError``. A pending animal or object
        (catalogue import) also leaves a tombstone, a salted hash of its kind, category and
        name, written before the delete, so importing the same catalogue again does not bring
        it back (SPEC 6). People are erased with ``forget`` and leave no tombstone (SPEC 12).
        """
        reg = self._reg()
        if self._is_person(record_id) or record_id in self.quarantined():
            reg.forget(record_id)
            return
        entity = reg.get_entity(record_id)
        if entity.status == "pending":
            reg.tombstones.add(entity.kind, entity.category, entity.display_name)
        reg.delete_entity(record_id)

    # ------------------------------------------------------------ internals

    def _reg(self) -> Registry:
        if self.registry is None:
            raise EnrollmentError(f"registry unavailable: {self._unavailable}")
        return self.registry

    def _is_person(self, record_id: str) -> bool:
        try:
            self._reg().get_person(record_id)
        except KeyError:
            return False
        return True

    def _spread_locked(self) -> float | None:
        known = [p for p in self._poses if p is not None]
        return max(known) - min(known) if len(known) >= 2 else None

    def _varied_locked(self) -> bool:
        """Enough change of head angle (always true when the threshold is 0)."""
        if self.cfg.min_pose_spread <= 0:
            return True
        spread = self._spread_locked()
        return spread is not None and spread >= self.cfg.min_pose_spread

    def _can_complete_locked(self) -> bool:
        n = len(self._samples)
        if self._state not in ACTIVE_STATES or not self._validated:
            return False
        if n < self.identity.enroll_min_embeddings:
            return False
        return self._varied_locked() or n >= self.identity.max_embeddings_per_person

    def _accept_locked(self, embedding: np.ndarray, pose: float | None) -> bool:
        """Add the sample if it agrees with the others; the first one waits for a second.

        The first sample is a candidate. A second one that resembles it validates both; one
        that does not replaces it (counted as inconsistent), so a wrong first face cannot
        lock the session onto another person.
        """
        if self._mean is None:
            self._samples, self._poses = [embedding], [pose]
            self._mean = embedding.copy()
            return True
        similarity = float(embedding @ self._mean)
        if similarity < self.cfg.consistency_min:
            self._rejected["inconsistent"] += 1
            if not self._validated:
                for sample in self._samples:
                    sample[:] = 0.0
                self._samples, self._poses = [embedding], [pose]
                self._mean = embedding.copy()
                return True
            return False
        self._validated = True
        self._samples.append(embedding)
        self._poses.append(pose)
        mean = np.mean(self._samples, axis=0)
        self._mean = mean / max(float(np.linalg.norm(mean)), EPSILON)
        return True

    def _reset_locked(self) -> None:
        for sample in self._samples:
            sample[:] = 0.0
        self._samples = []
        self._poses = []
        self._validated = False
        self._mean = None
        self._last_stamp = None
        self._track_id = None
        self._face_visible = False
        self._session = None
        self._state = IDLE
        self._rejected = dict.fromkeys(REJECT_REASONS, 0)
        self._token += 1

    def _expire_locked(self) -> None:
        if self._state in ACTIVE_STATES and self._clock() >= self._deadline:
            self._reset_locked()
            self._state = TIMED_OUT
            log.info("enrollment session timed out")

    def _pick(self, frame: Frame, tracks: list[Track]) -> Track | None:
        """The session's track while it is visible, else the largest and most central one.

        The lock on a track ends with the track: when it is gone the next call chooses afresh.
        """
        if not tracks:
            self._track_id = None
            return None
        for t in tracks:
            if t.track_id == self._track_id:
                return t
        height, width = frame.image.shape[:2]

        def weight(t: Track) -> float:
            x1, y1, x2, y2 = t.box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
            dx = ((x1 + x2) / 2 - width / 2) / (width / 2)
            dy = ((y1 + y2) / 2 - height / 2) / (height / 2)
            dist = min(1.0, (dx * dx + dy * dy) ** 0.5)
            return area * (1.0 - self.cfg.centre_weight * dist)

        chosen = max(tracks, key=weight)
        self._track_id = chosen.track_id
        return chosen

    def _reject_reason(self, frame: Frame, track: Track) -> str | None:
        det = track.detection
        assert det is not None
        if det.score < self.cfg.min_score:
            return "score"
        if det.min_side < self.cfg.min_face_px:
            return "small"
        height, width = frame.image.shape[:2]
        x1, y1, x2, y2 = det.box
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return "cropped"
        crop = frame.image[int(y1) : int(y2), int(x1) : int(x2)]
        if sharpness(crop, self.cfg.sharpness_size_px) < self.cfg.min_sharpness:
            return "blurred"
        return None


__all__ = [
    "CANCELLED",
    "COLLECTING",
    "COMPLETED",
    "COMPLETING",
    "IDLE",
    "READY",
    "TIMED_OUT",
    "UNAVAILABLE",
    "EnrollmentError",
    "EnrollmentService",
    "sharpness",
    "yaw_proxy",
]
