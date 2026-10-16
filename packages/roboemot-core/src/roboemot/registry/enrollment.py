"""Face enrollment from the live vision pipeline (SPEC 6, Enrollment; SPEC 12 Data).

The web UI (or any caller) drives one session at a time: ``start`` begins collecting,
``status`` reports progress, ``complete`` enrolls with a consent record, ``cancel`` discards.
While a session runs, the pipeline calls ``on_frame`` for every processed frame. The service
picks the largest, most central face track, rejects frames that are too small, low-scoring
or blurred, spaces the samples in time, and embeds the face itself on the vision thread (so
the ONNX session stays single-threaded). A sample whose embedding does not resemble the ones
collected so far is rejected too, so two people never mix in one session.

Privacy (SPEC 12): face crops are used for the sharpness check and dropped at once; embeddings
stay in this object's memory until ``complete`` hands them to ``Registry.enroll`` and are
zeroed on completion, cancellation or timeout. Neither embeddings nor names are logged.

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
TIMED_OUT = "timed_out"
COMPLETED = "completed"
CANCELLED = "cancelled"
UNAVAILABLE = "unavailable"
ACTIVE_STATES = (COLLECTING, READY)
REJECT_REASONS = ("score", "small", "blurred", "inconsistent")
LUMA = (0.114, 0.587, 0.299)  # BGR weights of the Rec. 601 luma
EPSILON = 1e-9
LAPLACIAN_CENTRE = -4.0


class EnrollmentError(RuntimeError):
    """The request does not fit the session state (busy, nothing to complete, no vision)."""


def sharpness(crop: np.ndarray, size: int) -> float:
    """Variance of the 4-neighbour Laplacian of a BGR crop, nearest-resized to ``size`` square.

    A blurred face has little edge energy; the value is scale-normalised by the resize, not by
    lighting. Returns 0.0 for an empty crop.
    """
    if crop.size == 0:
        return 0.0
    rows = np.linspace(0, crop.shape[0] - 1, size).astype(int)
    cols = np.linspace(0, crop.shape[1] - 1, size).astype(int)
    gray = crop[rows][:, cols].astype(np.float32) @ np.array(LUMA, np.float32)
    lap = (
        LAPLACIAN_CENTRE * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


class EnrollmentService:
    """One enrollment session at a time over a registry and a live pipeline."""

    def __init__(
        self,
        registry: Registry,
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
            if self._state in ACTIVE_STATES:
                raise EnrollmentError("an enrollment session is already running")
            self._reset_locked()
            self._state, self._session = COLLECTING, session_id
            self._token += 1
            self._deadline = self._clock() + self.cfg.timeout_s
        log.info("enrollment session started")

    def status(self) -> dict[str, Any]:
        """Progress for the UI: counts and flags only, never embeddings or names."""
        with self._lock:
            self._expire_locked()
            return {
                "session_id": self._session,
                "state": self._state if self._embedder is not None else UNAVAILABLE,
                "reason": self._unavailable if self._embedder is None else None,
                "collected": len(self._samples),
                "target": self.cfg.target_samples,
                "minimum": self.identity.enroll_min_embeddings,
                "can_complete": self._state in ACTIVE_STATES
                and len(self._samples) >= self.identity.enroll_min_embeddings,
                "face_visible": self._face_visible,
                "rejected": dict(self._rejected),
            }

    def complete(
        self,
        name: str,
        consent: Consent,
        declared_age_group: str | None = None,
        declared_gender: str | None = None,
    ) -> str:
        """Enroll the collected samples as ``name`` and return the new person id.

        Needs an active session with at least ``identity.enroll_min_embeddings`` samples. The
        registry checks the consent (guardian for possible minors, ``face`` scope). When it
        refuses (``ConsentError``, ``ValueError``) the samples are kept and the session stays
        open, so the caller can correct the input and call again; ``cancel`` discards them.
        On success the samples are zeroed and the session ends as ``completed``.
        """
        with self._lock:
            self._expire_locked()
            if self._state not in ACTIVE_STATES or self._embedder is None:
                raise EnrollmentError("no enrollment session is running")
            have, need = len(self._samples), self.identity.enroll_min_embeddings
            if have < need:
                raise EnrollmentError(f"only {have} of {need} samples collected")
            record = self.registry.enroll(
                name,
                np.vstack(self._samples),
                self._embedder.model_id,
                consent=consent,
                age_group=declared_age_group,
                gender=declared_gender,
            )
            self._reset_locked()
            self._state = COMPLETED
            person_id = record.person_id
        log.info("enrollment session completed")
        return person_id

    def cancel(self) -> None:
        """Discard the session and its samples; harmless when none is running."""
        with self._lock:
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
            if self._last_stamp is not None and frame.stamp - self._last_stamp < self.cfg.min_gap_s:
                return
            track = self._pick(frame, tracks)
            if track is None or track.detection is None:
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
        with self._lock:
            if self._state != COLLECTING or self._token != token:
                return
            if self._mean is not None and float(embedding @ self._mean) < self.cfg.consistency_min:
                self._rejected["inconsistent"] += 1
                return
            self._samples.append(embedding)
            mean = np.mean(self._samples, axis=0)
            self._mean = mean / max(float(np.linalg.norm(mean)), EPSILON)
            self._last_stamp, self._track_id = frame.stamp, track.track_id
            if len(self._samples) >= self.cfg.target_samples:
                self._state = READY

    # ------------------------------------------------------------ registry pass-throughs

    def list_records(self) -> list[RecordSummary]:
        """People, animals and objects without biometrics (``Registry.list``)."""
        return self.registry.list()

    def quarantined(self) -> list[str]:
        """Keys of people items that could not be read at startup (``Registry.quarantined``)."""
        return self.registry.quarantined()

    def rename(self, record_id: str, display_name: str) -> None:
        """Rename a person, animal or object; ``KeyError`` when the id is unknown."""
        if self._is_person(record_id):
            self.registry.rename_person(record_id, display_name)
        else:
            self.registry.rename_entity(record_id, display_name)

    def merge(self, keep_id: str, drop_id: str) -> None:
        """Merge two animals or objects (people cannot be merged); ``ValueError`` otherwise."""
        if self._is_person(keep_id) or self._is_person(drop_id):
            raise ValueError("people cannot be merged")
        self.registry.merge_entities(keep_id, drop_id)

    def delete(self, record_id: str) -> None:
        """Erase a person (``forget``: embeddings, fields, history) or an animal or object.

        An id the registry does not know raises ``KeyError``.
        """
        if self._is_person(record_id) or record_id in self.quarantined():
            self.registry.forget(record_id)
        else:
            self.registry.get_entity(record_id)
            self.registry.delete_entity(record_id)

    # ------------------------------------------------------------ internals

    def _is_person(self, record_id: str) -> bool:
        try:
            self.registry.get_person(record_id)
        except KeyError:
            return False
        return True

    def _reset_locked(self) -> None:
        for sample in self._samples:
            sample[:] = 0.0
        self._samples = []
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
        """The session's track while it is visible, else the largest and most central one."""
        if not tracks:
            return None
        if self._track_id is not None:
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

        return max(tracks, key=weight)

    def _reject_reason(self, frame: Frame, track: Track) -> str | None:
        det = track.detection
        assert det is not None
        if det.score < self.cfg.min_score:
            return "score"
        if det.min_side < self.cfg.min_face_px:
            return "small"
        x1, y1, x2, y2 = (int(v) for v in det.box)
        height, width = frame.image.shape[:2]
        crop = frame.image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if sharpness(crop, self.cfg.sharpness_size_px) < self.cfg.min_sharpness:
            return "blurred"
        return None


__all__ = [
    "CANCELLED",
    "COLLECTING",
    "COMPLETED",
    "IDLE",
    "READY",
    "TIMED_OUT",
    "UNAVAILABLE",
    "EnrollmentError",
    "EnrollmentService",
    "sharpness",
]
