"""ByteTrack-class multi-object tracker in numpy (SPEC 5, Tracking).

Each track follows its box with a constant-velocity Kalman filter. An update associates in two
passes, as ByteTrack does: high-score detections against all tracks by IoU, then low-score
detections against the tracks still unmatched that were matched at the previous update (this
keeps a track alive through a weak detection, such as a partly turned face). Unmatched
high-score detections start tracks; a track is reported only after ``confirm_hits`` matches,
and removed once unmatched for ``lost_s`` seconds of capture stamps (the rate varies with the
governor, so frame counts would be wrong). Association is greedy by descending IoU, which
equals the optimal assignment for the few, well separated faces of a home scene.

Track IDs come from an allocator that can be shared between trackers, so IDs are unique across
faces and objects. No wall clock is read: everything uses the stamps passed in.

An optional ``observer`` (``track_stats.TrackDiagnostics``) is told about starts, matches,
misses, confirmations, refused low-score detections and endings, for the debug report. It only
watches: the tracks and IDs are the same with or without it.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..config.schema import TrackerConfig
from .types import Box, Detection

if TYPE_CHECKING:
    from .track_stats import TrackDiagnostics

# Kalman noise as a fraction of the box height (the ByteTrack defaults).
POSITION_STD = 1.0 / 20
VELOCITY_STD = 1.0 / 160
INITIAL_POSITION_STD = 2 * POSITION_STD
INITIAL_VELOCITY_STD = 10 * VELOCITY_STD
STATE_DIM = 8
BOX_COLUMNS = 4
HEIGHT = 3
MEASURE_DIM = 4


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two ``(n, 4)`` and ``(m, 4)`` arrays of xyxy boxes.

    Returns an ``(n, m)`` matrix; pairs with no overlap, or boxes of zero area, give 0.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    x1 = np.maximum(ax1[:, None], bx1[None, :])
    y1 = np.maximum(ay1[:, None], by1[None, :])
    x2 = np.minimum(ax2[:, None], bx2[None, :])
    y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _cxcywh(box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


@dataclass(eq=False)
class Track:
    """A tracked object: stable ``track_id``, filtered box and the last matched detection."""

    track_id: int
    label: str
    score: float
    first_stamp: float
    last_stamp: float
    hits: int = 1
    confirmed: bool = False
    detection: Detection | None = field(default=None, repr=False)
    _mean: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM), repr=False)
    _cov: np.ndarray = field(default_factory=lambda: np.eye(STATE_DIM), repr=False)
    _stamp: float = 0.0
    _matched_last: bool = True

    @property
    def box(self) -> Box:
        """Current filtered box ``(x1, y1, x2, y2)`` in pixels."""
        cx, cy, w, h = self._mean[:MEASURE_DIM]
        return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))

    def _start(self, det: Detection, stamp: float) -> None:
        self._mean = np.concatenate([_cxcywh(det.box), np.zeros(MEASURE_DIM)])
        h = self._mean[HEIGHT]
        std = [INITIAL_POSITION_STD * h] * MEASURE_DIM + [INITIAL_VELOCITY_STD * h] * MEASURE_DIM
        self._cov = np.diag(np.array(std) ** 2)
        self._stamp = stamp

    def _predict(self, stamp: float) -> None:
        dt = max(0.0, stamp - self._stamp)
        h = max(self._mean[HEIGHT], 1.0)
        transition = np.eye(STATE_DIM)
        transition[:MEASURE_DIM, MEASURE_DIM:] = np.eye(MEASURE_DIM) * dt
        q_pos = (POSITION_STD * h) ** 2
        q_vel = (VELOCITY_STD * h) ** 2
        noise = np.diag([q_pos] * MEASURE_DIM + [q_vel] * MEASURE_DIM)
        self._mean = transition @ self._mean
        self._cov = transition @ self._cov @ transition.T + noise
        self._stamp = stamp

    def _correct(self, det: Detection) -> None:
        h = max(self._mean[HEIGHT], 1.0)
        meas_cov = np.eye(MEASURE_DIM) * (POSITION_STD * h) ** 2
        proj = np.zeros((MEASURE_DIM, STATE_DIM))
        proj[:, :MEASURE_DIM] = np.eye(MEASURE_DIM)
        innovation = _cxcywh(det.box) - proj @ self._mean
        s = proj @ self._cov @ proj.T + meas_cov
        gain = self._cov @ proj.T @ np.linalg.inv(s)
        self._mean = self._mean + gain @ innovation
        self._cov = (np.eye(STATE_DIM) - gain @ proj) @ self._cov


@dataclass
class TrackerResult:
    """Outcome of one update: visible confirmed tracks, newly confirmed IDs, removed IDs."""

    visible: list[Track]
    new_ids: set[int]
    removed_ids: list[int]


class Tracker:
    """Multi-object tracker; see the module docstring."""

    def __init__(self, cfg: TrackerConfig, ids: Iterator[int] | None = None):
        self.cfg = cfg
        self._ids = ids if ids is not None else itertools.count(1)
        self._tracks: list[Track] = []
        self.observer: TrackDiagnostics | None = None

    def reset(self, ids: Iterator[int] | None = None) -> None:
        """Drop all tracks; IDs keep counting from the allocator unless a new one is given.

        A replay passes a fresh allocator so track IDs restart and runs compare equal.
        """
        self._tracks.clear()
        if ids is not None:
            self._ids = ids

    @staticmethod
    def _greedy(
        tracks: list[Track], dets: list[Detection], min_iou: float
    ) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        iou = iou_matrix(
            np.array([t.box for t in tracks]).reshape(-1, BOX_COLUMNS),
            np.array([d.box for d in dets]).reshape(-1, BOX_COLUMNS),
        )
        same_label = np.array([[t.label == d.label for d in dets] for t in tracks]).reshape(
            len(tracks), len(dets)
        )
        iou = iou * same_label
        pairs = [(i, j) for i in range(len(tracks)) for j in range(len(dets))]
        pairs.sort(key=lambda p: -iou[p])
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched = []
        for i, j in pairs:
            if iou[i, j] <= 0 or iou[i, j] < min_iou:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            matched.append((tracks[i], dets[j]))
        rest_t = [t for i, t in enumerate(tracks) if i not in used_t]
        rest_d = [d for j, d in enumerate(dets) if j not in used_d]
        return matched, rest_t, rest_d

    def expire(self, stamp: float) -> list[int]:
        """Remove tracks unmatched for more than ``lost_s`` at ``stamp``; IDs of confirmed ones.

        ``update`` calls it first, so a track never associates after it has expired: a face
        back after a long gap gets a new ID. Callers may also call it between updates (when the
        detector did not run) so per-track state does not outlive the track.
        """
        gone = [t for t in self._tracks if stamp - t.last_stamp > self.cfg.lost_s]
        self._tracks = [t for t in self._tracks if t not in gone]
        if self.observer is not None:
            for t in gone:
                if t.confirmed:
                    self.observer.expired(t.track_id, t.box, t.last_stamp, stamp, self.cfg.lost_s)
                else:
                    self.observer.dropped(t.track_id)
        return [t.track_id for t in gone if t.confirmed]

    def update(self, detections: list[Detection], stamp: float) -> TrackerResult:
        """Fold one frame's detections (all labels share one tracker) into the tracks.

        Returns the confirmed tracks matched in this frame, the IDs confirmed by it, and the IDs
        of confirmed tracks removed by it (callers drop per-track state for those).
        Detections only associate with tracks of the same label.
        """
        expired = self.expire(stamp)
        for t in self._tracks:
            t._predict(stamp)
        high = [d for d in detections if d.score >= self.cfg.high_score]
        low = [d for d in detections if self.cfg.low_score <= d.score < self.cfg.high_score]
        matched, rest_t, rest_d = self._greedy(self._tracks, high, self.cfg.match_iou)
        recent = [t for t in rest_t if t._matched_last]
        matched_low, rest_t2, rest_low = self._greedy(recent, low, self.cfg.low_match_iou)
        stale = [t for t in rest_t if t not in recent]
        missed = stale + rest_t2
        if self.observer is not None:
            self._observe_low(self.observer, stale, rest_low, stamp)
        new_ids: set[int] = set()
        visible: list[Track] = []
        for t, d in matched + matched_low:
            t._correct(d)
            t.detection, t.score, t.last_stamp, t.hits = d, d.score, stamp, t.hits + 1
            t._matched_last = True
            if self.observer is not None:
                self.observer.matched(t.track_id)
            if not t.confirmed and t.hits >= self.cfg.confirm_hits:
                t.confirmed = True
                new_ids.add(t.track_id)
                if self.observer is not None:
                    self.observer.confirmed(t.track_id)
            if t.confirmed:
                visible.append(t)
        for t in missed:
            t._matched_last = False
            if self.observer is not None:
                self.observer.missed(t.track_id)
        for d in rest_d:
            t = Track(next(self._ids), d.label, d.score, stamp, stamp, detection=d)
            t._start(d, stamp)
            if self.observer is not None:
                near = {m.track_id: m.box for m in missed if m.confirmed and m.label == d.label}
                self.observer.started(t.track_id, d, stamp, near)
            if self.cfg.confirm_hits <= 1:
                t.confirmed = True
                new_ids.add(t.track_id)
                visible.append(t)
                if self.observer is not None:
                    self.observer.confirmed(t.track_id)
            self._tracks.append(t)
        dropped = [t for t in self._tracks if not t.confirmed and not t._matched_last]
        self._tracks = [t for t in self._tracks if t not in dropped]
        if self.observer is not None:
            for t in dropped:
                self.observer.dropped(t.track_id)
        visible.sort(key=lambda t: t.track_id)
        return TrackerResult(visible, new_ids, expired)

    def _observe_low(
        self, observer: TrackDiagnostics, stale: list[Track], low: list[Detection], stamp: float
    ) -> None:
        """Tell the observer which unmatched low-score detections lie near a track (IoU at least
        ``low_match_iou``, same label) that the low pass skipped because it missed the update
        before, and which overlap no live track of their label at all. Reads only;
        O(tracks x low), nothing when there is no low-score detection left."""
        if not low:
            return
        boxes = np.array([d.box for d in low]).reshape(-1, BOX_COLUMNS)
        live = iou_matrix(np.array([t.box for t in self._tracks]).reshape(-1, BOX_COLUMNS), boxes)
        unmatched = []
        for j, d in enumerate(low):
            near = [
                (live[i, j], t)
                for i, t in enumerate(self._tracks)
                if t.label == d.label and live[i, j] > 0
            ]
            refusable = [p for p in near if p[1] in stale and p[0] >= self.cfg.low_match_iou]
            if refusable:
                observer.refused_low(max(refusable, key=lambda p: p[0])[1].track_id, d)
            elif not near:
                unmatched.append(d)
        if unmatched:
            observer.low_unmatched(unmatched, stamp)
