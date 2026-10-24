"""ByteTrack-class multi-object tracker in numpy (SPEC 5, Tracking).

Each track follows its box with a constant-velocity Kalman filter. An update associates in two
passes, as ByteTrack does: high-score detections against all tracks, then low-score detections
against the tracks still unmatched that were matched at the previous update (this keeps a track
alive through a weak detection, such as a partly turned face). A pair matches when its IoU
reaches the pass's IoU gate or, failing that, when the detection's box centre lies within the
pass's distance gate, in heights of the track's predicted box (``match_distance_heights``,
``low_match_distance_heights``), and the two box heights differ by at most
``match_height_ratio``: a face seen side-on changes its box shape, so the IoU with its own
track drops below the gate (robot, 2026-10-23), but keeps its size, while a smaller face beside
or behind it does not. Association is greedy: IoU pairs first by descending IoU (for the few,
well separated faces of a home scene this matches the optimal assignment), then distance pairs
by ascending distance.

Unmatched high-score detections start tracks, except a duplicate: one whose centre lies within
``match_distance_heights`` of the corrected box of a confirmed track matched in this update,
with a height within ``match_height_ratio``. A track is reported only after
``confirm_hits`` matches, and removed once unmatched for ``lost_s`` seconds of capture stamps
(the rate varies with the governor, so frame counts would be wrong).

Track IDs come from an allocator that can be shared between trackers, so IDs are unique across
faces and objects. No wall clock is read: everything uses the stamps passed in.

An optional ``observer`` (``track_stats.TrackDiagnostics``) is told about starts, matches (and
which ones the distance gate made), duplicates not started, misses, confirmations, refused
low-score detections and endings, for the debug report. It only watches: the tracks and IDs are
the same with or without it.
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


def distance_matrix(tracks: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """Centre distance of each ``(n, 4)`` track box to each ``(m, 4)`` detection box, in
    heights of the track box; ``inf`` for a track box of zero height. Returns ``(n, m)``."""
    if len(tracks) == 0 or len(dets) == 0:
        return np.full((len(tracks), len(dets)), np.inf)
    tx1, ty1, tx2, ty2 = tracks.T
    dx1, dy1, dx2, dy2 = dets.T
    dx = (dx1 + dx2)[None, :] / 2 - (tx1 + tx2)[:, None] / 2
    dy = (dy1 + dy2)[None, :] / 2 - (ty1 + ty2)[:, None] / 2
    gap, height = np.hypot(dx, dy), ty2 - ty1
    return np.where(height[:, None] > 0, gap / np.where(height > 0, height, 1)[:, None], np.inf)


def gated_distance(tracks: np.ndarray, dets: np.ndarray, ratio: float) -> np.ndarray:
    """``distance_matrix``, with ``inf`` where the larger of the two box heights is more than
    ``ratio`` times the smaller (a box of zero height never passes)."""
    dist = distance_matrix(tracks, dets)
    if dist.size == 0:
        return dist
    _, ty1, _, ty2 = tracks.T
    _, dy1, _, dy2 = dets.T
    th, dh = (ty2 - ty1)[:, None], (dy2 - dy1)[None, :]
    small, large = np.minimum(th, dh), np.maximum(th, dh)
    return np.where((small > 0) & (large <= ratio * small), dist, np.inf)


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

    def _greedy(
        self, tracks: list[Track], dets: list[Detection], min_iou: float, max_distance: float
    ) -> tuple[list[tuple[Track, Detection, float | None]], list[Track], list[Detection]]:
        """Greedy association of one pass: IoU pairs at or above ``min_iou`` (and above 0) by
        descending IoU, then the remaining same-label pairs within ``max_distance`` track heights
        and ``match_height_ratio`` by ascending distance (none when it is 0). Matches carry the
        distance when the distance gate made them, else None; also returns the unmatched tracks
        and detections."""
        t_boxes = np.array([t.box for t in tracks]).reshape(-1, BOX_COLUMNS)
        d_boxes = np.array([d.box for d in dets]).reshape(-1, BOX_COLUMNS)
        same_label = np.array([[t.label == d.label for d in dets] for t in tracks]).reshape(
            len(tracks), len(dets)
        )
        iou = iou_matrix(t_boxes, d_boxes) * same_label
        ratio = self.cfg.match_height_ratio
        dist = np.where(same_label, gated_distance(t_boxes, d_boxes, ratio), np.inf)
        cells = [(i, j) for i in range(len(tracks)) for j in range(len(dets))]
        by_iou = sorted(
            (p for p in cells if iou[p] > 0 and iou[p] >= min_iou), key=lambda p: -iou[p]
        )
        rest = [p for p in cells if not (iou[p] > 0 and iou[p] >= min_iou)]
        near = [p for p in rest if max_distance > 0 and dist[p] <= max_distance]
        by_distance = sorted(near, key=lambda p: dist[p])
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: list[tuple[Track, Detection, float | None]] = []
        for by_dist, pairs in ((False, by_iou), (True, by_distance)):
            for i, j in pairs:
                if i in used_t or j in used_d:
                    continue
                used_t.add(i)
                used_d.add(j)
                matched.append((tracks[i], dets[j], float(dist[i, j]) if by_dist else None))
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
        cfg = self.cfg
        matched, rest_t, rest_d = self._greedy(
            self._tracks, high, cfg.match_iou, cfg.match_distance_heights
        )
        recent = [t for t in rest_t if t._matched_last]
        matched_low, rest_t2, rest_low = self._greedy(
            recent, low, cfg.low_match_iou, cfg.low_match_distance_heights
        )
        low_tracks = {id(m[0]) for m in matched_low}
        stale = [t for t in rest_t if t not in recent]
        missed = stale + rest_t2
        if self.observer is not None:
            self._observe_low(self.observer, stale, rest_low, stamp)
        new_ids: set[int] = set()
        visible: list[Track] = []
        for t, d, distance in matched + matched_low:
            t._correct(d)
            t.detection, t.score, t.last_stamp, t.hits = d, d.score, stamp, t.hits + 1
            t._matched_last = True
            if self.observer is not None:
                self.observer.matched(t.track_id, distance, low=id(t) in low_tracks)
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
        confirmed = [m[0] for m in matched + matched_low if m[0].confirmed]
        for d in self._not_duplicates(rest_d, confirmed):
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

    def _not_duplicates(self, dets: list[Detection], matched: list[Track]) -> list[Detection]:
        """The unmatched high-score detections that may start a track: those whose centre is
        farther than ``match_distance_heights`` from every confirmed track (same label, height
        within ``match_height_ratio``) matched in this update, measured on the corrected box.
        The others are duplicates of the face that track already follows; the observer counts
        them. All pass when the gate is 0."""
        gate = self.cfg.match_distance_heights
        if gate <= 0 or not dets or not matched:
            return dets
        dist = gated_distance(
            np.array([t.box for t in matched]).reshape(-1, BOX_COLUMNS),
            np.array([d.box for d in dets]).reshape(-1, BOX_COLUMNS),
            self.cfg.match_height_ratio,
        )
        keep = []
        for j, d in enumerate(dets):
            near = [(dist[i, j], t) for i, t in enumerate(matched) if t.label == d.label]
            near = [p for p in near if p[0] <= gate]
            if not near:
                keep.append(d)
            elif self.observer is not None:
                self.observer.duplicate(min(near, key=lambda p: p[0])[1].track_id, d)
        return keep

    def _observe_low(
        self, observer: TrackDiagnostics, stale: list[Track], low: list[Detection], stamp: float
    ) -> None:
        """Tell the observer which unmatched low-score detections the low pass would have matched
        to a track (IoU at least ``low_match_iou``, or centre within
        ``low_match_distance_heights`` with a height within ``match_height_ratio``, same label)
        that it skipped because the track missed the
        update before, and which neither overlap nor lie within that gate of any live track of
        their label. Reads only; O(tracks x low), nothing when no low-score detection is left."""
        if not low:
            return
        cfg = self.cfg
        boxes = np.array([d.box for d in low]).reshape(-1, BOX_COLUMNS)
        t_boxes = np.array([t.box for t in self._tracks]).reshape(-1, BOX_COLUMNS)
        live = iou_matrix(t_boxes, boxes)
        dist = gated_distance(t_boxes, boxes, cfg.match_height_ratio)
        gate = cfg.low_match_distance_heights
        unmatched = []
        for j, d in enumerate(low):
            near = [
                (live[i, j], dist[i, j], t)
                for i, t in enumerate(self._tracks)
                if t.label == d.label and (live[i, j] > 0 or (gate > 0 and dist[i, j] <= gate))
            ]
            refusable = [
                p
                for p in near
                if p[2] in stale
                and ((p[0] > 0 and p[0] >= cfg.low_match_iou) or (gate > 0 and p[1] <= gate))
            ]
            if refusable:
                best = max(refusable, key=lambda p: (p[0], -p[1]))
                observer.refused_low(best[2].track_id, d)
            elif not near:
                unmatched.append(d)
        if unmatched:
            observer.low_unmatched(unmatched, stamp)
