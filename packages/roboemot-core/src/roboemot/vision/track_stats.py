"""Face track diagnostics for the debug report (SPEC 5 Tracking, 12A): why tracks end and why
the same face comes back as a new track. Measurement only: nothing here changes tracking.

``Tracker`` calls the hooks of an observer it was given (the pipeline gives one to the face
tracker only). Per update the cost is O(tracks); memory is the live tracks plus windows of
``vision.report_window`` samples, like the lifetime summary.

- ``ended_by``: how confirmed tracks ended: ``expired`` (unmatched for longer than the
  effective ``lost_s``) or ``reset`` (the pipeline was reset). ``tentative_dropped`` counts
  tracks that ended before they were confirmed.
- ``expiry``: for each expired track, the updates it went unmatched (``misses``), the lifetime
  ``lost_s`` in force when it expired (SPEC 5: it follows the detection period), and how many
  expired tracks had a low-score detection near them refused during their final run of misses
  (``after_refused_low``; a match clears it).
- ``rebirths``: pairs of an ended (or ending) confirmed track and a confirmed track born after
  its last match, whose first box centre lies within ``vision.rebirth_distance_heights``
  heights of the old track's last (Kalman predicted) box. Pairs are found when the new track is
  confirmed (old track already ended, the new one born at most ``vision.rebirth_window_s``
  after the expiry, so the window does not shrink when ``lost_s`` grows with the detection
  period) or when the old track expires (new track already confirmed: ``overlapped``, the two
  lived side by side; no window, and the old box is the one the tracker predicted, and refused,
  at the new track's birth). Summaries: the gap from the old track's last match to the new
  track's first detection, the old track's ``lost_s`` at expiry (how much of the gap was
  tracker lifetime), the IoU and the centre distance between the two boxes, and, when the
  frames carry the head joint state and the robot gave a ``FrameSpec``, the head yaw and pitch
  change in degrees.
- ``low_score``: low-score detections refused because the near track (IoU at least
  ``low_match_iou``) missed the update before (``refused``, with a score summary); those that
  overlap no live track (``unmatched``, with a score summary: they never start a track); and,
  among these, those within the rebirth gate of a track that expired at most
  ``vision.rebirth_window_s`` before (``near_ended``, with a score summary: the face was there,
  too weak to restart a track).
- ``start_score``: scores of the detections that started tracks; ``start_near_iou``: for a start
  next to confirmed tracks the update left unmatched, the best IoU with their predicted boxes
  (the association that failed ``match_iou``: a split rather than a new face).
- ``below_min_px``: face detections dropped by ``vision.face_min_px`` before the tracker.

Numbers only, no boxes or images leave this module (SPEC 12).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.schema import VisionConfig
from ..stats import SUMMARY_LEVELS as LEVELS
from ..stats import summarise
from ..world import FrameSpec
from .tracker import iou_matrix
from .types import Box, Detection

Pose = tuple[float, float]


def _centre(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _iou(a: Box, b: Box) -> float:
    return float(iou_matrix(np.array([a]), np.array([b]))[0, 0])


def head_pose(joint_state: Mapping[str, float] | None, spec: FrameSpec | None) -> Pose | None:
    """Head yaw and pitch (rad, summed over the spec's dofs), or None when the frame has no
    joint state, the robot gave no spec, or a dof is missing."""
    if joint_state is None or spec is None or not (spec.yaw_dofs or spec.pitch_dofs):
        return None
    names = (*spec.yaw_dofs, *spec.pitch_dofs)
    if any(n not in joint_state for n in names):
        return None
    yaw = sum(joint_state[n] for n in spec.yaw_dofs)
    pitch = sum(joint_state[n] for n in spec.pitch_dofs)
    return yaw, pitch


@dataclass
class _Live:
    """What the diagnostics keep about one live track."""

    birth_box: Box
    birth_stamp: float
    birth_pose: Pose | None
    last_pose: Pose | None
    neighbours: dict[int, Box]
    misses: int = 0
    refused_low: bool = False
    confirmed: bool = False
    paired: bool = False


@dataclass
class _Ended:
    """An expired confirmed track waiting for a rebirth within the window."""

    track_id: int
    end_stamp: float
    last_stamp: float
    box: Box
    last_pose: Pose | None
    lost_s: float


class TrackDiagnostics:
    """Tracker observer that counts face track endings and rebirths; see the module docstring.

    ``pose`` is the head pose of the frame being tracked, set by the pipeline before each
    update. All state is guarded by one lock, so ``snapshot`` may run on another thread.
    """

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.pose: Pose | None = None
        self._lock = threading.Lock()
        self._live: dict[int, _Live] = {}
        window = cfg.report_window
        self._ended: deque[_Ended] = deque(maxlen=window)
        self._ended_by = {"expired": 0, "reset": 0}
        self._tentative_dropped = 0
        self._misses: deque[int] = deque(maxlen=window)
        self._lost_s: deque[float] = deque(maxlen=window)
        self._after_refused = 0
        self._rebirths = 0
        self._overlapped = 0
        self._gap_s: deque[float] = deque(maxlen=window)
        self._rebirth_lost_s: deque[float] = deque(maxlen=window)
        self._iou: deque[float] = deque(maxlen=window)
        self._distance: deque[float] = deque(maxlen=window)
        self._yaw_deg: deque[float] = deque(maxlen=window)
        self._pitch_deg: deque[float] = deque(maxlen=window)
        self._refused = 0
        self._refused_score: deque[float] = deque(maxlen=window)
        self._unmatched = 0
        self._unmatched_score: deque[float] = deque(maxlen=window)
        self._near_ended = 0
        self._near_ended_score: deque[float] = deque(maxlen=window)
        self._start_score: deque[float] = deque(maxlen=window)
        self._start_near_iou: deque[float] = deque(maxlen=window)
        self._below_min_px = 0

    # ------------------------------------------------------------ tracker hooks

    def started(
        self, track_id: int, det: Detection, stamp: float, neighbours: dict[int, Box]
    ) -> None:
        """A detection started a (tentative) track. ``neighbours`` are the predicted boxes of
        the confirmed tracks of its label that this update left unmatched, by track ID."""
        with self._lock:
            self._live[track_id] = _Live(det.box, stamp, self.pose, self.pose, neighbours)
            self._start_score.append(det.score)
            if neighbours:
                self._start_near_iou.append(max(_iou(det.box, b) for b in neighbours.values()))

    def matched(self, track_id: int) -> None:
        """The track was matched in this update."""
        with self._lock:
            live = self._live.get(track_id)
            if live is not None:
                live.misses = 0
                live.refused_low = False
                live.last_pose = self.pose

    def missed(self, track_id: int) -> None:
        """The track went unmatched in this update."""
        with self._lock:
            live = self._live.get(track_id)
            if live is not None:
                live.misses += 1

    def refused_low(self, track_id: int, det: Detection) -> None:
        """A low-score detection near the track was refused: the track missed the last update."""
        with self._lock:
            self._refused += 1
            self._refused_score.append(det.score)
            live = self._live.get(track_id)
            if live is not None:
                live.refused_low = True

    def low_unmatched(self, dets: list[Detection], stamp: float) -> None:
        """Low-score detections that overlap no live track; those within the rebirth gate of a
        track that expired in the window are counted apart (``near_ended``)."""
        window, limit = self.cfg.rebirth_window_s, self.cfg.rebirth_distance_heights
        with self._lock:
            recent = [old for old in self._ended if stamp - old.end_stamp <= window]
            for det in dets:
                self._unmatched += 1
                self._unmatched_score.append(det.score)
                if any(self._distance_heights(old.box, det.box) <= limit for old in recent):
                    self._near_ended += 1
                    self._near_ended_score.append(det.score)

    def below_min_px(self, count: int) -> None:
        """Face detections dropped before the tracker for a box side below ``face_min_px``."""
        with self._lock:
            self._below_min_px += count

    def dropped(self, track_id: int) -> None:
        """A track ended before it was confirmed."""
        with self._lock:
            self._live.pop(track_id, None)
            self._tentative_dropped += 1

    def confirmed(self, track_id: int) -> None:
        """A track was confirmed: pair it with the nearest ended track in the window, if any
        (the last ``report_window`` ended tracks are kept; the window is checked per pair)."""
        with self._lock:
            live = self._live.get(track_id)
            if live is None:
                return
            live.confirmed = True
            window = self.cfg.rebirth_window_s
            while self._ended and live.birth_stamp - self._ended[0].end_stamp > window:
                self._ended.popleft()  # confirmations come in birth order
            best = self._nearest(live, list(self._ended))
            if best is not None:
                self._ended.remove(best)
                live.paired = True
                self._record(best, live, overlapped=False)

    def expired(
        self, track_id: int, box: Box, last_stamp: float, stamp: float, lost_s: float
    ) -> None:
        """A confirmed track expired at ``stamp`` with ``lost_s`` in force; ``box`` is its last
        predicted box. It is paired with a confirmed live track born after its last match (the
        two overlapped: no window applies, and the box is the one predicted at that track's
        birth), or kept for a rebirth that confirms later."""
        with self._lock:
            live = self._live.pop(track_id, None)
            self._ended_by["expired"] += 1
            if live is None:
                return
            self._lost_s.append(lost_s)
            self._misses.append(live.misses)
            self._after_refused += live.refused_low
            ended = _Ended(track_id, stamp, last_stamp, box, live.last_pose, lost_s)
            born = [
                other
                for other in self._live.values()
                if other.confirmed and not other.paired and other.birth_stamp > last_stamp
            ]
            match = self._nearest_born(ended, born)
            if match is None:
                self._ended.append(ended)
                return
            match.paired = True
            self._record(ended, match, overlapped=True)

    def reset(self, confirmed: int) -> None:
        """The pipeline was reset: ``confirmed`` live tracks ended; nothing pairs across it."""
        with self._lock:
            self._ended_by["reset"] += confirmed
            self._live.clear()
            self._ended.clear()

    # ------------------------------------------------------------------ report

    def snapshot(self) -> dict[str, Any]:
        """The report fields (SPEC 12A); numbers only."""
        with self._lock:
            return {
                "ended_by": dict(self._ended_by),
                "tentative_dropped": self._tentative_dropped,
                "expiry": {
                    "misses": summarise(list(self._misses), LEVELS),
                    "lost_s": summarise(list(self._lost_s), LEVELS),
                    "after_refused_low": self._after_refused,
                },
                "rebirths": {
                    "count": self._rebirths,
                    "overlapped": self._overlapped,
                    "window_s": self.cfg.rebirth_window_s,
                    "distance_heights_max": self.cfg.rebirth_distance_heights,
                    "gap_s": summarise(list(self._gap_s), LEVELS),
                    "lost_s": summarise(list(self._rebirth_lost_s), LEVELS),
                    "iou": summarise(list(self._iou), LEVELS),
                    "distance_heights": summarise(list(self._distance), LEVELS),
                    "head_yaw_change_deg": summarise(list(self._yaw_deg), LEVELS),
                    "head_pitch_change_deg": summarise(list(self._pitch_deg), LEVELS),
                },
                "low_score": {
                    "refused": self._refused,
                    "refused_score": summarise(list(self._refused_score), LEVELS),
                    "unmatched": self._unmatched,
                    "unmatched_score": summarise(list(self._unmatched_score), LEVELS),
                    "near_ended": self._near_ended,
                    "near_ended_score": summarise(list(self._near_ended_score), LEVELS),
                },
                "start_score": summarise(list(self._start_score), LEVELS),
                "start_near_iou": summarise(list(self._start_near_iou), LEVELS),
                "below_min_px": self._below_min_px,
            }

    # ----------------------------------------------------------------- helpers

    def _distance_heights(self, old_box: Box, new_box: Box) -> float:
        _, top, _, bottom = old_box
        height = bottom - top
        if height <= 0:
            return math.inf
        (ox, oy), (nx, ny) = _centre(old_box), _centre(new_box)
        return math.hypot(nx - ox, ny - oy) / height

    def _nearest(self, live: _Live, ended: list[_Ended]) -> _Ended | None:
        window, limit = self.cfg.rebirth_window_s, self.cfg.rebirth_distance_heights
        best, best_d = None, math.inf
        for old in ended:
            if live.birth_stamp <= old.last_stamp or live.birth_stamp - old.end_stamp > window:
                continue
            d = self._distance_heights(old.box, live.birth_box)
            if d <= limit and d < best_d:
                best, best_d = old, d
        return best

    def _nearest_born(self, old: _Ended, born: list[_Live]) -> _Live | None:
        limit = self.cfg.rebirth_distance_heights
        best, best_d = None, math.inf
        for live in born:
            d = self._distance_heights(live.neighbours.get(old.track_id, old.box), live.birth_box)
            if d <= limit and d < best_d:
                best, best_d = live, d
        return best

    def _record(self, old: _Ended, new: _Live, *, overlapped: bool) -> None:
        self._rebirths += 1
        self._overlapped += overlapped
        self._gap_s.append(new.birth_stamp - old.last_stamp)
        self._rebirth_lost_s.append(old.lost_s)
        old_box = new.neighbours.get(old.track_id, old.box) if overlapped else old.box
        self._iou.append(_iou(new.birth_box, old_box))
        self._distance.append(self._distance_heights(old_box, new.birth_box))
        old_pose = old.last_pose
        if old_pose is not None and new.birth_pose is not None:
            self._yaw_deg.append(abs(math.degrees(new.birth_pose[0] - old_pose[0])))
            self._pitch_deg.append(abs(math.degrees(new.birth_pose[1] - old_pose[1])))
