"""Detect-then-track of the shown object (SPEC 5, 13).

During a look at a shown thing (the governor's ``objects`` stage active, SPEC 5A) the object
detector (NanoDet) runs at ``vision.rates.objects_active_hz``, costs a lot on the CM4, and a
missed or weak frame breaks its track. ``ShownFollower`` follows the shown track between the
detector runs with a cheap single-object tracker (VitTrack, ``stages.SingleObjectTracker``);
the pipeline folds each tracker box into the object tracker's track (``Tracker.follow``) and
publishes it, so the world entry and the gaze stay on the thing.

The detector stays the source of truth. On each detector run of a look (``detector_run``):

- no target yet: the tracker starts on the run's shown track, on the detector's box (``seed``);
- the run saw the target: the tracker box, moved on to the detector frame at the velocity of
  its last two boxes (at most one tracker interval ahead, ``_predicted``: the box can be a
  tracker period older than the frame), is confirmed when it overlaps the detector box at
  ``vision.object_track.reseed_iou`` or more (the confirm clock restarts, the template stays),
  else the tracker starts again on the detector box (``reseed``: it drifted, or the thing is
  elsewhere);
- the run saw the target but attention now follows another track: the target is dropped
  (``switched``) and the tracker starts on that one;
- the run missed the target but its shown track is the followed thing on a new track (the
  same class, or a box on the tracker's at ``reseed_iou``: a cup read as a bowl): the tracker
  starts again on that one (``reseed``);
- the run missed the target otherwise (not detected, too weak, or outside a crop): the tracker
  keeps it (``bridged``); a detector miss alone never ends the follow, and a thing of another
  class away from the tracker box (a couch behind the cup) never takes it over.

Between runs (``step``, at most ``vision.rates.object_track_hz`` under the governor's scales,
only on frames the detector does not take) the tracker updates its box. The follow ends when
its score falls below ``score_min`` (``score``), when its box is no longer sane (``box``: its
centre left the frame, or its area is off by more than ``box_area_ratio_max`` from the last
detector box it was started on or confirmed by), when the last detector confirm is more than
``confirm_max_s`` old (``timeout``), when the backend raises (``error``, logged by type), and,
from the pipeline, when the look ends (``episode_end``), the track ends (``track_ended``), the
switch goes off (``disabled``), the robot falls asleep (``asleep``), the governor's scale is 0
(``paused``) or the pipeline resets (``reset``). A lost box is never published or folded in.

The debug report (SPEC 12A, ``roboemot.vision`` ``object_track``) gives the state and the
counts: ``seeds``, ``reseeds``, ``updates``, ``confirms``, ``bridged`` and ``lost`` by reason.
Numbers only: the tracker's template is a crop held inside the backend, never logged (SPEC 12).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from ..config.schema import ObjectTrackConfig
from .stage_timing import OBJECT_TRACK, OBJECT_TRACK_INIT, StageTimer
from .stages import SingleObjectTracker
from .tracker import iou_matrix
from .types import Box

log = logging.getLogger(__name__)

SCORE, BOX, TIMEOUT, ERROR = "score", "box", "timeout", "error"
EPISODE_END, TRACK_ENDED, SWITCHED = "episode_end", "track_ended", "switched"
DISABLED, ASLEEP, PAUSED, RESET = "disabled", "asleep", "paused", "reset"
LOSSES = (
    SCORE,
    BOX,
    TIMEOUT,
    ERROR,
    EPISODE_END,
    TRACK_ENDED,
    SWITCHED,
    DISABLED,
    ASLEEP,
    PAUSED,
    RESET,
)
COUNTS = ("seeds", "reseeds", "updates", "confirms", "bridged")
OFF, UNAVAILABLE, IDLE, FOLLOWING = "off", "unavailable", "idle", "following"
SCORE_DIGITS = 2  # decimals of the last tracker score in the report
Seen = tuple[str, Box]  # a track a detector run saw: its class and the detector's box


def box_iou(a: Box, b: Box) -> float:
    """IoU of two xyxy boxes."""
    return float(iou_matrix(np.array([a]), np.array([b]))[0, 0])


class ShownFollower:
    """Follows the shown track between object detector runs; see the module docstring.

    ``tracker`` is the backend (None when it is unavailable, ``unavailable`` says why). ``cfg``
    is read on every call, so a hot reload applies at once. Runs on the vision thread, except
    ``stop`` (also the app thread's ``clear_overlay``) and ``snapshot`` (the report thread):
    the follow state is read and written under one lock, and a stop during a backend update
    voids that update's box (``_generation`` moves on every start and stop)."""

    def __init__(
        self,
        tracker: SingleObjectTracker | None,
        cfg: ObjectTrackConfig,
        timer: StageTimer,
        unavailable: str | None = None,
    ):
        self.tracker = tracker
        self.cfg = cfg
        self.timer = timer
        self.unavailable = unavailable
        self._lock = threading.RLock()
        self._target: int | None = None
        self._generation = 0
        self._label = ""
        self._box: Box | None = None  # the tracker's last box
        self._box_stamp = 0.0
        self._prev: tuple[Box, float] | None = None  # the box before it, for the velocity
        self._ref: Box | None = None  # the detector box it was started on or confirmed by
        self._confirmed_at = 0.0
        self._score: float | None = None
        self._counts = dict.fromkeys(COUNTS, 0)
        self._lost = dict.fromkeys(LOSSES, 0)

    @property
    def target(self) -> int | None:
        """The followed track's ID, None while nothing is followed."""
        with self._lock:
            return self._target

    @property
    def usable(self) -> bool:
        """Whether a follow may start: a backend and the switch on."""
        return self.tracker is not None and self.cfg.enabled

    def detector_run(
        self, image: np.ndarray, stamp: float, seen: dict[int, Seen], shown: int | None
    ) -> None:
        """Fold one detector run of a look into the follow. ``seen`` maps each confirmed track
        the run saw to its class and detector box, ``shown`` is the run's shown track (or
        None)."""
        target = self.target
        if not self.usable:
            if target is not None:
                self.stop(DISABLED)
            return
        if target is not None and target in seen:
            if shown is None or shown == target:
                self._check(image, stamp, target, seen[target][1])
                return
            self.stop(SWITCHED)
        elif target is not None:
            if shown is not None and shown in seen and self._takes_over(seen[shown], stamp):
                self._start(image, stamp, shown, seen[shown], "reseeds")
                return
            self._count("bridged")
            return
        if shown is not None and shown in seen:
            self._start(image, stamp, shown, seen[shown], "seeds")

    def _takes_over(self, other: Seen, stamp: float) -> bool:
        """Whether the run's shown track, while the followed one was missed, is the followed
        thing on a new track: the same class, or a box on the tracker's (a cup read as a
        bowl)."""
        label, box = other
        if label == self._label:
            return True
        predicted = self._predicted(stamp)
        return predicted is not None and box_iou(predicted, box) >= self.cfg.reseed_iou

    def _predicted(self, stamp: float) -> Box | None:
        """The tracker box moved on to ``stamp`` at the velocity of its last two boxes, at
        most one tracker interval ahead; the last box as it is without two."""
        with self._lock:
            box, prev, at = self._box, self._prev, self._box_stamp
        if box is None or prev is None:
            return box
        (px1, py1, px2, py2), before = prev
        interval = at - before
        if interval <= 0:
            return box
        ahead = min(max(stamp - at, 0.0), interval) / interval
        x1, y1, x2, y2 = box
        dx = ((x1 + x2) - (px1 + px2)) / 2 * ahead
        dy = ((y1 + y2) - (py1 + py2)) / 2 * ahead
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _check(self, image: np.ndarray, stamp: float, target: int, box: Box) -> None:
        """Confirm the tracker box, moved on to the detector frame, against the detector box,
        or start again on the latter."""
        predicted = self._predicted(stamp)
        if predicted is None:
            return
        if box_iou(predicted, box) >= self.cfg.reseed_iou:
            with self._lock:
                self._confirmed_at, self._ref = stamp, box
                self._counts["confirms"] += 1
            return
        self._start(image, stamp, target, (self._label, box), "reseeds")

    def _start(
        self, image: np.ndarray, stamp: float, track_id: int, seen: Seen, count: str
    ) -> None:
        """Start the tracker on a detector box (timed as ``object_track_init``). A backend that
        raises ends any follow and counts one ``error``."""
        if self.tracker is None:
            return
        label, box = seen
        wall0, cpu0 = self.timer.start()
        try:
            self.tracker.init(image, box)
        except Exception as exc:
            log.warning("shown-object tracker failed to start: %s", type(exc).__name__)
            with self._lock:
                self._target, self._box, self._prev = None, None, None
                self._generation += 1
                self._lost[ERROR] += 1
            return
        finally:
            self.timer.record(OBJECT_TRACK_INIT, wall0, cpu0)
        with self._lock:
            self._target, self._label, self._generation = track_id, label, self._generation + 1
            self._box, self._box_stamp, self._prev, self._ref = box, stamp, None, box
            self._confirmed_at = stamp
            self._counts[count] += 1

    def step(self, image: np.ndarray, stamp: float) -> tuple[Box, float] | None:
        """One tracker update between detector runs: the box and the tracker's score, or None
        when the follow ended (no target, the confirm timed out, the score fell, the box is
        not sane, the backend raised, or a stop came in meanwhile)."""
        with self._lock:
            target, generation, confirmed = self._target, self._generation, self._confirmed_at
        if target is None or self.tracker is None:
            return None
        if stamp - confirmed > self.cfg.confirm_max_s:
            self._lose(generation, TIMEOUT)
            return None
        wall0, cpu0 = self.timer.start()
        try:
            box, score = self.tracker.update(image)
        except Exception as exc:
            log.warning("shown-object tracker failed: %s", type(exc).__name__)
            self._lose(generation, ERROR)
            return None
        finally:
            self.timer.record(OBJECT_TRACK, wall0, cpu0)
        with self._lock:
            if self._generation != generation:
                return None
            self._score = score
        if score < self.cfg.score_min:
            self._lose(generation, SCORE)
            return None
        if not self._sane(box, image.shape[1], image.shape[0]):
            self._lose(generation, BOX)
            return None
        with self._lock:
            if self._generation != generation:
                return None
            if self._box is not None:
                self._prev = (self._box, self._box_stamp)
            self._box, self._box_stamp = box, stamp
            self._counts["updates"] += 1
        return box, score

    def _sane(self, box: Box, width: int, height: int) -> bool:
        """Whether a tracker box may stand for the thing: its centre inside the frame and its
        area within ``box_area_ratio_max`` of the last detector box it was started on or
        confirmed by, either way."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (0 <= cx <= width and 0 <= cy <= height):
            return False
        with self._lock:
            ref = self._ref
        if ref is None:
            return True
        rx1, ry1, rx2, ry2 = ref
        area = (x2 - x1) * (y2 - y1)
        ref_area = (rx2 - rx1) * (ry2 - ry1)
        if area <= 0 or ref_area <= 0:
            return False
        limit = self.cfg.box_area_ratio_max
        return 1 / limit <= area / ref_area <= limit

    def current(self, target: int) -> bool:
        """Whether ``target`` is still the followed track (a stop may come from another
        thread between a step and its publish)."""
        return self.target == target

    def _lose(self, generation: int, reason: str) -> None:
        """End the follow that ``generation`` names, unless a start or stop replaced it."""
        with self._lock:
            if self._generation == generation:
                self.stop(reason)

    def stop(self, reason: str) -> None:
        """End the follow (counted under ``reason`` when one was running); any thread."""
        with self._lock:
            if self._target is None:
                return
            self._target, self._box, self._prev = None, None, None
            self._generation += 1
            self._lost[reason] += 1

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def snapshot(self) -> dict[str, Any]:
        """The ``object_track`` report entry: state, reason, counts, the last tracker score."""
        with self._lock:
            following = self._target is not None
            score = self._score
            counts, lost = dict(self._counts), dict(self._lost)
        if self.tracker is None:
            state = UNAVAILABLE if self.unavailable is not None else OFF
        elif not self.cfg.enabled:
            state = OFF
        else:
            state = FOLLOWING if following else IDLE
        return {
            "state": state,
            "reason": self.unavailable if state == UNAVAILABLE else None,
            **counts,
            "lost": lost,
            "score": None if score is None else round(score, SCORE_DIGITS),
        }
