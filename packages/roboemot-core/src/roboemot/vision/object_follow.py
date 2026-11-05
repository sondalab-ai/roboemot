"""Detect-then-track of the shown object (SPEC 5, 13).

During a look at a shown thing (the governor's ``objects`` stage active, SPEC 5A) the object
detector (NanoDet) runs at ``vision.rates.objects_active_hz``, costs a lot on the CM4, and a
missed or weak frame breaks its track. ``ShownFollower`` follows the shown track between the
detector runs with a cheap single-object tracker (VitTrack, ``stages.SingleObjectTracker``);
the pipeline folds each tracker box into the object tracker's track (``Tracker.follow``) and
publishes it, so the world entry and the gaze stay on the thing.

The detector stays the source of truth. On each detector run of a look (``detector_run``):

- no target yet: the tracker starts on the run's shown track, on the detector's box (``seed``);
- the run saw the target: the tracker box is confirmed when it overlaps the detector box at
  ``vision.object_track.reseed_iou`` or more (the confirm clock restarts, the template stays),
  else the tracker starts again on the detector box (``reseed``: it drifted, or the thing is
  elsewhere);
- the run saw the target but attention now follows another track: the target is dropped
  (``switched``) and the tracker starts on that one;
- the run missed the target but its shown track is another one of the same class: the thing
  got a new track (the tracker's box, folded into the old track, drifted too far for the
  detection to match it), so the tracker starts again on that one (``reseed``);
- the run missed the target otherwise (not detected, too weak, or outside a crop): the tracker
  keeps it (``bridged``); a detector miss alone never ends the follow, and a thing of another
  class nearer the attention target (a couch behind the cup) never takes it over.

Between runs (``step``, at most ``vision.rates.object_track_hz`` under the governor's scales,
only on frames the detector does not take) the tracker updates its box. The follow ends when
its score falls below ``score_min`` (``score``), when the last detector confirm is more than
``confirm_max_s`` old (``timeout``), when the backend raises (``error``, logged by type), and,
from the pipeline, when the look ends (``episode_end``), the track ends (``track_ended``), the
switch goes off (``disabled``), the robot falls asleep (``asleep``) or the pipeline resets
(``reset``). A lost box is never published or folded in.

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
from .stage_timing import OBJECT_TRACK, StageTimer
from .stages import SingleObjectTracker
from .tracker import iou_matrix
from .types import Box

log = logging.getLogger(__name__)

SCORE, TIMEOUT, ERROR = "score", "timeout", "error"
EPISODE_END, TRACK_ENDED, SWITCHED = "episode_end", "track_ended", "switched"
DISABLED, ASLEEP, RESET = "disabled", "asleep", "reset"
LOSSES = (SCORE, TIMEOUT, ERROR, EPISODE_END, TRACK_ENDED, SWITCHED, DISABLED, ASLEEP, RESET)
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
    is read on every call, so a hot reload applies at once. Runs on the vision thread; the
    report thread reads ``snapshot`` under the lock."""

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
        self.target: int | None = None
        self._label = ""
        self._box: Box | None = None
        self._confirmed_at = 0.0
        self._score: float | None = None
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(COUNTS, 0)
        self._lost = dict.fromkeys(LOSSES, 0)

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
        if not self.usable:
            if self.target is not None:
                self.stop(DISABLED)
            return
        target = self.target
        if target is not None and target in seen:
            if shown is None or shown == target:
                self._check(image, stamp, seen[target][1])
                return
            self.stop(SWITCHED)
        elif target is not None:
            if shown is not None and shown in seen and seen[shown][0] == self._label:
                self._start(image, stamp, shown, seen[shown], "reseeds")
                return
            self._count("bridged")
            return
        if shown is not None and shown in seen:
            self._start(image, stamp, shown, seen[shown], "seeds")

    def _check(self, image: np.ndarray, stamp: float, box: Box) -> None:
        """Confirm the tracker box against the detector box, or start again on the latter."""
        assert self._box is not None and self.target is not None
        if box_iou(self._box, box) >= self.cfg.reseed_iou:
            self._confirmed_at = stamp
            self._count("confirms")
            return
        self._start(image, stamp, self.target, (self._label, box), "reseeds")

    def _start(
        self, image: np.ndarray, stamp: float, track_id: int, seen: Seen, count: str
    ) -> None:
        assert self.tracker is not None
        label, box = seen
        try:
            self.tracker.init(image, box)
        except Exception as exc:
            log.warning("shown-object tracker failed to start: %s", type(exc).__name__)
            self.target = track_id  # so the loss is counted against a follow
            self.stop(ERROR)
            return
        self.target, self._label, self._box, self._confirmed_at = track_id, label, box, stamp
        self._count(count)

    def step(self, image: np.ndarray, stamp: float) -> Box | None:
        """One tracker update between detector runs: the box, or None when the follow ended
        (no target, the confirm timed out, the score fell, the backend raised)."""
        if self.target is None or self.tracker is None:
            return None
        if stamp - self._confirmed_at > self.cfg.confirm_max_s:
            self.stop(TIMEOUT)
            return None
        wall0, cpu0 = self.timer.start()
        try:
            box, score = self.tracker.update(image)
        except Exception as exc:
            log.warning("shown-object tracker failed: %s", type(exc).__name__)
            self.stop(ERROR)
            return None
        finally:
            self.timer.record(OBJECT_TRACK, wall0, cpu0)
        with self._lock:
            self._score = score
        if score < self.cfg.score_min:
            self.stop(SCORE)
            return None
        self._box = box
        self._count("updates")
        return box

    def stop(self, reason: str) -> None:
        """End the follow (counted under ``reason`` when one was running)."""
        if self.target is None:
            return
        self.target, self._box = None, None
        with self._lock:
            self._lost[reason] += 1

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def snapshot(self) -> dict[str, Any]:
        """The ``object_track`` report entry: state, reason, counts, the last tracker score."""
        if self.tracker is None:
            state = UNAVAILABLE if self.unavailable is not None else OFF
        elif not self.cfg.enabled:
            state = OFF
        else:
            state = FOLLOWING if self.target is not None else IDLE
        with self._lock:
            score = self._score
            out: dict[str, Any] = {
                "state": state,
                "reason": self.unavailable if state == UNAVAILABLE else None,
                **self._counts,
                "lost": dict(self._lost),
            }
        out["score"] = None if score is None else round(score, SCORE_DIGITS)
        return out
