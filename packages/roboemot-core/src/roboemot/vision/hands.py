"""The hand check and the fist rule (SPEC 5, 12, 12A, 13).

``HandWatch`` runs a hand check on one captured frame per ``1 / vision.hands.rate_hz`` (capture
time), only while its caller asks for it (``wanted``; the pipeline's ``hands_wanted``, which the
radio drives). A check is the MediaPipe palm detector on the whole frame and, only when its best
palm scores at least ``vision.hands.palm_score_min``, the 21-point hand model on that palm
(``stages.PalmDetector``, ``stages.HandPoseEstimator``). The fist rule (:func:`classify`) turns
the points into ``fist``, ``open`` or ``unknown``; :class:`FistHold` turns consecutive fist
checks held for ``vision.hands.fist_hold_s`` into one :class:`HandGesture`, which the pipeline
publishes on ``percept.hand``.

**Thread.** In the live app the checks run on their own low-priority thread
(``roboemot-hands``, lowered like the embed worker by ``runtime.background_nice``): a check
costs about 136 ms of CPU on the CM4 without a palm and about 283 ms with one (int8 palm
detector plus fp32 hand model, ``scripts/bench/hands_cm4.py``), which inline would hold up the
8 Hz face detector and the object stage on the vision thread. The vision thread hands over a
copy of the frame (the only one) and never waits; a frame due while a check still runs is
skipped (``busy``). The results come back through :meth:`HandWatch.step` on the vision thread,
which applies the hold rule in check order, so the hold state lives on one thread.
Without a thread (replay, unit tests) the check runs inline in ``step``, so a replay stays
deterministic (SPEC 14).

**Seam.** ``step(stamp, image, wanted)``: ``wanted`` false costs one flag check. Going from
wanted to not wanted (or a ``reset``) starts a new generation: the frame waiting for the thread
is dropped, a result of the old generation is discarded (``discarded``) and the hold starts over.
After ``stop`` nothing is queued any more. ``step`` returns each event with the capture stamp of
the check that made it.

**Preview (SPEC 11A).** While the web UI's live preview is watched (:meth:`HandWatch.preview`
was called within the preview window, one check period plus ``vision.hands.preview_slack_s``,
capture time: the runner calls the preview overlay hook only then), a check that classifies a
fist also returns its palm box, and the watch keeps it with the hold so far as a
:class:`HandMark` for the preview overlay. A result that is not a fist removes it, and so does
the preview window (capture time from the frame the result was applied on) without a new
result, the end of the watch, ``clear_mark`` (sleep, from any thread) and ``reset``. A
``clear_mark`` bumps an epoch under the lock; ``step`` and ``preview`` capture it at entry and
write the kept fist or the watch only while it is unchanged, so a frame in flight during a sleep
cannot bring the box back.

**Privacy (SPEC 12).** The frame copy lives only in the waiting or running job; palm boxes,
keypoints and hand points live only inside one check, except the palm box of a fist while the
preview is watched, which goes only to the preview overlay. A result holds the check's capture
stamp and its class (and that box, hidden from repr), nothing else from the image; the event
holds the fact and the hold time. Failures are logged by exception type only. The report has
counts only.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .. import priority, shutdown
from ..config.schema import HandsConfig
from .governor import EPSILON_S
from .stage_timing import HAND_PALM, HAND_POSE, StageTimer
from .stages import HandPoseEstimator, PalmDetector
from .types import Box, HandMark

log = logging.getLogger(__name__)

THREAD_NAME = "roboemot-hands"
JOIN_TIMEOUT_S = 10.0
FIST = "fist"
OPEN = "open"
UNKNOWN = "unknown"
NO_PALM = "no_palm"
HandClass = Literal["fist", "open", "unknown"]
CheckResult = Literal["fist", "open", "unknown", "no_palm", "failed"]

# MediaPipe hand points (21): the wrist, then four per finger from the thumb, base to tip. The
# rule uses the wrist, the middle joint (PIP) and the tip of the four long fingers; the thumb
# folds across the palm in a fist and points anywhere, so it is left out.
WRIST = 0
FINGER_PIPS = (6, 10, 14, 18)  # index, middle, ring, little
FINGER_TIPS = (8, 12, 16, 20)
HAND_POINTS = 21
POINT_DIMS = 3  # x, y, z
MIN_REACH = 1e-9  # a PIP on the wrist: degenerate points, no class

COUNTS = (
    "checks",
    "palms",
    "pose_runs",
    "fists",
    "opens",
    "unknowns",
    "events",
    "busy",
    "failed",
    "discarded",
)
Clock = Callable[[], float]


def finger_ratios(points: np.ndarray) -> np.ndarray:
    """Per long finger (index, middle, ring, little), the tip's distance from the wrist over the
    PIP's distance from the wrist, on the 3D points. Distances are kept by any rotation and a
    ratio of two drops the scale, so the rule does not depend on how the hand is turned, how
    big it is or how far away."""
    wrist = points[WRIST]
    tips = np.linalg.norm(points[list(FINGER_TIPS)] - wrist, axis=1)
    pips = np.linalg.norm(points[list(FINGER_PIPS)] - wrist, axis=1)
    return tips / np.maximum(pips, MIN_REACH)


def classify(points: np.ndarray, presence: float, cfg: HandsConfig) -> HandClass:
    """The fist rule (SPEC 5): ``fist`` when the hand model is sure of a hand (``presence`` at
    least ``presence_min``) and every long finger's ratio (:func:`finger_ratios`) is below
    ``fist_ratio_max`` (each tip curled back closer to the wrist than its PIP); ``open`` when
    every ratio is above ``open_ratio_min`` (each finger straight out). Anything between (a
    half-closed hand, some fingers curled), an unsure model or points that are not 21 finite
    3D points is ``unknown``."""
    if presence < cfg.presence_min or points.shape != (HAND_POINTS, POINT_DIMS):
        return UNKNOWN
    if not np.all(np.isfinite(points)):
        return UNKNOWN
    wrist = points[WRIST]
    if np.min(np.linalg.norm(points[list(FINGER_PIPS)] - wrist, axis=1)) < MIN_REACH:
        return UNKNOWN
    ratios = finger_ratios(points)
    if np.all(ratios < cfg.fist_ratio_max):
        return FIST
    if np.all(ratios > cfg.open_ratio_min):
        return OPEN
    return UNKNOWN


class FistHold:
    """Fist checks in a row held for ``hold_s`` make one event; see :meth:`observe`."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._last: float | None = None
        self._fired = False

    def observe(self, stamp: float, fist: bool, hold_s: float, gap_s: float) -> float | None:
        """One check at ``stamp``. Returns the hold time (capture time from the first fist check
        of the run to this one) when the run reaches ``hold_s`` now, else None. A check that is
        not a fist (open, unknown, no palm, a failure) ends the run and re-arms; a run that
        fired fires again only after such a check. Two fist checks further apart than ``gap_s``
        (the watch passes one check period plus ``vision.hands.max_gap_s``: the camera paused,
        the robot asleep) are not one hold: the run starts again at the later one, still without
        re-arming."""
        if not fist:
            self.clear()
            return None
        if self._since is None or (self._last is not None and stamp - self._last > gap_s):
            self._since = stamp
        self._last = stamp
        held = stamp - self._since
        if self._fired or held < hold_s - EPSILON_S:
            return None
        self._fired = True
        return held

    def clear(self) -> None:
        self._since, self._last, self._fired = None, None, False

    def progress(self) -> tuple[float, bool]:
        """The run so far: capture time from its first fist check to its last (0 without a
        run) and whether it fired."""
        if self._since is None or self._last is None:
            return 0.0, self._fired
        return self._last - self._since, self._fired


@dataclass(frozen=True)
class HandGesture:
    """Payload of ``percept.hand``: the gesture and how long it was held (capture time). No
    image, box or point (SPEC 12)."""

    gesture: Literal["fist"]
    held_s: float


@dataclass(frozen=True)
class _Job:
    image: np.ndarray = field(repr=False)
    stamp: float
    generation: int
    boxed: bool = False  # the preview is watched: a fist returns its palm box


@dataclass(frozen=True)
class _Outcome:
    stamp: float
    generation: int
    result: CheckResult
    box: Box | None = field(default=None, repr=False)  # a fist's palm box, only when boxed


class HandWatch:
    """The hand check, its thread and its hold rule; see the module docstring.

    ``palm`` and ``pose`` are None when the hand stage is off or did not load (``unavailable``
    says why); the watch then never checks. ``cfg`` is replaced on hot reload. With
    ``threaded`` the checks run on :meth:`start`'s thread, else inline in :meth:`step`.
    """

    def __init__(
        self,
        cfg: HandsConfig,
        palm: PalmDetector | None = None,
        pose: HandPoseEstimator | None = None,
        timer: StageTimer | None = None,
        *,
        unavailable: str | None = None,
        threaded: bool = False,
        join_timeout_s: float = JOIN_TIMEOUT_S,
    ):
        self.cfg = cfg
        self.palm = palm
        self.pose = pose
        self.timer = timer
        self.unavailable = unavailable
        self.threaded = threaded
        self._join_timeout_s = join_timeout_s
        self._cond = threading.Condition()
        self._job: _Job | None = None
        self._busy = False
        self._done: list[_Outcome] = []
        self._stop = False
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._wanted = False
        self._last_check: float | None = None
        self._last_result: float | None = None
        self._hold = FistHold()
        self._counts = dict.fromkeys(COUNTS, 0)
        self._preview_at: float | None = None
        self._mark: tuple[HandMark, float] | None = None  # with the stamp it was applied at
        self._mark_epoch = 0  # bumped by ``clear_mark``; writes from an older epoch are dropped

    @property
    def ready(self) -> bool:
        """Whether the models are there and the switch is on."""
        return self.palm is not None and self.pose is not None and self.cfg.enabled

    def step(
        self, stamp: float, image: np.ndarray, wanted: bool
    ) -> list[tuple[float, HandGesture]]:
        """One captured frame at ``stamp`` (vision thread). Applies the finished checks in order
        and returns the events they made, each with its check's capture stamp; then, while
        ``wanted``, starts a check on ``image`` when one is due (a copy goes to the thread, or it
        runs inline here). After ``stop`` nothing is queued. The kept fist (``preview``) is
        dropped here once the preview is no longer watched or it lapsed."""
        epoch = self._epoch()
        boxed = self._watched(stamp)
        if not boxed or self._lapsed(self._mark, stamp):
            self._mark = None
        wanted = wanted and self.ready
        if not wanted and not self._wanted:
            return []
        if wanted != self._wanted:
            self._new_generation()
            self._wanted = wanted
        events = self._apply(self._drain(), stamp, boxed, epoch)
        if not wanted or not self._due(stamp):
            return events
        if not self.threaded:
            self._last_check = stamp
            outcome = self._check(_Job(image, stamp, self._generation, boxed))
            return events + self._apply([outcome], stamp, boxed, epoch)
        with self._cond:
            if self._stop:
                return events
            if self._busy or self._job is not None:
                self._counts["busy"] += 1
                return events
            self._last_check = stamp
            self._job = _Job(image.copy(), stamp, self._generation, boxed)
            self._cond.notify()
        return events

    def _window(self) -> float:
        """The preview window: one check period plus ``preview_slack_s`` (capture time)."""
        cfg = self.cfg
        return 1 / cfg.rate_hz + cfg.preview_slack_s + EPSILON_S

    def _epoch(self) -> int:
        with self._cond:
            return self._mark_epoch

    def _watched(self, stamp: float) -> bool:
        """Whether the preview asked (``preview``) within the preview window before ``stamp``."""
        at = self._preview_at
        return at is not None and stamp - at <= self._window()

    def _lapsed(self, mark: tuple[HandMark, float] | None, stamp: float) -> bool:
        """Whether the kept fist ``mark`` is older than the preview window at ``stamp`` (none
        counts as lapsed)."""
        return mark is None or stamp - mark[1] > self._window()

    def preview(self, stamp: float) -> HandMark | None:
        """The preview overlay hook's read for the frame at ``stamp`` (vision thread, called
        only while a viewer watches): marks the preview as watched, so the next checks return
        a fist's palm box, and returns the kept fist unless it lapsed. A ``clear_mark`` since
        entry wins: nothing is marked and nothing returned."""
        epoch = self._epoch()
        mark = self._mark  # one read: ``clear_mark`` may run on another thread (sleep)
        with self._cond:
            if epoch != self._mark_epoch:
                return None
            self._preview_at = stamp
        if mark is None or self._lapsed(mark, stamp):
            return None
        return mark[0]

    def clear_mark(self) -> None:
        """Forget the kept fist and that the preview is watched (the robot fell asleep; any
        thread). Bumps the epoch, so a ``step`` or ``preview`` already running writes neither
        back."""
        with self._cond:
            self._mark_epoch += 1
            self._mark = None
            self._preview_at = None

    def _keep_mark(self, mark: HandMark, stamp: float, epoch: int) -> None:
        """Keep ``mark`` applied at ``stamp`` unless ``clear_mark`` ran since ``epoch``."""
        with self._cond:
            if epoch == self._mark_epoch:
                self._mark = (mark, stamp)

    def _due(self, stamp: float) -> bool:
        last = self._last_check
        return last is None or stamp - last >= 1 / self.cfg.rate_hz - EPSILON_S

    def _new_generation(self) -> None:
        with self._cond:
            self._generation += 1
            self._job = None  # a result in flight or not drained yet is discarded by _apply
        self._hold.clear()
        self._last_check = None

    def _drain(self) -> list[_Outcome]:
        with self._cond:
            done, self._done = self._done, []
        return done

    def _apply(
        self, outcomes: list[_Outcome], stamp: float, boxed: bool, epoch: int
    ) -> list[tuple[float, HandGesture]]:
        """The hold rule over finished checks, in check order (vision thread); ``cfg`` is read
        at each check, so a hot reload of the rate or the hold applies to the run in progress.
        While the preview is watched (``boxed``) a fist with its box is kept for it, applied at
        ``stamp`` (the frame it comes back on) unless ``clear_mark`` ran since ``epoch``; any
        other result removes it."""
        events = []
        for o in sorted(outcomes, key=lambda o: o.stamp):
            if o.generation != self._generation:
                self._count("discarded")
                continue
            self._last_result = o.stamp
            cfg = self.cfg
            gap = 1 / cfg.rate_hz + cfg.max_gap_s
            held = self._hold.observe(o.stamp, o.result == FIST, cfg.fist_hold_s, gap)
            if held is not None:
                self._count("events")
                events.append((o.stamp, HandGesture(FIST, held)))
            self._mark = None
            if boxed and o.result == FIST and o.box is not None:
                so_far, fired = self._hold.progress()
                self._keep_mark(HandMark(o.box, so_far, cfg.fist_hold_s, fired), stamp, epoch)
        return events

    def _count(self, key: str) -> None:
        with self._cond:
            self._counts[key] += 1

    def _check(self, job: _Job) -> _Outcome:
        """Palm detector on the frame; the hand model and the fist rule only on a palm at or
        above ``palm_score_min``. A backend that raises makes a ``failed`` check (logged by
        type), which also ends a fist run."""
        result: CheckResult
        box = None
        try:
            result, box = self._classify(job.image)
        except Exception as exc:
            log.warning("hand check failed: %s", type(exc).__name__)
            result = "failed"
        counted = {FIST: "fists", OPEN: "opens", UNKNOWN: "unknowns", "failed": "failed"}
        with self._cond:
            self._counts["checks"] += 1
            if result in counted:
                self._counts[counted[result]] += 1
        kept = box if job.boxed and result == FIST else None
        return _Outcome(job.stamp, job.generation, result, kept)

    def _classify(self, image: np.ndarray) -> tuple[CheckResult, Box | None]:
        """The check's class and the box of the palm the hand model ran on (None without)."""
        palm_detector, pose_estimator, cfg = self.palm, self.pose, self.cfg
        assert palm_detector is not None and pose_estimator is not None
        with self._timed(HAND_PALM):
            palm = palm_detector.best(image)
        if palm is None or palm.score < cfg.palm_score_min:
            return NO_PALM, None
        self._count("palms")
        with self._timed(HAND_POSE):
            hand = pose_estimator.estimate(image, palm)
        self._count("pose_runs")
        return classify(hand.points, hand.presence, cfg), palm.box

    def _timed(self, stage: str) -> _Timed:
        return _Timed(self.timer, stage)

    def run_one(self, wait_s: float | None = 0.0) -> bool:
        """Run the waiting check on the calling thread, waiting up to ``wait_s`` for one (None:
        until one comes or ``stop``); returns whether one ran. Tests drive it directly."""
        with self._cond:
            if wait_s is None:
                while self._job is None and not self._stop:
                    self._cond.wait()
            elif self._job is None and wait_s > 0:
                self._cond.wait(wait_s)
            job = self._job
            if job is None or self._stop:
                return False
            self._job = None
            self._busy = True
        outcome = _Outcome(job.stamp, job.generation, "failed")
        try:
            outcome = self._check(job)
        finally:
            del job  # the frame copy goes with the run
            with self._cond:
                self._done.append(outcome)
                self._busy = False
                self._cond.notify_all()
        return True

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait until no check waits or runs (tests; the vision thread never waits)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._job is None and not self._busy, timeout_s)

    def start(self) -> None:
        """Start the thread (threaded watches with models only); a thread left from a ``stop``
        that timed out is kept, never doubled."""
        if not self.threaded or self.palm is None or self.pose is None:
            return
        if priority.left_out(THREAD_NAME):
            log.warning(
                "%s is not in runtime.background_threads: the hand check runs at normal priority",
                THREAD_NAME,
            )
        with self._cond:
            self._stop = False
            old = self._thread
            if old is not None and old.is_alive():
                log.warning("hand check thread still running from before: not started twice")
                return
            thread = threading.Thread(target=self._loop, name=THREAD_NAME, daemon=True)
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the thread, dropping a waiting frame, and wait for a check in flight, bounded
        by ``join_timeout_s``."""
        with self._cond:
            self._stop = True
            self._job = None
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            shutdown.join(thread, self._join_timeout_s)
            if thread.is_alive():
                log.warning("hand check thread did not stop within %.0f s", self._join_timeout_s)

    def _loop(self) -> None:
        priority.lower_current()
        while True:
            with self._cond:
                if self._stop:
                    return
            try:
                self.run_one(None)
            except Exception as exc:  # _check catches backend errors; belt and braces
                log.warning("hand check loop failed: %s", type(exc).__name__)

    def reset(self) -> None:
        """Forget the waiting frame, the hold and the counts (new session or replay start)."""
        self._new_generation()
        self._wanted = False
        self._last_result = None
        self.clear_mark()
        with self._cond:
            self._counts = dict.fromkeys(COUNTS, 0)

    def snapshot(self, now: float | None) -> dict[str, Any]:
        """For the debug report (SPEC 12A): the state, the counts since start and
        ``last_result_age_s``, the capture time from the last finished check to ``now`` (the
        last processed frame; null before a check). The state stays ``watching`` while no frame
        comes (the camera paused): the age then stops growing with it. Numbers only."""
        with self._cond:
            counts = dict(self._counts)
        if self.unavailable is not None:
            state = "unavailable"
        elif not self.ready:
            state = "off"
        else:
            state = "watching" if self._wanted else "idle"
        last = self._last_result
        age = None if last is None or now is None else max(0.0, now - last)
        out: dict[str, Any] = {"state": state, **counts, "last_result_age_s": age}
        if self.unavailable is not None:
            out["reason"] = self.unavailable
        return out


class _Timed:
    """Records one run of ``stage`` on the timer, also when it raises."""

    def __init__(self, timer: StageTimer | None, stage: str):
        self.timer, self.stage = timer, stage
        self.start = (0.0, 0.0)

    def __enter__(self) -> None:
        if self.timer is not None:
            self.start = self.timer.start()

    def __exit__(self, *exc: object) -> None:
        if self.timer is not None:
            self.timer.record(self.stage, *self.start)
