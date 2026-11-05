"""Animal and object appearance embeddings off the vision thread (SPEC 5, 12, 12A, 13).

DINOv2-S takes about 1.2 s per run on the CM4; inline it held up face detection, object
detection and tracking for that long. ``EmbedWorker`` runs it on its own low-priority thread
(``roboemot-embed``, lowered like the other background threads by ``runtime.background_nice``).
The vision thread decides when a track is due, cuts the crop (``cut``: a copy of the box, never
the frame) and hands over an ``EmbedJob``; the result comes back through ``drain``, which the
vision thread calls at the start of each ``process`` and applies there, with the job's capture
stamp, so every per-track state stays on one thread.

The queue keeps only the latest job per track: a newer job for a track replaces a waiting one
(``replaced``). It holds at most ``vision.embed_queue_max`` jobs; a job that would pass that
drops the oldest waiting one (``dropped_cap``: the fresher crop wins), and the vision thread
learns of it through ``drain`` so the track can be queued again. A job for a track that ended
is withdrawn (``cancel``, ``clear``) or its result discarded (``dropped_ended``). Crops and
embeddings live only in the job and the result; nothing is logged but exception type names.

``run_one`` is the thread's step; tests drive it directly. Counters and the queue wait (wall
time from ``submit`` to the start of the run) are numbers for the debug report.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .. import priority, shutdown
from ..stats import P50, P90, quantile
from .runner import MAX_RATE_HZ
from .stage_timing import APPEARANCE, MS_PER_S, StageTimer
from .stages import AppearanceEmbedder
from .types import Box

log = logging.getLogger(__name__)

THREAD_NAME = "roboemot-embed"
JOIN_TIMEOUT_S = 10.0
Clock = Callable[[], float]


def cut(image: np.ndarray, box: Box) -> tuple[np.ndarray, Box]:
    """A copy of the pixels of ``box`` and the box in the copy's pixels. The copy covers the
    same integer pixels a backend takes from the whole frame (``int`` of each edge, clipped to
    the image), so embedding the copy with the returned box gives the same crop."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1)), max(0, int(y1))
    crop = image[top : min(height, int(y2)), left : min(width, int(x2))].copy()
    return crop, (x1 - left, y1 - top, x2 - left, y2 - top)


@dataclass(frozen=True)
class EmbedJob:
    """One appearance embedding to run: what the vision thread decided when it queued it.

    ``token`` is the per-track state the job was queued for: a result is applied only while
    that very object is the track's state, so a track that ended (or a reset that reused its
    ID) never receives it. ``voting``, ``restart`` and ``measured`` (the detector box) are
    frozen at queue time; ``reacquired`` is the flag the queueing cleared, restored if the job
    is dropped."""

    track_id: int
    token: object = field(repr=False)
    crop: np.ndarray = field(repr=False)
    box: Box  # in the crop's pixels
    stamp: float  # capture stamp of the frame the crop was cut from
    reason: str
    voting: bool
    restart: bool
    measured: Box
    reacquired: bool
    queued_at: float = 0.0  # worker clock, set by ``submit``


@dataclass(frozen=True)
class EmbedOutcome:
    """A finished or dropped job, for the vision thread: ``embedding`` is None when the job was
    dropped at the cap (``dropped``) or the embedder raised (``error``, a type name)."""

    job: EmbedJob
    embedding: np.ndarray | None = field(default=None, repr=False)
    dropped: bool = False
    error: str | None = None


class EmbedWorker:
    """The appearance embedder on its own thread, with a latest-per-track queue; see the module
    docstring."""

    def __init__(
        self,
        embedder: AppearanceEmbedder,
        timer: StageTimer | None = None,
        *,
        clock: Clock = time.monotonic,
        max_window_s: float = 0.0,
    ):
        """``timer`` records ``appearance_embed`` runs (wall and CPU time of this thread);
        ``max_window_s`` (the upper bound of ``vision.timing_window_s``) sizes the ring of
        queue waits, which ``snapshot`` filters to the window asked for."""
        self.embedder = embedder
        self.timer = timer
        self._clock = clock
        self._cond = threading.Condition()
        self._queue: OrderedDict[int, EmbedJob] = OrderedDict()
        self._done: list[EmbedOutcome] = []
        self._busy = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._waits: deque[tuple[float, float]] = deque(
            maxlen=math.ceil(max_window_s * MAX_RATE_HZ) + 1
        )
        self._counts = dict.fromkeys(
            ("queued", "replaced", "dropped_cap", "dropped_ended", "finished", "failed"), 0
        )

    @property
    def model_id(self) -> str:
        return self.embedder.model_id

    def submit(self, job: EmbedJob, cap: int) -> None:
        """Queue ``job`` without blocking: it replaces a waiting job of its track, else, with
        ``cap`` jobs waiting, the oldest waiting job is dropped (handed back by ``drain``)."""
        with self._cond:
            now = self._clock()
            job = replace(job, queued_at=now)
            self._counts["queued"] += 1
            if job.track_id in self._queue:
                self._counts["replaced"] += 1
                self._queue[job.track_id] = job
            else:
                while len(self._queue) >= cap:
                    _, old = self._queue.popitem(last=False)
                    self._counts["dropped_cap"] += 1
                    self._done.append(EmbedOutcome(old, dropped=True))
                self._queue[job.track_id] = job
            self._cond.notify()

    def cancel(self, track_id: int) -> None:
        """Withdraw the waiting job of a track that ended (counted as ``dropped_ended``)."""
        with self._cond:
            if self._queue.pop(track_id, None) is not None:
                self._counts["dropped_ended"] += 1

    def clear(self) -> None:
        """Withdraw every waiting job and discard every finished result (a reset); a run in
        flight finishes and is discarded by the vision thread, whose states it no longer
        matches. The jobs and results count as ``dropped_ended`` (a cap drop is counted
        already)."""
        with self._cond:
            results = sum(1 for o in self._done if o.embedding is not None)
            self._counts["dropped_ended"] += len(self._queue) + results
            self._queue.clear()
            self._done.clear()

    def discarded(self) -> None:
        """Count a result the vision thread discarded because its track had ended."""
        with self._cond:
            self._counts["dropped_ended"] += 1

    def drain(self) -> list[EmbedOutcome]:
        """The finished and dropped jobs since the last call, in the order they ended."""
        with self._cond:
            done, self._done = self._done, []
        return done

    def run_one(self, wait_s: float | None = 0.0) -> bool:
        """Run the oldest waiting job on the calling thread, waiting up to ``wait_s`` for one
        (None: until one comes or ``stop``); returns whether one ran."""
        with self._cond:
            if wait_s is None:
                while not self._queue and not self._stop:
                    self._cond.wait()
            elif not self._queue and wait_s > 0:
                self._cond.wait(wait_s)
            if not self._queue or self._stop:
                return False
            _, job = self._queue.popitem(last=False)
            self._busy = True
            started = self._clock()
            self._waits.append((started, started - job.queued_at))
        try:
            outcome = self._run(job)
        except BaseException:
            with self._cond:
                self._busy = False
                self._cond.notify_all()
            raise
        with self._cond:
            self._counts["finished" if outcome.error is None else "failed"] += 1
            self._done.append(outcome)
            self._busy = False
            self._cond.notify_all()
        return True

    def _run(self, job: EmbedJob) -> EmbedOutcome:
        wall0, cpu0 = self.timer.start() if self.timer is not None else (0.0, 0.0)
        try:
            embedding = self.embedder.embed(job.crop, job.box)
        except Exception as exc:
            log.warning("appearance embed failed: %s", type(exc).__name__)
            return EmbedOutcome(job, error=type(exc).__name__)
        finally:
            if self.timer is not None:
                self.timer.record(APPEARANCE, wall0, cpu0)
        return EmbedOutcome(job, embedding)

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait until no job is waiting or running, up to ``timeout_s``; returns whether that
        holds (tests and a clean stop; the vision thread never waits)."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._queue and not self._busy, timeout_s)

    def start(self) -> None:
        """Start the worker thread."""
        with self._cond:
            self._stop = False
        thread = threading.Thread(target=self._loop, name=THREAD_NAME, daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Stop the thread and wait for the run in flight, bounded by ``JOIN_TIMEOUT_S`` (a run
        stuck in a native call is logged, not waited for). Waiting jobs are discarded."""
        with self._cond:
            self._stop = True
            self._queue.clear()
            self._cond.notify_all()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                log.warning("embed worker did not stop within %.0f s", JOIN_TIMEOUT_S)

    def _loop(self) -> None:
        priority.lower_current()  # the calling thread does part of the ONNX Runtime work
        while True:
            with self._cond:
                if self._stop:
                    return
            self.run_one(None)

    def snapshot(self, window_s: float) -> dict[str, Any]:
        """For the debug report (SPEC 12A): the counters since start, the jobs waiting now,
        whether a run is in flight and the queue wait (ms, p50 and p90 over the runs started in
        the last ``window_s`` of wall time; null without one). Numbers only."""
        with self._cond:
            counts = dict(self._counts)
            waiting, busy = len(self._queue), self._busy
            cutoff = self._clock() - window_s
            waits = sorted(w * MS_PER_S for at, w in self._waits if at >= cutoff)
        return {
            "state": "running" if self._thread is not None and self._thread.is_alive() else "idle",
            **counts,
            "waiting": waiting,
            "busy": busy,
            "queue_wait_ms": {
                "p50": quantile(waits, P50) if waits else None,
                "p90": quantile(waits, P90) if waits else None,
            },
        }
