"""Age and gender per track, in a low-priority worker (SPEC 5, Rules; 13; 12).

The estimator is too slow for the frame path on the CM4, so it never runs there. Per track the
pipeline collects the first ``vision.attribute_crops`` face crops and submits ONE job
that runs the estimator on each crop and aggregates the per-crop values with the registry
helpers (median age group, majority gender, confidence threshold). The age threshold is
``age_confidence_min`` when given (the age model's own score floor, ``vision.attribute_score_min``)
and the 0.8 rule otherwise; gender always uses the 0.8 rule. Until the result comes
back the track is ``unknown`` and so a possible minor. Crops are held in memory only for the
job and dropped when it ends, success or failure (SPEC 12); nothing is logged or written.

The executor is injectable: a one-thread pool live (so the estimator never competes with
itself), ``InlineExecutor`` for tests and replay so results land deterministically. Each job
carries the track's generation, so a result that arrives after the track ended (or its ID was
reused after a reset) is ignored by the pipeline.

For the debug report (SPEC 12A) the worker counts jobs (submitted, completed, failed, cancelled,
refused because the queue was full) and times each job and each crop's inference on its clock;
numbers only. Jobs run on the worker thread and the report reads on its own, so the counters
share the worker's lock.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.schema import IdentityConfig, VisionConfig
from ..registry import Estimate, estimate_age_group, estimate_gender
from ..stats import P50, P90, summarise
from .stages import AttributeEstimator

log = logging.getLogger(__name__)

THREAD_NAME = "roboemot-attributes"
NICE_INCREMENT = 10
MS_PER_S = 1000.0
DURATION_LEVELS = {"p50": P50, "p90": P90}
REPORT_WINDOW = VisionConfig.model_fields["report_window"].default
JOB_COUNTS = ("submitted", "completed", "failed", "cancelled", "queue_full_refusals")


def lower_thread_priority() -> None:
    """Make the calling thread nicer where the OS allows (Linux per-thread nice); else no-op."""
    try:
        tid = threading.get_native_id()
        os.setpriority(os.PRIO_PROCESS, tid, os.getpriority(os.PRIO_PROCESS, tid) + NICE_INCREMENT)
    except (AttributeError, OSError):
        pass


class InlineExecutor(Executor):
    """Runs each job immediately in the caller's thread (tests, replay)."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """Run ``fn`` now and return an already completed future."""
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


@dataclass(frozen=True)
class AttributeResult:
    """Aggregated attributes of one track generation and the stamp of the last crop used."""

    track_id: int
    generation: int
    age: Estimate
    gender: Estimate
    stamp: float


class AttributeWorker:
    """Runs one age and gender job per track and hands back the results on ``poll``.

    ``report_window`` sizes the job and crop duration windows of ``snapshot``; ``clock`` times
    them (seconds, monotonic). ``age_confidence_min`` replaces ``identity.estimate_confidence_min``
    for the age group (None keeps it).
    """

    def __init__(
        self,
        estimator: AttributeEstimator,
        config: IdentityConfig,
        queue_max: int,
        executor: Executor | None = None,
        *,
        report_window: int = REPORT_WINDOW,
        clock: Callable[[], float] = time.perf_counter,
        age_confidence_min: float | None = None,
    ):
        self.estimator = estimator
        self.config = config
        self.age_confidence_min = age_confidence_min
        self.queue_max = queue_max
        self.owns_executor = executor is None
        self.executor: Executor = executor or ThreadPoolExecutor(
            1, thread_name_prefix=THREAD_NAME, initializer=lower_thread_priority
        )
        self._results: queue.SimpleQueue[AttributeResult] = queue.SimpleQueue()
        self._missed: queue.SimpleQueue[tuple[int, int]] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pending = 0
        self._futures: set[Future[AttributeResult]] = set()
        self._closed = False
        self._clock = clock
        self._counts = dict.fromkeys(JOB_COUNTS, 0)
        self._running = 0
        self._job_ms: deque[float] = deque(maxlen=report_window)
        self._crop_ms: deque[float] = deque(maxlen=report_window)

    @property
    def pending(self) -> int:
        """Jobs submitted and not finished."""
        with self._lock:
            return self._pending

    @property
    def full(self) -> bool:
        """True when ``queue_max`` jobs are waiting or running."""
        return self.pending >= self.queue_max

    def submit(self, track_id: int, generation: int, crops: list[tuple[float, np.ndarray]]) -> bool:
        """Queue the job for a track; False (crops untouched) when full or closed.

        ``crops`` are ``(capture stamp, BGR crop)`` pairs; the worker gets its own list and
        clears it when done, so the caller can drop its copy right after a True return. If the
        executor refuses the job the slot is given back and the exception propagates.
        """
        with self._lock:
            if self._closed:
                return False
            if self._pending >= self.queue_max:
                self._counts["queue_full_refusals"] += 1
                return False
            self._pending += 1
        try:
            future = self.executor.submit(self._run, track_id, generation, list(crops))
        except BaseException:
            with self._lock:
                self._pending -= 1
            raise
        with self._lock:
            self._futures.add(future)
            self._counts["submitted"] += 1
        future.add_done_callback(lambda f: self._done(f, track_id, generation))
        return True

    def _run(
        self, track_id: int, generation: int, crops: list[tuple[float, np.ndarray]]
    ) -> AttributeResult:
        with self._lock:
            self._running += 1
        start = self._clock()
        try:
            samples = [self._timed_estimate(crop) for _, crop in crops]
            age = estimate_age_group(
                [(s.age_group, s.age_confidence) for s in samples],
                self.config,
                confidence_min=self.age_confidence_min,
            )
            gender = estimate_gender(
                [(s.gender, s.gender_confidence) for s in samples], self.config
            )
            return AttributeResult(track_id, generation, age, gender, crops[-1][0])
        finally:
            crops.clear()
            elapsed_ms = (self._clock() - start) * MS_PER_S
            with self._lock:
                self._running -= 1
                self._job_ms.append(elapsed_ms)

    def _timed_estimate(self, crop: np.ndarray) -> Any:
        """One crop through the estimator, its inference time kept for the report."""
        start = self._clock()
        sample = self.estimator.estimate(crop)
        elapsed_ms = (self._clock() - start) * MS_PER_S
        with self._lock:
            self._crop_ms.append(elapsed_ms)
        return sample

    def _done(self, future: Future[AttributeResult], track_id: int, generation: int) -> None:
        with self._lock:
            self._pending -= 1
            self._futures.discard(future)
        if future.cancelled():
            with self._lock:
                self._counts["cancelled"] += 1
            self._missed.put((track_id, generation))
            return
        exc = future.exception()
        if exc is not None:
            with self._lock:
                self._counts["failed"] += 1
            log.warning("attribute job failed: %s", type(exc).__name__)
            self._missed.put((track_id, generation))
            return
        with self._lock:
            self._counts["completed"] += 1
        self._results.put(future.result())

    def snapshot(self) -> dict[str, Any]:
        """Job counts, queue length (waiting, not running), whether a job runs now, and job and
        per-crop duration summaries in ms, for the debug report; numbers only."""
        with self._lock:
            counts = dict(self._counts)
            running, pending = self._running, self._pending
            job_ms, crop_ms = list(self._job_ms), list(self._crop_ms)
        return {
            **counts,
            "queue_len": max(0, pending - running),
            "busy": running > 0,
            "job_ms": summarise(job_ms, DURATION_LEVELS),
            "crop_ms": summarise(crop_ms, DURATION_LEVELS),
        }

    def poll_missed(self) -> list[tuple[int, int]]:
        """``(track_id, generation)`` of the jobs that failed or were cancelled since the last
        call, so the pipeline can retry them (SPEC 5); polled on the frame thread like ``poll``."""
        out = []
        while True:
            try:
                out.append(self._missed.get_nowait())
            except queue.Empty:
                return out

    def poll(self) -> list[AttributeResult]:
        """Results finished since the last call, oldest first.

        Failed or cancelled jobs produce no result; ``poll_missed`` reports them. The pipeline
        polls at the start of each frame, which keeps result delivery on the frame thread.
        """
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def cancel_pending(self) -> None:
        """Cancel jobs that have not started and drop results not yet read.

        A job already running finishes; its result carries an old generation and is ignored.
        """
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            future.cancel()
        self.poll()
        self.poll_missed()

    def close(self) -> None:
        """Refuse new jobs, cancel waiting ones and stop the pool if this worker created it.

        A running job finishes in the background and its result is never read. A
        caller-supplied executor is left alone.
        """
        with self._lock:
            self._closed = True
        self.cancel_pending()
        if self.owns_executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
