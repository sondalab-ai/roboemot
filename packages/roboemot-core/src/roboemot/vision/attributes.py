"""Age and gender per track, in a low-priority worker (SPEC 5, Rules; 13; 12).

The estimator is slow (two ViT-B models), so it never runs in the frame path. Per track the
pipeline collects the first ``vision.attribute_crops`` face crops and submits ONE job
that runs the estimator on each crop and aggregates the per-crop values with the registry
helpers (median age group, majority gender, confidence threshold). Until the result comes
back the track is ``unknown`` and so a possible minor. Crops are held in memory only for the
job and dropped when it ends, success or failure (SPEC 12); nothing is logged or written.

The executor is injectable: a one-thread pool live (so the estimator never competes with
itself), ``InlineExecutor`` for tests and replay so results land deterministically. Each job
carries the track's generation, so a result that arrives after the track ended (or its ID was
reused after a reset) is ignored by the pipeline.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.schema import IdentityConfig
from ..registry import Estimate, estimate_age_group, estimate_gender
from .stages import AttributeEstimator

log = logging.getLogger(__name__)

THREAD_NAME = "roboemot-attributes"
NICE_INCREMENT = 10


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
    """Runs one age and gender job per track and hands back the results on ``poll``."""

    def __init__(
        self,
        estimator: AttributeEstimator,
        config: IdentityConfig,
        queue_max: int,
        executor: Executor | None = None,
    ):
        self.estimator = estimator
        self.config = config
        self.queue_max = queue_max
        self.owns_executor = executor is None
        self.executor: Executor = executor or ThreadPoolExecutor(
            1, thread_name_prefix=THREAD_NAME, initializer=lower_thread_priority
        )
        self._results: queue.SimpleQueue[AttributeResult] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pending = 0
        self._futures: set[Future[AttributeResult]] = set()
        self._closed = False

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
            if self._closed or self._pending >= self.queue_max:
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
        future.add_done_callback(self._done)
        return True

    def _run(
        self, track_id: int, generation: int, crops: list[tuple[float, np.ndarray]]
    ) -> AttributeResult:
        try:
            samples = [self.estimator.estimate(crop) for _, crop in crops]
            age = estimate_age_group(
                [(s.age_group, s.age_confidence) for s in samples], self.config
            )
            gender = estimate_gender(
                [(s.gender, s.gender_confidence) for s in samples], self.config
            )
            return AttributeResult(track_id, generation, age, gender, crops[-1][0])
        finally:
            crops.clear()

    def _done(self, future: Future[AttributeResult]) -> None:
        with self._lock:
            self._pending -= 1
            self._futures.discard(future)
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            log.warning("attribute job failed: %s", type(exc).__name__)
            return
        self._results.put(future.result())

    def poll(self) -> list[AttributeResult]:
        """Results finished since the last call, oldest first.

        Failed or cancelled jobs produce no result (the track stays unknown). The pipeline
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
