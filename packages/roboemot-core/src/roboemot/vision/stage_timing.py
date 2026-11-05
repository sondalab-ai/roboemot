"""What each vision stage costs, for the debug report (SPEC 12A, 13).

``StageTimer`` keeps, per stage, a fixed ring of (end time, wall seconds, CPU seconds). Keys:
``face_detect`` (YuNet), ``face_embed`` (SFace), ``object_detect`` (NanoDet),
``appearance_embed`` (DINOv2) and ``process``, the whole ``VisionPipeline.process`` call of one
frame (frames the governor lets through without a stage included). The report never carries the
word "embedding" (privacy guard of the perception tests), hence ``_embed``.

``runs`` is the count since start. Every other field is taken over the last
``vision.timing_window_s`` of wall time: ``runs_per_s``, ``wall_ms`` per run (p50, p90, max;
null without a run in the window), ``cpu_share``, CPU-seconds per second of wall time (the
share of one core), and ``span_s``, the wall time these cover. The span starts at the stage's
first recorded run, so model loading and a stage that has not run yet never dilute the rate; it
is 0 before the first run. A run that raises still counts: its time was spent.

CPU time is ``time.thread_time`` of the thread that ran the stage: the vision thread, except
``appearance_embed``, which the embed worker records on its own thread in the live app
(``embed_worker``; its ``wall_ms`` is then what an embedding takes there, no longer time the
frame path waits). ONNX Runtime runs each model with ``intra_op_num_threads = vision.threads``
(2 by default; DINOv2-S ``vision.appearance_threads``, 2) and OpenCV with ``cv2.setNumThreads``
of the same value, so work done in their pool threads is not counted: ``cpu_share`` is a lower
bound, and ``wall_ms`` is the cost the calling thread sees.

Timing reads its own clocks (the pipeline only uses capture stamps); the values never feed a
decision, so replays stay deterministic. Recording a run writes three doubles into a ring sized
once for the largest window at the runner's highest frame rate: no allocation per run. The
report copies the raw rings under the lock and does the rest outside it, so reading the report
never holds up the vision thread for long. A stage that runs more than once per frame
(embeddings) can wrap its ring before the window ends; ``span_s`` then shrinks to the wall time
the ring still covers. Numbers only (SPEC 12).
"""

from __future__ import annotations

import math
import threading
import time
from array import array
from collections.abc import Callable
from typing import Any, NamedTuple

from ..stats import P50, P90, quantile
from .runner import MAX_RATE_HZ, MIN_SPAN_S

DETECTION = "face_detect"
EMBEDDING = "face_embed"
OBJECTS = "object_detect"
APPEARANCE = "appearance_embed"
PROCESS = "process"
STAGES = (DETECTION, EMBEDDING, OBJECTS, APPEARANCE, PROCESS)
MS_PER_S = 1000.0
FIELDS = 3  # end time, wall seconds, CPU seconds

Clock = Callable[[], float]


class _Copy(NamedTuple):
    """A ring's raw state, copied under the lock for the report."""

    data: array
    next: int
    count: int
    total: int
    first: float


class _Ring:
    """Fixed-size ring of samples, three doubles each; ``count`` saturates at capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = array("d", [0.0]) * (FIELDS * capacity)
        self.next = 0
        self.count = 0
        self.total = 0
        self.first = math.nan  # wall start of the first recorded run

    def add(self, start: float, end: float, cpu_s: float) -> None:
        if self.total == 0:
            self.first = start
        i = self.next * FIELDS
        self.data[i] = end
        self.data[i + 1] = end - start
        self.data[i + 2] = cpu_s
        self.next = (self.next + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.total += 1

    def copy(self) -> _Copy:
        return _Copy(self.data[:], self.next, self.count, self.total, self.first)


class StageTimer:
    """Per-stage wall and CPU time of the vision thread; see the module docstring.

    ``max_window_s`` sizes the rings (the upper bound of ``vision.timing_window_s``), so a hot
    reload of the window never needs a new buffer. ``start`` and ``record`` run on the vision
    thread, ``snapshot`` on the report thread; one lock guards the rings.
    """

    def __init__(
        self,
        max_window_s: float,
        *,
        wall_clock: Clock = time.perf_counter,
        cpu_clock: Clock = time.thread_time,
    ):
        capacity = math.ceil(max_window_s * MAX_RATE_HZ) + 1
        self._wall = wall_clock
        self._cpu = cpu_clock
        self._capacity = capacity
        self._rings = {stage: _Ring(capacity) for stage in STAGES}
        self._lock = threading.Lock()

    def start(self) -> tuple[float, float]:
        """Wall and CPU clock readings to hand back to ``record`` when the stage returns."""
        return self._wall(), self._cpu()

    def record(self, stage: str, wall0: float, cpu0: float) -> None:
        """Record one run of ``stage`` that started at ``wall0``/``cpu0`` (from ``start``)."""
        end, cpu = self._wall(), self._cpu()
        with self._lock:
            self._rings[stage].add(wall0, end, max(0.0, cpu - cpu0))

    def snapshot(self, window_s: float) -> dict[str, Any]:
        """Per stage over the last ``window_s`` of wall time (see the module docstring).

        Only the raw rings are copied under the lock; samples are built, filtered and sorted
        after it is released.
        """
        with self._lock:
            copies = {stage: ring.copy() for stage, ring in self._rings.items()}
        now = self._wall()
        out: dict[str, Any] = {"window_s": window_s}
        for stage, copy in copies.items():
            out[stage] = _summary(copy, self._capacity, now, window_s)
        return out


def _summary(copy: _Copy, capacity: int, now: float, window_s: float) -> dict[str, Any]:
    """One stage's report entry from a copy of its ring."""
    data, count = copy.data, copy.count
    oldest = (copy.next - count) % capacity
    cutoff = now - window_s
    walls: list[float] = []
    cpu_s = 0.0
    for k in range(count):
        i = ((oldest + k) % capacity) * FIELDS
        if data[i] >= cutoff:
            walls.append(data[i + 1] * MS_PER_S)
            cpu_s += data[i + 2]
    if copy.total == 0:
        span = 0.0
    else:
        first = copy.first
        if count == capacity:  # wrapped: the ring covers only from its oldest sample's start
            i = oldest * FIELDS
            first = max(first, data[i] - data[i + 1])
        span = max(MIN_SPAN_S, min(window_s, now - first))
    walls.sort()
    return {
        "runs": copy.total,
        "runs_per_s": len(walls) / span if span else 0.0,
        "wall_ms": {
            "p50": quantile(walls, P50) if walls else None,
            "p90": quantile(walls, P90) if walls else None,
            "max": walls[-1] if walls else None,
        },
        "cpu_share": cpu_s / span if span else 0.0,
        "span_s": span,
    }
