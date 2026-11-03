"""What each vision stage costs, for the debug report (SPEC 12A, 13).

``StageTimer`` keeps, per stage, a fixed ring of (end time, wall seconds, CPU seconds) and
reports over the last ``vision.timing_window_s`` of wall time: runs per second, wall time per
run in ms (p50, p90, max) and ``cpu_share``, CPU-seconds per second of wall time (the share of
one core). Keys: ``face_detect`` (YuNet), ``face_embed`` (SFace), ``object_detect`` (NanoDet),
``appearance_embed`` (DINOv2) and ``process``, the whole ``VisionPipeline.process`` call of one
frame (frames the governor lets through without a stage included). The report never carries the
word "embedding" (privacy guard of the perception tests), hence ``_embed``.

CPU time is ``time.thread_time`` of the vision thread. ONNX Runtime runs each model with
``intra_op_num_threads = vision.threads`` (2 by default) and OpenCV with ``cv2.setNumThreads``
of the same value, so work done in their pool threads is not counted: ``cpu_share`` is a lower
bound, and ``wall_ms`` is the cost the frame path sees.

Timing reads its own clocks (the pipeline only uses capture stamps); the values never feed a
decision, so replays stay deterministic. Recording a run writes three doubles into a ring sized
once for the largest window at the runner's highest frame rate: no allocation per run. A stage
that runs more than once per frame (embeddings) can wrap the ring before the window ends;
``span_s`` then says how much wall time the numbers cover. Numbers only (SPEC 12).
"""

from __future__ import annotations

import math
import threading
import time
from array import array
from collections.abc import Callable
from typing import Any

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


class _Ring:
    """Fixed-size ring of samples, three doubles each; ``count`` saturates at capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = array("d", [0.0]) * (FIELDS * capacity)
        self.next = 0
        self.count = 0
        self.total = 0

    def add(self, end: float, wall_s: float, cpu_s: float) -> None:
        i = self.next * FIELDS
        self.data[i] = end
        self.data[i + 1] = wall_s
        self.data[i + 2] = cpu_s
        self.next = (self.next + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.total += 1

    def samples(self) -> list[tuple[float, float, float]]:
        """The kept samples, oldest first."""
        start = (self.next - self.count) % self.capacity
        d = self.data
        out = []
        for k in range(self.count):
            i = ((start + k) % self.capacity) * FIELDS
            out.append((d[i], d[i + 1], d[i + 2]))
        return out


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
        self._started = wall_clock()

    def start(self) -> tuple[float, float]:
        """Wall and CPU clock readings to hand back to ``record`` when the stage returns."""
        return self._wall(), self._cpu()

    def record(self, stage: str, wall0: float, cpu0: float) -> None:
        """Record one run of ``stage`` that started at ``wall0``/``cpu0`` (from ``start``)."""
        end, cpu = self._wall(), self._cpu()
        with self._lock:
            self._rings[stage].add(end, end - wall0, max(0.0, cpu - cpu0))

    def reset(self) -> None:
        """Forget every sample; the window restarts now."""
        with self._lock:
            for ring in self._rings.values():
                ring.next = ring.count = ring.total = 0
            self._started = self._wall()

    def snapshot(self, window_s: float) -> dict[str, Any]:
        """Per stage over the last ``window_s`` of wall time: ``runs`` (all time),
        ``runs_per_s``, ``wall_ms`` (``p50``, ``p90``, ``max``; None before a run in the window),
        ``cpu_share`` and ``span_s``, the wall time the rate and share are taken over."""
        with self._lock:
            now = self._wall()
            kept = {s: (r.total, r.count, r.samples()) for s, r in self._rings.items()}
            started = self._started
        out: dict[str, Any] = {"window_s": window_s}
        for stage, (total, count, samples) in kept.items():
            out[stage] = _summary(samples, total, count, now, started, window_s, self._capacity)
        return out


def _summary(
    samples: list[tuple[float, float, float]],
    total: int,
    count: int,
    now: float,
    started: float,
    window_s: float,
    capacity: int,
) -> dict[str, Any]:
    """One stage's report entry from its kept samples (oldest first)."""
    cutoff = now - window_s
    first = started
    if count == capacity and samples:  # the ring wrapped: it covers only from its oldest sample
        oldest_end, oldest_wall, _ = samples[0]
        first = max(first, oldest_end - oldest_wall)
    span = max(MIN_SPAN_S, min(window_s, now - first))
    recent = [(wall, cpu) for end, wall, cpu in samples if end >= cutoff]
    walls = sorted(wall * MS_PER_S for wall, _ in recent)
    return {
        "runs": total,
        "runs_per_s": len(recent) / span,
        "wall_ms": {
            "p50": quantile(walls, P50) if walls else None,
            "p90": quantile(walls, P90) if walls else None,
            "max": walls[-1] if walls else None,
        },
        "cpu_share": sum(cpu for _, cpu in recent) / span,
        "span_s": span,
    }
