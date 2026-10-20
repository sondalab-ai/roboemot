"""Live frame delivery: a latest-wins slot between the camera and the pipeline (SPEC 13).

The camera thread calls ``submit``; the worker thread takes the newest frame and runs the
pipeline on it. A frame still waiting when the next one arrives is replaced and counted in
``dropped``, never queued, so a slow stage cannot build latency. The runner decides nothing
from a clock (frames carry their own capture stamps); ``clock`` only times the rate statistics,
and ``pump`` paces camera reads with its own injectable clock.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .. import shutdown
from .geometry import CameraModel
from .stages import PerceptionModule
from .types import Frame

log = logging.getLogger(__name__)

IDLE_WAIT_S = 0.01
GRID = 8
RATE_WINDOW_S = 10.0
MAX_RATE_HZ = 120.0  # vision.rates.camera_hz upper bound: sizes the rate windows
JOIN_TIMEOUT_S = 10.0
MIN_SPAN_S = 1e-3


class FrameSource(Protocol):
    """A camera: ``read`` returns the newest frame, or None when there is none yet."""

    def read(self) -> Frame | None: ...

    def camera_model(self, width: int, height: int) -> CameraModel | None:
        """The pinhole model for a ``width`` x ``height`` frame, or None when the camera does not
        know its intrinsics (vision then falls back to ``world.camera_fov_*_deg``). Every
        implementation declares this method; None means "not supported"."""
        ...


class LockedSource:
    """One lock around a ``FrameSource``, shared by every reader (the vision pump and the
    circadian idle light reader), so the SDK media is never read from two threads at once.
    ``camera_model`` passes through."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._lock = threading.Lock()

    def read(self) -> Frame | None:
        with self._lock:
            return self._source.read()

    def camera_model(self, width: int, height: int) -> CameraModel | None:
        return self._source.camera_model(width, height)


class LatestFrameRunner:
    """Runs a perception module on the newest submitted frame in its own thread."""

    def __init__(
        self,
        module: PerceptionModule,
        clock: Callable[[], float] = time.monotonic,
        max_hz: float = MAX_RATE_HZ,
    ):
        self.module = module
        self.dropped = 0
        self.processed = 0
        self.errors = 0
        self.last_error: str | None = None  # exception type name only, never the message
        self._clock = clock
        samples = math.ceil(RATE_WINDOW_S * max_hz) + 1
        self._done: deque[float] = deque(maxlen=samples)
        self._arrived: deque[float] = deque(maxlen=samples)
        self._started: float | None = None
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._stop = False
        self._thread: threading.Thread | None = None

    def submit(self, frame: Frame) -> None:
        """Hand over a frame, replacing one that has not been taken yet.

        Never blocks. A replaced frame is counted in ``dropped``; the worker always sees the
        newest frame, so a slow stage cannot build latency.
        """
        with self._cond:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._arrived.append(self._clock())
            self._cond.notify()

    def rates(self) -> dict[str, float]:
        """Frames per second handed in and processed over the last ``RATE_WINDOW_S`` seconds.

        Before the runner has run that long, the rate is over the time since ``start``. The
        sample buffers hold a full window at ``MAX_RATE_HZ``, so the rate does not saturate.
        """
        with self._cond:
            now = self._clock()
            cutoff = now - RATE_WINDOW_S
            since = self._started if self._started is not None else now
            span = max(MIN_SPAN_S, min(RATE_WINDOW_S, now - since))
            return {
                "submitted_hz": sum(t >= cutoff for t in self._arrived) / span,
                "processed_hz": sum(t >= cutoff for t in self._done) / span,
            }

    def start(self) -> None:
        """Start the worker thread."""
        self._stop = False
        self._started = self._clock()
        thread = threading.Thread(target=self._loop, name="roboemot-vision", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Stop the worker and wait for the frame in flight.

        A frame still waiting in the slot is discarded. The wait for the frame in flight is
        bounded by ``JOIN_TIMEOUT_S``: a stage stuck in a native call is logged, not waited
        for. Safe to call once after ``start``.
        """
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                log.warning("vision worker did not stop within %.0f s", JOIN_TIMEOUT_S)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._frame is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame, self._frame = self._frame, None
            try:
                self.module.process(frame)
                with self._cond:
                    self.processed += 1
                    self._done.append(self._clock())
            except Exception as exc:
                with self._cond:
                    self.errors += 1
                    self.last_error = type(exc).__name__
                log.warning("vision frame failed: %s", type(exc).__name__)


def fingerprint(image: np.ndarray) -> int:
    """Cheap hash of a sparse pixel grid (about 64 samples), to spot a repeated frame."""
    step_y = max(1, image.shape[0] // GRID)
    step_x = max(1, image.shape[1] // GRID)
    return hash(image[::step_y, ::step_x].tobytes())


def pump(
    source: FrameSource,
    runner: LatestFrameRunner,
    stop: threading.Event,
    max_hz: float,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Feed ``runner`` from ``source`` until ``stop`` is set (run it in the camera thread).

    Reads at most ``max_hz`` times per second (``vision.rates.camera_hz``) and sleeps on
    ``stop.wait`` between reads, so it never spins whether the SDK call blocks or returns at
    once. A frame identical to the previous one (same object or same sampled pixels) is not
    submitted again. ``IDLE_WAIT_S`` is the pause when the source has no frame.
    """
    period = 1.0 / max_hz
    last_key: int | None = None
    while not stop.is_set():
        started = clock()
        frame = source.read()
        pause = IDLE_WAIT_S
        if frame is not None:
            key = fingerprint(frame.image)
            if key != last_key:
                runner.submit(frame)
                last_key = key
            pause = 0.0
        stop.wait(max(pause, period - (clock() - started)))
