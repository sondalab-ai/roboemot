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
from typing import Any, NamedTuple, Protocol

import numpy as np

from .. import shutdown
from .geometry import CameraModel
from .stages import PerceptionModule
from .types import Frame

log = logging.getLogger(__name__)

IDLE_WAIT_S = 0.01
PAUSED_WAIT_S = 0.25  # between checks of the pause flag while the camera is paused
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


class Kept(NamedTuple):
    """The frame the live preview shows: the newest one the pipeline finished, its submit
    sequence number and the overlay computed on it (None without an overlay hook)."""

    frame: Frame
    seq: int
    overlay: Any


class LatestFrameRunner:
    """Runs a perception module on the newest submitted frame in its own thread."""

    def __init__(
        self,
        module: PerceptionModule,
        clock: Callable[[], float] = time.monotonic,
        max_hz: float = MAX_RATE_HZ,
        overlay: Callable[[], Any] | None = None,
    ):
        """``overlay`` is read on the worker thread right after a frame was processed, only
        while watched, and kept with that frame (the face boxes of the live preview)."""
        self.module = module
        self._overlay = overlay
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
        self._frame_seq = 0
        self._frame_at = float("-inf")
        self._watch_until = float("-inf")
        self._latest: Kept | None = None
        self._latest_at = float("-inf")
        self._seq = 0
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
            now = self._clock()
            self._arrived.append(now)
            self._seq += 1
            self._frame_seq, self._frame_at = self._seq, now
            if now >= self._watch_until:
                self._latest = None
            self._cond.notify()

    def watch(self, window_s: float) -> None:
        """Keep a reference to the newest processed frame for the next ``window_s`` seconds (the
        web UI's live preview, SPEC 11A). With nobody watching nothing is kept: one clock
        comparison per frame, no copy. The window lapses on its own and the next frame after it
        drops the reference."""
        with self._cond:
            self._watch_until = self._clock() + window_s

    def latest(self) -> Kept | None:
        """The newest frame the pipeline finished while watched, with its sequence number and
        the overlay computed on it, or None. Under overload it lags the camera, but the boxes
        always belong to the picture (they are the tracker's state after that frame: on a frame
        without a face detection, as of the last one). The frame is shared with the pipeline:
        read it, never write to it."""
        with self._cond:
            return self._latest

    def latest_age_s(self) -> float | None:
        """Seconds (on the runner's clock) since the kept frame was submitted, or None."""
        with self._cond:
            return None if self._latest is None else self._clock() - self._latest_at

    def unwatch(self) -> None:
        """Stop watching now and drop the kept frame (viewing stopped, flag off, robot asleep)."""
        with self._cond:
            self._watch_until = float("-inf")
            self._latest = None

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
            self.run_pending()

    def run_pending(self) -> bool:
        """Process the waiting frame on the calling thread, if there is one; returns whether
        there was. The worker thread's step (tests drive the runner with it); never call it
        while the worker thread runs."""
        with self._cond:
            if self._frame is None:
                return False
            frame, self._frame = self._frame, None
            seq, arrived = self._frame_seq, self._frame_at
        try:
            self.module.process(frame)
            self._keep(frame, seq, arrived)
            with self._cond:
                self.processed += 1
                self._done.append(self._clock())
        except Exception as exc:
            with self._cond:
                self.errors += 1
                self.last_error = type(exc).__name__
            log.warning("vision frame failed: %s", type(exc).__name__)
        return True

    def _keep(self, frame: Frame, seq: int, arrived: float) -> None:
        """While watched, keep ``frame`` with the overlay computed on it (worker thread, right
        after ``process``, so nothing newer has touched the pipeline's state)."""
        with self._cond:
            if self._clock() >= self._watch_until:
                return
        try:
            overlay = self._overlay() if self._overlay is not None else None
        except Exception as exc:  # the preview loses this frame, vision goes on
            log.warning("vision overlay failed: %s", type(exc).__name__)
            return
        with self._cond:
            if self._clock() < self._watch_until:  # unwatch() may have run meanwhile
                self._latest = Kept(frame, seq, overlay)
                self._latest_at = arrived


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
    paused: threading.Event | None = None,
) -> None:
    """Feed ``runner`` from ``source`` until ``stop`` is set (run it in the camera thread).
    While ``paused`` is set the camera is not read at all (the robot is asleep, SPEC 12A).

    Reads at most ``max_hz`` times per second (``vision.rates.camera_hz``) and sleeps on
    ``stop.wait`` between reads, so it never spins whether the SDK call blocks or returns at
    once. A frame identical to the previous one (same object or same sampled pixels) is not
    submitted again. ``IDLE_WAIT_S`` is the pause when the source has no frame.
    """
    period = 1.0 / max_hz
    last_key: int | None = None
    while not stop.is_set():
        if paused is not None and paused.is_set():
            stop.wait(PAUSED_WAIT_S)
            continue
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
