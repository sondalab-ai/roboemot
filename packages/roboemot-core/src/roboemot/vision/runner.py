"""Live frame delivery: a latest-wins slot between the camera and the pipeline (SPEC 13).

The camera thread calls ``submit``; the worker thread takes the newest frame and runs the
pipeline on it. A frame still waiting when the next one arrives is replaced and counted in
``dropped``, never queued, so a slow stage cannot build latency. The runner reads no clock
(frames carry their own capture stamps); only ``pump`` paces camera reads with an injectable clock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .stages import PerceptionModule
from .types import Frame

log = logging.getLogger(__name__)

IDLE_WAIT_S = 0.01
GRID = 8


class FrameSource(Protocol):
    """A camera: ``read`` returns the newest frame, or None when there is none yet."""

    def read(self) -> Frame | None: ...


class LatestFrameRunner:
    """Runs a perception module on the newest submitted frame in its own thread."""

    def __init__(self, module: PerceptionModule):
        self.module = module
        self.dropped = 0
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
            self._cond.notify()

    def start(self) -> None:
        """Start the worker thread."""
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="roboemot-vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker and wait for the frame in flight.

        A frame still waiting in the slot is discarded. Safe to call once after ``start``.
        """
        with self._cond:
            self._stop = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join()

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
            except Exception as exc:
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
