"""Microphone level from the Reachy Mini SDK media, for the loud-onset reflex (SPEC 4A, 10, 12).

The app owns the SDK client and its media (one process, SPEC 4A); this reads from the
``ReachyMini`` the daemon passed to ``run`` and never creates a client. Coded against
reachy-mini 1.12's documented interface, not exercised against a real robot or the simulator:
``mini.media.start_recording()`` starts the capture, ``mini.media.get_audio_sample()`` returns the
next chunk as a float32 ``ndarray`` (``(n,)`` or ``(n, channels)``, full scale 1.0) or ``None``
when nothing is queued, and ``mini.media.audio`` is ``None`` when the robot has no audio system
(the SDK then logs a warning on every call, so it is checked once).

Privacy (SPEC 12): a chunk lives in one local variable while its RMS is taken, and only that scalar
leaves this module. No buffer is kept, queued, published or logged.

``get_audio_sample`` hands each chunk to one reader. The M3 voice pipeline must take its audio from
this producer (or replace it), not read the SDK in parallel, or the two would split the chunks.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
JOIN_TIMEOUT_S = 2.0
IDLE_WAIT_S = 0.01


def chunk_rms(chunk: Any) -> float | None:
    """RMS of a float audio chunk (all channels), or None for anything but a non-empty ndarray."""
    if not isinstance(chunk, np.ndarray) or chunk.size == 0:
        return None
    x = chunk.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(x))))
    return rms if math.isfinite(rms) else None


class ReachyAudio:
    """Polls the SDK's audio chunks in a thread and hands each chunk's RMS to ``sink(rms, stamp)``.

    ``state`` is one of ``off`` (not started), ``unavailable`` (no audio system or start failed;
    ``reason`` says which), ``no_data`` (running, nothing usable yet) and ``running``. The stamp is
    the monotonic time of the read.
    """

    def __init__(
        self,
        mini: Any,
        sink: Callable[[float, float], None],
        poll_hz: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._mini, self._sink, self._clock = mini, sink, clock
        self._period = 1.0 / poll_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state, self._reason = "off", None
        self._chunks = 0
        self._last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        """State for the debug report: scalars and a reason, never audio."""
        with self._lock:
            return {"state": self._state, "reason": self._reason, "chunks": self._chunks}

    def _set(self, state: str, reason: str | None = None) -> None:
        with self._lock:
            self._state, self._reason = state, reason

    def start(self) -> None:
        """Start capturing in a daemon thread; a second start while running is a no-op. Degrades
        to ``unavailable`` (and starts nothing) when the robot has no audio system."""
        if self._thread is not None and self._thread.is_alive():
            return
        media = getattr(self._mini, "media", None)
        if media is None or getattr(media, "audio", True) is None:
            self._set("unavailable", "the SDK reports no audio system")
            return
        try:
            media.start_recording()
        except Exception as exc:
            self._set("unavailable", f"start_recording failed: {type(exc).__name__}")
            return
        self._set("no_data", "no audio chunk read yet")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="roboemot-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread, wait briefly, then stop the capture. A thread stuck in the SDK keeps
        its handle, so a later ``start`` does not run a second one beside it."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=JOIN_TIMEOUT_S)
            if thread.is_alive():
                return
            self._thread = None
            try:
                self._mini.media.stop_recording()
            except Exception as exc:
                log.warning("audio: stop_recording: %s", type(exc).__name__)
        self._set("off")

    def poll_once(self) -> bool:
        """Read one chunk and hand its RMS on. Returns whether a chunk was used."""
        try:
            chunk = self._mini.media.get_audio_sample()
            rms = chunk_rms(chunk)
            del chunk
            if rms is None:
                return False
            self._sink(rms, self._clock())
        except Exception as exc:
            msg = f"{type(exc).__name__}"
            if msg != self._last_error:  # once per kind, not per poll
                log.warning("audio: %s", msg)
                self._last_error = msg
            return False
        self._last_error = None
        with self._lock:
            self._chunks += 1
            self._state, self._reason = "running", None
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            # Drain what is queued, then wait out the rest of the period.
            used = self.poll_once()
            self._stop.wait(IDLE_WAIT_S if used else self._period)
