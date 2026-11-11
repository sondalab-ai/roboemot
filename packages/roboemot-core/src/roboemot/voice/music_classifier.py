"""The audio classifier that decides whether music plays (SPEC 8, 12A).

The beat detector cannot recognise music with no steady beat. A small pretrained classifier
(YAMNet) can: its scores go to the music detector (``MusicDetector.classified``), which applies
the start and keep rules of ``music.classifier``, and to the debug report.

Every ``music.classifier.interval_s`` the listener's last ``WINDOW_SAMPLES`` of contiguous audio go
through the model. A chunk that does not follow the last one (a hole, the robot speaking) restarts
the window, so the robot's own sound never reaches the model. The window stays in memory and
``snapshot`` has numbers only (SPEC 12).
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config.schema import MusicClassifierConfig
from .music import spread

log = logging.getLogger(__name__)

MODEL_RATE_HZ = 16000
WINDOW_SAMPLES = 15600  # YAMNet's fixed input: 0.975 s at 16 kHz
SPEECH_CLASS, MUSIC_CLASS = 0, 132  # output indices in the AudioSet class map of YAMNet
MS_PER_S = 1000.0
RECENT_RUNS = 60  # report only, not behaviour: two minutes at the default interval
SCORE_LEVELS = (0.3, 0.5)  # report only, not behaviour: the share of runs at or above each
RUNNING, UNAVAILABLE = "running", "unavailable"


class ClassifierBackend(Protocol):
    """A model that scores one window of ``WINDOW_SAMPLES`` mono float32 samples."""

    def scores(self, window: np.ndarray) -> tuple[float, float]:
        """The music score and the speech score of ``window``, each 0 to 1."""


class YamnetBackend:
    """YAMNet on the LiteRT interpreter. Raises RuntimeError without the ``music-classifier``
    extra. Load the file through ``roboemot.voice.models.model_file`` so its marker is checked."""

    def __init__(self, model_path: Path, threads: int) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError("the `music-classifier` extra (ai-edge-litert) is missing") from exc
        self._net = Interpreter(model_path=str(model_path), num_threads=threads)
        self._net.allocate_tensors()
        self._input = self._net.get_input_details()[0]["index"]
        self._output = self._net.get_output_details()[0]["index"]

    def scores(self, window: np.ndarray) -> tuple[float, float]:
        self._net.set_tensor(self._input, window)
        self._net.invoke()
        out = self._net.get_tensor(self._output)[0]
        return float(out[MUSIC_CLASS]), float(out[SPEECH_CLASS])


class MusicClassifier:
    """See the module docstring. ``feed`` runs on the listener thread and ``snapshot`` on any
    thread. Without a ``backend`` the state is unavailable, for ``reason``, and ``feed`` is idle."""

    def __init__(
        self,
        cfg: MusicClassifierConfig,
        rate_hz: int,
        backend: ClassifierBackend | None,
        reason: str = "",
        timer: Callable[[], float] = time.thread_time,
    ) -> None:
        self._interval_s = cfg.interval_s
        self._rate = rate_hz
        self._backend = backend
        self._reason = reason
        self._timer = timer
        self._ring = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._head = 0  # where the next sample goes in the ring
        self._filled = 0  # contiguous samples in the ring, at most WINDOW_SAMPLES
        self._expected = math.nan  # the stamp the next contiguous chunk starts at
        self._last_run_s = -math.inf
        self._runs = self._holes = self._errors = 0
        self._failing = False  # the last run raised
        self._recent: deque[tuple[float, float, float]] = deque(maxlen=RECENT_RUNS)

    @property
    def running(self) -> bool:
        """Whether the scores can be trusted: the model loaded and its last run did not raise."""
        return self._backend is not None and not self._failing

    def feed(self, samples: np.ndarray, start: float) -> tuple[float, float] | None:
        """Add one mono chunk that starts at ``start`` (seconds). Scores the window when it is
        full and the interval has passed, and returns (music, speech), else None. A model error
        is counted and never raised."""
        if self._backend is None:
            return None
        audio = np.asarray(samples, dtype=np.float32)
        chunk_s = audio.size / self._rate
        if abs(start - self._expected) > chunk_s:
            self._filled = 0
            self._holes += 1
        self._expected = start + chunk_s
        self._append(audio)
        if self._filled == WINDOW_SAMPLES and start - self._last_run_s >= self._interval_s:
            self._last_run_s = start
            return self._score(self._backend, self._window())
        return None

    def _append(self, audio: np.ndarray) -> None:
        """Write a chunk into the ring, keeping only its last WINDOW_SAMPLES. The model runs
        once per interval while chunks arrive every few ms, so copying into a fixed buffer
        replaces a new window allocation per chunk."""
        n = audio.size
        if n >= WINDOW_SAMPLES:
            self._ring[:] = audio[-WINDOW_SAMPLES:]
            self._head = 0
        else:
            first = min(n, WINDOW_SAMPLES - self._head)
            self._ring[self._head : self._head + first] = audio[:first]
            self._ring[: n - first] = audio[first:]
            self._head = (self._head + n) % WINDOW_SAMPLES
        self._filled = min(WINDOW_SAMPLES, self._filled + n)

    def _window(self) -> np.ndarray:
        """The ring as one contiguous array, oldest sample first: a new array, so the backend
        may keep it, built only when a run is due."""
        return np.concatenate((self._ring[self._head :], self._ring[: self._head]))

    def _score(self, backend: ClassifierBackend, window: np.ndarray) -> tuple[float, float] | None:
        """Run the model on the window; keep the scores and the CPU time for the report and
        return (music, speech), or None when the model raised."""
        t0 = self._timer()
        try:
            music, speech = backend.scores(window)
        except Exception as exc:
            self._errors += 1
            self._failing = True
            if self._errors == 1:
                log.warning("music classifier: %s", type(exc).__name__)
            return None
        self._failing = False
        self._recent.append((music, speech, (self._timer() - t0) * MS_PER_S))
        self._runs += 1
        return music, speech

    def snapshot(self) -> dict[str, Any]:
        """State, counts and the spread of the last scores and costs: numbers only (SPEC 12)."""
        recent = list(self._recent)
        music = [r[0] for r in recent]
        return {
            "state": UNAVAILABLE if self._backend is None else RUNNING,
            "reason": self._reason or None,
            "runs": self._runs,
            "holes": self._holes,
            "errors": self._errors,
            "music": spread(music),
            "music_share_at": {
                str(level): sum(m >= level for m in music) / len(music) if music else None
                for level in SCORE_LEVELS
            },
            "speech": spread([r[1] for r in recent]),
            "cost_ms": spread([r[2] for r in recent]),
        }
