"""The listener's real-time account (SPEC 8, 13): does it keep up with the microphone?

Per second of audio read, the wall time the listener thread spent on each stage (voice activity
detection, recognizer accept, start and finish, the barge-in check while the robot speaks) and
in total (``grammar_wait``: the listener waiting for a pass before the next utterance starts;
``handoff``: wake gate, prosody and delivery of an utterance, also when it runs between chunks),
the grammar pass time on its own thread, how far behind real time each chunk was when
read (``lag_s``: the age of its end, about the depth of the microphone queue) and the audio lost
before the listener read it (``dropped_s``: the holes in the chunk stamps after a gap).

Figures are kept since start and per window of ``voice.listener.report_window_s`` of audio (the
last full window is reported); ``over_budget`` flags a window whose total share of the audio time
passed ``voice.listener.budget_share``. Numbers only: no audio, no text.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..config.schema import ListenerConfig

MS_PER_S = 1000.0
STAGES = (
    "vad",
    "asr_accept",
    "asr_start",
    "asr_finish",
    "bargein",
    "grammar_wait",
    "handoff",
    "grammar",
    "decode",
)


@dataclass
class _Account:
    audio_s: float = 0.0
    busy_s: float = 0.0
    dropped_s: float = 0.0
    lag_max_s: float = 0.0
    stages: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STAGES, 0.0))

    def report(self) -> dict[str, Any]:
        per = MS_PER_S / self.audio_s if self.audio_s > 0 else 0.0
        return {
            "audio_s": self.audio_s,
            "dropped_s": self.dropped_s,
            "lag_max_s": self.lag_max_s,
            "share": self.busy_s / self.audio_s if self.audio_s > 0 else 0.0,
            "ms_per_audio_s": {
                **{stage: spent * per for stage, spent in self.stages.items()},
                "listener_total": self.busy_s * per,
            },
        }


class ListenerLoad:
    """See the module docstring. The listener thread calls ``chunk`` and ``stage``; the grammar
    worker calls ``stage("grammar", ...)``; ``snapshot`` runs anywhere."""

    def __init__(self, cfg: ListenerConfig) -> None:
        self._lock = threading.Lock()
        self._cfg = cfg
        self._total = _Account()
        self._window = _Account()
        self._last: _Account | None = None
        self._lag_s = 0.0

    def apply_config(self, cfg: ListenerConfig) -> None:
        self._cfg = cfg

    def stage(self, name: str, seconds: float) -> None:
        """Add ``seconds`` spent on stage ``name``."""
        with self._lock:
            self._total.stages[name] += seconds
            self._window.stages[name] += seconds

    def busy(self, seconds: float) -> None:
        """Listener work outside a chunk (a hand-off between two reads)."""
        with self._lock:
            self._total.busy_s += seconds
            self._window.busy_s += seconds

    def chunk(self, audio_s: float, busy_s: float, lag_s: float, dropped_s: float) -> None:
        """One chunk read: its length, the listener's time on it, its lag when read and the
        audio lost just before it. Closes the window once it holds enough audio."""
        with self._lock:
            self._lag_s = lag_s
            for acc in (self._total, self._window):
                acc.audio_s += audio_s
                acc.busy_s += busy_s
                acc.dropped_s += dropped_s
                acc.lag_max_s = max(acc.lag_max_s, lag_s)
            if self._window.audio_s >= self._cfg.report_window_s:
                self._last, self._window = self._window, _Account()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            last = self._last.report() if self._last is not None else None
            return {
                "lag_s": self._lag_s,
                "budget_share": self._cfg.budget_share,
                "over_budget": last is not None and last["share"] > self._cfg.budget_share,
                "total": self._total.report(),
                "last_window": last,
            }
