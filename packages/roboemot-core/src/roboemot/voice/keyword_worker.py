"""The stop-keyword recognizer run, off the listener thread (SPEC 8 stage 8).

Vosk takes 44 to 76 ms to build a recognizer and 65 to 224 ms to digest the ring of audio kept
from before the first candidate frame. Run on the listener thread, that work held every later
microphone chunk back, so a barge-in whose speech frames were already counted was cut off late.
:class:`KeywordWorker` runs the ASR calls on its own thread; ``BargeIn`` only submits audio
and polls for a hit, so the decision to cut the robot off never waits on the ASR.

Ownership: the ASR object is the listener's too. While a session is live the worker owns it; the
barge-in hands it back with :meth:`end` (``block=True``) before the listener touches it again,
which waits for the frame in hand only (the ring is fed one VAD frame at a time).
A *generation* number guards against stale work: every end bumps it and a queued operation
of an older generation is skipped. Text never leaves the ASR call: only the "stop heard" flag does.

``background=False`` runs each operation inline in the caller (unit tests, deterministic).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .asr import ASRBackend

log = logging.getLogger(__name__)
POLL_S = 0.1
JOIN_S = 2.0

Frame = tuple[float, np.ndarray]


class KeywordWorker:
    """Runs the ASR calls of the keyword path (see the module docstring)."""

    def __init__(
        self,
        asr: ASRBackend,
        spot: Callable[[str, str], bool],
        frame_s: float,
        *,
        background: bool = False,
    ) -> None:
        self._asr, self._spot, self._frame_s = asr, spot, frame_s
        self._background = background
        self._lock = threading.Lock()  # held by whoever is using the ASR
        self._gen = 0
        self._lang = ""
        self._dirty = False  # a session was started and the ASR not yet reset by a blocking end
        self._failed = -1  # generation whose session raised
        self._hit: tuple[int, float] | None = None
        self.errors = 0
        self._queue: queue.SimpleQueue[tuple[str, int, Any]] = queue.SimpleQueue()
        self._closed = False
        self._thread: threading.Thread | None = None

    # ---- caller side (the listener thread)

    def start(self, lang: str, frames: Sequence[Frame]) -> None:
        """Begin a session in ``lang`` and feed it the ``frames`` already heard."""
        self._dirty = True
        self._submit("start", (lang, list(frames)))

    def feed(self, stamp: float, frame: np.ndarray) -> None:
        self._submit("feed", (stamp, frame))

    def take_hit(self) -> float | None:
        """The end stamp of the frame after which "stop" was heard in the live session, once."""
        hit = self._hit
        if hit is None or hit[0] != self._gen:
            return None
        self._hit = None
        return hit[1]

    def end(self, *, block: bool) -> None:
        """Drop the session. Queued work is discarded. ``block`` waits for the frame in hand and
        resets the ASR before returning (the listener may use it next); otherwise the reset is
        queued."""
        self._gen += 1
        self._hit = None
        if not self._dirty:
            return
        if block:
            self._drain()
            with self._lock:
                self._reset_asr()
            self._dirty = False
        else:
            self._submit("reset", None)

    def close(self) -> None:
        """Stop the thread (waits briefly for the frame in hand) and release the ASR."""
        self._closed = True
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(JOIN_S)
        self.end(block=True)

    # ---- work

    def _submit(self, kind: str, payload: Any) -> None:
        op = (kind, self._gen, payload)
        if not self._background:
            self._run(op)
            return
        if self._thread is None and not self._closed:
            self._thread = threading.Thread(
                target=self._loop, name="roboemot-keywords", daemon=True
            )
            self._thread.start()
        self._queue.put(op)

    def _drain(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _loop(self) -> None:
        while not self._closed:
            try:
                op = self._queue.get(timeout=POLL_S)
            except queue.Empty:
                continue
            self._run(op)

    def _run(self, op: tuple[str, int, Any]) -> None:
        kind, gen, payload = op
        if kind == "start":
            lang, frames = payload
            ops: list[tuple[str, Any]] = [("start", lang), *(("feed", f) for f in frames)]
        else:
            ops = [(kind, payload)]
        for kind, payload in ops:
            with self._lock:  # a slice at a time: a waiter never waits for the whole ring
                if gen != self._gen or gen == self._failed:
                    return
                try:
                    self._apply(kind, gen, payload)
                except Exception as exc:
                    self.errors += 1
                    self._failed = gen
                    log.warning("barge-in: keyword recognizer: %s", type(exc).__name__)
                    self._reset_asr()
                    return

    def _apply(self, kind: str, gen: int, payload: Any) -> None:
        if kind == "start":
            self._lang = payload
            self._asr.start(payload)
        elif kind == "feed":
            if self._hit is not None and self._hit[0] == gen:
                return
            stamp, frame = payload
            event = self._asr.accept(frame)
            if event is not None and event.text and self._spot(event.text, self._lang):
                self._hit = (gen, stamp + self._frame_s)
        elif kind == "reset":
            self._reset_asr()

    def _reset_asr(self) -> None:
        try:
            self._asr.reset()
        except Exception as exc:
            log.warning("barge-in: recognizer reset: %s", type(exc).__name__)
