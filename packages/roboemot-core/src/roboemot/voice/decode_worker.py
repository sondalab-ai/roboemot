"""Whole-utterance decodes off the listener thread (SPEC 8, Realtime speech route; SPEC 13).

An utterance that skipped Vosk (its audio was meant for the realtime route) and still needs a
local final (the voice-at-face check rejected it, the server's transcript did not come, or the
burst was refused) is decoded from its buffer in one call: on the CM4 seconds of CPU for a long
utterance. :class:`DecodeWorker` runs those decodes in order on a thread of its own, so the
listener keeps reading the microphone; the listener hands the utterance off once its
:class:`DecodeJob` is done, or gives up at the job's deadline. A job whose deadline passed
before the thread got to it is skipped (``skipped_stale``). ``background=False`` runs each
decode inline in ``submit`` (unit tests, deterministic). No text or audio is logged.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Callable

from .. import shutdown

log = logging.getLogger(__name__)
JOIN_S = 2.0


class DecodeJob:
    """One decode: ``done`` once it ran (``result``, the text, empty on an error) or was
    skipped; ``ran`` only after it ran."""

    def __init__(self, task: Callable[[], str], deadline: float) -> None:
        self._task: Callable[[], str] | None = task  # holds the utterance's audio until run
        self.deadline = deadline
        self._done = threading.Event()
        self.result = ""
        self.ran = False

    def __repr__(self) -> str:
        return f"DecodeJob(done={self.done}, ran={self.ran})"

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout_s: float) -> bool:
        return self._done.wait(timeout_s)

    def run(self) -> None:
        task, self._task = self._task, None  # no audio kept once it ran
        try:
            if task is not None:
                self.result = task()
        except Exception as exc:
            log.warning("voice: decode: %s", type(exc).__name__)
        finally:
            self.ran = True
            self._done.set()

    def drop(self) -> None:
        self._task = None  # nor once it was dropped
        self._done.set()


class DecodeWorker:
    """See the module docstring. Jobs run first in, first out; ``on_spent`` gets each run's
    seconds (the listener's ``decode`` stage)."""

    def __init__(
        self,
        *,
        background: bool,
        on_spent: Callable[[float], None] = lambda seconds: None,
        timer: Callable[[], float] = time.perf_counter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._background, self._on_spent = background, on_spent
        self._timer, self._clock = timer, clock
        self._cond = threading.Condition()
        self._jobs: collections.deque[DecodeJob] = collections.deque()
        self._closed = False
        self._thread: threading.Thread | None = None
        self.skipped_stale = 0

    def submit(self, task: Callable[[], str], deadline: float) -> DecodeJob:
        """Queue ``task`` (useless after ``deadline``, ``clock`` time); returns its job."""
        job = DecodeJob(task, deadline)
        if not self._background:
            self._run(job)
            return job
        with self._cond:
            if self._closed:
                job.drop()
                return job
            self._jobs.append(job)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, name="roboemot-decode", daemon=True
                )
                self._thread.start()
            self._cond.notify()
        return job

    def close(self) -> None:
        """Stop the thread; queued jobs are dropped (a decode in flight is waited for at most
        ``JOIN_S``)."""
        with self._cond:
            self._closed = True
            while self._jobs:
                self._jobs.popleft().drop()
            self._cond.notify()
            thread, self._thread = self._thread, None
        if thread is not None and not shutdown.join(thread, JOIN_S):
            log.warning("voice: decode worker did not stop, left running")

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                job = self._jobs.popleft()
                if self._clock() >= job.deadline:
                    job.drop()
                    self.skipped_stale += 1
                    continue
            self._run(job)

    def _run(self, job: DecodeJob) -> None:
        t0 = self._timer()
        job.run()
        self._on_spent(self._timer() - t0)
