"""The grammar pass, off the listener thread (SPEC 8 stage 3, 13).

On the CM4 the pass took up to 0.76 s per utterance on the listener thread, while the microphone
kept filling its queue. :class:`GrammarWorker` runs it on its own thread: the listener submits the
pass of a finished utterance and keeps reading audio; it hands the utterance off when the
:class:`GrammarJob` is done or its wait (``voice.grammar.handoff_timeout_s``) is over.

One pass runs at a time and one waits: a new submission replaces a waiting one that never
started (its utterance was already handed off without it, counted ``replaced``). A pass that is
running is never interrupted (a Vosk decode cannot be); a result that comes after its utterance
was handed off is dropped by the listener. ``background=False`` runs each pass inline in
``submit`` (unit tests, deterministic). No text or audio is logged: exception types only.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .. import shutdown

log = logging.getLogger(__name__)
JOIN_S = 2.0

Task = Callable[[], frozenset[str]]


class GrammarJob:
    """One submitted pass: ``done`` once it ran (``result``, empty on an error) or was
    replaced unrun; ``spent_s`` is its run time."""

    def __init__(self, task: Task, timer: Callable[[], float]) -> None:
        self._task, self._timer = task, timer
        self._done = threading.Event()
        self.result: frozenset[str] = frozenset()
        self.ran = False
        self.spent_s = 0.0

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def run(self) -> None:
        t0 = self._timer()
        try:
            self.result = self._task()
        except Exception as exc:  # the pass counts its own errors; this is a last guard
            log.warning("voice: grammar worker: %s", type(exc).__name__)
        finally:
            self.spent_s = self._timer() - t0
            self.ran = True
            self._done.set()

    def drop(self) -> None:
        self._done.set()


class GrammarWorker:
    """See the module docstring."""

    def __init__(
        self,
        *,
        background: bool,
        on_spent: Callable[[float], None] = lambda seconds: None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._background, self._on_spent, self._timer = background, on_spent, timer
        self._cond = threading.Condition()
        self._next: GrammarJob | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self.replaced = 0

    def submit(self, task: Task) -> GrammarJob:
        """Queue ``task`` (inline without ``background``); returns its job."""
        job = GrammarJob(task, self._timer)
        if not self._background:
            self._run(job)
            return job
        with self._cond:
            if self._next is not None:
                self._next.drop()
                self.replaced += 1
            self._next = job
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(
                    target=self._loop, name="roboemot-grammar", daemon=True
                )
                self._thread.start()
            self._cond.notify()
        return job

    def close(self) -> None:
        """Stop the thread (a pass in flight is waited for at most ``JOIN_S``)."""
        with self._cond:
            self._closed = True
            if self._next is not None:
                self._next.drop()
                self._next = None
            self._cond.notify()
            thread, self._thread = self._thread, None
        if thread is not None and not shutdown.join(thread, JOIN_S):
            log.warning("voice: grammar worker did not stop, left running")

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._next is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                job, self._next = self._next, None
            self._run(job)

    def _run(self, job: GrammarJob) -> None:
        job.run()
        self._on_spent(job.spent_s)
