"""Addressed microphone audio to the realtime route (SPEC 8, Realtime speech route; SPEC 12).

With ``voice.realtime_route.input: audio`` and the ``privacy.realtime_audio`` opt-in, an
utterance the local wake gate addressed leaves the robot as audio, after the local decision,
never live. :class:`RealtimeInput` takes it from the listener (``submit``: never blocks) and,
on a thread of its own, sends it through the route's send gate (``RealtimeRoute.send_audio``)
as one burst: the utterance's PCM16 followed by ``close_silence_ms`` of zeros, so the server's
VAD (``server_silence_ms``, less than that) ends the turn within the burst. Nothing is sent
while the robot speaks.

The burst's transcript is collected from the server's events (client thread, ``on_event``):
every ``speech_started`` opens an item, and the job is done once each item has its
``transcription.completed`` and none is still open. A server that splits the utterance gives
several items; their texts are joined in the order the server heard them. The listener waits
for the job with its own deadline (``transcript_timeout_s``) and ``cancel``s it past that; one
burst is in flight at a time.

The server keeps the words as a user item of its conversation, which the next
``response.create`` answers. An item of an utterance the robot does not answer through the
route (a cancelled job, a transcript after the cancel, an empty transcript) is deleted at once,
so it is never answered. Nothing here logs or reports audio, a transcript or an item id.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import shutdown
from ..config.schema import RoboEmotConfig
from ..voice.asr import to_int16
from . import protocol

log = logging.getLogger(__name__)
GATE = "addressed"  # the gate this input sends under (voice.realtime_route.audio_gate)
MS_PER_S = 1000.0
JOIN_S = 2.0
DEAD_ITEMS = 64  # ids of cancelled bursts' items kept, so their late transcripts are deleted


@dataclass(frozen=True)
class Heard:
    """What the server heard of one burst: the joined text, its user items (in order) and the
    server session (``generation``) that holds them. Text and ids are never in repr."""

    text: str = field(repr=False)
    items: tuple[str, ...] = field(repr=False)
    generation: int


class InputJob:
    """One utterance's audio on its way to the server, read like a ``GrammarJob``: ``done``
    once it has a result (``result``, the :class:`Heard`), was refused before anything was
    sent (``refused``, a reason code) or ended without a transcript (cancelled, the session
    lost). ``ran`` is True only with a result; ``sent`` once its burst left the robot."""

    def __init__(self, samples: np.ndarray, rate_hz: int) -> None:
        self._samples: np.ndarray | None = samples
        self.rate_hz = rate_hz
        self._done = threading.Event()
        self.result: Heard | None = None
        self.refused: str | None = None
        self.generation: int | None = None  # the session it was sent on
        self.sent = False  # its burst left the robot
        self.cancelled = False

    def __repr__(self) -> str:
        return f"InputJob(done={self.done}, refused={self.refused!r})"

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def ran(self) -> bool:
        return self.result is not None

    def wait(self, timeout_s: float) -> bool:
        return self._done.wait(timeout_s)

    def _end(self, result: Heard | None = None, refused: str | None = None) -> bool:
        """End the job once; False when it had ended already."""
        if self._done.is_set():
            return False
        self.result, self.refused = result, refused
        self._samples = None
        self._done.set()
        return True


@dataclass
class _Burst:
    """The burst in flight: the server's items in the order heard, those still open, and the
    transcripts in (``closed``: the job ended, later transcripts are deleted)."""

    job: InputJob
    generation: int
    order: list[str | None] = field(default_factory=list)
    open: set[str | None] = field(default_factory=set)
    texts: dict[str | None, str] = field(default_factory=dict, repr=False)
    closed: bool = False


class RealtimeInput:
    """See the module docstring. ``route`` is the :class:`~roboemot.realtime.route.
    RealtimeRoute` (its input events come here); ``speaking()`` says whether the robot speaks
    now (nothing is sent then)."""

    def __init__(self, route: Any, *, speaking: Callable[[], bool] = lambda: False) -> None:
        self.route = route
        self._speaking = speaking
        self._lock = threading.Lock()
        self._queue: queue.Queue[InputJob | None] = queue.Queue()
        self._burst: _Burst | None = None
        self._dead: collections.deque[str] = collections.deque(maxlen=DEAD_ITEMS)
        self._thread: threading.Thread | None = None
        self._closed = False
        route.input = self

    # ----------------------------------------------------------------- listener side

    def wants(self, cfg: RoboEmotConfig) -> bool:
        """Whether an addressed utterance's audio would go now (cheap, any thread: no network
        and no lock held across it): audio input with its opt-in and gate, a session open and
        upload budget left. The route's send gate decides for real."""
        rt = cfg.voice.realtime_route
        return (
            rt.enabled
            and rt.input == "audio"
            and rt.audio_gate == GATE
            and cfg.privacy.realtime_audio
            and self.route.connected
            and self.route.upload_left_s(cfg) > 0
        )

    def submit(self, samples: np.ndarray, rate_hz: int) -> InputJob:
        """Queue one utterance's audio (float samples at ``rate_hz``, the robot's rate); never
        blocks. The returned job ends with the server's transcript, or without one."""
        job = InputJob(samples, rate_hz)
        with self._lock:
            if self._closed:
                job._end(refused="closed")
                return job
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._loop, name="realtime-input", daemon=True
                )
                self._thread.start()
        self._queue.put(job)
        return job

    def settled(self, job: InputJob) -> bool:
        """Whether ``job`` is over: done, or its session is gone (then it ends without a
        transcript). The listener asks this on each poll."""
        if job.done:
            return True
        if job.generation is not None and (
            not self.route.connected or self.route.generation != job.generation
        ):
            self._close(job)
        return job.done

    def cancel(self, job: InputJob) -> None:
        """The listener stopped waiting for ``job`` (its deadline, a newer utterance, a stop):
        it ends without a transcript, and its server items are deleted (those still to come as
        their transcripts arrive)."""
        job.cancelled = True
        self._close(job)

    def close(self) -> None:
        """Stop the thread; a job in flight ends without a transcript."""
        with self._lock:
            self._closed = True
            thread, self._thread = self._thread, None
            burst = self._burst
        if burst is not None:
            self._close(burst.job)
        self._queue.put(None)
        if thread is not None:
            shutdown.join(thread, JOIN_S)

    # ----------------------------------------------------------------- sending thread

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._send(job)
            except Exception as exc:  # one bad burst must not end the input
                log.warning("realtime input: %s", type(exc).__name__)
                self._close(job)

    def _send(self, job: InputJob) -> None:
        samples = job._samples
        if job.done or samples is None:
            return
        if self._speaking():
            job._end(refused="speaking")
            self.route.count("audio_refused")
            return
        cfg = self.route.stack.effective()
        rt = cfg.voice.realtime_route
        zeros = np.zeros(round(job.rate_hz * rt.close_silence_ms / MS_PER_S), np.float32)
        pcm = to_int16(np.concatenate([samples, zeros]))
        instructions = self.route.base_instructions()
        burst = _Burst(job, self.route.generation)
        with self._lock:
            self._burst = burst  # before the send: its first event may come at once
        job.generation = burst.generation
        job.sent = True  # before the send: a cancel meanwhile must count it as gone
        why = self.route.send_audio(pcm, gate=GATE, instructions=instructions)
        if why is not None:
            with self._lock:
                if self._burst is burst:
                    self._burst = None
            job.generation, job.sent = None, False
            job._end(refused=why)
            return
        # one burst at a time: the next waits until this one is over (the listener cancels it
        # at its deadline; this wait is the backstop)
        if not job.wait(rt.transcript_timeout_s):
            self._close(job)
        with self._lock:
            if self._burst is burst:
                self._burst = None

    # ----------------------------------------------------------------- client thread

    def on_event(self, event: protocol.ServerEvent) -> None:
        """An input event of the current session (``speech_started``/``stopped``, input
        transcripts): it belongs to the burst in flight; a transcript of no burst is deleted."""
        delete: str | None = None
        finished: tuple[InputJob, Heard | None, list[str]] | None = None
        with self._lock:
            burst = self._burst
            live = burst is not None and not burst.closed
            if isinstance(event, protocol.SpeechBoundary):
                if live:
                    assert burst is not None
                    if event.item_id not in burst.order:
                        burst.order.append(event.item_id)
                    if event.type == protocol.SPEECH_STARTED:
                        burst.open.add(event.item_id)
                    else:
                        burst.open.discard(event.item_id)
                elif burst is not None and event.item_id is not None:
                    self._dead.append(event.item_id)
                return
            if not isinstance(event, protocol.InputTranscript) or not event.final:
                return
            if not live or event.item_id in self._dead:
                delete = event.item_id
            else:
                assert burst is not None
                if event.item_id not in burst.order:
                    burst.order.append(event.item_id)
                burst.open.discard(event.item_id)
                burst.texts[event.item_id] = event.text
                if all(i in burst.texts for i in burst.order) and not burst.open:
                    burst.closed = True
                    finished = self._heard(burst)
        if delete is not None:
            self._delete([delete])
        if finished is not None:
            job, heard, empty = finished
            if heard is not None:
                self.route.count("transcripts")
                self.route.count("audio_items", len(heard.items))
            job._end(heard)
            self._delete(empty)

    def _heard(self, burst: _Burst) -> tuple[InputJob, Heard | None, list[str]]:
        """The burst's result (lock held): the joined text and items, or None (and its items
        to delete) when the server heard no words."""
        items = [i for i in burst.order if i is not None]
        text = " ".join(t for i in burst.order if (t := burst.texts.get(i, "").strip()))
        if not text:
            return burst.job, None, items
        return burst.job, Heard(text, tuple(items), burst.generation), []

    def _close(self, job: InputJob) -> None:
        """End ``job`` without a transcript; its burst's items are deleted, now for those
        heard and on arrival for the others."""
        with self._lock:
            burst = self._burst if self._burst is not None and self._burst.job is job else None
            heard: list[str] = []
            if burst is not None and not burst.closed:
                burst.closed = True
                heard = [i for i in burst.texts if i is not None]
                self._dead.extend(i for i in burst.order if i is not None and i not in heard)
        job._end()
        self._delete(heard)

    def _delete(self, items: list[str | None]) -> None:
        """Delete user items the robot will not answer through the route (no data: ungated)."""
        for item in items:
            if item is not None and self.route.send_control(protocol.item_delete(item)):
                self.route.count("audio_items_deleted")
