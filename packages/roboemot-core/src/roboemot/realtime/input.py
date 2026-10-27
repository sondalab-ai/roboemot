"""Addressed microphone audio to the realtime route (SPEC 8, Realtime speech route; SPEC 12).

With ``voice.realtime_route.input: audio`` and the ``privacy.realtime_audio`` opt-in, an
utterance the local wake gate addressed leaves the robot as audio, after the local decision,
never live. :class:`RealtimeInput` takes it from the listener (``submit``: never blocks) and,
on a thread of its own, sends it through the route's send gate (``RealtimeRoute.send_audio``)
as one burst: the utterance's PCM16 followed by ``close_silence_ms`` of zeros, so the server's
VAD (``server_silence_ms``, less than that) ends the turn within the burst. The gate decides,
with the send and under the route's lock, that the job is still wanted (not cancelled, its
listener deadline not past), that the robot is neither asleep nor speaking, and that no
realtime turn is in flight (the burst then waits: another utterance's words never land inside
a turn's tool round). One burst is in flight at a time.

The burst's transcript is collected from the server's events (client thread, ``on_event``):
every ``speech_started`` whose position in the session's input buffer (``audio_start_ms``)
lies inside the burst's span opens an item of the burst; one outside it belongs to an earlier
(cancelled) burst and is deleted, never joined. The job is done once each item has its
``transcription.completed`` and none is still open; a server that splits the utterance gives
several items, joined in the order the server heard them. A server that leaves ``item_id``
out still gives its text (the items are matched in order), but such items cannot be deleted:
they are counted (``audio_items_unnamed``, ``audio_items_undeletable``). While a cancelled
burst's id-less item may still come, an id-less transcript cannot be told apart from it: the
burst in flight then ends without a result (``audio_items_ambiguous``, a local decode), so a
late transcript is never joined into another burst. More items than
``transcript_max_items`` or text longer than ``transcript_max_chars`` drops the transcript.
The listener waits for the job with its own deadline (``transcript_timeout_s``) and
``cancel``s it past that.

The server keeps the words as a user item of its conversation, which the next
``response.create`` answers. An item of an utterance the robot does not answer through the
route (a cancelled job, a transcript after the cancel, an empty or oversized transcript) is
deleted at once, on the session that holds it, so it is never answered. Nothing here logs or
reports audio, a transcript or an item id.
"""

from __future__ import annotations

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
HOLD_POLL_S = 0.05  # how often a burst held by a realtime turn looks again
ANON = "anon"  # the key of an item the server sent without an id: (ANON, n)

Key = str | tuple[str, int]


@dataclass(frozen=True)
class Heard:
    """What the server heard of one burst: the joined text, its user items (in order) and the
    server session (``generation``) that holds them; ``unnamed`` items had no id (they cannot
    be deleted). Text and ids are never in repr."""

    text: str = field(repr=False)
    items: tuple[str, ...] = field(repr=False)
    generation: int
    unnamed: int = 0


class InputJob:
    """One utterance's audio on its way to the server, read like a ``GrammarJob``: ``done``
    once it has a result (``result``, the :class:`Heard`), was refused before anything was
    sent (``refused``, a reason code) or ended without a transcript (cancelled, the session
    lost). ``ran`` is True only with a result; ``sent`` once its burst left the robot. Past
    ``deadline`` (the listener's ``clock``) it is not sent any more."""

    def __init__(
        self,
        samples: np.ndarray,
        rate_hz: int,
        deadline: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._samples: np.ndarray | None = samples
        self.rate_hz = rate_hz
        self.deadline, self._clock = deadline, clock
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

    def expired(self) -> bool:
        """Its listener deadline passed: the listener no longer waits for it."""
        if self.deadline is None or self._clock is None:
            return False
        return self._clock() >= self.deadline

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
    """The burst in flight: its session and span of that session's input buffer, the server's
    items in the order heard, those still open, and the transcripts in (``closed``: the job
    ended, later transcripts are deleted). Ids and texts are never in repr."""

    job: Any
    generation: int
    start_ms: float = 0.0
    end_ms: float = 0.0
    max_items: int = 0
    max_chars: int = 0
    order: dict[Key, None] = field(default_factory=dict, repr=False)
    open: set[Key] = field(default_factory=set, repr=False)
    texts: dict[Key, str] = field(default_factory=dict, repr=False)
    anon: int = 0
    closed: bool = False

    def holds(self, position: int | None) -> bool:
        """Whether an item that starts at ``position`` of the input buffer is this burst's
        (no position: the burst in flight is the only candidate)."""
        return position is None or self.start_ms <= position < self.end_ms

    def add(self, key: Key) -> None:
        self.order.setdefault(key, None)

    def new_anon(self) -> Key:
        self.anon += 1
        key = (ANON, self.anon)
        self.add(key)
        return key

    def anon_waiting(self) -> Key | None:
        """The first item without an id that has no transcript yet."""
        return next((k for k in self.order if isinstance(k, tuple) and k not in self.texts), None)

    def over(self) -> bool:
        chars = sum(len(t) for t in self.texts.values())
        return len(self.order) > self.max_items or chars > self.max_chars

    def complete(self) -> bool:
        return bool(self.texts) and not self.open and all(k in self.texts for k in self.order)


class RealtimeInput:
    """See the module docstring. ``route`` is the :class:`~roboemot.realtime.route.
    RealtimeRoute` (its input events come here); ``speaking()`` and ``asleep()`` say whether
    the robot speaks or sleeps now (nothing is sent then; both are read under the route's lock,
    so they take only leaf locks)."""

    def __init__(
        self,
        route: Any,
        *,
        speaking: Callable[[], bool] = lambda: False,
        asleep: Callable[[], bool] = lambda: False,
    ) -> None:
        self.route = route
        self._speaking, self._asleep = speaking, asleep
        self._lock = threading.Lock()
        self._queue: queue.Queue[InputJob | None] = queue.Queue()
        self._burst: _Burst | None = None
        # items of ended bursts still to come: id -> its session (deleted on arrival)
        self._dead: dict[str, int] = {}
        self._dead_anon = 0  # items without id of ended bursts still to come
        # how many ended items are remembered (transcript_max_items, set by each burst's send)
        self._dead_limit = 0
        self._thread: threading.Thread | None = None
        self._closed = False
        self.held = 0  # bursts that waited for a realtime turn to end
        route.input = self

    # ----------------------------------------------------------------- listener side

    def wants(self, cfg: RoboEmotConfig) -> bool:
        """Whether an addressed utterance's audio would go now (cheap, any thread: no network
        and no lock held across it): audio input with its opt-in and gate, the robot awake, a
        session open and upload budget left. The route's send gate decides for real."""
        rt = cfg.voice.realtime_route
        return (
            rt.enabled
            and rt.input == "audio"
            and rt.audio_gate == GATE
            and cfg.privacy.realtime_audio
            and not self._asleep()
            and self.route.connected
            and self.route.upload_left_s(cfg) > 0
        )

    def submit(
        self,
        samples: np.ndarray,
        rate_hz: int,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> InputJob:
        """Queue one utterance's audio (float samples at ``rate_hz``, the robot's rate); never
        blocks. The returned job ends with the server's transcript, or without one; past
        ``deadline`` (``clock``'s time) it is not sent."""
        job = InputJob(samples, rate_hz, deadline, clock)
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
        it is never sent if it was not yet, it ends without a transcript, and its server items
        are deleted (those still to come as their transcripts arrive)."""
        job.cancelled = True
        self._close(job)

    def close(self) -> None:
        """Stop the thread: every queued job ends unsent, a job in flight ends without a
        transcript."""
        with self._lock:
            self._closed = True
            thread = self._thread
            burst = self._burst
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job._end(refused="closed")
        if burst is not None:
            self._close(burst.job)
        self._queue.put(None)
        if thread is not None:
            shutdown.join(thread, JOIN_S)

    def _thread_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

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
        if self._closed:
            job._end(refused="closed")
            return
        if job.done or samples is None:
            return
        cfg = self.route.stack.effective()
        rt = cfg.voice.realtime_route
        self._dead_limit = rt.transcript_max_items
        zeros = np.zeros(round(job.rate_hz * rt.close_silence_ms / MS_PER_S), np.float32)
        pcm = to_int16(np.concatenate([samples, zeros]))
        instructions = self.route.base_instructions()
        burst = _Burst(
            job, -1, max_items=rt.transcript_max_items, max_chars=rt.transcript_max_chars
        )

        def hold() -> str | None:  # under the route's lock, with the send
            if job.done or job.expired() or self._closed:
                return "cancelled"
            if self._asleep():
                return "asleep"
            if self._speaking():
                return "speaking"
            return None

        def track(generation: int, start_ms: float, end_ms: float) -> None:  # with the send
            burst.generation, burst.start_ms, burst.end_ms = generation, start_ms, end_ms
            job.generation, job.sent = generation, True
            with self._lock:
                self._burst = burst  # before the send: its first event may come at once

        waited = False
        while True:
            why = self.route.send_audio(
                pcm, gate=GATE, instructions=instructions, hold=hold, track=track
            )
            if why != "turn":
                break
            if not waited:
                waited = True
                with self._lock:
                    self.held += 1
            if job.wait(HOLD_POLL_S) or self._closed:
                why = "cancelled"
                break
        if why is not None:
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
        transcripts): it belongs to the burst in flight when its position is inside the
        burst's span; a transcript of no burst is deleted."""
        delete: list[tuple[str, int]] = []
        undeletable = 0
        ambiguous = False
        finished: tuple[InputJob, Heard | None] | None = None
        with self._lock:
            burst = self._burst
            live = burst if burst is not None and not burst.closed else None
            if isinstance(event, protocol.SpeechBoundary):
                self._boundary(event, live)
            elif isinstance(event, protocol.InputTranscript) and event.final:
                item = event.item_id
                if item is not None and item in self._dead:
                    delete.append((item, self._dead.pop(item)))
                elif live is None:
                    if item is not None:
                        delete.append((item, self.route.generation))
                    else:
                        undeletable += 1
                        self._dead_anon = max(0, self._dead_anon - 1)
                elif item is None and self._dead_anon > 0:
                    # it may be a cancelled burst's late words: never joined into this one;
                    # this burst ends without a result (the listener decodes it locally)
                    self._dead_anon -= 1
                    ambiguous = True
                    finished = (live.job, None)
                    gone, lost = self._retire(live)
                    delete, undeletable = delete + gone, undeletable + lost + 1
                else:
                    self._transcript(live, item, event.text)
            if finished is not None:
                pass  # ended above (ambiguous)
            elif live is not None and live.over():
                finished = (live.job, None)
                gone, lost = self._retire(live)
                delete, undeletable = delete + gone, undeletable + lost
            elif live is not None and live.complete():
                finished = self._heard(live)
                if finished[1] is None:
                    gone, lost = self._retire(live)
                    delete, undeletable = delete + gone, undeletable + lost
                else:
                    live.closed = True
        if ambiguous:
            self.route.count("audio_items_ambiguous")
        if undeletable:
            self.route.count("audio_items_undeletable", undeletable)
        self._delete(delete)
        if finished is not None:
            job, heard = finished
            if heard is not None:
                self.route.count("transcripts")
                self.route.count("audio_items", len(heard.items))
                if heard.unnamed:
                    self.route.count("audio_items_unnamed", heard.unnamed)
            job._end(heard)

    def _boundary(self, event: protocol.SpeechBoundary, live: _Burst | None) -> None:
        """A ``speech_started``/``stopped`` (lock held): credited to the burst in flight when
        its position is inside the burst's span, else the item is an ended burst's."""
        item = event.item_id
        if event.type == protocol.SPEECH_STARTED:
            if live is not None and live.holds(event.audio_ms):
                key: Key = item if item is not None else live.new_anon()
                live.add(key)
                live.open.add(key)
            elif item is not None:
                self._bury(item, self.route.generation if live is None else live.generation)
            else:
                self._dead_anon += 1
            return
        if live is None:
            return
        if item is not None:
            live.open.discard(item)
            return
        oldest = next((k for k in live.order if isinstance(k, tuple) and k in live.open), None)
        if oldest is not None:  # the oldest open item without an id
            live.open.discard(oldest)

    def _transcript(self, live: _Burst, item: str | None, text: str) -> None:
        """A transcript for the burst in flight (lock held); one without an id goes to the
        first item without an id still waiting for its text."""
        key = item if item is not None else (live.anon_waiting() or live.new_anon())
        live.add(key)
        live.open.discard(key)
        live.texts[key] = text

    def _bury(self, item: str, generation: int) -> None:
        """An ended burst's item, deleted when its transcript comes (lock held; bounded: no
        configuration read here, the stack's lock must never be taken under this one)."""
        self._dead[item] = generation
        while len(self._dead) > self._dead_limit:
            self._dead.pop(next(iter(self._dead)))

    def _heard(self, burst: _Burst) -> tuple[InputJob, Heard | None]:
        """The burst's result (lock held): the joined text and items, or None when the server
        heard no words."""
        text = " ".join(t for k in burst.order if (t := burst.texts.get(k, "").strip()))
        if not text:
            return burst.job, None
        items = tuple(k for k in burst.order if isinstance(k, str))
        unnamed = sum(1 for k in burst.order if isinstance(k, tuple))
        return burst.job, Heard(text, items, burst.generation, unnamed)

    def _close(self, job: InputJob) -> None:
        """End ``job`` without a transcript; its burst's items are deleted, now for those
        heard and on arrival for the others."""
        delete: list[tuple[str, int]] = []
        undeletable = 0
        with self._lock:
            burst = self._burst if self._burst is not None and self._burst.job is job else None
            if burst is not None and not burst.closed:
                delete, undeletable = self._retire(burst)
        job._end()
        if undeletable:
            self.route.count("audio_items_undeletable", undeletable)
        self._delete(delete)

    def _retire(self, burst: _Burst) -> tuple[list[tuple[str, int]], int]:
        """Close ``burst`` without a result (lock held): its items with a transcript are to be
        deleted now, the others when their transcript comes; returns those to delete and how
        many items without an id cannot be."""
        burst.closed = True
        delete: list[tuple[str, int]] = []
        undeletable = 0
        for k in burst.order:
            if isinstance(k, tuple):
                if k in burst.texts:
                    undeletable += 1
                else:
                    self._dead_anon += 1
            elif k in burst.texts:
                delete.append((k, burst.generation))
            else:
                self._bury(k, burst.generation)
        return delete, undeletable

    def _delete(self, items: list[tuple[str, int]]) -> None:
        """Delete user items the robot will not answer through the route, each on the session
        that holds it (an item of a session that is gone is left alone; no data: ungated)."""
        for item, generation in items:
            sent = self.route.delete_items(generation, [item])
            if sent:
                self.route.count("audio_items_deleted", sent)
