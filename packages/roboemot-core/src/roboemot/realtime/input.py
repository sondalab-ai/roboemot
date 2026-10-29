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

The burst's transcript is collected from the server's events (client thread, ``on_event``,
with the session they are of): every ``speech_started`` whose position in the session's input
buffer (``audio_start_ms``) is at or after the burst's start opens an item of the burst; one
before it belongs to an earlier burst and is deleted, never joined. The burst's start is where
the server says its buffer ends (the session's last ``speech_stopped`` ``audio_end_ms``), read
under this input's lock with the send (``track``): the server keeps no silence after the
speech, so a count of the appended audio runs ahead of it (Pollen's server, 2026-10-29).

A burst is covered once every item it opened has stopped and the last one ended within
``server_silence_ms`` of the utterance's end (``audio_end_ms`` at or after its start plus the
utterance's length, closing silence excluded, less ``server_silence_ms``): any later segment
would need that much silence inside the utterance before it, so the server would have reported
its start already. The job is done once the burst is covered and each item has its
``transcription.completed``; a server that splits the utterance gives several items, joined in
the order the server heard them. A server that sends no ``audio_end_ms`` never covers a burst:
its utterances are decoded locally. A burst that ends without its result and is not covered
resets the session (``RealtimeRoute.reset_session``), so its late items are stale and never
reach the next burst; the next burst waits for the new session while the reset's reconnect is
in flight (``audio_reopen_wait``), within its listener deadline, else it is decoded locally.

A server that leaves ``item_id`` out still gives its text (the items are matched in order), but
such items cannot be deleted: they are counted (``audio_items_unnamed``,
``audio_items_undeletable``). While a cancelled burst's id-less item may still come, an id-less
transcript cannot be told apart from it: the burst in flight then ends without a result
(``audio_items_ambiguous``, a local decode), so a late transcript is never joined into another
burst. More items than ``transcript_max_items`` or text longer than ``transcript_max_chars``
drops the transcript. The listener waits for the job with its own deadline
(``transcript_timeout_s``) and ``cancel``s it past that.

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
HOLD_POLL_S = 0.05  # how often a held burst looks again (a realtime turn, a reconnect)
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
    """The burst in flight: its session, its start in that session's input buffer, the
    utterance's length (``utt_ms``, the closing silence excluded) and the server VAD's end
    silence (``silence_ms``), the server's items in the order heard, those still open, the
    last end the server reported for them, and the transcripts in (``closed``: the job ended,
    later transcripts are deleted). Ids and texts are never in repr."""

    job: Any
    generation: int
    start_ms: float = 0.0
    utt_ms: float = 0.0
    silence_ms: float = 0.0
    last_end_ms: float | None = None
    max_items: int = 0
    max_chars: int = 0
    order: dict[Key, None] = field(default_factory=dict, repr=False)
    open: set[Key] = field(default_factory=set, repr=False)
    texts: dict[Key, str] = field(default_factory=dict, repr=False)
    anon: int = 0
    closed: bool = False
    overflow: bool = False  # more late items after it ended than the cap: never covered

    def holds(self, position: int | None) -> bool:
        """Whether an item that starts at ``position`` of the input buffer is this burst's
        (no position: the burst in flight is the only candidate). Only earlier bursts can lie
        before its start, and a burst that is not covered resets the session, so any later
        position is its own."""
        return position is None or position >= self.start_ms

    def covered(self) -> bool:
        """Whether the server reported all of it (module docstring): an item started, every
        one stopped, and the last end lies within ``silence_ms`` of the utterance's end."""
        if not self.order or self.open or self.overflow or self.last_end_ms is None:
            return False
        return self.last_end_ms >= self.start_ms + self.utt_ms - self.silence_ms

    def stopped(self, key: Key, end_ms: int | None) -> None:
        """Item ``key`` of this burst stopped at ``end_ms`` of the input buffer (if given)."""
        self.open.discard(key)
        if end_ms is not None:
            self.last_end_ms = max(self.last_end_ms or 0.0, float(end_ms))

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
        return bool(self.texts) and all(k in self.texts for k in self.order) and self.covered()


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
        # items of ended bursts still to come, as (session, id): deleted on arrival
        self._dead: dict[tuple[int, str], None] = {}
        self._dead_anon = 0  # items without id of ended bursts still to come
        # how many ended items are remembered (transcript_max_items, set by each burst's send)
        self._dead_limit = 0
        self._thread: threading.Thread | None = None
        self._closed = False
        self._reopening = False  # this input reset the session: a burst waits for the new one
        # where the server's input buffer ends, as (session, its last audio_end_ms)
        self._end: tuple[int, float] = (-1, 0.0)
        self._reset_gen = -1  # the last session this input reset: its late events are dropped
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
            job,
            -1,
            utt_ms=len(samples) / job.rate_hz * MS_PER_S,
            silence_ms=rt.server_silence_ms,
            max_items=rt.transcript_max_items,
            max_chars=rt.transcript_max_chars,
        )

        def hold() -> str | None:  # under the route's lock, with the send
            if job.done or job.expired() or self._closed:
                return "cancelled"
            if self._asleep():
                return "asleep"
            if self._speaking():
                return "speaking"
            return None

        def track(generation: int) -> None:  # under the route's lock, with the send
            job.generation, job.sent = generation, True
            with self._lock:  # the start and the server's end in one step (on_event moves it)
                session, end_ms = self._end
                burst.generation = generation
                burst.start_ms = end_ms if session == generation else 0.0  # a new session: 0
                self._burst = burst  # before the send: its first event may come at once

        waited: set[str] = set()
        while True:
            if self._reopening and not self.route.connected and self.route.opening():
                why: str | None = "reopening"  # this input's reset reconnects: not a refusal
            else:
                why = self.route.send_audio(
                    pcm, gate=GATE, instructions=instructions, hold=hold, track=track
                )
            if why not in ("turn", "reopening"):
                break
            if why not in waited:  # counted once per burst, however long the wait
                waited.add(why)
                if why == "turn":
                    with self._lock:
                        self.held += 1
                self.route.count("audio_held" if why == "turn" else "audio_reopen_wait")
            if job.wait(HOLD_POLL_S) or self._closed or job.expired():
                why = "cancelled"
                break
        self._reopening = False  # only the first burst after a reset waits for the new session
        if why is not None:
            job._end(refused=why)
            return
        # one burst at a time: the next waits until this one is over (the listener cancels it
        # at its deadline; this wait is the backstop)
        if not job.wait(rt.transcript_timeout_s):
            self._close(job)
        with self._lock:
            reset = job.result is None and not burst.covered()
        if reset and self.route.reset_session(burst.generation):
            self._reopening = True  # the next burst waits for the new session (its deadline)
            with self._lock:  # the old session's items die with it
                self._reset_gen = max(self._reset_gen, burst.generation)
                self._dead = {k: None for k in self._dead if k[0] > burst.generation}
                self._dead_anon = 0
        with self._lock:
            if self._burst is burst:
                self._burst = None

    # ----------------------------------------------------------------- client thread

    def on_event(self, event: protocol.ServerEvent, generation: int | None = None) -> None:
        """An input event (``speech_started``/``stopped``, input transcripts) of ``generation``'s
        session (None: the current one): it belongs to the burst in flight on that session when
        its position is at or after the burst's start; a transcript of no burst is deleted."""
        gen = self.route.generation if generation is None else generation
        delete: list[tuple[str, int]] = []
        undeletable = 0
        ambiguous = False
        finished: tuple[InputJob, Heard | None] | None = None
        with self._lock:
            if gen <= self._reset_gen:
                return  # a reset session's event still in flight: it ended with the session
            burst = (
                self._burst if self._burst is not None and self._burst.generation == gen else None
            )
            live = burst if burst is not None and not burst.closed else None
            if isinstance(event, protocol.SpeechBoundary):
                self._boundary(event, gen, burst, live)
            elif isinstance(event, protocol.InputTranscript) and event.final:
                item = event.item_id
                if item is not None and (gen, item) in self._dead:
                    del self._dead[(gen, item)]
                    delete.append((item, gen))
                elif live is None:
                    if item is not None:
                        delete.append((item, gen))
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

    def _boundary(
        self, event: protocol.SpeechBoundary, gen: int, burst: _Burst | None, live: _Burst | None
    ) -> None:
        """A ``speech_started``/``stopped`` of session ``gen`` (lock held). A start at or after
        the burst's start is the burst's item (still counted for its coverage once the burst
        ended, and then to be deleted); any other is an ended burst's. A stop moves where the
        server's buffer ends and, for an item of the burst, its last end."""
        item = event.item_id
        if event.type == protocol.SPEECH_STARTED:
            if burst is not None and burst.holds(event.audio_ms):
                if live is None and len(burst.order) > burst.max_items:
                    burst.overflow = True  # bounded: an ended burst's items are not kept
                else:
                    key: Key = item if item is not None else burst.new_anon()
                    burst.add(key)
                    burst.open.add(key)
                if live is not None:
                    return
            if item is not None:
                self._bury(item, gen)
            else:
                self._dead_anon += 1
            return
        if event.audio_ms is not None:
            session, end_ms = self._end
            if gen > session:
                self._end = (gen, float(event.audio_ms))
            elif gen == session:
                self._end = (gen, max(end_ms, float(event.audio_ms)))
        if burst is None:
            return
        if item is not None:
            if item in burst.order:
                burst.stopped(item, event.audio_ms)
            return
        oldest = next((k for k in burst.order if isinstance(k, tuple) and k in burst.open), None)
        if oldest is not None:  # the oldest open item without an id
            burst.stopped(oldest, event.audio_ms)

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
        self._dead[(generation, item)] = None
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
