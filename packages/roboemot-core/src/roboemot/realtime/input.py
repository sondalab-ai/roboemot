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

Live streams (``voice.realtime_route.turns: server``, SPEC 8 and 12): the listener opens a
stream once a speech segment passed its gates (``open_stream``, the pre-roll and any held
audio), hands over each piece of the segment as it is heard (``stream``) and closes it at the
segment's end (``close_stream``: ``close_silence_ms`` of zeros, so the server's VAD ends the
turn; no commit). A segment cut part way (the robot starts speaking, a chunk gap, a pause) gets
the closing silence too and its job is cancelled (``cut_stream``): its items are deleted. The
pieces go in order on a thread of their own, each through the route's send gate
(``RealtimeRoute.stream_audio``); a piece refused part way cuts the stream there. A stream is a
burst whose audio arrives in pieces: its start is read with its first piece, its length is
known at its close, and its items and transcripts are collected the same way, except that
several streams may wait for their transcripts at once (a person who pauses and goes on). The
server's buffer is in the order the streams were sent, so a ``speech_started`` belongs to the
oldest stream the server has not covered yet, and a stream's start is where the server's buffer
ends when it becomes that oldest one. A stream that ended without a result and is still not
covered ``transcript_timeout_s`` later resets the session, as a burst does; one the server heard
nothing of is dropped then.

The server answers a stream on its own (``create_response``, S1b): each ``speech_stopped`` of a
stream is owed one response, and the route asks at each ``response.created`` what it answers
(``claim``): a stream that ended whole (``ANSWER``: its items then leave with the server turn,
never deleted here), a fragment (``FRAGMENT``: cut, cancelled, revoked, still streaming, or
ended by the server before the segment's end) or a second answer (``DUPLICATE``). A cut and a
close mark the stream before its closing silence goes, so the server's answer to that silence
is judged on them. While a stream streams or owed speech is fresh (``unsettled``), the robot's
own requests wait.

A barge-in's stream (``playback``, S2) is opened while the robot speaks: its pieces are not held
for the robot's speech (every other check holds), and the server's ``speech_started`` for it
calls ``on_confirm`` (the barge-in's confirmation), outside the lock.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import shutdown
from ..config.schema import RoboEmotConfig
from ..voice.asr import to_int16
from . import protocol
from .route import (
    ANSWER,
    DUPLICATE,
    FRAGMENT,
    STREAM_GATE,
    WAIT,
    close_samples,
    server_turns,
    streams_allowed,
    turns_fallback,
)

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
        self.sent = False  # its burst (or a stream's opening piece) left the robot
        self.cancelled = False
        self.stream = False  # a live stream's job (turns: server), not a burst's
        self.epoch = 0  # a stream's: the input's revocation epoch when it was opened
        # a stream's segment as the listener heard it: its start and its last speech (stamps of
        # the listener's clock), set when it ended; the server turn's latency starts there
        self.start_stamp: float | None = None
        self.speech_end: float | None = None
        # a server-created response answers it, or its words joined a turn (set by the worker
        # once it took them): its items are that turn's, never deleted here
        self.answered = False
        # what its opening showed the server (the worker's: skills, choices, conversation state)
        self.shown: Any = None
        self.claimed = False  # a server response was claimed for it: a second is a duplicate
        self.judged: str | None = None  # the verdict on a response held for it (``_judge``)
        # a stream opened during the robot's speech by a barge-in (S2): its pieces go while the
        # robot speaks, and the server's speech_started for it confirms the barge-in
        self.playback = False
        self.confirmed = False  # the server's speech_started for that stream came (S2)
        # dropped unconfirmed by the listener at its transcript wait's end (S3): a later
        # speech_started is late (``playback_unconfirmed_late``) and confirms nothing
        self.dropped = False

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
    # a live stream: its start is not known until it is the oldest one (None); when it ended
    # without a result (the input's clock), so a stream the server never covers is swept
    placed: bool = True
    ended_at: float | None = None
    sent_ms: float = 0.0  # a stream's audio sent so far (its length when it is cut)
    cut: bool = False  # a stream cut part way: set before its closing silence, never answered
    # server turns (S1b): when it ended or was cut (the input's clock: the server may still
    # answer it for server_answer_wait_s); a response the server created for it, and where
    # the server ended it, while it still streamed (judged when it ends: ``_judge``)
    end_at: float | None = None
    held_end: int | None = None
    holding: bool = False

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
        if not self.placed:
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


@dataclass
class _Owed:
    """Stream speech the server ended (its ``speech_stopped``) on a session that answers on its
    own: the response it may create for it (``claim``). ``stream``: the stream that holds it
    (None: none does); ``end_ms``: where the server ended it; ``at``: when (the input's
    clock)."""

    generation: int
    stream: _Burst | None
    end_ms: int | None
    at: float


@dataclass(frozen=True)
class Claim:
    """What a server-created response answers (``RealtimeInput.claim``): ``verdict``
    (``ANSWER``, ``FRAGMENT`` or ``DUPLICATE``), the stream's job and its items so far."""

    verdict: str
    job: Any = field(default=None, repr=False)
    items: tuple[str, ...] = field(default=(), repr=False)


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
        self._stream_reopening = False  # ... the same for a stream's opening
        self._idless = -1  # the last session that reported an input item without an id
        self._late_confirms = 0  # confirmations of dropped streams, counted after the lock
        # where the server's input buffer ends, as (session, its last audio_end_ms)
        self._end: tuple[int, float] = (-1, 0.0)
        self._reset_gen = -1  # the last session this input reset: its late events are dropped
        self.held = 0  # bursts that waited for a realtime turn to end
        # live streams (turns: server) still collecting the server's items, oldest first; a
        # session's input events go to them while it has some and no burst (_streams_hold)
        self._streams: list[_Burst] = []
        self._ops: queue.Queue[tuple[str, InputJob, Any] | None] = queue.Queue()
        self._stream_thread: threading.Thread | None = None
        self._clock: Callable[[], float] = time.monotonic  # when a stream ended (the sweep)
        # bumped by a reload that no longer allows streams: an open queued before it is refused
        self._epoch = 0
        # stream speech the server ended, oldest first, each waiting for the response the
        # server creates for it on its own (turns: server; ``claim``)
        self._owed: list[_Owed] = []
        # the barge-in's confirmation (S2): called (client thread, no lock held) when the
        # server's VAD starts speech of a stream opened during the robot's speech
        self.on_confirm: Callable[[InputJob], None] | None = None
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
            and not server_turns(cfg)  # the server ends the turns: audio only streams
            and rt.audio_gate in (GATE, STREAM_GATE)  # robot turns (a fallback included)
            and cfg.privacy.realtime_audio
            and not self._asleep()
            and self.route.connected
            and self.route.upload_left_s(cfg) > 0
        )

    def stream_refusal(self, cfg: RoboEmotConfig, *, playback: bool = False) -> str | None:
        """Why a speech segment that starts now cannot stream (``turns: server``; cheap, any
        thread, no lock held across it), None when it may: the route on with server turns, the
        opt-ins (``turns: server`` without ``privacy.realtime_audio_unaddressed`` is
        ``no_unaddressed_opt_in``: robot turns run), ``audio_gate: server_gates``, the robot
        awake (always) and silent, a session open, no realtime turn in flight and upload
        budget left beyond the closing silence. ``playback``: a barge-in's stream during the
        robot's speech (S2), the same checks but the robot's silence. The route's send gate
        decides each piece for real."""
        rt = cfg.voice.realtime_route
        if not rt.enabled:
            return "turns"
        if not server_turns(cfg):
            return turns_fallback(cfg) or "turns"
        if not cfg.privacy.realtime_audio:
            return "no_opt_in"
        if rt.audio_gate != STREAM_GATE:
            return "gate"
        if self._asleep():
            return "asleep"
        if not playback and self._speaking():
            return "speaking"
        if not self.route.connected:
            return "no_session"
        if self.route.input_held():
            return "turn"
        if self.route.upload_left_s(cfg) <= rt.close_silence_ms / MS_PER_S:
            return "upload_budget"
        return None

    def open_stream(
        self,
        samples: np.ndarray,
        rate_hz: int,
        *,
        clock: Callable[[], float] | None = None,
        playback: bool = False,
    ) -> InputJob:
        """Open a live stream with its first audio (float samples at ``rate_hz``: the pre-roll
        and any audio held for the gates); never blocks. The job ends with the server's
        transcript of the segment, or without one (refused, cut, cancelled, the session
        lost). ``playback``: a barge-in opened it during the robot's speech (S2): its pieces
        are not held while the robot speaks, and its server ``speech_started`` confirms the
        barge-in (``on_confirm``)."""
        job = InputJob(np.zeros(0, np.float32), rate_hz, None, clock)
        job.stream, job.epoch, job.playback = True, self._epoch, playback
        return self._op("open", job, samples)

    def stream(self, job: InputJob, samples: np.ndarray) -> None:
        """One more piece of the stream's audio, as it was heard (never blocks)."""
        self._op("audio", job, samples)

    def close_stream(self, job: InputJob, utt_ms: float, deadline: float | None) -> None:
        """The segment ended: ``utt_ms`` of it is speech (the pre-roll included, the end silence
        not); the closing silence goes and the job waits for its transcript, until ``deadline``
        (the listener's clock) at most."""
        job.deadline = deadline
        self._op("close", job, utt_ms)

    def cut_stream(self, job: InputJob) -> None:
        """The segment was cut part way: the closing silence goes (the server ends the turn)
        and the job is cancelled, so its items are deleted."""
        job.cancelled = True
        self._op("cut", job, None)

    def _op(self, kind: str, job: InputJob, arg: Any) -> InputJob:
        with self._lock:
            if self._closed:
                job._end(refused="closed")
                return job
            if self._stream_thread is None:
                self._stream_thread = threading.Thread(
                    target=self._stream_loop, name="realtime-stream", daemon=True
                )
                self._stream_thread.start()
        self._ops.put((kind, job, arg))
        return job

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

    def drop_unconfirmed(self, job: InputJob) -> bool:
        """The listener's transcript wait for a barge-in's stream ended (S3): drop it unless
        the server confirmed it. Decided under the lock the confirmation is set under, so a
        ``speech_started`` that lands just after is late and confirms nothing (counted
        ``playback_unconfirmed_late``). True when dropped."""
        with self._lock:
            if job.confirmed:
                return False
            job.dropped = True
            return True

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
            stream_thread = self._stream_thread
            streams = list(self._streams)
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job._end(refused="closed")
        while True:
            try:
                op = self._ops.get_nowait()
            except queue.Empty:
                break
            if op is not None:
                op[1]._end(refused="closed")
        if burst is not None:
            self._close(burst.job)
        for stream in streams:
            self._close(stream.job)
        self._queue.put(None)
        self._ops.put(None)
        if thread is not None:
            shutdown.join(thread, JOIN_S)
        if stream_thread is not None:
            shutdown.join(stream_thread, JOIN_S)

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

    # ----------------------------------------------------------------- stream thread

    def _stream_loop(self) -> None:
        while True:
            op = self._ops.get()
            if op is None:
                return
            kind, job, arg = op
            try:
                self._stream_op(kind, job, arg)
            except Exception as exc:  # one bad piece must not end the input
                log.warning("realtime input: stream: %s", type(exc).__name__)
                self._close(job)

    def _stream_op(self, kind: str, job: InputJob, arg: Any) -> None:
        """One stream operation, in the order the listener queued them (stream thread). A job
        opened before a reload that forbade streams (``_epoch``) sends nothing more."""
        if kind == "save":  # the day's upload count, off the listener thread (disk)
            self.route.save_upload()
            return
        if self._closed:
            job._end(refused="closed")
            return
        if job.epoch != self._epoch:
            if kind == "open":
                job.cancelled = True
                job._end(refused="revoked")
            return
        cfg = self.route.stack.effective()
        if kind == "open":
            self._sweep(cfg)
            self._open(job, arg, cfg)
            return
        with self._lock:
            burst = next((s for s in self._streams if s.job is job), None)
        if burst is None or burst.closed:  # refused at its opening, or over: nothing more goes
            return
        if kind == "audio":
            if burst.utt_ms != math.inf:
                return  # closed already
            why = self.route.stream_audio(
                self._pcm(arg), hold=self._hold(job), generation=burst.generation
            )
            if why is None:
                burst.sent_ms += arg.size / job.rate_hz * MS_PER_S
            else:  # refused part way (the robot speaks, the budget, the opt-in, a new session)
                self._cut(job, burst, cfg)
            return
        if kind == "cut" or job.done:  # a cut, or a close of a job cancelled meanwhile
            self._cut(job, burst, cfg)
            return
        with self._lock:  # before the closing silence: the server ends the turn on it (claim)
            burst.utt_ms = float(arg)
            burst.end_at = self._clock()
        self._judge(burst)  # a response held while it streamed: its end is known now
        self._closing(job, burst, cfg)  # "close": the segment ended
        with self._lock:
            finished, delete, undeletable = self._results([burst])
        self._conclude(finished, delete, undeletable)

    def _open(self, job: InputJob, samples: np.ndarray, cfg: RoboEmotConfig) -> None:
        """The stream's opening piece: ``session.update`` with fresh instructions, then its
        first audio. Its start is where the server's buffer ends when no other stream of the
        session waits; otherwise it is placed once the older ones are covered. After this
        input reset the session, it waits for the reset's reconnect while that is in flight
        (``audio_reopen_wait``, at most ``connect_timeout_s``), as a burst does; any other
        refusal (a realtime turn in flight, no session) ends the job unsent."""
        rt = cfg.voice.realtime_route
        self._dead_limit = rt.transcript_max_items
        burst = _Burst(
            job,
            -1,
            utt_ms=math.inf,  # known at its close
            silence_ms=rt.server_silence_ms,
            max_items=rt.transcript_max_items,
            max_chars=rt.transcript_max_chars,
        )

        def track(generation: int) -> None:  # under the route's lock, with the send
            job.generation, job.sent = generation, True
            with self._lock:
                session, end_ms = self._end
                burst.generation = generation
                if any(s.generation == generation for s in self._streams):
                    burst.placed = False
                else:
                    burst.start_ms = end_ms if session == generation else 0.0
                # with the send: a revoke from now on knows the stream holds audio (its reset)
                burst.sent_ms = samples.size / job.rate_hz * MS_PER_S
                self._streams.append(burst)  # before the send: its first event may come at once

        self._await_reopen(job, rt.connect_timeout_s)
        instructions, tools = self.route.stream_session(job)  # what the server's own answer reads
        why = self.route.stream_audio(
            self._pcm(samples),
            instructions=instructions,
            tools=tools,
            hold=self._hold(job),
            track=track,
            interrupt=job.playback,  # S2: only a barge-in's stream lets the server interrupt
        )
        if why is not None:
            job._end(refused=why)
            return
        with self._lock:
            ended = job.done and not burst.closed
        if ended:  # cancelled between its send and here: it never got its burst retired
            self._cut(job, burst, cfg)

    def _await_reopen(self, job: InputJob, limit_s: float) -> None:
        """Wait (bounded) for the reconnect of a reset this input made, once."""
        if self._stream_reopening and not self.route.connected and self.route.opening():
            self.route.count("audio_reopen_wait")
            end = self._clock() + limit_s
            while not self.route.connected and self.route.opening() and self._clock() < end:
                if job.wait(HOLD_POLL_S) or self._closed:
                    break
        self._stream_reopening = False

    def _hold(self, job: InputJob) -> Callable[[], str | None]:
        """What stops a piece (under the route's lock, with the send: leaf locks only). Asleep
        always does: it is no configurable gate."""

        def hold() -> str | None:
            if job.done or job.cancelled or self._closed or job.epoch != self._epoch:
                return "cancelled"
            if self._asleep():
                return "asleep"
            if not job.playback and self._speaking():  # a barge-in's stream goes over it (S2)
                return "speaking"
            return None

        return hold

    def _closing(self, job: InputJob, burst: _Burst, cfg: RoboEmotConfig) -> bool:
        """``close_silence_ms`` of zeros on the stream's session, so the server's VAD ends the
        turn (no sound: not held while the robot speaks; the send gate and the budget still
        apply). Whether it went."""
        zeros = np.zeros(close_samples(job.rate_hz, cfg))
        why = self.route.stream_audio(self._pcm(zeros), closing=True, generation=burst.generation)
        return why is None

    def _cut(self, job: InputJob, burst: _Burst, cfg: RoboEmotConfig) -> None:
        """End a stream part way: its closing silence, then the job ends without a transcript
        and its items are deleted. Without the closing silence the server would keep the audio
        in its input buffer for the next commit, and on a session whose items have no id the
        cut turn's transcript could not be deleted: the session is reset instead. It is marked
        cut first: a response the server creates on that closing silence answers a fragment
        (``claim``) and never plays."""
        with self._lock:
            burst.cut = True
            burst.end_at = self._clock()
        self._judge(burst)  # a response held while it streamed answers a fragment
        closed = self._closing(job, burst, cfg)
        with self._lock:
            burst.utt_ms = burst.sent_ms
            idless = self._idless == burst.generation
        if self._close(job):  # counted once: a revoke may have closed it meanwhile
            self.route.count("streams_cut")
        if not closed or idless:
            self._reset_for(burst, force=True)

    def _reset_for(self, burst: _Burst, *, force: bool = False) -> None:
        """Reset the session of a stream that sent audio its server will never end (``force``:
        no closing silence went, or its items cannot be deleted) or that is not covered: its
        buffered audio and late items die with the session (as a burst's)."""
        with self._lock:
            needed = burst.sent_ms > 0 and (force or not burst.covered())
        if needed and self.route.reset_session(burst.generation):
            self._stream_reopening = True
            with self._lock:
                self._reset_gen = max(self._reset_gen, burst.generation)
                self._dead = {k: None for k in self._dead if k[0] > burst.generation}
                self._dead_anon = 0
                self._streams = [s for s in self._streams if s.generation > burst.generation]

    @staticmethod
    def _pcm(samples: np.ndarray) -> bytes:
        return to_int16(np.asarray(samples, np.float32))

    def _sweep(self, cfg: RoboEmotConfig) -> None:
        """Before a stream opens: a stream of a session that is gone is dropped; one that ended
        without a result ``transcript_timeout_s`` ago is dropped when the server heard nothing
        of it, and resets the session when it is still not covered (its late items could not be
        told from the new stream's)."""
        timeout = cfg.voice.realtime_route.transcript_timeout_s
        generation = self.route.generation
        with self._lock:
            now = self._clock()
            stale = {
                id(s)
                for s in self._streams
                if s.closed and s.ended_at is not None and now - s.ended_at >= timeout
            }
            reset = next(
                (s for s in self._streams if id(s) in stale and s.order and not s.covered()), None
            )
            self._streams = [
                s
                for s in self._streams
                if s.generation == generation and not (id(s) in stale and not s.order)
            ]
        if reset is not None:
            self._reset_for(reset)

    # ----------------------------------------------------------------- server turns (S1b)

    def claim(self, generation: int, max_age_s: float) -> Claim | None:
        """A ``response.created`` on ``generation``'s session (client thread): what the server
        created it for. The owed stream speech it ended longest ago (``speech_stopped``), and:

        - ``ANSWER``: that stream ended whole (``_whole``);
        - ``WAIT``: it still streams (the server ended its turn on the trailing silence before
          the listener closed it): the response is held and judged when it ends (``_judge``);
        - ``FRAGMENT``: cut, cancelled, revoked, ended by the server inside the segment, or no
          stream holds it; also, failing closed, when more than one owed entry is fresh (a
          response the server skipped makes order matching unsafe);
        - ``DUPLICATE``: a response was already claimed for that stream (one still held for it
          is released as a fragment instead: the server ended the stream again later, so the
          held one answered part of it, and this one is judged in its place).

        None when no stream speech waits: the robot's own request, or a second response. Speech
        the server ended ``max_age_s`` (``server_answer_wait_s``) ago or more waits no longer."""
        now = self._clock()
        with self._lock:
            self._owed = [
                o for o in self._owed if o.generation == generation and now - o.at < max_age_s
            ]
            if not self._owed:
                return None
            ambiguous = len(self._owed) > 1
            owed = self._owed.pop(0)
            s = owed.stream
            if ambiguous or s is None:
                return Claim(FRAGMENT)
            released = s.holding  # the server split it: the held response was a part of it
            if s.job.claimed and not released:
                return Claim(DUPLICATE, s.job)
            items = tuple(k for k in s.order if isinstance(k, str))
            if released:
                s.holding, s.job.judged = False, FRAGMENT
            claim = self._judge_claim(s, owed.end_ms, items)
        if released:
            self.route.judged(s.job, FRAGMENT, items)
        return claim

    def _judge_claim(self, s: _Burst, end_ms: int | None, items: tuple[str, ...]) -> Claim:
        """The verdict on a fresh claim for stream ``s`` (lock held): held while it streams,
        else whole or a fragment; a fragment leaves the stream unclaimed (a later end of the
        same stream may still answer it)."""
        if self._streaming(s):
            s.job.claimed = True
            s.held_end, s.holding, s.job.judged = end_ms, True, None
            return Claim(WAIT, s.job, items)
        if not self._whole(s, end_ms):
            return Claim(FRAGMENT)
        s.job.claimed = True
        return Claim(ANSWER, s.job, items)

    def poison(self, generation: int) -> None:
        """An ``error`` about no client event (client thread): it may be the server declining
        to answer the oldest owed stream speech (an active response), so that entry is
        resolved now: no response is waited for (``unsettled``), and none is matched to it (a
        response that comes anyway finds the next entry, or none: it fails closed)."""
        with self._lock:
            owed = next((o for o in self._owed if o.generation == generation), None)
            if owed is not None:
                self._owed.remove(owed)

    def _streaming(self, s: _Burst) -> bool:
        """Stream ``s`` still streams (lock held): the listener has not ended it yet, and
        nothing cut, cancelled or revoked it."""
        job = s.job
        alive = not (s.cut or s.closed or job.cancelled or job.epoch != self._epoch)
        return alive and s.utt_ms == math.inf

    def _whole(self, s: _Burst, end_ms: int | None) -> bool:
        """Whether the server ended stream ``s`` at its end (lock held): the listener ended the
        segment (its length is known: set before its closing silence goes), it was neither cut
        nor cancelled nor revoked (both flags are set before the closing silence of a cut
        goes), and the server's end lies within ``server_silence_ms`` of the segment's end, as
        for coverage. A server that sends no end position never makes a stream whole; one that
        ended the turn at a pause inside the segment answered a fragment."""
        job = s.job
        if s.cut or job.cancelled or job.epoch != self._epoch or s.utt_ms == math.inf:
            return False
        if not s.placed or end_ms is None:
            return False
        return end_ms >= s.start_ms + s.utt_ms - s.silence_ms

    def _judge(self, s: _Burst) -> None:
        """Stream ``s`` ended (closed, cut, cancelled or revoked): a server response held for it
        (``WAIT``) is judged now, whole or a fragment, and the route told (``judged``)."""
        with self._lock:
            if not s.holding:
                return
            s.holding = False
            verdict = ANSWER if self._whole(s, s.held_end) else FRAGMENT
            s.job.judged = verdict  # the route reads it should it register the hold later
            if verdict == FRAGMENT:
                s.job.claimed = False  # a later end of the same stream may still answer it
            items = tuple(k for k in s.order if isinstance(k, str))
        self.route.judged(s.job, verdict, items)

    def unsettled(self, generation: int, max_age_s: float) -> bool:
        """Whether the server may still create a response on its own on ``generation``'s
        session: owed stream speech less than ``max_age_s`` (``server_answer_wait_s``) old, a
        response held for a stream that still streams, or a stream there that sent audio and is
        not covered yet, while it streams and for ``max_age_s`` after it ended or was cut (the
        server's ``speech_stopped`` for it may still be on its way). The robot's own request
        waits meanwhile (``RealtimeRoute.busy``), so the two are never taken for each other."""
        now = self._clock()
        with self._lock:
            if any(o.generation == generation and now - o.at < max_age_s for o in self._owed):
                return True
            for s in self._streams:
                if s.generation != generation or s.sent_ms <= 0:
                    continue
                if s.holding:
                    return True
                recent = s.end_at is None or now - s.end_at < max_age_s
                if recent and not s.covered():
                    return True
            return False

    # ----------------------------------------------------------------- client thread

    def on_event(self, event: protocol.ServerEvent, generation: int | None = None) -> None:
        """An input event (``speech_started``/``stopped``, input transcripts) of ``generation``'s
        session (None: the current one): it belongs to the burst in flight on that session when
        its position is at or after the burst's start; a transcript of no burst is deleted."""
        gen = self.route.generation if generation is None else generation
        named = isinstance(event, protocol.SpeechBoundary | protocol.InputTranscript)
        if self._streams_hold(gen, event.item_id if named else None):
            self._stream_event(event, gen)
            return
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

    def _streams_hold(self, gen: int, item: str | None) -> bool:
        """Whether an input event of session ``gen`` about ``item`` goes to the live streams: a
        stream of that session holds the item (whatever burst is set: a stream's words never
        join a burst), or the session has streams and no burst. Decided per event, so a mode
        that changes at runtime (the unaddressed opt-in turned off) takes the burst path at its
        next burst."""
        with self._lock:
            streams = [s for s in self._streams if s.generation == gen]
            if item is not None and any(item in s.order for s in streams):
                return True
            burst = self._burst
            if burst is not None and burst.generation == gen:
                return False
            return bool(streams)

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload (the route passes it on): when live streams are no longer allowed (an
        opt-in turned off, robot turns back, another gate), the open ones are cancelled and
        their items deleted on the server; nothing more of them is sent (counted
        ``streams_revoked``), and an open still queued is refused (``_epoch``). The server would
        keep their audio in its input buffer for the next commit, so the session is reset."""
        if streams_allowed(cfg):
            return
        with self._lock:
            self._epoch += 1
            self._owed.clear()  # robot turns: no response is the server's own any more
            streams = [s for s in self._streams if not s.closed]
        revoked = []
        for stream in streams:
            stream.job.cancelled = True
            if self._close(stream.job):  # a cut on the stream thread may have closed it first
                revoked.append(stream)
        for stream in streams:
            self._reset_for(stream, force=True)  # no closing silence goes now
        if revoked:
            self.route.count("streams_revoked", len(revoked))

    def _stream_event(self, event: protocol.ServerEvent, gen: int) -> None:
        """``on_event`` for live streams (module docstring): a start goes to the oldest stream
        of the session the server has not covered, a stop and a transcript to the stream that
        holds the item; an item of no open stream is deleted."""
        delete: list[tuple[str, int]] = []
        undeletable = 0
        with self._lock:
            if gen <= self._reset_gen:
                return  # a reset session's event still in flight: it ended with the session
            streams = [s for s in self._streams if s.generation == gen]
            cut = None
            confirm: InputJob | None = None
            if isinstance(event, protocol.SpeechBoundary):
                confirm = self._stream_boundary(event, gen, streams)
                if event.item_id is None:  # learnt after a cut: that turn cannot be deleted
                    cut = next((s for s in streams if s.closed and s.job.result is None), None)
            elif isinstance(event, protocol.InputTranscript) and event.final:
                item = event.item_id
                owner = next((s for s in streams if item is not None and item in s.order), None)
                if item is None:
                    owner = next((s for s in streams if s.anon_waiting() is not None), None)
                if item is not None and (gen, item) in self._dead:
                    del self._dead[(gen, item)]
                    delete.append((item, gen))
                elif owner is not None and owner.closed and owner.job.answered:
                    pass  # a server turn answered it: its items leave with that turn
                elif owner is None or owner.closed:
                    if item is not None:
                        delete.append((item, gen))
                    else:
                        undeletable += 1
                        self._dead_anon = max(0, self._dead_anon - 1)
                else:
                    self._transcript(owner, item, event.text)
            finished, gone, lost = self._results(streams)
            late, self._late_confirms = self._late_confirms, 0
        if late:
            self.route.count("playback_unconfirmed_late", late)
        if confirm is not None:
            self.route.count("playback_confirms")
        if confirm is not None and self.on_confirm is not None:  # first: the barge-in's stop
            try:
                self.on_confirm(confirm)
            except Exception as exc:
                log.warning("realtime input: barge-in confirm: %s", type(exc).__name__)
        self._conclude(finished, delete + gone, undeletable + lost)
        if cut is not None:
            self._reset_for(cut, force=True)

    def _stream_boundary(
        self, event: protocol.SpeechBoundary, gen: int, streams: list[_Burst]
    ) -> InputJob | None:
        """A ``speech_started``/``stopped`` of session ``gen`` for the live streams (lock
        held). The job of a barge-in's stream (``playback``, S2) when it is that stream's
        start: the server heard the person over the robot, which confirms the barge-in, also
        once the stream was cut (its release) or closed. A start of any older stream is not."""
        item = event.item_id
        if item is None:
            self._idless = gen  # its items cannot be deleted: a cut resets it
        if event.type == protocol.SPEECH_STARTED:
            head = next((s for s in streams if not s.covered()), None)
            if head is not None and not head.placed:  # the older ones are covered: it is next
                session, end_ms = self._end
                head.start_ms, head.placed = (end_ms if session == gen else 0.0), True
            confirm = None
            if head is not None and head.job.playback and head.holds(event.audio_ms):
                if head.job.dropped:  # after the listener dropped it: never stops the robot
                    self._late_confirms += 1  # counted after the lock (never the route's in it)
                else:
                    confirm = head.job
                    confirm.confirmed = True
            if head is not None and head.holds(event.audio_ms):
                if head.closed and len(head.order) > head.max_items:
                    head.overflow = True  # bounded: an ended stream's items are not kept
                else:
                    key: Key = item if item is not None else head.new_anon()
                    head.add(key)
                    head.open.add(key)
                if not head.closed:
                    return confirm
            if item is not None:
                self._bury(item, gen)
            else:
                self._dead_anon += 1
            return confirm
        if event.audio_ms is not None:
            session, end_ms = self._end
            if gen > session:
                self._end = (gen, float(event.audio_ms))
            elif gen == session:
                self._end = (gen, max(end_ms, float(event.audio_ms)))
        owner = next((s for s in streams if item is not None and item in s.order), None)
        if item is None:  # the oldest open item without an id
            for s in streams:
                key = next((k for k in s.order if isinstance(k, tuple) and k in s.open), None)
                if key is not None:
                    s.stopped(key, event.audio_ms)
                    owner = s
                    break
        elif owner is not None:
            owner.stopped(item, event.audio_ms)
        # the server ended a turn: with server turns it creates a response for it on its own
        self._owed.append(_Owed(gen, owner, event.audio_ms, self._clock()))
        while len(self._owed) > self._dead_limit:  # bounded: one per item at most
            self._owed.pop(0)
        return None

    def _results(
        self, streams: list[_Burst]
    ) -> tuple[list[tuple[InputJob, Heard | None]], list[tuple[str, int]], int]:
        """The streams of ``streams`` whose result is in (lock held): too many items or too
        much text ends one without a result, a complete one with its transcript. Returns the
        jobs to end, the items to delete and how many items cannot be."""
        finished: list[tuple[InputJob, Heard | None]] = []
        delete: list[tuple[str, int]] = []
        undeletable = 0
        for s in streams:
            if s.closed:
                continue
            heard: tuple[InputJob, Heard | None] | None = None
            if s.over():
                heard = (s.job, None)
            elif s.complete():
                heard = self._heard(s)
            if heard is None:
                continue
            finished.append(heard)
            if heard[1] is None:
                gone, lost = self._retire(s)
                delete, undeletable = delete + gone, undeletable + lost
                s.ended_at = self._clock()
            else:
                s.closed = True
        self._prune()
        return finished, delete, undeletable

    def _conclude(
        self,
        finished: list[tuple[InputJob, Heard | None]],
        delete: list[tuple[str, int]],
        undeletable: int,
    ) -> None:
        """The side effects of ``_results`` (outside the lock): counts, deletes, the jobs."""
        if undeletable:
            self.route.count("audio_items_undeletable", undeletable)
        self._delete(delete)
        for job, heard in finished:
            if heard is not None:
                self.route.count("transcripts")
                self.route.count("audio_items", len(heard.items))
                if heard.unnamed:
                    self.route.count("audio_items_unnamed", heard.unnamed)
            job._end(heard)

    def _prune(self) -> None:
        """Drop the streams that are over and covered (lock held): no later start is theirs."""
        self._streams = [s for s in self._streams if not (s.closed and s.covered())]

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

    def _close(self, job: InputJob) -> bool:
        """End ``job`` without a transcript; its burst's items are deleted, now for those
        heard and on arrival for the others. Whether this call retired its burst."""
        delete: list[tuple[str, int]] = []
        undeletable = 0
        retired = False
        with self._lock:
            burst = self._burst if self._burst is not None and self._burst.job is job else None
            if burst is None:
                burst = next((s for s in self._streams if s.job is job), None)
            if burst is not None and not burst.closed:
                retired = True
                if burst.utt_ms == math.inf:  # a stream ended part way: its length is what went
                    burst.utt_ms = burst.sent_ms
                    burst.cut = True  # only the listener's close ends a stream whole
                delete, undeletable = self._retire(burst)
                burst.ended_at = self._clock()
                if burst.end_at is None:
                    burst.end_at = burst.ended_at
                self._prune()
        if burst is not None:
            self._judge(burst)  # cancelled or revoked: a held response answers a fragment
        job._end()
        if job.stream and job.sent and not self._closed:
            # the pieces since its opening count after a restart too; disk, so never on the
            # caller's thread (the listener's): the stream thread saves it
            self._ops.put(("save", job, None))
        if undeletable:
            self.route.count("audio_items_undeletable", undeletable)
        self._delete(delete)
        return retired

    def _retire(self, burst: _Burst) -> tuple[list[tuple[str, int]], int]:
        """Close ``burst`` without a result (lock held): its items with a transcript are to be
        deleted now, the others when their transcript comes; returns those to delete and how
        many items without an id cannot be."""
        burst.closed = True
        delete: list[tuple[str, int]] = []
        undeletable = 0
        if burst.job.answered:  # a server turn answered it: its items leave with that turn
            return delete, undeletable
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
