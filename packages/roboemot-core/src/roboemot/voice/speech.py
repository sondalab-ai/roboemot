"""Speech output: sentences in, audio on the speaker out (SPEC 8, stage 7; 13 latency budgets).

:class:`SpeechOutput` owns a thread with an event loop (the pattern of the persona traits
worker), so the caller, the conversation worker or the app tick, never blocks on synthesis.
``say`` composes the sentence's :class:`~roboemot.voice.prosody.Delivery` from the voice
configuration, the persona effects, the register, the circadian state and the mood the caller
passes, plus the sentence's variation (seeded by the utterance id and the sentence's index in it;
a question is raised; the rest of a sentence the splitter cut keeps the delivery of its first
piece), queues it, and the loop streams it through the backend into the sink,
which resamples (the robot's speaker). ``stop`` is the barge-in path: it cuts the sink first,
then cancels the synthesis in flight and drops the queue; an epoch counter makes any chunk that
was already on its way fall on the floor, so nothing plays after ``stop`` returns.

``play`` queues raw samples (the self-soothing sounds, SPEC 7B) through the same queue, epoch and
push path: a sound never interleaves with a sentence, ``stop`` cuts it, and it is not speech (it
leaves ``ended_at``, ``question_end``, the counters and the speech timeline alone, so speaking
motion ignores it). ``cancel`` drops one such job only.

The bus gets ``speech.say`` with ids, lengths, language and mood, never the text (SPEC 12).

For the debug report (SPEC 12A) the output measures, for any backend:

- the real-time factor of each chunk (time waited for it since the previous push, or since the
  request, over its audio duration) and of each finished sentence (all its waits over all its
  audio; a backend that synthesizes a sentence whole, like Piper, puts all of it on the first
  chunk, so the per-sentence value is the one that says whether synthesis keeps up);
- underruns: a push of a sentence that already pushed audio, arriving after the sink's
  ``playing_until`` (the speaker ran dry in the middle of a sentence), with the gap in ms. Only
  here is it known that a sentence has more audio to come;
- late sentences: the next sentence of one answer arriving after the previous one ended by more
  than its planned pause (the silence beyond the pause, in ms), and ``boundary_dry``, how often the
  sink ran dry at all between two sentences of one answer;
- ``push_ms``: the time spent in the sink's push (the speaker resamples there);
- ``timeline_ms``: the time each push spends writing the speech timeline (wall time, under the
  lock ``stop`` takes), so its cost is visible on the robot.

A sink with no finite ``playing_until`` (its output never opened) counts no gaps. Numbers only.

Barge-in ducking (SPEC 8 stage 8): with a :class:`~roboemot.voice.duck.DuckGain` (``duck``, set
when barge-in runs) every push goes to the sink with the duck's gain on it, keyed to the time the
push starts to play, so the speaker's echo reference records the audio as it really plays. The
speech timeline gets the samples *before* the gain: the speaking motion follows the sentence as
spoken, not the duck, so a duck neither stops nor shrinks it. ``stop`` clears the duck.

``stream`` plays audio that arrives already synthesized, in pieces, from another thread (the
realtime speech route's reply, SPEC 8, Realtime speech route): a :class:`StreamFeed` the caller
fills. The stream job sits in the same queue, under the same epoch, and pushes through the same
``_push(..., speech=True)`` path as a sentence, so the duck, the speech timeline, the echo
reference (the sink's), underruns, ``played``, ``ended_at`` and ``question_end`` work unchanged.
It holds ``voice.realtime_route.prebuffer_ms`` of audio before its first push (network jitter),
applies the caller's volume as gain, and counts as one sentence. Its question flag can come late
(``mark_question``: the reply's transcript arrives while its audio plays). ``stop`` cuts it like a
sentence, and its feed then refuses more audio.

Every push also feeds :attr:`SpeechOutput.timeline` (:mod:`roboemot.voice.envelope`): energy
frames, sentence bounds, phrase gaps and emphasis peaks on the playback clock, for the speaking
motion. ``stop`` truncates it under the same lock, so nothing after a barge-in stays on it.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .. import shutdown
from ..bus import Bus
from ..config.schema import RegisterParams, RoboEmotConfig, StateEffects
from ..persona import effects
from ..soothe.synth import db_to_gain
from ..stats import P50, P90, summarise
from .duck import DuckGain
from .envelope import SpeechTimeline
from .prosody import Delivery, compose_delivery
from .tts import NO_VOICE, AudioChunk, TTSBackend, TTSError

log = logging.getLogger(__name__)

MS_PER_S = 1000.0
LATENCY_WINDOW = 20
PLAYED_WINDOW = 64  # sentences whose playback start is remembered, for ``played``
READY_TIMEOUT_S = 5.0
JOIN_TIMEOUT_S = 2.0
RTF_LEVELS = {"p50": P50, "p90": P90}
SAID_UTTERANCES_MAX = 64  # answers whose sentence count and last delivery are remembered


PCM16_FULL_SCALE = 32768.0  # int16 sample to float in [-1, 1)
QUESTION_MARKS = ("?", "？")
CLOSERS = "\"')]}»”’›"


def ends_with_question(text: str) -> bool:
    """Whether a sentence ends in a question mark (closing quotes and brackets and spaces after
    it are ignored). This is how the robot knows it just asked something."""
    return text.rstrip().rstrip(CLOSERS).rstrip().endswith(QUESTION_MARKS)


class AudioSink(Protocol):
    """What plays the audio: the robot's speaker (it resamples to the output rate itself).
    ``playing_until`` is the monotonic stamp at which everything pushed so far has played.
    ``end_stream`` is called after each sentence (also a failed one, not one a stop cut): a sink
    that resamples as a stream flushes its held-back tail (a no-op for one that does not)."""

    def push_audio_sample(self, samples: np.ndarray, rate_hz: int) -> float: ...
    def playing_until(self) -> float: ...
    def end_stream(self) -> float: ...
    def stop(self) -> None: ...


@runtime_checkable
class DuckingSink(Protocol):
    """An ``AudioSink`` that applies barge-in's duck itself, where audio leaves for the player
    (a pacing speaker, at its hand-off: the duck then reaches all but its lead). ``take_duck``
    returns True when it does; the speech output then never applies the gain a second time.
    ``pacing`` is its report section (numbers only)."""

    def take_duck(self, duck: DuckGain | None) -> bool: ...
    def pacing(self) -> dict[str, Any]: ...


@dataclass
class _RawJob:
    """Samples to play as they are (``SpeechOutput.play``); never synthesized, not speech."""

    id: int
    epoch: int
    samples: np.ndarray
    rate_hz: int


@dataclass
class _Job:
    id: int
    utterance: int
    text: str
    language: str
    delivery: Delivery
    epoch: int
    said_at: float
    question: bool = False  # this piece's own text ends in a question mark
    continued: bool = False  # the rest of a sentence the splitter cut


class StreamFeed:
    """Audio for one stream job, filled from any thread (``put``, then ``end``). PCM16 bytes
    (little endian, mono) or float samples. Once the job is cut by a stop or has finished, the
    feed is closed and ``put`` returns False: nothing more will play."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: collections.deque[tuple[np.ndarray, int]] = collections.deque()
        self._ended = False
        self._closed = False
        self._buffered_s = 0.0
        self._wake: Callable[[], None] | None = None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def put(self, samples: bytes | np.ndarray, rate_hz: int) -> bool:
        """Queue a piece of audio; False when the feed no longer takes any."""
        if isinstance(samples, bytes | bytearray | memoryview):
            pcm = np.frombuffer(samples, "<i2").astype(np.float32) / np.float32(PCM16_FULL_SCALE)
        else:
            pcm = np.asarray(samples, np.float32)
        with self._lock:
            if self._closed or self._ended:
                return False
            if pcm.size and rate_hz > 0:
                self._chunks.append((pcm, rate_hz))
                self._buffered_s += pcm.size / rate_hz
            wake = self._wake
        if wake is not None:
            wake()
        return True

    def end(self) -> None:
        """No more audio: the job plays what it holds and finishes."""
        with self._lock:
            self._ended = True
            wake = self._wake
        if wake is not None:
            wake()

    def close(self) -> None:
        """Refuse more audio and drop what is held (the job is over or was cut)."""
        with self._lock:
            self._closed = True
            self._chunks.clear()

    def _bind(self, wake: Callable[[], None]) -> None:
        with self._lock:
            self._wake = wake

    def _ready(self, prebuffer_s: float) -> bool:
        with self._lock:
            return self._ended or self._closed or self._buffered_s >= prebuffer_s

    def _take(self) -> tuple[list[tuple[np.ndarray, int]], bool]:
        """Everything held, and whether the feed has ended (or was closed)."""
        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()
            return chunks, self._ended or self._closed


@dataclass
class _StreamJob(_Job):
    """Audio from a :class:`StreamFeed`, played as one sentence (``SpeechOutput.stream``)."""

    feed: StreamFeed = field(default_factory=StreamFeed)
    gain: float = 1.0
    prebuffer_s: float = 0.0
    pushed_s: float = 0.0


class SpeechOutput:
    """Queue of sentences, a TTS backend and a sink (see the module docstring)."""

    def __init__(
        self,
        config: RoboEmotConfig,
        sink: AudioSink,
        backend: TTSBackend,
        bus: Bus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config, self.sink, self.backend, self.bus, self.clock = (
            config,
            sink,
            backend,
            bus,
            clock,
        )
        self._lock = threading.Lock()
        self._jobs: collections.deque[_Job | _RawJob] = collections.deque()
        self._epoch = 0
        # stops waiting for ``_lock``: the output thread refuses to push meanwhile, so it cannot
        # re-take the lock push after push (CPython locks are not fair) and starve the stop
        self._stopping = 0
        self._stopping_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._utterances = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._main: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._current_epoch = -1
        self._raw_current: int | None = None  # id of the raw job taken off the queue, if any
        self._raw_cancelled: set[int] = set()
        self._raw_pushed: int | None = None  # id of the raw job whose samples are in the sink
        self._ready = threading.Event()
        self._closed = False
        self._last_utterance = 0
        self._last_rate_hz = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
        self._first_audio: collections.OrderedDict[int, float] = collections.OrderedDict()
        # job id -> clock stamp at which its first sample starts to play; ids a stop cut off
        # before that are in ``_unplayed``
        self._audible: collections.OrderedDict[int, float] = collections.OrderedDict()
        self._unplayed: set[int] = set()
        self._ended: collections.OrderedDict[int, float] = collections.OrderedDict()
        self._question_end: float | None = None  # end of the last spoken sentence, if a question
        # utterance -> (index of its next sentence, delivery of its last one)
        self._said: collections.OrderedDict[int, tuple[int, Delivery]] = collections.OrderedDict()
        window = config.voice.report_window
        self._rtf: collections.deque[float] = collections.deque(maxlen=window)
        self._rtf_sentence: collections.deque[float] = collections.deque(maxlen=window)
        self._push_ms: collections.deque[float] = collections.deque(maxlen=window)
        self._timeline_ms: collections.deque[float] = collections.deque(maxlen=window)
        self._late_sentences = 0
        self._late_ms = 0.0
        self._boundary_dry = 0
        self._underruns = 0
        self._underrun_ms = 0.0
        self._underrun_sentences = 0
        self._underrun_job = 0  # last sentence counted in _underrun_sentences
        self.sentences_spoken = 0
        self.sentences_dropped = 0
        self.no_voice = 0  # sentences refused because there is no local voice (backend none)
        self.failures = 0
        # energy, phrase and emphasis stamps of what is pushed, for the speaking motion (SPEC 8)
        self.timeline = SpeechTimeline(config.speech_motion, clock)
        self._duck: DuckGain | None = None  # barge-in ducking, set by the runtime with barge-in
        self._sink_ducks = False  # the sink applies the duck itself (DuckingSink)
        self._duck_utterance: int | None = None  # utterance of the last speech push
        self._last_speech: int | None = None  # job id of the last speech push
        # stream jobs by id (``mark_question``, ``stream_progress``), the last PLAYED_WINDOW
        self._streams: collections.OrderedDict[int, _StreamJob] = collections.OrderedDict()

    @property
    def duck(self) -> DuckGain | None:
        """Barge-in's duck gain (None: no ducking)."""
        return self._duck

    @duck.setter
    def duck(self, duck: DuckGain | None) -> None:
        """Hand the duck to a sink that applies it at its own hand-off, else apply it here."""
        self._duck = duck
        self._sink_ducks = isinstance(self.sink, DuckingSink) and self.sink.take_duck(duck)

    def apply_config(self, config: RoboEmotConfig) -> None:
        """Hot reload: the next sentence reads ``config``; the backend gets ``config.voice`` for
        the settings it keeps itself (Piper's timeouts)."""
        self.config = config
        self.backend.apply_config(config.voice)
        self.timeline.apply_config(config.speech_motion)

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the loop thread and warm the default language's voice."""
        self._thread = threading.Thread(target=self._run, name="speech-output", daemon=True)
        self._thread.start()
        if not self._ready.wait(READY_TIMEOUT_S):
            log.warning("speech: the output thread did not start in time")

    def close(self) -> None:
        """Stop speaking, cancel the loop and wait briefly for the thread."""
        self._closed = True
        self.stop()
        loop, main = self._loop, self._main
        if loop is not None and main is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.backend.cancel)
                loop.call_soon_threadsafe(main.cancel)
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)

    def _run(self) -> None:
        loop = self._loop = asyncio.new_event_loop()
        try:
            self._main = loop.create_task(self._serve())
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(self._main)
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(self.backend.aclose())
            loop.close()

    async def _serve(self) -> None:
        self._wake = asyncio.Event()
        self._ready.set()
        cfg = self.config
        try:
            await self.backend.warm(cfg.conversation.language, cfg.voice)
        except Exception as exc:  # a failed warm-up must not end the output thread
            log.warning("speech: warm-up failed: %s", type(exc).__name__)
        while True:
            with self._lock:  # a pending stop drops the queue: take nothing from it
                job = self._jobs.popleft() if self._jobs and not self._stopping else None
                if isinstance(job, _RawJob):
                    self._raw_current = job.id
            if job is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            if job.epoch != self._epoch:
                self._release_raw(job)
                continue
            self._current_epoch = job.epoch
            self._current = asyncio.ensure_future(self._run_job(job))
            await asyncio.wait({self._current})
            self._release_raw(job)

    def _run_job(self, job: _Job | _RawJob) -> Any:
        if isinstance(job, _RawJob):
            return self._play_raw(job)
        if isinstance(job, _StreamJob):
            return self._stream(job)
        return self._speak(job)

    # ----------------------------------------------------------------- API

    def new_utterance(self) -> int:
        """Id grouping the sentences of one answer (they get a pause between them)."""
        return next(self._utterances)

    def say(
        self,
        text: str,
        *,
        language: str,
        register: RegisterParams,
        circadian: StateEffects,
        mood: str | None,
        intensity: float,
        utterance: int | None = None,
        continued: bool = False,
    ) -> int | None:
        """Queue one sentence; returns its id, or ``None`` when it was dropped (queue full or
        closed). Safe from any thread. The caller resolves ``mood`` (affect label, or the LLM's
        tag through ``resolve_tag``) and passes the register and circadian state of this moment.
        ``continued`` (the splitter's mark on the rest of a cut sentence) reuses the delivery of
        the utterance's previous piece, so one sentence is spoken with one delivery."""
        cfg = self.config
        if self.backend.name == NO_VOICE:  # voice.backend none: nothing speaks a sentence
            with self._lock:
                self.no_voice += 1
            return None
        if utterance is None:
            utterance = self.new_utterance()
        with self._lock:
            previous = self._said.get(utterance)
        if continued and previous is not None:
            index, delivery = previous
        else:
            index = previous[0] if previous is not None else 0
            delivery = compose_delivery(
                cfg.voice,
                register=register,
                circadian=circadian,
                mood=mood,
                intensity=intensity,
                persona=effects(cfg.persona),
                seed=utterance,
                index=index,
                question=ends_with_question(text),
            )
            index += 1
        with self._lock:
            self._said.pop(utterance, None)
            self._said[utterance] = (index, delivery)
            while len(self._said) > SAID_UTTERANCES_MAX:
                self._said.popitem(last=False)
        with self._lock:
            if self._closed or len(self._jobs) >= cfg.voice.queue_max_sentences:
                self.sentences_dropped += 1
                return None
            job = _Job(
                next(self._ids),
                utterance,
                text,
                language,
                delivery,
                self._epoch,
                self.clock(),
                ends_with_question(text),
                continued,
            )
            self._jobs.append(job)
        if self.bus is not None:
            self.bus.publish(
                "speech.say",
                {
                    "id": job.id,
                    "utterance": job.utterance,
                    "chars": len(text),
                    "language": language,
                    "mood": delivery.mood,
                },
                job.said_at,
            )
        self._poke()
        return job.id

    def stream(
        self,
        chunks: StreamFeed,
        *,
        utterance: int,
        mood: str | None,
        volume_db: float,
        language: str = "",
    ) -> int | None:
        """Queue audio that arrives in ``chunks`` (filled from any thread) as one sentence of
        answer ``utterance``; returns its job id, or ``None`` when the queue is full or closed
        (the feed is then closed). ``volume_db`` is applied as gain (the delivery's volume:
        base, register, circadian and mood); ``mood`` is the timeline's. Safe from any thread."""
        cfg = self.config
        delivery = Delivery(1.0, 0.0, volume_db, 0.0, 1.0, mood=mood)
        with self._lock:
            if self._closed or len(self._jobs) >= cfg.voice.queue_max_sentences:
                self.sentences_dropped += 1
                chunks.close()
                return None
            job = _StreamJob(
                next(self._ids),
                utterance,
                "",
                language,
                delivery,
                self._epoch,
                self.clock(),
                feed=chunks,
                gain=db_to_gain(volume_db),
                prebuffer_s=cfg.voice.realtime_route.prebuffer_ms / MS_PER_S,
            )
            self._jobs.append(job)
            self._streams[job.id] = job
            while len(self._streams) > PLAYED_WINDOW:
                self._streams.popitem(last=False)
        if self.bus is not None:
            self.bus.publish(
                "speech.say",
                {
                    "id": job.id,
                    "utterance": utterance,
                    "chars": 0,
                    "language": language,
                    "mood": mood,
                    "stream": True,
                },
                job.said_at,
            )
        self._poke()
        return job.id

    def mark_question(self, job_id: int) -> None:
        """Stream job ``job_id`` ends in a question (its transcript came while it played): the
        timeline and ``question_end`` learn it, also after its audio was pushed whole."""
        with self._lock:
            job = self._streams.get(job_id)
            if job is None or job.question:
                return
            job.question = True
            self.timeline.set_question(job_id)
            if self._last_speech == job_id and job_id in self._ended:
                self._question_end = self._ended[job_id]

    def stream_progress(self, job_id: int) -> tuple[int, int] | None:
        """``(played_ms, pushed_ms)`` of stream job ``job_id``: how much of its audio has played
        (up to now, or to the stop that cut it) and how much was pushed. None when it never
        began to play or is no longer remembered."""
        with self._lock:
            job = self._streams.get(job_id)
            start = self._audible.get(job_id)
            end = self._ended.get(job_id)
            if job is None or start is None or end is None or job_id in self._unplayed:
                return None
            played_s = max(0.0, min(end, self.clock()) - start)
            return round(played_s * MS_PER_S), round(job.pushed_s * MS_PER_S)

    def stop(self) -> None:
        """Barge-in: cut the audio now, drop the queue, cancel the synthesis in flight.

        The epoch moves under the lock that guards every push, so a chunk already on its way
        either was pushed before (and the sink flush below drops it) or sees the new epoch and
        is discarded: nothing plays after ``stop`` returns. Only a sentence from before this
        stop is cancelled; one said right after it is left alone. While it waits for the lock,
        the output thread pushes nothing more (``_stopping``): it waits for the push in flight
        only, never for the rest of the sentence (SPEC 13 cut-off)."""
        with self._stopping_lock:
            self._stopping += 1
        with self._lock:
            with self._stopping_lock:
                self._stopping -= 1
            self._epoch += 1
            epoch = self._epoch
            self.sentences_dropped += sum(isinstance(j, _Job) for j in self._jobs)
            for queued in self._jobs:
                if isinstance(queued, _StreamJob):
                    queued.feed.close()
            self._jobs.clear()
            now = self.clock()
            self.timeline.truncate(now)
            self._unplayed.update(i for i, at in self._audible.items() if at > now)
            for job_id, end in self._ended.items():
                self._ended[job_id] = min(end, now)
            if self._question_end is not None:
                self._question_end = min(self._question_end, now)
            if self._duck is not None:  # under the lock: no push applies the old episode after
                self._duck.clear()
        self.sink.stop()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_before, epoch)

    def play(self, samples: np.ndarray, rate_hz: int, volume_db: float = 0.0) -> int | None:
        """Queue raw mono ``samples`` behind whatever is queued; returns the job id, or ``None``
        when the queue is full or closed. The job holds the queue while it plays, so a sentence
        never interleaves with it, and ``stop`` cuts it. It is not speech: nothing is synthesized
        and ``ended_at``, ``question_end`` and the counters stay as they are. ``volume_db`` is
        the circadian offset of the moment (``StateEffects.volume_db``). A raw job counts against
        ``voice.queue_max_sentences`` like a sentence. It writes no speech-timeline event (no
        envelope, no accent, no breath), so speaking motion never reacts to it. Safe from any
        thread."""
        gain = db_to_gain(volume_db)
        louder = np.clip(np.asarray(samples, np.float32) * np.float32(gain), -1.0, 1.0)
        with self._lock:
            if self._closed or len(self._jobs) >= self.config.voice.queue_max_sentences:
                return None
            job = _RawJob(next(self._ids), self._epoch, louder, rate_hz)
            self._jobs.append(job)
        self._poke()
        return job.id

    def cancel(self, job_id: int) -> None:
        """Drop raw job ``job_id`` only: removed if still queued; if it is the one in flight,
        nothing more is pushed for it and, once it has pushed, the sink is flushed. The job
        starts only after earlier audio has played, so the flush cuts nothing else. Other jobs
        and an id that already finished are untouched. The check, the flush and the push share
        the lock, so a finished job is never followed by a flush of the next one."""
        with self._lock:
            for job in self._jobs:
                if isinstance(job, _RawJob) and job.id == job_id:
                    self._jobs.remove(job)
                    return
            if self._raw_current != job_id:
                return
            self._raw_cancelled.add(job_id)
            if self._raw_pushed == job_id:
                self.sink.stop()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_raw, job_id)

    def live(self, job_id: int) -> bool:
        """Whether raw job ``job_id`` is still queued or is the one playing. False once it has
        played to its end, was cancelled, or a ``stop`` (barge-in) dropped it, so the caller
        can tell its sound is over without a clock of its own. Safe from any thread."""
        with self._lock:
            return self._raw_current == job_id or any(
                isinstance(job, _RawJob) and job.id == job_id for job in self._jobs
            )

    def waiting(self, exclude: int | None = None) -> int:
        """Jobs still queued (sentences and raw sounds), not counting job ``exclude``. The idle
        runner passes its own sound, so only somebody else's audio counts (SPEC 7B)."""
        with self._lock:
            return sum(job.id != exclude for job in self._jobs)

    def played(self, job_id: int) -> bool:
        """Whether sentence ``job_id`` began to play: its audio reached the sink and was not
        cut before its first sample. A sentence a ``stop`` interrupted halfway counts (it was
        heard in part); one still queued or in synthesis does not. Remembered for the last
        ``PLAYED_WINDOW`` sentences, so the conversation keeps in its history only what the
        person heard."""
        with self._lock:
            return job_id in self._audible and job_id not in self._unplayed

    def ended_at(self, job_id: int) -> float | None:
        """Clock stamp at which the audio pushed so far for sentence ``job_id`` has played
        (the end of the sink's queue after its last push); None before its first push or
        after the last ``PLAYED_WINDOW`` sentences. It moves while the sentence is still
        being synthesized, so a caller needs the sentence to be over first. A stop moves it
        to the stop."""
        with self._lock:
            return self._ended.get(job_id)

    def question_end(self) -> float | None:
        """Clock stamp at which the last sentence that began to play ends (or ended, or was cut
        by a stop) when it ended in a question mark; None when that sentence was not a question
        or nothing has been said. The next sentence that plays replaces it, so a question
        followed by a statement is no longer pending. Sentences a stop dropped before they
        played never replace or revive it, and a stop only ever moves the stamp earlier."""
        with self._lock:
            return self._question_end

    def first_audio_at(self, utterance: int) -> float | None:
        """Clock stamp when the first audio of answer ``utterance`` was pushed, None until then
        (or after the last ``LATENCY_WINDOW`` answers): end-to-end latency counts from it."""
        with self._lock:
            return self._first_audio.get(utterance)

    def snapshot(self) -> dict[str, Any]:
        """Counts and timings for the debug report; never text."""
        with self._lock:
            lat, queued = list(self._latencies), len(self._jobs)
            rtf, rtf_sentence, push_ms, timeline_ms = (
                list(self._rtf),
                list(self._rtf_sentence),
                list(self._push_ms),
                list(self._timeline_ms),
            )
            gaps = {
                "underruns": self._underruns,
                "underrun_ms_total": self._underrun_ms,
                "underrun_sentences": self._underrun_sentences,
                "late_sentences": self._late_sentences,
                "late_ms_total": self._late_ms,
                "boundary_dry": self._boundary_dry,
            }
        return {
            "queued": queued,
            "spoken": self.sentences_spoken,
            "dropped": self.sentences_dropped,
            "no_voice": self.no_voice,
            "failures": self.failures,
            "first_audio_ms_last": lat[-1] if lat else None,
            "first_audio_ms_max": max(lat) if lat else None,
            "backend": self.backend.name,
            "tts_rtf": summarise(rtf, RTF_LEVELS),
            "tts_rtf_sentence": summarise(rtf_sentence, RTF_LEVELS),
            "push_ms": summarise(push_ms, RTF_LEVELS),
            "timeline_ms": summarise(timeline_ms, RTF_LEVELS),
            **gaps,
            "timeline": self.timeline.snapshot(),
            **({"pacing": self.sink.pacing()} if isinstance(self.sink, DuckingSink) else {}),
        }

    # ----------------------------------------------------------------- loop thread

    def _poke(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)

    def _cancel_before(self, epoch: int) -> None:
        """Cancel the sentence in flight if it was queued before the stop that moved the epoch to
        ``epoch``. Cancelling its task closes its backend generator, which cancels that request
        only (the Piper worker is told to skip it; an HTTP stream is closed)."""
        task = self._current
        if task is not None and not task.done() and self._current_epoch < epoch:
            task.cancel()

    def _release_raw(self, job: _Job | _RawJob) -> None:
        """The loop is done with ``job``: if it was a raw job, it is no longer the one playing
        (done on the loop thread, so a task cancelled before it ran is released too)."""
        if isinstance(job, _RawJob):
            with self._lock:
                self._raw_cancelled.discard(job.id)
                if self._raw_current == job.id:
                    self._raw_current = None
                if self._raw_pushed == job.id:
                    self._raw_pushed = None

    def _cancel_raw(self, job_id: int) -> None:
        """Cancel the task of raw job ``job_id`` if it is the one in flight (loop thread)."""
        task = self._current
        if task is not None and not task.done() and self._raw_current == job_id:
            task.cancel()

    def _push(
        self, job: _Job | _RawJob, samples: np.ndarray, rate_hz: int, *, speech: bool = False
    ) -> bool:
        """Push unless a stop happened since ``job`` was queued, or one is waiting for the lock.
        The check, the push and the notes about it (when the sentence starts to play, where its
        audio ends) are one step under the lock ``stop`` takes to move the epoch, so a sentence
        never counts as played when a stop beat its first push."""
        with self._lock:
            if self._stale(job) or job.id in self._raw_cancelled:
                return False
            now, until = self.clock(), self.sink.playing_until()
            start = max(now, until)
            if speech and job.id in self._audible and math.isfinite(until) and now > until:
                self._count_underrun(job, now - until)
            if speech and job.utterance != self._duck_utterance:
                if self._duck_utterance is not None and self._duck is not None:
                    self._duck.clear()  # a new answer starts at full gain
                self._duck_utterance = job.utterance
            duck = None if self._sink_ducks else self._duck
            played = samples if duck is None else duck.apply(samples, rate_hz, start)
            self.sink.push_audio_sample(played, rate_hz)
            if isinstance(job, _RawJob):
                self._raw_pushed = job.id
            if speech:
                self._last_speech = job.id
                self._push_ms.append((self.clock() - now) * MS_PER_S)
                began = time.perf_counter()
                self.timeline.push(
                    job.id,
                    samples,
                    rate_hz,
                    start,
                    utterance=job.utterance,
                    question=job.question,
                    mood=job.delivery.mood,
                    continued=job.continued,
                )
                self._timeline_ms.append((time.perf_counter() - began) * MS_PER_S)
            elif not isinstance(job, _RawJob):
                self.timeline.pad(job.id, start)
            if speech and job.id not in self._audible:
                self._audible[job.id] = start
                while len(self._audible) > PLAYED_WINDOW:
                    self._unplayed.discard(self._audible.popitem(last=False)[0])
            if speech:
                self._ended[job.id] = self._queue_end()
                self._question_end = self._ended[job.id] if job.question else None
                while len(self._ended) > PLAYED_WINDOW:
                    self._ended.popitem(last=False)
            return True

    async def _play_raw(self, job: _RawJob) -> None:
        """Wait until the audio already in the sink has played (cancellable, so a stop or a
        close never hangs), push the samples in one piece and hold the queue until they have
        played, so the next job starts behind them and a cancel reaches them while audible."""
        try:
            while True:
                with self._lock:
                    wait_s = self._queue_end() - self.clock()
                if wait_s <= 0:
                    break
                await asyncio.sleep(wait_s)
            if not self._push(job, job.samples, job.rate_hz):
                return
            with self._lock:
                hold_s = self._queue_end() - self.clock()
            await asyncio.sleep(hold_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a sink bug must not end the output thread
            self.failures += 1
            log.warning("speech %d (raw) crashed: %s", job.id, type(exc).__name__)

    def _stale(self, job: _Job | _RawJob) -> bool:
        """Whether ``job`` must not reach the sink: a stop since it was queued, or one waiting
        for the lock (it moves the epoch next); the lock is held."""
        return self._stopping > 0 or job.epoch != self._epoch

    def _count_underrun(self, job: _Job, gap_s: float) -> None:
        """A push of ``job`` found the sink dry for ``gap_s``; the lock is held."""
        self._underruns += 1
        self._underrun_ms += gap_s * MS_PER_S
        if self._underrun_job != job.id:
            self._underrun_job = job.id
            self._underrun_sentences += 1

    def _note_rtf(self, chunk: AudioChunk, waited_s: float) -> float:
        """Real-time factor of one chunk: the time waited for it over its audio duration.
        Returns that duration in seconds (0 for an empty or rateless chunk, not recorded)."""
        if chunk.rate_hz <= 0 or chunk.samples.size == 0:
            return 0.0
        audio_s = chunk.samples.size / chunk.rate_hz
        with self._lock:
            self._rtf.append(waited_s / audio_s)
        return audio_s

    def _note_sentence_rtf(self, waited_s: float, audio_s: float) -> None:
        """Real-time factor of a finished sentence: all its waits over all its audio."""
        if audio_s > 0:
            with self._lock:
                self._rtf_sentence.append(waited_s / audio_s)

    async def _speak(self, job: _Job) -> None:
        cfg, first = self.config, True
        try:
            asked = self.clock()
            waited_s = audio_s = 0.0
            async for chunk in self.backend.synthesize(
                job.text,
                job.delivery,
                language=job.language,
                voice=cfg.voice,
            ):
                waited = self.clock() - asked
                waited_s += waited
                audio_s += self._note_rtf(chunk, waited)
                if first:
                    first = False
                    if not self._pad(job, chunk.rate_hz):
                        return
                    self._note_first_audio(job)
                    log.debug("speech %d: first audio after %.0f ms", job.id, self._latencies[-1])
                self._last_rate_hz = chunk.rate_hz
                if not self._push(job, chunk.samples, chunk.rate_hz, speech=True):
                    return
                asked = self.clock()
            self._note_sentence_rtf(waited_s, audio_s)
            self.sentences_spoken += 1
        except asyncio.CancelledError:
            raise
        except TTSError as exc:
            self.failures += 1
            log.warning("speech %d failed: %s", job.id, type(exc).__name__)
        except Exception as exc:  # a backend bug must not end the output thread
            self.failures += 1
            log.warning("speech %d crashed: %s", job.id, type(exc).__name__)
        finally:
            self._finish_stream(job)
            self._last_utterance = job.utterance

    async def _stream(self, job: _StreamJob) -> None:
        """Play a stream job: wait for ``prebuffer_s`` of audio (or the feed's end), then push
        each piece as it arrives, with the gain, until the feed ends. One sentence."""
        feed, first = job.feed, True
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        feed._bind(lambda: loop.call_soon_threadsafe(wake.set))
        try:
            while not feed._ready(job.prebuffer_s):
                wake.clear()
                if feed._ready(job.prebuffer_s):
                    break
                await wake.wait()
            while True:
                wake.clear()
                chunks, ended = feed._take()
                for samples, rate_hz in chunks:
                    if first:
                        first = False
                        if not self._pad(job, rate_hz):
                            return
                        self._note_first_audio(job)
                    self._last_rate_hz = rate_hz
                    played = np.clip(samples * np.float32(job.gain), -1.0, 1.0)
                    if not self._push(job, played, rate_hz, speech=True):
                        return
                    job.pushed_s += samples.size / rate_hz
                if ended:
                    break
                if not chunks:
                    await wake.wait()
            if not first:
                self.sentences_spoken += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a sink bug must not end the output thread
            self.failures += 1
            log.warning("speech %d (stream) crashed: %s", job.id, type(exc).__name__)
        finally:
            feed.close()
            self._finish_stream(job)
            self._last_utterance = job.utterance

    def _note_first_audio(self, job: _Job) -> None:
        """The first audio of ``job``: its latency and, once per answer, the answer's first
        audio stamp."""
        with self._lock:
            now = self.clock()
            self._latencies.append((now - job.said_at) * MS_PER_S)
            if job.utterance not in self._first_audio:
                self._first_audio[job.utterance] = now
                while len(self._first_audio) > LATENCY_WINDOW:
                    self._first_audio.popitem(last=False)

    def _finish_stream(self, job: _Job) -> None:
        """End the sink's stream after ``job`` (its tail extends where the sentence ends),
        unless a stop since it was queued already reset the sink, or one is waiting for the lock."""
        with self._lock:
            if self._stale(job):
                return
            try:
                self.sink.end_stream()
            except Exception as exc:
                log.warning("speech: end_stream: %s", type(exc).__name__)
                return
            if job.id in self._ended:
                self._ended[job.id] = self._queue_end()
                self.timeline.finish(job.id, self._ended[job.id])
                self._question_end = self._ended[job.id] if job.question else None

    def _queue_end(self) -> float:
        """Where the sink's queue ends: when a sample pushed now would play (the clock when it
        has no timeline)."""
        now = self.clock()
        return max(now, self.sink.playing_until())

    def _note_boundary(self, idle_s: float, pause_s: float) -> None:
        """Count a dry sink between two sentences of one answer, and a late sentence when the
        silence already there is longer than the planned pause."""
        with self._lock:
            self._boundary_dry += idle_s > 0
            if idle_s > pause_s:
                self._late_sentences += 1
                self._late_ms += (idle_s - pause_s) * MS_PER_S

    def _pad(self, job: _Job, rate_hz: int) -> bool:
        """Silence between two sentences of one answer, less the idle gap already there.
        False if a stop made the sentence stale."""
        if job.utterance != self._last_utterance:
            return job.epoch == self._epoch
        pause_s = self.config.voice.sentence_pause_ms / MS_PER_S * job.delivery.pause_scale
        until = self.sink.playing_until()
        idle_s = max(0.0, self.clock() - until) if math.isfinite(until) else 0.0
        self._note_boundary(idle_s, pause_s)
        pause_s -= idle_s
        if pause_s > 0:
            return self._push(job, np.zeros(round(pause_s * rate_hz), np.float32), rate_hz)
        return job.epoch == self._epoch
