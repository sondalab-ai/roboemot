"""Speech output: sentences in, audio on the speaker out (SPEC 8, stage 7; 13 latency budgets).

:class:`SpeechOutput` owns a thread with an event loop (the pattern of the persona traits
worker), so the caller, the conversation worker or the app tick, never blocks on synthesis.
``say`` composes the sentence's :class:`~roboemot.voice.prosody.Delivery` from the voice
configuration, the persona effects, the register, the circadian state and the mood the caller
passes, queues it, and the loop streams it through the backend into the sink, which resamples
(``ReachySpeaker``). ``stop`` is the barge-in path: it cuts the sink first, then cancels the
synthesis in flight and drops the queue; an epoch counter makes any chunk that was already on its
way fall on the floor, so nothing plays after ``stop`` returns.

The bus gets ``speech.say`` with ids, lengths, language and mood, never the text (SPEC 12).
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..bus import Bus
from ..config.schema import RegisterParams, RoboEmotConfig, StateEffects
from ..persona import effects
from .prosody import Delivery, compose_delivery
from .tts import TTSBackend, TTSError

log = logging.getLogger(__name__)

MS_PER_S = 1000.0
LATENCY_WINDOW = 20
PLAYED_WINDOW = 64  # sentences whose playback start is remembered, for ``played``
READY_TIMEOUT_S = 5.0
JOIN_TIMEOUT_S = 2.0


class AudioSink(Protocol):
    """What plays the audio: ``ReachySpeaker`` (it resamples to the output rate itself)."""

    def push_audio_sample(self, samples: np.ndarray, rate_hz: int) -> float: ...
    def stop(self) -> None: ...


@dataclass
class _Job:
    id: int
    utterance: int
    text: str
    language: str
    delivery: Delivery
    epoch: int
    said_at: float
    minor: bool = False  # a minor was present when the sentence was written (SPEC 12)


class SpeechOutput:
    """Queue of sentences, a TTS backend and a sink (see the module docstring)."""

    def __init__(
        self,
        config: RoboEmotConfig,
        sink: AudioSink,
        backend: TTSBackend,
        bus: Bus | None = None,
        minor_present: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config, self.sink, self.backend, self.bus, self.clock = (
            config,
            sink,
            backend,
            bus,
            clock,
        )
        self._minor_present = minor_present  # read when synthesis starts, never at enqueue
        self._lock = threading.Lock()
        self._jobs: collections.deque[_Job] = collections.deque()
        self._epoch = 0
        self._ids = itertools.count(1)
        self._utterances = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._main: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._current_epoch = -1
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
        self.sentences_spoken = 0
        self.sentences_dropped = 0
        self.failures = 0

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
            self._thread.join(JOIN_TIMEOUT_S)

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
            with self._lock:
                job = self._jobs.popleft() if self._jobs else None
            if job is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            if job.epoch != self._epoch:
                continue
            self._current_epoch = job.epoch
            self._current = asyncio.ensure_future(self._speak(job))
            await asyncio.wait({self._current})

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
        minor: bool = False,
    ) -> int | None:
        """Queue one sentence; returns its id, or ``None`` when it was dropped (queue full or
        closed). Safe from any thread. The caller resolves ``mood`` (affect label, or the LLM's
        tag through ``resolve_tag``) and passes the register and circadian state of this moment.
        ``minor`` says a minor was present when the sentence was written: a sentence written
        without minor rules is dropped, and the speech stopped, if one is present by the time
        it would play (SPEC 12)."""
        cfg = self.config
        delivery = compose_delivery(
            cfg.voice,
            register=register,
            circadian=circadian,
            mood=mood,
            intensity=intensity,
            persona=effects(cfg.persona),
        )
        with self._lock:
            if self._closed or len(self._jobs) >= cfg.voice.queue_max_sentences:
                self.sentences_dropped += 1
                return None
            job = _Job(
                next(self._ids),
                utterance if utterance is not None else self.new_utterance(),
                text,
                language,
                delivery,
                self._epoch,
                self.clock(),
                minor,
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

    def stop(self) -> None:
        """Barge-in: cut the audio now, drop the queue, cancel the synthesis in flight.

        The epoch moves under the lock that guards every push, so a chunk already on its way
        either was pushed before (and the sink flush below drops it) or sees the new epoch and
        is discarded: nothing plays after ``stop`` returns. Only a sentence from before this
        stop is cancelled; one said right after it is left alone."""
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self.sentences_dropped += len(self._jobs)
            self._jobs.clear()
            now = self.clock()
            self._unplayed.update(i for i, at in self._audible.items() if at > now)
        self.sink.stop()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_before, epoch)

    def played(self, job_id: int) -> bool:
        """Whether sentence ``job_id`` began to play: its audio reached the sink and was not
        cut before its first sample. A sentence a ``stop`` interrupted halfway counts (it was
        heard in part); one still queued or in synthesis does not. Remembered for the last
        ``PLAYED_WINDOW`` sentences, so the conversation keeps in its history only what the
        person heard."""
        with self._lock:
            return job_id in self._audible and job_id not in self._unplayed

    def first_audio_at(self, utterance: int) -> float | None:
        """Clock stamp when the first audio of answer ``utterance`` was pushed, None until then
        (or after the last ``LATENCY_WINDOW`` answers): end-to-end latency counts from it."""
        with self._lock:
            return self._first_audio.get(utterance)

    def snapshot(self) -> dict[str, Any]:
        """Counts and timings for the debug report; never text."""
        with self._lock:
            lat, queued = list(self._latencies), len(self._jobs)
        return {
            "queued": queued,
            "spoken": self.sentences_spoken,
            "dropped": self.sentences_dropped,
            "failures": self.failures,
            "first_audio_ms_last": lat[-1] if lat else None,
            "first_audio_ms_max": max(lat) if lat else None,
            "backend": self.backend.name,
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

    def _minor(self) -> bool:
        """Whether a minor is present now; fail closed (True) if the signal cannot be read."""
        try:
            return bool(self._minor_present())
        except Exception as exc:
            log.warning("speech: minor signal failed, assuming a minor: %s", type(exc).__name__)
            return True

    def _stale(self, job: _Job) -> bool:
        """Whether ``job`` was written for adults and a minor is present now. The speech is
        stopped (queue dropped, audio cut) and the caller abandons the sentence."""
        if job.minor or not self._minor():
            return False
        self.stop()
        return True

    def _push(self, job: _Job, samples: np.ndarray, rate_hz: int) -> bool:
        """Push unless a stop happened since ``job`` was queued. The check and the push are one
        step under the lock ``stop`` takes to move the epoch."""
        with self._lock:
            if job.epoch != self._epoch:
                return False
            self.sink.push_audio_sample(samples, rate_hz)
            return True

    async def _speak(self, job: _Job) -> None:
        cfg, first, audible = self.config, True, None
        try:
            if self._stale(job):
                return
            async for chunk in self.backend.synthesize(
                job.text,
                job.delivery,
                language=job.language,
                voice=cfg.voice,
                minor_present=self._minor(),
            ):
                if self._stale(job):
                    return
                if first:
                    first = False
                    if not self._pad(job, chunk.rate_hz):
                        return
                    audible = self._audible_start()
                    with self._lock:
                        now = self.clock()
                        self._latencies.append((now - job.said_at) * MS_PER_S)
                        if job.utterance not in self._first_audio:
                            self._first_audio[job.utterance] = now
                            while len(self._first_audio) > LATENCY_WINDOW:
                                self._first_audio.popitem(last=False)
                    log.debug("speech %d: first audio after %.0f ms", job.id, self._latencies[-1])
                self._last_rate_hz = chunk.rate_hz
                if not self._push(job, chunk.samples, chunk.rate_hz):
                    return
                if audible is not None:  # only a sentence that really reached the sink
                    self._mark_audible(job, audible)
                    audible = None
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
            self._last_utterance = job.utterance

    def _audible_start(self) -> float:
        """When a sample pushed now will play: when the audio queued before it ends (the
        sink's timeline, silence pad included), else now."""
        now = self.clock()
        playing_until = getattr(self.sink, "playing_until", None)
        return max(now, playing_until()) if callable(playing_until) else now

    def _mark_audible(self, job: _Job, start: float) -> None:
        """Note that ``job`` reached the sink and its first sample plays at ``start``."""
        with self._lock:
            self._audible[job.id] = start
            while len(self._audible) > PLAYED_WINDOW:
                self._unplayed.discard(self._audible.popitem(last=False)[0])

    def _pad(self, job: _Job, rate_hz: int) -> bool:
        """Silence between two sentences of one answer, less the idle gap already there.
        False if a stop made the sentence stale."""
        if job.utterance != self._last_utterance:
            return job.epoch == self._epoch
        pause_s = self.config.voice.sentence_pause_ms / MS_PER_S * job.delivery.pause_scale
        playing_until = getattr(self.sink, "playing_until", None)
        if callable(playing_until):
            pause_s -= max(0.0, self.clock() - playing_until())
        if pause_s > 0:
            return self._push(job, np.zeros(round(pause_s * rate_hz), np.float32), rate_hz)
        return job.epoch == self._epoch
