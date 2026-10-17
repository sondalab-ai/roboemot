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
    minor_present: bool
    epoch: int
    said_at: float


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
        self._jobs: collections.deque[_Job] = collections.deque()
        self._epoch = 0
        self._ids = itertools.count(1)
        self._utterances = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._main: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._ready = threading.Event()
        self._closed = False
        self._last_utterance = 0
        self._last_rate_hz = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=LATENCY_WINDOW)
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
        await self.backend.warm(cfg.conversation.language, cfg.voice)
        while True:
            with self._lock:
                job = self._jobs.popleft() if self._jobs else None
            if job is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            if job.epoch != self._epoch:
                continue
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
        minor_present: bool = False,
    ) -> int | None:
        """Queue one sentence; returns its id, or ``None`` when it was dropped (queue full or
        closed). Safe from any thread. The caller resolves ``mood`` (affect label, or the LLM's
        tag through ``resolve_tag``) and passes the register and circadian state of this moment."""
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
                minor_present,
                self._epoch,
                self.clock(),
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
        """Barge-in: cut the audio now, drop the queue, cancel the synthesis in flight."""
        with self._lock:
            self._epoch += 1
            self.sentences_dropped += len(self._jobs)
            self._jobs.clear()
        self.sink.stop()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_current)

    def snapshot(self) -> dict[str, Any]:
        """Counts and timings for the debug report; never text."""
        lat = list(self._latencies)
        return {
            "queued": len(self._jobs),
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

    def _cancel_current(self) -> None:
        self.backend.cancel()
        if self._current is not None and not self._current.done():
            self._current.cancel()

    async def _speak(self, job: _Job) -> None:
        cfg, first = self.config, True
        try:
            async for chunk in self.backend.synthesize(
                job.text,
                job.delivery,
                language=job.language,
                voice=cfg.voice,
                minor_present=job.minor_present,
            ):
                if job.epoch != self._epoch:
                    return
                if first:
                    first = False
                    self._pad(job, chunk.rate_hz)
                    self._latencies.append((self.clock() - job.said_at) * MS_PER_S)
                    log.debug("speech %d: first audio after %.0f ms", job.id, self._latencies[-1])
                self._last_rate_hz = chunk.rate_hz
                self.sink.push_audio_sample(chunk.samples, chunk.rate_hz)
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

    def _pad(self, job: _Job, rate_hz: int) -> None:
        """Silence between two sentences of one answer, less the idle gap already there."""
        if job.utterance != self._last_utterance:
            return
        pause_s = self.config.voice.sentence_pause_ms / MS_PER_S * job.delivery.pause_scale
        playing_until = getattr(self.sink, "playing_until", None)
        if callable(playing_until):
            pause_s -= max(0.0, self.clock() - playing_until())
        if pause_s > 0:
            self.sink.push_audio_sample(np.zeros(round(pause_s * rate_hz), np.float32), rate_hz)
