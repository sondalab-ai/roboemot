"""The voice listener: microphone chunks to utterances (SPEC 8 stages 2-4).

One thread reads ``source(timeout)`` (the audio hub's voice queue), and for each chunk:

1. drops it while the robot speaks (``SpeechActivity``; barge-in is slice 6) and resets the
   pipeline on that transition and on a gap, since the audio is no longer contiguous;
2. runs the VAD endpointer; the ASR only ever sees speech (pre-roll included);
3. on speech start picks the language, runs the fast path (head turn, ``speech.heard``) and
   starts the recognizer; on each partial runs the keyword spotter;
4. on speech end finishes the ASR and builds an ``Utterance``: handed to ``on_utterance`` (the
   conversation, in-process) and announced on the bus as ``percept.utterance`` without its text.

Chunk stamps are the monotonic time the chunk *ends*; the endpointer wants its start. The work
per utterance is bounded by ``voice.vad.max_utterance_s``. Audio and text are never logged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..bus import Bus
from ..config.schema import RoboEmotConfig
from .activity import SpeechActivity
from .asr import ASRBackend, ASREvent, resolve_language
from .fastpath import FastPath
from .speech_features import Prosody
from .vad import Endpointer, SpeechAudio, SpeechEnd, SpeechStart
from .wake import WakeGate

log = logging.getLogger(__name__)
TOPIC = "percept.utterance"
POLL_S = 0.1
JOIN_S = 2.0


class VoiceChunkLike(Protocol):
    samples: np.ndarray
    rate_hz: int
    stamp: float
    gap: bool


@dataclass(frozen=True)
class Utterance:
    """A finished utterance. ``text`` is the user's speech: in-process only, hidden from repr.

    Stamps are monotonic seconds; latencies count from ``last_speech_stamp``. ``addressed`` is
    the wake-word decision: the conversation answers only addressed utterances.
    """

    text: str = field(repr=False)
    language: str
    start_stamp: float
    last_speech_stamp: float
    endpoint_stamp: float
    addressed: bool
    wake_word: bool
    keywords: frozenset[str]
    prosody: Prosody
    final: bool = True


@dataclass(frozen=True)
class UtteranceInfo:
    """The ``percept.utterance`` payload: everything about an utterance except what was said."""

    language: str
    chars: int
    words: int
    start_stamp: float
    last_speech_stamp: float
    endpoint_stamp: float
    addressed: bool
    wake_word: bool
    rms_db: float
    pitch_hz: float | None
    words_per_s: float | None


class VoiceListener:
    """See the module docstring. ``process`` is the synchronous step; ``start`` runs it."""

    def __init__(
        self,
        cfg: RoboEmotConfig,
        source: Callable[[float], VoiceChunkLike | None],
        endpointer: Endpointer,
        asr: ASRBackend,
        speech: SpeechActivity,
        fastpath: FastPath,
        on_utterance: Callable[[Utterance], None],
        bus: Bus | None = None,
        *,
        speaker_language: Callable[[], str | None] = lambda: None,
        enrolled_in_view: Callable[[float], bool] = lambda now: False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._endpointer = endpointer
        self._asr = asr
        self._speech = speech
        self._fast = fastpath
        self._on_utterance = on_utterance
        self._bus = bus
        self._speaker_language = speaker_language
        self._enrolled_in_view = enrolled_in_view
        self._clock = clock
        self._gate = WakeGate("always", 0.0)
        self.apply_config(cfg)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._counts = {"utterances": 0, "empty": 0, "gaps": 0, "dropped_speaking": 0, "errors": 0}
        self._active = False
        self._muted = False
        self._language = ""
        self._last_error = ""

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: keyword lists, wake policy and window, default language."""
        self._cfg = cfg
        self._fast.apply_config(cfg.voice)
        self._gate.apply(cfg.voice.wake_word_policy, cfg.voice.fastpath.addressed_window_s)

    # ---- lifecycle

    def start(self) -> None:
        """Load the ASR models, then read the audio in a daemon thread (no-op when running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._asr.load(self._cfg.voice.asr.models)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="roboemot-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread (waits briefly for the chunk in hand)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(JOIN_S)
            self._thread = None

    def snapshot(self) -> dict[str, Any]:
        """Counters and fast-path latencies for the debug report: numbers only, no content."""
        with self._lock:
            snap: dict[str, Any] = dict(self._counts)
        snap["latency"] = self._fast.stats.snapshot()
        return snap

    def _run(self) -> None:
        while not self._stop.is_set():
            chunk = self._source(POLL_S)
            if chunk is None:
                continue
            try:
                self.process(chunk)
            except Exception as exc:  # one bad chunk must not end the listener
                self._abort()
                self._count("errors")
                name = type(exc).__name__
                if name != self._last_error:
                    self._last_error = name
                    log.warning("voice listener: %s", name)

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    # ---- one chunk

    def _abort(self) -> None:
        """Drop the utterance in progress and every buffered sample."""
        self._endpointer.reset()
        self._asr.reset()
        self._fast.reset()
        self._active = False

    def process(self, chunk: VoiceChunkLike) -> None:
        """Handle one chunk (see the module docstring)."""
        start = chunk.stamp - chunk.samples.size / chunk.rate_hz
        if chunk.gap:
            self._count("gaps")
            self._abort()
        if self._speech.speaking(start) or self._speech.speaking(chunk.stamp):
            if not self._muted:
                self._muted = True
                self._abort()
            self._count("dropped_speaking")
            return
        self._muted = False
        buffered: list[np.ndarray] = []
        for ev in self._endpointer.feed(chunk.samples, start):
            if isinstance(ev, SpeechAudio):
                buffered.append(ev.samples)
                continue
            self._feed(buffered, chunk.stamp)
            buffered = []
            if isinstance(ev, SpeechStart):
                self._begin(ev, chunk.stamp)
            elif isinstance(ev, SpeechEnd):
                self._end(ev)
        self._feed(buffered, chunk.stamp)

    def _feed(self, parts: list[np.ndarray], audio_end: float) -> None:
        if parts and self._active:
            self._partial(self._asr.accept(np.concatenate(parts)), audio_end)

    def _partial(self, event: ASREvent | None, audio_end: float) -> None:
        if event is not None and event.text:
            self._fast.partial(event.text, audio_end)

    # ---- utterance edges

    def _begin(self, ev: SpeechStart, audio_end: float) -> None:
        models = self._cfg.voice.asr.models
        try:
            speaker = self._speaker_language()
        except Exception:
            speaker = None
        self._language = resolve_language(models, speaker, self._cfg.conversation.language)
        self._fast.begin(audio_end, self._language)
        self._asr.start(self._language)
        self._active = True
        self._partial(self._asr.accept(ev.pre_roll), audio_end)

    def _end(self, ev: SpeechEnd) -> None:
        if not self._active:
            return
        self._active = False
        text = self._asr.finish().strip()
        if not text:
            self._count("empty")
            self._fast.reset()
            return
        rate = self._cfg.voice.sample_rate_hz
        first = max(0, round((ev.start_stamp - ev.audio_stamp) * rate))
        last = round((ev.last_speech_stamp - ev.audio_stamp + self._endpointer.frame_s) * rate)
        fast = self._fast.end(
            text,
            ev.audio[first:last],
            rate,
            last_speech_stamp=ev.last_speech_stamp,
            endpoint_stamp=ev.endpoint_stamp,
        )
        addressed = self._gate.decide(
            wake_word=fast.wake_word,
            enrolled_in_view=self._enrolled(ev.start_stamp),
            start=ev.start_stamp,
            last_speech=ev.last_speech_stamp,
        )
        utterance = Utterance(
            text,
            self._language,
            ev.start_stamp,
            ev.last_speech_stamp,
            ev.endpoint_stamp,
            addressed,
            fast.wake_word,
            fast.keywords,
            fast.prosody,
        )
        self._fast.stats.record("final_vs_last_speech", self._clock() - ev.last_speech_stamp)
        self._count("utterances")
        self._publish(utterance, len(text), len(text.split()))
        try:
            self._on_utterance(utterance)
        except Exception as exc:
            log.warning("voice listener: utterance consumer raised %s", type(exc).__name__)

    def _enrolled(self, stamp: float) -> bool:
        try:
            return self._enrolled_in_view(stamp)
        except Exception:
            return False

    def _publish(self, u: Utterance, chars: int, words: int) -> None:
        if self._bus is None:
            return
        info = UtteranceInfo(
            u.language,
            chars,
            words,
            u.start_stamp,
            u.last_speech_stamp,
            u.endpoint_stamp,
            u.addressed,
            u.wake_word,
            u.prosody.rms_db,
            u.prosody.pitch_hz,
            u.prosody.words_per_s,
        )
        try:
            self._bus.publish(TOPIC, info, u.endpoint_stamp)
        except Exception as exc:
            log.warning(
                "voice listener: percept.utterance subscriber raised %s", type(exc).__name__
            )
