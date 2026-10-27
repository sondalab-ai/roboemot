"""The voice listener: microphone chunks to utterances (SPEC 8 stages 2-4).

One thread reads ``source(timeout)`` (the audio hub's voice queue), and for each chunk:

1. drops it while the robot speaks (``SpeechActivity``) after handing it to
   ``on_speaking_chunk`` (the barge-in detector, SPEC 8 stage 8), and resets the pipeline on that
   transition and on a gap, since the audio is no longer contiguous. ``resume`` is the detector's
   way back: it feeds the audio from the barge-in onset and keeps the gate open until the echo
   tail is over, so the user's first words are not lost;
2. runs the VAD endpointer; the ASR only ever sees speech (pre-roll included);
3. on speech start picks the language, runs the fast path (head turn, ``speech.heard``) and
   starts the recognizer; on each partial runs the keyword spotter;
4. on speech end finishes the ASR and submits the utterance's grammar pass to the grammar
   worker (its own thread, ``roboemot.voice.grammar_worker``), then keeps reading audio;
5. hands the utterance off once the pass is done, its wait (``voice.grammar.handoff_timeout_s``)
   is over (free-text keywords only, counted ``timeouts``). The next utterance starting, or a
   gap, waits for the pass within the same deadline first (on a lagging listener the next
   utterance is already queued); the robot speaking, a barge-in or a pause hand off at once with
   the result so far (counted ``superseded``). A finished utterance is never dropped. The
   hand-off runs the wake gate (with the voice-at-face check, whose ``direction`` samples are
   taken on speech start and on each piece of utterance audio) and builds an ``Utterance``:
   handed to ``on_utterance`` (the conversation, in-process) and announced on the bus as
   ``percept.utterance`` without its text. ``on_activity`` sees the utterance in progress until
   then.

With the realtime route's audio input (``bind_realtime``, SPEC 8, Realtime speech route), an
utterance is decided ``addressable`` when its speech starts, with the wake gate's own predicates
(``WakeGate.addressable``: the wake word not required, or a follow-up in the open window). An
addressable one skips Vosk and is only buffered; at its end the voice-at-face check runs, and
only if the gate then addresses it does its audio go (``RealtimeInput``, off this thread). One
the check rejects is decoded by Vosk from the buffer and goes only if the wake word is in it.
Speech that is not addressable runs Vosk as before; if the wake decision addresses it, its
buffered audio goes. The final of an utterance whose audio went is the server's transcript: it
waits in the same hand-off queue with the deadline ``transcript_timeout_s``, so this thread
never waits on the network. Past the deadline (or when the audio was refused) the utterance is
decoded locally from its buffer and handed off for the chat routes. Hand-offs keep the order
of the utterances; each carries the fast path's state of its utterance, so a newer utterance
can begin while an older one waits.

Chunk stamps are the monotonic time the chunk *ends*; the endpointer wants its start. The work
per utterance is bounded by ``voice.vad.max_utterance_s``. The ``realtime`` block of the report
(``ListenerLoad``) says whether the listener keeps up: its time per second of audio by stage, its
lag behind the microphone and the audio lost before it read it (SPEC 13). Audio and text are
never logged.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .. import shutdown
from ..bus import Bus
from ..config.schema import RoboEmotConfig
from .activity import SpeechActivity
from .asr import ASRBackend, ASREvent, resolve_language
from .direction import VoiceDirection
from .fastpath import FastPath, GrammarPass, UtteranceState
from .grammar_worker import GrammarJob, GrammarWorker
from .listener_load import ListenerLoad
from .speech_features import Prosody
from .vad import Endpointer, SpeechAudio, SpeechEnd, SpeechStart
from .wake import WakeGate

log = logging.getLogger(__name__)
TOPIC = "percept.utterance"
POLL_S = 0.1
JOIN_S = 2.0
MS_PER_S = 1000.0


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
    # The realtime route's audio input (SPEC 8, Realtime speech route): the server session
    # (generation) and its user items that hold this utterance's words; None when its audio
    # did not go. ``local_only``: its audio went but no transcript came, the chat routes answer.
    realtime: tuple[int, tuple[str, ...]] | None = field(default=None, repr=False)
    local_only: bool = False


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


@dataclass
class _Handoff:
    """A finished utterance waiting until ``deadline`` for its grammar pass (``job``) or for
    the realtime route's transcript of its audio (``rt``). ``state`` is the fast path's state
    of the utterance (held at its end); ``addressed`` the wake decision once made (None
    before); ``face`` the voice-at-face check's answer once asked (None: not asked);
    ``heard`` what its grammar pass heard."""

    ev: SpeechEnd
    text: str = field(repr=False)
    language: str
    grammar_pass: GrammarPass | None
    job: GrammarJob | None
    ended: float
    deadline: float
    state: UtteranceState | None = None
    rt: Any = None  # the RealtimeInput job (realtime.input.InputJob)
    addressed: bool | None = None
    face: bool | None = None
    heard: frozenset[str] = frozenset()


class VoiceListener:
    """See the module docstring. ``process`` is the synchronous step; ``start`` runs it.
    ``grammar_background`` runs the grammar pass on its own thread (the app); without it the
    pass runs inline at the end of the utterance (tests, deterministic)."""

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
        speaker_id: Callable[[], str | None] = lambda: None,
        on_speaking_chunk: Callable[[VoiceChunkLike], None] | None = None,
        on_speaking_end: Callable[[], None] | None = None,
        before_asr_reset: Callable[[], None] | None = None,
        on_activity: Callable[[bool], None] | None = None,
        enrolled_in_view: Callable[[float], bool] = lambda now: False,
        direction: VoiceDirection | None = None,
        clock: Callable[[], float] = time.monotonic,
        timer: Callable[[], float] = time.perf_counter,
        grammar_background: bool = False,
    ) -> None:
        self._source = source
        self._endpointer = endpointer
        self._asr = asr
        self._speech = speech
        self._fast = fastpath
        self._on_utterance = on_utterance
        self._bus = bus
        self._speaker_language = speaker_language
        self._speaker_id = speaker_id
        self._on_speaking_chunk = on_speaking_chunk
        self._on_speaking_end = on_speaking_end
        self._before_asr_reset = before_asr_reset
        self._on_activity = on_activity
        self._enrolled_in_view = enrolled_in_view
        self._clock = clock
        self._timer = timer
        self._direction = direction
        self._gate = WakeGate("always", 0.0)  # configured by apply_config
        self.load = ListenerLoad(cfg.voice.listener)
        self._last_end: float | None = None  # stamp of the end of the last chunk read
        self._grammar = GrammarWorker(
            background=grammar_background,
            on_spent=lambda seconds: self.load.stage("grammar", seconds),
            timer=timer,
            clock=clock,
        )
        # finished utterances waiting for their final, in order (the head goes first)
        self._queue: collections.deque[_Handoff] = collections.deque()
        self._handoff = {"waits": 0, "timeouts": 0, "superseded": 0, "dropped_on_stop": 0}
        self._realtime: Any = None  # the realtime route's audio input (bind_realtime)
        self._audio = {
            "vosk_skipped": 0,
            "sent": 0,
            "transcribed": 0,
            "transcript_timeouts": 0,
            "local_fallbacks": 0,
        }
        self._vosk = True  # the utterance in progress runs the recognizer
        self._wait_last_s = 0.0
        self._wait_max_s = 0.0
        self.apply_config(cfg)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._counts = {
            "utterances": 0,
            "empty": 0,
            "gaps": 0,
            "dropped_speaking": 0,
            "barge_ins": 0,
            "errors": 0,
        }
        # Written by the listener thread, read by snapshot(): single-word stores, atomic in
        # CPython, so no lock; the counters have their own.
        self._active = False  # an utterance is in progress or waiting for its hand-off
        self._recording = False  # the recognizer is taking the utterance's audio
        self._muted = False
        self._open_until = -math.inf  # barge-in handoff: chunks starting before this pass the gate
        self._state = "off"
        self._language = ""
        self._last_error = ""

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: keyword lists, wake policy and window, default language, VAD settings."""
        self._cfg = cfg
        self._endpointer.apply_config(cfg.voice.vad)
        self._fast.apply_config(cfg.voice)
        fp = cfg.voice.fastpath
        self._gate.apply(cfg.voice.wake_word_policy, fp.addressed_window_s, fp.window_max_s)
        self.load.apply_config(cfg.voice.listener)
        if self._direction is not None:
            self._direction.apply_config(cfg.voice.direction)

    def bind_realtime(self, realtime: Any) -> None:
        """Send addressed utterances' audio to the realtime route through ``realtime`` (a
        ``RealtimeInput``; None stops it). Bound after the route exists (start-up)."""
        self._realtime = realtime

    @property
    def config(self) -> RoboEmotConfig:
        """The config of the last ``apply_config``: the live values for the voice wiring."""
        return self._cfg

    # ---- lifecycle

    def start(self) -> None:
        """Read the audio in a daemon thread (no-op when running). The thread loads the ASR
        models first, so a slow disk never holds up the caller; a model that fails to load is
        logged (type only) and leaves the listener ``unavailable``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="roboemot-listener", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Stop the thread (waits briefly for the chunk in hand)."""
        self._stop.set()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_S)
            if not self._thread.is_alive():
                self._drop_pending()
            self._thread = None
        else:
            self._drop_pending()
        self._grammar.close()

    def pause_listening(self) -> None:
        """Run no VAD or recognizer until :meth:`resume_listening` (the robot is asleep, SPEC
        12A). The thread drops the utterance in hand, then keeps emptying the audio queue (so
        stale audio is not recognised on wake) and idles."""
        self._paused.set()

    def resume_listening(self) -> None:
        """Listen again after :meth:`pause_listening`, on the audio that arrives from now."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def snapshot(self) -> dict[str, Any]:
        """Counters and fast-path latencies for the debug report: numbers only, no content."""
        with self._lock:
            snap: dict[str, Any] = {
                "state": self._state,
                "paused": self._paused.is_set(),
                **self._counts,
            }
        snap["latency"] = self._fast.stats.snapshot()
        snap["wake"] = self._gate.snapshot(self._clock())
        if self._direction is not None:
            snap["direction"] = self._direction.snapshot()
        snap["wake_phrases_out_of_vocab"] = self._fast.wake_out_of_vocab()
        grammar = self._fast.grammar_snapshot()
        if grammar is not None:
            snap["grammar"] = grammar
        snap["vad"] = self._endpointer.snapshot()
        snap["realtime"] = {**self.load.snapshot(), "recognizers": self._asr.stats()}
        with self._lock:
            snap["realtime_audio"] = dict(self._audio)
            snap["grammar_handoff"] = {
                **self._handoff,
                "pending": bool(self._queue),
                "replaced": self._grammar.replaced,
                "skipped_stale": self._grammar.skipped_stale,
                "wait_ms": {
                    "last": self._wait_last_s * MS_PER_S,
                    "max": self._wait_max_s * MS_PER_S,
                },
            }
        return snap

    def _run(self) -> None:
        models = list(self._cfg.voice.asr.models)
        primary = resolve_language(models, None, self._cfg.conversation.language)
        try:
            self._asr.load([primary])
        except Exception as exc:
            self._state = "unavailable"
            log.warning("voice listener: models: %s", type(exc).__name__)
            self._discard()
            return
        self._name_vocabulary()
        self._state = "running"
        for language in models:  # the others while already listening; utterances before that
            if language != primary and not self._stop.is_set():  # use the languages loaded so far
                try:
                    self._asr.load([language])
                    self._name_vocabulary()
                except Exception as exc:
                    log.warning("voice listener: model %s: %s", language, type(exc).__name__)
        was_paused = False
        while not self._stop.is_set():
            if self._paused.is_set():
                if not was_paused:  # on this thread: the endpointer and recognizer are ours
                    was_paused = True
                    try:
                        self._abort()
                    except Exception as exc:
                        log.warning("voice listener: pause: %s", type(exc).__name__)
                while (chunk := self._source(0.0)) is not None:
                    self._last_end = chunk.stamp  # a pause is not lost audio
                try:  # an utterance whose audio went still gets its final
                    self.poll_handoff()
                except Exception as exc:
                    log.warning("voice listener: paused hand-off: %s", type(exc).__name__)
                self._stop.wait(POLL_S)
                continue
            was_paused = False
            head = self._queue[0] if self._queue else None
            wait = POLL_S if head is None else min(POLL_S, head.deadline - self._clock())
            chunk = self._source(max(0.0, wait))
            try:
                if chunk is not None:
                    self.process(chunk)
                t0 = self._timer()
                self.poll_handoff()  # a hand-off here is listener work outside ``process``
                self.load.busy(self._timer() - t0)
            except Exception as exc:  # one bad chunk must not end the listener
                self._count("errors")
                name = type(exc).__name__
                if name != self._last_error:
                    self._last_error = name
                    log.warning("voice listener: %s", name)
                try:
                    self._abort()
                except Exception as again:  # nor a recognizer that fails to reset
                    log.warning("voice listener: abort: %s", type(again).__name__)

    def _name_vocabulary(self) -> None:
        """A model loaded: check the robot's name against its vocabulary now (logs once)."""
        self._fast.names.refresh()
        self._fast.apply_config(self._cfg.voice)

    def _discard(self) -> None:
        """Without a working recognizer keep draining the hub's queue so it never backs up."""
        while not self._stop.is_set():
            self._source(POLL_S)

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def _count_audio(self, key: str) -> None:
        with self._lock:
            self._audio[key] += 1

    def _set_active(self, active: bool) -> None:
        """Track "an utterance is in progress" and tell ``on_activity`` (the Live tab's phase) when
        it changes. A raising hook is logged, never allowed to break the audio path."""
        changed = self._active != active
        self._active = active
        if changed and self._on_activity is not None:
            try:
                self._on_activity(active)
            except Exception as exc:
                log.warning("voice listener: activity hook: %s", type(exc).__name__)

    # ---- one chunk

    def _abort(self, wait: bool = False) -> None:
        """Drop the utterance in progress and every buffered sample; one that already ended is
        handed off first (``wait``: for its grammar pass, within its deadline). The barge-in gives
        the recognizer back first (``before_asr_reset``): its keyword worker may be using it."""
        self._settle(wait)
        self._recording = False
        if self._before_asr_reset is not None:
            try:
                self._before_asr_reset()
            except Exception as exc:
                log.warning("voice listener: asr hand-back: %s", type(exc).__name__)
        self._endpointer.reset()
        try:
            self._asr.reset()
        except Exception as exc:
            log.warning("voice listener: asr reset: %s", type(exc).__name__)
        self._fast.reset()
        self._set_active(False)

    def process(self, chunk: VoiceChunkLike) -> None:
        """Handle one chunk (see the module docstring) and account for it (``realtime``)."""
        t0 = self._timer()
        lag = self._clock() - chunk.stamp
        duration = chunk.samples.size / chunk.rate_hz
        lost = 0.0
        if chunk.gap and self._last_end is not None:
            lost = max(0.0, chunk.stamp - duration - self._last_end)
        self._last_end = chunk.stamp
        try:
            self._process(chunk, chunk.stamp - duration)
        finally:
            self.load.chunk(duration, self._timer() - t0, lag, lost)

    def _timed(self, stage: str, fn: Callable[..., Any], *args: Any) -> Any:
        """``fn(*args)``, its wall time added to ``stage`` of the realtime account."""
        t0 = self._timer()
        try:
            return fn(*args)
        finally:
            self.load.stage(stage, self._timer() - t0)

    def _process(self, chunk: VoiceChunkLike, start: float) -> None:
        if chunk.gap:
            self._count("gaps")
            self._abort(wait=True)
        speaking = self._speech.speaking(start) or self._speech.speaking(chunk.stamp)
        if speaking and start >= self._open_until:
            if not self._muted:
                self._muted = True
                self._abort()
            self._count("dropped_speaking")
            if self._on_speaking_chunk is not None:
                try:  # its hand-back (resume) replays audio: counted here and in its stages
                    self._timed("bargein", self._on_speaking_chunk, chunk)
                except Exception as exc:
                    log.warning("voice listener: speaking-chunk hook: %s", type(exc).__name__)
            return
        if self._muted and not speaking:
            self._muted = False
            self._speaking_ended()
        self._ingest(chunk.samples, start, chunk.stamp)

    def resume(self, audio: np.ndarray, start: float) -> None:
        """Barge-in handoff (SPEC 8 stage 8), called from the speaking-chunk hook after the
        speech was stopped: ``audio`` runs from the user's speech onset (pre-roll included)
        and starts at monotonic ``start``. The pipeline restarts on it, and the gate stays open
        until the speaking window the stop left behind (the echo tail) is over."""
        self._abort()
        self._muted = False
        self._open_until = self._speech.until()
        self._ingest(audio, start, start + audio.size / self._cfg.voice.sample_rate_hz)
        self._count("barge_ins")

    def _speaking_ended(self) -> None:
        if self._on_speaking_end is not None:
            try:
                self._on_speaking_end()
            except Exception as exc:
                log.warning("voice listener: speaking-end hook: %s", type(exc).__name__)

    def _ingest(self, samples: np.ndarray, start: float, audio_end: float) -> None:
        """Run ``samples`` (starting at ``start``) through the endpointer and the recognizer."""
        buffered: list[np.ndarray] = []
        for ev in self._timed("vad", self._endpointer.feed, samples, start):
            if isinstance(ev, SpeechAudio):
                buffered.append(ev.samples)
                continue
            self._feed(buffered, audio_end)
            buffered = []
            if isinstance(ev, SpeechStart):
                self._begin(ev, audio_end)
            elif isinstance(ev, SpeechEnd):
                self._end(ev)
        self._feed(buffered, audio_end)

    def _feed(self, parts: list[np.ndarray], audio_end: float) -> None:
        if parts and self._recording:
            if self._vosk:
                self._partial(
                    self._timed("asr_accept", self._asr.accept, np.concatenate(parts)), audio_end
                )
            self._sample_direction(audio_end)

    def _partial(self, event: ASREvent | None, audio_end: float) -> None:
        if event is not None and event.text:
            self._fast.partial(event.text, audio_end)

    # ---- utterance edges

    def _begin(self, ev: SpeechStart, audio_end: float) -> None:
        self._settle(wait=True)  # the per-utterance state below is the pending one's until then
        try:
            speaker = self._speaker_language()
        except Exception:
            speaker = None
        self._language = resolve_language(
            self._asr.languages, speaker, self._cfg.conversation.language
        )
        self._fast.begin(ev.stamp, audio_end, self._language)
        self._vosk = not self._addressable(ev.stamp)
        if self._vosk:
            self._timed("asr_start", self._asr.start, self._language)
        else:
            self._count_audio("vosk_skipped")  # its audio goes if the gate addresses it
        self._recording = True
        self._set_active(True)
        if self._direction is not None:
            self._direction.begin()
        self._sample_direction(audio_end)
        if self._vosk:
            self._partial(self._timed("asr_accept", self._asr.accept, ev.pre_roll), audio_end)

    def _wants_audio(self) -> bool:
        """Whether an addressed utterance's audio would go to the realtime route now (cheap:
        no network, no lock held across it)."""
        rt = self._realtime
        if rt is None:
            return False
        try:
            return bool(rt.wants(self._cfg))
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return False

    def _addressable(self, start: float) -> bool:
        """Whether the utterance starting at ``start`` skips Vosk: its audio would go, and the
        wake gate would address it without the wake word (its voice-at-face check aside)."""
        if not self._wants_audio():
            return False
        return self._gate.addressable(
            enrolled_in_view=self._enrolled(start), start=start, speaker=self._speaker()
        )

    def _end(self, ev: SpeechEnd) -> None:
        """The utterance ended: the recognizer's final (seconds on a loaded robot) and the
        grammar pass submitted. The Live phase stays ``listening`` until it is handed off or
        dropped."""
        if not self._recording:
            return
        self._recording = False
        face: bool | None = None
        if not self._vosk:  # buffered only: addressed now, or decoded from the buffer
            face = self._face()
            if self._send_unheard(ev, face):
                return
            text = self._decode(self._language, ev)
        else:
            try:
                text = self._timed("asr_finish", self._asr.finish).strip()
            except BaseException:
                self._set_active(bool(self._queue))
                raise
        if not text:
            self._count("empty")
            self._fast.reset()
            self._set_active(bool(self._queue))
            return
        grammar_pass = self._fast.grammar_pass(text, ev.audio[: self._last_sample(ev)])
        now = self._clock()
        deadline = now + self._cfg.voice.grammar.handoff_timeout_s
        job = None if grammar_pass is None else self._grammar.submit(grammar_pass.run, deadline)
        held = self._fast.hold()
        self._queue.append(
            _Handoff(ev, text, self._language, grammar_pass, job, now, deadline, held, face=face)
        )
        self.poll_handoff()

    def _face(self) -> bool | None:
        """The voice-at-face check, asked once for an utterance; None without one."""
        check = self._at_face()
        return None if check is None else bool(check())

    def _send_unheard(self, ev: SpeechEnd, face: bool | None) -> bool:
        """An utterance that skipped Vosk ended: if the wake gate addresses it without the
        wake word (its voice-at-face check passed) its audio goes and its final waits for the
        server. False: it is decoded locally (the check rejected it, or the window closed)."""
        if self._realtime is None:  # unbound meanwhile
            return False
        start, speaker = ev.start_stamp, self._speaker()
        enrolled = self._enrolled(start)
        if face is False or not self._gate.addressable(
            enrolled_in_view=enrolled, start=start, speaker=speaker
        ):
            return False
        addressed = self._gate.decide(
            wake_word=False,
            enrolled_in_view=enrolled,
            start=start,
            last_speech=ev.last_speech_stamp,
            speaker=speaker,
            at_face=None if face is None else (lambda: face),
        )
        now = self._clock()
        held = self._fast.hold()
        p = _Handoff(ev, "", self._language, None, None, now, now, held, face=face)
        p.addressed = addressed
        if addressed:
            self._send(p)
        else:  # decided already (a reload changed the policy meanwhile): decoded, not re-decided
            p.text = self._decode(self._language, ev)
            p.deadline = now
        self._queue.append(p)
        self.poll_handoff()
        return True

    def _send(self, p: _Handoff) -> None:
        """Hand ``p``'s audio to the realtime input (never blocks); its final now waits for the
        server's transcript until ``transcript_timeout_s``."""
        rate = self._cfg.voice.sample_rate_hz
        p.deadline = self._clock() + self._cfg.voice.realtime_route.transcript_timeout_s
        # past its deadline it is never sent: the listener has decoded it locally by then
        p.rt = self._realtime.submit(
            self._speech_audio(p.ev), rate, deadline=p.deadline, clock=self._clock
        )
        self._count_audio("sent")

    def _speech_audio(self, ev: SpeechEnd) -> np.ndarray:
        """The utterance's audio from its first to its last speech frame, pre-roll included."""
        return ev.audio[: self._last_sample(ev)]

    def _decode(self, language: str, ev: SpeechEnd) -> str:
        """The local final of an utterance whose recognizer did not run: a whole-utterance
        decode on a recognizer of its own (the one in progress is not disturbed)."""
        try:
            text = self._timed("asr_decode", self._asr.decode, language, self._speech_audio(ev))
        except Exception as exc:
            log.warning("voice listener: decode: %s", type(exc).__name__)
            return ""
        return str(text).strip()

    def _last_sample(self, ev: SpeechEnd) -> int:
        """Index in ``ev.audio`` just past the last speech frame (the end silence off)."""
        rate = self._cfg.voice.sample_rate_hz
        return round((ev.last_speech_stamp - ev.audio_stamp + self._endpointer.frame_s) * rate)

    def poll_handoff(self) -> None:
        """Hand waiting utterances off, in order, while the first one's final is in: its
        grammar pass is done or its wait is over, or the realtime route's transcript came or
        its deadline passed (the listener thread calls this after every chunk and every empty
        read)."""
        while self._queue:
            p = self._queue[0]
            if p.rt is not None:
                if not self._rt_settled(p):
                    if self._clock() < p.deadline:
                        return
                    self._count_audio("transcript_timeouts")
                    self._rt_timeout(p)
                self._hand_off(frozenset())
            elif p.job is None or (p.job.done and p.job.ran):
                self._hand_off(p.job.result if p.job is not None else frozenset())
            elif p.job.done or self._clock() >= p.deadline:  # skipped as stale, or out of time
                self._count_handoff("timeouts")
                self._hand_off(frozenset())
            else:
                return

    def _rt_settled(self, p: _Handoff) -> bool:
        try:
            return bool(self._realtime.settled(p.rt))
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return True  # handled as a failure: decoded locally

    def _rt_timeout(self, p: _Handoff) -> None:
        """No transcript within ``transcript_timeout_s``: the job is cancelled (its server
        items are deleted) and the utterance is decoded locally."""
        try:
            self._realtime.cancel(p.rt)
            self._realtime.route.count("transcript_timeouts")
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _settle(self, wait: bool) -> None:
        """Hand the waiting utterances off now, in order; with ``wait``, after waiting for a
        grammar pass up to its deadline (else a timeout), without, with its result only if it
        is in (else superseded). An utterance waiting for the realtime route's transcript is
        never waited for (that is the network): it and those behind it wait on, handed off by
        ``poll_handoff``."""
        while self._queue:
            p = self._queue[0]
            if p.rt is not None:
                self.poll_handoff()
                if self._queue and self._queue[0] is p:
                    return
                continue
            if wait and p.job is not None and not p.job.done:
                self._timed("grammar_wait", self._wait_for, p)
                if self._stop.is_set() and not p.job.done:
                    self._drop_pending()
                    return
            if p.job is not None and not p.job.done:
                self._count_handoff("timeouts" if wait else "superseded")
                self._hand_off(frozenset())
            else:
                self.poll_handoff()
                if self._queue and self._queue[0] is p:
                    return

    def _wait_for(self, p: _Handoff) -> None:
        """Wait for ``p``'s pass until its deadline (as wall time from now), or ``stop``."""
        assert p.job is not None
        end = self._timer() + max(0.0, p.deadline - self._clock())
        while not p.job.done and not self._stop.is_set():
            left = end - self._timer()
            if left <= 0:
                return
            p.job.wait(min(POLL_S, left))

    def _drop_pending(self) -> None:
        """The listener stops with utterances still waiting: drop them, counted (a realtime
        job is cancelled: its server items are deleted)."""
        if not self._queue:
            return
        while self._queue:
            p = self._queue.popleft()
            if p.rt is not None and self._realtime is not None:
                try:
                    self._realtime.cancel(p.rt)
                except Exception as exc:
                    log.warning("voice listener: realtime input: %s", type(exc).__name__)
            self._count_handoff("dropped_on_stop")
        self._set_active(False)

    def _count_handoff(self, key: str) -> None:
        with self._lock:
            self._handoff[key] += 1

    def _hand_off(self, heard: frozenset[str]) -> None:
        if not self._queue:
            return
        p = self._queue.popleft()
        if p.rt is None:
            waited = self._clock() - p.ended
            with self._lock:
                self._handoff["waits"] += 1
                self._wait_last_s = waited
                self._wait_max_s = max(self._wait_max_s, waited)
        try:
            again = self._timed("handoff", self._deliver, p, heard)
            if again is not None:  # its audio went: it waits for the server's final first
                self._queue.appendleft(again)
        finally:
            self._set_active(self._recording or bool(self._queue))

    def _deliver(self, p: _Handoff, heard: frozenset[str]) -> _Handoff | None:
        """Decide and hand ``p`` off; returns it instead when its audio just went to the
        realtime route (its final then comes from the server)."""
        ev = p.ev
        if p.rt is not None:
            return self._deliver_heard(p)
        text = p.text
        if p.addressed is None:
            wake = self._fast.wakes(text, grammar_pass=p.grammar_pass, heard=heard, state=p.state)
            face = p.face
            p.addressed = self._gate.decide(
                wake_word=wake,
                enrolled_in_view=self._enrolled(ev.start_stamp),
                start=ev.start_stamp,
                last_speech=ev.last_speech_stamp,
                speaker=self._speaker(),
                at_face=self._at_face() if face is None else (lambda: face),
            )
            if p.addressed and self._wants_audio():
                p.heard = heard
                self._send(p)
                return p
        if not text:  # decoded to nothing after the decision
            self._count("empty")
            return None
        self._finish(p, text, p.grammar_pass, heard)
        return None

    def _deliver_heard(self, p: _Handoff) -> None:
        """``p``'s audio went to the realtime route: its final is the server's transcript
        (keywords, the wake word and the speaking rate come from it), else a local decode of
        its buffer, which the chat routes answer when the audio did leave."""
        job = p.rt
        result = job.result
        if result is not None:
            self._count_audio("transcribed")
            heard = (result.generation, result.items)
            self._finish(p, result.text, None, frozenset(), heard, server=True)
            return None
        self._count_audio("local_fallbacks")
        text = p.text or self._decode(p.language, p.ev)
        if not text:
            self._count("empty")
            return None
        self._finish(p, text, p.grammar_pass, p.heard, local_only=job.sent)
        return None

    def _finish(
        self,
        p: _Handoff,
        text: str,
        grammar_pass: GrammarPass | None,
        heard: frozenset[str],
        realtime: tuple[int, tuple[str, ...]] | None = None,
        *,
        local_only: bool = False,
        server: bool = False,
    ) -> None:
        """The utterance with its final ``text`` (its wake decision made): keywords, prosody
        and impulses, then the hand-off to the conversation and the bus."""
        ev = p.ev
        rate = self._cfg.voice.sample_rate_hz
        first = max(0, round((ev.start_stamp - ev.audio_stamp) * rate))
        fast = self._fast.end(
            text,
            ev.audio[first : self._last_sample(ev)],
            rate,
            last_speech_stamp=ev.last_speech_stamp,
            endpoint_stamp=ev.endpoint_stamp,
            grammar_pass=grammar_pass,
            heard=heard,
            state=p.state,
            server=server,
        )
        addressed = bool(p.addressed)
        self._fast.impulses(fast, addressed, state=p.state)
        utterance = Utterance(
            text,
            p.language,
            ev.start_stamp,
            ev.last_speech_stamp,
            ev.endpoint_stamp,
            addressed,
            fast.wake_word,
            fast.keywords,
            fast.prosody,
            realtime=realtime,
            local_only=local_only,
        )
        self._fast.stats.record("final_vs_last_speech", self._clock() - ev.last_speech_stamp)
        self._count("utterances")
        self._publish(utterance, len(text), len(text.split()))
        try:
            self._on_utterance(utterance)
        except Exception as exc:
            log.warning("voice listener: utterance consumer raised %s", type(exc).__name__)

    def _sample_direction(self, stamp: float) -> None:
        """One voice-at-face sample at ``stamp`` (skipped while the check is off: it scans the
        world model). Muted while the robot speaks: its sound direction is off then (barge-in)."""
        d = self._direction
        if d is not None and d.enabled:
            d.sample(stamp, muted=self._speech.speaking(stamp))

    def _at_face(self) -> Callable[[], bool] | None:
        """The voice-at-face check for the wake gate, None when there is none or it is off."""
        d = self._direction
        return d.check if d is not None and d.enabled else None

    def _speaker(self) -> str | None:
        try:
            return self._speaker_id()
        except Exception:
            return None

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
