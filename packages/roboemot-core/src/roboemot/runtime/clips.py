"""The robot's status lines in its own cloud voice (SPEC 8, Realtime speech route, Lost reply
retry; SPEC 12, Data; SPEC 12A).

Three lines per language (``voice.realtime_route.retry``: ``retrying_lines``, ``sorry_lines``,
``thinking_lines``) are kept as saved clips (``voice.clip_cache``), so the robot can say them
while the cloud does not answer.

Recording: a thread looks every ``RECORD_POLL_S``. When a realtime session is up, the
conversation is idle (``conversation_idle``) and no media plays, it records one missing clip:
the conversation worker has the open session say the line once (``record_line``, holding the
audio input) into a :class:`CaptureSpeech`, which keeps the audio and plays none of it. The
clip is kept only when the reply's transcript says the line (case, accents and punctuation
ignored); at most ``retry.clip_record_attempts`` attempts per clip key in a day. It never opens
a session just to record.

Playback (``play``): the clip of the current key through the speech output at the circadian
volume (``volume_db``), like a cue; never over robot speech or audible media, and never while
someone's speech is in progress or starting (``hearing``, the listener's ``voice_active``): a
clip opens the speaking window and the listener drops what it hears then, so it would cut off
the person's first words (counted ``skipped_voice_active``). ``play`` runs on the listener's and
the worker's threads and reads no file: it plays only the clip loaded for the current key. The
recorder thread loads them (``preload``) before each look for a missing clip, so a new key's clip
plays within ``RECORD_POLL_S`` and an old key's never.

The key is the provider host and model of the configured realtime route, the voice
``session.update`` asks for (``realtime_voice``), ``conversation.language`` and the line: all
from the configuration, so a clip plays also while the session is down. A language with no line
has no clip.

Privacy (SPEC 12): the report and the logs hold kinds, states and counts, never a line.
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import logging
import threading
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

from .. import shutdown
from ..config.schema import RetryConfig, RoboEmotConfig, realtime_voice
from ..voice.clip_cache import CLIP_KINDS, Clip, ClipCache, ClipKey, ClipKind, ClipPlay
from ..voice.speech import StreamFeed
from .ticks import conversation_idle, media_on, robot_speaking

log = logging.getLogger(__name__)

RecordOutcome = Literal["skipped", "failed", "rejected_mismatch", "recorded"]
RECORD_POLL_S = 1.0  # how often the recorder looks for an idle moment and a missing clip
RESULT_POLL_S = 0.1  # how often a wait for a recording looks whether the recorder closed
SECONDS_PER_DAY = 86400.0
JOIN_TIMEOUT_S = 2.0
LINES: dict[ClipKind, Callable[[RetryConfig], dict[str, str]]] = {
    "retrying": lambda r: r.retrying_lines,
    "sorry": lambda r: r.sorry_lines,
    "thinking": lambda r: r.thinking_lines,
}


class CaptureSpeech:
    """The speech output a recording's response streams into: each stream job's audio is kept
    (``clip``), nothing plays. The worker sets ``outcome`` once the recording ended."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._feeds: list[StreamFeed] = []
        self.outcome: RecordOutcome = "skipped"

    def new_utterance(self) -> int:
        """An id for the recording's speech jobs (one group)."""
        return next(self._ids)

    def stream(
        self,
        chunks: StreamFeed,
        *,
        utterance: int,
        mood: str | None,
        volume_db: float,
        language: str = "",
    ) -> int:
        """Keep ``chunks`` (filled from the client's thread); its job id."""
        self._feeds.append(chunks)
        return next(self._ids)

    def mark_question(self, job_id: int) -> None:
        """A captured line plays nothing: a question mark changes nothing."""

    def clip(self) -> Clip | None:
        """The audio of every job, in order, as one clip; None when there is none or its
        pieces came at different rates."""
        pieces: list[np.ndarray] = []
        rates: set[int] = set()
        for feed in self._feeds:
            chunks, _ = feed.take()
            for samples, rate_hz in chunks:
                pieces.append(samples)
                rates.add(rate_hz)
        if not pieces or len(rates) != 1:
            return None
        return Clip(samples=np.concatenate(pieces).astype(np.float32), rate_hz=rates.pop())


class VoiceClips:
    """See the module docstring. ``worker`` is the conversation worker (``record_line``),
    ``host`` the runtime host (``realtime_route``, ``speech_output``, ``media`` and what
    ``conversation_idle`` reads), ``cfg`` the live configuration, ``cache`` the clip files,
    ``hearing`` whether someone's speech is in progress or starting, ``volume_db`` the
    circadian volume offset now."""

    def __init__(
        self,
        worker: Any,
        host: Any,
        cfg: Callable[[], RoboEmotConfig],
        cache: ClipCache,
        clock: Callable[[], float],
        *,
        hearing: Callable[[], bool],
        volume_db: Callable[[], float],
    ) -> None:
        self.worker, self.host, self.cfg, self.cache, self.clock = worker, host, cfg, cache, clock
        self.hearing, self.volume_db = hearing, volume_db
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # leaf: the loaded clips, the attempts and the counts
        # per kind, the key last loaded and its clip (None: no file for that key)
        self._loaded: dict[ClipKind, tuple[ClipKey, Clip | None]] = {}
        self._attempts: dict[ClipKey, collections.deque[float]] = {}
        self._counts: dict[ClipKind, collections.Counter[str]] = {
            kind: collections.Counter() for kind in CLIP_KINDS
        }
        self._inflight: concurrent.futures.Future | None = None

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the recorder's thread (once)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="roboemot-voice-clips", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the recorder; a recording asked and not yet done is given up. Close it before
        the conversation."""
        self._stop.set()
        inflight = self._inflight
        if inflight is not None:
            inflight.cancel()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.preload()
                self.record_next()
            except Exception as exc:
                log.warning("voice clips: recording failed (%s)", type(exc).__name__)
            self._stop.wait(RECORD_POLL_S)

    # ----------------------------------------------------------------- the key and the clip

    def key(self, kind: ClipKind) -> ClipKey | None:
        """What a clip of ``kind`` is recorded with now (see the module docstring); None with no
        realtime route, or no line for ``conversation.language``."""
        route = self.host.realtime_route
        if route is None:
            return None
        cfg = self.cfg()
        llm = route.route(cfg)
        if llm is None:
            return None
        settings = route.settings(cfg)
        language = cfg.conversation.language
        line = LINES[kind](settings.retry).get(language)
        if not line:
            return None
        provider = urlsplit(llm.base_url).hostname or llm.base_url
        return ClipKey(provider, llm.model, realtime_voice(llm, settings), language, line)

    def preload(self) -> None:
        """Load from its file the clip of each kind whose current key is not the one loaded
        (none there is remembered too, so a missing file is not read again each poll). The
        recorder's thread, off the audio path; ``play`` reads only what this loaded."""
        for kind in CLIP_KINDS:
            key = self.key(kind)
            if key is None or self._loaded_for(kind, key):
                continue
            clip = self.cache.get(kind, key)
            with self._lock:
                self._loaded[kind] = (key, clip)

    def _loaded_for(self, kind: ClipKind, key: ClipKey) -> bool:
        """Whether ``kind``'s entry in memory is for ``key`` (its clip, or that none exists)."""
        with self._lock:
            loaded = self._loaded.get(kind)
        return loaded is not None and loaded[0] == key

    def _clip(self, kind: ClipKind, key: ClipKey) -> Clip | None:
        """The clip of ``kind`` for ``key`` in memory; None when it is not loaded for that key.
        Never reads a file."""
        with self._lock:
            loaded = self._loaded.get(kind)
        return loaded[1] if loaded is not None and loaded[0] == key else None

    # ----------------------------------------------------------------- playback

    def play(self, kind: ClipKind) -> ClipPlay:
        """Queue the clip of ``kind`` on the speech output at the circadian volume. ``refused``
        with no line or no output, while someone speaks (counted), media is audible or the robot
        speaks, or when the output refuses it; else ``missing`` when its clip is not loaded.
        Any thread; reads no file."""
        key = self.key(kind)
        output = self.host.speech_output
        if key is None or output is None:
            return "refused"
        if self.hearing():
            with self._lock:
                self._counts[kind]["skipped_voice_active"] += 1
            return "refused"
        if media_on(self.host) or robot_speaking(self.host, self.clock()):
            return "refused"
        clip = self._clip(kind, key)
        if clip is None:
            return "missing"
        if output.play(clip.samples, clip.rate_hz, self.volume_db()) is None:
            return "refused"
        with self._lock:
            self._counts[kind]["played"] += 1
        return "played"

    # ----------------------------------------------------------------- recording

    def record_next(self) -> ClipKind | None:
        """Record one missing clip, when a session is up, the conversation is idle and no media
        plays (waits for it, on the caller's thread); the kind attempted, None when nothing was
        (a skip by the worker does not count as an attempt)."""
        route = self.host.realtime_route
        if route is None or not route.up():
            return None
        if not conversation_idle(self.host, self.clock()) or media_on(self.host):
            return None
        retry = self.cfg().voice.realtime_route.retry
        if not retry.enabled:
            return None  # no retry, no clip: nothing goes to the cloud for one
        attempts = retry.clip_record_attempts
        self.preload()  # a clip already in its file is not recorded again
        for kind in CLIP_KINDS:
            key = self.key(kind)
            if key is None or self._clip(kind, key) is not None or not self._may_try(key, attempts):
                continue
            capture = CaptureSpeech()
            future = self._inflight = self.worker.record_line(key.line, key.language, capture)
            clip = self._result(future, capture)
            self._inflight = None
            if capture.outcome == "skipped":
                return None
            with self._lock:
                self._attempts.setdefault(key, collections.deque()).append(self.clock())
                self._counts[kind][capture.outcome] += 1
            if clip is not None:
                self._store(kind, key, clip)
            return kind
        return None

    def _may_try(self, key: ClipKey, attempts: int) -> bool:
        """Whether ``key`` has had fewer than ``attempts`` recordings in the last day."""
        since = self.clock() - SECONDS_PER_DAY
        with self._lock:
            stamps = self._attempts.get(key)
            while stamps and stamps[0] <= since:
                stamps.popleft()
            return not stamps or len(stamps) < attempts

    def _result(self, future: concurrent.futures.Future, capture: CaptureSpeech) -> Clip | None:
        """``future``'s clip once done; None after a failure, a cancel or a close. A recording
        that raised or was cancelled other than by the close is ``failed`` (it counts as an
        attempt: its request may have gone out); only the worker's own skip stays ``skipped``."""
        while True:
            try:
                return future.result(timeout=RESULT_POLL_S)
            except TimeoutError:
                if self._stop.is_set():
                    future.cancel()
                    return None
            except concurrent.futures.CancelledError:
                if not self._stop.is_set():
                    capture.outcome = "failed"
                return None
            except Exception as exc:
                log.warning("voice clips: recording failed (%s)", type(exc).__name__)
                capture.outcome = "failed"
                return None

    def _store(self, kind: ClipKind, key: ClipKey, clip: Clip) -> None:
        """Keep ``clip`` in memory and in its file (a write error costs the file only)."""
        with self._lock:
            self._loaded[kind] = (key, clip)
        try:
            self.cache.put(kind, key, clip)
        except OSError as exc:
            log.warning("voice clips: %s clip not stored (%s)", kind, type(exc).__name__)

    # ----------------------------------------------------------------- report

    def snapshot(self) -> dict[str, Any]:
        """Per kind: ``cached`` (the clip of the current key is loaded), ``missing`` (a line
        exists for the language and its clip is not loaded), and the counts ``recorded``,
        ``rejected_mismatch``, ``failed`` (no reply, or no audio), ``played`` and
        ``skipped_voice_active`` (not played: someone was speaking) (SPEC 12A). Never a line."""
        out: dict[str, Any] = {}
        for kind in CLIP_KINDS:
            key = self.key(kind)
            cached = key is not None and self._clip(kind, key) is not None
            with self._lock:
                counts = self._counts[kind]
                out[kind] = {
                    "cached": cached,
                    "missing": key is not None and not cached,
                    "recorded": counts["recorded"],
                    "rejected_mismatch": counts["rejected_mismatch"],
                    "failed": counts["failed"],
                    "played": counts["played"],
                    "skipped_voice_active": counts["skipped_voice_active"],
                }
        return out
