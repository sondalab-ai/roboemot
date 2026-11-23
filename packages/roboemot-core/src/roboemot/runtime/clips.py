"""The robot's status lines in its own cloud voice (SPEC 8, Realtime speech route, Lost reply
retry; SPEC 12, Data; SPEC 12A).

Three lines per language (``voice.realtime_route.retry``: ``retrying_lines``, ``sorry_lines``,
``thinking_lines``) are kept as saved clips (``voice.clip_cache``), so the robot can say them
while the cloud does not answer.

Recording: a thread looks every ``RECORD_POLL_S``. When a realtime session is up, the
conversation is idle (``conversation_idle``) and no media plays, it records one missing clip:
the conversation worker has the open session say the line once (``record_line``) into a
:class:`CaptureSpeech`, which keeps the audio and plays none of it. The clip is kept only when
the reply's transcript says the line (case, accents and punctuation ignored); at most
``retry.clip_record_attempts`` attempts per clip key in a day. It never opens a session just to
record.

Playback (``play``): the clip of the current key through the speech output, like a cue; never
over robot speech or audible media.

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
from ..voice.clip_cache import CLIP_KINDS, Clip, ClipCache, ClipKey, ClipKind
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
    ``conversation_idle`` reads), ``cfg`` the live configuration, ``cache`` the clip files."""

    def __init__(
        self,
        worker: Any,
        host: Any,
        cfg: Callable[[], RoboEmotConfig],
        cache: ClipCache,
        clock: Callable[[], float],
    ) -> None:
        self.worker, self.host, self.cfg, self.cache, self.clock = worker, host, cfg, cache, clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()  # leaf: the loaded clips, the attempts and the counts
        self._loaded: dict[ClipKind, tuple[ClipKey, Clip]] = {}
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

    def _clip(self, kind: ClipKind, key: ClipKey) -> Clip | None:
        """The clip of ``kind`` for ``key``: the one in memory, else read from its file."""
        with self._lock:
            loaded = self._loaded.get(kind)
        if loaded is not None and loaded[0] == key:
            return loaded[1]
        clip = self.cache.get(kind, key)
        if clip is not None:
            with self._lock:
                self._loaded[kind] = (key, clip)
        return clip

    # ----------------------------------------------------------------- playback

    def play(self, kind: ClipKind) -> bool:
        """Queue the clip of ``kind`` on the speech output; False when it is missing, media is
        audible or the robot speaks. Any thread."""
        key = self.key(kind)
        if key is None or media_on(self.host) or robot_speaking(self.host, self.clock()):
            return False
        clip = self._clip(kind, key)
        output = self.host.speech_output
        if clip is None or output is None:
            return False
        if output.play(clip.samples, clip.rate_hz) is None:
            return False
        with self._lock:
            self._counts[kind]["played"] += 1
        return True

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
        """Per kind: ``cached`` (the clip of the current key is there), ``missing`` (a line
        exists for the language and its clip does not), and the counts ``recorded``,
        ``rejected_mismatch``, ``failed`` (no reply, or no audio) and ``played`` (SPEC 12A).
        Never a line."""
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
                }
        return out
