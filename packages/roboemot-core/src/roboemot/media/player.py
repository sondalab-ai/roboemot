"""The media player (SPEC 8, Radio playback; SPEC 11, Voice tools, Media player; SPEC 12).

One network stream at a time, decoded to mono PCM at the voice hub's rate
(``voice.sample_rate_hz``) and pushed in pieces of ``tools.media.chunk_ms`` through
``SpeechOutput.play_media`` at ``tools.media.volume_db`` plus the ``radio_volume`` steps.

Pause, not mix (owner decision 2026-11-16): the feeder pushes nothing while robot speech, a cue
or an announcement is queued or playing (``SpeechOutput.media_blocked``, and ``play_media``
refuses it under the output's lock anyway), and while an addressed turn is open: the conversation
holds a turn open, served or waiting from the moment ``submit`` accepts an addressed utterance
(the wake word's included) until its reply ends (``ConversationWorker.busy``), and the phase is
``thinking`` or ``speaking``. It never reads the VAD, ``voice_active()`` or the ``listening``
phase: a robot without hardware echo cancellation hears the radio itself, and the music must
not pause on its own lyrics. While held it keeps reading and drops what it reads, so the stream
stays live, and it rejoins ``tools.media.resume_after_s`` after the last hold. ``pause(reason)``
(the announcer's) holds it now and cuts what is still in the speaker. An announcement's reply
window (``VoiceListener.reply_window_open``) holds it too, so the person can answer the robot
without its name (owner decision 2026-11-17).

Audible (SPEC 8, Radio playback): the music can be heard while a station is set and no hold
covers the moment, from a hold's start to its last moment plus ``resume_after_s``
(``audible_at``). The player keeps those quiet spans for ``music.history_s`` so the listener
decides an utterance by its start stamp: the radio rule (only the wake word addresses the robot)
applies only while the music is audible; while it is held the normal rules apply. ``playing``
(a station set, held or not) stays the nap's and the self-soothing's check.

Dance (SPEC 8, Radio playback): a robot whose microphone goes through a hardware echo canceller
does not hear its own radio, so the music detector cannot hear it from the microphone. Each
piece that went to the speaker is handed to ``on_piece(samples, rate_hz, start)`` (the
listener's ``hear_media``): decoded once, at the stream's own level (before the volume), and
stamped with the moment it starts to play (the stamp ``play_media`` returns, less the piece's
length). It runs on the feeder thread and must not block.

It stops on ``stop``, on a rest (``circadian.state`` ``resting``: a requested nap, the quiet
hours; while it plays the circadian runner starts no idle nap, owner decision 2026-11-17), on
``robot.asleep``, after ``tools.media.max_play_s``, and at a stream end, stall or error. It
publishes ``media.playing`` (``{"station"}``) and ``media.stopped`` (``{"station", "reason"}``)
with the station name only.
Logs carry the station name and a reason, never the URL, an exception's text or stream metadata.

It also stops when a config change (``on_config``, a ``ConfigStack`` listener) turns
``tools.radio.enabled`` off (reason ``disabled``) or removes the playing station from
``tools.radio.stations`` (reason ``removed``): withdrawing the opt-in ends the outbound stream.
On the realtime audio path an addressed utterance's final waits for the server's transcript
after the wake decision; the turn counts as open from then (``awaiting_transcript``).

Threads: one feeder thread (``media-player``) reads and pushes, and it alone closes a stream (a
stopped or replaced one is killed by whoever stops it and closed by the feeder, so a pipe is
never closed under a read in flight). ``play`` runs on the caller's thread (a skill's), kills
the stream that plays before it opens the new one, and blocks up to the connect timeout;
``stop`` and the bus callbacks (the sleep thread's ``robot.asleep``, the circadian runner's state)
only swap the state under the lock and kill the decoder, which ends the feeder's blocking read;
``close`` joins the feeder, then closes what is left.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from .. import shutdown
from ..bus import Message
from ..circadian.model import CircadianState
from ..config.schema import MEDIA_VOLUME_MAX_DB, MEDIA_VOLUME_MIN_DB, RoboEmotConfig, Station
from ..conversation.phase import SPEAKING, THINKING
from . import decode
from .decode import PcmStream

log = logging.getLogger(__name__)

PLAYING_TOPIC = "media.playing"
STOPPED_TOPIC = "media.stopped"
RESTING = "resting"
MS_PER_S = 1000.0
JOIN_TIMEOUT_S = 2.0
PlayStatus = Literal["ok", "unreachable", "disabled"]
Opener = Callable[[str, int, float], PcmStream]


class MediaPlayer:
    """See the module docstring. ``output`` is the speech output (``play_media``,
    ``media_blocked``, ``media_ahead_s``, ``cut_media``); ``host`` gives ``bus``,
    ``conversation`` and ``phase`` (each may be None); ``cfg`` reads the live configuration.
    ``opener`` (``decode.open_stream``), ``clock`` and ``wait`` (seconds -> whether the player
    closed) are injectable for tests."""

    def __init__(
        self,
        output: Any,
        host: Any,
        cfg: Callable[[], RoboEmotConfig],
        *,
        opener: Opener = decode.open_stream,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.output, self.host, self.cfg = output, host, cfg
        self._opener, self._clock = opener, clock
        self._closed = threading.Event()
        self._wait = wait if wait is not None else self._closed.wait
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._station: Station | None = None
        self._stream: PcmStream | None = None
        self._gen = 0  # moves on every play and stop: a feeder step on an old stream stops
        self._started = 0.0
        self._held_at = -math.inf  # the last moment something held the music
        # spans in which the music was not audible: each hold from its start to its last
        # moment plus ``resume_after_s``, merged when they touch, oldest first (``audible_at``)
        self._quiet: list[tuple[float, float]] = []
        self._offset_db = 0.0  # the radio_volume steps on top of tools.media.volume_db
        self._asleep = False
        self._retired: list[PcmStream] = []  # killed streams the feeder still has to close
        self._unsubscribe: list[Callable[[], None]] = []
        self._thread: threading.Thread | None = None
        # each piece that went to the speaker, with its rate and play stamp (the dance's audio)
        self.on_piece: Callable[[np.ndarray, int, float], None] | None = None
        self._piece_errors = 0  # consecutive on_piece failures (feeder thread)

    # ----------------------------------------------------------------- lifecycle

    def start(self, feeder: bool = True) -> None:
        """Listen for a nap and the platform's sleep on the bus and start the feeder thread
        (``feeder`` False: the subscriptions only, a test drives :meth:`step`)."""
        bus = self.host.bus
        if bus is not None:
            self._unsubscribe = [
                bus.subscribe("robot.asleep", self._on_asleep),
                bus.subscribe("robot.awake", self._on_awake),
                bus.subscribe("circadian.state", self._on_circadian),
            ]
        if feeder:
            self._thread = threading.Thread(target=self._run, name="media-player", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Stop what plays, drop the subscriptions and join the feeder."""
        self.stop("closed")
        self._closed.set()
        self._wake.set()
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe = []
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)
        if self._thread is None or not self._thread.is_alive():
            self._close_retired()  # no reader is left: closing here races no read

    def on_config(self, change: Any) -> None:
        """A ``ConfigStack`` listener: stop when ``change.config`` turns the radio off
        (``disabled``) or no longer lists the station that plays (``removed``)."""
        station = self._station
        if station is None:
            return
        radio = change.config.tools.radio
        if not radio.enabled:
            self.stop("disabled")
        elif station not in radio.stations:
            self.stop("removed")

    # ----------------------------------------------------------------- API

    def play(self, station: Station, connect_timeout_s: float | None = None) -> PlayStatus:
        """Play ``station`` instead of what plays now: ``ok`` once its first audio came,
        ``unreachable`` when none came within ``connect_timeout_s`` (default
        ``tools.media.connect_timeout_s``) or the decoder failed, ``disabled`` while the robot
        sleeps or the player is closed. Blocks the caller up to that timeout. What plays is
        stopped (``replaced``) before the new stream is opened, so two decoders never run."""
        cfg = self.cfg()
        if self._closed.is_set() or self._asleep:
            log.info("media: %s not played (disabled)", station.name)
            return "disabled"
        self.stop("replaced")
        timeout = (
            cfg.tools.media.connect_timeout_s if connect_timeout_s is None else connect_timeout_s
        )
        try:
            stream = self._opener(station.url, cfg.voice.sample_rate_hz, timeout)
        except Exception as exc:  # its text may quote the URL: the type only (SPEC 12)
            log.info("media: %s unreachable (%s)", station.name, type(exc).__name__)
            return "unreachable"
        with self._lock:
            if self._closed.is_set() or self._asleep:
                old, replaced, gone = None, None, stream
            else:  # another play may have won the race meanwhile: it is replaced too
                old, replaced, gone = self._stream, self._station, None
                self._stream, self._station = stream, station
                self._gen += 1
                self._started, self._held_at = self._clock(), -math.inf
                self._quiet = []
        if gone is not None:
            gone.close()  # never handed to the feeder: this thread is its only reader
            return "disabled"
        if old is not None:
            self._retire(old)
        if replaced is not None:
            self._publish(STOPPED_TOPIC, {"station": replaced.name, "reason": "replaced"})
        self._publish(PLAYING_TOPIC, {"station": station.name})
        log.info("media: playing %s", station.name)
        self._wake.set()
        return "ok"

    def stop(self, reason: str) -> None:
        """Stop what plays (nothing when nothing does) and cut the music still in the speaker.
        Any thread: it never waits for the feeder."""
        self._stop(reason, None)

    def _stop(self, reason: str, gen: int | None) -> None:
        """``stop``, only while the stream of generation ``gen`` plays (None: any)."""
        with self._lock:
            stream, station = self._stream, self._station
            if stream is None or (gen is not None and gen != self._gen):
                return
            self._stream = self._station = None
            self._gen += 1
        self._retire(stream)
        self.output.cut_media()
        assert station is not None
        self._publish(STOPPED_TOPIC, {"station": station.name, "reason": reason})
        log.info("media: %s stopped (%s)", station.name, reason)

    def _retire(self, stream: PcmStream) -> None:
        """Kill ``stream`` now (any thread) and leave its close to the feeder."""
        stream.kill()
        with self._lock:
            self._retired.append(stream)
        self._wake.set()

    def _close_retired(self) -> None:
        """Close the killed streams (the feeder's thread, or after it is gone)."""
        with self._lock:
            retired, self._retired = self._retired, []
        for stream in retired:
            stream.close()

    def pause(self, reason: str) -> None:
        """Hold the music now (``reason``: who asked, the announcer's ``announce``) and cut what
        is still in the speaker; it rejoins after ``tools.media.resume_after_s`` with no
        speech and no open addressed turn."""
        cfg = self.cfg()
        resume_after_s = cfg.tools.media.resume_after_s
        with self._lock:
            if self._stream is None:
                return
            fresh = self._clock() - self._held_at >= resume_after_s
            self._note_hold(self._clock(), cfg)
        self.output.cut_media()
        if fresh:  # a caller that keeps the music held calls this on every poll
            log.debug("media: paused (%s)", reason)

    def volume(self, delta_db: float) -> float:
        """Step the music volume by ``delta_db``, kept within ``tools.media.volume_db``'s
        range; the new volume in dB."""
        base = self.cfg().tools.media.volume_db
        with self._lock:
            level = _clamp(base + self._offset_db + delta_db)
            self._offset_db = level - base
        return level

    def volume_db(self) -> float:
        """The music volume now: ``tools.media.volume_db`` plus the steps, in its range."""
        with self._lock:
            offset = self._offset_db
        return _clamp(self.cfg().tools.media.volume_db + offset)

    def now_playing(self) -> Station | None:
        """The station that plays (paused or not), None when none does. Lock-free."""
        return self._station

    def playing(self) -> bool:
        """Whether a station is set (``media.playing`` holds), held or not: no idle nap and no
        self-soothing sound starts while it is."""
        return self._station is not None

    def audible(self) -> bool:
        """Whether the music can be heard now: ``audible_at`` the player's clock now."""
        return self.audible_at(self._clock())

    def audible_at(self, at: float) -> bool:
        """Whether the music could be heard at ``at`` (the monotonic clock): a station is set,
        it was set by then, and no hold covered ``at`` (from a hold's start to its last moment
        plus ``tools.media.resume_after_s``). The listener's radio rule (the wake gate's media
        rule, the stream gate, the name-first shortcut, the dance's audio source). Any thread:
        it takes only this player's leaf lock and reads no config, so the wake gate may ask it
        under its own lock. A hold is known from the feeder's next step (at most one piece
        later) or from ``hold_from``."""
        with self._lock:
            if self._station is None or at < self._started:
                return False
            return not any(lo <= at <= hi for lo, hi in self._quiet)

    def hold_from(self, at: float) -> None:
        """Record a hold from ``at`` (the monotonic clock, possibly past): an addressed
        utterance ended at ``at`` while the music was audible, so its turn holds the music from
        then. The feeder only sees the turn at its next step, after the endpoint and the wake
        decision; an utterance that starts in between (the person goes on after the name) must
        find the music held (SPEC 8, Radio playback). Changes nothing else: the feeder pauses
        the music itself once it sees the turn. Any thread."""
        cfg = self.cfg()
        with self._lock:
            if self._station is None:
                return
            self._note_hold(at, cfg)

    # ----------------------------------------------------------------- feeder

    def _run(self) -> None:
        while not self._closed.is_set():
            self._close_retired()  # the reader's own thread: no read of them is in flight
            if self._stream is None:
                self._wake.wait()
                self._wake.clear()
                continue
            try:
                self.step()
            except Exception as exc:  # a feeder bug stops the music, never the run
                log.warning("media: feeder: %s", type(exc).__name__)
                self.stop("error")

    def step(self) -> None:
        """One feeder step: read one piece (blocking up to the connect timeout), then drop it
        while held or push it once the speaker holds at most one piece of music."""
        with self._lock:
            stream, gen, started = self._stream, self._gen, self._started
        if stream is None:
            return
        cfg = self.cfg()
        media = cfg.tools.media
        if self._clock() - started >= media.max_play_s:
            self._end(gen, "max_play")
            return
        rate = cfg.voice.sample_rate_hz
        piece_s = media.chunk_ms / MS_PER_S
        pcm = stream.read(max(1, round(piece_s * rate)))
        if pcm is None:
            self._end(gen, "dropped")
            return
        if not self._ready(gen, media.resume_after_s):
            return  # dropped: the live stream stays live
        while self.output.media_ahead_s() > piece_s:
            if self._wait(self.output.media_ahead_s() - piece_s) or gen != self._gen:
                return
        until = self.output.play_media(pcm, rate, self.volume_db())
        if not until:
            self._hold()  # speech came between the check and the push
            return
        self._hand_on(pcm, rate, until - pcm.size / rate)

    def _hand_on(self, pcm: np.ndarray, rate: int, start: float) -> None:
        """Give a piece that went to the speaker to ``on_piece``; a consumer error costs the
        piece, never the music."""
        on_piece = self.on_piece
        if on_piece is None:
            return
        try:
            on_piece(pcm, rate, start)
        except Exception as exc:  # logged once per run of failures, not once per piece
            self._piece_errors += 1
            if self._piece_errors == 1:
                log.warning("media: piece consumer: %s", type(exc).__name__)
            return
        self._piece_errors = 0

    def _ready(self, gen: int, resume_after_s: float) -> bool:
        """Whether a piece may be pushed now: the same stream still plays, nothing holds the
        music, and nothing held it for ``resume_after_s``."""
        if gen != self._gen:
            return False
        if self.output.media_blocked() or self._turn_open():
            self._hold()
            self.output.cut_media()  # the music already in the speaker stops now too
            return False
        with self._lock:
            return self._clock() - self._held_at >= resume_after_s

    def _hold(self) -> None:
        cfg = self.cfg()
        with self._lock:
            self._note_hold(self._clock(), cfg)

    def _note_hold(self, at: float, cfg: RoboEmotConfig) -> None:
        """Something held the music at ``at`` (lock held): it rejoins ``resume_after_s`` after
        the last hold, and ``[at, at + resume_after_s]`` joins the quiet spans, merged with any
        it touches. Spans that ended over ``music.history_s`` ago are dropped: the listener
        asks about an utterance's start no further back (as the music detector's intervals)."""
        self._held_at = max(self._held_at, at)
        lo, hi = at, at + cfg.tools.media.resume_after_s
        quiet = self._quiet
        if not quiet or lo > quiet[-1][1]:
            quiet.append((lo, hi))  # the feeder's usual case: a new span after the last
        elif lo >= quiet[-1][0]:
            quiet[-1] = (quiet[-1][0], max(hi, quiet[-1][1]))  # it goes on: grow the last
        else:  # back-dated (``hold_from``) before the last span: merge what it touches
            kept = [(a, b) for a, b in quiet if not (a <= hi and lo <= b)]
            for a, b in quiet:
                if a <= hi and lo <= b:
                    lo, hi = min(lo, a), max(hi, b)
            quiet[:] = sorted([*kept, (lo, hi)])
        horizon = self._clock() - cfg.music.history_s
        while quiet and quiet[0][1] < horizon:
            quiet.pop(0)

    def _turn_open(self) -> bool:
        """Whether an addressed turn is open: the conversation holds a turn open, served or
        waiting, an addressed utterance's audio waits for the server's transcript (the wake
        decision is taken, the final not yet), an announcement's reply window is open (owner
        decision 2026-11-17), or the phase is thinking or speaking. Never the VAD or the
        listening phase."""
        conversation = self.host.conversation
        if conversation is not None and conversation.busy():
            return True
        listener = self.host.voice_listener
        if listener is not None and (
            listener.awaiting_transcript() or listener.reply_window_open()
        ):
            return True
        phase = self.host.phase
        return phase is not None and phase.current()[0] in (THINKING, SPEAKING)

    def _end(self, gen: int, reason: str) -> None:
        """Stop the stream of generation ``gen`` for ``reason`` unless a play or a stop came
        meanwhile."""
        self._stop(reason, gen)

    # ----------------------------------------------------------------- bus

    def _on_asleep(self, msg: Message) -> None:
        self._asleep = True
        self.stop("asleep")

    def _on_awake(self, msg: Message) -> None:
        self._asleep = False

    def _on_circadian(self, msg: Message) -> None:
        if isinstance(msg.payload, CircadianState) and msg.payload.state == RESTING:
            self.stop("nap")

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        bus = self.host.bus
        if bus is None:
            return
        try:
            bus.publish(topic, payload, self._clock())
        except Exception as exc:
            log.warning("media: %s subscriber raised %s", topic, type(exc).__name__)


def _clamp(level_db: float) -> float:
    """``level_db`` within the range of ``tools.media.volume_db``."""
    return min(MEDIA_VOLUME_MAX_DB, max(MEDIA_VOLUME_MIN_DB, level_db))
