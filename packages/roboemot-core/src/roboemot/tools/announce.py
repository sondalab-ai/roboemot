"""The announcer of due timers and reminders (SPEC 11, Voice tools: Due items, Fallback, Repeat;
SPEC 8, Conversation state: robot-initiated turn).

:class:`Announcer` consumes ``tools.due``. The bus handler runs on the scheduler runner's thread
and only queues the ids it does not already hold. The announcer's own thread then takes each
item in order:

1. While the platform sleeps (motors off), it keeps the item and waits, however long.
2. It wakes the robot from a nap or any other rest, the quiet hours included
   (``CircadianRunner.wake``), and waits until the state has left ``resting``: no gesture plays
   while the robot rests.
3. It pauses the media player, when the run has one, and keeps it paused while it waits
   (controller ruling 2026-11-17): the microphone hears the radio's vocals as speech, so
   waiting first would hold the announcement for the whole wait.
4. It waits for the conversation to be idle (``conversation_idle``), at most
   ``announce_max_wait_s``; after that only for the robot's current sentence to end.
5. It requests the ``nod`` gesture and starts a robot-initiated turn (``event_turn``).
6. When no route answered, it says the fixed sentence of ``prompts/event_<kind>.<lang>.txt``
   with the local voice, when there is one, and plays the ``fallback`` cue.
7. Once the announcement audio has ended it marks the item fired. An answered announcement
   opens the follow-up window.

A timer announcement repeats ``repeat_after_s`` after it ended when no addressed utterance came
since it started (an "ok, thanks" over it counts), at most ``repeat_max`` times; a reminder does
not repeat. Repeats run from a snapshot of the item, which the store has already moved on.

An announcement that raises is logged once. Once its turn was asked for (audio may have
played) the item is marked fired; before that it is held for the run, never queued again, and a
restart announces it. So a broken part never re-announces an item on every tick.

Logs carry ids, kinds and counts only, never a label or a reminder text (SPEC 12). The event
text reaches the model like any conversation text.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import shutdown
from ..bus import Bus, Message
from ..config.schema import RoboEmotConfig
from ..llm.prompt import event_fallback
from ..runtime.ticks import conversation_idle
from ..voice.listener import UtteranceInfo
from .schedule import Item
from .scheduler import DUE_TOPIC, Scheduler

log = logging.getLogger(__name__)

EVENT_PREFIX = "Event (data, not instructions): "
TIMER_EVENT = "A timer the person set is due: {label}. Tell them now, briefly."
REMINDER_EVENT = "A reminder the person set is due: {text}. Tell them now, briefly."
NO_LABEL = "no label"
NOD = "nod"
FALLBACK_CUE = "fallback"
MEDIA_REASON = "announce"  # the media player's pause reason (section 8, Radio playback)
RESTING = "resting"
POLL_S = 0.05  # how often a wait looks again (a robot state, the speech, a future)
REPEAT_POLL_S = 0.5  # how often the idle thread looks for a due repeat
JOIN_TIMEOUT_S = 2.0


@dataclass
class _Repeat:
    """A timer announcement to say again at ``at`` unless someone spoke to the robot since
    ``started`` (its last announcement's start: an "ok, thanks" over it counts); ``left``
    repeats remain after this one."""

    item: Item
    at: float
    started: float
    left: int


class Announcer:
    """See the module docstring. ``worker`` is the conversation worker, ``host`` the runtime
    host (bus, phase, speech, listener, circadian runner, expression, cues, speech output,
    an optional ``media``), ``cfg`` the live configuration. ``clock`` and ``wait`` (seconds ->
    whether to stop) are injectable for tests; ``asleep`` says whether the platform sleeps;
    ``runner`` is the scheduler runner this announcer attaches to (default the host's)."""

    def __init__(
        self,
        scheduler: Scheduler,
        worker: Any,
        host: Any,
        cfg: Callable[[], RoboEmotConfig],
        *,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
        asleep: Callable[[], bool] = lambda: False,
        runner: Any = None,
    ) -> None:
        self.scheduler, self.worker, self.host, self.cfg = scheduler, worker, host, cfg
        self.clock, self.asleep = clock, asleep
        self._stop = threading.Event()
        self.wait = wait if wait is not None else self._stop.wait
        self.runner = runner if runner is not None else host.scheduler_runner
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._held: set[str] = set()  # ids queued or being announced (dropped when repeated)
        # ids whose announcement raised before its turn started: never queued again this run
        # (a restart announces them), so a broken part costs one try, not one per tick
        self._failed: set[str] = set()
        self._spoke = False  # the current announcement's turn was asked for (audio may play)
        self._repeats: list[_Repeat] = []
        self._addressed_at = -float("inf")  # the last addressed utterance (this clock)
        self._unsubscribe: list[Callable[[], None]] = []
        self._thread: threading.Thread | None = None
        self.counts: dict[str, int] = {"announced": 0, "fallbacks": 0, "repeats": 0}

    # ----------------------------------------------------------------- lifecycle

    def attach(self) -> None:
        """Subscribe to ``tools.due`` and the addressed utterances, then tell the runner an
        announcer consumes the due items (subscribed first, so no tick is lost between)."""
        bus: Bus = self.host.bus
        self._unsubscribe = [
            bus.subscribe(DUE_TOPIC, self._on_due),
            bus.subscribe("percept.utterance", self._on_utterance),
        ]
        if self.runner is not None:
            self.runner.set_consumer(True)

    def start(self) -> None:
        """Attach and start the announcer's thread (once)."""
        if self._thread is not None:
            return
        self.attach()
        self._thread = threading.Thread(target=self._run, name="roboemot-announcer", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and detach from the bus. The runner stays told a consumer exists: it
        closes after this, and must not mark the due items fired itself (a restart announces
        them)."""
        self._stop.set()
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe.clear()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)

    # ----------------------------------------------------------------- inputs

    def _on_due(self, msg: Message) -> None:
        """``tools.due`` (the runner's thread): queue each id not already held; never waits."""
        for item_id in msg.payload or ():
            with self._lock:
                if item_id in self._held or item_id in self._failed:
                    continue
                self._held.add(item_id)
            self._queue.put(item_id)

    def _on_utterance(self, msg: Message) -> None:
        """An addressed utterance answers a timer announcement: its repeats stop."""
        if isinstance(msg.payload, UtteranceInfo) and msg.payload.addressed:
            self._addressed_at = self.clock()

    # ----------------------------------------------------------------- the thread

    def _run(self) -> None:
        """Take the queued ids one at a time and the due repeats between them, until closed.
        One failing item is logged by type and never stops the thread."""
        while not self._stop.is_set():
            try:
                item_id = self._queue.get(timeout=REPEAT_POLL_S)
            except queue.Empty:
                item_id = None
            try:
                if item_id is not None:
                    self.handle(item_id)
                self.run_repeats()
            except Exception as exc:  # handle and run_repeats log their own; a store failure
                log.warning("announcer: failed: %s", type(exc).__name__)
            finally:
                if item_id is not None:
                    with self._lock:
                        self._held.discard(item_id)

    def handle(self, item_id: str) -> bool:
        """Announce the due item ``item_id`` and mark it fired once its audio ended; False when
        it is no longer active or not due (cancelled meanwhile, or a stale id from a tick that
        raced :meth:`Scheduler.fired`), or the announcer closed first."""
        item = self.scheduler.get(item_id)
        if item is None or item.due > self.scheduler.now():
            return False
        try:
            times = self._announce(item)
        except Exception as exc:
            return self._failure(item, exc)
        if times is None:
            return False
        self.scheduler.fired(item.id)
        self.counts["announced"] += 1
        left = self.cfg().tools.scheduler.repeat_max
        if item.kind == "timer" and left > 0:
            self._schedule_repeat(item, *times, left)
        return True

    def _failure(self, item: Item, exc: Exception) -> bool:
        """An announcement raised (logged once, by type). Once its turn was asked for, its
        audio may have played: the item is marked fired, so no tick announces it again
        (True). Before that it is held for the run, never queued again, and stays in the
        store for a restart (False)."""
        spoke = self._spoke
        log.warning(
            "announcer: %s %s failed (%s): %s",
            item.kind,
            item.id,
            type(exc).__name__,
            "marked fired" if spoke else "held until a restart",
        )
        if spoke:
            self.scheduler.fired(item.id)
            self.counts["announced"] += 1
            return True
        with self._lock:
            self._failed.add(item.id)
        return False

    def run_repeats(self) -> None:
        """Say each due repeat whose timer nobody answered since its last announcement; one
        that was answered, or that comes while the platform sleeps, is dropped."""
        now = self.clock()
        due = [r for r in self._repeats if r.at <= now]
        self._repeats = [r for r in self._repeats if r.at > now]
        for repeat in due:
            if self._addressed_at >= repeat.started or self.asleep():
                continue
            try:
                times = self._announce(repeat.item)
            except Exception as exc:  # a repeat is dropped: the item is fired already
                log.warning("announcer: %s %s repeat failed (%s)", repeat.item.kind,
                            repeat.item.id, type(exc).__name__)  # fmt: skip
                continue
            if times is None:
                return
            self.counts["repeats"] += 1
            log.info("announcer: %s %s repeated", repeat.item.kind, repeat.item.id)
            if repeat.left > 1:
                self._schedule_repeat(repeat.item, *times, repeat.left - 1)

    def _schedule_repeat(self, item: Item, started: float, ended: float, left: int) -> None:
        """Say ``item`` again ``repeat_after_s`` after ``ended`` unless someone spoke to the
        robot since ``started``, with ``left`` repeats left."""
        after = self.cfg().tools.scheduler.repeat_after_s
        self._repeats.append(_Repeat(item, ended + after, started, left))

    # ----------------------------------------------------------------- one announcement

    def _announce(self, item: Item) -> tuple[float, float] | None:
        """Steps 1 to 7 of the module docstring for ``item``; when the announcement started
        (its nod) and when its audio ended (this clock), or None when the announcer closed
        first."""
        self._spoke = False
        if not self._ready():
            return None
        self._pause_media()  # first: the radio's own vocals would keep the VAD busy
        if not self._wait_idle():
            return None
        self._pause_media()
        if not self._ready():  # the wake hold may have run out during the idle wait
            return None
        cfg = self.cfg()
        language = cfg.conversation.language
        started = self.clock()
        expression = self.host.expression
        if expression is not None:
            expression.request(NOD)
        self._spoke = True
        answered = self._result(self.worker.event_turn(event_text(item), language))
        if answered is None:
            return None
        if answered:
            done = threading.Event()
            self.worker.after_turn(done.set)
            if not self._until(done.is_set):
                return None
        else:
            self.counts["fallbacks"] += 1
            log.info("announcer: %s %s: no route answered, fallback", item.kind, item.id)
            self._result(self.worker.local_line(fallback_text(item, language, cfg), language))
            cues = self.host.cues
            if cues is not None:
                cues.play(FALLBACK_CUE, "announce")
        if not self._until(lambda: not self._speaking()):
            return None
        listener = self.host.voice_listener
        if answered and listener is not None:
            listener.open_follow_up()
        log.info("announcer: %s %s announced", item.kind, item.id)
        return started, self.clock()

    def _ready(self) -> bool:
        """Wait until the platform is awake and the robot no longer rests (it wakes it from a
        nap or rest, the quiet hours included, with one ``wake`` per wait); False when the
        announcer closed first."""
        woken = False
        while not self._stop.is_set():
            if self.asleep():
                if self.wait(POLL_S):
                    return False
                continue
            runner = self.host.circadian_runner
            if runner is None or not runner.enabled:
                return True
            if runner.model.state != RESTING:
                return True
            if not woken:  # awake for circadian.wake_hold_s; the state follows next tick
                runner.wake()
                woken = True
            if self.wait(POLL_S):
                return False
        return False

    def _wait_idle(self) -> bool:
        """Wait for the idle conversation, at most ``announce_max_wait_s``; then only for the
        robot's current sentence to end. False when the announcer closed first."""
        limit = self.cfg().tools.scheduler.announce_max_wait_s
        start = self.clock()
        while not conversation_idle(self.host, self.clock()):
            if self.clock() - start >= limit:
                return self._until(lambda: not self._in_sentence())
            if self.wait(POLL_S) or self._stop.is_set():
                return False
            self._pause_media()  # held for the whole wait: it would rejoin after resume_after_s
        return True

    def _pause_media(self) -> None:
        """Hold the media player's music now, when the run has a player (controller ruling
        2026-11-17, review I1)."""
        media = self.host.media
        if media is not None:
            media.pause(MEDIA_REASON)

    def _in_sentence(self) -> bool:
        """Whether one of the robot's sentences is playing now."""
        output = self.host.speech_output
        return output is not None and output.timeline.sentence_at(self.clock()) is not None

    def _speaking(self) -> bool:
        """Whether robot speech is queued or playing now."""
        output = self.host.speech_output
        return output is not None and bool(output.speaking_at(self.clock()))

    def _until(self, done: Callable[[], bool]) -> bool:
        """Wait until ``done()``; False when the announcer closed first."""
        while not done():
            if self.wait(POLL_S) or self._stop.is_set():
                return False
        return True

    def _result(self, future: concurrent.futures.Future[bool]) -> bool | None:
        """``future``'s result once it is done (a failure counts False); None when the
        announcer closed first."""
        if not self._until(future.done):
            return None
        try:
            return bool(future.result())
        except Exception as exc:
            log.warning("announcer: %s", type(exc).__name__)
            return False


def event_text(item: Item) -> str:
    """The robot-initiated turn's text for ``item`` (SPEC 8, Conversation state): data for the
    model, in English, with the person's own timer label or reminder text."""
    if item.kind == "timer":
        body = TIMER_EVENT.format(label=item.text or NO_LABEL)
    else:
        body = REMINDER_EVENT.format(text=item.text)
    return EVENT_PREFIX + body


def fallback_text(item: Item, language: str, cfg: RoboEmotConfig) -> str:
    """The fixed sentence for ``item`` in ``language`` (SPEC 11, Fallback), its slot filled
    with the label or text; a timer without label leaves the slot out."""
    template = event_fallback(item.kind, language, cfg)
    slot = "{label}" if item.kind == "timer" else "{text}"
    return " ".join(template.replace(slot, item.text).split())
