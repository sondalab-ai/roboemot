"""The robot's own reactions, worded by the model (SPEC 8, Robot expression; owner decisions of
2026-11-20).

Triggers offer a :class:`ReactionRequest` (``offer``: any thread, never waits). The offer is
refused at once when the reactions or the situation are off, while the robot sleeps, rests or
the quiet hours run (``dropped_asleep``), while the radio plays (the situation's gesture alone,
``gesture_only_music``) and within the situation's cooldown for the same key
(``dropped_cooldown``; the cooldown starts at acceptance). The queue holds at most one request
per situation: a newer one replaces the older (``dropped_replaced``).

The expressor's thread takes one request at a time (``handle``):

1. Older than its situation's ``max_age_s``, or no longer valid (a shown thing that is no longer
   shown): ``dropped_stale``.
2. The robot sleeps, rests or the quiet hours run: ``dropped_asleep``. A reaction never wakes
   the robot.
3. The radio plays (a station held by the player, or audible): the gesture alone,
   ``gesture_only_music``. The player is only read, never paused, lowered or stopped, and the
   line is never said later.
4. It waits for the idle conversation (``conversation_idle``, and no announcement being
   handled), checking 1 to 3 again on every poll; past the max age: ``dropped_busy``.
5. Offline: the fixed line of ``prompts/expression_<situation>.<lang>.txt`` with the local voice
   (``offline_line``), else the situation's cue (``cue``) or nothing (``offline_silent``).
6. Else the situation's gesture and a reaction turn (``ConversationWorker.reaction_turn``) with
   an event text built from fixed English templates and data only (``spoken``). The worker
   gives way to a person (``busy``: back to 4); a route that does not answer in
   ``turn_timeout_s`` goes to 5. A spoken reaction opens the follow-up window once its audio
   ended, as an answered announcement does.

Privacy (SPEC 12): logs, decision events and the report carry the situation, the outcome, an age
and counts, never a name, a station or a category. The request's ``key`` (a cooldown key) and
``token`` never leave the robot.
"""

from __future__ import annotations

import collections
import concurrent.futures
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from .. import shutdown
from ..config.schema import RoboEmotConfig, Situation, SituationConfig
from ..diag import EventSink, no_events
from ..llm.prompt import expression_fallback
from ..tools.announce import EVENT_PREFIX
from .ticks import conversation_idle, room_quiet

log = logging.getLogger(__name__)

Outcome = Literal[
    "spoken",
    "gesture_only_music",
    "dropped_stale",
    "dropped_busy",
    "dropped_asleep",
    "dropped_cooldown",
    "dropped_replaced",
    "offline_line",
    "cue",
    "offline_silent",
]
OUTCOMES: tuple[str, ...] = get_args(Outcome)

DIDNT_CATCH_EVENT = (
    "Someone just spoke to you and you did not catch what they said. Ask them in one short "
    "sentence to say it again."
)
FAILED_ACTION_EVENT = (
    "The radio station {station} you were playing stopped: it did not load or it dropped. "
    "Tell the person in one short sentence."
)
SHOWN_OBJECT_EVENT = (
    "A person is showing you something that looks like a {category}. Comment on it or ask about "
    "it in one short sentence."
)
NO_MEMORY = (
    "Never mention anything from an earlier conversation, their preferences or what they told "
    "you before."
)
RETURNING_EVENT = (
    "{who} just came back after being away. Greet them by name in one short sentence. " + NO_MEMORY
)
WAKE_EVENT = (
    "You just woke up and {who} is here. Greet them for the time of day in one short sentence. "
    + NO_MEMORY
)
ANONYMOUS = "a person you know"  # {who} without an enrolled name
SLOTS = ("station", "category", "who")  # the data keys a template or a fixed line may hold
# the cue of a situation whose line cannot be said (SPEC 8, Non-verbal cues), and its reason
OFFLINE_CUES: dict[str, str] = {"didnt_catch": "not_heard", "failed_action": "fallback"}
CUE_REASON = "no_route"
RESTING = "resting"
RETURNING = "returning_person"
POLL_S = 0.05  # how often a wait looks again (the idle room, a future, the speech)
IDLE_POLL_S = 0.5  # how often the idle thread looks at its queue without a signal
JOIN_TIMEOUT_S = 2.0
SPACE_BEFORE_MARK = re.compile(r"\s+([!?.,;:])")


@dataclass(frozen=True)
class ReactionRequest:
    """One reaction to offer: its situation, the data its event text may hold (``station``,
    ``category``, ``who``, ``wake``), when its cause happened (the expressor's clock), the
    cooldown ``key`` ("", a station, a category or a person id) and the shown look's ``token``.
    ``key`` and ``token`` are never sent."""

    situation: Situation
    data: Mapping[str, str] = field(repr=False)
    created_at: float
    key: str = field(default="", repr=False)
    token: str = field(default="", repr=False)


class Expressor:
    """See the module docstring. ``worker`` is the conversation worker, ``host`` the runtime
    host (bus, phase, speech, listener, circadian runner, expression, cues, speech output, an
    optional ``media`` and ``announcer``), ``cfg`` the live configuration. ``online`` says
    whether a route can word a line now, ``asleep`` whether the platform sleeps. ``clock`` and
    ``wait`` (seconds -> whether to stop) are injectable for tests."""

    def __init__(
        self,
        worker: Any,
        host: Any,
        cfg: Callable[[], RoboEmotConfig],
        *,
        online: Callable[[], bool],
        asleep: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.worker, self.host, self.cfg = worker, host, cfg
        self.online, self.asleep, self.clock = online, asleep, clock
        self._stop = threading.Event()
        self.wait = wait if wait is not None else self._stop.wait
        self.events: EventSink = no_events  # the Live tab's decision events (wired at start)
        self._lock = threading.Lock()  # leaf: the queue, the cooldowns and the counts
        self._signal = threading.Event()
        self._queue: collections.OrderedDict[str, ReactionRequest] = collections.OrderedDict()
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._counts: dict[str, collections.Counter[str]] = {}
        self._unsubscribe: list[Callable[[], None]] = []
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------------------- lifecycle

    def attach(self) -> None:
        """Subscribe to the bus topics whose events offer reactions."""
        self._unsubscribe = []

    def start(self) -> None:
        """Attach and start the expressor's thread (once)."""
        if self._thread is not None:
            return
        self.attach()
        self._thread = threading.Thread(target=self._run, name="roboemot-expressor", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and detach from the bus. Close it before the conversation."""
        self._stop.set()
        self._signal.set()
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe.clear()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)

    # ----------------------------------------------------------------- offers

    def offer(self, request: ReactionRequest) -> bool:
        """Queue ``request`` for a spoken attempt (any thread; it never waits). In order: the
        reactions and the situation are on; the robot is awake (else ``dropped_asleep``); no
        music plays (else the gesture alone, ``gesture_only_music``); the situation's cooldown
        for the request's key has passed (else ``dropped_cooldown``). Then it is queued,
        replacing a queued request of its situation (``dropped_replaced``). True when queued."""
        reactions = self.cfg().expression.reactions
        sit = reactions.situation(request.situation)
        if not (reactions.enabled and sit.enabled):
            return False
        if self.resting():
            self._outcome(request, "dropped_asleep")
            return False
        if self.music():
            self._gesture(sit)
            self._outcome(request, "gesture_only_music")
            return False
        now = self.clock()
        replaced = None
        with self._lock:
            key = (request.situation, request.key)
            last = self._cooldowns.get(key)
            cooling = last is not None and now - last < sit.cooldown_s
            if not cooling:
                self._cooldowns[key] = now
                replaced = self._queue.pop(request.situation, None)
                self._queue[request.situation] = request
        if cooling:
            self._outcome(request, "dropped_cooldown")
            return False
        if replaced is not None:
            self._outcome(replaced, "dropped_replaced")
        self._signal.set()
        return True

    def _take(self) -> ReactionRequest | None:
        """The oldest queued request, taken off the queue (None when empty)."""
        with self._lock:
            if not self._queue:
                self._signal.clear()
                return None
            return self._queue.popitem(last=False)[1]

    # ----------------------------------------------------------------- the thread

    def _run(self) -> None:
        """Handle the queued requests one at a time until closed; one that raises is logged by
        type and never stops the thread."""
        while not self._stop.is_set():
            request = self._take()
            if request is None:
                self._signal.wait(IDLE_POLL_S)
                continue
            try:
                self.handle(request)
            except Exception as exc:
                log.warning(
                    "expressor: %s failed (%s)", request.situation, type(exc).__name__
                )  # fmt: skip

    def handle(self, request: ReactionRequest) -> Outcome:
        """One request, on the expressor's thread (steps 1 to 6 of the module docstring)."""
        reactions = self.cfg().expression.reactions
        sit = reactions.situation(request.situation)
        outcome = self._handle(request, sit)
        self._outcome(request, outcome)
        return outcome

    def _handle(self, request: ReactionRequest, sit: SituationConfig) -> Outcome:
        """``handle`` without the outcome's count and event."""
        waited = gestured = False
        while True:
            if self.clock() - request.created_at > sit.max_age_s:
                return "dropped_busy" if waited else "dropped_stale"
            if not self.valid(request):
                return "dropped_stale"
            if self.resting():
                return "dropped_asleep"
            if self.music():
                self._gesture(sit)
                return "gesture_only_music"
            if not self.idle():
                waited = True
                if self.wait(POLL_S) or self._stop.is_set():
                    return "dropped_busy"
                continue
            if not self.online():
                return self._offline(request)
            if not gestured:
                self._gesture(sit)
                gestured = True
            result = self._turn(request)
            if result == "busy":
                waited = True
                continue
            if result == "spoken":
                self._after_spoken()
                return "spoken"
            return self._offline(request)

    def _turn(self, request: ReactionRequest) -> str:
        """Start the reaction turn and wait for its result (``no_route`` when it raised or the
        expressor closed first)."""
        cfg = self.cfg()
        reactions = cfg.expression.reactions
        future = self.worker.reaction_turn(
            event_text(request),
            cfg.conversation.language,
            timeout_s=reactions.turn_timeout_s,
            quiet=lambda: room_quiet(self.host, self.clock()),
            profile=request.situation != RETURNING,
        )
        result = self._result(future)
        return "no_route" if result is None else str(result)

    def _after_spoken(self) -> None:
        """Once the reaction's audio ended, open the follow-up window, so the person can answer
        without the robot's name (as an answered announcement does)."""
        done = threading.Event()
        self.worker.after_turn(done.set)
        if not self._until(done.is_set) or not self._until(lambda: not self._speaking()):
            return
        listener = self.host.voice_listener
        if listener is not None:
            listener.open_follow_up()

    def _offline(self, request: ReactionRequest) -> Outcome:
        """No route words the line: the fixed line with the local voice, else the situation's
        cue, else nothing."""
        cfg = self.cfg()
        language = cfg.conversation.language
        if self.worker.local_voice():
            line = fallback_text(request, language, cfg)
            said = self._result(self.worker.local_line(line, language))
            if said:
                return "offline_line"
        cue = OFFLINE_CUES.get(request.situation)
        cues = self.host.cues
        if cue is None or cues is None:
            return "offline_silent"
        cues.play(cue, CUE_REASON)
        return "cue"

    # ----------------------------------------------------------------- checks

    def music(self) -> bool:
        """Whether the robot's radio plays: a station held by the player, or audible now. Only
        read (never paused, lowered or stopped)."""
        media = self.host.media
        return media is not None and (media.playing() or media.audible())

    def resting(self) -> bool:
        """Whether the platform sleeps, the robot rests or the quiet hours run (the rhythm on):
        a reaction never wakes the robot. Never called with the expressor's lock held."""
        if self.asleep():
            return True
        runner = self.host.circadian_runner
        if runner is None or not runner.enabled:
            return False
        return runner.model.state == RESTING or bool(runner.quiet_now())

    def idle(self) -> bool:
        """Whether the conversation is idle and no announcement is being handled."""
        if not conversation_idle(self.host, self.clock()):
            return False
        announcer = self.host.announcer
        return announcer is None or not announcer.busy()

    def valid(self, request: ReactionRequest) -> bool:
        """Whether ``request`` still applies (a shown thing still shown)."""
        return True

    def _speaking(self) -> bool:
        """Whether robot speech is queued or playing now."""
        output = self.host.speech_output
        return output is not None and bool(output.speaking_at(self.clock()))

    def _gesture(self, sit: SituationConfig) -> None:
        """Play the situation's gesture, when it has one (a manifest ``gesture_map`` name)."""
        expression = self.host.expression
        if sit.gesture and expression is not None:
            expression.request(sit.gesture)

    def _until(self, done: Callable[[], bool]) -> bool:
        """Wait until ``done()``; False when the expressor closed first."""
        while not done():
            if self.wait(POLL_S) or self._stop.is_set():
                return False
        return True

    def _result(self, future: concurrent.futures.Future) -> Any:
        """``future``'s result once done (a failure counts None); None when closed first."""
        if not self._until(future.done):
            return None
        try:
            return future.result()
        except Exception as exc:
            log.warning("expressor: %s", type(exc).__name__)
            return None

    # ----------------------------------------------------------------- report

    def _outcome(self, request: ReactionRequest, outcome: Outcome) -> None:
        """Count ``outcome`` and publish its decision event (codes and the age only)."""
        age = max(0.0, self.clock() - request.created_at)
        with self._lock:
            self._counts.setdefault(request.situation, collections.Counter())[outcome] += 1
        log.info("expressor: %s %s", request.situation, outcome)
        self.events("expression", outcome, situation=request.situation, age_s=age)

    def snapshot(self) -> dict[str, Any]:
        """Whether reactions are on, the queue length and the counts per situation and outcome
        (codes only, SPEC 12A)."""
        enabled = self.cfg().expression.reactions.enabled
        with self._lock:
            return {
                "enabled": enabled,
                "queued": len(self._queue),
                "counts": {s: dict(c) for s, c in self._counts.items()},
            }


def event_text(request: ReactionRequest) -> str:
    """The reaction turn's text (SPEC 8, 12): a fixed English template filled with data only
    (a station name, a detector category, an enrolled first name, a wake flag)."""
    data = request.data
    situation = request.situation
    if situation == "didnt_catch":
        body = DIDNT_CATCH_EVENT
    elif situation == "failed_action":
        body = FAILED_ACTION_EVENT.format(station=data.get("station", ""))
    elif situation == "shown_object":
        body = SHOWN_OBJECT_EVENT.format(category=data.get("category", ""))
    else:
        template = WAKE_EVENT if data.get("wake") else RETURNING_EVENT
        body = template.format(who=data.get("who") or ANONYMOUS)
    return EVENT_PREFIX + body


def fallback_text(request: ReactionRequest, language: str, cfg: RoboEmotConfig) -> str:
    """The fixed line for ``request`` in ``language`` (SPEC 8, Robot expression), its slots
    filled with the request's data; an empty slot leaves no space before the punctuation."""
    text = expression_fallback(request.situation, language, cfg)
    for slot in SLOTS:
        text = text.replace("{" + slot + "}", request.data.get(slot, ""))
    return SPACE_BEFORE_MARK.sub(r"\1", " ".join(text.split()))
