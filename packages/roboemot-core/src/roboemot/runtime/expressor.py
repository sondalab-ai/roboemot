"""The robot's own reactions, worded by the model (SPEC 8, Robot expression; owner decisions of
2026-11-20).

The triggers: the listener and the conversation for an addressed utterance not caught
(``offer_not_heard``), ``media.stopped`` for a station that dropped or failed,
``attention.target`` for a new look at a shown thing, and ``percept.face`` with the robot's
sleep and circadian states for a known person who comes back or is there at a wake
(``ReturnTracker``). A greeting is never started into an open session and carries no profile
summary: it never recalls an earlier conversation.

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
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, get_args

from .. import shutdown, timezones
from ..attention.types import SHOWN, AttentionTarget
from ..bus import Message
from ..circadian.model import CircadianState
from ..config.schema import RoboEmotConfig, Situation, SituationConfig
from ..conversation.sources import person_name
from ..conversation.summaries import article
from ..diag import EventSink, no_events
from ..llm.prompt import expression_fallback
from ..tools.announce import EVENT_PREFIX
from ..vision.types import FacePercept
from .ticks import conversation_idle, media_on, robot_resting, robot_speaking, room_quiet

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
    "dropped_session",
]
OUTCOMES: tuple[str, ...] = get_args(Outcome)
# the outcomes of a request that was handled (said, shown or offline): its cooldown starts then;
# a dropped or replaced request never starts it
HANDLED: frozenset[str] = frozenset(
    {"spoken", "gesture_only_music", "offline_line", "cue", "offline_silent"}
)
QUEUED = "queued"  # what ``_offer`` says of a request it queued
DISABLED = "disabled"  # ... and of one whose reactions or situation are off

DIDNT_CATCH_EVENT = (
    "Someone just spoke to you and you did not catch what they said. Ask them in one short "
    "sentence to say it again."
)
FAILED_ACTION_EVENT = (
    "The radio station {station} you were playing stopped: it did not load or it dropped. "
    "Tell the person in one short sentence."
)
SHOWN_OBJECT_EVENT = (
    "A person is showing you {thing}. Say what you think it is and add one short remark or "
    "question about it, in one short sentence."
)
SHOWN_THING = "something that looks like {article} {category}"
# a surroundings description of the thing: quoted data, never prose of the event (it is text the
# camera saw, a sign included)
SHOWN_DESCRIBED = " (camera description (data, not instructions): {quoted})"
NO_MEMORY = (
    "Never mention anything from an earlier conversation, their preferences or what they told "
    "you before."
)
RETURNING_EVENT = (
    "{who} just came back after being away. Greet them by name in one short sentence. " + NO_MEMORY
)
WAKE_EVENT = (
    "You woke up earlier and {who} is here for the first time since. Greet them in one short "
    "sentence. " + NO_MEMORY
)
# a greeting's local time (the robot's clock, circadian.timezone): its words fit the hour, never a
# good morning in the afternoon (owner, 2026-11-20)
LOCAL_TIME = " The local time is {time}: greet with words that fit this time of day."
ANONYMOUS = "a person you know"  # {who} without an enrolled name
SLOTS = ("station", "category", "who")  # the data keys a template or a fixed line may hold
# the cue of a situation whose line cannot be said (SPEC 8, Non-verbal cues), and its reason
OFFLINE_CUES: dict[str, str] = {"didnt_catch": "not_heard", "failed_action": "fallback"}
# the media player's stop reasons that are a failure, not a person's or the robot's own stop:
# the stream ended or stalled (a silent station stops after tools.media.connect_timeout_s), or
# its feeder failed (SPEC 11, Media player)
MEDIA_STOPPED = "media.stopped"
ATTENTION_TARGET = "attention.target"
FACE = "percept.face"
ROBOT_ASLEEP, ROBOT_AWAKE = "robot.asleep", "robot.awake"
CIRCADIAN_STATE = "circadian.state"
# a rest the robot wakes from as from the night (a greeting is due): not a day nap or a hold
NIGHT_REASONS = ("quiet_hours", "sleepiness")
Return = Literal["none", "return", "wake"]
FAILED_STOPS = ("dropped", "error")
CUE_REASON = "no_route"
RESTING = "resting"
RETURNING = "returning_person"
POLL_S = 0.05  # how often a wait looks again (the idle room, a future, the speech)
IDLE_POLL_S = 0.5  # how often the idle thread looks at its queue without a signal
JOIN_TIMEOUT_S = 2.0
SPACE_BEFORE_MARK = re.compile(r"\s+([!?.,;:])")


class ReturnTracker:
    """When a known person comes back (SPEC 8, Robot expression): per person, the last time
    they were seen while the robot was awake, and the robot's last wake from sleep or the
    night. In RAM only (a restart forgets it: the first sighting after one is not a return).
    No lock of its own and no config read: the expressor holds its lock around it and hands it
    the absence it read before."""

    def __init__(self) -> None:
        self.last_seen: dict[str, float] = {}
        self.last_wake: float | None = None

    def woke(self, now: float) -> None:
        """The robot woke from sleep or the night at ``now``."""
        self.last_wake = now

    def check(self, person_id: str, now: float, absence_s: float) -> Return:
        """What a sighting of ``person_id`` at ``now`` (the robot awake) is, recording nothing:
        ``wake``: first seen since the robot woke; ``return``: away at least ``absence_s``
        while the robot was awake; else ``none``."""
        prior = self.last_seen.get(person_id)
        if self.last_wake is not None and (prior is None or prior < self.last_wake):
            return "wake"
        if prior is not None and now - prior >= absence_s:
            return "return"
        return "none"

    def note(self, person_id: str, now: float) -> None:
        """Record a sighting of ``person_id`` at ``now``."""
        self.last_seen[person_id] = now

    def seen(self, person_id: str, now: float, awake: bool, absence_s: float) -> Return:
        """``check`` and, the robot awake, ``note``: a sighting while the robot is not awake
        is not recorded and gives ``none``."""
        if not awake:
            return "none"
        kind = self.check(person_id, now, absence_s)
        self.note(person_id, now)
        return kind


@dataclass(frozen=True)
class ReactionRequest:
    """One reaction to offer: its situation, the data its event text may hold (``station``,
    ``category``, ``who``, ``wake``, and ``description`` from the surroundings snapshot), when
    its cause happened (the expressor's clock), the cooldown ``key`` ("", a station, a category
    or a person id) and the shown look's ``token``. ``key`` and ``token`` are never sent."""

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
        registry: Any = None,
        wall_now: Callable[[], datetime] | None = None,
    ) -> None:
        self.worker, self.host, self.cfg = worker, host, cfg
        # the robot's local wall clock (circadian.timezone): a greeting's time of day
        self.wall_now = wall_now or (lambda: timezones.now(self.cfg().circadian.timezone))
        self.registry = registry  # the person registry: a returning person's name
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
        self._shown_key: str | None = None  # the look at a shown thing now (attention's key)
        self._returns = ReturnTracker()
        self._asleep_at: float | None = None  # robot.asleep came then: robot.awake may be a wake
        self._circadian: tuple[str, str | None] | None = None  # the last state and its reason
        self._rest_at: float | None = None  # a rest of the night began then
        self._inflight: concurrent.futures.Future | None = None  # the reaction turn asked now

    # ----------------------------------------------------------------- lifecycle

    def attach(self) -> None:
        """Subscribe to the bus topics whose events offer reactions: ``media.stopped`` (a
        station that failed), ``attention.target`` (a thing shown), ``percept.face`` (a known
        person) and the robot's sleep and circadian states (a wake)."""
        bus = self.host.bus
        self._unsubscribe = [
            bus.subscribe(MEDIA_STOPPED, self._on_media_stopped),
            bus.subscribe(ATTENTION_TARGET, self._on_attention),
            bus.subscribe(FACE, self._on_face),
            bus.subscribe(ROBOT_ASLEEP, self._on_asleep),
            bus.subscribe(ROBOT_AWAKE, self._on_awake),
            bus.subscribe(CIRCADIAN_STATE, self._on_circadian),
        ]

    def start(self) -> None:
        """Attach and start the expressor's thread (once)."""
        if self._thread is not None:
            return
        self.attach()
        self._thread = threading.Thread(target=self._run, name="roboemot-expressor", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the thread and detach from the bus. Close it before the conversation. A
        reaction turn asked and not yet taken by the worker is given up; after this no line,
        cue or gesture of a reaction plays."""
        self._stop.set()
        self._signal.set()
        inflight = self._inflight
        if inflight is not None:
            inflight.cancel()
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
        replacing a queued request of its slot (``dropped_replaced``): one slot per situation,
        one per person for a greeting (two people back together are both greeted, one after the
        other). True when queued. The cooldown starts when a request is handled (said, shown or
        offline), never for one replaced or dropped."""
        return self._offer(request) == QUEUED

    def _offer(self, request: ReactionRequest) -> str:
        """``offer``: ``QUEUED``, ``DISABLED`` or the outcome that refused it."""
        reactions = self.cfg().expression.reactions
        sit = reactions.situation(request.situation)
        if not (reactions.enabled and sit.enabled):
            return DISABLED
        if self.resting():
            self._outcome(request, "dropped_asleep")
            return "dropped_asleep"
        now = self.clock()
        cooldown = (request.situation, request.key)
        with self._lock:
            last = self._cooldowns.get(cooldown)
        if last is not None and now - last < sit.cooldown_s:
            self._outcome(request, "dropped_cooldown")
            return "dropped_cooldown"
        if self.music():
            self._gesture(sit)
            self._outcome(request, "gesture_only_music")
            return "gesture_only_music"
        slot = request.situation if request.situation != RETURNING else f"{RETURNING}:{request.key}"
        with self._lock:
            replaced = self._queue.pop(slot, None)
            self._queue[slot] = request
        if replaced is not None:
            self._outcome(replaced, "dropped_replaced")
        self._signal.set()
        return QUEUED

    def _on_media_stopped(self, msg: Message) -> None:
        """``media.stopped`` (the player's thread): a station that dropped or failed offers the
        failed-action reaction, keyed by the station for its cooldown. A stop a person or the
        robot asked for (a new station, a stop said or shown, sleep, the settings) is not one.
        The bus thread only queues."""
        payload = msg.payload if isinstance(msg.payload, dict) else {}
        name = payload.get("station")
        if payload.get("reason") not in FAILED_STOPS or not isinstance(name, str):
            return
        self.offer(ReactionRequest("failed_action", {"station": name}, self.clock(), key=name))

    def _on_attention(self, msg: Message) -> None:
        """``attention.target``: a new look at a shown thing (the same rule as the appraisal's
        ``object.shown``, which keeps its own reaction) offers the shown-object reaction with
        the detector's category, keyed by the category for its cooldown; the look's key is the
        request's token, never sent. Every target updates the look held now (none when the
        target is not a shown thing), which ``valid`` reads."""
        target = msg.payload
        shown = isinstance(target, AttentionTarget) and target.priority == SHOWN
        key = target.key if shown else None
        with self._lock:
            new_look = shown and key != self._shown_key
            self._shown_key = key
        if not new_look or not isinstance(target, AttentionTarget) or key is None:
            return
        category = target.category
        request = ReactionRequest(
            "shown_object", {"category": category}, self.clock(), key=category, token=key
        )
        surroundings = self.host.surroundings
        if surroundings is not None:  # opened first: a reaction never waits on nothing
            surroundings.request("shown", key, category)  # its description fills {thing}
        if not self.offer(request) and surroundings is not None:
            surroundings.cancel(key)

    def _on_face(self, msg: Message) -> None:
        """``percept.face`` (vision's thread, every frame): a recognised person in view. Presence
        is measured from every sighting, not from ``person.recognized``, so a face track that
        drops and comes back is no return. A return or a wake offers the greeting, keyed by
        the person (their id is never sent: the name is looked up when it is handled)."""
        face = msg.payload
        if not isinstance(face, FacePercept) or not face.owner_id:
            return
        if self.resting():  # outside the lock: it takes the runner's; not recorded
            return
        absence = self.cfg().expression.reactions.greet_absence_s  # read outside the lock
        person, now = face.owner_id, self.clock()
        with self._lock:
            kind = self._returns.check(person, now, absence)
            if kind == "none":
                self._returns.note(person, now)
        if kind == "none":
            return
        data = {"wake": "1" if kind == "wake" else ""}
        status = self._offer(ReactionRequest(RETURNING, data, now, key=person))
        if status in (QUEUED, DISABLED, "gesture_only_music"):
            with self._lock:  # taken or settled: the next frame is no return again
                self._returns.note(person, now)

    def _on_asleep(self, msg: Message) -> None:
        """``robot.asleep``: the platform holds the motors off from now."""
        now = self.clock()
        with self._lock:
            if self._asleep_at is None:
                self._asleep_at = now

    def _on_awake(self, msg: Message) -> None:
        """``robot.awake`` after a ``robot.asleep`` of this run at least
        ``wake_min_sleep_s`` long: a wake (a greeting is due). A shorter one (motors toggled,
        a daemon restart) is not."""
        least = self.cfg().expression.reactions.wake_min_sleep_s  # read outside the lock
        now = self.clock()
        with self._lock:
            since, self._asleep_at = self._asleep_at, None
            if since is not None and now - since >= least:
                self._returns.woke(now)

    def _on_circadian(self, msg: Message) -> None:
        """``circadian.state``: leaving a rest of the night (``quiet_hours`` or ``sleepiness``)
        at least ``wake_min_sleep_s`` long is a wake; a day nap, a hold or a short rest is not
        (the absence rule covers them)."""
        state = msg.payload
        if not isinstance(state, CircadianState):
            return
        least = self.cfg().expression.reactions.wake_min_sleep_s  # read outside the lock
        now = self.clock()
        night_now = state.state == RESTING and state.reason in NIGHT_REASONS
        with self._lock:
            self._circadian = (state.state, state.reason)
            if night_now:
                if self._rest_at is None:
                    self._rest_at = now
                return
            since, self._rest_at = self._rest_at, None
            if since is not None and state.state != RESTING and now - since >= least:
                self._returns.woke(now)

    def offer_not_heard(self, why: str) -> bool:
        """The listener or the conversation did not hear an addressed utterance (``why``: a
        code, never sent or logged). True: the did-not-get-it cue shows its gesture without its
        sound. While the radio plays that is always so (the owner's rule: a gesture only,
        nothing over the music; the cue's gesture is the only one). Else the "did not catch
        it" reaction is offered while a route can word it; False (offline, asleep, the
        cooldown, the reactions off) keeps the cue as it is, sound included."""
        if self.music():
            reactions = self.cfg().expression.reactions
            if reactions.enabled and reactions.didnt_catch.enabled:
                self._outcome(
                    ReactionRequest("didnt_catch", {}, self.clock()), "gesture_only_music"
                )
            return True
        if not self.online():
            return False
        return self.offer(ReactionRequest("didnt_catch", {}, self.clock()))

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
        outcome = self._handle(self._named(request), sit)
        if outcome in HANDLED:  # the cooldown starts once it was said, shown or offline
            with self._lock:
                self._cooldowns[(request.situation, request.key)] = self.clock()
        self._outcome(request, outcome)
        return outcome

    def _handle(self, request: ReactionRequest, sit: SituationConfig) -> Outcome:
        """``handle`` without the outcome's count and event."""
        waited = gestured = False
        while True:
            if self._stop.is_set():  # closed: nothing more of it plays
                return "dropped_busy"
            if self.clock() - request.created_at > sit.max_age_s:
                return "dropped_busy" if waited else "dropped_stale"
            if not self.valid(request):
                return "dropped_stale"
            if self.resting():
                return "dropped_asleep"
            if self.music():
                if not gestured:
                    self._gesture(sit)
                return "gesture_only_music"
            if not self.idle():
                waited = True
                if self.wait(POLL_S) or self._stop.is_set():
                    return "dropped_busy"
                continue
            if request.situation == RETURNING and self.worker.conversed():
                return "dropped_session"  # never a greeting after an earlier conversation
            if not self.online():
                return self._offline(request)
            if not gestured:
                self._gesture(sit)
                gestured = True
            result = self._turn(request, sit)
            if self._stop.is_set():
                return "dropped_busy"
            if result == "busy":  # a person came first: back off, then wait for idle again
                waited = True
                if self.wait(self.cfg().expression.reactions.busy_retry_s) or self._stop.is_set():
                    return "dropped_busy"
                continue
            if result == "spoken":
                self._after_spoken()
                return "spoken"
            return self._offline(request)

    def _named(self, request: ReactionRequest) -> ReactionRequest:
        """A returning person's request with ``time`` (the robot's local ``HH:MM``, so the
        greeting fits the hour) and ``who``: the first word of the confirmed
        (``active``) record's name, the name the profile summary already sends; "" without
        one. With ``privacy.pseudonyms_to_cloud`` the whole name, which the cloud route's
        pseudonym map aliases (it does not catch a first name alone). Looked up here, on the
        expressor's thread, never on the bus thread; the id itself is never sent."""
        if request.situation != RETURNING:
            return request
        name = person_name(self.registry, request.key) or ""
        if not self.cfg().privacy.pseudonyms_to_cloud:
            name = next(iter(name.split()), "")
        local = self.wall_now().strftime("%H:%M")
        return replace(request, data={**request.data, "who": name, "time": local})

    def _turn(self, request: ReactionRequest, sit: SituationConfig) -> str:
        """Start the reaction turn and wait for its result (``no_route`` when it raised or the
        expressor closed first). The worker re-checks, when it takes the turn and before its
        text goes out, that it is not past its max age and still valid (``fresh``)."""
        cfg = self.cfg()
        reactions = cfg.expression.reactions
        request = self._described(request, cfg)

        def fresh() -> bool:
            young = self.clock() - request.created_at <= sit.max_age_s
            return young and not self._stop.is_set() and self.valid(request)

        future = self._inflight = self.worker.reaction_turn(
            event_text(request),
            cfg.conversation.language,
            timeout_s=reactions.turn_timeout_s,
            quiet=lambda: room_quiet(self.host, self.clock()),
            profile=request.situation != RETURNING,
            fresh=fresh,
        )
        if self._stop.is_set():  # closed while asking: give it up
            future.cancel()
        try:
            result = self._result(future)
        finally:
            self._inflight = None
        return "no_route" if result is None else str(result)

    def _described(self, request: ReactionRequest, cfg: RoboEmotConfig) -> ReactionRequest:
        """A shown-object request with the surroundings description of its thing, waited for at
        most ``surroundings.shown.wait_s`` from the look (SPEC 8, Surroundings snapshots); as is
        without one."""
        surroundings = self.host.surroundings
        if request.situation != "shown_object" or surroundings is None:
            return request
        left = cfg.surroundings.shown.wait_s - (self.clock() - request.created_at)
        text = surroundings.wait_description(request.token, max(0.0, left))
        if not text:
            return request
        return replace(request, data={**request.data, "description": text})

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
        if self._stop.is_set():
            return "dropped_busy"
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
        return media_on(self.host)

    def resting(self) -> bool:
        """Whether the platform sleeps, the robot rests or the quiet hours run (the rhythm on):
        a reaction never wakes the robot. Never called with the expressor's lock held."""
        return robot_resting(self.host, self.asleep)

    def idle(self) -> bool:
        """Whether the conversation is idle, no confirmation question waits for its answer and
        no announcement is being handled."""
        if not conversation_idle(self.host, self.clock()) or self.worker.confirming():
            return False
        announcer = self.host.announcer
        return announcer is None or not announcer.busy()

    def valid(self, request: ReactionRequest) -> bool:
        """Whether ``request`` still applies (checked at the start, on every wait poll and, by
        the worker, right before its text goes out): a shown thing is still the look it was
        shown in; a "did not catch it" question has no person's turn begun since (the person
        said it again, or something else)."""
        if request.situation == "didnt_catch":
            return self.worker.person_turn_at() <= request.created_at
        if request.situation != "shown_object":
            return True
        with self._lock:
            return request.token == self._shown_key

    def _speaking(self) -> bool:
        """Whether robot speech is queued or playing now."""
        return robot_speaking(self.host, self.clock())

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
        """``future``'s result once done (a failure or a cancel counts None); None when closed
        first."""
        if not self._until(future.done) or future.cancelled():
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
    (a station name, a detector category, an enrolled first name, a wake flag, a local time)."""
    data = request.data
    situation = request.situation
    if situation == "didnt_catch":
        body = DIDNT_CATCH_EVENT
    elif situation == "failed_action":
        body = FAILED_ACTION_EVENT.format(station=data.get("station", ""))
    elif situation == "shown_object":
        category = data.get("category", "")
        thing = SHOWN_THING.format(article=article(category), category=category)
        if data.get("description"):
            thing += SHOWN_DESCRIBED.format(
                quoted=json.dumps(data["description"], ensure_ascii=False)
            )
        body = SHOWN_OBJECT_EVENT.format(thing=thing)
    else:
        template = WAKE_EVENT if data.get("wake") else RETURNING_EVENT
        body = template.format(who=data.get("who") or ANONYMOUS)
        if data.get("time"):
            body += LOCAL_TIME.format(time=data["time"])
    return EVENT_PREFIX + body


def fallback_text(request: ReactionRequest, language: str, cfg: RoboEmotConfig) -> str:
    """The fixed line for ``request`` in ``language`` (SPEC 8, Robot expression), its slots
    filled with the request's data; an empty slot leaves no space before the punctuation."""
    text = expression_fallback(request.situation, language, cfg)
    for slot in SLOTS:
        text = text.replace("{" + slot + "}", request.data.get(slot, ""))
    return SPACE_BEFORE_MARK.sub(r"\1", " ".join(text.split()))
