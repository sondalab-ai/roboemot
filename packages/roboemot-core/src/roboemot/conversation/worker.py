"""The conversation worker (SPEC 8 stage 6, Conversation state; 12 Behaviour and content).

Addressed final utterances in, spoken sentences out. One thread with its own event loop (the
pattern of the persona traits worker) runs one turn at a time: it composes the request
(``system_prompt`` with the persona-shaped register, the speaker and scene summaries under their
caps, the affect state, the session's history), streams the answer through the shared router,
cuts it into sentences and hands each to the speech output as soon as it is complete.

Sessions: the first addressed utterance starts one; it ends ``session_timeout_min`` after the
last utterance or the last time the app reported the person in view (``note_presence``), and the
history of whole turns (RAM only, at most ``history_turns``) is dropped by a timer on the loop,
not on the next utterance. Nothing is persisted.

One behaviour for everyone (owner decision 2026-10-23, SPEC 8, 12): no age or possible-minor
signal changes the prompt, the register, the lines or what is spoken. Replies are cut to the
register's sentence cap.

Realtime speech route (SPEC 8, Realtime speech route): with a ready ``RealtimeRoute`` the turn
goes to the server instead. The transcript is sent as ``input_text``, ``session.update`` carries
parts 1 to 4 (without the mood-tag rule) and the tools, and each ``response.create`` carries parts
1 to 5, the mood's voice instructions and, after a reconnect, the turns the server's session does
not hold (data, not instructions). The server's voice plays through ``SpeechOutput.stream``; tool
calls run through the same executor; the history mirror keeps what was heard and deletes the
server's items of turns beyond ``history_turns``. A failure before any audio played answers the
turn through the chat routes; after audio played the reply is cut like a barge-in. Fixed lines
(the confirmation question and replies, the fallback, the offline cue) use the server's voice
while the route is up and Piper only when it is down. The session end closes the server session.

Privacy (SPEC 12): neither logs nor the bus carry a transcript, prompt or reply, only ids,
lengths and exception types.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import shutdown
from ..bus import Bus
from ..config.schema import RegisterParams, RoboEmotConfig, StateEffects
from ..config.stack import ConfigStack
from ..llm.hub import RouterHub
from ..llm.prompt import (
    CHARS_PER_TOKEN,
    LANGUAGE_NAMES,
    max_reply_sentences,
    realtime_instructions,
    system_prompt,
)
from ..llm.pseudonyms import Pseudonyms
from ..llm.router import (
    ActiveRoute,
    Done,
    NoRouteAvailable,
    Offline,
    TextDelta,
    ToolCall,
    build_messages,
    tool_arguments,
)
from ..persona import apply_to_register, effects
from ..realtime import protocol as rt_protocol
from ..realtime.turn import (
    SKIP_TOOL,
    Reply,
    ResponseRun,
    heard_words,
    line_instructions,
    replay_block,
    skip_tool,
    truncate_played,
    wait_in_order,
)
from ..skills import Skill, SkillContext, SkillRegistry, ToolOutcome, execute
from ..voice.keywords import normalise
from ..voice.listener import Utterance
from ..voice.prosody import Delivery, compose_delivery, resolve_tag
from ..voice.robot_name import RobotName
from ..voice.speech_features import Prosody
from ..voice.tts import Sentence, SentenceSplitter
from .confirm import Pending, PreferenceStore, reply_kind, sanitise_preference
from .lines import pick
from .phase import ConversationPhase
from .transcript import NOT_RECORDED, TranscriptRing

log = logging.getLogger(__name__)

TOPIC = "conversation.turn"
SECONDS_PER_MIN = 60.0
MS_PER_S = 1000.0
JOIN_TIMEOUT_S = 2.0
READY_TIMEOUT_S = 5.0
INTERRUPTED_MARK = " …"  # ends an assistant message the person cut off (history, SPEC 8)
TOO_MANY_ROUNDS = '{"error": "too_many_rounds"}'  # a tool call after the realtime round cap


class Speech(Protocol):
    """What the worker needs of ``SpeechOutput``."""

    def new_utterance(self) -> int: ...
    def say(self, text: str, **kwargs: Any) -> int | None: ...
    def stop(self) -> None: ...
    def first_audio_at(self, utterance: int) -> float | None: ...
    def played(self, job_id: int) -> bool: ...
    def ended_at(self, job_id: int) -> float | None: ...


@dataclass
class _Session:
    last_active: float
    speaker: str | None = None  # registry id of the person it belongs to; None: unknown


@dataclass
class _Turn:
    """State of the answer being prepared."""

    utterance: Utterance
    cfg: RoboEmotConfig
    language: str
    register: RegisterParams
    speech_id: int
    cap: int
    spoken: list[str] = field(default_factory=list)
    queued: list[tuple[int, str]] = field(default_factory=list)  # (speech job id, text)
    exchange: list[dict[str, Any]] = field(default_factory=list)  # tool calls and results
    speaker: str | None = None
    interrupted: bool = False  # a barge-in is under way: nothing more is said (set first)
    stopped: bool = False  # ... and the speech was stopped: history keeps what was heard
    reply_seq: int | None = None  # this reply's entry in the live transcript
    ring_gen: int = NOT_RECORDED  # the transcript generation of the utterance it answers
    speech_stopped: bool = False  # the speech was stopped during the turn (barge-in, stop)
    done: bool = False  # the turn is over: a late skill callback does nothing
    exchange_spoken: int = 0  # queued sentences already in the exchange messages
    cut: bool = False  # stop_talking cut it
    tool_calls: int = 0
    rounds: int = 0
    sentences: int = 0
    outcome: str = "ok"
    route: str | None = None
    tokens: int = 0
    reasoning_chars: int = 0  # the route's non-standard reasoning deltas (a count, never text)
    out_chars: int = 0
    stamps: dict[str, float] = field(default_factory=dict)
    active: ActiveRoute = field(default_factory=ActiveRoute)
    voice: str = "local"  # "realtime": the server's voice speaks this turn
    lines: list[tuple[str, tuple[str, str, str] | None]] = field(
        default_factory=list, repr=False
    )  # fixed lines to say; a question carries the preference (person's data)
    rt_gen: int = 0  # the realtime session generation this turn's items belong to
    rt_items: list[str] = field(default_factory=list)  # server item ids of this turn
    rt_jobs: list[tuple[int, str | None]] = field(default_factory=list)  # (speech job, item)
    rt_partial: bool = False  # a realtime reply was cut part way (history: interrupted)
    rt_tools: bool = False  # a skill ran in this realtime turn: it is never answered again
    rt_round: tuple[list[ToolCall], list[ToolOutcome], int] | None = None  # round in flight
    rt_run: Any = None  # the ResponseRun of that round
    rt_audio: bool = False  # the server's own items of the utterance's audio were answered
    skipped: str | None = None  # the model called ignore_utterance: the route counter to move


@dataclass
class _Tail:
    """The latest turn's history entry, so a barge-in after the stream ended (the speech is
    still playing) can cut it back to what was heard."""

    messages: list[dict[str, Any]]
    queued: list[tuple[int, str]]
    cut: bool = False
    rt: list[tuple[int, str | None]] | None = None  # realtime: (speech job, server item)
    rt_gen: int = 0


class _NoSend:
    """A route stand-in that sends nothing (the server session the items belonged to is gone)."""

    def send_control(self, event: dict[str, Any]) -> bool:
        return False


class ConversationWorker:
    """See the module docstring. Callables are injected so core stays robot-agnostic:

    ``register()`` the persona-shaped active register (None before the first tick),
    ``circadian()`` the circadian state's effects, ``profile(max_chars)`` and
    ``scene(max_chars)`` the summaries (None when empty).
    ``skills`` are the tools offered to the model (SPEC 11) and ``preferences`` where a
    confirmed preference is written (SPEC 6); without them the model gets no tools.
    ``known_names()`` gives the registry display names (every record), which
    ``privacy.pseudonyms_to_cloud`` swaps for aliases on routes that are not local (SPEC 12).
    """

    def __init__(
        self,
        stack: ConfigStack,
        hub: RouterHub,
        speech: Speech,
        *,
        affect: Any,
        responder: Any,
        register: Callable[[], RegisterParams | None],
        circadian: Callable[[], StateEffects],
        profile: Callable[[int], str | None] = lambda cap: None,
        scene: Callable[[int], str | None] = lambda cap: None,
        speaker: Callable[[], str | None] = lambda: None,
        skills: SkillRegistry | None = None,
        preferences: PreferenceStore | None = None,
        bus: Bus | None = None,
        clock: Callable[[], float] = time.monotonic,
        names: RobotName | None = None,
        known_names: Callable[[], Iterable[str]] = lambda: (),
        phase: ConversationPhase | None = None,
        transcript: TranscriptRing | None = None,
        realtime: Any = None,
    ) -> None:
        self.phase = phase if phase is not None else ConversationPhase()
        self.transcript = transcript  # the Live tab's opt-in ring of turns (SPEC 11A, 12)
        self._reply: tuple[int, list[tuple[int, str]]] | None = (
            None  # (ring seq, queued) of the last reply
        )
        self.names = names if names is not None else RobotName()
        self.stack, self.hub, self.speech = stack, hub, speech
        self.affect, self.responder = affect, responder
        self._register, self._circadian = register, circadian
        self._profile, self._scene, self._speaker = profile, scene, speaker
        self.skills, self.preferences = skills, preferences
        self.bus, self.clock = bus, clock
        self._known_names = known_names
        self._pseudonyms = Pseudonyms()
        self._refused_names = 0  # last count of registry names that cannot be aliased
        self._active_turn: _Turn | None = None
        self._inner: asyncio.Future | None = None
        self._confirm: Pending | None = None
        self._tail: _Tail | None = None
        self._barge_at = -math.inf  # when the last barge-in or stop keyword cut the robot off
        self._lock = threading.Lock()
        # (utterance, the live transcript's generation when it was recorded)
        self._pending: collections.deque[tuple[Utterance, int]] = collections.deque()
        self._turns: collections.deque[list[dict[str, Any]]] = collections.deque()
        self._max_turns = 0
        self._session: _Session | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._ready = threading.Event()
        self._closed = False
        self._counts: collections.Counter[str] = collections.Counter()
        self._last: dict[str, Any] | None = None
        self.realtime = realtime  # the realtime speech route (SPEC 8), or None
        self._rt_lock: asyncio.Lock | None = None  # one realtime response at a time
        self._tagged_noted = False
        if realtime is not None:
            realtime.base_instructions = self._base_instructions
        self.apply_config(stack.effective())

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the worker thread and wait briefly for its loop."""
        self._thread = threading.Thread(target=self._run, name="conversation", daemon=True)
        self._thread.start()
        if not self._ready.wait(READY_TIMEOUT_S):
            log.warning("conversation: worker did not start in time")

    def close(self) -> None:
        """Cancel the turn in flight (its stream closes and its tokens are charged), stop the
        thread and drop the session. Close it before the speech output."""
        self._closed = True
        loop, main = self._loop, self._main
        if loop is not None and main is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(main.cancel)
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_TIMEOUT_S)
        self._end_session("closed")
        with self._lock:
            self._pending.clear()

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: the history cap follows the configuration (the rest is read from the stack
        at each turn)."""
        with self._lock:
            self._max_turns = cfg.conversation.history_turns
            dropped = self._cap_history()
        self._delete_items(dropped)

    def _cap_history(self) -> list[list[dict[str, Any]]]:
        """Drop the oldest turns beyond the cap (lock held); returns them."""
        dropped = []
        while len(self._turns) > self._max_turns:
            dropped.append(self._turns.popleft())
        return dropped

    def _delete_items(self, turns: list[list[dict[str, Any]]]) -> None:
        """Delete the realtime server's items of turns that left the history (SPEC 8,
        Conversation state): only those of the server session still open."""
        route = self.realtime
        if route is None:
            return
        for turn in turns:
            mark = turn[0].get("rt") if turn else None
            if mark:
                route.delete_items(*mark)

    # ----------------------------------------------------------------- input

    def submit(self, utterance: Utterance) -> bool:
        """Queue an utterance for an answer; only addressed final ones with text count. While a
        turn runs, at most ``conversation.pending_utterances`` wait and a newer one replaces the
        oldest. Safe from any thread (the listener's). Returns whether it was taken."""
        if self._closed or not (utterance.final and utterance.addressed and utterance.text.strip()):
            self._drop_audio(utterance)  # not answered: the server must not answer it either
            return False
        cfg = self.stack.effective()
        if self.clock() - self._barge_at <= cfg.conversation.stop_drop_window_s and (
            self._only_stop(utterance, cfg)
        ):
            self._counts["stop_consumed"] += 1  # the command that already stopped the robot
            self._drop_audio(utterance)
            return False
        limit = cfg.conversation.pending_utterances
        gen = NOT_RECORDED
        if self.transcript is not None:
            _, gen = self.transcript.begin(
                "user", utterance.text.strip(), utterance.language or cfg.conversation.language
            )
        replaced = []
        with self._lock:
            while len(self._pending) >= limit:
                replaced.append(self._pending.popleft()[0])
                self._counts["replaced"] += 1
            self._pending.append((utterance, gen))
        for old in replaced:
            self._drop_audio(old)  # never answered: the server must not answer it either
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)
        return True

    def _drop_audio(self, utterance: Utterance) -> None:
        """Delete the server's user items of an utterance's audio (the realtime route's audio
        input) that no realtime response answered: otherwise the next ``response.create``
        would answer it (SPEC 8, Realtime speech route)."""
        if utterance.realtime is not None and self.realtime is not None:
            generation, items = utterance.realtime
            self.realtime.delete_items(generation, list(items))

    @staticmethod
    def _only_stop(utterance: Utterance, cfg: RoboEmotConfig) -> bool:
        """Whether the text is nothing but stop phrases of its language ("stop", "basta
        basta")."""
        keywords = cfg.voice.keywords.get(utterance.language)
        phrases = sorted(
            (p.split() for p in (normalise(x) for x in (keywords.stop if keywords else [])) if p),
            key=len,
            reverse=True,
        )
        words = normalise(utterance.text).split()
        while words:
            hit = next((p for p in phrases if words[: len(p)] == p), None)
            if hit is None:
                return False
            words = words[len(hit) :]
        return True

    def note_presence(self, now: float, speaker: str | None = None) -> None:
        """The session's person is still in view (app tick): keeps it alive. ``speaker`` is who
        is in view now (registry id, None when unknown); anyone else does not refresh it."""
        with self._lock:
            if self._session is not None and self._session.speaker == speaker:
                self._session.last_active = now

    # ----------------------------------------------------------------- session

    def in_session(self) -> bool:
        """Whether a conversation session is open (the idle runner waits while it is)."""
        with self._lock:
            return self._session is not None

    def expire(self, now: float) -> bool:
        """End the session when it has been idle for ``session_timeout_min``; returns whether it
        ended. The loop's timer calls it; tests call it with their own clock."""
        timeout = self._timeout_s()
        with self._lock:
            session = self._session
            if session is None or now - session.last_active < timeout:
                return False
        self._end_session("timeout")
        return True

    def _timeout_s(self) -> float:
        return self.stack.effective().conversation.session_timeout_min * SECONDS_PER_MIN

    def _end_session(self, why: str) -> None:
        with self._lock:
            had = self._session is not None
            self._session = None
            self._turns.clear()
            self._confirm = None
            self._tail = None
        if had:
            log.debug("conversation: session ended (%s)", why)
            if self.realtime is not None and why != "closed":
                self.realtime.close_soon("conversation_end")  # the server session ends with it

    def _who(self) -> str | None:
        try:
            return self._speaker()
        except Exception as exc:
            log.warning("conversation: speaker unavailable (%s)", type(exc).__name__)
            return None

    def _touch(self, now: float, speaker: str | None) -> None:
        """Start the session if none is open and stamp the activity. A session belongs to the
        person who opened it: another speaker (a different known person, or an unknown one
        taking over a known person's session, or the reverse) ends it first, so nobody inherits
        someone else's history. Two unknown speakers cannot be told apart and share a session."""
        with self._lock:
            s = self._session
            other = s is not None and s.speaker != speaker
        if other:
            self._end_session("speaker changed")
            self._counts["speaker_changes"] += 1
        with self._lock:
            if self._session is None:
                self._session = _Session(now, speaker)
                self._counts["sessions"] += 1
            else:
                self._session.last_active = now
        self._arm_timer(self._timeout_s())

    def _arm_timer(self, delay_s: float) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = loop.call_later(delay_s, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        now = self.clock()
        if self.expire(now):
            return
        with self._lock:
            session = self._session
        if session is not None:
            self._arm_timer(max(0.0, session.last_active + self._timeout_s() - now))

    # ----------------------------------------------------------------- loop thread

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except asyncio.CancelledError:
            pass
        finally:
            self._ready.set()

    async def _serve(self) -> None:
        self._loop, self._main = asyncio.get_running_loop(), asyncio.current_task()
        self._wake = asyncio.Event()
        self._rt_lock = asyncio.Lock()
        self._ready.set()
        while not self._closed:
            with self._lock:
                item = self._pending.popleft() if self._pending else None
            if item is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            try:
                await self._answer(*item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one bad turn must not end the worker
                self._counts["errors"] += 1
                log.warning("conversation: turn failed (%s)", type(exc).__name__)

    # ----------------------------------------------------------------- one turn

    def _register_for(self, cfg: RoboEmotConfig) -> RegisterParams:
        """The persona-shaped active register; the configured register, persona-shaped, before
        the first tick or when it cannot be read."""
        try:
            active = self._register()
        except Exception as exc:
            log.warning("conversation: register unavailable (%s)", type(exc).__name__)
            active = None
        if active is None:
            return apply_to_register(cfg.register.params, effects(cfg.persona))
        return active

    def _summary(self, source: Callable[[int], str | None], cap: int) -> str | None:
        try:
            return source(cap)
        except Exception as exc:  # a missing record must not stop the answer
            log.warning("conversation: summary unavailable (%s)", type(exc).__name__)
            return None

    def _history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {**m, "content": m["content"] + INTERRUPTED_MARK} if m.get("interrupted") else m
                for turn in self._turns
                for m in turn
            ]

    # ----------------------------------------------------------------- barge-in

    def mark_interrupting(self) -> None:
        """First step of a barge-in, before the speech is stopped: from now on the turn in
        flight says nothing more, so a sentence it queues between the stop and ``interrupt``
        cannot play. Any thread."""
        with self._lock:
            if self._active_turn is not None:
                self._active_turn.interrupted = True

    def interrupt(self) -> bool:
        """Barge-in or the stop keyword (SPEC 8 stage 8), from any thread, after the speech was
        stopped: cancel the answer in flight (its stream is closed and charged), or, when the
        stream already ended and only the speech was running, cut the turn's history entry back
        to the sentences that were heard. A question cut off before it finished is no longer
        pending. Returns whether there was anything to cut."""
        with self._lock:
            self._barge_at = self.clock()
            turn = self._active_turn
            if turn is not None:
                turn.interrupted = turn.stopped = turn.speech_stopped = True
            self._drop_unheard_question()
        if turn is None:
            self._settle_reply()
            return self._trim_tail()
        self._counts["interrupts"] += 1
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_inner)
        return True

    def _settle_reply(self) -> None:
        """A barge-in after the stream ended: the last reply in the live transcript becomes the
        sentences that played, marked interrupted (nothing played removes it)."""
        with self._lock:
            reply, self._reply = self._reply, None
        if reply is None or self.transcript is None:
            return
        seq, queued = reply
        heard = [text for job, text in queued if self.speech.played(job)]
        if len(heard) < len(queued):
            self.transcript.settle(seq, " ".join(heard), interrupted=True)

    def _drop_unheard_question(self) -> None:
        """Forget a pending confirmation unless its question finished playing before now
        (called with the lock held). A cut-off question was not heard to the end."""
        pending = self._confirm
        if pending is None:
            return
        ended = self.speech.ended_at(pending.job)
        if not self.speech.played(pending.job) or ended is None or ended > self.clock():
            self._confirm = None
            self._counts["confirm_cut"] += 1

    def _cancel_inner(self) -> None:
        inner = self._inner
        if inner is not None and not inner.done():
            inner.cancel()

    def _trim_tail(self) -> bool:
        with self._lock:
            tail = self._tail
            if tail is None or tail.cut or not any(tail.messages is m for m in self._turns):
                return False
            tail.cut = True
            if tail.rt is not None:
                return self._trim_realtime_tail(tail)
            played = [text for job, text in tail.queued if self.speech.played(job)]
            if len(played) == len(tail.queued):
                return False
            if played:
                tail.messages[-1].update(content=" ".join(played), interrupted=True)
            else:
                tail.messages.pop()
            self._counts["history_trimmed"] += 1
        return True

    def _trim_realtime_tail(self, tail: _Tail) -> bool:
        """``_trim_tail`` for a realtime reply still playing after its response ended: the
        server's items are truncated at what played (when its session is still the same) and
        the history keeps the words heard (lock held)."""
        route = self.realtime
        assert tail.rt is not None
        same = route is not None and route.generation == tail.rt_gen
        heard = truncate_played(route if same else _NoSend(), self.speech, tail.rt)
        kept, partial = [], False
        for job, text in tail.queued:
            fraction = heard.get(job)
            if fraction is None:
                partial = True
                continue
            if fraction < 1.0:
                text, partial = heard_words(text, fraction), True
            if text:
                kept.append(text)
        if not partial:
            return False
        if kept:
            tail.messages[-1].update(content=" ".join(kept), interrupted=True)
        else:
            tail.messages.pop()
        self._counts["history_trimmed"] += 1
        return True

    async def _answer(self, utterance: Utterance, ring_gen: int = NOT_RECORDED) -> None:
        now = self.clock()
        cfg = self.stack.effective()
        self.apply_config(cfg)
        router = self.hub.sync(cfg)
        self.expire(now)
        speaker = self._who()
        self._touch(now, speaker)
        register = self._register_for(cfg)
        turn = _Turn(
            utterance,
            cfg,
            utterance.language or cfg.conversation.language,
            register,
            self.speech.new_utterance(),
            max_reply_sentences(cfg, register),
            speaker=speaker,
        )
        turn.ring_gen = ring_gen
        turn.stamps["heard"] = utterance.last_speech_stamp
        self.affect.new_turn()
        with self._lock:
            self._active_turn = turn
        try:
            if await self._realtime_ready(turn):
                turn.voice = "realtime"
            answered = self._confirmation_reply(turn, now)
            if not answered and turn.voice == "realtime":
                answered = await self._realtime_turn(turn)
            if not answered:
                turn.voice = "local"
                await self._converse(turn, router)
            await self._speak_lines(turn)
        finally:
            self._inner = None
            self.phase.end_thinking()
            if not turn.rt_audio:
                self._drop_audio(utterance)
        self._finish(turn)

    # ----------------------------------------------------------------- realtime route

    async def _realtime_ready(self, turn: _Turn) -> bool:
        """Whether this turn goes to the realtime route: it is configured, may answer and is
        connected (a connect runs off the loop, within ``connect_timeout_s``). A route in
        realtime mode that is not ready is a fallback at start (the local pipeline answers)."""
        route = self.realtime
        if route is None or not turn.cfg.voice.realtime_route.enabled:
            return False
        if turn.utterance.local_only:  # its audio went, no transcript came: the chat routes
            return False
        loop = asyncio.get_running_loop()
        try:
            ready = await loop.run_in_executor(None, route.ready, turn.cfg)
        except Exception as exc:
            log.warning("conversation: realtime route: %s", type(exc).__name__)
            ready = False
        if not ready:
            route.count("fallbacks_start")
        elif turn.cfg.voice.mood_source == "llm_tagged" and not self._tagged_noted:
            self._tagged_noted = True
            log.info("conversation: realtime route: mood_source llm_tagged acts as auto")
        return bool(ready) and not turn.interrupted

    def _base_instructions(self) -> str:
        """Parts 1 to 4 for the realtime route's ``session.update`` (no mood-tag rule)."""
        cfg = self.stack.effective()
        return realtime_instructions(
            cfg, self._register_for(cfg), robot_name=self._prompt_name(cfg)
        )

    def _delivery(self, turn: _Turn) -> Delivery:
        """The delivery of a realtime reply: the volume (applied as gain) and the mood's
        voice instructions; rate, pitch and pauses are lost (SPEC 8)."""
        cfg = turn.cfg
        state = self.affect.state(self.clock())
        return compose_delivery(
            cfg.voice,
            register=turn.register,
            circadian=self._circadian(),
            mood=state.label,
            intensity=state.intensity,
            persona=effects(cfg.persona),
            seed=turn.speech_id,
        )

    def _rt_instructions(self, turn: _Turn, delivery: Delivery, *, line: bool = False) -> str:
        """``response.create`` instructions: parts 1 to 5, the turns the server's session does
        not hold (after a reconnect), and the mood's voice instructions. A fixed ``line`` leaves
        out the skip rule (the server must say it, not judge it)."""
        cfg, conv = turn.cfg, turn.cfg.conversation
        state = self.affect.state(self.clock())
        system = build_messages(
            system=realtime_instructions(
                cfg,
                turn.register,
                language=turn.language,
                robot_name=self._prompt_name(cfg),
                skip_rule=not line,
            ),
            profile_summary=self._summary(self._profile, conv.profile_summary_max_chars),
            scene_summary=self._summary(self._scene, conv.scene_summary_max_chars),
            history=[],
            user_text="",
            affect=f"{state.label}, intensity {state.intensity:.1f}",
            summary_max_chars=(conv.profile_summary_max_chars, conv.scene_summary_max_chars),
        )[0]["content"]
        generation = self.realtime.generation
        with self._lock:
            missing = [t for t in self._turns if not t or t[0].get("rt", (None,))[0] != generation]
        mood = delivery.instructions if cfg.voice.realtime_route.mood_instructions else ""
        return "\n".join(p for p in (system, replay_block(missing), mood) if p)

    async def _realtime_turn(self, turn: _Turn) -> bool:
        """Answer through the realtime route; False when nothing played and the session was
        lost (or refused the response): the chat routes then answer this turn."""
        inner = self._inner = asyncio.ensure_future(self._realtime_rounds(turn))
        if turn.interrupted:
            inner.cancel()
        try:
            return bool(await inner)
        except asyncio.CancelledError:
            if not ((turn.interrupted or turn.stopped) and inner.cancelled()):
                inner.cancel()
                raise
            turn.outcome = "interrupted"
            return True
        except Exception as exc:
            log.warning("conversation: realtime turn failed (%s)", type(exc).__name__)
            # a response in flight was already abandoned (cancelled) by its run; the failing
            # round's audio is stopped with the rest
            run = turn.rt_run
            if run is not None:
                turn.rt_jobs += [j for j in run.reply.jobs if j not in turn.rt_jobs]
            answered = self._realtime_lost(turn)
            turn.outcome = "error"
            return answered

    async def _realtime_rounds(self, turn: _Turn) -> bool:
        """The transcript as ``input_text``, then one response per round: tool calls run as
        they arrive, their results go back as ``function_call_output`` and one more response
        is asked, until the reply has no call, a skill ends the turn or
        ``conversation.max_tool_rounds`` rounds ran (the next request has ``tool_choice:
        none``; a call after that gets an error result and ends the turn)."""
        route = self.realtime
        assert self._rt_lock is not None
        async with self._rt_lock:
            # no other utterance's audio lands between this turn's response.create calls
            route.hold_input()
            try:
                return await self._realtime_rounds_held(turn)
            finally:
                route.release_input()

    async def _realtime_rounds_held(self, turn: _Turn) -> bool:
        """``_realtime_rounds`` with ``_rt_lock`` held and the audio input held."""
        route, conv = self.realtime, turn.cfg.conversation
        # nothing goes to a session an abandoned request still holds: wait it out first
        # (an expiry resets it), then make sure a session is up for this turn
        await wait_in_order(route, self.clock)
        loop = asyncio.get_running_loop()
        if not route.up() and not await loop.run_in_executor(None, route.ready, turn.cfg):
            return False
        turn.rt_gen = route.generation
        turn.route = route.settings(turn.cfg).route
        offered = self._offer(turn)
        choices = SkillRegistry.snapshot_choices(offered, conv.tool_enum_max_items)
        tools = SkillRegistry.tools(
            offered,
            enum_max=conv.tool_enum_max_items,
            max_chars=conv.tool_arg_max_chars,
            choices=choices,
        )
        skip = turn.cfg.voice.realtime_route.skip_tool
        functions = rt_protocol.function_tools(tools) if tools else []
        functions += [skip_tool()] if skip else []
        sent = route.send(route.session_update(turn.cfg, self._base_instructions(), functions))
        if not sent:
            return False
        audio = turn.utterance.realtime
        if audio is not None and audio[0] == route.generation:
            # the server heard the audio: its own user items hold the words, answered by
            # the response below (no input_text: the words would be there twice)
            turn.rt_items += list(audio[1])
            turn.rt_audio = True
        else:  # text input, or the session that heard the audio is gone: the words as text
            item = route.new_item_id()
            if not route.send(rt_protocol.text_item(turn.utterance.text, item)):
                return False
            turn.rt_items.append(item)
        route.count("utterances_sent")
        turn.stamps["request"] = self.clock()
        self.phase.begin_thinking()
        self.phase.set_route(turn.route)
        delivery = self._delivery(turn)
        while True:
            in_cap = turn.rounds < conv.max_tool_rounds and bool(offered)
            calls: list[ToolCall] = []
            results: list[ToolOutcome] = []
            on_call = self._rt_caller(turn, choices, in_cap, calls, results)
            instructions = self._rt_instructions(turn, delivery)
            run = ResponseRun(
                route,
                self.speech,
                utterance=turn.speech_id,
                mood=delivery.mood,
                volume_db=delivery.volume_db,
                language=turn.language,
                clock=self.clock,
                on_call=on_call,
                input_chars=len(turn.utterance.text),
                generation=turn.rt_gen,  # the session that holds the turn's words
                mute_on=SKIP_TOOL if skip else None,
            )
            round_from = len(turn.spoken)
            turn.rt_round = (calls, results, round_from)
            turn.rt_run = run
            try:
                ask = in_cap or (skip and not turn.rounds)  # the skip tool: first round only
                choice = "auto" if ask else "none" if offered or skip else None
                reply = await run.run(instructions, choice)
            except asyncio.CancelledError:
                self._absorb(turn, run.reply, instructions)
                if turn.skipped:
                    self._skip(turn)
                    raise
                self._cut_realtime(turn)
                self._keep_round(turn)  # a skill that ran stays in the history
                raise
            self._absorb(turn, reply, instructions)
            if turn.skipped:  # before a loss: the chat routes must not answer it either
                return self._skip(turn)
            if reply.closed or reply.status == "error":
                return self._realtime_lost(turn)
            if turn.cut:
                self._cut_realtime(turn)
            ended = turn.cut or any(r.end_turn for r in results)
            for call, result in zip(calls, results, strict=True):
                out_id = route.new_item_id()
                if not route.send(rt_protocol.function_output(call.id, result.content, out_id)):
                    if ended:
                        break  # no response follows: nothing is lost
                    return self._realtime_lost(turn)  # a skill ran: B1 (cut, Piper line)
                turn.rt_items.append(out_id)
            if ended or not calls:
                if turn.rounds and not (ended or turn.spoken):
                    turn.outcome = "no_text"
                    self._say_fallback(turn)
                return True
            turn.rounds += 1
            said = " ".join(turn.spoken[round_from:]) or None
            turn.exchange += self._exchange(calls, results, turn.rounds, said)
            turn.exchange_spoken = len(turn.queued)
            turn.rt_round = None

    def _realtime_lost(self, turn: _Turn) -> bool:
        """The session was lost (or refused the response, or the turn failed) during the turn.
        Audio that arrived but did not play yet is dropped. If nothing played and no skill ran,
        False: the chat routes answer the turn. Otherwise it is never answered again (a skill
        would run twice): the reply is cut like a barge-in, the history is marked interrupted
        and, when a skill ran, the fallback line is said with Piper."""
        route = self.realtime
        if turn.rt_jobs:
            self.speech.stop()  # first: whether anything played is then settled
        played = any(self.speech.played(job) for job, _ in turn.rt_jobs)
        if not played:
            turn.spoken.clear()
            turn.queued.clear()
            turn.rt_jobs.clear()
            turn.sentences = turn.exchange_spoken = 0
            if self.transcript is not None and turn.reply_seq is not None:
                self.transcript.settle(turn.reply_seq, "", interrupted=False)
                turn.reply_seq = None
            if not turn.rt_tools:
                route.count("fallbacks_mid_turn")
                # the server must not keep a question the chat routes answer: its items go,
                # and the turn (no server items) is replayed like a local one
                route.delete_items(turn.rt_gen, turn.rt_items)
                turn.rt_items.clear()
                return False
        route.count("cut_mid_turn")
        turn.outcome = "realtime_lost"
        self._keep_round(turn)
        if played:
            turn.speech_stopped = turn.stopped = True
            self._cut_realtime(turn)
        else:
            turn.rt_partial = True  # the fallback line below closes an unfinished turn
        if turn.rt_tools:
            turn.voice = "local"  # the route is down: Piper (owner decision 4)
            self._say_fallback(turn)
        return True

    def _keep_round(self, turn: _Turn) -> None:
        """The tool calls of the round the loss interrupted go into the history."""
        if turn.rt_round is None or not turn.rt_round[0]:
            return
        calls, results, _ = turn.rt_round
        n = min(len(calls), len(results))
        if n:
            turn.exchange += self._exchange(calls[:n], results[:n], turn.rounds + 1, None)
        turn.rt_round = None

    def _rt_caller(
        self,
        turn: _Turn,
        choices: dict[str, Any],
        in_cap: bool,
        calls: list[ToolCall],
        results: list[ToolOutcome],
    ) -> Callable[[rt_protocol.FunctionCall], Awaitable[bool]]:
        """The tool-call handler of one realtime round: runs the call through the executor
        (the per-round cap applies) and says whether it ends the turn. After the round cap a
        call gets an error result and ends the turn."""
        cap = turn.cfg.conversation.max_tool_calls_per_round

        async def on_call(ev: rt_protocol.FunctionCall) -> bool:
            if turn.skipped or (ev.name == SKIP_TOOL and turn.cfg.voice.realtime_route.skip_tool):
                self._skip_call(turn)  # nothing runs after a skip, in this response either
                return True
            call = ToolCall(ev.call_id, ev.name, tool_arguments(ev.arguments))
            calls.append(call)
            if not in_cap:
                self._counts["tool_errors"] += 1
                results.append(ToolOutcome(TOO_MANY_ROUNDS, False, True, ev.name, "rounds"))
                return True
            turn.rt_tools = True
            results.append(await self._call(turn, call, choices, len(calls) > cap))
            self.realtime.count("tool_calls")
            return turn.cut or results[-1].end_turn

        return on_call

    def _skip_call(self, turn: _Turn) -> None:
        """``ignore_utterance`` (SPEC 8, skip tool): the response plays nothing more (muted, then
        the speech is stopped like a barge-in); ``skipped_late`` once any of its audio played,
        else ``skipped``. Its items go when the response ends (``_skip``)."""
        if turn.skipped:
            return
        run = turn.rt_run
        if run is not None:
            run.mute()
        jobs = turn.rt_jobs + (run.reply.jobs if run is not None else [])
        if jobs:
            self.speech.stop()  # first: whether anything played is then settled
        late = any(self.speech.played(job) for job, _ in jobs)
        turn.skipped = "skipped_late" if late else "skipped"

    def _skip(self, turn: _Turn) -> bool:
        """End a skipped turn: no tool result and no further response (the call item is deleted
        with the rest), the user's and the response's items deleted on the server, nothing kept
        in the history mirror or the live transcript (it is not a turn of ``history_turns``)."""
        route = self.realtime
        route.delete_items(turn.rt_gen, turn.rt_items)
        turn.rt_items.clear()
        turn.spoken.clear()
        turn.queued.clear()
        turn.rt_jobs.clear()
        turn.exchange.clear()
        turn.lines.clear()
        turn.rt_round = None
        turn.sentences = turn.exchange_spoken = 0
        if self.transcript is not None and turn.reply_seq is not None:
            self.transcript.settle(turn.reply_seq, "", interrupted=False)
            turn.reply_seq = None
        route.count(turn.skipped or "skipped")
        turn.outcome = "skipped"
        return True

    def _absorb(self, turn: _Turn, reply: Reply, instructions: str) -> None:
        """Book a response into the turn: its audio items (history, live transcript), its
        server items, the latency and the budget (usage, or the chars/4 estimate)."""
        route = self.realtime
        for job, item in reply.jobs:
            turn.rt_jobs.append((job, item))
            text = reply.texts.get(item, "").strip()
            if text:
                turn.spoken.append(text)
                turn.queued.append((job, text))
                self._note_reply(turn, text)
            turn.sentences += 1
        turn.rt_items += reply.item_ids()
        if reply.first_audio is not None and "token" not in turn.stamps:
            turn.stamps["token"] = turn.stamps.setdefault("say", reply.first_audio)
            heard = turn.utterance.last_speech_stamp
            route.note_first_audio((reply.first_audio - heard) * MS_PER_S)
        if reply.status is None and reply.closed is None and not reply.cancelled:
            return
        text = reply.text()
        turn.out_chars += len(text)
        if reply.tokens is not None:
            tokens = reply.tokens
        else:
            chars = len(instructions) + len(turn.utterance.text) + len(text)
            tokens = chars // CHARS_PER_TOKEN
        turn.tokens += tokens  # the report; the route charges every response once itself

    def _cut_realtime(self, turn: _Turn) -> None:
        """After a stop: truncate the server's audio items at what played and keep, in the
        history and the live transcript, the words heard (an estimate by played time)."""
        route = self.realtime
        heard = truncate_played(route, self.speech, turn.rt_jobs)
        for i, (job, text) in enumerate(turn.queued):
            fraction = heard.get(job)
            if fraction is not None and fraction < 1.0:
                cut = heard_words(text, fraction)
                turn.queued[i] = (job, cut)
                if i < len(turn.spoken) and turn.spoken[i] == text:
                    turn.spoken[i] = cut
                turn.rt_partial = True
        turn.speech_stopped = True
        route.count("cuts")

    async def _speak_lines(self, turn: _Turn) -> None:
        """Say the fixed lines the turn left for the server's voice (the confirmation question
        and replies, the fallback): one response each while the route is up, Piper once it is
        down. A question that played leaves the confirmation pending."""
        while turn.lines and not turn.interrupted:
            text, ask = turn.lines.pop(0)
            job = None
            if turn.voice == "realtime" and self.realtime.up():
                inner = self._inner = asyncio.ensure_future(self._rt_line(turn, text))
                try:
                    job = await inner
                except asyncio.CancelledError:
                    if not (turn.interrupted and inner.cancelled()):
                        raise
                    return
            if job is None:
                if turn.voice == "realtime" and self.realtime.up() and ask is None:
                    continue  # the server spoke nothing: the line is not said twice
                # (a confirmation question is retried with Piper: Pending needs its audio)
                before = len(turn.queued)
                mood, intensity = self._mood(turn.cfg, None)
                self._say_local(turn, text, mood, intensity)
                job = turn.queued[-1][0] if len(turn.queued) > before else None
            if ask is not None and job is not None:
                with self._lock:
                    if not (turn.done or turn.interrupted):
                        expires = self.clock() + turn.cfg.conversation.confirm_timeout_s
                        self._confirm = Pending(*ask, expires, job)

    async def _rt_line(self, turn: _Turn, text: str) -> int | None:
        """One fixed line in the server's voice; returns its first speech job (None when no
        audio came)."""
        route = self.realtime
        assert self._rt_lock is not None
        async with self._rt_lock:
            if not route.up():
                return None  # went down (or spent its budget) while the line waited
            delivery = self._delivery(turn)
            language = LANGUAGE_NAMES.get(turn.language, turn.language)
            instructions = line_instructions(
                self._rt_instructions(turn, delivery, line=True), text, language
            )
            run = ResponseRun(
                route,
                self.speech,
                utterance=turn.speech_id,
                mood=delivery.mood,
                volume_db=delivery.volume_db,
                language=turn.language,
                clock=self.clock,
            )
            try:
                reply = await run.run(instructions, "none")
            except asyncio.CancelledError:
                self._absorb(turn, run.reply, instructions)
                self._cut_realtime(turn)
                raise
            self._absorb(turn, reply, instructions)
            route.count("lines")
            return reply.jobs[0][0] if reply.jobs else None

    def say_line(self, text: str, language: str) -> bool:
        """A fixed line from outside a turn (the offline cue) in the server's voice while the
        route is up; False when it is not (the caller then says it with Piper). Any thread."""
        route, loop = self.realtime, self._loop
        if route is None or loop is None or self._closed or not route.up():
            return False
        asyncio.run_coroutine_threadsafe(self._cue_line(text, language), loop)
        return True

    async def _cue_line(self, text: str, language: str) -> None:
        cfg, now = self.stack.effective(), self.clock()
        quiet = Prosody(0.0, None, None, 0.0)
        utterance = Utterance("", language, now, now, now, False, False, frozenset(), quiet)
        turn = _Turn(
            utterance,
            cfg,
            language,
            self._register_for(cfg),
            self.speech.new_utterance(),
            0,
            voice="realtime",
        )
        try:
            job = await self._rt_line(turn, text)
        except Exception as exc:
            log.warning("conversation: realtime line failed (%s)", type(exc).__name__)
            job = None
        if job is None and not self.realtime.up():  # the route went down: Piper says it
            mood, intensity = self._mood(cfg, None)
            self._say_local(turn, text, mood, intensity)

    async def _converse(self, turn: _Turn, router: Any) -> None:
        """Request, stream, run tool calls, repeat (see ``_rounds``); the errors that end a turn
        become the spoken fallbacks. The rounds run as an inner task, so ``interrupt`` can
        cancel the answer without touching the worker's own shutdown."""
        cfg, conv = turn.cfg, turn.cfg.conversation
        state = self.affect.state(self.clock())
        try:
            protect = self._learn_names(turn)  # the summary caps must not cut a name
        except Exception as exc:  # without the map a cap could cut inside a name: no request
            turn.outcome = "error"
            log.warning("conversation: registry names unavailable (%s)", type(exc).__name__)
            self._say_fallback(turn)
            return
        messages = build_messages(
            system=system_prompt(
                cfg,
                turn.register,
                language=turn.language,
                robot_name=self._prompt_name(cfg),
            ),
            profile_summary=self._summary(self._profile, conv.profile_summary_max_chars),
            scene_summary=self._summary(self._scene, conv.scene_summary_max_chars),
            history=self._history(),
            user_text=turn.utterance.text,
            affect=f"{state.label}, intensity {state.intensity:.1f}",
            summary_max_chars=(conv.profile_summary_max_chars, conv.scene_summary_max_chars),
            protect=protect,
        )
        splitter = SentenceSplitter(cfg.voice.sentence_max_chars, strip_tags=True)
        turn.stamps["request"] = self.clock()
        self.phase.begin_thinking()
        inner = self._inner = asyncio.ensure_future(self._rounds(turn, router, messages, splitter))
        if turn.interrupted:  # a barge-in landed before the task existed
            inner.cancel()
        try:
            await inner
        except Offline:
            turn.outcome = "offline"
            self.responder.speech_while_offline(self.clock(), turn.language)
        except NoRouteAvailable:
            turn.outcome = "no_route"
            self._say_fallback(turn)
        except asyncio.CancelledError:
            if not ((turn.interrupted or turn.stopped) and inner.cancelled()):
                inner.cancel()
                raise
            turn.outcome = "interrupted"
        except Exception as exc:
            turn.outcome = "error"
            log.warning("conversation: stream failed (%s)", type(exc).__name__)
            if not turn.spoken:
                self._say_fallback(turn)

    async def _rounds(
        self, turn: _Turn, router: Any, messages: list[dict[str, Any]], splitter: SentenceSplitter
    ) -> None:
        """One request per round: the text is spoken as it streams, each tool call runs as soon
        as it is complete, and a round that called tools is followed by another request with
        the results, until the model answers in text, a skill ends the turn, or
        ``conversation.max_tool_rounds`` rounds ran (the next request then has no tools)."""
        conv = turn.cfg.conversation
        cap = conv.max_tool_calls_per_round
        while True:
            offered = self._offer(turn) if turn.rounds < conv.max_tool_rounds else []
            choices = SkillRegistry.snapshot_choices(offered, conv.tool_enum_max_items)
            tools = (
                SkillRegistry.tools(
                    offered,
                    enum_max=conv.tool_enum_max_items,
                    max_chars=conv.tool_arg_max_chars,
                    choices=choices,
                )
                or None
            )
            round_from = len(turn.spoken)
            pseudonyms = self._learn_names(turn)
            active = turn.active = ActiveRoute()
            stream = router.stream(messages, tools, active=active, pseudonyms=pseudonyms)
            calls: list[ToolCall] = []
            results: list[ToolOutcome] = []
            cut = False
            async with contextlib.aclosing(stream):
                async for ev in stream:
                    if active.route is not None:
                        self.phase.set_route(active.route.name)
                    if isinstance(ev, TextDelta):
                        turn.stamps.setdefault("token", self.clock())
                        turn.out_chars += len(ev.text)
                        cut = not self._speak_all(turn, splitter.feed(ev.text))
                    elif isinstance(ev, ToolCall) and offered:  # (none offered: ignored)
                        # what the model said before the call is said before the skill acts
                        cut = not self._speak_all(turn, splitter.flush())
                        if not cut:
                            calls.append(ev)
                            results.append(await self._call(turn, ev, choices, len(calls) > cap))
                            cut = turn.cut or results[-1].end_turn
                    elif isinstance(ev, Done):
                        turn.route, turn.tokens = ev.route, turn.tokens + ev.tokens
                        turn.reasoning_chars += ev.reasoning_chars
                    if cut:
                        break
            if not cut:
                self._speak_all(turn, splitter.flush())
            ended = any(r.end_turn for r in results)
            if cut or ended or not calls:
                if turn.rounds and not (cut or ended or turn.spoken):
                    turn.outcome = "no_text"
                    self._say_fallback(turn)
                return
            turn.rounds += 1
            said = " ".join(turn.spoken[round_from:]) or None
            turn.exchange += self._exchange(calls, results, turn.rounds, said)
            turn.exchange_spoken = len(turn.queued)
            messages += turn.exchange[-(len(calls) + 1) :]

    def _learn_names(self, turn: _Turn) -> Pseudonyms | None:
        """With ``privacy.pseudonyms_to_cloud``: bring the map in line with the registry as it is
        now (new names get aliases, forgotten ones are dropped) and return it for the router,
        which applies it to routes that are not local. Read before every request. If the names
        cannot be read the exception ends the turn: sending without the map would put real names
        on the cloud route. None when the option is off. A change in the number of registry names
        that cannot be aliased (they reach a cloud route as typed) is logged for the admin, as a
        count only."""
        cfg = turn.cfg
        if not cfg.privacy.pseudonyms_to_cloud:
            return None
        self._pseudonyms.set_structural_keys(
            self.skills.structural_keys() if self.skills is not None else ()
        )
        self._pseudonyms.sync(
            self._known_names(),
            cfg.privacy.pseudonym_templates,
            turn.language,
            cfg.conversation.language,
        )
        if self._pseudonyms.refused != self._refused_names:
            self._refused_names = self._pseudonyms.refused
            log.warning(
                "conversation: %d registry names cannot be aliased for cloud routes (no letter, "
                "a template word or a request keyword); rename them to keep them private",
                self._refused_names,
            )
        return self._pseudonyms

    def _offer(self, turn: _Turn) -> list[Skill]:
        """The skills the model may call in this request (visible and listed)."""
        if self.skills is None:
            return []
        return self.skills.offered(turn.cfg.conversation.llm_skills)

    @staticmethod
    def _exchange(
        calls: list[ToolCall], results: list[ToolOutcome], round_no: int, said: str | None
    ) -> list[dict[str, Any]]:
        """The assistant message carrying the calls (and what was spoken in the round, as
        ``content``) and one ``tool`` message per result. Arguments that are not an object, or
        broken, are not sent back (they are the model's raw text)."""
        ids = [c.id or f"call_{round_no}_{i}" for i, c in enumerate(calls)]
        assistant = {
            "role": "assistant",
            "content": said,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": (
                            json.dumps(c.arguments, ensure_ascii=False)
                            if isinstance(c.arguments, dict) and "_raw" not in c.arguments
                            else "{}"
                        ),
                    },
                }
                for call_id, c in zip(ids, calls, strict=True)
            ],
        }
        tool = [
            {"role": "tool", "tool_call_id": call_id, "content": r.content}
            for call_id, r in zip(ids, results, strict=True)
        ]
        return [assistant, *tool]

    async def _call(
        self, turn: _Turn, call: ToolCall, choices: dict[str, Any], over_cap: bool
    ) -> ToolOutcome:
        conv = turn.cfg.conversation
        if over_cap:
            self._counts["tool_errors"] += 1
            return ToolOutcome('{"error": "too_many_calls"}', False, False, call.name, "too_many")
        ctx = SkillContext(
            turn.speaker,
            turn.language,
            self.clock(),
            stop_speech=lambda: self._stop_talking(turn),
            ask_confirmation=lambda key, value: self._ask(turn, key, value),
        )
        outcome = await execute(
            self.skills or SkillRegistry(),
            set(choices),
            call.name,
            call.arguments,
            ctx,
            timeout_s=conv.tool_timeout_s,
            max_chars=conv.tool_arg_max_chars,
            enum_max=conv.tool_enum_max_items,
            choices=choices,
        )
        turn.tool_calls += 1
        self._counts["tool_calls" if outcome.ok else "tool_errors"] += 1
        return outcome

    def _stop_talking(self, turn: _Turn) -> None:
        """``stop_talking``: stop the speech, drop the rest of the answer."""
        if turn.done or turn.interrupted:
            return
        self.speech.stop()
        turn.cut = turn.speech_stopped = True
        turn.outcome = "stopped"

    # ----------------------------------------------------------------- confirmation

    def _ask(self, turn: _Turn, key: Any, value: Any) -> str:
        """``remember_preference``: speak the question and leave a pending request (SPEC 8, 12).
        Returns ``asked`` or why not. Nothing is stored until the same person's next utterance
        says yes."""
        conv = turn.cfg.conversation
        person, store = turn.speaker, self.preferences
        if turn.done or turn.interrupted:
            return "cancelled"
        if person is None or store is None:
            return "not_allowed"
        pair = sanitise_preference(
            key, value, conv.preference_key_max_chars, conv.preference_value_max_chars
        )
        if pair is None:
            return "invalid"
        key, value = pair
        template = pick(conv.confirm_ask_lines, turn.language, conv.language)
        if not store.allowed(person, key) or not template:
            return "not_allowed"
        text = template.replace("{key}", key).replace("{value}", value)
        if turn.voice == "realtime":  # asked in the server's voice once the response is over
            turn.lines.append((text, (person, key, value)))
            self._counts["confirm_asked"] += 1
            return "asked"
        mood, intensity = self._mood(turn.cfg, None)
        before = len(turn.queued)
        self._say(turn, text, mood, intensity)
        if len(turn.queued) == before:
            return "unavailable"
        with self._lock:
            if turn.done or turn.interrupted:
                return "cancelled"
            self._confirm = Pending(
                person, key, value, self.clock() + conv.confirm_timeout_s, turn.queued[-1][0]
            )
        self._counts["confirm_asked"] += 1
        return "asked"

    def _confirmation_reply(self, turn: _Turn, now: float) -> bool:
        """If a question is pending, the next utterance is its answer, whatever it says: it
        counts only from the same person (the attended face, until speaker ID exists), in time,
        spoken after the question finished playing, as a short yes (stored) or a no
        (declined); anything else drops the question and the turn goes on as usual. An
        utterance that started before the question ended (queued behind it, or a voice over
        it) is not the answer and leaves the question pending. True when the turn is done."""
        conv = turn.cfg.conversation
        with self._lock:
            pending = self._confirm
            if pending is None:
                return False
            ended = self.speech.ended_at(pending.job)
            if not self.speech.played(pending.job):
                self._confirm = None
                self._counts["confirm_cut"] += 1
                return False
            if ended is None or turn.utterance.start_stamp <= ended:
                self._counts["confirm_early"] += 1
                return False
            self._confirm = None
        if now > pending.expires:
            self._counts["confirm_expired"] += 1
            return False
        if turn.speaker is None or turn.speaker != pending.person_id:
            self._counts["confirm_other"] += 1
            return False
        kind = reply_kind(
            turn.utterance.text,
            turn.language,
            conv.confirm_yes,
            conv.confirm_no,
            conv.confirm_fillers,
            conv.confirm_max_words,
        )
        if kind == "other":
            self._counts["confirm_ignored"] += 1
            return False
        stored = (
            kind == "yes"
            and self.preferences is not None
            and self.preferences.store(pending.person_id, pending.key, pending.value)
        )
        if stored:
            lines = conv.confirm_saved_lines
        elif kind == "yes":
            lines = conv.confirm_failed_lines  # said yes, but it could not be kept: say so
        else:
            lines = conv.confirm_declined_lines
        mood, intensity = self._mood(turn.cfg, None)
        self._say(turn, pick(lines, turn.language, conv.language) or "", mood, intensity)
        turn.outcome = "confirmed" if stored else "declined"
        self._counts["confirm_saved" if stored else "confirm_declined"] += 1
        return True

    def _speak_all(self, turn: _Turn, sentences: list[Sentence]) -> bool:
        """Speak the sentences; False when the answer must stop (cap reached)."""
        return all(self._speak(turn, sentence) for sentence in sentences)

    def _speak(self, turn: _Turn, sentence: Sentence) -> bool:
        if turn.sentences >= turn.cap:
            turn.outcome = "capped"
            return False
        mood, intensity = self._mood(turn.cfg, sentence.mood)
        self._say(turn, sentence.text, mood, intensity, continued=sentence.continued)
        turn.sentences += 1
        if turn.sentences >= turn.cap:
            turn.outcome = "capped"
            return False
        return True

    def _mood(self, cfg: RoboEmotConfig, tag: str | None) -> tuple[str, float]:
        """Mood label and intensity for a sentence per ``voice.mood_source``: the affect label,
        or the LLM's tag when it is close enough to the affect state (``llm_tagged``); ``fixed``
        is applied by ``compose_delivery``."""
        state = self.affect.state(self.clock())
        label = state.label
        if cfg.voice.mood_source == "llm_tagged":
            label = resolve_tag(
                tag,
                affect_label=state.label,
                valence=state.valence,
                arousal=state.arousal,
                affect=cfg.affect,
                voice=cfg.voice,
            )
        return label, state.intensity

    def _say(
        self, turn: _Turn, text: str, mood: str | None, intensity: float, continued: bool = False
    ) -> None:
        """Say a sentence or a fixed line: in a realtime turn it waits for the server's voice
        (``_speak_lines``), else Piper says it now."""
        if text and not turn.interrupted and turn.voice == "realtime":
            turn.lines.append((text, None))
            return
        self._say_local(turn, text, mood, intensity, continued)

    def _say_local(
        self, turn: _Turn, text: str, mood: str | None, intensity: float, continued: bool = False
    ) -> None:
        if not text or turn.interrupted:  # a barge-in is under way: nothing more is said
            return
        job = self.speech.say(
            text,
            language=turn.language,
            register=turn.register,
            circadian=self._circadian(),
            mood=mood,
            intensity=intensity,
            utterance=turn.speech_id,
            continued=continued,
        )
        if job is not None:
            turn.stamps.setdefault("say", self.clock())
            turn.spoken.append(text)
            turn.queued.append((job, text))
            if self.transcript is not None:
                self._note_reply(turn, text)

    def _note_reply(self, turn: _Turn, text: str) -> None:
        """Add a queued sentence to the live transcript's reply, if the ring still accepts the
        turn's generation."""
        ring = self.transcript
        if ring is None:
            return
        if turn.reply_seq is None:
            turn.reply_seq = ring.add("robot", text, turn.language, generation=turn.ring_gen)
        else:
            ring.extend(turn.reply_seq, text)

    def _prompt_name(self, cfg: RoboEmotConfig) -> str:
        """The robot's name as safe to say and to put in a prompt ('' without one)."""
        self.names.configure(cfg.voice)
        return self.names.prompt_name(cfg.voice.wake_word)

    def _say_fallback(self, turn: _Turn) -> None:
        """No route answered: the "can't think right now" line."""
        text = pick(
            turn.cfg.conversation.fallback_lines, turn.language, turn.cfg.conversation.language
        )
        state = self.affect.state(self.clock())
        self._say(turn, text or "", state.label, state.intensity)
        self._counts["fallbacks"] += 1

    def _finish(self, turn: _Turn) -> None:
        """Keep the whole turn (the user's words, the tool exchange and what was actually
        spoken) in the session's history, record the latency stamps and announce the turn
        without its text. After a barge-in or stop_talking only the sentences that began to play
        are kept, and the message says it was cut. Sentences already in the exchange messages
        are not repeated in the closing one."""
        with self._lock:
            turn.done = True
            cut = turn.stopped or turn.cut
            items = turn.queued[turn.exchange_spoken :]
            texts = turn.spoken[turn.exchange_spoken :]
            kept = [t for job, t in items if self.speech.played(job)] if cut else texts
            spoken = " ".join(t for t in kept if t)
            dropped: list[list[dict[str, Any]]] = []
            if (spoken or turn.exchange) and self._max_turns > 0:
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": turn.utterance.text},
                    *turn.exchange,
                ]
                # the server's items of this turn (deleted with it); _history_message sends a
                # chat route only the chat fields, so this mark never leaves the robot
                if turn.rt_items:
                    messages[0]["rt"] = (turn.rt_gen, list(turn.rt_items))
                if spoken:
                    messages.append({"role": "assistant", "content": spoken})
                    if len(kept) < len(texts) or turn.rt_partial:
                        messages[-1]["interrupted"] = True
                self._turns.append(messages)
                dropped = self._cap_history()
                rt = list(turn.rt_jobs) if turn.voice == "realtime" else None
                self._tail = (
                    _Tail(messages, list(items), cut=cut, rt=rt, rt_gen=turn.rt_gen)
                    if spoken
                    else None
                )
            elif turn.rt_items and self.realtime is not None:
                dropped = [[{"role": "user", "rt": (turn.rt_gen, list(turn.rt_items))}]]
                self._tail = None
            else:
                self._tail = None
            self._active_turn = None
            self._reply = None
            if turn.reply_seq is not None and self.transcript is not None:
                if cut or turn.speech_stopped:  # the transcript keeps what was heard
                    heard = [t for job, t in turn.queued if self.speech.played(job)]
                    self.transcript.settle(
                        turn.reply_seq, " ".join(heard), interrupted=len(heard) < len(turn.queued)
                    )
                else:  # a barge-in while the last sentences still play settles it later
                    self._reply = (turn.reply_seq, list(turn.queued))
        self._delete_items(dropped)
        self._counts["turns"] += 1
        self._counts[turn.outcome] += 1
        with self._lock:
            self._last = {
                "speech_id": turn.speech_id,
                "reasoning_chars": turn.reasoning_chars,
                **turn.stamps,
            }
        if self.bus is not None:
            self.bus.publish(
                TOPIC,
                {
                    "turn": self._counts["turns"],
                    "language": turn.language,
                    "chars_in": len(turn.utterance.text),
                    "chars_out": len(spoken),
                    "sentences": turn.sentences,
                    "route": turn.route,
                    "tokens": turn.tokens,
                    "reasoning_chars": turn.reasoning_chars,
                    "tool_calls": turn.tool_calls,
                    "rounds": turn.rounds,
                    "outcome": turn.outcome,
                },
                self.clock(),
            )

    # ----------------------------------------------------------------- report

    def snapshot(self) -> dict[str, Any]:
        """Counts and the last turn's latencies (ms from the last speech frame) for the debug
        report; never text."""
        with self._lock:
            last = dict(self._last) if self._last else None
            out: dict[str, Any] = {
                "session": self._session is not None,
                "history_turns": len(self._turns),
                "pending": len(self._pending),
                **dict(self._counts),
            }
        if last is not None:
            heard = last["heard"]
            first_audio = self.speech.first_audio_at(last["speech_id"])

            def ms(stamp: float | None) -> float | None:
                return None if stamp is None else (stamp - heard) * MS_PER_S

            out["last_reasoning_chars"] = last["reasoning_chars"]
            out["latency_ms"] = {
                "request": ms(last.get("request")),
                "first_token": ms(last.get("token")),
                "first_sentence": ms(last.get("say")),
                "first_audio": ms(first_audio),
            }
        return out
