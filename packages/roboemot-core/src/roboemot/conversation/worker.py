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

Children (user decision 2026-10-17): while a minor is present only routes cleared for minors are
used (``Router.eligible``), the request carries the minor content rules and, when the active
register is not already a minor one, the child register; replies are cut to the register's
sentence cap; every sentence passes the output filter before TTS; with no cleared route a
built-in child-safe line is spoken instead of silence; a minor appearing mid-answer stops the
speech and cancels the stream. The minor signal is read through one fail-closed wrapper.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..bus import Bus
from ..config.schema import RegisterParams, RoboEmotConfig, StateEffects
from ..config.stack import ConfigStack
from ..llm.hub import RouterHub
from ..llm.prompt import max_reply_sentences, system_prompt
from ..llm.pseudonyms import Pseudonyms
from ..llm.router import (
    ActiveRoute,
    Done,
    NoRouteAvailable,
    Offline,
    TextDelta,
    ToolCall,
    build_messages,
)
from ..persona import apply_to_register, effects
from ..skills import Skill, SkillContext, SkillRegistry, ToolOutcome, execute
from ..voice.keywords import normalise
from ..voice.listener import Utterance
from ..voice.prosody import resolve_tag
from ..voice.robot_name import RobotName
from ..voice.tts import Sentence, SentenceSplitter
from .confirm import Pending, PreferenceStore, reply_kind, sanitise_preference
from .lines import line_text, minor_line, pick
from .output_filter import OutputFilter

log = logging.getLogger(__name__)

TOPIC = "conversation.turn"
SECONDS_PER_MIN = 60.0
MS_PER_S = 1000.0
JOIN_TIMEOUT_S = 2.0
READY_TIMEOUT_S = 5.0
INTERRUPTED_MARK = " …"  # ends an assistant message the person cut off (history, SPEC 8)


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
    minor: bool = False  # the minor state of the latest turn (history is written under it)


@dataclass
class _Turn:
    """State of the answer being prepared."""

    utterance: Utterance
    cfg: RoboEmotConfig
    language: str
    minor: bool
    register: RegisterParams
    speech_id: int
    cap: int
    spoken: list[str] = field(default_factory=list)
    queued: list[tuple[int, str]] = field(default_factory=list)  # (speech job id, text)
    exchange: list[dict[str, Any]] = field(default_factory=list)  # tool calls and results
    speaker: str | None = None
    interrupted: bool = False  # a barge-in is under way: nothing more is said (set first)
    stopped: bool = False  # ... and the speech was stopped: history keeps what was heard
    done: bool = False  # the turn is over: a late skill callback does nothing
    exchange_spoken: int = 0  # queued sentences already in the exchange messages
    cut: bool = False  # stop_talking cut it
    tool_calls: int = 0
    rounds: int = 0
    sentences: int = 0
    outcome: str = "ok"
    route: str | None = None
    tokens: int = 0
    out_chars: int = 0
    stamps: dict[str, float] = field(default_factory=dict)
    active: ActiveRoute = field(default_factory=ActiveRoute)


@dataclass
class _Tail:
    """The latest turn's history entry, so a barge-in after the stream ended (the speech is
    still playing) can cut it back to what was heard."""

    messages: list[dict[str, Any]]
    queued: list[tuple[int, str]]
    cut: bool = False


class ConversationWorker:
    """See the module docstring. Callables are injected so core stays robot-agnostic:

    ``register()`` the persona-shaped active register (None before the first tick),
    ``circadian()`` the circadian state's effects, ``minor_present()`` the one minor signal,
    ``profile(max_chars)`` and ``scene(max_chars)`` the summaries (None when empty).
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
        minor_present: Callable[[], bool],
        profile: Callable[[int], str | None] = lambda cap: None,
        scene: Callable[[int], str | None] = lambda cap: None,
        speaker: Callable[[], str | None] = lambda: None,
        skills: SkillRegistry | None = None,
        preferences: PreferenceStore | None = None,
        bus: Bus | None = None,
        clock: Callable[[], float] = time.monotonic,
        names: RobotName | None = None,
        known_names: Callable[[], Iterable[str]] = lambda: (),
    ) -> None:
        self.names = names if names is not None else RobotName()
        self.stack, self.hub, self.speech = stack, hub, speech
        self.affect, self.responder = affect, responder
        self._register, self._circadian, self._minor_present = register, circadian, minor_present
        self._profile, self._scene, self._speaker = profile, scene, speaker
        self.skills, self.preferences = skills, preferences
        self.bus, self.clock = bus, clock
        self._known_names = known_names
        self._pseudonyms = Pseudonyms()
        self._active_turn: _Turn | None = None
        self._inner: asyncio.Future | None = None
        self._confirm: Pending | None = None
        self._tail: _Tail | None = None
        self._barge_at = -math.inf  # when the last barge-in or stop keyword cut the robot off
        self._lock = threading.Lock()
        self._pending: collections.deque[Utterance] = collections.deque()
        self._turns: collections.deque[list[dict[str, Any]]] = collections.deque()
        self._max_turns = 0
        self._session: _Session | None = None
        self._filter = OutputFilter({})
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._ready = threading.Event()
        self._closed = False
        self._counts: collections.Counter[str] = collections.Counter()
        self._last: dict[str, Any] | None = None
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
            self._thread.join(JOIN_TIMEOUT_S)
        self._end_session("closed")
        with self._lock:
            self._pending.clear()

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: the history cap and the output filter follow the configuration (the rest
        is read from the stack at each turn)."""
        with self._lock:
            self._max_turns = cfg.conversation.history_turns
            while len(self._turns) > self._max_turns:
                self._turns.popleft()
        if cfg.safety.minor_output_blocklist != self._filter.source:
            self._filter = OutputFilter(cfg.safety.minor_output_blocklist)

    # ----------------------------------------------------------------- input

    def submit(self, utterance: Utterance) -> bool:
        """Queue an utterance for an answer; only addressed final ones with text count. While a
        turn runs, at most ``conversation.pending_utterances`` wait and a newer one replaces the
        oldest. Safe from any thread (the listener's). Returns whether it was taken."""
        if self._closed or not (utterance.final and utterance.addressed and utterance.text.strip()):
            return False
        cfg = self.stack.effective()
        if self.clock() - self._barge_at <= cfg.conversation.stop_drop_window_s and (
            self._only_stop(utterance, cfg)
        ):
            self._counts["stop_consumed"] += 1  # the command that already stopped the robot
            return False
        limit = cfg.conversation.pending_utterances
        with self._lock:
            while len(self._pending) >= limit:
                self._pending.popleft()
                self._counts["replaced"] += 1
            self._pending.append(utterance)
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(wake.set)
        return True

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

    def _who(self) -> str | None:
        try:
            return self._speaker()
        except Exception as exc:
            log.warning("conversation: speaker unavailable (%s)", type(exc).__name__)
            return None

    def _touch(self, now: float, speaker: str | None, minor: bool) -> None:
        """Start the session if none is open and stamp the activity. A session belongs to the
        person who opened it and to one minor state: another speaker (a different known
        person, or an unknown one taking over a known person's session, or the reverse), or a
        turn whose minor state differs from the session's latest turn (either way), ends it
        first, so nobody inherits someone else's history and a turn written without the minor
        rules is never sent for a child. Two unknown speakers cannot be told apart and share a
        session while the minor state stays the same."""
        with self._lock:
            s = self._session
            other = s is not None and (s.speaker != speaker or s.minor != minor)
        if other:
            self._end_session("speaker changed")
            self._counts["speaker_changes"] += 1
        with self._lock:
            if self._session is None:
                self._session = _Session(now, speaker, minor)
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
        self._ready.set()
        while not self._closed:
            with self._lock:
                utterance = self._pending.popleft() if self._pending else None
            if utterance is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            try:
                await self._answer(utterance)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one bad turn must not end the worker
                self._counts["errors"] += 1
                log.warning("conversation: turn failed (%s)", type(exc).__name__)

    # ----------------------------------------------------------------- one turn

    def _minor(self) -> bool:
        """Whether a minor is present now; True (fail closed) when the signal cannot be read.
        The one read of the signal: routes, prompt, filter and the mid-answer check use it."""
        try:
            return bool(self._minor_present())
        except Exception as exc:
            log.warning(
                "conversation: minor signal failed, assuming a minor (%s)", type(exc).__name__
            )
            return True

    def _register_for(self, cfg: RoboEmotConfig, minor: bool) -> RegisterParams:
        """The persona-shaped active register. While a minor is present but the register is an
        adult one (the minor hold outlasts the child leaving the view) the child register is
        used, so wording and reply length stay child-appropriate."""
        fx = effects(cfg.persona)
        try:
            active = self._register()
        except Exception as exc:
            log.warning("conversation: register unavailable (%s)", type(exc).__name__)
            active = None
        if active is None:
            active = apply_to_register(cfg.register.unknown, fx)
        if minor and not active.minor_content_policy:
            return apply_to_register(cfg.register.child, fx)
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
                turn.interrupted = turn.stopped = True
            self._drop_unheard_question()
        if turn is None:
            return self._trim_tail()
        self._counts["interrupts"] += 1
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_inner)
        return True

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
            played = [text for job, text in tail.queued if self.speech.played(job)]
            if len(played) == len(tail.queued):
                return False
            if played:
                tail.messages[-1].update(content=" ".join(played), interrupted=True)
            else:
                tail.messages.pop()
            self._counts["history_trimmed"] += 1
        return True

    async def _answer(self, utterance: Utterance) -> None:
        now = self.clock()
        cfg = self.stack.effective()
        self.apply_config(cfg)
        router = self.hub.sync(cfg)
        minor = self._minor()
        self.expire(now)
        speaker = self._who()
        self._touch(now, speaker, minor)
        register = self._register_for(cfg, minor)
        turn = _Turn(
            utterance,
            cfg,
            utterance.language or cfg.conversation.language,
            minor,
            register,
            self.speech.new_utterance(),
            max_reply_sentences(cfg, register),
            speaker=speaker,
        )
        turn.stamps["heard"] = utterance.last_speech_stamp
        self.affect.new_turn()
        with self._lock:
            self._active_turn = turn
        try:
            if not self._confirmation_reply(turn, now):
                await self._converse(turn, router)
        finally:
            self._inner = None
        self._finish(turn)

    async def _converse(self, turn: _Turn, router: Any) -> None:
        """Request, stream, run tool calls, repeat (see ``_rounds``); the errors that end a turn
        become the spoken fallbacks. The rounds run as an inner task, so ``interrupt`` can
        cancel the answer without touching the worker's own shutdown."""
        cfg, conv, minor = turn.cfg, turn.cfg.conversation, turn.minor
        state = self.affect.state(self.clock())
        messages = build_messages(
            system=system_prompt(
                cfg,
                turn.register,
                language=turn.language,
                minor_present=minor,
                robot_name=self._prompt_name(cfg),
            ),
            profile_summary=self._summary(self._profile, conv.profile_summary_max_chars),
            scene_summary=self._summary(self._scene, conv.scene_summary_max_chars),
            history=self._history(),
            user_text=turn.utterance.text,
            affect=f"{state.label}, intensity {state.intensity:.1f}",
            summary_max_chars=(conv.profile_summary_max_chars, conv.scene_summary_max_chars),
        )
        splitter = SentenceSplitter(cfg.voice.sentence_max_chars, strip_tags=True)
        turn.stamps["request"] = self.clock()
        inner = self._inner = asyncio.ensure_future(self._rounds(turn, router, messages, splitter))
        if turn.interrupted:  # a barge-in landed before the task existed
            inner.cancel()
        try:
            await inner
        except Offline:
            turn.outcome = "offline"
            self.responder.speech_while_offline(self.clock())
            if turn.minor or self._minor():  # never silence for a child
                self._say_minor_line(turn, "no_route")
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
        ``conversation.max_tool_rounds`` rounds ran (the next request then has no tools). The
        minor rules apply to every round: routes, the filter, the mid-answer switch."""
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
            if turn.rounds and self._minor_appeared(turn, turn.active):
                return
            active = turn.active = ActiveRoute()
            stream = router.stream(
                messages, tools, minor_present=turn.minor, active=active, pseudonyms=pseudonyms
            )
            calls: list[ToolCall] = []
            results: list[ToolOutcome] = []
            cut = False
            async with contextlib.aclosing(stream):
                async for ev in stream:
                    if isinstance(ev, TextDelta):
                        turn.stamps.setdefault("token", self.clock())
                        turn.out_chars += len(ev.text)
                        if self._minor_appeared(turn, active):
                            cut = True
                        else:
                            cut = not self._speak_all(turn, splitter.feed(ev.text))
                    elif isinstance(ev, ToolCall) and offered:  # (none offered: ignored)
                        if self._minor_appeared(turn, active):
                            cut = True
                        else:
                            # what the model said before the call is said before the skill acts
                            cut = not self._speak_all(turn, splitter.flush())
                            if not cut:
                                calls.append(ev)
                                results.append(
                                    await self._call(turn, ev, choices, len(calls) > cap)
                                )
                                cut = turn.cut or results[-1].end_turn
                    elif isinstance(ev, Done):
                        turn.route, turn.tokens = ev.route, turn.tokens + ev.tokens
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
        """With ``privacy.pseudonyms_to_cloud``: give aliases to the registry names not seen yet
        and return the map for the router, which applies it to routes that are not local. The
        names are read before every request (a record may have been added, the ``look_at``
        choices are of this moment). If they cannot be read the exception ends the turn: sending
        without the map would put real names on the cloud route. None when the option is off."""
        cfg = turn.cfg
        if not cfg.privacy.pseudonyms_to_cloud:
            return None
        self._pseudonyms.learn(
            self._known_names(),
            cfg.privacy.pseudonym_templates,
            turn.language,
            cfg.conversation.language,
        )
        return self._pseudonyms

    def _offer(self, turn: _Turn) -> list[Skill]:
        """The skills the model may call in this request (visible, listed, allowed with a minor
        present: asked again for every round)."""
        if self.skills is None:
            return []
        return self.skills.offered(
            turn.cfg.conversation.llm_skills, minor=turn.minor or self._minor()
        )

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
                            json.dumps(c.arguments)
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
            turn.minor or self._minor(),
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
        turn.cut = True
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
        if (turn.minor or self._minor()) and self._filter.blocked(text, " ".join(turn.spoken)):
            return "not_allowed"
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

    def _cleared(self, active: ActiveRoute) -> bool:
        """Whether the route answering is cleared for minors in the configuration as it is now
        (not the router's, which may be stale): looked up by name, a missing route is not."""
        if active.route is None:
            return False
        cfg = self.stack.effective()
        route = next((r for r in cfg.llm_routes if r.name == active.route.name), None)
        return route is not None and route.allowed_for_minors and cfg.safety.minors_allowed_llm

    def _minor_appeared(self, turn: _Turn, active: ActiveRoute) -> bool:
        """A minor showed up after the request was composed, or the route answering is not
        cleared while one is present: stop the speech, drop the rest of the answer (it was
        written without the minor rules). The caller cancels the stream, which charges it."""
        if not self._minor():
            return False
        if not turn.minor or not self._cleared(active):
            self.speech.stop()
            turn.outcome = "minor_cut"
            return True
        return False

    def _speak_all(self, turn: _Turn, sentences: list[Sentence]) -> bool:
        """Speak the sentences; False when the answer must stop (cap reached or filtered)."""
        return all(self._speak(turn, sentence) for sentence in sentences)

    def _speak(self, turn: _Turn, sentence: Sentence) -> bool:
        if turn.sentences >= turn.cap:
            turn.outcome = "capped"
            return False
        if self._minor_appeared(turn, turn.active):
            return False
        if (turn.minor or self._minor()) and self._filter.blocked(
            sentence.text, " ".join(turn.spoken)
        ):
            self._say_minor_line(turn, "filtered")
            turn.outcome = "filtered"
            return False
        mood, intensity = self._mood(turn.cfg, sentence.mood)
        self._say(turn, sentence.text, mood, intensity)
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
        self, turn: _Turn, text: str, mood: str | None, intensity: float, *, minor: bool = False
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
            minor=turn.minor or minor,
        )
        if job is not None:
            turn.stamps.setdefault("say", self.clock())
            turn.spoken.append(text)
            turn.queued.append((job, text))

    def _prompt_name(self, cfg: RoboEmotConfig) -> str:
        """The robot's name as safe to say and to put in a prompt ('' without one)."""
        self.names.configure(cfg.voice)
        return self.names.prompt_name(cfg.voice.wake_word)

    def _say_minor_line(self, turn: _Turn, which: str) -> None:
        """A built-in child-safe line (``safety.minor_lines``), never asking a route."""
        if which == "filtered":
            text = line_text(
                turn.cfg.safety.minor_lines,
                turn.language,
                turn.cfg.conversation.language,
                "filtered",
            )
        else:
            text = minor_line(
                turn.cfg.safety.minor_lines,
                turn.language,
                turn.cfg.conversation.language,
                turn.utterance.keywords,
                self._prompt_name(turn.cfg),
            )
        state = self.affect.state(self.clock())
        self._say(turn, text, state.label, state.intensity, minor=True)
        self._counts["minor_lines"] += 1

    def _say_fallback(self, turn: _Turn) -> None:
        """No route answered: while a minor is present a child-safe line (an uncleared route is
        never asked), otherwise the register's "can't think right now" line."""
        if turn.minor or self._minor():
            self._say_minor_line(turn, "no_route")
            return
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
            spoken = " ".join(kept)
            if (spoken or turn.exchange) and self._max_turns > 0:
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": turn.utterance.text},
                    *turn.exchange,
                ]
                if spoken:
                    messages.append({"role": "assistant", "content": spoken})
                    if len(kept) < len(texts):
                        messages[-1]["interrupted"] = True
                self._turns.append(messages)
                while len(self._turns) > self._max_turns:
                    self._turns.popleft()
                self._tail = _Tail(messages, list(items), cut=cut) if spoken else None
            else:
                self._tail = None
            self._active_turn = None
        self._counts["turns"] += 1
        self._counts[turn.outcome] += 1
        with self._lock:
            self._last = {"speech_id": turn.speech_id, **turn.stamps}
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
                    "minor": turn.minor,
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

            out["latency_ms"] = {
                "request": ms(last.get("request")),
                "first_token": ms(last.get("token")),
                "first_sentence": ms(last.get("say")),
                "first_audio": ms(first_audio),
            }
        return out
