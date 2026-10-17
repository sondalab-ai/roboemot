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
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..bus import Bus
from ..config.schema import RegisterParams, RoboEmotConfig, StateEffects
from ..config.stack import ConfigStack
from ..llm.hub import RouterHub
from ..llm.prompt import max_reply_sentences, system_prompt
from ..llm.router import (
    ActiveRoute,
    Done,
    NoRouteAvailable,
    Offline,
    TextDelta,
    build_messages,
)
from ..persona import apply_to_register, effects
from ..voice.listener import Utterance
from ..voice.prosody import resolve_tag
from ..voice.tts import Sentence, SentenceSplitter
from .lines import minor_line, pick
from .output_filter import OutputFilter

log = logging.getLogger(__name__)

TOPIC = "conversation.turn"
SECONDS_PER_MIN = 60.0
MS_PER_S = 1000.0
JOIN_TIMEOUT_S = 2.0
READY_TIMEOUT_S = 5.0


class Speech(Protocol):
    """What the worker needs of ``SpeechOutput``."""

    def new_utterance(self) -> int: ...
    def say(self, text: str, **kwargs: Any) -> int | None: ...
    def stop(self) -> None: ...
    def first_audio_at(self, utterance: int) -> float | None: ...


@dataclass
class _Session:
    last_active: float


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
    sentences: int = 0
    outcome: str = "ok"
    route: str | None = None
    tokens: int = 0
    out_chars: int = 0
    stamps: dict[str, float] = field(default_factory=dict)


class ConversationWorker:
    """See the module docstring. Callables are injected so core stays robot-agnostic:

    ``register()`` the persona-shaped active register (None before the first tick),
    ``circadian()`` the circadian state's effects, ``minor_present()`` the one minor signal,
    ``profile(max_chars)`` and ``scene(max_chars)`` the summaries (None when empty).
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
        bus: Bus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stack, self.hub, self.speech = stack, hub, speech
        self.affect, self.responder = affect, responder
        self._register, self._circadian, self._minor_present = register, circadian, minor_present
        self._profile, self._scene = profile, scene
        self.bus, self.clock = bus, clock
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
        limit = self.stack.effective().conversation.pending_utterances
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

    def note_presence(self, now: float) -> None:
        """The person is still in view (app tick): keeps the session alive."""
        with self._lock:
            if self._session is not None:
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
        if had:
            log.debug("conversation: session ended (%s)", why)

    def _touch(self, now: float) -> None:
        """Start the session if none is open and stamp the activity."""
        with self._lock:
            if self._session is None:
                self._session = _Session(now)
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
            return [m for turn in self._turns for m in turn]

    async def _answer(self, utterance: Utterance) -> None:
        now = self.clock()
        cfg = self.stack.effective()
        self.apply_config(cfg)
        router = self.hub.sync(cfg)
        self.expire(now)
        self._touch(now)
        minor = self._minor()
        register = self._register_for(cfg, minor)
        conv = cfg.conversation
        turn = _Turn(
            utterance,
            cfg,
            utterance.language or conv.language,
            minor,
            register,
            self.speech.new_utterance(),
            max_reply_sentences(cfg, register),
        )
        turn.stamps["heard"] = utterance.last_speech_stamp
        self.affect.new_turn()
        state = self.affect.state(now)
        messages = build_messages(
            system=system_prompt(cfg, register, language=turn.language, minor_present=minor),
            profile_summary=self._summary(self._profile, conv.profile_summary_max_chars),
            scene_summary=self._summary(self._scene, conv.scene_summary_max_chars),
            history=self._history(),
            user_text=utterance.text,
            affect=f"{state.label}, intensity {state.intensity:.1f}",
            summary_max_chars=(conv.profile_summary_max_chars, conv.scene_summary_max_chars),
        )
        splitter = SentenceSplitter(cfg.voice.sentence_max_chars, strip_tags=True)
        active = ActiveRoute()
        turn.stamps["request"] = self.clock()
        stream = router.stream(messages, None, minor_present=minor, active=active)
        try:
            async with contextlib.aclosing(stream):
                cut = False
                async for ev in stream:
                    if isinstance(ev, TextDelta):
                        turn.stamps.setdefault("token", self.clock())
                        turn.out_chars += len(ev.text)
                        if self._minor_appeared(turn, active):
                            cut = True
                        else:
                            cut = not self._speak_all(turn, splitter.feed(ev.text))
                    elif isinstance(ev, Done):
                        turn.route, turn.tokens = ev.route, ev.tokens
                    if cut:
                        break
                if not cut:
                    self._speak_all(turn, splitter.flush())
        except Offline:
            turn.outcome = "offline"
            self.responder.speech_while_offline(self.clock())
        except NoRouteAvailable:
            turn.outcome = "no_route"
            self._say_fallback(turn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            turn.outcome = "error"
            log.warning("conversation: stream failed (%s)", type(exc).__name__)
            if not turn.spoken:
                self._say_fallback(turn)
        self._finish(turn)

    def _cleared(self, cfg: RoboEmotConfig, active: ActiveRoute) -> bool:
        route = active.route
        return route is not None and route.allowed_for_minors and cfg.safety.minors_allowed_llm

    def _minor_appeared(self, turn: _Turn, active: ActiveRoute) -> bool:
        """A minor showed up after the request was composed, or the route answering is not
        cleared while one is present: stop the speech, drop the rest of the answer (it was
        written without the minor rules). The caller cancels the stream, which charges it."""
        if not self._minor():
            return False
        if not turn.minor or not self._cleared(turn.cfg, active):
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
        if self._minor() and self._filter.blocked(sentence.text):
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

    def _say(self, turn: _Turn, text: str, mood: str | None, intensity: float) -> None:
        if not text:
            return
        job = self.speech.say(
            text,
            language=turn.language,
            register=turn.register,
            circadian=self._circadian(),
            mood=mood,
            intensity=intensity,
            utterance=turn.speech_id,
        )
        if job is not None:
            turn.stamps.setdefault("say", self.clock())
            turn.spoken.append(text)

    def _say_minor_line(self, turn: _Turn, which: str) -> None:
        """A built-in child-safe line (``safety.minor_lines``), never asking a route."""
        lines = pick(turn.cfg.safety.minor_lines, turn.language, turn.cfg.conversation.language)
        if which == "filtered":
            text = lines.filtered if lines else ""
        else:
            text = minor_line(
                turn.cfg.safety.minor_lines,
                turn.language,
                turn.cfg.conversation.language,
                turn.utterance.keywords,
            )
        state = self.affect.state(self.clock())
        self._say(turn, text, state.label, state.intensity)
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
        """Keep the whole turn (the user's words and what was actually spoken) in the session's
        history, record the latency stamps and announce the turn without its text."""
        spoken = " ".join(turn.spoken)
        if spoken and self._max_turns > 0:
            with self._lock:
                self._turns.append(
                    [
                        {"role": "user", "content": turn.utterance.text},
                        {"role": "assistant", "content": spoken},
                    ]
                )
                while len(self._turns) > self._max_turns:
                    self._turns.popleft()
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
