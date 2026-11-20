"""Fakes for the expressor tests (SPEC 8, Robot expression): a clock moved only by ``wait``,
a worker that records reaction turns, a media player spy, a circadian runner and the host."""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from typing import Any

from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.runtime.expressor import Expressor, ReactionRequest


class Clock:
    """Monotonic seconds moved only by ``wait`` (no real sleeps)."""

    def __init__(self) -> None:
        self.t = 0.0
        self.waits: list[Any] = []  # callbacks run on each wait (a scene that changes)

    def __call__(self) -> float:
        return self.t

    def wait(self, seconds: float) -> bool:
        self.t += seconds
        for hook in list(self.waits):
            hook(self.t)
        return False


def done(value: Any) -> concurrent.futures.Future:
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(value)
    return future


class FakeWorker:
    """Records ``reaction_turn`` calls and answers from ``answers`` (the last one repeats);
    ``busy_until`` is when the conversation stops being busy (the idle check)."""

    def __init__(self, clock: Clock, answers=("spoken",)) -> None:
        self.clock = clock
        self.answers = list(answers)
        self.turns: list[dict[str, Any]] = []
        self.lines: list[tuple[str, str]] = []
        self.voice = True
        self.session = False
        self.busy_until = 0.0
        self.after = 0
        self.on_turn: Any = None  # called with the call's index before it answers

    def reaction_turn(self, text, language, *, timeout_s, quiet, profile=True):
        self.turns.append(
            {
                "t": self.clock(),
                "text": text,
                "language": language,
                "timeout_s": timeout_s,
                "quiet": quiet(),
                "profile": profile,
            }
        )
        if self.on_turn is not None:
            self.on_turn(len(self.turns) - 1)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return done(answer)

    def local_line(self, text, language):
        self.lines.append((text, language))
        return done(self.voice)

    def local_voice(self) -> bool:
        return self.voice

    def after_turn(self, callback):
        self.after += 1
        callback()

    def busy(self) -> bool:
        return self.clock() < self.busy_until

    def in_session(self) -> bool:
        return self.session


class MediaSpy:
    """A media player that records every call; only ``playing`` and ``audible`` are reads."""

    def __init__(self) -> None:
        self.is_playing = False
        self.is_audible = False
        self.calls: list[str] = []

    def playing(self) -> bool:
        return self.is_playing

    def audible(self) -> bool:
        return self.is_audible

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args: Any, **kw: Any) -> None:
            self.calls.append(name)

        return call


class FakeRunner:
    def __init__(self) -> None:
        self.enabled = True
        self.state = "awake"
        self.reason: str | None = None
        self.quiet = False
        self.wakes = 0
        self.model = self

    def quiet_now(self) -> bool:
        return self.quiet

    def wake(self) -> None:
        self.wakes += 1


class FakeExpression:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def request(self, label, now=None):
        self.requests.append(label)
        return True


class FakeCues:
    def __init__(self) -> None:
        self.played: list[tuple[str, str, dict]] = []

    def play(self, kind, reason="", **detail):
        self.played.append((kind, reason, detail))
        return True


class Rig:
    def __init__(self, *, answers=("spoken",), app=None, online=True) -> None:
        self.clock = Clock()
        self.stack = ConfigStack(app={"conversation": {"language": "it"}, **(app or {})})
        self.worker = FakeWorker(self.clock, answers)
        self.media = MediaSpy()
        self.runner = FakeRunner()
        self.online = online
        self.asleep = False
        self.events: list[tuple[str, str, dict]] = []
        listener = SimpleNamespace(voice_active=lambda: False, opened=[])
        listener.open_follow_up = lambda: listener.opened.append(self.clock())
        self.host = SimpleNamespace(
            phase=None,
            speech=SimpleNamespace(speaking=lambda now: False),
            speech_output=SimpleNamespace(speaking_at=lambda now: False),
            voice_listener=listener,
            conversation=self.worker,
            circadian_runner=self.runner,
            expression=FakeExpression(),
            cues=FakeCues(),
            bus=Bus(),
            media=self.media,
            announcer=None,
        )
        self.expressor = Expressor(
            self.worker,
            self.host,
            self.stack.effective,
            online=lambda: self.online,
            asleep=lambda: self.asleep,
            clock=self.clock,
            wait=self.clock.wait,
        )
        self.expressor.events = lambda kind, reason, **f: self.events.append((kind, reason, f))

    def request(self, situation: str, data=None, at: float | None = None, **kw) -> ReactionRequest:
        created = self.clock() if at is None else at
        return ReactionRequest(situation, dict(data or {}), created, **kw)  # type: ignore[arg-type]

    def shown(self, category: str = "cup", entry_id: str = "e1") -> ReactionRequest:
        """Show the robot a thing (a new look, as attention publishes it) and take the request
        it queued, as the expressor's thread would."""
        from roboemot.attention.types import SHOWN, AttentionTarget

        target = AttentionTarget(entry_id, "object", 0.0, 0.0, SHOWN, "new", 0.0, category)
        self.expressor._on_attention(SimpleNamespace(payload=target))
        with self.expressor._lock:
            return self.expressor._queue.pop("shown_object")

    def counts(self, situation: str) -> dict[str, int]:
        return self.expressor.snapshot()["counts"].get(situation, {})
