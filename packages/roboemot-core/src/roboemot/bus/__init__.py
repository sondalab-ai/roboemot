"""In-process message bus with the SPEC 11 topic names.

The production transport (Zenoh or iceoryx2 with shared-memory frames) is chosen in M0; it
will implement this same interface. Subscribers are called synchronously in publish order;
async consumers use :meth:`Bus.queue`, which keeps only the newest ``maxsize`` messages so a
slow consumer never blocks a producer (latest-wins, right for frames and state).
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TOPICS = (
    "percept.face",
    "percept.face_frame",
    "percept.pose",
    "percept.utterance",
    "percept.entity",
    "world.entity_appeared",
    "world.entity_moved",
    "world.entity_gone",
    "world.snapshot",
    "person.recognized",
    "affect.state",
    "attention.target",
    "motion.request",
    "motion.state",
    "speech.say",
    "safety.event",
    "circadian.state",
    "config.changed",
    "frames.camera",
    "frames.audio_in",
    "frames.audio_out",
)


@dataclass
class Message:
    topic: str
    payload: Any
    stamp: float
    meta: dict[str, Any] = field(default_factory=dict)


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, fn: Callable[[Message], None]) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs[topic]:
                    self._subs[topic].remove(fn)

        return unsubscribe

    def publish(self, topic: str, payload: Any, stamp: float, **meta: Any) -> None:
        msg = Message(topic, payload, stamp, meta)
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for fn in subs:
            fn(msg)

    def queue(
        self, topic: str, maxsize: int = 1, loop: asyncio.AbstractEventLoop | None = None
    ) -> asyncio.Queue[Message]:
        """Async queue fed from any thread; drops the oldest message when full."""
        q: asyncio.Queue[Message] = asyncio.Queue(maxsize)
        target_loop = loop or asyncio.get_running_loop()

        def put(msg: Message) -> None:
            def _put() -> None:
                if q.full():
                    q.get_nowait()
                q.put_nowait(msg)

            target_loop.call_soon_threadsafe(_put)

        self.subscribe(topic, put)
        return q
