"""A conversation worker answering through the realtime route against the fake server: the real
``RealtimeRoute``, ``RealtimeClient``, ``SpeechOutput`` (fake backend, a paced sink feeding an
echo reference), router and network monitor; the chat routes are the httpx fake of the worker
tests (SPEC 8, Realtime speech route)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from roboemot.bus import Bus
from roboemot.config import ConfigStack
from roboemot.config.schema import LinkPolicy, StateEffects
from roboemot.conversation import ConversationWorker
from roboemot.health.network import NetworkMonitor
from roboemot.llm.hub import RouterHub
from roboemot.realtime.route import RealtimeRoute
from roboemot.realtime.session import DirectSession
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.speech import SpeechOutput
from test_conversation_worker import LAN, FakeAffect, FakeResponder, Server, utt, wait_for
from test_speech import FakeBackend
from turn_server import RATE, TurnServer


class PacedSink:
    """Plays in real time on paper: ``playing_until`` moves with what was pushed; every push
    feeds the echo reference, as the robot's speaker does."""

    def __init__(self, reference: PlaybackReference) -> None:
        self.reference = reference
        self.until = 0.0
        self.pushed: list[np.ndarray] = []
        self.stops = 0

    def push_audio_sample(self, samples: np.ndarray, rate_hz: int) -> float:
        now = time.monotonic()
        start = max(now, self.until)
        self.pushed.append(np.asarray(samples, np.float32).copy())
        self.reference.add(start, np.asarray(samples, np.float32))
        self.until = start + samples.size / rate_hz
        return self.until

    def playing_until(self) -> float:
        return self.until

    def end_stream(self) -> float:
        return self.until

    def stop(self) -> None:
        self.stops += 1
        self.until = time.monotonic()

    def peak(self) -> float:
        return max((float(np.max(np.abs(p))) for p in self.pushed if p.size), default=0.0)


@dataclass
class RtRig:
    stack: ConfigStack
    server: TurnServer
    chat: Server
    hub: RouterHub
    network: NetworkMonitor
    route: RealtimeRoute
    speech: SpeechOutput
    sink: PacedSink
    worker: ConversationWorker
    bus: Bus
    circadian: list = field(default_factory=lambda: [StateEffects()])
    events: list = field(default_factory=list)

    def say(self, text: str = "ciao", **kw: Any) -> int:
        n = self.worker.snapshot().get("turns", 0)
        kw.setdefault("stamp", time.monotonic())
        assert self.worker.submit(utt(text, **kw))
        return n + 1

    def turn(self, text: str = "ciao", **kw: Any) -> None:
        n = self.say(text, **kw)
        assert wait_for(lambda: self.worker.snapshot().get("turns", 0) >= n, 10.0)

    def chat_hosts(self) -> list[str]:
        return self.chat.hosts()

    def history(self) -> list[list[dict]]:
        with self.worker._lock:
            return [list(t) for t in self.worker._turns]

    def close(self) -> None:
        self.worker.close()
        self.route.close()
        self.speech.close()


def rt_route(server: TurnServer, **kw: Any) -> dict[str, Any]:
    return {
        "name": "rt",
        "base_url": server.http_base(),
        "model": "s2s",
        "kind": "realtime",
        "locality": "local",
        "api_key": "none",
        "first_token_timeout_s": 3,
        "stall_timeout_s": 3,
        **kw,
    }


def build_rt(
    server: TurnServer,
    *,
    rt: dict[str, Any] | None = None,
    route: dict[str, Any] | None = None,
    extra: bool = True,
    provider: Any = None,
    app: dict[str, Any] | None = None,
) -> RtRig:
    cfg: dict[str, Any] = {
        "llm_routes": [LAN, route or rt_route(server)],
        "conversation": {"language": "it"},
        "voice": {
            "sample_rate_hz": RATE,
            "realtime_route": {
                "enabled": True,
                "route": "rt",
                "prebuffer_ms": 0,
                "connect_timeout_s": 2,
                "send_timeout_s": 1,
                **(rt or {}),
            },
        },
    }
    for key, value in (app or {}).items():
        cfg[key] = {**cfg.get(key, {}), **value}
    stack = ConfigStack(app=cfg)
    chat = Server()
    network = NetworkMonitor(
        stack.effective().llm_routes, LinkPolicy(), probe=lambda host, port, timeout: True
    )

    def make(c: Any) -> Any:
        r = chat.router(c)
        r.monitor = network
        return r

    hub = RouterHub(make(stack.effective()), make)
    reference = PlaybackReference(RATE, 30.0)
    sink = PacedSink(reference)
    speech = SpeechOutput(stack.effective(), sink, FakeBackend())
    speech.start()
    provider = provider if provider is not None else DirectSession()
    route_part = RealtimeRoute(stack, lambda: hub.router, network, provider, extra=lambda: extra)
    bus = Bus()
    rig = RtRig(stack, server, chat, hub, network, route_part, speech, sink, None, bus)  # type: ignore[arg-type]
    rig.worker = ConversationWorker(
        stack,
        hub,
        speech,
        affect=FakeAffect(),
        responder=FakeResponder(),
        register=lambda: None,
        circadian=lambda: rig.circadian[0],
        bus=bus,
        realtime=route_part,
    )
    bus.subscribe("conversation.turn", lambda m: rig.events.append(m.payload))
    rig.worker.start()
    return rig


__all__ = ["PacedSink", "RtRig", "build_rt", "rt_route", "wait_for"]
