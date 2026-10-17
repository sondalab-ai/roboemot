"""SPEC 8, stage 7 / 13: SpeechOutput (queue, persona effects, stop latency, privacy, minors)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import httpx
import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RegisterParams, RoboEmotConfig, StateEffects
from roboemot.voice.speech import SpeechOutput
from roboemot.voice.tts import AudioChunk, FallbackTTS, TTSError
from test_tts import SpeechServer, openai_backend

RATE = 16000
REG, CIRC = RegisterParams(), StateEffects()


class FakeSink:
    def __init__(self) -> None:
        self.pushed: list[tuple[int, int]] = []
        self.stopped_at: list[float] = []
        self.pushed_at: list[float] = []

    def push_audio_sample(self, samples, rate_hz):
        self.pushed.append((samples.size, rate_hz))
        self.pushed_at.append(time.monotonic())
        return 0.0

    def stop(self):
        self.stopped_at.append(time.monotonic())


class FakeBackend:
    name, controls = "fake", frozenset()

    def __init__(self, pieces=1, delay=0.0, fail=False) -> None:
        self.pieces, self.delay, self.fail = pieces, delay, fail
        self.deliveries, self.minors, self.cancelled = [], [], threading.Event()
        self.started = threading.Event()

    def available(self, *, minor_present=False):
        return True

    async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
        self.deliveries.append(delivery)
        self.minors.append(minor_present)
        self.started.set()
        if self.fail:
            raise TTSError("boom")
        for _ in range(self.pieces):
            await asyncio.sleep(self.delay)
            yield AudioChunk(np.ones(RATE // 10, np.float32), RATE)

    def cancel(self):
        self.cancelled.set()

    async def warm(self, language, voice):
        pass

    async def aclose(self):
        pass


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def say(out, text="Ciao.", **kw):
    args = {"language": "it", "register": REG, "circadian": CIRC, "mood": None, "intensity": 0.0}
    return out.say(text, **args | kw)


@pytest.fixture()
def make():
    made = []

    def build(backend=None, cfg=None, sink=None, **kw):
        out = SpeechOutput(
            cfg or RoboEmotConfig(), sink or FakeSink(), backend or FakeBackend(), **kw
        )
        out.start()
        made.append(out)
        return out

    yield build
    for out in made:
        out.close()


def test_sentences_are_spoken_in_order_with_a_pause_inside_one_utterance(make):
    sink, backend = FakeSink(), FakeBackend()
    out = make(backend, sink=sink)
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    say(out, "Due.", utterance=u)
    assert wait_for(lambda: out.sentences_spoken == 2)
    sizes = [n for n, _ in sink.pushed]
    pause = round(out.config.voice.sentence_pause_ms / 1000 * RATE)
    assert sizes == [RATE // 10, pause, RATE // 10]  # no pause before the first sentence


def test_first_audio_latency_is_measured_from_say(make):
    out = make(FakeBackend(delay=0.05))
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    ms = out.snapshot()["first_audio_ms_last"]
    assert 40 <= ms < 1000


def test_persona_effects_reach_compose_delivery(make):
    cfg = RoboEmotConfig.model_validate({"persona": {"traits": {"energy": 1.0, "warmth": 1.0}}})
    plain = RoboEmotConfig()
    a, b = FakeBackend(), FakeBackend()
    out_a, out_b = make(a, cfg), make(b, plain)
    say(out_a)
    say(out_b)
    assert wait_for(lambda: a.deliveries and b.deliveries)
    assert a.deliveries[0].rate > b.deliveries[0].rate  # energy 1 speaks faster (voice_rate_x)
    assert a.deliveries[0].pitch_st != b.deliveries[0].pitch_st  # warmth moves the pitch


def test_mood_register_and_circadian_shape_the_delivery(make):
    backend = FakeBackend()
    out = make(backend)
    say(out, mood="joy", intensity=1.0, register=RegisterParams(speech_rate=0.8))
    assert wait_for(lambda: backend.deliveries)
    d = backend.deliveries[0]
    assert d.mood == "joy" and d.instructions and d.rate != out.config.voice.base.rate


def test_stop_cuts_the_sink_at_once_and_cancels_the_synthesis_within_budget(make):
    sink, backend = FakeSink(), FakeBackend(pieces=50, delay=0.05)
    out = make(backend, sink=sink)
    for _ in range(3):
        say(out)
    assert backend.started.wait(2) and wait_for(lambda: sink.pushed)
    t0 = time.monotonic()
    out.stop()
    assert sink.stopped_at and sink.stopped_at[0] - t0 < 0.05
    assert backend.cancelled.wait(0.2)
    n = len(sink.pushed)
    time.sleep(0.3)
    assert len(sink.pushed) == n  # nothing plays after stop, queued sentences are gone
    assert out.snapshot()["queued"] == 0
    say(out)  # the output works again after a stop
    assert wait_for(lambda: len(sink.pushed) > n)


def test_a_failing_backend_does_not_end_the_output(make):
    backend = FakeBackend(fail=True)
    out = make(backend)
    say(out)
    assert wait_for(lambda: out.failures == 1)
    backend.fail = False
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)


def test_queue_is_bounded(make):
    cfg = RoboEmotConfig.model_validate({"voice": {"queue_max_sentences": 2}})
    out = make(FakeBackend(pieces=30, delay=0.05), cfg)
    ids = [say(out) for _ in range(6)]
    assert None in ids and out.snapshot()["dropped"] >= 1


def test_bus_and_logs_carry_no_text(make, caplog):
    caplog.set_level(logging.DEBUG)
    bus, seen = Bus(), []
    bus.subscribe("speech.say", seen.append)
    out = make(FakeBackend(fail=True), bus=bus)
    say(out, "frase riservata", mood="joy", intensity=1.0)
    assert wait_for(lambda: out.failures == 1)
    payload = seen[0].payload
    assert payload["chars"] == len("frase riservata") and payload["mood"] == "joy"
    assert "riservata" not in repr(payload) and "riservata" not in caplog.text
    assert "riservata" not in repr(out.snapshot())


def test_minor_appearing_after_enqueue_blocks_the_remote_route(make):
    """The minors flag is read when synthesis starts, not when the sentence was queued."""
    server = SpeechServer()
    remote, _ = openai_backend(server)
    local = FakeBackend()
    local.name = "piper"
    minor = {"present": False}
    gate = threading.Event()

    sink = FakeSink()
    out = make(FallbackTTS(remote, local), sink=sink, minor_present=lambda: minor["present"])
    # first sentence keeps the loop busy while the minor walks in
    original = out.backend.synthesize

    async def gated(*a, **k):
        gate.wait(2)
        async for c in original(*a, **k):
            yield c

    out.backend.synthesize = gated
    say(out)  # queued while no minor is present
    minor["present"] = True
    gate.set()
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert server.requests == []  # the remote route was never contacted
    assert local.minors == [True]


def test_adult_sentence_uses_the_remote_route(make):
    server = SpeechServer()
    remote, _ = openai_backend(server)
    out = make(FallbackTTS(remote, FakeBackend()))
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1) and len(server.requests) == 1
    assert isinstance(server.requests[0], httpx.Request)


def test_factory_picks_backends_and_default_locations():
    from roboemot import paths
    from roboemot.llm import Router
    from roboemot.voice.factory import build_backend, piper_locations
    from roboemot.voice.piper_backend import PiperBackend

    cfg = RoboEmotConfig()
    router = Router([], cfg.safety)
    python, voices = piper_locations(cfg)
    assert python == paths.piper_python(cfg.data_dir)
    assert voices.parts[-2:] == ("voice", "piper")
    assert isinstance(build_backend(cfg, router), PiperBackend)
    remote = RoboEmotConfig.model_validate({"voice": {"backend": "openai_speech"}})
    chain = build_backend(remote, router)
    assert isinstance(chain, FallbackTTS) and chain.primary.name == "openai_speech"
    assert chain.fallback.name == "piper" and not chain.primary.available()  # no route set
    with pytest.raises(ValueError, match="kokoro"):
        build_backend(RoboEmotConfig.model_validate({"voice": {"backend": "kokoro"}}), router)
