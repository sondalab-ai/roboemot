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
from roboemot.voice.speech import SpeechOutput, ends_with_question
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

    def playing_until(self):
        return time.monotonic()

    def stop(self):
        self.stopped_at.append(time.monotonic())


class FakeBackend:
    name, controls = "fake", frozenset()

    def __init__(self, pieces=1, delay=0.0, fail=False) -> None:
        self.pieces, self.delay, self.fail = pieces, delay, fail
        self.deliveries, self.minors, self.cancelled = [], [], threading.Event()
        self.texts: list[str] = []
        self.started = threading.Event()

    def available(self, *, minor_present=False):
        return True

    async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
        self.deliveries.append(delivery)
        self.minors.append(minor_present)
        self.texts.append(text)
        self.started.set()
        if self.fail:
            raise TTSError("boom")
        try:
            for _ in range(self.pieces):
                await asyncio.sleep(self.delay)
                yield AudioChunk(np.ones(RATE // 10, np.float32), RATE)
        except asyncio.CancelledError:
            self.cancelled.set()  # the task running this request was cancelled
            raise

    def cancel(self):
        self.cancelled.set()

    async def warm(self, language, voice):
        pass

    async def aclose(self):
        pass


class HoldingBackend(FakeBackend):
    """The first request answers at once; every later one waits (until the stop cancels it), so a
    queued sentence can never reach the sink before the test's own ``stop``."""

    HOLD_S = 30.0

    async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
        if self.texts:
            self.texts.append(text)
            await asyncio.sleep(self.HOLD_S)
            return
        async for chunk in super().synthesize(
            text, delivery, language=language, voice=voice, minor_present=minor_present
        ):
            yield chunk


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
    assert wait_for(lambda: sink.stopped_at)
    assert server.requests == [] and local.minors == []  # no route was contacted
    assert out.sentences_spoken == 0 and sink.pushed == []


def test_a_sentence_written_for_a_minor_uses_the_remote_route(make):
    """SPEC 8 (owner decision 2026-10-22): a minor present does not skip the speech route."""
    server = SpeechServer()
    remote, _ = openai_backend(server)
    local = FakeBackend()
    local.name = "piper"
    out = make(FallbackTTS(remote, local), minor_present=lambda: True)
    say(out, minor=True)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert len(server.requests) == 1 and local.minors == []


def test_queued_sentences_are_dropped_when_a_minor_appears_after_the_stream_ended(make):
    minor = {"present": False}
    sink, backend = FakeSink(), FakeBackend(pieces=3, delay=0.05)
    out = make(backend, sink=sink, minor_present=lambda: minor["present"])
    u = out.new_utterance()
    for text in ("Uno.", "Due.", "Tre."):
        say(out, text, utterance=u)
    assert wait_for(lambda: sink.pushed)  # the first sentence is playing, 2 and 3 are queued
    minor["present"] = True
    assert wait_for(lambda: sink.stopped_at)
    time.sleep(0.4)
    assert out.sentences_spoken == 0 and backend.texts == ["Uno."]
    assert out.snapshot()["queued"] == 0


def test_a_minor_appearing_during_a_stall_cuts_the_sentence_in_flight(make):
    minor = {"present": False}
    gate = threading.Event()
    sink, backend = FakeSink(), FakeBackend(pieces=5, delay=0.02)
    out = make(backend, sink=sink, minor_present=lambda: minor["present"])
    original = backend.synthesize

    async def stalled(*a, **k):
        async for chunk in original(*a, **k):
            yield chunk
            while not gate.is_set() and len(sink.pushed) == 1:
                await asyncio.sleep(0.01)  # the LLM side is stalled, the sentence is held

    backend.synthesize = stalled
    say(out, "Uno due tre.")
    assert wait_for(lambda: len(sink.pushed) == 1)
    minor["present"] = True
    gate.set()
    assert wait_for(lambda: sink.stopped_at)
    time.sleep(0.2)
    assert len(sink.pushed) == 1 and out.sentences_spoken == 0


def test_adult_sentence_uses_the_remote_route(make):
    server = SpeechServer()
    remote, _ = openai_backend(server)
    out = make(FallbackTTS(remote, FakeBackend()))
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1) and len(server.requests) == 1
    assert isinstance(server.requests[0], httpx.Request)


def test_factory_picks_backends_and_resolves_voices_from_the_voice_registry(tmp_path):
    from roboemot import paths
    from roboemot.llm import Router
    from roboemot.vision import models as vm
    from roboemot.voice import models as voice_models
    from roboemot.voice.factory import build_backend, piper_python
    from roboemot.voice.piper_backend import PiperBackend, manifest_voice_files

    cfg = RoboEmotConfig()
    router = Router([], cfg.safety)
    assert piper_python(cfg) == paths.piper_python(cfg.data_dir)
    assert voice_models.models_dir(cfg.voice.models_dir, cfg.data_dir).name == "voice-models"
    assert isinstance(build_backend(cfg, lambda: router), PiperBackend)
    remote = RoboEmotConfig.model_validate({"voice": {"backend": "openai_speech"}})
    chain = build_backend(remote, lambda: router)
    assert isinstance(chain, FallbackTTS) and chain.primary.name == "openai_speech"
    assert chain.fallback.name == "piper" and not chain.primary.available()  # no route set
    kokoro = RoboEmotConfig.model_validate({"voice": {"backend": "kokoro"}})
    with pytest.raises(ValueError, match="kokoro"):
        build_backend(kokoro, lambda: router)

    resolve = manifest_voice_files(tmp_path)  # default layout: <voice-models>/piper/<voice>.onnx
    voice = cfg.voice.voices["it"]
    with pytest.raises(voice_models.ModelError):  # nothing fetched
        resolve(voice)
    spec = next(s for s in voice_models.load_manifest().values() if s.name == "piper-it-paola")
    for f in spec.files:
        target = tmp_path / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * f.size_bytes)
    with pytest.raises(voice_models.ModelError):  # present but no verified marker
        resolve(voice)
    for f in spec.files:
        vm._write_marker(tmp_path / f.path, f.sha256)
    model, conf = resolve(voice)
    assert model == tmp_path / "piper" / f"{voice}.onnx" and conf.name == f"{voice}.onnx.json"
    with pytest.raises(voice_models.ModelError):  # a voice outside the registry
        resolve("evil")


def test_config_refuses_unsafe_voice_names_and_relative_interpreters():
    for bad in ("../x", "a/b", "a b", ""):
        with pytest.raises(ValueError):
            RoboEmotConfig.model_validate({"voice": {"voices": {"it": bad}}})
    with pytest.raises(ValueError, match="absolute"):
        RoboEmotConfig.model_validate({"voice": {"piper": {"python": "python3"}}})
    ok = RoboEmotConfig.model_validate({"voice": {"piper": {"python": "/opt/pv/bin/python"}}})
    assert ok.voice.piper.python == "/opt/pv/bin/python"


def test_piper_interpreter_is_admin_only_and_restart():
    from roboemot.config.stack import field_meta

    meta = field_meta("voice.piper.python")
    assert meta.get("admin_only") and meta.get("restart")


def test_stop_then_say_before_the_loop_runs_still_speaks_the_new_sentence(make):
    sink, backend = FakeSink(), FakeBackend(pieces=40, delay=0.05)
    out = make(backend, sink=sink)
    say(out, "Vecchia.")
    assert backend.started.wait(2)
    out.stop()
    say(out, "Nuova.")  # queued right after the stop, before the loop handles the cancel
    assert wait_for(lambda: backend.texts == ["Vecchia.", "Nuova."])
    n = len(sink.pushed)
    assert wait_for(lambda: len(sink.pushed) > n)  # the new sentence is audible
    assert backend.cancelled.is_set()  # the old one was cancelled


def test_no_chunk_is_pushed_after_stop_returns(make):
    class SlowSink(FakeSink):
        def push_audio_sample(self, samples, rate_hz):
            time.sleep(0.03)
            return super().push_audio_sample(samples, rate_hz)

    sink, backend = SlowSink(), FakeBackend(pieces=100, delay=0.0)
    out = make(backend, sink=sink)
    say(out)
    assert wait_for(lambda: len(sink.pushed_at) >= 2)
    out.stop()
    returned = time.monotonic()
    pushed = len(sink.pushed_at)
    time.sleep(0.3)
    assert len(sink.pushed_at) == pushed and max(sink.pushed_at) <= returned
    assert sink.stopped_at and sink.stopped_at[0] >= max(sink.pushed_at)


def test_a_failing_minor_signal_fails_closed(make):
    backend = FakeBackend()

    def broken():
        raise RuntimeError("no signal")

    out = make(backend, minor_present=broken)
    say(out, minor=True)
    assert wait_for(lambda: backend.minors == [True])
    say(out)  # written for adults, and the signal cannot be read: not spoken
    time.sleep(0.2)
    assert out.sentences_spoken == 1


def test_a_failing_warm_up_does_not_kill_the_output(make):
    class Cold(FakeBackend):
        async def warm(self, language, voice):
            raise RuntimeError("cold")

    out = make(Cold())
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)


class TimelineSink(FakeSink):
    """Pushes are faster than real time; ``playing_until`` is where the queue ends, as in the
    Reachy speaker."""

    def __init__(self) -> None:
        super().__init__()
        self.until = 0.0

    def push_audio_sample(self, samples, rate_hz):
        self.until = max(self.until, time.monotonic()) + samples.size / rate_hz
        return super().push_audio_sample(samples, rate_hz)

    def playing_until(self):
        return self.until

    def stop(self):
        self.until = time.monotonic()
        super().stop()


def test_played_is_true_only_for_sentences_that_began_to_play_before_the_stop(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)  # 1 s of audio per sentence
    u = out.new_utterance()
    first, second = say(out, "Uno.", utterance=u), say(out, "Due.", utterance=u)
    third = say(out, "Tre.", utterance=u)
    # all of it is pushed within a few ms of the clock, but only the first second is audible
    assert wait_for(lambda: out.sentences_spoken >= 2 and len(sink.pushed) >= 20)
    out.stop()
    assert out.played(first) is True  # playing when the stop came
    assert out.played(second) is False  # queued in the sink behind it, never heard
    assert out.played(third) is False
    assert out.played(9999) is False


def test_without_a_timeline_every_pushed_sentence_counts_as_played(make):
    out = make(FakeBackend(pieces=1))
    a = say(out, "Uno.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    out.stop()
    assert out.played(a) is True


def test_a_sentence_whose_first_push_lost_the_race_with_a_stop_never_counts_as_played(make):
    holder = {}

    class Racing(FakeBackend):
        async def synthesize(self, text, delivery, *, language, voice, minor_present=False):
            holder["out"].stop()  # a barge-in lands after the sentence started, before its push
            async for chunk in super().synthesize(
                text, delivery, language=language, voice=voice, minor_present=minor_present
            ):
                yield chunk

    sink = TimelineSink()
    out = holder["out"] = make(Racing(), sink=sink)
    job = say(out, "Uno.")
    assert wait_for(lambda: out.sentences_dropped >= 0 and not out._jobs)
    time.sleep(0.2)
    assert sink.pushed == [] and out.played(job) is False and out.ended_at(job) is None


def test_ended_at_follows_the_sinks_timeline_and_a_stop_cuts_it(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)  # 1 s of audio
    job = say(out, "Uno.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.ended_at(job) == pytest.approx(sink.until, abs=0.01)
    assert out.ended_at(job) > time.monotonic() + 0.5  # still playing
    out.stop()
    assert out.ended_at(job) <= time.monotonic()
    assert out.ended_at(9999) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Vuoi un caffe?", True),
        ("Vuoi un caffe? ", True),
        ('Did he say "yes?"', True),
        ("(Are you sure?)", True),
        ("¿Qué tal?", True),
        ("Vuoi un caffe? Ti serve.", False),
        ("Ciao.", False),
        ("Che bello!", False),
        ("", False),
    ],
)
def test_ends_with_question(text, expected):
    assert ends_with_question(text) is expected


def test_question_end_tracks_the_last_spoken_sentence_only(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)
    assert out.question_end() is None
    say(out, "Vuoi un caffe?")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.question_end() == pytest.approx(sink.until, abs=0.01)
    say(out, "Ok, lo preparo.")
    assert wait_for(lambda: out.sentences_spoken == 2)
    assert out.question_end() is None  # a statement after the question ends the window


def test_a_stop_moves_the_question_end_to_the_stop(make):
    out = make(FakeBackend(pieces=10), sink=TimelineSink())
    say(out, "Vuoi un caffe?")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.question_end() > time.monotonic() + 0.5
    out.stop()
    assert out.question_end() <= time.monotonic()


def test_dropped_sentences_neither_replace_nor_extend_the_question_end(make):
    out = make(HoldingBackend(pieces=1), sink=TimelineSink())
    say(out, "Vuoi un caffe?")
    assert wait_for(lambda: out.sentences_spoken == 1)
    before = out.question_end()
    assert before is not None
    for _ in range(3):
        say(out, "Una frase che non verra detta.")
    out.stop()  # drops the queue (barge-in, the stop keyword)
    after = out.question_end()
    assert after is not None and after <= before and after <= time.monotonic()
