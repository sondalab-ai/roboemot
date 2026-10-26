"""SPEC 8, stage 7 / 13: SpeechOutput (queue, persona effects, stop latency, privacy)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import threading
import time

import httpx
import numpy as np
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import BargeInDuckConfig, RegisterParams, RoboEmotConfig, StateEffects
from roboemot.voice.duck import DuckGain
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

    def end_stream(self):
        return 0.0

    def stop(self):
        self.stopped_at.append(time.monotonic())


class FakeBackend:
    name, controls = "fake", frozenset()

    def __init__(self, pieces=1, delay=0.0, fail=False) -> None:
        self.pieces, self.delay, self.fail = pieces, delay, fail
        self.deliveries, self.cancelled = [], threading.Event()
        self.texts: list[str] = []
        self.started = threading.Event()

    def available(self):
        return True

    async def synthesize(self, text, delivery, *, language, voice):
        self.deliveries.append(delivery)
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

    def apply_config(self, voice):
        pass

    async def aclose(self):
        pass


class HoldingBackend(FakeBackend):
    """The first request answers at once; every later one waits (until the stop cancels it), so a
    queued sentence can never reach the sink before the test's own ``stop``."""

    HOLD_S = 30.0

    async def synthesize(self, text, delivery, *, language, voice):
        if self.texts:
            self.texts.append(text)
            await asyncio.sleep(self.HOLD_S)
            return
        async for chunk in super().synthesize(text, delivery, language=language, voice=voice):
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


def test_every_sentence_may_use_the_remote_route(make):
    """Owner decision 2026-10-23: no possible-minor rule keeps a sentence on Piper."""
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
        async def synthesize(self, text, delivery, *, language, voice):
            holder["out"].stop()  # a barge-in lands after the sentence started, before its push
            async for chunk in super().synthesize(text, delivery, language=language, voice=voice):
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


def sound(seconds: float, level: float = 0.5) -> np.ndarray:
    return np.full(round(seconds * RATE), level, np.float32)


def test_play_pushes_the_samples_and_applies_the_circadian_volume(make):
    seen = []

    class RecordingSink(FakeSink):
        def push_audio_sample(self, samples, rate_hz):
            seen.append((float(samples.max()), rate_hz))
            return super().push_audio_sample(samples, rate_hz)

    out = make(sink=RecordingSink())
    assert out.play(sound(0.1), RATE) is not None
    assert out.play(sound(0.1), RATE, volume_db=-6.0) is not None
    assert wait_for(lambda: len(seen) == 2)
    assert seen[0] == (0.5, RATE)
    assert seen[1][0] == pytest.approx(0.5 * 10 ** (-6 / 20), rel=1e-4)


def test_stop_cuts_a_raw_job_playing_and_drops_the_queued_ones(make):
    sink = TimelineSink()
    out = make(sink=sink)
    out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)
    out.play(sound(1), RATE)  # queued behind the first, which holds the queue while it plays
    out.stop()
    assert sink.stopped_at
    time.sleep(0.2)
    assert len(sink.pushed) == 1  # the queued one never played


def test_live_is_true_while_a_raw_job_is_queued_or_playing_and_false_after_cancel_or_stop(make):
    sink = TimelineSink()
    out = make(sink=sink)
    first = out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)
    second = out.play(sound(1), RATE)  # queued behind the first
    assert out.live(first) and out.live(second) and not out.live(999)
    out.cancel(second)
    assert not out.live(second)
    out.stop()
    assert wait_for(lambda: not out.live(first))


def test_waiting_counts_the_queued_jobs_except_the_excluded_one(make):
    sink = TimelineSink()
    out = make(sink=sink)
    first = out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)  # taken off the queue: it holds the output
    assert out.waiting() == 0
    second = out.play(sound(1), RATE)
    assert out.waiting() == 1 and out.waiting(exclude=second) == 0 and out.waiting(first) == 1


def test_cancel_removes_only_its_own_job(make):
    sink = TimelineSink()
    out = make(sink=sink)
    first = out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)
    second, third = out.play(sound(1), RATE), out.play(sound(2), RATE)
    out.cancel(second)  # still queued: removed, the sink is not touched
    assert not sink.stopped_at
    out.cancel(first)  # playing now: the sink is flushed and the queue moves on
    assert sink.stopped_at
    assert wait_for(lambda: len(sink.pushed) == 2)
    assert sink.pushed[1][0] == 2 * RATE  # the third; the second was never pushed
    out.cancel(third)  # whatever its state, cancelling must not raise
    out.cancel(9999)


def test_a_queued_sentence_still_plays_after_a_raw_job_is_cancelled(make):
    sink, backend = TimelineSink(), FakeBackend()
    out = make(backend, sink=sink)
    raw = out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)
    say(out, "Ciao.")
    assert not wait_for(lambda: backend.texts, timeout=0.2)  # held behind the sound
    out.cancel(raw)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert backend.texts == ["Ciao."]


def test_a_raw_job_is_not_speech(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)
    say(out, "Vuoi un caffe?")
    assert wait_for(lambda: out.sentences_spoken == 1)
    before = out.question_end()
    assert before is not None
    job = out.play(sound(0.2), RATE)
    assert wait_for(lambda: len(sink.pushed) == 11)
    assert out.question_end() == before
    assert out.sentences_spoken == 1 and out.ended_at(job) is None and not out.played(job)


def test_a_raw_job_writes_nothing_on_the_speech_timeline(make):
    """SPEC 7, 7B: a soothe sound is not speech, so the speaking motion has no frame, sentence,
    pad or event to react to; a spoken sentence after it still lands on the timeline."""
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)
    out.play(sound(0.5, level=0.9), RATE)
    assert wait_for(lambda: len(sink.pushed) == 1)
    time.sleep(0.1)
    assert out.timeline.snapshot().get("frames", 0) == 0
    assert out.timeline.snapshot().get("sentences", 0) == 0
    assert not out.timeline._pads
    assert out.snapshot()["push_ms"] == {"count": 0}  # push latency covers speech only
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.timeline.snapshot().get("sentences", 0) == 1


def test_nothing_is_pushed_after_cancel_returns(make):
    class SlowSink(TimelineSink):
        def push_audio_sample(self, samples, rate_hz):
            time.sleep(0.05)
            return super().push_audio_sample(samples, rate_hz)

    sink = SlowSink()
    out = make(sink=sink)
    job = out.play(sound(20), RATE)
    assert wait_for(lambda: out._raw_current == job)  # the loop took it, around the slow push
    out.cancel(job)
    returned = time.monotonic()
    pushed = len(sink.pushed_at)
    assert pushed == 1  # the cancel waited for the push under the lock, then flushed it
    time.sleep(0.3)
    assert len(sink.pushed_at) == pushed and all(at <= returned for at in sink.pushed_at)
    assert sink.stopped_at[0] >= sink.pushed_at[-1]
    assert out.play(sound(0.1), RATE) is not None  # the output is still usable
    assert wait_for(lambda: len(sink.pushed) == pushed + 1)


def test_cancel_before_the_push_does_not_flush_the_speech_still_in_the_sink(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)  # 1 s of speech audio
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    spoken = len(sink.pushed)
    raw = out.play(sound(5), RATE)
    assert wait_for(lambda: out._raw_current == raw)  # waiting for the speech to play out
    out.cancel(raw)
    assert not sink.stopped_at
    time.sleep(0.2)
    assert len(sink.pushed) == spoken  # the cancelled sound never reached the sink
    out.play(sound(0.1), RATE)
    assert wait_for(lambda: len(sink.pushed) == spoken + 1, timeout=3)  # after the speech played


def test_a_raw_job_starts_only_after_earlier_audio_has_played(make):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=10), sink=sink)
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    start = time.monotonic()
    out.play(sound(0.1), RATE)
    assert wait_for(lambda: len(sink.pushed) == 11, timeout=3)
    assert time.monotonic() - start > 0.5


def test_a_stale_or_finished_id_is_a_no_op_while_the_sink_is_mid_play(make):
    sink = TimelineSink()
    out = make(sink=sink)
    done = out.play(sound(0.05), RATE)
    assert wait_for(lambda: out._raw_current is None and len(sink.pushed) == 1)
    live = out.play(sound(20), RATE)
    assert wait_for(lambda: len(sink.pushed) == 2)
    out.cancel(done)
    out.cancel(live + 100)
    assert not sink.stopped_at
    out.cancel(live)
    assert len(sink.stopped_at) == 1


@pytest.mark.parametrize("ahead", [False, True])
def test_close_returns_promptly_while_a_raw_job_waits_or_plays(make, ahead):
    sink = TimelineSink()
    out = make(FakeBackend(pieces=30), sink=sink)  # 3 s of speech audio
    if ahead:
        say(out, "Ciao.")
        assert wait_for(lambda: out.sentences_spoken == 1)
    out.play(sound(20), RATE)
    assert wait_for(lambda: out._raw_current is not None)
    start = time.monotonic()
    out.close()
    assert time.monotonic() - start < 1.0


def test_a_cancel_racing_the_natural_end_never_flushes_the_next_job(make):
    class Sink(TimelineSink):
        def __init__(self) -> None:
            super().__init__()
            self.queue: list[tuple[float, int]] = []
            self.flushed: list[int] = []

        def push_audio_sample(self, samples, rate_hz):
            super().push_audio_sample(samples, rate_hz)
            self.queue.append((self.until, samples.size))
            return 0.0

        def stop(self):
            now = time.monotonic()
            self.flushed += [size for end, size in self.queue if end > now]
            self.queue = []
            super().stop()

    sink, backend = Sink(), FakeBackend()
    out = make(backend, sink=sink)
    rng = random.Random(0)
    for i in range(25):
        raw = out.play(sound(0.03), RATE)
        say(out, "Ciao.")
        time.sleep(rng.uniform(0.0, 0.05))
        out.cancel(raw)
        assert wait_for(lambda i=i: out.sentences_spoken == i + 1)
    assert RATE // 10 not in sink.flushed  # a speech chunk was never in a flush


def test_a_reload_reaches_the_backend(make):
    class Configurable(FakeBackend):
        def apply_config(self, voice):
            self.applied = voice

    backend = Configurable()
    out = make(backend)
    cfg = RoboEmotConfig(conversation={"language": "en"})
    out.apply_config(cfg)
    assert out.config is cfg and backend.applied is cfg.voice


# ------------------------------------------------- report counters (SPEC 12A)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ClockedSink(FakeSink):
    """Timeline on a fake clock, like the Reachy speaker's ``playing_until``."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock, self.until = clock, 0.0

    def push_audio_sample(self, samples, rate_hz):
        self.until = max(self.until, self.clock()) + samples.size / rate_hz
        return super().push_audio_sample(samples, rate_hz)

    def playing_until(self):
        return self.until


class ClockedBackend(FakeBackend):
    """Request n yields one 100 ms chunk per value of ``synth_s[n]``, each taking that long to
    synthesize on the fake clock."""

    def __init__(self, clock: FakeClock, synth_s: list[list[float]]) -> None:
        super().__init__()
        self.clock, self.synth_s = clock, list(synth_s)

    async def synthesize(self, text, delivery, *, language, voice):
        for took in self.synth_s.pop(0):
            self.clock.now += took
            yield AudioChunk(np.ones(RATE // 10, np.float32), RATE)


def test_speech_reports_tts_rtf_and_underruns_within_a_sentence(make):
    clock = FakeClock()
    sink = ClockedSink(clock)
    # 100 ms chunks: the third takes 250 ms, so the speaker runs dry for 100 ms before it
    out = make(ClockedBackend(clock, [[0.05, 0.05, 0.25, 0.05]]), sink=sink, clock=clock)
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    snap = out.snapshot()
    rtf = snap["tts_rtf"]
    assert rtf["count"] == 4
    assert rtf["p50"] == pytest.approx(0.5) and rtf["max"] == pytest.approx(2.5)
    assert snap["underruns"] == 1 and snap["underrun_sentences"] == 1
    assert snap["underrun_ms_total"] == pytest.approx(100.0)
    # the sentence waited 0.4 s in all for 0.4 s of audio
    sentence = snap["tts_rtf_sentence"]
    assert sentence["count"] == 1 and sentence["max"] == pytest.approx(1.0)
    json.dumps(snap)


def test_the_gap_before_a_sentence_and_its_pause_is_not_an_underrun(make):
    clock = FakeClock()
    sink = ClockedSink(clock)
    # two one-chunk sentences of one answer; the second arrives long after the first ended
    out = make(ClockedBackend(clock, [[0.05], [0.5]]), sink=sink, clock=clock)
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    say(out, "Due.", utterance=u)
    assert wait_for(lambda: out.sentences_spoken == 2)
    snap = out.snapshot()
    assert snap["underruns"] == 0 and snap["underrun_ms_total"] == 0.0
    assert snap["tts_rtf"]["count"] == 2


def test_report_window_bounds_the_rtf_summary(make):
    clock = FakeClock()
    cfg = RoboEmotConfig(voice={"report_window": 10})
    out = make(ClockedBackend(clock, [[0.01] * 15]), cfg=cfg, sink=ClockedSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.snapshot()["tts_rtf"]["count"] == 10
    assert RoboEmotConfig().voice.report_window == 200


class DeadSink(ClockedSink):
    """A speaker whose output never opened: nothing is queued and ``playing_until`` stays -inf."""

    def push_audio_sample(self, samples, rate_hz):
        return 0.0

    def playing_until(self):
        return -math.inf


def test_a_dead_sink_counts_no_underruns_and_no_late_sentences(make):
    clock = FakeClock()
    out = make(ClockedBackend(clock, [[0.05, 0.25], [0.5]]), sink=DeadSink(clock), clock=clock)
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    say(out, "Due.", utterance=u)
    assert wait_for(lambda: out.sentences_spoken == 2)
    snap = out.snapshot()
    assert snap["underruns"] == 0 and snap["underrun_ms_total"] == 0.0
    assert snap["late_sentences"] == 0 and snap["late_ms_total"] == 0.0
    json.dumps(snap, allow_nan=False)


def test_underrun_sentences_counts_each_sentence_that_ran_dry_once(make):
    clock = FakeClock()
    sink = ClockedSink(clock)
    out = make(ClockedBackend(clock, [[0.05, 0.25, 0.3], [0.05, 0.25]]), sink=sink, clock=clock)
    say(out, "Uno.")  # two answers: no pause between them
    say(out, "Due.")
    assert wait_for(lambda: out.sentences_spoken == 2)
    snap = out.snapshot()
    # first: dry 150 ms then 200 ms; second: queued behind the first, then dry 100 ms
    assert snap["underruns"] == 3 and snap["underrun_sentences"] == 2
    assert snap["underrun_ms_total"] == pytest.approx(450.0)


def test_a_push_refused_after_a_stop_counts_no_underrun(make):
    clock = FakeClock()
    holder = {}

    class Stopping(ClockedBackend):
        async def synthesize(self, text, delivery, *, language, voice):
            n = 0
            async for chunk in super().synthesize(text, delivery, language=language, voice=voice):
                if n == 1:
                    holder["out"].stop()  # barge-in after the sink ran dry
                n += 1
                yield chunk

    out = holder["out"] = make(Stopping(clock, [[0.05, 0.5]]), sink=ClockedSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: out._current is not None and out._current.done())
    snap = out.snapshot()
    assert snap["underruns"] == 0 and snap["tts_rtf_sentence"] == {"count": 0}


def test_a_sentence_later_than_its_pause_is_counted_late(make):
    clock = FakeClock()
    cfg = RoboEmotConfig(voice={"sentence_pause_ms": 0})
    sink = ClockedSink(clock)
    out = make(ClockedBackend(clock, [[0.05], [0.5]]), cfg=cfg, sink=sink, clock=clock)
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    say(out, "Due.", utterance=u)
    assert wait_for(lambda: out.sentences_spoken == 2)
    snap = out.snapshot()
    # the first ends at 0.15, the second arrives at 0.55 with no pause planned
    assert snap["late_sentences"] == 1 and snap["late_ms_total"] == pytest.approx(400.0)
    assert snap["boundary_dry"] == 1


def test_a_sentence_inside_its_pause_is_not_late(make):
    clock = FakeClock()
    cfg = RoboEmotConfig(voice={"sentence_pause_ms": 500})
    out = make(
        ClockedBackend(clock, [[0.05], [0.1]]), cfg=cfg, sink=ClockedSink(clock), clock=clock
    )
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    say(out, "Due.", utterance=u)
    assert wait_for(lambda: out.sentences_spoken == 2)
    snap = out.snapshot()
    assert snap["late_sentences"] == 0 and snap["late_ms_total"] == 0.0
    assert snap["boundary_dry"] == 0


def test_push_ms_times_each_push_into_the_sink(make):
    clock = FakeClock()

    class SlowSink(ClockedSink):
        def push_audio_sample(self, samples, rate_hz):
            clock.now += 0.003
            return super().push_audio_sample(samples, rate_hz)

    out = make(ClockedBackend(clock, [[0.05, 0.05]]), sink=SlowSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    push = out.snapshot()["push_ms"]
    assert push["count"] == 2 and push["max"] == pytest.approx(3.0)


class StreamSink(ClockedSink):
    """A speaker that resamples as a stream: ``end_stream`` flushes a 10 ms tail."""

    TAIL_S = 0.01

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.ends = 0

    def end_stream(self):
        self.ends += 1
        self.until = max(self.until, self.clock()) + self.TAIL_S
        return self.until


def test_each_finished_sentence_ends_the_sinks_stream_and_its_tail_counts(make):
    clock = FakeClock()
    sink = StreamSink(clock)
    out = make(ClockedBackend(clock, [[0.05, 0.05], [0.05]]), sink=sink, clock=clock)
    first = say(out, "Vero?")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert sink.ends == 1
    assert out.ended_at(first) == pytest.approx(sink.until)  # the flushed tail included
    assert out.question_end() == pytest.approx(sink.until)


def test_a_failing_sentence_still_ends_the_stream_but_a_stopped_one_does_not(make):
    clock = FakeClock()
    sink = StreamSink(clock)

    class FailsMidway(ClockedBackend):
        async def synthesize(self, text, delivery, *, language, voice):
            async for chunk in super().synthesize(text, delivery, language=language, voice=voice):
                yield chunk
            raise TTSError("boom")

    out = make(FailsMidway(clock, [[0.05]]), sink=sink, clock=clock)
    say(out)
    assert wait_for(lambda: out.failures == 1)
    assert wait_for(lambda: sink.ends == 1)
    holder = {}

    class Stopping(ClockedBackend):
        async def synthesize(self, text, delivery, *, language, voice):
            async for chunk in super().synthesize(text, delivery, language=language, voice=voice):
                yield chunk
                holder["out"].stop()

    sink2 = StreamSink(clock)
    out2 = holder["out"] = make(Stopping(clock, [[0.05, 0.05]]), sink=sink2, clock=clock)
    say(out2)
    assert wait_for(lambda: out2._current is not None and out2._current.done())
    assert sink2.ends == 0  # the stop already reset the sink's stream


def test_sentences_of_one_reply_vary_by_index_and_a_question_is_raised(make):
    """SPEC 8, variation: the reply's id seeds each sentence's offsets, so the same reply gets the
    same delivery again; a question gets the configured raise on top."""
    from roboemot.persona import effects
    from roboemot.voice.prosody import compose_delivery

    backend = FakeBackend()
    out = make(backend)
    u = out.new_utterance()
    for text in ("Uno.", "Due.", "Tre?"):
        say(out, text, utterance=u)
    assert wait_for(lambda: len(backend.deliveries) == 3)
    cfg = out.config

    def expected(index, question):
        return compose_delivery(
            cfg.voice, register=REG, circadian=CIRC, mood=None, intensity=0.0,
            persona=effects(cfg.persona), seed=u, index=index, question=question,
        )  # fmt: skip

    assert backend.deliveries == [expected(0, False), expected(1, False), expected(2, True)]
    assert len({d.pitch_st for d in backend.deliveries}) == 3


def test_the_pieces_of_a_cut_sentence_share_one_delivery(make):
    """Review S3: a continuation reuses the first piece's delivery (index, variation and the
    question raise, which only the full sentence could tell), and the next sentence goes on."""
    backend = FakeBackend()
    out = make(backend)
    u = out.new_utterance()
    say(out, "Una frase lunga, molto lunga,", utterance=u)
    say(out, "che finisce con una domanda?", utterance=u, continued=True)
    say(out, "Poi un'altra.", utterance=u)
    assert wait_for(lambda: len(backend.deliveries) == 3)
    head, tail, nxt = backend.deliveries
    assert tail == head
    assert nxt != head


def test_a_dropped_sentence_does_not_repeat_a_delivery_of_its_reply(make):
    backend = HoldingBackend()
    cfg = RoboEmotConfig.model_validate({"voice": {"queue_max_sentences": 1}})
    out = make(backend, cfg=cfg)
    u = out.new_utterance()
    say(out, "Uno.", utterance=u)
    assert wait_for(lambda: backend.started.is_set())
    assert say(out, "Due.", utterance=u) is not None  # queued
    assert say(out, "Tre.", utterance=u) is None  # dropped: queue full
    out.stop()
    later = say(out, "Quattro.", utterance=u)
    assert later is not None
    assert out._said[u][0] == 4  # the drop consumed index 2, the sequence goes on


# ------------------------------------------------------------------ barge-in ducking (SPEC 8)


class PlayingSink(FakeSink):
    """Keeps the samples and a playback timeline, as the robot's speaker does."""

    def __init__(self) -> None:
        super().__init__()
        self.samples: list[np.ndarray] = []
        self.until = -math.inf

    def push_audio_sample(self, samples, rate_hz):
        self.samples.append(np.array(samples, np.float32))
        self.until = max(self.until, time.monotonic()) + samples.size / rate_hz
        return super().push_audio_sample(samples, rate_hz)

    def playing_until(self):
        return self.until


def spoken_with(make, duck):
    sink = PlayingSink()
    out = make(FakeBackend(pieces=3), sink=sink)
    out.duck = duck
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    frames = [db for _, db in out.timeline.frames_between(-math.inf, math.inf)]
    return np.concatenate(sink.samples), frames


def test_a_duck_lowers_what_the_sink_gets_but_not_the_speech_timeline(make):
    duck = DuckGain(BargeInDuckConfig(gain=0.25))
    duck.duck(0.0)  # every sample plays inside the episode
    ducked, ducked_frames = spoken_with(make, duck)
    plain, plain_frames = spoken_with(make, None)
    assert ducked == pytest.approx(0.25 * plain)
    # the speaking motion reads the timeline: it sees the sentence as spoken, not the duck
    assert ducked_frames and ducked_frames == pytest.approx(plain_frames)


def test_stop_clears_the_duck_so_the_next_answer_plays_at_full_gain(make):
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink)
    out.duck = DuckGain(BargeInDuckConfig(gain=0.25))
    out.duck.duck(0.0)
    out.stop()
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert np.concatenate(sink.samples) == pytest.approx(1.0)


def test_a_soothe_sound_through_play_is_ducked_too(make):
    """SPEC 7B, 8 stage 8: a raw sound goes through the same push, so a candidate ducks it."""
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink)
    out.duck = DuckGain(BargeInDuckConfig(gain=0.25))
    out.duck.duck(0.0)
    assert out.play(np.full(RATE // 10, 0.5, np.float32), RATE) is not None
    assert wait_for(lambda: sink.samples)
    assert np.concatenate(sink.samples) == pytest.approx(0.125)


def test_a_cut_off_during_a_large_ducked_push_waits_only_for_that_push(make, monkeypatch):
    """The duck runs under the output lock with the push: a stop from another thread waits at
    most for the push in flight, and nothing is pushed after it returns."""
    sink = PlayingSink()
    out = make(FakeBackend(pieces=3), sink=sink)
    duck = DuckGain(BargeInDuckConfig(gain=0.25))
    out.duck = duck
    duck.duck(0.0)
    inside, hold_s = threading.Event(), 0.1
    real = duck.apply

    def slow_apply(samples, rate_hz, start):
        inside.set()
        time.sleep(hold_s)
        return real(samples, rate_hz, start)

    monkeypatch.setattr(duck, "apply", slow_apply)
    say(out, "Ciao.")
    assert inside.wait(2.0)
    began = time.monotonic()
    out.stop()
    waited = time.monotonic() - began
    pushed = len(sink.samples)
    time.sleep(0.3)
    assert waited < hold_s + 0.1  # one push in flight, not the sentence
    assert len(sink.samples) == pushed  # nothing after the stop


class TakingSink(PlayingSink):
    """A sink that applies the duck itself (a pacing speaker)."""

    def __init__(self) -> None:
        super().__init__()
        self.taken = None

    def take_duck(self, duck):
        self.taken = duck
        return True

    def pacing(self):
        return {"lead_ms": 200.0}


def test_a_sink_that_takes_the_duck_gets_the_samples_unscaled_and_reports_its_pacing(make):
    sink = TakingSink()
    out = make(FakeBackend(), sink=sink)
    duck = DuckGain(BargeInDuckConfig(gain=0.25))
    out.duck = duck
    duck.duck(0.0)
    assert sink.taken is duck
    say(out, "Ciao.")
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert np.concatenate(sink.samples) == pytest.approx(1.0)  # never applied twice
    assert out.snapshot()["pacing"] == {"lead_ms": 200.0}
