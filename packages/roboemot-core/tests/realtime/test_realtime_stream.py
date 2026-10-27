"""``SpeechOutput.stream``: audio that arrives already synthesized (the realtime route's reply)
plays through the same queue, epoch and push path as a sentence (SPEC 8, Realtime speech route;
stage 8 barge-in ducking)."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
from roboemot.config.schema import BargeInDuckConfig, RoboEmotConfig
from roboemot.soothe.synth import db_to_gain
from roboemot.voice.duck import DuckGain
from roboemot.voice.speech import StreamFeed
from test_speech import (
    RATE,
    FakeBackend,
    PlayingSink,
    make,  # noqa: F401  (fixture)
    wait_for,
)

LEVEL = 0.5
CHUNK = RATE // 10  # 100 ms


def tone(n: int = CHUNK, level: float = LEVEL) -> np.ndarray:
    return np.full(n, level, np.float32)


def cfg(prebuffer_ms: float = 0.0) -> RoboEmotConfig:
    return RoboEmotConfig(voice={"realtime_route": {"prebuffer_ms": prebuffer_ms}})


def stream(out, feed, volume_db=0.0, utterance=None):
    return out.stream(
        feed,
        utterance=utterance if utterance is not None else out.new_utterance(),
        mood="joy",
        volume_db=volume_db,
        language="it",
    )


def test_a_stream_plays_as_one_sentence_with_the_gain(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg())
    feed = StreamFeed()
    job = stream(out, feed, volume_db=-6.0)
    for _ in range(3):
        assert feed.put(tone(), RATE)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 1)
    played = np.concatenate(sink.samples)
    assert played.size == 3 * CHUNK
    assert played == pytest.approx(LEVEL * db_to_gain(-6.0))
    assert out.played(job) and out.ended_at(job) is not None
    assert out.timeline.snapshot()["sentences"] == 1
    assert out.snapshot()["first_audio_ms_last"] is not None


def test_pcm16_bytes_are_read_as_samples(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg())
    feed = StreamFeed()
    stream(out, feed)
    feed.put((np.full(CHUNK, 16384, "<i2")).tobytes(), RATE)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert np.concatenate(sink.samples) == pytest.approx(0.5)


def test_the_prebuffer_holds_playback_until_enough_audio_arrived(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg(prebuffer_ms=250))
    feed = StreamFeed()
    stream(out, feed)
    feed.put(tone(), RATE)
    feed.put(tone(), RATE)
    time.sleep(0.1)
    assert sink.samples == []  # 200 ms held, 250 needed
    feed.put(tone(), RATE)
    assert wait_for(lambda: len(sink.samples) == 3)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 1)


def test_the_duck_applies_to_a_stream_but_not_to_its_timeline(make):  # noqa: F811
    """ZV ducking sits on the same push path: a realtime reply is ducked too (SPEC 8 stage 8)."""

    def run(duck):
        sink = PlayingSink()
        out = make(FakeBackend(), sink=sink, cfg=cfg())
        out.duck = duck
        feed = StreamFeed()
        stream(out, feed)
        feed.put(tone(), RATE)
        feed.end()
        assert wait_for(lambda: out.sentences_spoken == 1)
        frames = [db for _, db in out.timeline.frames_between(-math.inf, math.inf)]
        return np.concatenate(sink.samples), frames

    duck = DuckGain(BargeInDuckConfig(gain=0.25))
    duck.duck(time.monotonic())
    ducked, ducked_frames = run(duck)
    plain, plain_frames = run(None)
    assert ducked == pytest.approx(0.25 * plain)
    assert ducked_frames and ducked_frames == pytest.approx(plain_frames)


def test_stop_cuts_a_stream_and_its_feed_takes_no_more(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg())
    feed = StreamFeed()
    job = stream(out, feed)
    feed.put(tone(RATE * 2), RATE)  # 2 s
    assert wait_for(lambda: sink.samples)
    time.sleep(0.1)
    out.stop()
    assert wait_for(lambda: feed.closed)
    assert feed.put(tone(), RATE) is False
    played_ms, pushed_ms = out.stream_progress(job)
    assert pushed_ms == 2000 and 0 < played_ms < pushed_ms


def test_a_stream_queued_behind_a_stop_is_dropped_with_its_feed(make):  # noqa: F811
    out = make(FakeBackend(), sink=PlayingSink(), cfg=cfg())
    first, second = StreamFeed(), StreamFeed()
    stream(out, first)  # holds the queue: never ends
    stream(out, second)
    out.stop()
    assert first.closed or wait_for(lambda: first.closed)
    assert second.closed


def test_a_late_question_mark_reaches_question_end_and_the_timeline(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg())
    feed = StreamFeed()
    job = stream(out, feed)
    feed.put(tone(), RATE)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert out.question_end() is None
    out.mark_question(job)  # the transcript came after the audio
    assert out.question_end() == out.ended_at(job)
    with out.timeline._lock:
        assert [s.question for s in out.timeline._sentences] == [True]


def test_feeding_from_another_thread_while_it_plays(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=cfg())
    feed = StreamFeed()
    stream(out, feed)

    def producer():
        for _ in range(5):
            feed.put(tone(), RATE)
            time.sleep(0.01)
        feed.end()

    threading.Thread(target=producer).start()
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert np.concatenate(sink.samples).size == 5 * CHUNK
