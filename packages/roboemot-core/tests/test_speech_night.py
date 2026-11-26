"""The night stage of ``SpeechOutput`` (SPEC 7A, 8): inside the quiet hours every sound gets the
night gain, a job queued at day volume (an announcement) skips it, and the speech timeline gets
the samples before it."""

from __future__ import annotations

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.soothe.synth import db_to_gain
from roboemot.voice.speech import StreamFeed
from test_speech import (
    RATE,
    AsideSink,
    FakeBackend,
    MediaSink,
    PlayingSink,
    blip,
    make,  # noqa: F401  (fixture)
    say,
    wait_for,
)

NIGHT = db_to_gain(-20.0)
NO_PREBUFFER = RoboEmotConfig(voice={"realtime_route": {"prebuffer_ms": 0.0}})


class Night:
    """A fake ``NightVolume.extra_db``: returns ``db`` and records each ``state_applied``."""

    def __init__(self, db: float = -20.0) -> None:
        self.db, self.calls = db, []

    def __call__(self, *, state_applied: bool) -> float:
        self.calls.append(state_applied)
        return self.db


def test_night_gain_on_raw_play(make):  # noqa: F811
    sink = PlayingSink()
    out = make(sink=sink)
    out.night_db = night = Night()
    assert out.play(blip(level=0.5), RATE) is not None
    assert wait_for(lambda: sink.samples)
    assert float(sink.samples[0].max()) == pytest.approx(0.5 * NIGHT, rel=1e-4)
    assert night.calls and all(night.calls)


def test_night_gain_on_speech_and_stream(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink)
    out.night_db = Night()
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert float(np.concatenate(sink.samples).max()) == pytest.approx(NIGHT, rel=1e-4)
    sink.samples.clear()
    feed = StreamFeed()
    out.stream(feed, utterance=out.new_utterance(), mood=None, volume_db=0.0)
    feed.put(blip(level=0.5), RATE)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 2)
    assert float(np.concatenate(sink.samples).max()) == pytest.approx(0.5 * NIGHT, rel=1e-4)


def test_night_gain_on_aside(make):  # noqa: F811
    sink = AsideSink()
    out = make(sink=sink)
    out.night_db = night = Night()
    assert out.play_aside(blip(), RATE)
    assert sink.asides[0][2] == pytest.approx(0.5 * NIGHT, rel=1e-4)
    assert night.calls == [True]


def test_night_gain_on_media_carries_no_state_offset(make):  # noqa: F811
    sink = MediaSink()
    out = make(sink=sink)
    out.night_db = night = Night()
    assert out.play_media(blip(), RATE)
    assert sink.media[0][2] == pytest.approx(0.5 * NIGHT, rel=1e-4)
    assert night.calls == [False]


def test_day_volume_skips_night_gain(make):  # noqa: F811
    sink = PlayingSink()
    out = make(sink=sink)
    out.night_db = Night()
    out.day_volume = True
    assert out.play(blip(level=0.5), RATE) is not None
    assert wait_for(lambda: sink.samples)
    assert float(sink.samples[0].max()) == pytest.approx(0.5)


def test_day_volume_captured_at_queue_time(make):  # noqa: F811
    """A sound queued during an announcement keeps the day volume when it plays after the
    announcement released the flag; the sentence queued before it does not."""
    sink = PlayingSink()
    out = make(FakeBackend(delay=0.2), sink=sink)
    out.night_db = Night()
    say(out)  # holds the queue for 0.2 s
    out.day_volume = True
    assert out.play(blip(level=0.5), RATE) is not None
    out.day_volume = False
    assert wait_for(lambda: len(sink.samples) == 2)
    assert float(sink.samples[0].max()) == pytest.approx(NIGHT, rel=1e-4)
    assert float(sink.samples[1].max()) == pytest.approx(0.5)


def test_night_level_read_per_push(make):  # noqa: F811
    """The window starts in the middle of a reply: the next piece drops to the night level."""
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=NO_PREBUFFER)
    out.night_db = night = Night(0.0)
    feed = StreamFeed()
    out.stream(feed, utterance=out.new_utterance(), mood=None, volume_db=0.0)
    feed.put(blip(level=0.5), RATE)
    assert wait_for(lambda: len(sink.samples) == 1)
    night.db = -20.0
    feed.put(blip(level=0.5), RATE)
    feed.end()
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert float(sink.samples[0].max()) == pytest.approx(0.5)
    assert float(np.concatenate(sink.samples[1:]).max()) == pytest.approx(0.5 * NIGHT, rel=1e-4)


def test_timeline_gets_samples_before_night_gain(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink)
    out.night_db = Night()
    seen = []
    push = out.timeline.push

    def recording(job_id, samples, *args, **kw):
        seen.append(float(np.max(samples)))
        return push(job_id, samples, *args, **kw)

    out.timeline.push = recording
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert seen == [pytest.approx(1.0)]


def test_no_night_stage_by_default(make):  # noqa: F811
    sink = PlayingSink()
    out = make(sink=sink)
    assert out.night_db(state_applied=True) == 0.0 and out.day_volume is False
    samples = blip(level=0.37)
    assert out.play(samples, RATE) is not None
    assert wait_for(lambda: sink.samples)
    assert np.array_equal(sink.samples[0], samples)
    assert out.snapshot()["night_db"] == 0.0
