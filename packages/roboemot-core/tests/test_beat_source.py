"""SPEC 7 (Dance motion), 7B: the beat source and the hum's beat grid, from the moment the speech
output really starts the sound."""

from __future__ import annotations

import random
import time

import pytest
from roboemot.config.schema import RoboEmotConfig, SootheConfig
from roboemot.motion.beat import Beat, BeatSource
from roboemot.soothe import SootheRunner
from roboemot.soothe.synth import plan_hum, plan_sound, render_plan, synthesize
from roboemot.voice.speech import SpeechOutput
from test_speech import FakeBackend, TimelineSink, wait_for

RATE = 8000
LENGTH_S = 1.0
INTERVAL = 30.0


def variant(sound):
    return {
        "label": "calm",
        "sound": sound,
        "shape": {"length_s": (LENGTH_S, LENGTH_S), "tempo_bpm": (90.0, 90.0)},
    }


def config(sound):
    return SootheConfig(
        interval_s=INTERVAL,
        jitter=0,
        idle_min_s=0,
        cooldown_s=0,
        rate_hz=RATE,
        variants=[variant(sound)],
    )


class Output:
    """A fake output: job 1 is queued at ``queued`` and the sink starts it at ``started``."""

    def __init__(self, started=None):
        self.started = started

    def play(self, samples, rate_hz, volume_db=0.0):
        return 1

    def cancel(self, job_id):
        self.started = None

    def live(self, job_id):
        return True

    def started_at(self, job_id):
        return self.started


def runner(sound, out):
    r = SootheRunner(config(sound), out, on_event=lambda *a: None, rng=random.Random(7))
    r.tick(0.0, True, "awake", False)  # draws the schedule
    r.tick(INTERVAL, True, "awake", False)  # plays
    assert r.job == 1
    return r


def test_the_room_is_asked_only_without_an_own_beat():
    own = Beat(0.5, 1.0, 1.0, "own")
    room = Beat(0.6, 2.0, 0.7, "room")
    assert BeatSource(lambda now: own, lambda now: room).beat_at(0.0) is own
    assert BeatSource(lambda now: None, lambda now: room).beat_at(0.0) is room
    assert BeatSource(lambda now: None, lambda now: None).beat_at(0.0) is None


def test_accents_default_to_none():
    assert BeatSource(lambda now: None, lambda now: None).accents(0.0, 1.0) == []


def test_hum_publishes_grid_from_start_stamp():
    out = Output(started=INTERVAL + 2.5)  # the queue held something: it starts late
    r = runner("hum", out)
    plan = r._plan
    b = r.beat_at(INTERVAL + 2.6)
    assert b is not None and b.source == "own" and b.strength == 1
    assert b.period_s == plan.beat_s
    assert b.phase_s == INTERVAL + 2.5  # the start stamp, not the queue time (INTERVAL)
    assert r.beat_at(INTERVAL + 2.4) is None  # not started yet
    assert r.beat_at(INTERVAL + 2.5 + LENGTH_S) is None  # over


def test_a_hum_not_started_yet_gives_no_beat():
    r = runner("hum", Output(started=None))
    assert r.beat_at(INTERVAL + 1.0) is None


def test_blirp_gives_accents_no_beat():
    out = Output(started=INTERVAL + 1.0)
    r = runner("blirp", out)
    assert r.beat_at(INTERVAL + 1.1) is None
    plan = r._plan
    stamps = [INTERVAL + 1.0 + e.start_s for e in plan.events]
    assert r.accents(0.0, INTERVAL + 10.0) == stamps
    first, second = stamps[0], stamps[1]
    assert r.accents(first, second) == [second]  # (since, now]


def test_a_hum_has_no_accents():
    r = runner("hum", Output(started=INTERVAL))
    assert r.accents(0.0, INTERVAL + 10.0) == []


def test_no_beat_after_the_sound_is_cut():
    out = Output(started=INTERVAL)
    r = runner("hum", out)
    assert r.beat_at(INTERVAL + 0.1) is not None
    r.close()
    assert r.beat_at(INTERVAL + 0.1) is None and r.accents(0.0, INTERVAL + 9.0) == []


def test_the_plan_keeps_the_seeded_sound_and_gives_its_beat():
    """``plan_sound`` then ``render_plan`` is exactly ``synthesize`` for the same seed."""
    shape = config("hum").variants[0].shape
    made = synthesize("hum", shape, random.Random(3), RATE)
    plan = plan_sound("hum", shape, random.Random(3))
    assert (render_plan(plan, RATE) == made).all()
    assert plan.beat_s == pytest.approx(plan.notes[-1].dur_s / 2.0)  # the held tonic is 2 beats
    assert plan_hum(shape, random.Random(3)).beat_s == plan.beat_s


def test_real_output_stamps_the_sink_start_not_the_queue_time():
    sink = TimelineSink()
    out = SpeechOutput(RoboEmotConfig(), sink, FakeBackend())
    out.start()
    try:
        samples = synthesize("hum", config("hum").variants[0].shape, random.Random(1), RATE)
        queued = time.monotonic()
        busy_until = sink.until = queued + 0.3  # the sink still plays 0.3 s of something else
        job = out.play(samples, RATE)
        assert wait_for(lambda: out.started_at(job) is not None, timeout=3.0)
        assert out.started_at(job) >= busy_until  # behind that audio, not at the queue time
    finally:
        out.close()
