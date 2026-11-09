"""SPEC 7 (Dance motion), 7B: the beat source and the hum's beat grid, from the moment the speech
output really starts the sound."""

from __future__ import annotations

import random
import time
from types import SimpleNamespace

import pytest
from roboemot.config.schema import RoboEmotConfig, SootheConfig
from roboemot.motion.beat import Beat, BeatSource, room_beat
from roboemot.runtime.assembly import dance_beat, speech_playing
from roboemot.soothe import SootheRunner
from roboemot.soothe.synth import plan_hum, plan_sound, render_plan, synthesize
from roboemot.voice.music import MusicState
from roboemot.voice.speech import SpeechOutput, _Job, _RawJob
from test_soothe_runner import RealRig
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


def test_no_beat_right_after_a_barge_in_stop():
    """``stop`` drops the sound; the beat goes at once, before the runner's next tick."""
    rig = RealRig()
    try:
        job = rig.start_sound()
        assert wait_for(lambda: rig.out.started_at(job) is not None)
        at = rig.out.started_at(job) + 0.01
        assert rig.runner.beat_at(at) is not None
        rig.out.stop()
        assert wait_for(lambda: not rig.out.live(job))
        assert rig.runner.beat_at(at) is None and rig.runner.job == job
    finally:
        rig.out.close()


# ------------------------------------------------------------------ room music (SPEC 7, 8)


class Music:
    """A detector stand-in: a settable state with the time since the last beat."""

    def __init__(self, bpm=120.0, anchor=0.0, confidence=0.8, playing=True) -> None:
        self.bpm, self.anchor, self.confidence, self.playing = bpm, anchor, confidence, playing

    def state(self, now):
        if not self.playing:
            return MusicState(False, None, None, self.confidence, now)
        since = (now - self.anchor) % (60.0 / self.bpm)
        return MusicState(True, self.bpm, since, self.confidence, now)


def test_own_beat_wins_over_room():
    own = Beat(0.5, 0.0, 1.0, "own")
    music = Music()
    source = BeatSource(lambda now: own, lambda now: room_beat(music.state(now)))
    assert source.beat_at(1.0) is own
    room_only = BeatSource(lambda now: None, lambda now: room_beat(music.state(now)))
    assert room_only.beat_at(1.0).source == "room"


def test_room_beat_maps_the_detector_state():
    got = room_beat(Music(bpm=120.0, confidence=0.7).state(10.1))
    assert got is not None
    assert got.period_s == pytest.approx(0.5)
    assert got.strength == pytest.approx(0.7)
    assert got.source == "room"
    assert got.phase_s == pytest.approx(10.0)


@pytest.mark.parametrize(
    "state",
    [
        None,
        MusicState(False, None, None, 0.9, 1.0),
        MusicState(True, None, 0.1, 0.9, 1.0),
        MusicState(True, 120.0, None, 0.9, 1.0),
    ],
)
def test_room_beat_is_none_without_music(state):
    assert room_beat(state) is None


# ------------------------------------------------------------------ wiring (SPEC 7)


def test_assembly_without_a_listener_gives_no_room_beat():
    host = SimpleNamespace(voice_listener=None, soothe_runner=None)
    assert dance_beat(host).beat_at(1.0) is None


def test_assembly_reads_the_listener_late():
    host = SimpleNamespace(voice_listener=None, soothe_runner=None)
    source = dance_beat(host)
    music = Music(bpm=120.0)
    host.voice_listener = SimpleNamespace(music_state=music.state)
    assert source.beat_at(1.0).source == "room"
    host.voice_listener = SimpleNamespace(music_state=lambda now: None)  # no detector
    assert source.beat_at(1.0) is None


def test_assembly_without_speech_output_is_not_speaking():
    host = SimpleNamespace(speech_output=None)
    assert speech_playing(host)(1.0) is False
    host.speech_output = SimpleNamespace(speaking_at=lambda now: True)
    assert speech_playing(host)(1.0) is True


def test_speaking_at_covers_the_gap_between_sentences_and_not_a_raw_sound():
    out = SpeechOutput(RoboEmotConfig(), TimelineSink(), FakeBackend())
    assert not out.speaking_at(1.0)
    out._jobs.append(object.__new__(_RawJob))
    assert not out.speaking_at(1.0)  # a queued hum is not speech
    out._jobs.clear()
    out._jobs.append(object.__new__(_Job))
    assert out.speaking_at(1.0)  # the next sentence waits: the gap counts
    out._jobs.clear()
    out._in_flight = True
    assert out.speaking_at(1.0)  # a sentence being synthesized
    out._raw_current = 7
    assert not out.speaking_at(1.0)  # the job in flight is the hum
    out._in_flight, out._raw_current = False, None
    out.timeline._sentence_at = lambda t: object()
    assert out.speaking_at(1.0)  # the last sentence still plays out of the sink
