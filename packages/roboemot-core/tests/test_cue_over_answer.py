"""S3: a cue never delays the answer's first audio and never plays over it (SPEC 8, Non-verbal
cues). The listening blip still in the speaker is cut by the answer's first push; a one-shot
cue's sound (a raw job) gives way to an answer queued behind it; a self-soothing sound does
not; a listening cue begun after its utterance ended is ignored."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.speech import StreamFeed
from test_cues import make as make_cues
from test_speech import RATE, FakeBackend, PlayingSink, make, say, wait_for  # noqa: F401

BLIP_S = 0.15  # shorter than voice.vad.min_speech_ms (200 ms), long enough to measure
CUE_S = 0.6  # a one-shot cue sound (0.5 to 0.7 s on Reachy Mini)
EPS_S = 0.05


class AsideSink(PlayingSink):
    """A speaker that plays the blip like any audio (it occupies the player) and can cut it."""

    def __init__(self) -> None:
        super().__init__()
        self.asides: list[int] = []
        self.aside_until = -math.inf
        self.cuts: list[float] = []
        self.cut_calls = 0
        self.begins: list[float] = []  # where each speech push starts to play

    def push_aside(self, samples, rate_hz):
        self.asides.append(samples.size)
        self.until = max(self.until, time.monotonic()) + samples.size / rate_hz
        self.aside_until = self.until
        return self.until

    def cut_aside(self):
        self.cut_calls += 1
        now = time.monotonic()
        if self.aside_until <= now:
            return False
        self.cuts.append(now)
        self.until, self.aside_until = now, -math.inf
        return True

    def push_audio_sample(self, samples, rate_hz):
        self.begins.append(max(self.until, time.monotonic()))
        return super().push_audio_sample(samples, rate_hz)


def blip(seconds=BLIP_S):
    return np.full(round(seconds * RATE), 0.2, np.float32)


def stream_answer(out, pieces=2):
    feed = StreamFeed()
    job = out.stream(feed, utterance=out.new_utterance(), mood=None, volume_db=0.0)
    for _ in range(pieces):
        feed.put(np.full(RATE // 10, 0.5, np.float32), RATE)
    feed.end()
    return job


def no_prebuffer():
    return RoboEmotConfig(voice={"realtime_route": {"prebuffer_ms": 0}})


# ------------------------------------------------------------------ the listening blip


@pytest.mark.parametrize("answer", ["stream", "sentence"])
def test_the_answers_first_audio_cuts_a_blip_still_playing(make, answer):  # noqa: F811
    sink = AsideSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    assert out.play_aside(blip(), RATE)
    t0 = time.monotonic()
    if answer == "stream":
        stream_answer(out)
    else:
        say(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert len(sink.cuts) == 1 and sink.cuts[0] <= sink.pushed_at[0]
    # the answer starts where it is pushed, not behind the blip's rest
    assert sink.begins[0] - sink.pushed_at[0] < EPS_S
    assert sink.begins[0] < t0 + BLIP_S
    assert out.snapshot()["aside"] == {"played": 1, "cut": 1}


def test_a_blip_that_already_played_is_not_cut(make):  # noqa: F811
    sink = AsideSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    assert out.play_aside(blip(0.05), RATE)
    time.sleep(0.08)
    stream_answer(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert sink.cuts == [] and out.snapshot()["aside"] == {"played": 1}


def test_a_blip_is_cut_once_not_at_every_push(make):  # noqa: F811
    sink = AsideSink()
    out = make(FakeBackend(pieces=3), sink=sink, cfg=no_prebuffer())
    assert out.play_aside(blip(), RATE)
    say(out)
    say(out)
    assert wait_for(lambda: out.sentences_spoken == 2)
    assert len(sink.cuts) == 1


class SlowAsideSink(AsideSink):
    """The aside push is held: an answer is pushed between the output's decision and it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def push_aside(self, samples, rate_hz):
        self.entered.set()
        assert self.release.wait(5.0)
        return super().push_aside(samples, rate_hz)


def test_an_answer_pushed_while_the_blip_is_being_pushed_still_cuts_it(make):  # noqa: F811
    """Review of #200: between ``play_aside``'s decision and its push the output must already
    treat the blip as playing, so an answer pushed then asks the sink to cut it."""
    sink = SlowAsideSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    played: list[bool] = []
    t = threading.Thread(target=lambda: played.append(out.play_aside(blip(), RATE)))
    t.start()
    assert sink.entered.wait(5.0)
    stream_answer(out)
    assert wait_for(lambda: sink.pushed)
    assert sink.cut_calls == 1  # the answer asked for the cut although the blip was not in yet
    sink.release.set()
    t.join(5.0)
    assert out._aside_until < math.inf  # narrowed after the push


# ------------------------------------------------------------------ one-shot cue sounds


def cue_sound():
    return np.full(round(CUE_S * RATE), 0.3, np.float32)


def test_a_cue_sound_playing_gives_way_to_the_answer_at_once(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    job = out.play(cue_sound(), RATE, yields=True)
    assert wait_for(lambda: sink.pushed)  # the cue is in the speaker
    t0 = time.monotonic()
    stream_answer(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert sink.stopped_at and sink.stopped_at[0] - t0 < EPS_S  # the cue was flushed
    assert sink.pushed_at[1] - t0 < CUE_S / 2  # the answer did not wait for its end
    assert not out.live(job)
    assert out.snapshot()["cue_sounds_cut"] == 1


def test_a_queued_cue_sound_is_dropped_when_an_answer_is_queued(make):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    first = out.play(cue_sound(), RATE, yields=True)
    second = out.play(cue_sound(), RATE, yields=True)
    assert wait_for(lambda: sink.pushed)
    stream_answer(out)
    assert wait_for(lambda: out.sentences_spoken == 1)
    assert not out.live(first) and not out.live(second)
    cues_pushed = [n for n, _ in sink.pushed if n == cue_sound().size]
    assert len(cues_pushed) == 1  # the first cue, then the answer; the second never played
    assert out.snapshot()["cue_sounds_cut"] == 2


@pytest.mark.parametrize("answer", ["stream", "sentence"])
def test_a_self_soothing_sound_is_not_a_cue_and_keeps_its_place(make, answer):  # noqa: F811
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    out.play(np.full(round(0.2 * RATE), 0.3, np.float32), RATE)
    assert wait_for(lambda: sink.pushed)
    t0 = time.monotonic()
    if answer == "stream":
        stream_answer(out)
    else:
        say(out)
    assert wait_for(lambda: out.sentences_spoken == 1, 3.0)
    assert sink.stopped_at == [] and sink.pushed_at[1] - t0 > 0.1  # it waited behind it
    assert out.snapshot()["cue_sounds_cut"] == 0


def test_a_self_soothing_sound_queued_behind_a_cue_sound_does_not_cut_it(make):  # noqa: F811
    """Only an answer cuts a cue's sound; another raw sound waits behind it as before."""
    sink = PlayingSink()
    out = make(FakeBackend(), sink=sink, cfg=no_prebuffer())
    cue = out.play(cue_sound(), RATE, yields=True)
    assert wait_for(lambda: sink.pushed)
    t0 = time.monotonic()
    out.play(np.full(round(0.1 * RATE), 0.3, np.float32), RATE)
    assert wait_for(lambda: len(sink.pushed) == 2, 3.0)
    assert sink.stopped_at == [] and sink.pushed_at[1] - t0 > CUE_S / 2  # after the cue
    assert out.snapshot()["cue_sounds_cut"] == 0 and not out.live(cue)


def test_the_cues_play_their_one_shot_sounds_as_sounds_that_give_way():
    cues, _, out, _ = make_cues()
    assert cues.play("fallback")
    assert out.yields == [True]


# ------------------------------------------------------------------ the listening cue's keys


def test_a_listening_cue_begun_after_its_utterance_ended_is_ignored():
    """The server's answer may end an utterance before the listener begins its cue (a barge-in's
    stream, S2): that cue would hold until the hold limit over the answer."""
    cues, clock, out, state = make_cues()
    cues.listening_end("u1")
    cues.listening_begin("u1")
    assert out.asides == []  # no blip
    clock.t += 1.0
    assert cues.commands(clock.t) == []  # no gesture
    counts = cues.snapshot()["counts"]["listening"]
    assert counts == {"late_end": 1, "begin_after_end": 1}


@pytest.mark.parametrize("spoken", [True, False])
def test_the_fallback_cue_never_plays_with_its_own_spoken_line(spoken):
    """Review of #200: a cue gives way to an answer, so no cue may be meant to play together
    with its own line. The worker shows ``cue`` only when no voice said the line (here the
    fallback line in a local turn): one or the other, never both."""
    from types import SimpleNamespace

    from roboemot.conversation.worker import ConversationWorker

    said, cued = [], []
    fake = SimpleNamespace(
        _say_local=lambda turn, text, mood, intensity, continued=False: said.append(text) or spoken,
        _cue=lambda turn, cue: cued.append(cue),
    )
    turn = SimpleNamespace(voice="local", interrupted=False, lines=[])
    ConversationWorker._say(fake, turn, "Non riesco a pensare.", None, 0.0, cue="fallback")
    assert said == ["Non riesco a pensare."]
    assert cued == ([] if spoken else ["fallback"])


def test_a_begin_for_a_key_evicted_past_the_memory_still_starts_no_cue():
    """Review of #200: the ended keys are a bounded memory (``ENDED_KEYS_MAX``); keys are the
    utterances' start stamps, so a begin at or before an evicted one is just as late."""
    from roboemot.voice.cues import ENDED_KEYS_MAX

    cues, clock, out, _ = make_cues()
    for i in range(ENDED_KEYS_MAX + 5):
        cues.listening_end(float(i))
    cues.listening_begin(0.0)  # evicted from the memory
    clock.t += 1.0
    assert cues.commands(clock.t) == [] and out.asides == []
    assert cues.snapshot()["counts"]["listening"]["begin_after_end"] == 1
    cues.listening_begin(1000.0)  # a new utterance still begins
    assert out.asides
