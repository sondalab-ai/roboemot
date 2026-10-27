"""SPEC 8 (design ZT slice 1): the speech timeline written from the robot's own TTS output."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig, SpeechMotionConfig
from roboemot.voice.envelope import SpeechEvent, SpeechSentence, SpeechTimeline
from roboemot.voice.speech import SpeechOutput
from roboemot.voice.tts import AudioChunk
from test_speech import RATE, ClockedSink, FakeBackend, FakeClock, say, wait_for

TONE_HZ = 250  # whole cycles in a 20 ms frame, so a steady tone gives steady frames
CHUNKINGS = [None, 337]  # one whole-sentence chunk (Piper); chunks that do not divide a frame


def tone(amp: float, s: float) -> np.ndarray:
    t = np.arange(round(s * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)


def silence(s: float) -> np.ndarray:
    return np.zeros(round(s * RATE), np.float32)


class ScriptBackend(FakeBackend):
    """Request n yields ``signals[n]`` whole, or in chunks of ``chunk`` samples; the clock does
    not move, so stamps are positions in the sink's queue."""

    def __init__(self, signals: list[np.ndarray], chunk: int | None = None) -> None:
        super().__init__()
        self.signals, self.chunk = list(signals), chunk

    async def synthesize(self, text, delivery, *, language, voice):
        self.deliveries.append(delivery)
        self.texts.append(text)
        x = self.signals.pop(0)
        step = self.chunk or x.size
        for i in range(0, x.size, step):
            yield AudioChunk(x[i : i + step], RATE)


@pytest.fixture()
def make():
    made = []

    def build(backend, *, sink, clock):
        out = SpeechOutput(RoboEmotConfig(), sink, backend, clock=clock)
        out.start()
        made.append(out)
        return out

    yield build
    for out in made:
        out.close()


def ended(out, n: int) -> bool:
    return out.timeline.snapshot()["sentence_end"] >= n


def kinds(events: list[SpeechEvent], kind: str) -> list[SpeechEvent]:
    return [e for e in events if e.kind == kind]


@pytest.mark.parametrize("chunk", CHUNKINGS)
def test_tone_bursts_separated_by_gaps_give_phrase_gaps_at_the_right_stamps(make, chunk):
    clock = FakeClock()
    x = np.concatenate(
        [
            silence(0.1),  # leading silence: never a gap
            tone(0.3, 0.4),
            silence(0.3),  # a phrase gap: 0.5 .. 0.8
            tone(0.3, 0.4),
            silence(0.1),  # shorter than phrase_gap_ms (180 ms)
            tone(0.3, 0.4),
            silence(0.2),  # trailing silence: the sentence end, not a gap
        ]
    )
    out = make(ScriptBackend([x], chunk), sink=ClockedSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: ended(out, 1))
    events = out.timeline.events_between(-math.inf, math.inf)
    gaps = kinds(events, "phrase_gap")
    assert len(gaps) == 1
    assert gaps[0].at_s == pytest.approx(0.5) and gaps[0].until_s == pytest.approx(0.8)
    assert gaps[0].index == 0
    (start,), (end,) = kinds(events, "sentence_start"), kinds(events, "sentence_end")
    assert start.at_s == 0.0 and end.at_s == pytest.approx(x.size / RATE)
    assert out.timeline.level_at(0.3) > out.config.speech_motion.silence_dbfs
    assert out.timeline.level_at(0.65) < out.config.speech_motion.silence_dbfs
    frames = out.timeline.frames_between(-math.inf, math.inf)
    assert len(frames) == math.ceil(x.size / 320)  # 20 ms frames at 16 kHz; chunking cuts none
    assert [a for a, _ in frames[:3]] == pytest.approx([0.0, 0.02, 0.04])


@pytest.mark.parametrize("chunk", CHUNKINGS)
def test_a_louder_burst_gives_one_emphasis_and_a_steady_voice_none(make, chunk):
    clock = FakeClock()
    steady = np.concatenate([tone(0.1, 1.0), tone(0.1, 0.2), tone(0.1, 1.0)])
    loud = np.concatenate([tone(0.1, 1.0), tone(0.4, 0.2), tone(0.1, 1.0)])  # +12 dB
    out = make(ScriptBackend([steady, loud], chunk), sink=ClockedSink(clock), clock=clock)
    say(out, "Uno.")
    say(out, "Due.")
    assert wait_for(lambda: ended(out, 2))
    events = out.timeline.events_between(-math.inf, math.inf)
    first, second = (s.sentence for s in kinds(events, "sentence_start"))
    assert not [e for e in kinds(events, "emphasis") if e.sentence == first]
    (peak,) = [e for e in kinds(events, "emphasis") if e.sentence == second]
    begin = out.timeline.sentence_at(2.2 + 0.5).start_s  # the second sentence started at 2.2 s
    assert begin == pytest.approx(2.2)
    assert begin + 1.0 <= peak.at_s < begin + 1.2
    assert peak.dbfs == pytest.approx(20 * math.log10(0.4 / math.sqrt(2)), abs=0.1)


def test_question_and_mood_flags_are_set_per_sentence(make):
    clock = FakeClock()
    backend = ScriptBackend([tone(0.3, 0.2), tone(0.3, 0.2)])
    out = make(backend, sink=ClockedSink(clock), clock=clock)
    say(out, "Vieni anche tu?", mood="joy")
    say(out, "Ciao.", mood="calm")
    assert wait_for(lambda: ended(out, 2))
    asked, told = out.timeline.sentence_at(0.1), out.timeline.sentence_at(0.3)
    assert (asked.question, asked.mood) == (True, backend.deliveries[0].mood)
    assert (told.question, told.mood) == (False, backend.deliveries[1].mood)
    assert asked.mood == "joy" and told.mood == "calm"


def test_a_continued_sentence_piece_keeps_its_own_flags(make):
    clock = FakeClock()
    backend = ScriptBackend([tone(0.3, 0.2), tone(0.3, 0.2)])
    out = make(backend, sink=ClockedSink(clock), clock=clock)
    u = out.new_utterance()
    say(out, "Allora, ti va di venire", mood="joy", utterance=u)
    say(out, "al cinema domani?", mood="sadness", utterance=u, continued=True)
    assert wait_for(lambda: ended(out, 2))
    # the continued piece reuses the first piece's delivery (its mood; no question raise) ...
    assert backend.deliveries[1] is backend.deliveries[0]
    head = out.timeline.sentence_at(0.1)
    second = kinds(out.timeline.events_between(-math.inf, math.inf), "sentence_start")[1]
    tail = out.timeline.sentence_at(second.at_s + 0.1)
    # ... but the timeline takes the question flag from the piece's own text
    assert (head.question, head.continued) == (False, False)
    assert (tail.question, tail.continued) == (True, True)
    assert head.mood == tail.mood == "joy"
    # the pause pushed before the second piece is on the timeline (the breath rides on it)
    assert head.pad_start_s is None and tail.pad_start_s == pytest.approx(head.end_s)
    assert tail.start_s > tail.pad_start_s


class StoppingBackend(FakeBackend):
    """Yields 100 ms chunks of ``x``; after ``at_chunk`` of them moves the clock to ``cut_s``,
    calls ``stop()`` and then yields one more chunk, which races the stop."""

    def __init__(self, clock, x, at_chunk, cut_s) -> None:
        super().__init__()
        self.clock, self.x, self.at_chunk, self.cut_s = clock, x, at_chunk, cut_s
        self.out = None
        self.raced = False

    async def synthesize(self, text, delivery, *, language, voice):
        step = RATE // 10
        for n, i in enumerate(range(0, self.x.size, step)):
            if n == self.at_chunk:
                self.clock.now = self.cut_s
                self.out.stop()
                self.raced = True
            yield AudioChunk(self.x[i : i + step], RATE)


def test_nothing_remains_after_stop_including_a_chunk_racing_it(make):
    clock = FakeClock()
    x = np.concatenate([tone(0.3, 0.4), silence(0.3), tone(0.3, 0.4), tone(0.3, 1.0)])
    # 0.9 s pushed (the gap 0.4 .. 0.7 is confirmed), then a stop at 0.6 s: the gap's resume
    # is never heard, so the gap goes too
    backend = StoppingBackend(clock, x, at_chunk=9, cut_s=0.6)
    out = make(backend, sink=ClockedSink(clock), clock=clock)
    backend.out = out
    say(out)
    assert wait_for(lambda: backend.raced and out._current is not None and out._current.done())
    t = out.timeline
    frames = t.frames_between(-math.inf, math.inf)
    assert frames and max(a for a, _ in frames) < 0.6
    assert t.level_at(0.6) is None and t.level_at(0.59) is not None
    assert t.events_between(0.6, math.inf) == []
    assert not kinds(t.events_between(-math.inf, math.inf), "phrase_gap")
    cut = t.sentence_at(0.3)
    assert cut.cut and cut.end_s == pytest.approx(0.6)
    assert t.last_cut() == pytest.approx(0.6)
    snap = t.snapshot()
    assert snap["cuts"] == 1 and snap["frames"] == 45  # 0.9 s of 20 ms frames, none after
    assert snap["sentence_end"] == 0  # the stop made the sentence stale: no end was written


def test_a_sentence_after_a_stop_starts_a_fresh_timeline(make):
    clock = FakeClock()
    out = make(ScriptBackend([tone(0.3, 0.2)]), sink=ClockedSink(clock), clock=clock)
    out.timeline.truncate(0.0)  # a barge-in while nothing was pushed
    say(out)
    assert wait_for(lambda: ended(out, 1))
    s = out.timeline.sentence_at(0.1)
    assert s is not None and not s.cut and s.end_s == pytest.approx(0.2)


def test_the_timeline_holds_no_text_and_no_samples(make):
    canary = "Canarino segreto 4242?"
    clock = FakeClock()
    out = make(ScriptBackend([tone(0.3, 0.6)], chunk=337), sink=ClockedSink(clock), clock=clock)
    say(out, canary, mood="joy")
    assert wait_for(lambda: ended(out, 1))
    # the record types carry stamps, flags, ids and energies only
    assert {f.name for f in dataclasses.fields(SpeechEvent)} == {
        "kind", "at_s", "until_s", "sentence", "utterance", "index", "dbfs",
    }  # fmt: skip
    assert {f.name for f in dataclasses.fields(SpeechSentence)} == {
        "id", "utterance", "start_s", "end_s", "question", "mood", "continued", "pad_start_s",
        "cut",
    }  # fmt: skip
    seen: list = []
    walk(out.timeline, seen, set())
    assert not [v for v in seen if isinstance(v, np.ndarray)]
    assert not [v for v in seen if isinstance(v, str) and ("Canarino" in v or "4242" in v)]
    assert all(isinstance(v, int) for v in out.timeline.snapshot().values())


def walk(obj, seen: list, ids: set) -> None:
    """Every value reachable from ``obj`` through attributes, slots and containers."""
    if id(obj) in ids or callable(obj) and not hasattr(obj, "__dict__"):
        return
    ids.add(id(obj))
    seen.append(obj)
    if isinstance(obj, (str, bytes, int, float, bool, type(None), np.ndarray)):
        return
    if isinstance(obj, dict):
        children = [*obj.keys(), *obj.values()]
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children = list(obj)
    elif dataclasses.is_dataclass(obj):
        children = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        children = [v for k, v in vars(obj).items() if k not in {"clock", "_lock", "config"}]
    else:
        children = []
    for child in children:
        walk(child, seen, ids)


def test_a_hot_reload_changes_the_frame_length_from_the_next_sentence(make):
    clock = FakeClock()
    backend = ScriptBackend([tone(0.3, 0.4), tone(0.3, 0.4)])
    out = make(backend, sink=ClockedSink(clock), clock=clock)
    say(out, "Uno.")
    assert wait_for(lambda: ended(out, 1))
    out.apply_config(RoboEmotConfig(speech_motion={"envelope_frame_ms": 40}))
    say(out, "Due.")
    assert wait_for(lambda: ended(out, 2))
    starts = [a for a, _ in out.timeline.frames_between(-math.inf, math.inf)]
    assert starts[1] - starts[0] == pytest.approx(0.02)
    assert starts[-1] - starts[-2] == pytest.approx(0.04)


def test_history_drops_old_entries():
    clock = FakeClock()
    t = SpeechTimeline(SpeechMotionConfig(history_s=1.0), clock)
    x = tone(0.3, 0.4)
    t.push(1, x, RATE, 0.0, utterance=1, question=False, mood=None, continued=False)
    t.finish(1, 0.4)
    clock.now = 1.5
    t.push(2, x, RATE, 1.5, utterance=2, question=False, mood=None, continued=False)
    assert t.sentence_at(0.2) is None and t.level_at(0.2) is None
    assert t.events_between(-math.inf, 1.0) == []
    assert t.sentence_at(1.6) is not None


class NoTimelineSink(ClockedSink):
    """A sink that cannot say where its queue ends: every push starts at the clock."""

    def playing_until(self):
        return -math.inf


def test_frames_stay_in_order_on_a_sink_without_a_timeline(make):
    clock = FakeClock()
    backend = ScriptBackend([tone(0.3, 0.4), tone(0.3, 0.2)], chunk=640)  # two frames a chunk
    out = make(backend, sink=NoTimelineSink(clock), clock=clock)
    say(out, "Uno.")
    say(out, "Due.")
    assert wait_for(lambda: ended(out, 2))
    starts = [a for a, _ in out.timeline.frames_between(-math.inf, math.inf)]
    assert len(starts) == 30
    assert np.diff(starts) == pytest.approx(np.full(29, 0.02))


# ------------------------------------------------- fix round 1 (review of ZT1)


def seg(db: float, s: float) -> np.ndarray:
    """A steady tone whose frames have an RMS of ``db`` dBFS."""
    return tone(10 ** (db / 20) * math.sqrt(2), s)


def push(t: SpeechTimeline, sentence: int, x: np.ndarray, start: float, **kw) -> float:
    """Push ``x`` whole for ``sentence`` at ``start`` and finish it; returns its end."""
    args = {"utterance": 1, "question": False, "mood": None, "continued": False} | kw
    t.push(sentence, x, RATE, start, **args)
    end = start + x.size / RATE
    t.finish(sentence, end)
    return end


def emphases(t: SpeechTimeline) -> list[float]:
    return [e.at_s for e in kinds(t.events_between(-math.inf, math.inf), "emphasis")]


@pytest.mark.parametrize("continued", [False, True])
def test_the_emphasis_interval_holds_across_back_to_back_sentences(continued):
    t = SpeechTimeline(SpeechMotionConfig(emphasis_warmup_ms=0), FakeClock())
    end = push(t, 1, np.concatenate([seg(-26, 0.6), seg(-14, 0.04), seg(-26, 0.1)]), 0.0)
    # the next sentence (or the rest of the same one) peaks 0.24 s after the first peak
    s2 = np.concatenate([seg(-26, 0.1), seg(-14, 0.04), seg(-26, 0.6)])
    end = push(t, 2, s2, end, continued=continued)
    assert emphases(t) == pytest.approx([0.62])
    # a peak a full interval later still fires
    push(t, 3, np.concatenate([seg(-26, 1.0), seg(-14, 0.04), seg(-26, 0.2)]), end)
    assert emphases(t) == pytest.approx([0.62, end + 1.02])


def test_a_cut_resets_the_emphasis_interval():
    t = SpeechTimeline(SpeechMotionConfig(emphasis_warmup_ms=0), FakeClock())
    end = push(t, 1, np.concatenate([seg(-26, 0.6), seg(-14, 0.04), seg(-26, 0.1)]), 0.0)
    t.truncate(end)  # nothing after the cut plays, so the first peak no longer limits the next
    push(t, 2, np.concatenate([seg(-26, 0.1), seg(-14, 0.04), seg(-26, 0.6)]), end)
    assert emphases(t) == pytest.approx([0.62, end + 0.12])


@pytest.mark.parametrize("warmup_ms, expected", [(250, 0), (0, 1)])
def test_a_sentence_onset_ramp_is_not_an_emphasis(warmup_ms, expected):
    t = SpeechTimeline(SpeechMotionConfig(emphasis_warmup_ms=warmup_ms), FakeClock())
    ramp = [seg(db, 0.02) for db in (-40, -30, -22, -18)]
    push(t, 1, np.concatenate([*ramp, seg(-20, 1.0)]), 0.0)
    # the -18 frame clears the median of the ramp by 8 dB; only the warm-up keeps it out
    assert len(emphases(t)) == expected


def test_the_sentence_end_covers_its_frames_on_a_sink_without_a_timeline(make):
    clock = FakeClock()
    out = make(ScriptBackend([tone(0.3, 1.0)]), sink=NoTimelineSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: ended(out, 1))
    t = out.timeline
    assert t.sentence_at(0.5) is not None and t.sentence_at(0.5).end_s == pytest.approx(1.0)
    (end,) = kinds(t.events_between(-math.inf, math.inf), "sentence_end")
    assert end.at_s >= max(a for a, _ in t.frames_between(-math.inf, math.inf)) + 0.02 - 1e-9


def test_the_report_has_the_time_spent_writing_the_timeline(make):
    clock = FakeClock()
    out = make(ScriptBackend([tone(0.3, 0.5)], chunk=1600), sink=ClockedSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: ended(out, 1))
    timeline_ms = out.snapshot()["timeline_ms"]
    assert timeline_ms["count"] == 5 and timeline_ms["max"] >= 0  # one per speech push


class DryingBackend(FakeBackend):
    """Two 1000-sample chunks (3 frames and a 40-sample rest each); between them the clock moves
    past the end of the sink's queue, as when a streamed backend falls behind."""

    def __init__(self, clock, dry_until: float) -> None:
        super().__init__()
        self.clock, self.dry_until = clock, dry_until

    async def synthesize(self, text, delivery, *, language, voice):
        yield AudioChunk(tone(0.3, 1000 / RATE), RATE)
        self.clock.now = self.dry_until
        yield AudioChunk(tone(0.3, 1000 / RATE), RATE)


def test_a_sink_running_dry_mid_sentence_leaves_no_frame_across_the_gap(make):
    clock = FakeClock()
    out = make(DryingBackend(clock, 0.5), sink=ClockedSink(clock), clock=clock)
    say(out)
    assert wait_for(lambda: ended(out, 1))
    t = out.timeline
    starts = [a for a, _ in t.frames_between(-math.inf, math.inf)]
    assert starts[:5] == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.5])
    assert t.level_at(0.062) is not None  # the flushed rest ends where the audio stopped ...
    assert t.level_at(0.063) is None and t.level_at(0.3) is None  # ... and nothing spans the gap
    assert t.level_at(0.5) is not None
    (gap,) = kinds(t.events_between(-math.inf, math.inf), "phrase_gap")  # the pause is heard
    assert gap.at_s == pytest.approx(0.0625) and gap.until_s == pytest.approx(0.5)


def test_a_sample_rate_change_mid_sentence_gives_frames_at_the_new_rate():
    t = SpeechTimeline(SpeechMotionConfig(), FakeClock())
    args = {"utterance": 1, "question": False, "mood": None, "continued": False}
    t.push(1, tone(0.3, 1000 / RATE), RATE, 0.0, **args)  # 3 frames and a 2.5 ms rest
    slow = (0.3 * np.sin(2 * np.pi * TONE_HZ * np.arange(800) / 8000)).astype(np.float32)
    t.push(1, slow, 8000, 1000 / RATE, **args)  # 100 ms at 8 kHz: 160-sample frames
    t.finish(1, 1000 / RATE + 0.1)
    starts = [a for a, _ in t.frames_between(-math.inf, math.inf)]
    assert starts == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.0625, 0.0825, 0.1025, 0.1225, 0.1425])
    assert t.level_at(0.0624) is not None and t.level_at(0.15) is not None


def test_a_stop_with_nothing_on_the_timeline_is_not_a_cut():
    t = SpeechTimeline(SpeechMotionConfig(), FakeClock())
    t.truncate(5.0)
    assert t.snapshot()["cuts"] == 0 and t.last_cut() is None
    end = push(t, 1, tone(0.3, 0.2), 6.0)
    t.truncate(end + 1.0)  # after the sentence ended: still nothing to cut
    assert t.snapshot()["cuts"] == 0
    t.push(2, tone(0.3, 0.2), RATE, 8.0, utterance=1, question=False, mood=None, continued=False)
    t.truncate(8.1)
    assert t.snapshot()["cuts"] == 1 and t.last_cut() == 8.1


def test_the_phrase_index_survives_a_long_pause_inside_one_utterance():
    clock = FakeClock()
    t = SpeechTimeline(SpeechMotionConfig(history_s=1.0), clock)
    gapped = np.concatenate([tone(0.3, 0.2), silence(0.3), tone(0.3, 0.2)])
    push(t, 1, gapped, 0.0)
    clock.now = 10.0  # an LLM stall far longer than the history between two sentences, ...
    push(t, 2, tone(0.3, 0.2), 10.0, utterance=2)  # ... and a cue of another answer meanwhile
    clock.now = 20.0
    push(t, 3, gapped, 20.0)
    gaps = kinds(t.events_between(-math.inf, math.inf), "phrase_gap")
    assert [(g.utterance, g.index) for g in gaps] == [(1, 1)]  # the count outlived the history
