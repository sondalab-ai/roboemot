"""Endpointer state machine on synthetic signals with the energy VAD (SPEC 8 stage 2)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from roboemot.config.schema import VadConfig
from roboemot.voice.vad import (
    Endpointer,
    EnergyVAD,
    SpeechAudio,
    SpeechEnd,
    SpeechStart,
    VADBackend,
)

RATE = 16000
FRAME = 320  # 20 ms
FRAME_S = FRAME / RATE
T0 = 1000.0
LOUD = 0.3  # rms 0.21, probability 1 with the default reference
MID = 0.0566  # rms 0.04, probability 0.4: between the off and on thresholds


def tone(duration_s: float, amp: float) -> np.ndarray:
    n = round(duration_s * RATE)
    t = np.arange(n) / RATE
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def signal(*segments: tuple[float, float]) -> np.ndarray:
    return np.concatenate([tone(d, a) for d, a in segments])


def cfg(**kw) -> VadConfig:
    base = {"min_speech_ms": 100, "end_silence_ms": 200, "pre_roll_ms": 100, "max_utterance_s": 2}
    return VadConfig(**{**base, **kw})


def run(x: np.ndarray, config: VadConfig | None = None, chunk: int = FRAME) -> list:
    ep = Endpointer(EnergyVAD(RATE, FRAME), config or cfg())
    events = []
    for i in range(0, len(x), chunk):
        events += ep.feed(x[i : i + chunk], T0 + i / RATE)
    return events


def kinds(events: list, kind: type) -> list:
    return [e for e in events if isinstance(e, kind)]


def test_energy_vad_is_a_backend():
    vad = EnergyVAD(RATE, FRAME)
    assert isinstance(vad, VADBackend)
    assert vad.speech_prob(tone(FRAME_S, LOUD)) == 1.0
    assert vad.speech_prob(np.zeros(FRAME, np.float32)) == 0.0


def test_speech_burst_gives_one_start_and_one_end():
    events = run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0)))
    starts, ends = kinds(events, SpeechStart), kinds(events, SpeechEnd)
    assert len(starts) == 1 and len(ends) == 1
    assert events.index(starts[0]) < events.index(ends[0])
    assert ends[0].reason == "silence"


def test_start_stamp_is_the_first_speech_frame():
    start = kinds(run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechStart)[0]
    assert start.stamp == pytest.approx(T0 + 0.5)


def test_last_speech_stamp_and_endpoint():
    end = kinds(run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechEnd)[0]
    # The last speech frame starts one frame before the speech stops at 1.1 s.
    assert end.last_speech_stamp == pytest.approx(T0 + 1.1 - FRAME_S)
    # The endpoint is the end of the frame that completed 200 ms of silence.
    assert end.endpoint_stamp == pytest.approx(T0 + 1.1 + 0.2)
    assert end.start_stamp == pytest.approx(T0 + 0.5)


def test_audio_includes_pre_roll_and_runs_to_the_endpoint():
    end = kinds(run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechEnd)[0]
    assert end.audio.dtype == np.float32
    assert end.audio_stamp == pytest.approx(T0 + 0.5 - 0.1)
    assert len(end.audio) / RATE == pytest.approx(end.endpoint_stamp - end.audio_stamp)


def test_pre_roll_is_capped_by_available_audio():
    end = kinds(run(signal((0.04, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechEnd)[0]
    assert end.audio_stamp == pytest.approx(T0)


def test_streamed_audio_matches_the_final_audio():
    events = run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0)))
    start = kinds(events, SpeechStart)[0]
    end = kinds(events, SpeechEnd)[0]
    streamed = np.concatenate([start.pre_roll, *[e.samples for e in kinds(events, SpeechAudio)]])
    np.testing.assert_array_equal(streamed, end.audio)
    assert start.pre_roll_stamp == pytest.approx(end.audio_stamp)


def test_short_burst_below_min_speech_is_ignored():
    assert run(signal((0.5, 0.0), (0.06, LOUD), (0.8, 0.0))) == []


def test_pause_shorter_than_end_silence_stays_one_utterance():
    events = run(signal((0.3, LOUD), (0.12, 0.0), (0.3, LOUD), (0.6, 0.0)))
    assert len(kinds(events, SpeechStart)) == 1
    assert len(kinds(events, SpeechEnd)) == 1


def test_pause_longer_than_end_silence_splits_utterances():
    events = run(signal((0.3, LOUD), (0.4, 0.0), (0.3, LOUD), (0.6, 0.0)))
    assert len(kinds(events, SpeechStart)) == 2
    assert len(kinds(events, SpeechEnd)) == 2


def test_hysteresis_mid_level_does_not_start_but_keeps_speaking():
    assert run(signal((0.5, MID), (0.5, 0.0))) == []
    # Once speaking, a mid-level stretch longer than end_silence does not end the utterance.
    events = run(signal((0.3, LOUD), (0.5, MID), (0.3, LOUD), (0.6, 0.0)))
    assert len(kinds(events, SpeechEnd)) == 1
    end = kinds(events, SpeechEnd)[0]
    assert end.last_speech_stamp == pytest.approx(T0 + 1.1 - FRAME_S)


def test_zero_margin_makes_mid_level_silence():
    events = run(signal((0.3, LOUD), (0.5, MID), (0.6, 0.0)), cfg(off_margin=0, threshold=0.5))
    end = kinds(events, SpeechEnd)[0]
    assert end.last_speech_stamp == pytest.approx(T0 + 0.3 - FRAME_S)


def test_max_utterance_cuts_and_continues():
    events = run(signal((5.0, LOUD), (0.6, 0.0)))
    ends = kinds(events, SpeechEnd)
    assert ends[0].reason == "max_length"
    assert ends[0].endpoint_stamp - ends[0].start_stamp == pytest.approx(2.0, abs=FRAME_S)
    assert len(kinds(events, SpeechStart)) == len(ends) >= 3
    assert ends[-1].reason == "silence"


@pytest.mark.parametrize("chunk", [1, 7, 160, 321, 512, 1000, 4096])
def test_arbitrary_chunk_sizes_give_the_same_events(chunk):
    x = signal((0.5, 0.0), (0.6, LOUD), (0.1, 0.0), (0.4, LOUD), (0.8, 0.0), (0.3, LOUD), (0.5, 0))
    ref = run(x)
    got = run(x, chunk=chunk)
    assert [type(e) for e in got if not isinstance(e, SpeechAudio)] == [
        type(e) for e in ref if not isinstance(e, SpeechAudio)
    ]
    for a, b in zip(kinds(ref, SpeechEnd), kinds(got, SpeechEnd), strict=True):
        assert a.last_speech_stamp == pytest.approx(b.last_speech_stamp)
        assert a.endpoint_stamp == pytest.approx(b.endpoint_stamp)
        np.testing.assert_array_equal(a.audio, b.audio)


def test_chunk_stamps_may_have_gaps_between_chunks_only_by_the_caller():
    ep = Endpointer(EnergyVAD(RATE, FRAME), cfg())
    out = ep.feed(tone(0.3, LOUD), T0)
    assert len(kinds(out, SpeechStart)) == 1
    assert ep.in_speech


def test_reset_drops_the_utterance_in_progress():
    ep = Endpointer(EnergyVAD(RATE, FRAME), cfg())
    ep.feed(tone(0.3, LOUD), T0)
    ep.reset()
    assert not ep.in_speech
    assert ep.feed(tone(0.5, 0.0), T0 + 1) == []


def test_rejects_wrong_dtype_and_shape():
    ep = Endpointer(EnergyVAD(RATE, FRAME), cfg())
    with pytest.raises(ValueError):
        ep.feed(np.zeros(FRAME, np.float64), T0)
    with pytest.raises(ValueError):
        ep.feed(np.zeros((FRAME, 2), np.float32), T0)


def test_events_never_carry_text_and_repr_hides_audio():
    end = kinds(run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechEnd)[0]
    assert "array" not in repr(end)


def test_max_length_cut_loses_no_frames_and_does_not_reset_the_backend():
    resets = []

    class Counting(EnergyVAD):
        def reset(self) -> None:
            resets.append(1)

    ep = Endpointer(Counting(RATE, FRAME), cfg())
    x = signal((5.0, LOUD), (0.6, 0.0))
    events = []
    for i in range(0, len(x), FRAME):
        events += ep.feed(x[i : i + FRAME], T0 + i / RATE)
    ends = kinds(events, SpeechEnd)
    starts = kinds(events, SpeechStart)
    assert [e.reason for e in ends[:-1]] == ["max_length"] * (len(ends) - 1)
    for prev, nxt in zip(ends, ends[1:], strict=False):
        assert nxt.audio_stamp == pytest.approx(prev.endpoint_stamp - FRAME_S)
    assert starts[1].stamp == pytest.approx(ends[0].endpoint_stamp - FRAME_S)
    assert len(resets) == 1 + 1  # constructor reset plus the final silence end only


def test_endpoint_is_end_silence_plus_one_frame_after_the_last_speech_frame():
    end = kinds(run(signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))), SpeechEnd)[0]
    assert end.endpoint_stamp - end.last_speech_stamp == pytest.approx(0.2 + FRAME_S)


def test_mutating_the_input_after_feed_does_not_change_the_audio():
    ep = Endpointer(EnergyVAD(RATE, FRAME), cfg())
    x = signal((0.5, 0.0), (0.6, LOUD), (0.8, 0.0))
    ref = kinds(run(x), SpeechEnd)[0].audio
    buf = np.empty(FRAME + 100, np.float32)
    events = []
    for i in range(0, len(x), len(buf)):
        chunk = x[i : i + len(buf)]
        buf[: len(chunk)] = chunk
        events += ep.feed(buf[: len(chunk)], T0 + i / RATE)
        buf[:] = 0.0
    np.testing.assert_array_equal(kinds(events, SpeechEnd)[0].audio, ref)


class ScriptedVAD:
    """Backend that returns the scripted probabilities, one per frame (then 0)."""

    rate_hz = RATE
    frame_samples = FRAME

    def __init__(self, probs):
        self.probs = list(probs)
        self.resets = 0

    def speech_prob(self, frame):
        return self.probs.pop(0) if self.probs else 0.0

    def reset(self):
        self.resets += 1


def scripted(probs, config=None, extra_frames=0):
    """An Endpointer over ``ScriptedVAD(probs)`` after feeding one frame per probability."""
    ep = Endpointer(ScriptedVAD(probs), config or cfg())
    silence = np.zeros(FRAME, np.float32)
    for i in range(len(probs) + extra_frames):
        ep.feed(silence, T0 + i * FRAME_S)
    return ep


def test_a_burst_that_starts_speech_is_recorded_as_started():
    # min_speech_ms 100 = 5 frames; the burst is 2 frames at 0.3, then 6 frames at 0.8/0.9
    ep = scripted([0.0, 0.3, 0.35, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8])
    snap = ep.snapshot()
    assert snap["started"]["peak_prob"] == {
        "count": 1,
        "p50": 0.9,
        "p90": 0.9,
        "p99": 0.9,
        "max": 0.9,
    }
    assert snap["started"]["longest_on_run_ms"]["max"] == pytest.approx(5 * FRAME_S * 1000)
    assert snap["not_started"] == {
        "peak_prob": {"count": 0},
        "longest_on_run_ms": {"count": 0},
    }
    assert snap["starts"] == 1


def test_a_burst_that_does_not_start_speech_keeps_its_peak_and_longest_run():
    # frames at or above 0.5 form runs of 2 and 3 inside one burst (0.3 keeps it open)
    ep = scripted([0.0, 0.6, 0.7, 0.3, 0.5, 0.55, 0.9, 0.3, 0.0, 0.1])
    snap = ep.snapshot()
    assert snap["starts"] == 0
    peak = snap["not_started"]["peak_prob"]
    assert peak["count"] == 1 and peak["max"] == 0.9
    run = snap["not_started"]["longest_on_run_ms"]
    assert run["max"] == pytest.approx(3 * FRAME_S * 1000) and run["p50"] == run["max"]
    assert snap["started"]["peak_prob"] == {"count": 0}


def test_a_burst_ends_at_the_first_frame_below_watch_prob():
    ep = scripted([0.3, 0.3, 0.1, 0.4, 0.0])
    peaks = ep.snapshot()["not_started"]["peak_prob"]
    assert peaks["count"] == 2 and peaks["p50"] == 0.3 and peaks["max"] == 0.4
    assert ep.snapshot()["not_started"]["longest_on_run_ms"]["max"] == 0.0


def test_reset_in_the_middle_of_a_burst_records_nothing_and_keeps_counters():
    ep = scripted([0.0, 0.6, 0.7])
    ep.reset()
    ep.feed(np.zeros(FRAME, np.float32), T0 + 10)  # the next frame (prob 0) must not close it
    snap = ep.snapshot()
    assert snap["not_started"]["peak_prob"] == {"count": 0}
    assert snap["frames"] == 4 and snap["frames_on"] == 2


def test_counters_follow_frames_starts_and_both_ways_to_end():
    # 6 on frames start speech, 10 silent frames (200 ms) end it; 70 more (1.4 s) are cut at 1 s
    probs = [0.9] * 6 + [0.0] * 10 + [0.9] * 70
    ep = scripted(probs, cfg(max_utterance_s=1))
    snap = ep.snapshot()
    assert snap["frames"] == len(probs)
    assert snap["frames_on"] == 76
    assert snap["ends_silence"] == 1
    assert snap["ends_max_length"] == 1
    assert snap["starts"] == 3  # the first, the second, and the one the cut continues into


def test_watch_prob_above_threshold_watches_from_threshold():
    ep = scripted([0.55, 0.0, 0.2, 0.0], cfg(threshold=0.5, watch_prob=0.9))
    snap = ep.snapshot()
    assert snap["watch_prob"] == 0.5
    assert snap["not_started"]["peak_prob"]["count"] == 1


def test_watch_window_keeps_only_the_latest_bursts():
    ep = scripted([0.3, 0.0] * 12, cfg(watch_window=10))
    snap = ep.snapshot()
    assert snap["window"] == 10 and snap["not_started"]["peak_prob"]["count"] == 10


def test_snapshot_is_plain_json():
    ep = scripted([np.float32(0.3), np.float32(0.0), np.float32(0.9)] * 4)
    snap = ep.snapshot()
    assert json.loads(json.dumps(snap)) == snap


def test_a_nan_probability_closes_the_burst_and_keeps_the_report_valid_json():
    ep = scripted([float("nan"), 0.0, 0.3, 0.0])
    snap = ep.snapshot()
    json.loads(json.dumps(snap, allow_nan=False))
    assert snap["not_started"]["peak_prob"]["count"] == 1


def test_watch_settings_have_defaults_and_ranges():
    from pydantic import ValidationError
    from roboemot.config.schema import ReflexAudioConfig

    vad, audio = VadConfig(), ReflexAudioConfig()
    assert (vad.watch_prob, vad.watch_window) == (0.2, 500)
    assert (audio.level_block_s, audio.level_window) == (1.0, 300)
    for model, field, bad in [
        (VadConfig, "watch_prob", 0.0),
        (VadConfig, "watch_prob", 0.96),
        (VadConfig, "watch_window", 9),
        (VadConfig, "watch_window", 10001),
        (ReflexAudioConfig, "level_block_s", 0.05),
        (ReflexAudioConfig, "level_block_s", 61),
        (ReflexAudioConfig, "level_window", 9),
        (ReflexAudioConfig, "level_window", 10001),
    ]:
        with pytest.raises(ValidationError):
            model(**{field: bad})


def feed_probs(ep, probs, start=0):
    """Feed one silent frame per probability of ``ep``'s ScriptedVAD from frame ``start``;
    returns the events and the next frame index."""
    ep._vad.probs.extend(probs)
    events = []
    for i in range(start, start + len(probs)):
        events += ep.feed(np.zeros(FRAME, np.float32), T0 + i * FRAME_S)
    return events, start + len(probs)


def test_a_hot_reload_changes_threshold_and_min_speech_ms_of_a_running_endpointer():
    ep = Endpointer(ScriptedVAD([]), cfg(threshold=0.5, min_speech_ms=100))  # 5 frames
    events, n = feed_probs(ep, [0.4] * 8)
    assert kinds(events, SpeechStart) == []
    ep.apply_config(cfg(threshold=0.35, min_speech_ms=40))  # 2 frames
    events, n = feed_probs(ep, [0.0, 0.4, 0.4], n)
    starts = kinds(events, SpeechStart)
    assert len(starts) == 1 and starts[0].stamp == pytest.approx(T0 + 9 * FRAME_S)


def test_a_reload_mid_utterance_keeps_it_and_applies_the_new_end_silence():
    ep = Endpointer(ScriptedVAD([]), cfg(end_silence_ms=200))  # 10 frames
    events, n = feed_probs(ep, [0.9] * 6 + [0.0] * 3)
    assert ep.in_speech
    ep.apply_config(cfg(end_silence_ms=100))  # 5 frames: 2 more silent frames end it
    events, n = feed_probs(ep, [0.0, 0.0], n)
    ends = kinds(events, SpeechEnd)
    assert len(ends) == 1 and ends[0].reason == "silence"
    assert ends[0].audio_stamp == pytest.approx(T0) and len(ends[0].audio) == n * FRAME


def test_a_reload_while_a_run_is_counting_keeps_the_true_start_stamp():
    for min_ms in (40, 200):  # shrink to 2 frames, grow to 10
        ep = Endpointer(ScriptedVAD([]), cfg(min_speech_ms=100))
        events, n = feed_probs(ep, [0.0, 0.9, 0.9, 0.9])
        ep.apply_config(cfg(min_speech_ms=min_ms))
        events, n = feed_probs(ep, [0.9] * 8, n)
        starts = kinds(events, SpeechStart)
        assert len(starts) == 1 and starts[0].stamp == pytest.approx(T0 + FRAME_S)


def test_a_reload_with_no_pre_roll_and_a_shorter_run_never_misplaces_the_start():
    ep = Endpointer(ScriptedVAD([]), cfg(pre_roll_ms=0, min_speech_ms=100))
    events, n = feed_probs(ep, [0.9] * 4)
    ep.apply_config(cfg(pre_roll_ms=0, min_speech_ms=40))
    events, n = feed_probs(ep, [0.9], n)
    start = kinds(events, SpeechStart)[0]
    assert start.pre_roll_stamp <= start.stamp <= T0 + 4 * FRAME_S
    assert start.pre_roll_stamp == pytest.approx(start.stamp)


def test_a_reload_moves_the_watch_level_with_the_threshold_only():
    ep = Endpointer(ScriptedVAD([]), cfg(threshold=0.5, watch_prob=0.9, watch_window=10))
    ep.apply_config(cfg(threshold=0.35, watch_prob=0.1, watch_window=50))
    feed_probs(ep, [0.0])
    snap = ep.snapshot()
    assert snap["watch_prob"] == 0.35 and snap["window"] == 10  # restart-only fields stay


def test_a_shorter_min_speech_starts_a_run_that_already_qualifies_without_more_speech():
    ep = Endpointer(ScriptedVAD([]), cfg(min_speech_ms=200))  # 10 frames
    events, n = feed_probs(ep, [0.0] + [0.9] * 6)
    ep.apply_config(cfg(min_speech_ms=40))  # 2 frames: the run of 6 already qualifies
    events, n = feed_probs(ep, [0.0], n)
    starts = kinds(events, SpeechStart)
    assert len(starts) == 1 and starts[0].stamp == pytest.approx(T0 + FRAME_S)
    assert ep.in_speech


def test_a_reload_after_a_max_length_cut_keeps_the_continued_utterance():
    ep = Endpointer(ScriptedVAD([]), cfg(max_utterance_s=1))
    events, n = feed_probs(ep, [0.9] * 55)  # cut at 1 s, the speech goes on in a new utterance
    assert kinds(events, SpeechEnd)[0].reason == "max_length" and ep.in_speech
    ep.apply_config(cfg(max_utterance_s=1, min_speech_ms=200, pre_roll_ms=0, end_silence_ms=100))
    events, n = feed_probs(ep, [0.9] * 3 + [0.0] * 5, n)
    ends = kinds(events, SpeechEnd)
    assert kinds(events, SpeechStart) == [] and len(ends) == 1 and ends[0].reason == "silence"
    assert ends[0].start_stamp == pytest.approx(T0 + 49 * FRAME_S)
    assert len(ends[0].audio) == (n - 49) * FRAME
