"""Endpointer state machine on synthetic signals with the energy VAD (SPEC 8 stage 2)."""

from __future__ import annotations

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
