"""Speaker: chunked non-blocking playback, queued duration and barge-in (SPEC 4A, 8)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import BargeInDuckConfig, RegisterParams, RoboEmotConfig, StateEffects
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.duck import DuckGain
from roboemot.voice.speech import SpeechOutput
from roboemot.voice.tts import AudioChunk
from roboemot_reachymini.speaker import ReachySpeaker


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def fake(out_rate=16000, audio="present"):
    log = SimpleNamespace(pushed=[], started=0, stopped=0, cleared=0)
    audio_obj = SimpleNamespace(clear_player=lambda: setattr(log, "cleared", log.cleared + 1))
    media = SimpleNamespace(
        audio=None if audio is None else audio_obj,
        start_playing=lambda: setattr(log, "started", log.started + 1),
        stop_playing=lambda: setattr(log, "stopped", log.stopped + 1),
        push_audio_sample=lambda x: log.pushed.append(x),
        get_output_audio_samplerate=lambda: out_rate,
    )
    return SimpleNamespace(media=media), log


def test_audio_is_pushed_in_chunks_and_the_queued_duration_is_counted():
    mini, log = fake()
    clock = Clock()
    sp = ReachySpeaker(mini, 100, clock=clock)
    assert sp.start_playing() and log.started == 1
    until = sp.push_audio_sample(np.zeros(16000, np.float32), 16000)  # one second
    assert [len(p) for p in log.pushed] == [1600] * 10
    assert until == pytest.approx(101.0) and sp.playing_until() == pytest.approx(101.0)
    assert sp.queued_s() == pytest.approx(1.0)
    clock.t += 0.4
    assert sp.queued_s() == pytest.approx(0.6)
    sp.push_audio_sample(np.zeros(8000, np.float32), 16000)  # queued behind the first
    assert sp.playing_until() == pytest.approx(101.5)
    clock.t += 10
    sp.push_audio_sample(np.zeros(1600, np.float32), 16000)  # idle: starts from now
    assert sp.playing_until() == pytest.approx(clock.t + 0.1)


def test_audio_is_resampled_to_the_output_rate_the_sdk_reports():
    mini, log = fake(out_rate=48000)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    sp.push_audio_sample(np.zeros(22050, np.float32), 22050)  # one second at Piper's rate
    sp.end_stream()  # the converter's held-back tail
    assert sum(len(p) for p in log.pushed) == 48000
    assert all(p.dtype == np.float32 and p.ndim == 1 for p in log.pushed)
    assert sp.queued_s() == pytest.approx(1.0)


def test_stop_flushes_through_clear_player_and_resets_the_queue():
    mini, log = fake()
    clock = Clock()
    act = SpeechActivity(300)
    sp = ReachySpeaker(mini, 100, act, clock)
    sp.start_playing()
    sp.push_audio_sample(np.zeros(32000, np.float32), 16000)
    assert act.speaking(clock.t + 1.5)
    clock.t += 0.5
    sp.stop()
    assert log.cleared == 1 and log.stopped == 0
    assert sp.queued_s() == 0.0 and sp.playing_until() == clock.t
    assert act.speaking(clock.t + 0.1) and not act.speaking(clock.t + 0.3)  # only the tail


def test_a_failing_clear_player_is_logged_and_the_queue_is_still_reset():
    mini, log = fake()
    mini.media.audio.clear_player = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    clock = Clock()
    sp = ReachySpeaker(mini, 100, clock=clock)
    sp.start_playing()
    sp.push_audio_sample(np.zeros(1600, np.float32), 16000)
    sp.stop()
    assert sp.queued_s() == 0.0 and log.stopped == 0


def test_a_stop_during_a_push_abandons_the_chunks_not_yet_pushed():
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    real = mini.media.push_audio_sample

    def push(x):
        real(x)
        if len(log.pushed) == 3:
            sp.stop()

    mini.media.push_audio_sample = push
    sp.push_audio_sample(np.zeros(16000, np.float32), 16000)
    assert len(log.pushed) == 3 and sp.queued_s() == 0.0


def test_the_speech_gate_follows_the_pushed_audio():
    mini, _ = fake()
    clock = Clock()
    act = SpeechActivity(300)
    sp = ReachySpeaker(mini, 100, act, clock)
    sp.start_playing()
    sp.push_audio_sample(np.zeros(16000, np.float32), 16000)
    assert act.speaking(100.5) and act.speaking(101.2) and not act.speaking(101.3)


def test_no_audio_system_or_bad_rate_means_a_silent_speaker():
    mini, log = fake(audio=None)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    assert not sp.start_playing() and log.started == 0
    assert sp.push_audio_sample(np.zeros(100, np.float32), 16000) == 0.0 and log.pushed == []
    mini, log = fake(out_rate=-1)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    assert not sp.start_playing()
    sp.close()


def test_a_failing_push_is_logged_by_type_and_ends_the_push(caplog):
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    mini.media.push_audio_sample = lambda x: (_ for _ in ()).throw(RuntimeError("secret"))
    sp.push_audio_sample(np.zeros(16000, np.float32), 16000)
    assert sp.queued_s() == 0.0
    assert all("secret" not in r.getMessage() for r in caplog.records)


def test_close_flushes_and_stops_the_pipeline():
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    sp.close()
    assert log.cleared == 1 and log.stopped == 1
    sp.close()
    assert log.stopped == 1


def test_a_stop_during_a_push_leaves_the_gate_reset_not_re_extended():
    mini, log = fake()
    clock = Clock()
    act = SpeechActivity(0)
    sp = ReachySpeaker(mini, 100, act, clock)
    sp.start_playing()
    real = mini.media.push_audio_sample

    def push(x):
        real(x)
        if len(log.pushed) == 2:
            sp.stop()

    mini.media.push_audio_sample = push
    sp.push_audio_sample(np.zeros(16000, np.float32), 16000)
    assert not act.speaking(clock.t + 0.01)  # reset won; the interrupted piece did not extend


def test_stop_and_pushes_from_two_threads_never_leave_the_gate_ahead_of_the_speaker():
    import threading

    mini, _ = fake()
    clock = Clock()
    act = SpeechActivity(0)
    sp = ReachySpeaker(mini, 10, act, clock)
    sp.start_playing()
    pusher = threading.Thread(
        target=lambda: [sp.push_audio_sample(np.zeros(16000, np.float32), 16000) for _ in range(20)]
    )
    pusher.start()
    for _ in range(50):
        sp.stop()
    pusher.join()
    sp.stop()
    assert act.until() <= clock.t and sp.queued_s() == 0.0


def test_the_return_value_is_zero_when_nothing_was_queued_and_bad_rates_are_refused():
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    assert sp.push_audio_sample(np.zeros(0, np.float32), 16000) == 0.0
    assert sp.push_audio_sample(np.zeros(100, np.float32), 0) == 0.0
    assert sp.push_audio_sample(np.zeros(100, np.float32), -5) == 0.0
    assert log.pushed == []
    assert sp.push_audio_sample(np.zeros(100, np.float32), 16000) > 0.0


def test_the_speaker_keeps_a_timestamped_reference_for_the_echo_check_and_cuts_it_on_stop():
    from roboemot.voice.bargein import PlaybackReference

    mini, log = fake(out_rate=48000)
    clock = Clock()
    ref = PlaybackReference(16000, keep_s=5.0)
    sp = ReachySpeaker(mini, 100, clock=clock, reference=ref)
    sp.start_playing()
    tone = np.sin(np.arange(22050) * 0.05).astype(np.float32)
    sp.push_audio_sample(tone, 22050)  # one second at Piper's rate, resampled twice on the way
    sp.end_stream()
    start = clock.t
    w = ref.window(start, start + 1.0)
    assert w.size == 16000 and float(np.abs(w).max()) > 0.5  # at the reference rate
    clock.t += 0.5
    sp.stop()
    assert float(np.abs(ref.window(start + 0.6, start + 1.0)).max()) == 0.0  # never played
    assert float(np.abs(ref.window(start + 0.1, start + 0.4)).max()) > 0.5


# ------------------------------------------------- streamed resampling (SPEC 8)


def sine(rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * 233.3 * t)).astype(np.float32)


def push_in_pieces(sp, x, rate, piece_s=0.2):
    step = int(rate * piece_s)
    for i in range(0, x.size, step):
        sp.push_audio_sample(x[i : i + step], rate)


@pytest.mark.parametrize("out_rate", [16000, 48000])
def test_chunked_playback_matches_the_whole_sentence_resampled_at_once(out_rate):
    from roboemot_reachymini.resample import resample

    mini, log = fake(out_rate=out_rate)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    x = sine(22050)
    push_in_pieces(sp, x, 22050)
    ref = resample(x, 22050, out_rate)
    before = np.concatenate(log.pushed)
    # no edge transient at the 200 ms boundaries: what is out so far is the whole's prefix
    assert np.abs(before - ref[: before.size]).max() < 1e-5
    sp.end_stream()
    out = np.concatenate(log.pushed)
    assert out.size == ref.size and np.abs(out - ref).max() < 1e-5
    assert sp.snapshot()["pushed_samples"] == ref.size


def test_a_pitch_shifted_rate_is_snapped_so_the_stream_starts_at_once():
    from roboemot_reachymini.resample import resample
    from roboemot_reachymini.speaker import RATE_GRID_HZ

    mini, log = fake(out_rate=48000)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    rate = 23907  # Piper's 22050 Hz pitched up 1.4 semitones
    x = sine(rate)
    sp.push_audio_sample(x[: int(rate * 0.2)], rate)
    first = sum(p.size for p in log.pushed)
    assert first >= 48000 * (0.2 - 0.1)  # less than one playback chunk held back
    sp.push_audio_sample(x[int(rate * 0.2) :], rate)
    sp.end_stream()
    snapped = round(rate / RATE_GRID_HZ) * RATE_GRID_HZ
    ref = resample(x, snapped, 48000)
    out = np.concatenate(log.pushed)
    assert out.size == ref.size and np.abs(out - ref).max() < 1e-5


def test_a_stop_starts_a_fresh_stream():
    from roboemot_reachymini.resample import resample

    mini, log = fake(out_rate=48000)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    x = sine(22050, 0.5)
    sp.push_audio_sample(x, 22050)  # cut by a barge-in before its tail is flushed
    sp.stop()
    log.pushed.clear()
    sp.push_audio_sample(x, 22050)
    sp.end_stream()
    out, ref = np.concatenate(log.pushed), resample(x, 22050, 48000)
    assert out.size == ref.size and np.abs(out - ref).max() < 1e-5


def test_a_rate_change_ends_the_previous_stream_first():
    from roboemot_reachymini.resample import resample

    mini, log = fake(out_rate=48000)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    a, b = sine(22050, 0.3), sine(16000, 0.3)
    sp.push_audio_sample(a, 22050)
    sp.push_audio_sample(b, 16000)
    sp.end_stream()
    ref = np.concatenate([resample(a, 22050, 48000), resample(b, 16000, 48000)])
    out = np.concatenate(log.pushed)
    assert out.size == ref.size and np.abs(out - ref).max() < 1e-5
    assert sp.end_stream() == 0.0  # nothing left to flush


class ToneBackend:
    """One sentence of a 300 Hz tone at 0.5 amplitude, in 100 ms chunks."""

    name, controls = "tone", frozenset()

    async def synthesize(self, text, delivery, *, language, voice):
        t = np.arange(1600) / 16000
        for _ in range(5):
            yield AudioChunk((0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 16000)

    def available(self):
        return True

    def cancel(self):
        pass

    async def warm(self, language, voice):
        pass

    def apply_config(self, voice):
        pass

    async def aclose(self):
        pass


def test_the_echo_reference_holds_the_ducked_audio_the_speaker_really_played():
    """SPEC 8 stage 8: the echo guard compares the microphone with what played, after the duck."""
    mini, log = fake()
    ref = PlaybackReference(16000, 2.0)
    sp = ReachySpeaker(mini, 100, clock=time.monotonic, reference=ref)
    sp.start_playing()
    out = SpeechOutput(RoboEmotConfig(), sp, ToneBackend())
    out.duck = DuckGain(BargeInDuckConfig(gain=0.25))
    out.duck.duck(time.monotonic())  # open: within max_episode_s of now
    out.start()
    try:
        out.say(
            "Ciao.",
            language="it",
            register=RegisterParams(),
            circadian=StateEffects(),
            mood=None,
            intensity=0.0,
        )
        end = time.monotonic() + 3.0
        while out.sentences_spoken == 0 and time.monotonic() < end:
            time.sleep(0.005)
        assert out.sentences_spoken == 1
    finally:
        out.close()
    pushed = np.concatenate(log.pushed)
    assert float(np.abs(pushed).max()) == pytest.approx(0.125, abs=0.01)  # 0.5 at 0.25 gain
    until = sp.playing_until()
    played = ref.window(until - 0.5, until)
    assert float(np.abs(played).max()) == pytest.approx(0.125, abs=0.01)
