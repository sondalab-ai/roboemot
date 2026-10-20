"""Speaker: chunked non-blocking playback, queued duration and barge-in (SPEC 4A, 8)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.voice.activity import SpeechActivity
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
    start = clock.t
    w = ref.window(start, start + 1.0)
    assert w.size == 16000 and float(np.abs(w).max()) > 0.5  # at the reference rate
    clock.t += 0.5
    sp.stop()
    assert float(np.abs(ref.window(start + 0.6, start + 1.0)).max()) == 0.0  # never played
    assert float(np.abs(ref.window(start + 0.1, start + 0.4)).max()) > 0.5
