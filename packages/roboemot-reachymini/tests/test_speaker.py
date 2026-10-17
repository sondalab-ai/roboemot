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


def fake(out_rate=16000, clear=True, audio="present"):
    log = SimpleNamespace(pushed=[], started=0, stopped=0, cleared=0)
    audio_obj = SimpleNamespace(clear_player=lambda: setattr(log, "cleared", log.cleared + 1))
    if not clear:
        audio_obj = object()
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


def test_stop_falls_back_to_restarting_the_player_without_clear_player():
    mini, log = fake(clear=False)
    sp = ReachySpeaker(mini, 100, clock=Clock())
    sp.start_playing()
    sp.push_audio_sample(np.zeros(1600, np.float32), 16000)
    sp.stop()
    assert log.stopped == 1 and log.started == 2 and sp.queued_s() == 0.0


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
