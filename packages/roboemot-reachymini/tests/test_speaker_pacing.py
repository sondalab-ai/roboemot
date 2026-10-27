"""Speaker pacing: a bounded lead in the SDK player, so barge-in ducking reaches the rest
(SPEC 4A, 8 stage 8, 13). ``pace_thread=False`` and ``feed()`` stand in for the pacer thread on
a fake clock; one test runs the real thread."""

from __future__ import annotations

import time

import numpy as np
import pytest
from roboemot.config.schema import BargeInDuckConfig, RegisterParams, RoboEmotConfig, StateEffects
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import PlaybackReference
from roboemot.voice.duck import DuckGain
from roboemot.voice.speech import SpeechOutput
from roboemot_reachymini.speaker import ReachySpeaker
from test_speaker import Clock, ToneBackend, fake

RATE = 16000
SECOND = np.ones(RATE, np.float32)


def paced(lead_ms=150, **kw):
    """A speaker with a fake clock and no pacer thread. The default lead (150 ms, not a
    multiple of the 100 ms piece) keeps the hand-off decisions off exact float boundaries."""
    mini, log = fake()
    clock = Clock()
    sp = ReachySpeaker(mini, 100, clock=clock, lead_ms=lead_ms, pace_thread=False, **kw)
    assert sp.start_playing()
    return sp, log, clock


def handed_s(log) -> float:
    return sum(len(p) for p in log.pushed) / RATE


def drain(sp, clock, step=0.1):
    """Run the pacer on time (every ``step``) until nothing is pending."""
    for _ in range(1000):
        if sp.pacing()["pending_ms"] == 0:
            return
        clock.t += step
        sp.feed()


def test_only_the_lead_is_handed_to_the_player_and_the_rest_follows_as_it_drains():
    sp, log, clock = paced()
    until = sp.push_audio_sample(SECOND, RATE)
    # at once (no wait for first audio): pieces while the player holds less than the lead,
    # so it ends up holding at most the lead plus one piece
    assert handed_s(log) == pytest.approx(0.2)
    assert until == pytest.approx(101.0) and sp.playing_until() == pytest.approx(101.0)
    clock.t += 0.1
    sp.feed()
    assert handed_s(log) == pytest.approx(0.3)
    drain(sp, clock)
    assert handed_s(log) == pytest.approx(1.0) and sp.pacing()["underruns"] == 0
    assert sp.playing_until() == pytest.approx(101.0)


def test_a_lead_of_zero_hands_everything_over_at_once_as_before():
    sp, log, _ = paced(lead_ms=0)
    sp.push_audio_sample(SECOND, RATE)
    assert handed_s(log) == pytest.approx(1.0) and sp.playing_until() == pytest.approx(101.0)


def test_a_short_dry_gap_keeps_the_sdk_timeline_and_counts_late():
    """Up to the SDK's 200 ms gap reset the SDK places the next buffer right after the last
    one (no DISCONT): the piece starts where the handed audio ended, and the gap is lost."""
    ref = PlaybackReference(RATE, 2.0)
    sp, log, clock = paced(reference=ref)
    sp.push_audio_sample(SECOND, RATE)  # handed up to 100.2
    clock.t += 0.27  # the pacer woke 70 ms after the player ran dry
    sp.feed()
    pacing = sp.pacing()
    assert (pacing["underruns"], pacing["late"], pacing["restamped"]) == (1, 1, 0)
    assert pacing["late_ms"] == pytest.approx(70) and pacing["underrun_ms"] == pytest.approx(70)
    assert sp.playing_until() == pytest.approx(101.0)  # on the SDK's timeline, not shifted
    assert float(np.abs(ref.window(100.2, 100.25)).min()) > 0.9  # stamped where the SDK put it


def test_a_dry_gap_past_the_sdk_gap_reset_restamps_to_now():
    ref = PlaybackReference(RATE, 2.0)
    sp, log, clock = paced(reference=ref)
    sp.push_audio_sample(SECOND, RATE)  # handed up to 100.2
    clock.t += 0.5  # 300 ms dry: the SDK restamps the next buffer at its running time
    sp.feed()
    pacing = sp.pacing()
    assert (pacing["underruns"], pacing["late"], pacing["restamped"]) == (1, 0, 1)
    assert pacing["restamped_ms"] == pytest.approx(300)
    assert sp.playing_until() == pytest.approx(101.3)  # the rest plays that much later
    assert float(np.abs(ref.window(100.21, 100.49)).max()) == 0.0  # silence in the gap
    assert float(np.abs(ref.window(100.5, 100.55)).min()) > 0.9


@pytest.mark.parametrize("lead_ms", [50, 99, 100, 150])
def test_a_lead_shorter_than_a_piece_never_leaves_gaps_and_the_pacer_terminates(lead_ms):
    """The pacer driven exactly on the due times it returns, as the thread waits on them."""
    sp, log, clock = paced(lead_ms=lead_ms)
    sp.push_audio_sample(SECOND, RATE)
    for _ in range(100):
        due = sp.feed()
        if due is None:
            break
        clock.t += due
    assert due is None and handed_s(log) == pytest.approx(1.0)
    assert sp.pacing()["underruns"] == 0


def test_stop_drops_the_pending_audio_and_reports_how_much():
    act = SpeechActivity(0)
    sp, log, clock = paced(activity=act)
    sp.push_audio_sample(SECOND, RATE)
    clock.t += 0.1
    sp.feed()
    sp.stop()
    pacing = sp.pacing()
    assert pacing["pending_at_stop_last_ms"] == pytest.approx(700)
    assert pacing["pending_ms"] == 0 and log.cleared == 1
    assert sp.playing_until() == clock.t and not act.speaking(clock.t + 0.01)
    clock.t += 1.0
    sp.feed()
    assert handed_s(log) == pytest.approx(0.3)  # nothing more after the stop


def test_the_echo_reference_is_stamped_at_hand_off_and_the_gate_at_accept():
    ref = PlaybackReference(RATE, 2.0)
    act = SpeechActivity(0)
    sp, log, clock = paced(reference=ref, activity=act)
    sp.push_audio_sample(SECOND, RATE)
    assert act.until() == pytest.approx(101.0)  # the whole answer, as before pacing
    assert float(np.abs(ref.window(100.2, 100.3)).max()) == 0.0  # not handed yet
    clock.t += 0.1
    sp.feed()
    assert float(np.abs(ref.window(100.2, 100.3)).min()) > 0.9


def test_a_pacer_stall_longer_than_the_echo_tail_keeps_the_speaking_window_open():
    act = SpeechActivity(300)
    sp, log, clock = paced(activity=act)
    sp.push_audio_sample(SECOND, RATE)  # handed up to 100.2
    clock.t += 0.6  # the pacer stalls well past the 300 ms tail
    assert act.speaking(clock.t)  # no unmute, no barge-in reset mid-answer
    sp.feed()  # it resumes (restamped): the window covers the shifted end too
    assert act.speaking(sp.playing_until() + 0.2)


def test_a_duck_reaches_audio_that_is_still_pending_and_the_reference_records_it():
    ref = PlaybackReference(RATE, 2.0)
    sp, log, clock = paced(reference=ref)
    duck = DuckGain(BargeInDuckConfig(gain=0.25))
    assert sp.take_duck(duck)
    sp.push_audio_sample(SECOND, RATE)  # handed: 100.0 to 100.2 at full gain
    duck.duck(100.05)  # a candidate: the user may be speaking
    clock.t += 0.1
    sp.feed()  # 100.2 to 100.3 handed now, ducked after the 10 ms attack
    assert float(np.abs(log.pushed[-1][160:]).max()) == pytest.approx(0.25)
    assert float(np.abs(ref.window(100.21, 100.3)).max()) == pytest.approx(0.25, abs=0.01)
    assert float(np.abs(ref.window(100.0, 100.04)).min()) == pytest.approx(1.0)


def test_the_gain_is_applied_once_when_the_speaker_takes_the_duck():
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=time.monotonic, lead_ms=200)
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
        while (out.sentences_spoken == 0 or sp.pacing()["pending_ms"] > 0) and (
            time.monotonic() < end
        ):
            time.sleep(0.01)
        assert out.sentences_spoken == 1 and sp.pacing()["pending_ms"] == 0
        assert "pacing" in out.snapshot()
    finally:
        out.close()
        sp.close()
    pushed = np.concatenate(log.pushed)
    assert float(np.abs(pushed).max()) == pytest.approx(0.125, abs=0.01)  # not 0.03


def test_the_pacer_thread_feeds_the_player_in_real_time_and_close_joins_it():
    mini, log = fake()
    sp = ReachySpeaker(mini, 100, clock=time.monotonic, lead_ms=150)
    sp.start_playing()
    sp.push_audio_sample(np.ones(RATE // 2, np.float32), RATE)
    assert handed_s(log) == pytest.approx(0.2)
    end = time.monotonic() + 2.0
    while handed_s(log) < 0.5 and time.monotonic() < end:
        time.sleep(0.01)
    assert handed_s(log) == pytest.approx(0.5)
    sp.close()
    assert not sp.pacer_alive()


def test_the_resampler_tail_of_end_stream_follows_the_pending_audio_in_order():
    sp, log, clock = paced()
    sp.push_audio_sample(np.ones(22050, np.float32), 22050)
    sp.end_stream()
    drain(sp, clock)
    assert handed_s(log) == pytest.approx(1.0, abs=0.002)
    assert sp.pacing()["pending_ms"] == 0


def test_a_failed_push_in_the_pacer_abandons_the_rest_of_that_push_only():
    sp, log, clock = paced()
    real = sp._mini.media.push_audio_sample
    fail = [True]

    def push(x):
        if len(log.pushed) == 2 and fail[0]:
            fail[0] = False
            raise RuntimeError("sdk")
        real(x)

    sp._mini.media.push_audio_sample = push
    sp.push_audio_sample(SECOND, RATE)  # pieces 1 and 2 handed, the rest pending
    clock.t += 0.1
    sp.feed()  # piece 3 fails: the pending rest of the sentence is dropped
    assert sp.pacing()["pending_ms"] == 0
    # the speech thread still holds the generation of that push: its next pieces are abandoned
    generation = sp._generation - 1
    assert not sp._push_pieces(np.ones(1600, np.float32), generation)
    sp.push_audio_sample(np.ones(1600, np.float32), RATE)  # the next sentence plays
    assert handed_s(log) == pytest.approx(0.3)


def test_a_restart_never_starts_a_second_pacer_while_the_first_is_alive(monkeypatch):
    import threading

    import roboemot_reachymini.speaker as speaker_module

    monkeypatch.setattr(speaker_module, "PACER_JOIN_S", 0.0)
    mini, _ = fake()
    sp = ReachySpeaker(mini, 100, clock=time.monotonic, lead_ms=150)
    sp.start_playing()
    first = sp._pacer
    with sp._lock:  # the pacer cannot exit while we hold its lock: close's join times out
        sp.close()
        assert first.is_alive()
        sp.start_playing()
        assert sp._pacer is first and sp.pacing()["pacer_reused"] == 1
    pacers = [t for t in threading.enumerate() if t.name == "speaker-pacer"]
    assert pacers == [first]
    monkeypatch.setattr(speaker_module, "PACER_JOIN_S", 1.0)
    sp.close()
    assert not sp.pacer_alive()
