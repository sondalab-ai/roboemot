"""SpeechActivity: speaking until the playback end plus the echo tail (SPEC 4A, 8, 10)."""

import math
import threading

from roboemot.voice.activity import SpeechActivity


def test_silent_until_something_is_queued():
    act = SpeechActivity(300)
    assert not act.speaking(0.0) and act.until() == -math.inf


def test_speaking_through_the_playback_and_the_echo_tail_only():
    act = SpeechActivity(300)
    act.extend(10.0)
    assert act.speaking(5.0) and act.speaking(10.29)
    assert not act.speaking(10.3) and not act.speaking(11.0)
    assert act.until() == 10.3


def test_extend_never_moves_the_end_earlier():
    act = SpeechActivity(0)
    act.extend(10.0)
    act.extend(4.0)
    act.extend(math.nan)
    assert act.until() == 10.0


def test_reset_ends_playback_now_but_keeps_the_tail():
    act = SpeechActivity(300)
    act.extend(60.0)
    act.reset(5.0)
    assert act.speaking(5.2) and not act.speaking(5.3)
    act.extend(8.0)  # speaking again after a reset
    assert act.speaking(7.0)


def test_the_tail_can_be_reloaded():
    act = SpeechActivity(300)
    act.extend(1.0)
    act.set_echo_tail_ms(1000)
    assert act.speaking(1.9) and not act.speaking(2.0)


def test_concurrent_extends_keep_the_latest_end():
    act = SpeechActivity(0)
    threads = [threading.Thread(target=act.extend, args=(float(i),)) for i in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert act.until() == 49.0
