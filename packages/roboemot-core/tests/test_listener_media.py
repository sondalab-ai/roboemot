"""SPEC 8 (Radio playback: lyrics, dance) with the voice listener's test rig: while the robot's
own radio plays, only the wake word addresses the robot, and with ``tools.radio.dance`` off the
music detector ignores music."""

from __future__ import annotations

from types import SimpleNamespace

from reachy_config import reachy_config
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.wake import WakeGate
from test_voice_listener import Rig, SpyMusic, silence


def cfg(policy="unless_enrolled_in_view", **tools):
    """The Reachy Mini run's config (the robot's name is its wake word) with ``policy`` and
    ``tools.radio`` overrides."""
    data = reachy_config().model_dump()
    data["voice"]["wake_word_policy"] = policy
    data["tools"]["radio"].update(tools)
    return RoboEmotConfig.model_validate(data)


def radio_rig(script, final=None, policy="unless_enrolled_in_view", **kw):
    """A rig whose listener is told the media player plays (``playing[0]``)."""
    r = Rig(script, final=final, cfg=cfg(policy), **kw)
    playing = [True]
    r.listener.media_playing = lambda: playing[0]
    return r, playing


def test_follow_up_window_does_not_address_while_media_plays():
    """Review Focus 3: the follow-up window would answer the lyrics of the robot's own radio."""
    r, playing = radio_rig(["ciao reati"], final="ciao reati metti il jazz")
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word  # the wake word opens it
    r.asr.script, r.asr.final_text = ["amore mio"], "amore mio"  # a lyric, inside the window
    r.say()
    assert not r.utterances[-1].addressed
    snap = r.listener.snapshot()["wake"]["decisions"]
    assert snap["rejected_media"] == 1
    playing[0] = False  # the same follow-up without the radio is addressed
    r.asr.script, r.asr.final_text = ["ciao reati"], "ciao reati"
    r.say()
    r.asr.script, r.asr.final_text = ["dimmi"], "dimmi di piu"
    r.say()
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word


def test_policy_and_a_face_do_not_address_while_media_plays():
    for policy, kw in (
        ("never", {}),
        ("unless_enrolled_in_view", {"enrolled_in_view": lambda now: True}),
    ):
        r, playing = radio_rig(["come stai"], policy=policy, **kw)
        seen = []
        r.listener.events = lambda kind, reason, seen=seen, **fields: seen.append((kind, reason))
        r.say()
        assert not r.utterances[-1].addressed
        assert ("unanswered", "media") in seen
        playing[0] = False
        r.say()
        assert r.utterances[-1].addressed


def test_the_wake_word_addresses_the_robot_while_media_plays():
    """The robot's name and a generic wake phrase both still work."""
    for final in ("reati spegni la radio", "ehi robot spegni la radio"):
        r, _ = radio_rig([final], final=final, policy="always")
        r.say()
        assert r.utterances[-1].addressed and r.utterances[-1].wake_word, final


def test_the_media_gate_never_lets_a_segment_stream_to_the_server():
    """Server turns: while the radio plays a segment stays local and the wake gate decides it."""
    r, playing = radio_rig(["ciao"])
    s = SimpleNamespace(start=r.t, face=None, face_info={})
    assert r.listener._stream_gate(s, r.t) == "media"
    playing[0] = False
    assert r.listener._stream_gate(s, r.t) != "media"


def test_dance_off_hides_music_while_playing():
    r = Rig(["ciao"], cfg=cfg(dance=False))
    playing = [True]
    r.listener.media_playing = lambda: playing[0]
    spy = r.listener.music = SpyMusic(True)
    r.feed(silence(0.5))
    assert spy.fed == [] and spy.gaps  # the detector ignores the robot's radio
    assert not r.listener._music_during(r.t - 0.5, r.t)
    assert not r.listener.music_state(r.t).playing
    playing[0] = False  # the radio stopped: music in the room counts again
    r.feed(silence(0.5))
    assert spy.fed
    assert r.listener._music_during(r.t - 0.5, r.t)


def test_dance_on_lets_the_detector_hear_the_robots_own_radio():
    r = Rig(["ciao"], cfg=cfg(dance=True))
    r.listener.media_playing = lambda: True
    spy = r.listener.music = SpyMusic(True)
    r.feed(silence(0.5))
    assert spy.fed  # it may start the dance, with tempo and beat from the microphone
    assert r.listener._music_during(r.t - 0.5, r.t)


def test_the_wake_gate_media_rule_counts_rejected_media():
    gate = WakeGate("never", 5.0, media=lambda: True)
    out = gate.outcome(wake_word=False, enrolled_in_view=True, start=1.0, last_speech=2.0)
    assert out == "rejected_media"
    assert not gate.addressable(enrolled_in_view=True, start=1.0)
    assert gate.outcome(wake_word=True, enrolled_in_view=False, start=3, last_speech=4) == (
        "wake_word"
    )


def test_a_long_name_led_command_with_degraded_doa_wakes_while_media_plays():
    """Controller ruling (review I5): the radio is the robot's own sound, so the sound direction
    is muted and the voice-at-face check runs degraded; while media plays the wake word is
    required anyway, so the name opening a long command still addresses the robot."""
    from test_voice_listener import OFF_FACE_RAD, direction_rig

    command = "reati metti una stazione jazz per favore adesso"
    r, _, _ = direction_rig(command, OFF_FACE_RAD, cfg=cfg("never"))
    r.say()
    assert not r.utterances[-1].addressed  # without the radio the check rejects it
    assert r.listener.snapshot()["name_first_no_face"] == 1
    r.listener.media_playing = lambda: True
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    assert r.listener.snapshot()["name_first_no_face"] == 1  # no second rejection


def test_the_announcements_reply_window_counts_while_a_station_is_set():
    """Review I2 (owner decision 2026-11-17): an announcement's reply window counts while a
    station is set (the radio stays paused in it), so "ok, thanks" without the name is
    answered and the timer's repeats stop; once it closes the lyrics gate applies again."""
    from test_voice_listener import FACE_RAD, direction_rig, say_text

    r, _, _ = direction_rig("ok grazie", FACE_RAD, cfg=cfg("always"))
    r.listener.media_playing = lambda: True
    r.feed(silence(0.5))
    assert not r.listener.reply_window_open()
    r.listener.open_follow_up()  # the announcement's audio ended
    assert r.listener.reply_window_open()  # the media player stays paused (test_media_player)
    s = SimpleNamespace(start=r.t, face=None, face_info={})
    assert r.listener._stream_gate(s, r.t) != "media"  # server turns hear the reply too
    r.say()
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word
    assert r.infos[-1].addressed  # percept.utterance: the announcer drops the repeats
    window = r.cfg.voice.fastpath.addressed_window_s
    r.feed(silence(window + 1.0))
    assert not r.listener.reply_window_open()  # closed: the radio rejoins
    say_text(r, "amore mio")
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1


def test_the_wake_gate_reply_window_over_media_and_a_wake_word_window_does_not():
    gate = WakeGate("always", 5.0, 120.0, media=lambda: True)
    gate.open_window(10.0)
    assert gate.reply_window(12.0)
    assert gate.addressable(enrolled_in_view=False, start=12.0)
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=12.0, last_speech=13.0)
    assert out == "window"  # a follow-up extends it, as any window
    assert gate.reply_window(17.5)
    assert not gate.reply_window(18.5)
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=19.0, last_speech=20.0)
    assert out == "rejected_media"  # closed: only the wake word over the radio
    gate.outcome(wake_word=True, enrolled_in_view=False, start=30.0, last_speech=31.0)
    assert not gate.reply_window(32.0)  # the wake word's window is not a reply window
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=32.0, last_speech=33.0)
    assert out == "rejected_media"
    gate.open_window(40.0)
    gate.close_window()  # a requested nap started
    assert not gate.reply_window(41.0)
