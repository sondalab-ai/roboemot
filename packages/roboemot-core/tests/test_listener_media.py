"""SPEC 8 (Radio playback: lyrics, dance) with the voice listener's test rig: while the robot's
own radio plays, only the wake word addresses the robot, and with ``tools.radio.dance`` off the
music detector ignores music."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from audio_signals import drum_loop
from reachy_config import reachy_config
from roboemot.config.schema import RoboEmotConfig
from roboemot.media.player import MediaPlayer
from roboemot.voice.wake import WakeGate
from test_media_player import JAZZ, FakeOutput, FakeStream
from test_voice_listener import (
    FACE_RAD,
    RATE,
    Rig,
    SpyMusic,
    direction_rig,
    say_text,
    silence,
    voice,
)


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
    r.listener.media_audible = lambda at: playing[0]
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


LYRIC = "neva ferma la zebracanto"  # a local final over the radio the wake gate rejects


def live_rig(enabled=True, transcript=True, policy="never", radio=True, **live):
    """A rig with ``webui.live`` set, the radio audible or not, and its decision events kept in
    ``seen`` as ``(kind, reason, fields)``, as the listener passed them (before ``diag``)."""
    data = cfg(policy).model_dump()
    data["webui"]["live"].update(enabled=enabled, transcript=transcript, **live)
    r = Rig([LYRIC], cfg=RoboEmotConfig.model_validate(data))
    if radio:
        r.listener.media_audible = lambda at: True
    seen = []
    r.listener.events = lambda kind, reason, **fields: seen.append((kind, reason, fields))
    return r, seen


def test_a_local_media_reject_carries_the_local_words_with_the_live_transcript_on(caplog):
    """The owner's robot run (2026-11-18): "neve ferma la musica" over the radio was rejected by
    the local wake gate and the Live tab could not say what the local recognizer wrote. With the
    Live tab and its transcript on, ``rejected_media`` carries the local final as ``heard``,
    cut to ``webui.live.transcript_max_chars``, and no log has it (SPEC 8, 12). It is not the
    server's ``media_unconfirmed``. Replaces PR 273's test that this event carried no words."""
    caplog.set_level(logging.DEBUG)
    r, seen = live_rig(transcript_max_chars=80)
    r.say()
    assert not r.utterances[-1].addressed
    media = [f for k, why, f in seen if (k, why) == ("unanswered", "media")]
    assert media and all(f["heard"] == LYRIC for f in media)
    assert all(why != "media_unconfirmed" for _, why, _ in seen)
    assert "zebracanto" not in caplog.text


@pytest.mark.parametrize(
    ("enabled", "transcript"),
    [(True, False), (False, True), (False, False)],  # the transcript flag alone is not enough
)
def test_a_local_media_reject_carries_no_words_without_the_live_tab_and_its_transcript(
    enabled, transcript
):
    r, seen = live_rig(enabled=enabled, transcript=transcript)
    r.say()
    media = [f for k, why, f in seen if (k, why) == ("unanswered", "media")]
    assert media and all(f["heard"] is None for f in media)


@pytest.mark.parametrize(
    ("policy", "room_music", "reason"),
    [
        ("always", False, "no_window"),  # the wake word is required
        ("never", True, "music"),  # music in the room (the microphone), not the radio
    ],
)
def test_no_other_rejection_carries_words_even_with_the_live_transcript_on(
    policy, room_music, reason
):
    """SPEC 12: only the radio rejection carries the local final; without the radio the same
    utterance is rejected for another rule and its event holds no ``heard`` at all."""
    r, seen = live_rig(policy=policy, radio=False)
    if room_music:
        r.feed(drum_loop(120, 10.0, RATE) * 0.5)
        assert r.listener.music_state(r.now).playing
    r.say()
    assert not r.utterances[-1].addressed
    unanswered = [(why, f) for k, why, f in seen if k == "unanswered"]
    assert [why for why, _ in unanswered] == [reason]
    assert all("heard" not in f for _, f in unanswered)


@pytest.mark.parametrize(
    ("enabled", "transcript", "want"),
    [(True, True, "ehi neva"), (False, True, None), (True, False, None)],
)
def test_the_local_words_need_the_live_tab_and_its_transcript(enabled, transcript, want):
    data = cfg().model_dump()
    data["webui"]["live"].update(enabled=enabled, transcript=transcript)
    r = Rig(["ciao"], cfg=RoboEmotConfig.model_validate(data))
    assert r.listener._live_words("  ehi neva  ") == want


def test_the_local_words_are_cut_as_a_transcript_turn():
    r, _ = live_rig(transcript_max_chars=80)
    assert r.listener._live_words("x" * 200) == "x" * 80


@pytest.mark.parametrize(
    ("enabled", "transcript", "server", "want"),
    [
        (True, True, True, "ehi neva"),
        (False, True, True, None),  # the Live tab off: the transcript flag alone is not enough
        (True, False, True, None),
        (True, True, False, None),  # a local final carries no words
    ],
)
def test_the_heard_words_need_the_live_tab_its_transcript_and_a_server_final(
    enabled, transcript, server, want
):
    data = cfg().model_dump()
    data["webui"]["live"].update(enabled=enabled, transcript=transcript)
    r = Rig(["ciao"], cfg=RoboEmotConfig.model_validate(data))
    assert r.listener._heard("  ehi neva  ", server) == want


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
    r.listener.media_audible = lambda at: playing[0]
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
    """The robot (2026-11-17): behind the hardware echo canceller the microphone read -72.7
    dBFS while the radio played at -26.7 dBFS, so the dance never started. The detector now
    hears the radio's own pieces, stamped when they play, and never the microphone's chunks
    (AEC-cleaned room audio on another timeline): no microphone chunk is fed, no hole either."""
    r = Rig(["ciao"], cfg=cfg(dance=True))
    r.listener.media_audible = lambda at: True
    spy = r.listener.music = SpyMusic(True)
    piece = np.full(3200, 0.25, np.float32)  # 200 ms at 16 kHz
    r.listener.hear_media(piece, RATE, r.t + 0.1)
    r.listener.hear_media(piece, RATE, r.t + 0.3)
    r.feed(silence(0.5))
    assert spy.fed == [(pytest.approx(r.t - 0.4), 3200), (pytest.approx(r.t - 0.2), 3200)]
    assert spy.gaps == []
    assert r.listener._music_during(r.t - 0.5, r.t)  # it may start the dance


def test_dance_off_keeps_no_radio_piece():
    r = Rig(["ciao"], cfg=cfg(dance=False))
    r.listener.media_audible = lambda at: True
    spy = r.listener.music = SpyMusic(True)
    r.listener.hear_media(np.ones(3200, np.float32), RATE, r.t)
    r.feed(silence(0.5))
    assert spy.fed == [] and spy.gaps  # the microphone's chunks are holes, no radio piece
    assert len(r.listener._media_pieces) == 0


def test_the_microphone_feeds_the_detector_again_while_the_radio_is_held():
    """Held (the robot speaks, a turn is open), the radio plays nothing: the microphone hears
    the room again, from the end of the radio's last piece, after one hole."""
    r = Rig(["ciao"], cfg=cfg(dance=True))
    audible = [True]
    r.listener.media_audible = lambda at: audible[0]
    spy = r.listener.music = SpyMusic(True)
    r.listener.hear_media(np.ones(3200, np.float32), RATE, r.t + 0.1)  # plays until +0.3
    r.feed(silence(0.128))
    assert len(spy.fed) == 1
    radio_end = spy.fed[0][0] + 0.2
    audible[0] = False  # the robot answers: the player holds the radio
    r.listener.hear_media(np.ones(3200, np.float32), RATE, r.t + 0.5)  # stale: never fed
    r.feed(silence(0.5))
    mic = spy.fed[1:]
    assert mic and all(start >= radio_end for start, _ in mic)  # never back into the radio's
    assert spy.gaps == [pytest.approx(radio_end)]  # one hole where the radio's piece ended
    assert len(r.listener._media_pieces) == 0


def test_the_radio_reaches_the_detector_through_the_player_on_one_thread():
    """Wired as the run wires it: the player's feeder hands each piece it pushed to the
    listener (``on_piece``), which feeds the detector on its own thread only."""
    r = Rig(["ciao"], cfg=cfg(dance=True))
    player, _ = with_player(r)
    spy = r.listener.music = SpyMusic(True)
    fed_on: set[str] = set()
    feed = spy.feed

    def on_thread(samples, start):
        fed_on.add(threading.current_thread().name)
        feed(samples, start)

    spy.feed = on_thread
    pieces: list[float] = []
    hear = r.listener.hear_media

    def from_another_thread(samples, rate_hz, start):
        pieces.append(start)
        worker = threading.Thread(target=hear, args=(samples, rate_hz, start), name="media")
        worker.start()
        worker.join()

    player.on_piece = from_another_thread
    r.feed(silence(1.0))
    assert pieces and [s for s, _ in spy.fed] == pytest.approx(pieces[: len(spy.fed)])
    assert fed_on == {threading.current_thread().name}  # the listener's thread only


def test_the_wake_gate_media_rule_counts_rejected_media():
    gate = WakeGate("never", 5.0, media=lambda at: True)
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
    r.listener.media_audible = lambda at: True
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    assert r.listener.snapshot()["name_first_no_face"] == 1  # no second rejection


def test_the_announcements_reply_window_counts_while_a_station_is_set():
    """Review I2 (owner decision 2026-11-17): an announcement's reply window counts while a
    station is set (the radio stays paused in it), so "ok, thanks" without the name is
    answered and the timer's repeats stop; once it closes and the radio is audible again the
    lyrics gate applies again. Since the radio rule follows the player's ``audible_at`` (SPEC 8,
    Radio playback), the window counts because the player holds the radio in it, not through an
    exception of the wake gate: a real player on the rig's clock."""
    r, _, _ = direction_rig("ok grazie", FACE_RAD, cfg=cfg("always"))
    player, _ = with_player(r)
    r.feed(silence(0.5))
    assert player.audible() and not r.listener.reply_window_open()
    player.pause("announce")  # the announcement's speech holds the radio...
    r.listener.open_follow_up()  # ...and its reply window opens as its audio ends
    r.feed(silence(0.2))
    assert r.listener.reply_window_open() and not player.audible()
    s = SimpleNamespace(start=r.t, face=None, face_info={})
    assert r.listener._stream_gate(s, r.t) != "media"  # server turns hear the reply too
    r.say()
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word
    assert r.infos[-1].addressed  # percept.utterance: the announcer drops the repeats
    window = r.cfg.voice.fastpath.addressed_window_s
    r.feed(silence(window + r.cfg.tools.media.resume_after_s + 1.0))
    assert not r.listener.reply_window_open() and player.audible()  # the radio rejoined
    say_text(r, "amore mio")
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1


def test_the_wake_gate_media_rule_follows_the_players_audible_only():
    """The reply window is no exception of the gate any more (SPEC 8, Radio playback): the
    player holds the radio while it is open, so ``media(start)`` is False in it. The gate keeps
    ``reply_window`` only as the player's reason to hold."""
    held = [True]
    gate = WakeGate("always", 5.0, 120.0, media=lambda at: not held[0])
    gate.open_window(10.0)
    assert gate.reply_window(12.0)
    assert gate.addressable(enrolled_in_view=False, start=12.0)
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=12.0, last_speech=13.0)
    assert out == "window"  # a follow-up extends it, as any window
    assert gate.reply_window(17.5)
    assert not gate.reply_window(18.5)
    held[0] = False  # the window closed and the radio rejoined
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=19.0, last_speech=20.0)
    assert out == "rejected_media"  # only the wake word over the radio
    gate.outcome(wake_word=True, enrolled_in_view=False, start=30.0, last_speech=31.0)
    assert not gate.reply_window(32.0)  # the wake word's window is not a reply window
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=32.0, last_speech=33.0)
    assert out == "rejected_media"
    gate.open_window(40.0)
    gate.close_window()  # a requested nap started
    assert not gate.reply_window(41.0)
    gate.open_window(50.0)  # a reply window over audible music (no player hold): rejected
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=51.0, last_speech=52.0)
    assert out == "rejected_media"


class Turn:
    """The conversation as the media player sees it: ``open`` while an addressed turn is."""

    open = False

    def busy(self) -> bool:
        return self.open


class DrumStream:
    """A live stream with a steady beat (120 BPM), so the real music detector confirms music
    on the radio's own pieces, as on the robot."""

    def __init__(self) -> None:
        self.audio = drum_loop(120, 120.0, RATE)
        self.at = 0

    def read(self, frames: int) -> np.ndarray:
        piece = self.audio[self.at : self.at + frames]
        self.at += frames
        return piece

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


def with_player(r, steps=True, stream=FakeStream, echo_cancelled=False):
    """A real ``MediaPlayer`` on the rig's clock, wired to the listener as the run wires it
    (``runtime.startup.start_media``), playing a fake endless ``stream``; with ``steps`` its
    feeder steps while the rig feeds, at most as fast as the speaker plays (once the speaker
    holds at most one piece). Returns the player and the conversation's turn flag."""
    turn = Turn()
    host = SimpleNamespace(bus=None, conversation=turn, phase=None, voice_listener=r.listener)
    output = FakeOutput(lambda: r.now)
    player = MediaPlayer(
        output,
        host,
        lambda: r.cfg,
        opener=lambda url, rate_hz, timeout_s: stream(),
        clock=lambda: r.now,
        wait=lambda seconds: False,
        echo_cancelled=echo_cancelled,
    )
    piece_s = r.cfg.tools.media.chunk_ms / 1000.0
    r.listener.media_audible = player.audible_at
    r.listener.media_hold = player.hold_from
    r.listener.media_set = player.playing
    r.listener.media_speech_start = player.duck_from
    r.listener.media_speech_end = player.release_from
    player.on_piece = r.listener.hear_media
    process = r.listener.process

    def process_and_step(chunk):
        process(chunk)
        if output.until - r.now <= piece_s:
            player.step()

    if steps:
        r.listener.process = process_and_step
    assert player.play(JAZZ) == "ok"
    return player, turn


def test_a_command_after_the_name_is_a_follow_up_while_the_radio_is_held():
    """The robot (2026-11-17): "neve" ended at a short pause and woke the robot, the radio was
    held for the turn and the robot said "Eccomi", then "stoppa la musica" without the name was
    rejected as ``rejected_media``. While the radio is held no lyric can be heard, so it is a
    follow-up in the window the name opened; once the radio is audible again, a lyric is not."""
    r = Rig(["reati"], final="reati", cfg=cfg("always"))
    player, turn = with_player(r)
    r.feed(silence(0.5))
    assert player.audible()
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    turn.open = True  # the turn is served: the robot says "Eccomi"
    r.feed(silence(2.0))
    assert not player.audible()
    turn.open = False  # its reply ended
    r.feed(silence(0.2))
    say_text(r, "stoppa la musica")
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word
    decisions = r.listener.snapshot()["wake"]["decisions"]
    assert decisions["window"] == 1 and decisions["rejected_media"] == 0
    r.feed(silence(r.cfg.tools.media.resume_after_s + 0.5))
    assert player.audible()  # the radio rejoined: the window is still open, the lyrics are not
    say_text(r, "amore mio")
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1


def test_the_turn_holds_the_radio_from_the_names_last_speech():
    """The feeder sees the turn only at its next step, after the endpoint and the wake decision:
    words that start before it must still find the radio held (``hold_from``)."""
    r = Rig(["reati"], final="reati", cfg=cfg("always"))
    player, _ = with_player(r, steps=False)  # the feeder never runs
    r.feed(silence(0.5))
    r.say()
    first = r.utterances[-1]
    assert first.addressed and first.wake_word
    assert not player.audible_at(first.last_speech_stamp)
    say_text(r, "stoppa la musica")  # starts before the feeder saw the turn
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word


def test_speech_over_the_radio_ducks_it_from_its_start_to_after_its_end():
    """The robot run of 2026-11-18 (owner decision): behind a hardware echo canceller the VAD
    start ducks the audible radio, and it comes back up ``duck_release_s`` after the endpoint.
    A lyric line it rejects (no wake word) still ducked the music: the VAD cannot tell."""
    r = Rig(["amore mio"], final="amore mio", cfg=cfg("always"))
    player, _ = with_player(r, echo_cancelled=True)
    r.feed(silence(0.5))
    r.feed(voice(seconds=0.6))
    assert player.ducked()  # the VAD start reached the player
    r.feed(silence(r.cfg.voice.vad.end_silence_ms / 1000 + 0.2))
    assert not r.utterances[-1].addressed
    assert player.ducked()  # within duck_release_s of the endpoint
    r.feed(silence(r.cfg.tools.media.duck_release_s + 0.2))
    assert not player.ducked()
    assert player.snapshot()["ducks"] == 1
    assert player.audible()


def test_no_duck_over_the_radio_without_a_hardware_echo_canceller():
    r = Rig(["amore mio"], final="amore mio", cfg=cfg("always"))
    player, _ = with_player(r)
    r.feed(silence(0.5))
    r.feed(voice(seconds=0.6))
    assert not player.ducked()
    assert player.snapshot()["ducks"] == 0


def test_a_dropped_utterance_releases_the_duck():
    """An audio gap drops the speech in progress (no endpoint will come): the duck must not
    stay down."""
    r = Rig(["amore mio"], final="amore mio", cfg=cfg("always"))
    player, _ = with_player(r, echo_cancelled=True)
    r.feed(silence(0.5))
    r.feed(voice(seconds=0.6))
    assert player.ducked()
    r.feed(silence(0.1), gap=True)
    r.feed(silence(r.cfg.tools.media.duck_release_s + 0.2))
    assert not player.ducked()


def test_the_wake_word_over_a_ducked_radio_holds_it_and_it_rejoins_at_the_normal_level():
    r = Rig(["reati"], final="reati", cfg=cfg("always"))
    player, turn = with_player(r, echo_cancelled=True)
    output = player.output
    r.feed(silence(0.5))
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    assert not player.ducked()  # the hold took over from the duck
    turn.open = True
    r.feed(silence(1.0))
    turn.open = False
    pushed = len(output.samples)
    r.feed(silence(r.cfg.tools.media.resume_after_s + 1.0))
    rejoined = output.samples[pushed:]
    assert rejoined and all(np.allclose(p, 0.25) for p in rejoined)  # FakeStream's level
    assert player.snapshot()["ducks"] == 1


def test_an_utterance_inside_a_reply_window_over_a_station_never_deadlocks():
    """With ``tools.radio.dance`` off the wake gate's music predicate asked whether the radio
    played, which asked the reply window, which took the gate's own lock again: a hang on any
    utterance inside an announcement's reply window while a station was set. The predicate now
    takes only the player's leaf lock."""
    r, _, _ = direction_rig("ok grazie", FACE_RAD, cfg=cfg("always", dance=False))
    player, _ = with_player(r)
    r.feed(silence(0.5))
    player.pause("announce")
    r.listener.open_follow_up()
    r.feed(silence(0.2))
    done = threading.Event()

    def speak():
        r.say()
        done.set()

    thread = threading.Thread(target=speak, daemon=True)
    thread.start()
    thread.join(5.0)
    assert done.is_set()
    assert r.utterances[-1].addressed


def test_music_the_detector_heard_on_the_radio_never_rejects_a_held_follow_up():
    """With the detector hearing the radio's own pieces it confirms music on the robot's jazz
    and carries it through the robot's reply (a hole does not count toward the release). That
    music is the radio's: while the radio is audible the media rule covers it, and while it is
    held no lyric can be heard, so it must not turn "stoppa la musica" into rejected_music."""
    r = Rig(["reati"], final="reati", cfg=cfg("always"))
    player, turn = with_player(r, stream=DrumStream)
    r.feed(silence(10.0))
    assert r.listener.music_state(r.now).playing  # the dance may start on the radio
    r.say()
    assert r.utterances[-1].addressed and r.utterances[-1].wake_word
    turn.open = True  # the robot says "Eccomi"
    r.feed(silence(2.0))
    turn.open = False
    r.feed(silence(0.2))
    assert r.listener.music_state(r.now).playing  # the flywheel keeps the dance's beat
    say_text(r, "stoppa la musica")
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word
    decisions = r.listener.snapshot()["wake"]["decisions"]
    assert decisions["rejected_music"] == 0 and decisions["window"] == 1
    r.feed(silence(r.cfg.tools.media.resume_after_s + 0.5))
    say_text(r, "amore mio")  # the radio is audible again: a lyric is rejected_media
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1


def test_the_reply_window_over_a_radio_the_detector_danced_to():
    r, _, _ = direction_rig("ok grazie", FACE_RAD, cfg=cfg("always"))
    player, _ = with_player(r, stream=DrumStream)
    r.feed(silence(10.0))
    assert r.listener.music_state(r.now).playing
    player.pause("announce")
    r.listener.open_follow_up()
    r.feed(silence(0.2))
    r.say()
    assert r.utterances[-1].addressed and not r.utterances[-1].wake_word
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_music"] == 0


def test_room_music_from_the_microphone_still_needs_the_wake_word():
    """Music the detector heard from the microphone keeps the music rule (2026-11-04)."""
    r = Rig(["come stai"], cfg=cfg("never"))
    r.feed(drum_loop(120, 10.0, RATE) * 0.5)
    assert r.listener.music_state(r.now).playing
    r.say()
    assert not r.utterances[-1].addressed
    assert r.listener.snapshot()["wake"]["decisions"]["rejected_music"] == 1


def test_room_music_right_after_the_radio_counts_again():
    """Review: the radio's music must stop being ignored once the microphone is the detector's
    source again, so music in the room after the radio still needs the wake word."""
    r = Rig(["come stai"], cfg=cfg("never"))
    player, _ = with_player(r, stream=DrumStream)
    r.feed(silence(10.0))
    assert r.listener.music_state(r.now).playing
    assert not r.listener._music_during(r.now - 1.0, r.now)  # the radio's: the media rule's
    player.stop("user")
    r.feed(drum_loop(120, 3.0, RATE) * 0.5)  # a song in the room, no hole long enough to stop
    assert r.listener.music_state(r.now).playing
    assert r.listener._music_during(r.now - 1.0, r.now)


def test_turning_the_dance_off_drops_the_waiting_radio_pieces():
    r = Rig(["ciao"], cfg=cfg(dance=True))
    r.listener.media_audible = lambda at: True
    spy = r.listener.music = SpyMusic(True)
    r.listener.hear_media(np.ones(3200, np.float32), RATE, r.t)
    r.listener.apply_config(cfg(dance=False))
    r.listener.apply_config(cfg(dance=True))
    r.feed(silence(0.1))
    assert spy.fed == []  # nothing stale after it is turned on again


def test_a_hot_reload_while_the_player_appends_never_raises():
    """Review: the reload rebuilt the deque by iterating the live one while the player's thread
    appended (RuntimeError: deque mutated during iteration)."""
    r = Rig(["ciao"], cfg=cfg(dance=True))
    r.listener.music = SpyMusic(True)
    sizes = [
        cfg(dance=True),
        RoboEmotConfig.model_validate(
            {**cfg(dance=True).model_dump(), "tools": {"media": {"chunk_ms": 100}}}
        ),
    ]
    stop, errors = threading.Event(), []
    piece = np.ones(16, np.float32)

    def feeder():
        try:
            while not stop.is_set():
                r.listener.hear_media(piece, RATE, 0.0)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=feeder, daemon=True)
    thread.start()
    try:
        for i in range(300):
            r.listener.apply_config(sizes[i % 2])
    finally:
        stop.set()
        thread.join(5.0)
    assert errors == []
