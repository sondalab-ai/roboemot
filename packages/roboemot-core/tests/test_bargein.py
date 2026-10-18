"""SPEC 8 stage 8 / 7A / 13: barge-in over the robot's own voice, with the echo guard, the stop
keyword path and the listener handoff, on a fake microphone that hears the robot's playback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.voice.bargein import BargeIn, PlaybackReference, echo_correlation
from roboemot.voice.vad import EnergyVAD
from test_voice_listener import CHUNK, FRAME, RATE, T0, Rig

ECHO_LAG_S = 0.08
ECHO_GAIN = 0.5
ROBOT_RMS = 0.2


def colored(seconds, seed, rms):
    """Band-limited noise: speech-like in that it is not periodic or white."""
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(round(seconds * RATE) + 40), np.ones(8) / 8, "same")
    x = x[20:-20].astype(np.float32)
    return (x * rms / float(np.sqrt(np.mean(x * x)))).astype(np.float32)


def user_voice(seconds, rms):
    t = np.arange(round(seconds * RATE)) / RATE
    x = (
        np.sin(2 * np.pi * 180 * t)
        + 0.5 * np.sin(2 * np.pi * 360 * t + 1)
        + 0.25 * np.sin(2 * np.pi * 540 * t + 2)
    )
    return (x * rms / float(np.sqrt(np.mean(x * x)))).astype(np.float32)


class World:
    """A fake microphone in a room with the robot's speaker. The robot plays ``robot`` from
    ``play_start``; the microphone hears it ``ECHO_LAG_S`` later at ``ECHO_GAIN``, until the
    stop. The reference is written the way the speaker writes it."""

    def __init__(self, script=("ciao",), cfg=None, reference=True, hear_echo=True, **bargein):
        self.cfg = cfg or RoboEmotConfig()
        self.rig = Rig(
            script,
            cfg=self.cfg,
            on_speaking_chunk=lambda c: self.bargein.on_chunk(c),
            on_speaking_end=lambda: self.bargein.reset(),
        )
        self.reference = PlaybackReference(RATE, self.cfg.voice.barge_in.echo_reference_s)
        self.events: list[str] = []
        self.stops: list[float] = []
        self.interrupts = 0
        self.hear_echo = hear_echo
        self.robot = np.empty(0, np.float32)
        self.play_start = self.play_end = self.stopped_at = None
        self.bargein = BargeIn(
            self.cfg,
            EnergyVAD(RATE, FRAME),
            self.rig.asr,
            self.reference if reference else None,
            language=lambda: "it",
            emit=lambda name, now, payload: self.events.append(name),
            clock=lambda: self.rig.now,
            **bargein,
        )
        self.bargein.bind(self.stop, self.interrupt, self.rig.listener.resume)

    def stop(self):
        now = self.rig.now
        self.stops.append(now)
        self.stopped_at = now
        self.rig.speech.reset(now)  # what ReachySpeaker.stop does
        self.reference.truncate(now)

    def interrupt(self):
        self.interrupts += 1

    def speak(self, seconds):
        """Queue ``seconds`` of the robot's voice (pushed ahead of time, as the speaker does)."""
        self.robot = colored(seconds, 1, ROBOT_RMS)
        self.play_start = self.rig.t
        self.play_end = self.play_start + seconds
        self.stopped_at = None
        self.reference.add(self.play_start, self.robot)
        self.rig.speech.extend(self.play_end)

    def _echo(self, t0, n):
        out = np.zeros(n, np.float32)
        if self.play_start is None or not self.hear_echo:
            return out
        end = self.play_end if self.stopped_at is None else min(self.play_end, self.stopped_at)
        for i in range(n):
            t = t0 + i / RATE - ECHO_LAG_S
            if self.play_start <= t < end:
                out[i] = (
                    ECHO_GAIN
                    * self.robot[min(round((t - self.play_start) * RATE), self.robot.size - 1)]
                )
        return out

    def feed(self, seconds, user=None, user_at=None):
        """Run the microphone for ``seconds``; ``user`` (samples) starts at ``user_at`` seconds
        from now."""
        rig = self.rig
        n_total = round(seconds * RATE)
        origin = rig.t
        for i in range(0, n_total, CHUNK):
            n = min(CHUNK, n_total - i)
            t0 = rig.t
            mic = self._echo(t0, n) + 0.0005 * np.random.default_rng(i).standard_normal(n).astype(
                np.float32
            )
            if user is not None:
                lo = round(((origin + (user_at or 0.0)) - t0) * RATE)
                for j in range(n):
                    k = j - lo
                    if 0 <= k < user.size:
                        mic[j] += user[k]
            end = t0 + n / RATE
            rig.now = end
            rig.listener.process(
                SimpleNamespace(samples=mic.astype(np.float32), rate_hz=RATE, stamp=end, gap=False)
            )
            rig.t = end


def test_the_robots_own_playback_in_the_microphone_is_not_a_barge_in():
    w = World()
    w.speak(3.0)
    w.feed(3.0)
    snap = w.bargein.snapshot()
    assert w.stops == [] and w.interrupts == 0 and snap["triggers"] == 0
    assert snap["echo_frames"] > 20  # the VAD saw speech all along: the echo guard held it back
    assert w.rig.asr.started == [] and w.rig.listener.snapshot()["dropped_speaking"] > 0


def test_the_echo_alone_would_have_triggered_without_the_guard():
    cfg = RoboEmotConfig.model_validate({"voice": {"barge_in": {"echo_corr_max": 1.0}}})
    w = World(cfg=cfg)
    w.speak(3.0)
    w.feed(3.0)
    assert len(w.stops) == 1  # the guard is what keeps the robot from interrupting itself


def test_user_voice_over_the_robot_cuts_it_off_within_the_budget_and_the_words_are_kept():
    w = World(["ciao", "ciao robot come stai"])
    w.speak(5.0)
    w.feed(1.0)
    onset = w.rig.t + 0.5
    w.feed(2.0, user_voice(1.5, 0.21), user_at=0.5)
    assert len(w.stops) == 1 and w.interrupts == 1
    snap = w.bargein.snapshot()
    assert snap["triggers"] == 1
    assert snap["latency"]["onset_to_stop"]["last_ms"] < 200  # SPEC 13
    assert w.stops[0] - onset < 0.2
    w.feed(2.0)  # the user stopped talking
    (u,) = w.rig.utterances
    assert u.text == "ciao robot come stai"
    assert u.start_stamp <= onset + 0.1  # the first words were not lost to the speaking gate
    assert w.rig.listener.snapshot()["barge_ins"] == 1


def test_the_robot_going_quiet_after_a_barge_in_closes_the_gate_again():
    w = World(["ciao"])
    w.speak(5.0)
    w.feed(1.0)
    w.feed(1.5, user_voice(1.0, 0.21), user_at=0.2)
    w.feed(1.5)
    assert len(w.rig.utterances) == 1
    w.speak(3.0)  # the next answer: its echo must not be taken for the user
    w.feed(3.0)
    assert len(w.rig.utterances) == 1 and len(w.stops) == 1


def test_a_quiet_user_below_the_threshold_does_not_cut_the_robot():
    w = World()
    w.speak(4.0)
    w.feed(
        4.0, user_voice(3.0, 0.05), user_at=0.5
    )  # probability 0.5: under voice.barge_in.threshold
    assert w.stops == []


def test_bound_reports_whether_the_actions_are_connected():
    w = World()
    assert w.bargein.bound and w.bargein.snapshot()["bound"] is True
    w.bargein._stop = None  # as before ``bind``
    assert not w.bargein.bound and w.bargein.snapshot()["bound"] is False


def test_barge_in_can_be_disabled():
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"barge_in": {"enabled": False, "keyword_enabled": False}}}
    )
    w = World(cfg=cfg)
    w.speak(4.0)
    w.feed(3.0, user_voice(2.0, 0.21), user_at=0.5)
    assert w.stops == [] and w.interrupts == 0 and w.rig.asr.started == []


def test_without_a_reference_there_is_no_echo_evidence_so_loud_speech_counts():
    w = World(reference=False, hear_echo=False)
    w.speak(4.0)
    w.feed(2.0, user_voice(1.5, 0.21), user_at=0.3)
    assert len(w.stops) == 1


# ------------------------------------------------------------------ the stop keyword


def quiet_stop_world(script, hear_echo=False):
    """The robot speaks; the user says something at probability 0.65: above the keyword
    candidate threshold (0.5), below the barge-in one (0.8)."""
    w = World(script, hear_echo=hear_echo)
    w.speak(4.0)
    return w


def test_stop_is_heard_while_speaking_even_below_the_barge_in_threshold():
    w = quiet_stop_world(["stop"])
    w.feed(2.0, user_voice(1.0, 0.065), user_at=0.3)
    assert len(w.stops) == 1 and w.interrupts == 1
    assert w.events == ["speech.keyword.stop"]
    snap = w.bargein.snapshot()
    assert snap["triggers"] == 0 and snap["keyword_triggers"] == 1
    assert w.rig.listener.snapshot()["barge_ins"] == 0  # a command, not an utterance to answer


def test_another_word_does_not_stop_it_and_the_recognizer_is_released():
    w = quiet_stop_world(["ciao"])
    w.feed(3.0, user_voice(1.0, 0.065), user_at=0.3)
    assert w.stops == [] and w.events == []
    assert w.rig.asr.resets >= 1  # the session ended with the speech


def test_the_robot_saying_stop_does_not_stop_itself():
    w = quiet_stop_world(["stop"], hear_echo=True)  # loud echo of its own voice
    w.feed(3.5)
    assert w.stops == [] and w.rig.asr.started == []


def test_the_keyword_path_can_be_switched_off():
    cfg = RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}})
    w = World(["stop"], cfg=cfg, hear_echo=False)
    w.speak(4.0)
    w.feed(2.0, user_voice(1.0, 0.065), user_at=0.3)
    assert w.stops == [] and w.rig.asr.started == []


def test_a_gap_in_the_audio_drops_the_stretch_in_progress():
    w = World()
    w.speak(4.0)
    w.feed(0.4, user_voice(1.0, 0.21), user_at=0.0)  # a barge-in is half-way: 2 frames of 3
    w.bargein.on_chunk(
        SimpleNamespace(samples=np.zeros(CHUNK, np.float32), rate_hz=RATE, stamp=w.rig.t, gap=True)
    )
    assert w.bargein._run == 0


# ------------------------------------------------------------------ the pieces


def test_echo_correlation_finds_a_delayed_copy_and_ignores_the_rest():
    ref = colored(1.0, 7, 0.2)
    mic = 0.5 * ref[4000 : 4000 + 2048]  # a copy, somewhere in the reference
    assert echo_correlation(mic, ref) > 0.98
    other = colored(1.0, 8, 0.2)[:2048]
    assert echo_correlation(other, ref) < 0.3
    mixed = mic + colored(1.0, 9, 0.5)[:2048]  # the copy under louder, different audio
    assert 0.2 < echo_correlation(mixed, ref) < 0.6
    assert echo_correlation(np.zeros(2048, np.float32), ref) == 0.0
    assert echo_correlation(mic, np.zeros(RATE, np.float32)) == 0.0
    assert echo_correlation(mic, ref[:100]) == 0.0


def test_the_reference_is_a_timeline_with_silence_and_a_cut():
    r = PlaybackReference(1000, keep_s=1.0)
    r.add(10.0, np.ones(500, np.float32))  # plays 10.0 to 10.5
    r.add(10.5, np.full(500, 2.0, np.float32))
    w = r.window(9.9, 10.6)
    assert w.size == 700 and w[:100].sum() == 0 and w[100] == 1.0 and w[-1] == 2.0
    assert r.window(10.4, 10.6).tolist() == [1.0] * 100 + [2.0] * 100
    r.truncate(10.55)  # flushed 50 ms into the second piece
    assert r.window(10.5, 10.7).tolist() == [2.0] * 50 + [0.0] * 150
    r.add(20.0, np.ones(10, np.float32))  # old audio falls out of the keep window
    assert r.window(10.0, 10.5).sum() == 0


def test_the_config_rejects_a_reference_too_short_for_the_echo_window():
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate(
            {"voice": {"barge_in": {"echo_reference_s": 0.5, "echo_max_lag_ms": 600}}}
        )


def test_the_snapshot_has_numbers_only():
    w = World()
    w.speak(2.0)
    w.feed(1.0)
    snap = w.bargein.snapshot()
    assert set(snap) >= {"enabled", "triggers", "keyword_triggers", "echo_frames", "latency"}
    assert T0 > 0


def test_a_config_reload_between_chunks_keeps_working():
    w = World()
    w.speak(3.0)
    w.feed(0.5)
    cfg = RoboEmotConfig.model_validate(
        {"voice": {"barge_in": {"echo_window_ms": 256, "min_speech_ms": 64}}}
    )
    w.bargein.apply_config(cfg)
    w.feed(2.0, user_voice(1.0, 0.21), user_at=0.3)
    assert len(w.stops) == 1


def test_the_conversation_is_told_before_the_speech_is_stopped():
    w = World()
    order = []
    w.bargein.bind(
        lambda: (order.append("stop"), w.stop()),
        lambda: (order.append("interrupt"), w.interrupt()),
        w.rig.listener.resume,
        lambda: order.append("prepare"),
    )
    w.speak(4.0)
    w.feed(0.5)
    w.feed(1.5, user_voice(1.0, 0.3), user_at=0.2)
    assert order == ["prepare", "stop", "interrupt"]


class ReverbWorld(World):
    """The room smears the echo: two later reflections and a low-pass, as a small room does."""

    def _echo(self, t0, n):
        dry = super()._echo(t0, n)
        wet = dry + 0.5 * super()._echo(t0 - 0.03, n) + 0.25 * super()._echo(t0 - 0.07, n)
        return np.convolve(wet, np.ones(8) / 8, "same").astype(np.float32)


def test_the_documented_limit_a_user_quieter_than_the_echo_is_not_heard():
    """Pins the echo guard's known limit (SPEC 8 stage 8): with a smeared echo, a user whose
    power is far below the echo's is taken for echo; a loud one still gets through."""
    quiet = ReverbWorld()
    quiet.speak(4.0)
    quiet.feed(3.0, user_voice(2.5, 0.04), user_at=0.3)  # power 1/9 of the echo's
    assert quiet.stops == []
    loud = ReverbWorld()
    loud.speak(4.0)
    loud.feed(3.0, user_voice(2.5, 0.4), user_at=0.3)  # about ten times the echo's power
    assert len(loud.stops) == 1


def test_the_snapshot_counts_the_audio_fed_to_the_keyword_recognizer():
    w = World(["ciao"], hear_echo=False)
    w.speak(4.0)
    w.feed(2.0, user_voice(1.0, 0.065), user_at=0.3)
    snap = w.bargein.snapshot()
    assert 0.5 < snap["keyword_seconds"] < 3.0
