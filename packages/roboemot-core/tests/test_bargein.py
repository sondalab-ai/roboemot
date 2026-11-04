"""SPEC 8 stage 8 / 7A / 13: barge-in over the robot's own voice, with the echo guard, the stop
keyword path and the listener handoff, on a fake microphone that hears the robot's playback."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.motion.state import MotionSample, MotionState
from roboemot.voice.bargein import LABEL_MAX, BargeIn, PlaybackReference, echo_correlation
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
            before_asr_reset=lambda: self.bargein.handback_asr(),
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


# ------------------------------------------------------------------ the onset path (fricatives)


class ScriptedVAD:
    """Scores frame ``i`` as ``probs[i]`` (0 after the script)."""

    rate_hz = RATE
    frame_samples = FRAME

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def speech_prob(self, frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        pass


def hiss(frames, rms=0.02, seed=5):
    """High-pass noise (an /s/): nearly all its energy is above 2 kHz."""
    x = np.random.default_rng(seed).standard_normal(frames * FRAME)
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(x.size, 1 / RATE) < 3000] = 0
    x = np.fft.irfft(spec, x.size)
    return (x * rms / np.std(x)).astype(np.float32)


def onset_run(probs, audio, reference=None, motion=None, face_offset=None, **cfg):
    """Feed ``audio`` in microphone chunks through a scripted VAD; return (stops, BargeIn)."""
    config = (
        RoboEmotConfig.model_validate({"voice": {"barge_in": cfg}}) if cfg else RoboEmotConfig()
    )
    now = [0.0]
    stops: list[float] = []
    bi = BargeIn(
        config,
        ScriptedVAD(probs),
        SimpleNamespace(
            start=lambda *a: None, accept=lambda a: None, reset=lambda: None, finish=lambda: ""
        ),
        reference,
        language=lambda: "en",
        clock=lambda: now[0],
        motion=motion,
        face_offset=face_offset,
    )
    bi.bind(lambda: stops.append(now[0]), lambda: None, None)
    for i in range(0, audio.size, CHUNK):
        part = audio[i : i + CHUNK]
        now[0] = (i + part.size) / RATE
        bi.on_chunk(SimpleNamespace(samples=part, rate_hz=RATE, stamp=now[0], gap=False))
    return stops, bi


def test_a_fricative_onset_scored_between_the_thresholds_cuts_in_two_frames():
    stops, bi = onset_run([0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False)
    assert len(stops) == 1 and bi.snapshot()["triggers"] == 1
    assert stops[0] <= 2 * FRAME / RATE + CHUNK / RATE  # two frames, one chunk of slack


def test_the_same_scores_on_a_voiced_low_frequency_sound_do_not_cut():
    stops, _ = onset_run([0.5] * 8, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    assert stops == []


def test_a_fricative_that_ends_before_two_frames_or_scores_too_low_does_not_cut():
    stops, _ = onset_run([0.5, 0.1, 0.6, 0.1, 0.5, 0.1], hiss(6), keyword_enabled=False)
    assert stops == []
    stops, _ = onset_run([0.3] * 6, hiss(6), keyword_enabled=False)
    assert stops == []


def test_a_click_is_one_frame_even_at_a_high_score():
    audio = np.zeros(8 * FRAME, np.float32)
    audio[FRAME * 2 : FRAME * 2 + 4] = 0.9
    stops, _ = onset_run([0.0, 0.0, 0.95, 0.0], audio, keyword_enabled=False)
    assert stops == []


def test_the_onset_path_can_be_switched_off():
    stops, _ = onset_run([0.5, 0.6, 0.55, 0.5], hiss(4), onset_enabled=False, keyword_enabled=False)
    assert stops == []


def high_band(x: np.ndarray, cutoff_hz: float = 2000.0) -> np.ndarray:
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(x.size, 1 / RATE) < cutoff_hz] = 0
    return np.fft.irfft(spec, x.size).astype(np.float32)


def test_the_robots_own_hiss_in_the_microphone_is_dropped_by_the_high_band_guard():
    audio = hiss(4)
    ref = PlaybackReference(RATE, 2.0)
    ref.add(0.0, audio)
    # The premise, measured: the copy correlates in the high band, an independent hiss does not.
    assert echo_correlation(high_band(audio), high_band(audio)) > 0.99
    assert echo_correlation(high_band(hiss(4, seed=9)), high_band(audio)) < 0.4
    stops, bi = onset_run([0.5, 0.6, 0.55, 0.5], audio, ref, keyword_enabled=False)
    assert stops == [] and bi.snapshot()["onset_echo_frames"] == 4
    stops, bi = onset_run(
        [0.5, 0.6, 0.55, 0.5], hiss(4, seed=9), ref, keyword_enabled=False
    )  # a user's hiss while the robot plays something else
    assert len(stops) == 1 and bi.snapshot()["onset_echo_frames"] == 0


def test_a_confident_frame_with_a_middling_echo_correlation_cuts_on_the_main_path_only():
    robot = colored(0.5, 3, 0.1)[: 6 * FRAME]
    mic = 0.3 * robot + 0.5 * user_voice(6 * FRAME / RATE, 0.1)
    ref = PlaybackReference(RATE, 2.0)
    ref.add(0.0, robot)
    corr = echo_correlation(mic, robot)
    b = RoboEmotConfig().voice.barge_in
    assert b.onset_echo_corr_max <= corr < b.echo_corr_max, corr  # measured, not assumed
    # The main path needs five frames here (150 ms); the onset path would cut after two.
    cfg = {"keyword_enabled": False, "min_speech_ms": 150}
    stops, bi = onset_run([0.9] * 6, mic, ref, **cfg)
    assert len(stops) == 1 and stops[0] > 4 * FRAME / RATE  # the main path, at the fifth frame
    assert bi.snapshot()["onset_echo_frames"] >= 3  # the onset path's guard held
    stops, _ = onset_run([0.9] * 6, mic, None, **cfg)  # no reference: nothing to be stricter on
    assert len(stops) == 1 and stops[0] <= 4 * FRAME / RATE


def tones(frames: int, hf_share: float, rms: float = 0.02) -> np.ndarray:
    """A 500 Hz and a 4 kHz tone whose energy shares are ``1 - hf_share`` and ``hf_share``."""
    t = np.arange(frames * FRAME) / RATE
    x = np.sqrt(1 - hf_share) * np.sin(2 * np.pi * 500 * t) + np.sqrt(hf_share) * np.sin(
        2 * np.pi * 4000 * t
    )
    return (x * rms / np.std(x)).astype(np.float32)


def test_the_fricative_test_flips_around_the_high_frequency_share_of_0_7():
    stops, _ = onset_run([0.5] * 4, tones(4, 0.74), keyword_enabled=False)
    assert len(stops) == 1
    stops, _ = onset_run([0.5] * 4, tones(4, 0.66), keyword_enabled=False)
    assert stops == []


def test_silent_and_constant_frames_are_never_fricatives():
    for audio in (np.zeros(6 * FRAME, np.float32), np.full(6 * FRAME, 0.05, np.float32)):
        stops, bi = onset_run([0.5] * 6, audio, keyword_enabled=False)
        assert stops == [] and bi.snapshot()["errors"] == 0


def test_a_fricative_under_the_level_floor_is_not_counted():
    stops, _ = onset_run([0.5] * 4, hiss(4, rms=0.0005), keyword_enabled=False)  # -66 dBFS
    assert stops == []
    stops, _ = onset_run([0.5] * 4, hiss(4, rms=0.02), keyword_enabled=False)  # -34 dBFS
    assert len(stops) == 1


def test_a_run_does_not_survive_the_end_of_speech_and_nothing_fires_after_the_stop():
    audio = hiss(12)
    now = [0.0]
    stops: list[float] = []
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.5] * 12),
        SimpleNamespace(start=lambda *a: None, accept=lambda a: None, reset=lambda: None),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: stops.append(now[0]), lambda: None, None)

    def feed(lo, hi):
        for i in range(lo * FRAME, hi * FRAME, FRAME):
            now[0] = (i + FRAME) / RATE
            bi.on_chunk(
                SimpleNamespace(samples=audio[i : i + FRAME], rate_hz=RATE, stamp=now[0], gap=False)
            )

    feed(0, 1)  # one frame of an /s/: the robot stops speaking here (the listener's reset)
    bi.reset()
    assert bi._weak_run == 0 and bi._run == 0
    feed(1, 2)  # a second frame in the tail after the last sentence: not a run of two
    assert stops == []
    feed(2, 4)  # two fresh ones do cut
    assert len(stops) == 1
    feed(4, 12)  # the echo tail of the cut: the detector is deaf, nothing fires again
    assert len(stops) == 1


def test_the_onset_thresholds_may_not_be_stricter_than_the_main_path():
    with pytest.raises(ValueError, match="onset_threshold"):
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"onset_threshold": 0.9}}})
    with pytest.raises(ValueError, match="onset_echo_corr_max"):
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"onset_echo_corr_max": 0.7}}})


def test_the_pre_roll_of_an_onset_cut_starts_before_the_fricative():
    audio = np.concatenate([np.zeros(2 * FRAME, np.float32), hiss(4)])
    handed = []
    now = [0.0]
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.0, 0.0, 0.5, 0.6, 0.55, 0.5]),
        SimpleNamespace(start=lambda *a: None, accept=lambda a: None, reset=lambda: None),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: None, lambda: None, lambda samples, stamp: handed.append((samples, stamp)))
    for i in range(0, audio.size, CHUNK):
        now[0] = (i + CHUNK) / RATE
        bi.on_chunk(
            SimpleNamespace(samples=audio[i : i + CHUNK], rate_hz=RATE, stamp=now[0], gap=False)
        )
    assert len(handed) == 1
    onset_s = 2 * FRAME / RATE
    assert handed[0][1] == pytest.approx(onset_s - 0.064)  # pre_roll_ms before the /s/


# ------------------------------------------------------------------ ducking (candidate stage)

FRAME_MS = FRAME / RATE * 1000


def duck_snap(bi):
    return bi.snapshot()["duck"]


def test_music_like_non_speech_does_not_duck():
    probs = [0.2, 0.3, 0.25, 0.35] * 6  # under onset_threshold and threshold: never a candidate
    stops, bi = onset_run(probs, user_voice(len(probs) * FRAME / RATE, 0.2), keyword_enabled=False)
    assert stops == []
    assert duck_snap(bi)["started"] == 0 and not bi.duck.ducking(0.5)


def test_a_cough_ducks_then_releases_without_a_cut():
    probs = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # one confident frame: a cough, a click
    stops, bi = onset_run(probs, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    assert stops == []
    snap = duck_snap(bi)
    assert (snap["started"], snap["released"], snap["cut"], snap["capped"]) == (1, 1, 0, 0)
    # from the cough frame to the end of release_ms (64 ms: two frames) of non-speech
    assert snap["last_ms"] == pytest.approx(3 * FRAME_MS) and snap["max_ms"] == snap["last_ms"]
    cough = 2 * FRAME / RATE
    assert bi.duck.ducking(cough + 0.01)
    assert not bi.duck.ducking(cough + 3 * FRAME / RATE)  # released: later audio at full gain


def test_candidate_frames_inside_the_release_window_keep_one_episode():
    probs = [0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    stops, bi = onset_run(probs, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    snap = duck_snap(bi)
    assert stops == [] and (snap["started"], snap["released"]) == (1, 1)
    assert snap["last_ms"] == pytest.approx(5 * FRAME_MS)


def test_a_fricative_onset_ducks_and_its_cut_counts_as_a_cut_duck():
    stops, bi = onset_run([0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False)
    snap = duck_snap(bi)
    assert len(stops) == 1
    assert (snap["started"], snap["released"], snap["cut"]) == (1, 0, 1)
    assert snap["last_ms"] == pytest.approx(2 * FRAME_MS)  # the two onset frames
    assert not bi.duck.ducking(0.01)  # the cut clears the duck: the next answer is at full gain


def test_ducking_can_be_switched_off_and_the_cut_still_happens():
    stops, bi = onset_run(
        [0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False, duck={"enabled": False}
    )
    assert len(stops) == 1 and duck_snap(bi)["started"] == 0


def test_the_robots_own_echo_does_not_duck():
    w = World()
    w.speak(3.0)
    w.feed(3.0)
    assert w.bargein.snapshot()["echo_frames"] > 20 and duck_snap(w.bargein)["started"] == 0


def test_real_speech_ducks_then_cuts_at_the_same_latency_as_without_ducking():
    def run(enabled):
        cfg = RoboEmotConfig.model_validate({"voice": {"barge_in": {"duck": {"enabled": enabled}}}})
        w = World(["ciao", "ciao robot come stai"], cfg=cfg)
        w.speak(5.0)
        w.feed(1.0)
        w.feed(2.0, user_voice(1.5, 0.21), user_at=0.5)
        assert len(w.stops) == 1
        return w.bargein.snapshot()

    on, off = run(True), run(False)
    assert on["latency"]["onset_to_stop"] == off["latency"]["onset_to_stop"]  # SPEC 13
    assert on["latency"]["detect_to_stop"] == off["latency"]["detect_to_stop"]
    assert on["latency"]["onset_to_stop"]["last_ms"] < 200
    assert (on["duck"]["started"], on["duck"]["cut"]) == (1, 1) and off["duck"]["started"] == 0


def test_the_end_of_the_robots_speech_releases_the_duck_and_clears_the_gain():
    probs = [0.0, 0.9, 0.0]
    stops, bi = onset_run(probs, user_voice(3 * FRAME / RATE, 0.1), keyword_enabled=False)
    assert duck_snap(bi)["started"] == 1 and duck_snap(bi)["released"] == 0
    bi.reset()  # the listener: the robot stopped speaking
    snap = duck_snap(bi)
    assert (snap["released"], snap["cut"]) == (1, 0)
    assert not bi.duck.ducking(FRAME / RATE + 0.01)
    bi.reset()  # nothing more to count
    assert duck_snap(bi)["released"] == 1


def test_the_snapshot_reports_the_audio_out_of_reach_when_the_duck_started():
    now = [0.0]
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.9, 0.0, 0.0, 0.0]),
        SimpleNamespace(start=lambda *a: None, accept=lambda a: None, reset=lambda: None),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: None, lambda: None, None)
    bi.duck.apply(np.zeros(RATE * 3, np.float32), RATE, 0.0)  # the speech output pushed 3 s
    audio = user_voice(4 * FRAME / RATE, 0.1)
    for i in range(0, audio.size, CHUNK):
        now[0] = (i + CHUNK) / RATE
        bi.on_chunk(
            SimpleNamespace(samples=audio[i : i + CHUNK], rate_hz=RATE, stamp=now[0], gap=False)
        )
    snap = duck_snap(bi)
    # the duck starts on the first frame, decided when the first chunk arrives (now = 64 ms):
    # what was already handed to the player beyond that moment is out of the duck's reach
    assert snap["handed_at_start_last_ms"] == pytest.approx(3000 - 64)
    assert snap["handed_at_start_max_ms"] == snap["handed_at_start_last_ms"]
    assert all(isinstance(v, (int, float)) or v is None for v in snap.values())


def test_the_duck_stays_open_while_candidates_continue_and_closes_at_the_release():
    """SPEC 8 stage 8: a long push lands while the user is still talking (candidates under the
    cut-off, e.g. the main path's run broken every other frame): it stays ducked throughout."""
    probs = [0.9, 0.0] * 10 + [0.0] * 4  # 20 frames of on-and-off speech, then quiet
    stops, bi = onset_run(probs, user_voice(len(probs) * FRAME / RATE, 0.1), keyword_enabled=False)
    assert stops == []
    snap = duck_snap(bi)
    assert (snap["started"], snap["released"], snap["cut"]) == (1, 1, 0)
    frame_s = FRAME / RATE
    # the release is decided after two quiet frames following the last candidate (frame 18)
    assert snap["last_ms"] == pytest.approx(21 * FRAME_MS)
    assert all(bi.duck.ducking(i * frame_s) for i in range(21))
    assert not bi.duck.ducking(21 * frame_s)


def test_while_candidates_continue_a_one_second_push_is_ducked_for_its_whole_length():
    probs = [0.9, 0.0] * 4
    _, bi = onset_run(probs, user_voice(len(probs) * FRAME / RATE, 0.1), keyword_enabled=False)
    # the episode is open (frame 7 is the first quiet frame after the last candidate)
    assert duck_snap(bi)["released"] == 0
    out = bi.duck.apply(np.ones(RATE, np.float32), RATE, 0.3)  # plays from 0.3 s for 1 s
    assert out[RATE // 100 :] == pytest.approx(0.25)


def test_a_gap_mid_episode_ramps_the_voice_back_but_keeps_one_episode():
    """A gap in the microphone audio loses track of the user: the voice ramps back up, and the
    candidates after it continue the same episode (counted once)."""
    now = [0.0]
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.9, 0.0, 0.9, 0.0, 0.0, 0.0]),
        SimpleNamespace(start=lambda *a: None, accept=lambda a: None, reset=lambda: None),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: None, lambda: None, None)
    audio = user_voice(6 * FRAME / RATE, 0.1)
    frame_s = FRAME / RATE
    for i in range(6):  # the chunk of frame 2 comes after a gap
        now[0] = (i + 1) * frame_s
        bi.on_chunk(
            SimpleNamespace(
                samples=audio[i * FRAME : (i + 1) * FRAME], rate_hz=RATE, stamp=now[0], gap=i == 2
            )
        )
    snap = duck_snap(bi)
    assert (snap["started"], snap["released"], snap["cut"]) == (1, 1, 0)
    assert snap["last_ms"] == pytest.approx(5 * FRAME_MS)


def test_a_gap_releases_the_gain_with_the_ramp_not_a_step():
    now = [0.0]
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.9, 0.0]),
        SimpleNamespace(start=lambda *a: None, accept=lambda a: None, reset=lambda: None),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: None, lambda: None, None)
    frame_s = FRAME / RATE
    bi.duck.apply(np.ones(FRAME, np.float32), RATE, 0.0)  # playing up to the first frame's end
    now[0] = frame_s
    bi.on_chunk(
        SimpleNamespace(samples=user_voice(frame_s, 0.1), rate_hz=RATE, stamp=now[0], gap=False)
    )
    assert duck_snap(bi)["started"] == 1
    played = bi.duck.apply(np.ones(FRAME, np.float32), RATE, frame_s)
    assert played[-1] == pytest.approx(0.25)
    now[0] = 2 * frame_s
    bi.on_chunk(
        SimpleNamespace(samples=user_voice(frame_s, 0.1), rate_hz=RATE, stamp=now[0], gap=True)
    )
    after = bi.duck.apply(np.ones(RATE // 10, np.float32), RATE, 2 * frame_s)
    ramp = round(0.05 * RATE)  # ramp_ms 50: from 0.25 back to 1 without a step
    assert after[0] == pytest.approx(0.25, abs=0.01)
    assert np.all(np.diff(after) >= -1e-6) and np.max(np.diff(after)) < 0.01
    assert after[ramp:] == pytest.approx(1.0)
    assert duck_snap(bi)["released"] == 0  # the episode itself goes on


# ------------------------------------------------------------------ trigger records (SPEC 8, 12A)

SCALARS = (bool, int, float, type(None))
BY_GROUP = ("commanded_dps", "measured_dps")  # the two mappings: manifest group label -> deg/s


class FakeMotion:
    """A ``MotionProbe`` with a fixed reading; counts the derivations."""

    def __init__(self, commanded=None, measured=None, external=False, reflex=False, stamp=0.0):
        self.sample = MotionSample(stamp, {}, None, external, reflex)
        self.state = MotionState(stamp, commanded or {}, measured or {}, external, reflex)
        self.derived = 0

    def motion_sample(self):
        return self.sample

    def motion_state(self, sample):
        assert sample is self.sample
        self.derived += 1
        return self.state


def assert_scalars_only(record):
    for key, value in record.items():
        if key in BY_GROUP and value is not None:
            assert all(len(g) <= LABEL_MAX and isinstance(v, float) for g, v in value.items())
        elif isinstance(value, str):
            assert len(value) <= LABEL_MAX, key
        else:
            assert isinstance(value, SCALARS), key


def test_an_onset_cut_records_its_path_outcome_level_and_scores():
    stops, bi = onset_run([0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False)
    assert len(stops) == 1
    snap = bi.snapshot()
    (rec,) = snap["recent"]
    assert (rec["outcome"], rec["fired"], rec["confirmed"]) == ("cut", "onset", "onset")
    assert rec["released_by"] is None
    assert rec["level_db"] == pytest.approx(-34.0, abs=0.5)  # rms 0.02
    assert rec["hf_ratio"] > 0.9  # the /s/ is all above the cutoff: the onset test's own value
    assert (rec["vad_prob"], rec["vad_peak"]) == (0.5, 0.6)
    assert rec["speech_ms"] == pytest.approx(2 * FRAME_MS)
    assert rec["episode_ms"] == pytest.approx(2 * FRAME_MS)
    assert rec["echo_corr"] == 0.0  # no reference: no echo evidence
    assert rec["robot_playing"] is None and rec["robot_level_db"] is None
    assert rec["moving"] is None and rec["face_offset_deg"] is None  # no probe, no lookup
    assert rec["speaking_ms"] == 0.0 and rec["lag_ms"] >= 0.0
    assert snap["cuts_by_path"] == {"vad": 0, "onset": 1, "keyword": 0}
    assert snap["record_errors"] == 0
    assert_scalars_only(rec)


def test_the_high_band_share_is_empty_when_the_onset_test_did_not_run():
    probs = [0.0, 0.9, 0.95, 0.9, 0.9, 0.9]
    _, bi = onset_run(
        probs, user_voice(6 * FRAME / RATE, 0.2), keyword_enabled=False, onset_enabled=False
    )
    assert bi.recent()[0]["hf_ratio"] is None  # no second FFT for the record


def test_a_main_path_cut_records_vad_as_the_path_that_fired_and_confirmed():
    probs = [0.0, 0.9, 0.95, 0.9, 0.9, 0.9]
    stops, bi = onset_run(  # the onset path off: it would confirm a confident run first
        probs, user_voice(6 * FRAME / RATE, 0.2), keyword_enabled=False, onset_enabled=False
    )
    assert len(stops) == 1
    (rec,) = bi.snapshot()["recent"]
    assert (rec["outcome"], rec["fired"], rec["confirmed"]) == ("cut", "vad", "vad")
    assert rec["vad_prob"] == 0.9 and rec["vad_peak"] == 0.95
    assert rec["speaking_ms"] == pytest.approx(FRAME_MS)  # one frame of speaking before it
    assert bi.snapshot()["cuts_by_path"] == {"vad": 1, "onset": 0, "keyword": 0}


def test_a_released_duck_records_released_after_the_quiet_frames():
    probs = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # a cough
    stops, bi = onset_run(probs, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    assert stops == []
    snap = bi.snapshot()
    (rec,) = snap["recent"]
    assert (rec["outcome"], rec["released_by"], rec["confirmed"]) == ("released", "quiet", None)
    assert rec["candidate_frames"] == 1 and rec["speech_ms"] == pytest.approx(FRAME_MS)
    assert rec["episode_ms"] == pytest.approx(3 * FRAME_MS)  # matches the duck episode
    assert snap["duck"]["last_ms"] == pytest.approx(rec["episode_ms"])
    assert sum(snap["cuts_by_path"].values()) == 0


def test_records_are_kept_with_ducking_off():
    stops, bi = onset_run(
        [0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False, duck={"enabled": False}
    )
    assert len(stops) == 1 and bi.snapshot()["duck"]["started"] == 0
    (rec,) = bi.recent()
    assert (rec["outcome"], rec["confirmed"]) == ("cut", "onset")
    _, bi = onset_run(
        [0.0, 0.9, 0.0, 0.0, 0.0],
        user_voice(5 * FRAME / RATE, 0.1),
        duck={"enabled": False},
        keyword_enabled=False,
    )
    assert [(r["outcome"], r["released_by"]) for r in bi.recent()] == [("released", "quiet")]


def test_an_episode_open_at_the_end_of_speech_is_released_by_it():
    _, bi = onset_run([0.9], user_voice(FRAME / RATE, 0.1), keyword_enabled=False)
    assert bi.snapshot()["recent"][0]["outcome"] == "open"  # shown mid-episode
    bi.reset()  # the robot stopped speaking
    (rec,) = bi.snapshot()["recent"]
    assert (rec["outcome"], rec["released_by"]) == ("released", "speech_end")


def test_a_gap_keeps_the_record_open():
    _, bi = onset_run([0.9], user_voice(FRAME / RATE, 0.1), keyword_enabled=False)
    bi.reset(gap=True)
    assert [r["outcome"] for r in bi.recent()] == ["open"]


def test_the_ring_keeps_the_last_n_records():
    probs = [p for k in range(5) for p in (0.81 + k / 100, 0.0, 0.0, 0.0)]  # five coughs
    _, bi = onset_run(
        probs, user_voice(len(probs) * FRAME / RATE, 0.1), keyword_enabled=False, recent_triggers=3
    )
    recent = bi.snapshot()["recent"]
    assert [r["vad_prob"] for r in recent] == [0.83, 0.84, 0.85]
    assert recent[0]["ago_s"] > recent[-1]["ago_s"]
    assert bi.snapshot()["duck"]["started"] == 5


def test_a_hot_reload_resizes_the_ring():
    probs = [p for _ in range(4) for p in (0.9, 0.0, 0.0, 0.0)]
    _, bi = onset_run(probs, user_voice(len(probs) * FRAME / RATE, 0.1), keyword_enabled=False)
    assert len(bi.recent()) == 4
    bi.apply_config(RoboEmotConfig.model_validate({"voice": {"barge_in": {"recent_triggers": 2}}}))
    assert len(bi.recent()) == 2


@pytest.mark.parametrize(
    ("motion", "moving"),
    [
        (FakeMotion({"head": 40.0}, {"head": 30.0}, reflex=True, stamp=0.1), True),
        (FakeMotion({"head": 40.0}, {"head": 1.0}), False),  # measured wins over commanded
        (FakeMotion({"head": 40.0}), True),  # no measured pose: commanded
        (FakeMotion({"ears": 300.0, "head": 1.0}), False),  # another group's speed is not it
        (FakeMotion({}, external=True), True),
    ],
)
def test_motion_at_the_first_candidate_frame_is_recorded_from_the_probe(motion, moving):
    _, bi = onset_run([0.5, 0.6, 0.55], hiss(3), keyword_enabled=False, motion=motion)
    (rec,) = bi.recent()
    state = motion.state
    assert rec["moving"] is moving
    assert rec["commanded_dps"] == state.commanded_dps
    assert rec["measured_dps"] == state.measured_dps
    assert (rec["external"], rec["reflex"]) == (state.external, state.reflex)
    # the sample's tick against the end of the first candidate frame
    assert rec["motion_offset_ms"] == pytest.approx((state.stamp - FRAME / RATE) * 1000)
    assert motion.derived == 1  # derived once, at close, not on the frame path
    assert_scalars_only(rec)


def test_the_moving_group_and_threshold_are_configurable():
    motion = FakeMotion({"ears": 4.0, "head": 0.0})
    _, bi = onset_run(
        [0.5, 0.6, 0.55],
        hiss(3),
        keyword_enabled=False,
        motion=motion,
        moving_min_speed_dps=3.0,
        moving_group="ears",
    )
    assert bi.recent()[0]["moving"] is True
    _, bi = onset_run(
        [0.5, 0.6, 0.55], hiss(3), keyword_enabled=False, motion=motion, moving_group=None
    )
    assert bi.recent()[0]["moving"] is False  # the fastest group, 4 deg/s, is under 5


def test_the_sound_direction_offset_from_the_nearest_face_is_recorded():
    seen = []

    def lookup(now):
        seen.append(now)
        return np.pi / 6

    _, bi = onset_run([0.5, 0.6, 0.55], hiss(3), keyword_enabled=False, face_offset=lookup)
    assert bi.recent()[0]["face_offset_deg"] == pytest.approx(30.0)
    assert seen == [pytest.approx(0.0)]  # once, at close, for the episode's first frame


def test_a_failing_probe_or_lookup_leaves_the_fields_empty_and_still_cuts():
    class Broken:
        def motion_sample(self):
            raise RuntimeError("boom")

    def no_face(now):
        raise RuntimeError("boom")

    stops, bi = onset_run(
        [0.5, 0.6, 0.55], hiss(3), keyword_enabled=False, motion=Broken(), face_offset=no_face
    )
    assert len(stops) == 1
    snap = bi.snapshot()
    assert snap["record_errors"] == 2 and snap["triggers"] == 1


def test_a_raising_reference_window_still_cuts_and_interrupts(caplog):
    """The recording's reference read fails: the cut, the interrupt and the reset still happen,
    the error is counted and logged once per exception type."""

    class Ref:
        rate_hz = RATE

        def set_keep_s(self, keep_s):
            pass

        def window(self, t0, t1):
            raise ValueError("boom")

    interrupts = []
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD([0.5, 0.6, 0.55]),
        SimpleNamespace(),
        Ref(),
        language=lambda: "en",
        clock=lambda: 1.0,
    )
    bi._echo = bi._echo_hf = lambda end: 0.0  # the decision's own reads stay working
    stops = []
    bi.bind(lambda: stops.append(1), lambda: interrupts.append(1), None)
    audio = hiss(3)
    bi.on_chunk(SimpleNamespace(samples=audio, rate_hz=RATE, stamp=audio.size / RATE, gap=False))
    assert stops == [1] and interrupts == [1]
    assert bi.snapshot()["record_errors"] == 1 and bi._rec is None and bi._run == 0
    assert sum("trigger record: ValueError" in r.message for r in caplog.records) == 1


def test_a_cut_over_the_robots_voice_records_the_echo_and_its_own_audio_playing():
    w = World(["ciao", "ciao robot come stai"])
    w.speak(5.0)
    w.feed(1.0)
    w.feed(2.0, user_voice(1.5, 0.21), user_at=0.5)
    assert len(w.stops) == 1
    rec = next(r for r in w.bargein.recent() if r["outcome"] == "cut")
    assert rec["robot_playing"] is True and rec["robot_level_db"] > -30
    assert 0.0 <= rec["echo_corr"] < w.cfg.voice.barge_in.echo_corr_max
    assert rec["speaking_ms"] > 1000  # the robot had spoken for about 1.5 s
    for r in w.bargein.recent():
        assert_scalars_only(r)


def test_a_keyword_stop_with_no_episode_open_gets_a_record_of_its_own():
    w = quiet_stop_world(["stop"])
    w.feed(2.0, user_voice(1.0, 0.065), user_at=0.3)
    assert len(w.stops) == 1
    snap = w.bargein.snapshot()
    assert snap["cuts_by_path"] == {"vad": 0, "onset": 0, "keyword": 1}
    rec = snap["recent"][-1]
    assert (rec["outcome"], rec["fired"], rec["confirmed"]) == ("cut", "keyword", "keyword")
    assert_scalars_only(rec)


@pytest.mark.parametrize("prob", [0.9, 0.5])
def test_a_keyword_stop_during_an_open_episode_confirms_it(prob):
    audio = hiss(1) if prob == 0.5 else user_voice(FRAME / RATE, 0.1)
    stops, bi = onset_run([prob], audio, keyword_enabled=False, onset_min_speech_ms=64)
    assert stops == [] and bi.recent()[0]["outcome"] == "open"
    bi._worker.take_hit = lambda: 0.05  # the keyword worker heard "stop"
    assert bi._poll_keyword()
    (rec,) = bi.recent()
    fired = "vad" if prob == 0.9 else "onset"
    assert (rec["outcome"], rec["fired"], rec["confirmed"]) == ("cut", fired, "keyword")
    assert bi.snapshot()["cuts_by_path"] == {"vad": 0, "onset": 0, "keyword": 1}


def test_recent_can_be_read_while_a_chunk_is_processed():
    """The report thread reads the ring while the listener writes it; the stop action (inside
    ``on_chunk``) reads it too without a deadlock."""
    seen = []
    done = threading.Event()

    def reader():
        while not done.is_set():
            for r in bi.recent():
                assert_scalars_only(r)

    probs = [p for _ in range(40) for p in (0.9, 0.0, 0.0, 0.0)] + [0.5, 0.6, 0.55]
    audio = np.concatenate([user_voice(160 * FRAME / RATE, 0.1), hiss(3)])
    # The clock follows the chunk stamps: with the real monotonic clock the echo-tail deafness
    # after the cut depended on the machine's uptime against T0, and on a fresh CI runner the
    # trailing hiss frame opened a new (rightly open) episode after the cut record.
    now = [T0]
    bi = BargeIn(
        RoboEmotConfig.model_validate({"voice": {"barge_in": {"keyword_enabled": False}}}),
        ScriptedVAD(probs),
        SimpleNamespace(),
        None,
        language=lambda: "en",
        clock=lambda: now[0],
    )
    bi.bind(lambda: seen.append(bi.recent()), lambda: None, None)
    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for i in range(0, audio.size, CHUNK):
            part = audio[i : i + CHUNK]
            now[0] = T0 + (i + part.size) / RATE
            bi.on_chunk(SimpleNamespace(samples=part, rate_hz=RATE, stamp=now[0], gap=False))
    finally:
        done.set()
        thread.join()
    assert len(seen) == 1 and seen[0][-1]["outcome"] == "open"  # the cut is recorded after stop
    assert bi.recent()[-1]["outcome"] == "cut" and bi.snapshot()["record_errors"] == 0


def test_recording_does_not_change_any_decision():
    """The same audio with a probe and a lookup cuts at the same moment as without."""
    a, _ = onset_run([0.5, 0.6, 0.55, 0.3], hiss(4), keyword_enabled=False)
    b, _ = onset_run(
        [0.5, 0.6, 0.55, 0.3],
        hiss(4),
        keyword_enabled=False,
        motion=FakeMotion({"head": 90.0}, external=True),
        face_offset=lambda now: 1.0,
    )
    assert a == b and len(a) == 1


# ------------------------------------------------------------------ a sound that is not speech


def test_speech_over_the_robots_own_sound_cuts_it_without_cancelling_an_answer():
    """SPEC 7B, 8 (owner decision 2026-11-04): a person speaking over a self-soothing hum cuts
    the hum (``cut_own``), not the robot's speech: no prepare, stop or interrupt, and the
    person's words are kept."""
    w = World(["ciao", "neve mi racconti una barzelletta"])
    order, cuts, decisions = [], [], []
    w.bargein.events = lambda kind, reason, **fields: decisions.append((kind, reason))

    def cut_own():
        cuts.append(w.rig.now)
        w.stop()  # the hum is cut: what ReachySpeaker.stop does to the speaking window
        return True

    w.bargein.bind(
        lambda: order.append("stop"),
        lambda: order.append("interrupt"),
        w.rig.listener.resume,
        lambda: order.append("prepare"),
        cut_own=cut_own,
    )
    w.speak(5.0)  # the hum
    w.feed(1.0)
    onset = w.rig.t + 0.5
    w.feed(2.0, user_voice(1.5, 0.21), user_at=0.5)
    assert len(cuts) == 1 and order == []
    snap = w.bargein.snapshot()
    assert snap["own_sound_cuts"] == 1 and snap["triggers"] == 0
    assert [r["outcome"] for r in snap["recent"]] == ["cut_own"]  # not the user cutting it off
    assert decisions == [("bargein", "cut_own")]
    w.feed(2.0)
    (u,) = w.rig.utterances
    assert u.text == "neve mi racconti una barzelletta"
    assert u.start_stamp <= onset + 0.1  # the first words were not lost to the hum


def test_over_the_robots_speech_cut_own_declines_and_the_cut_off_is_as_before():
    w = World(["ciao"])
    order = []
    w.bargein.bind(
        lambda: (order.append("stop"), w.stop()),
        lambda: order.append("interrupt"),
        w.rig.listener.resume,
        lambda: order.append("prepare"),
        cut_own=lambda: False,  # a sentence is playing
    )
    w.speak(4.0)
    w.feed(0.5)
    w.feed(1.5, user_voice(1.0, 0.3), user_at=0.2)
    assert order == ["prepare", "stop", "interrupt"]
    assert w.bargein.snapshot()["own_sound_cuts"] == 0
