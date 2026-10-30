"""Non-verbal cues (SPEC 8, Non-verbal cues; SPEC 4A): gesture plus sound, the listening hold."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.config.schema import CueConfig
from roboemot.motion.primitives import Keyframe, Primitive
from roboemot.voice.cues import (
    Cues,
    CueSound,
    CueSoundError,
    load_cue_sounds,
    peak_time,
    validate_cue_sounds,
)

RATE = 16000
PEAK = 0.3


def prim(name: str) -> Primitive:
    return Primitive(
        name,
        (
            Keyframe(0.0, {"antenna_left": 0.0}),
            Keyframe(PEAK, {"antenna_left": 0.4}),
            Keyframe(0.6, {"antenna_left": 0.0}),
        ),
        additive=True,
    )


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Output:
    def __init__(self) -> None:
        self.played: list[tuple[int, int, float]] = []
        self.cancelled: list[int] = []

    def play(self, samples, rate_hz, volume_db=0.0):
        self.played.append((samples.size, rate_hz, volume_db))
        return len(self.played)

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def sound(seconds: float = 0.2) -> CueSound:
    return CueSound(np.full(int(seconds * RATE), 0.1, np.float32), RATE)


def make(gestures=("cue_listening", "cue_fallback", "cue_not_heard"), sounds=None, **cfg):
    clock, out, speaking = Clock(), Output(), SimpleNamespace(on=False, muted=False)
    cues = Cues(
        CueConfig(**cfg),
        gesture=lambda name: prim(name) if name in gestures else None,
        sounds=sounds if sounds is not None else {"fallback": sound(), "listening": sound(0.1)},
        output=lambda: out,
        volume_db=lambda: -3.0,
        muted=lambda: speaking.muted,
        speaking=lambda now: speaking.on,
        clock=clock,
    )
    return cues, clock, out, speaking


def test_a_one_shot_cue_plays_its_gesture_and_its_sound_quietly():
    cues, clock, out, _ = make()
    assert cues.play("fallback")
    assert out.played == [(int(0.2 * RATE), RATE, -6.0 - 3.0)]  # cue level plus circadian
    cmds = cues.commands(clock.t + PEAK)
    assert [c.source for c in cmds] == ["cue:fallback"] and cmds[0].additive
    assert cmds[0].targets["antenna_left"] == pytest.approx(0.4)
    assert cues.commands(clock.t + 1.0) == []
    assert cues.snapshot()["counts"]["fallback"] == {"played": 1}


def test_a_cue_the_robot_cannot_show_is_a_counted_no_op():
    cues, _, out, _ = make(gestures=(), sounds={})
    assert not cues.play("offline")
    assert out.played == []
    snap = cues.snapshot()
    assert snap["counts"]["offline"] == {"unsupported": 1}
    assert snap["gestures"] == [] and snap["sounds"] == []


def test_cues_are_silent_while_muted_or_disabled():
    cues, _, out, state = make()
    state.muted = True
    assert not cues.play("fallback")
    off, _, out_off, _ = make(enabled=False)
    assert not off.play("fallback")
    assert out.played == out_off.played == []
    assert cues.snapshot()["counts"]["fallback"] == {"muted": 1}
    assert off.snapshot()["counts"]["fallback"] == {"disabled": 1}


def test_listening_shows_after_the_onset_holds_at_its_peak_and_ends_on_first_audio():
    cues, clock, out, state = make(listening_onset_s=0.3)
    cues.listening_begin()
    assert cues.commands(clock.t + 0.1) == []  # before the onset
    clock.t += 0.3
    assert cues.commands(clock.t)[0].targets["antenna_left"] == pytest.approx(0.0)
    held = cues.commands(clock.t + 5.0)  # still no answer: held at the peak
    assert held[0].source == "cue:listening"
    assert held[0].targets["antenna_left"] == pytest.approx(0.4)
    assert out.played == []  # listening_sound is off by default
    state.on = True  # the answer's first audio
    clock.t += 5.0
    cues.commands(clock.t)
    back = cues.commands(clock.t + 0.3)  # returns over the rest of the gesture
    assert back == [] or back[0].targets["antenna_left"] == pytest.approx(0.0, abs=1e-6)
    assert cues.snapshot()["counts"]["listening"] == {"shown": 1, "released_speech": 1}


def test_listening_ends_when_the_turn_ends_without_audio():
    cues, clock, _, _ = make(listening_onset_s=0.0)
    cues.listening_begin()
    cues.commands(clock.t + 1.0)
    clock.t += 1.0
    cues.listening_end()  # a timeout, a skip, an error: no audio came
    assert not cues.snapshot()["listening"]["holding"]
    assert cues.commands(clock.t + 1.0) == []
    assert cues.snapshot()["counts"]["listening"] == {"shown": 1, "released_ended": 1}


def test_an_answer_before_the_onset_shows_no_listening_cue():
    cues, clock, _, state = make(listening_onset_s=0.5)
    cues.listening_begin()
    state.on = True
    assert cues.commands(clock.t + 0.6) == []
    assert cues.snapshot()["counts"]["listening"] == {"skipped_speech": 1}


def test_listening_waits_for_every_utterance_that_began_it():
    cues, clock, _, _ = make(listening_onset_s=0.0)
    cues.listening_begin()
    cues.listening_begin()
    cues.commands(clock.t + 0.5)
    cues.listening_end()
    assert cues.snapshot()["listening"]["holding"]
    cues.listening_end()
    assert not cues.snapshot()["listening"]["holding"]


def test_the_hold_limit_always_releases_listening():
    cues, clock, _, _ = make(listening_onset_s=0.0, listening_hold_max_s=2.0)
    cues.listening_begin()
    cues.commands(clock.t)
    cues.commands(clock.t + 2.5)
    snap = cues.snapshot()
    assert not snap["listening"]["holding"] and snap["listening"]["outstanding"] == 0
    assert snap["counts"]["listening"]["released_hold_max"] == 1


def test_the_listening_sound_is_opt_in_and_cut_on_release():
    cues, clock, out, _ = make(listening_onset_s=0.0, listening_sound=True)
    cues.listening_begin()
    cues.commands(clock.t)
    assert len(out.played) == 1
    cues.listening_end()
    assert out.cancelled == [1]


def test_peak_time_is_the_keyframe_with_the_largest_offset():
    assert peak_time(prim("x")) == PEAK


def test_cue_sounds_are_validated():
    assert validate_cue_sounds({"fallback": (np.zeros(800, np.float32), RATE)})["fallback"]
    for bad, why in [
        ({"hello": sound()}, "unknown kind"),
        ({"fallback": sound(1.5)}, "at most"),
        ({"fallback": CueSound(np.full(100, 2.0, np.float32), RATE)}, "within"),
        ({"fallback": CueSound(np.zeros((2, 100), np.float32), RATE)}, "mono"),
        ({"fallback": CueSound(np.zeros(100, np.float32), 4000)}, "rate"),
    ]:
        with pytest.raises(CueSoundError, match=why):
            validate_cue_sounds(bad)


def test_a_robot_without_a_cue_library_gets_an_empty_one():
    manifest = SimpleNamespace(robot=SimpleNamespace(id="no-such-robot"))
    assert load_cue_sounds(manifest) == {}
