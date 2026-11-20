"""WakeGate decision counters for the debug report (SPEC 8 stage 3, 12A): numbers only."""

from __future__ import annotations

import json

import pytest
from roboemot.voice.wake import AT_FACE, NO_FACE_AT_VOICE, WakeGate

KEYS = {
    "not_required",
    "wake_word",
    "window",
    "rejected_no_window",
    "rejected_other_speaker",
    "rejected_no_face_at_voice",
    "rejected_not_facing",
    "rejected_no_facing_data",
    "rejected_music",
    "rejected_media",
    "face_media",
}
WINDOW_S = 5.0


def decide(
    gate, *, wake_word=False, enrolled=False, start=0.0, last=1.0, speaker=None, at_face=None
):
    return gate.decide(
        wake_word=wake_word,
        enrolled_in_view=enrolled,
        start=start,
        last_speech=last,
        speaker=speaker,
        at_face=at_face,
    )


def counts(gate):
    return gate.snapshot(0.0)["decisions"]


def test_all_keys_are_present_and_zero_before_use():
    assert counts(WakeGate("always", WINDOW_S)) == dict.fromkeys(KEYS, 0)


@pytest.mark.parametrize(
    "policy,enrolled", [("never", False), ("never", True), ("unless_enrolled_in_view", True)]
)
def test_not_required(policy, enrolled):
    gate = WakeGate(policy, WINDOW_S)
    assert decide(gate, enrolled=enrolled) and decide(gate, enrolled=enrolled, wake_word=True)
    assert counts(gate) == {**dict.fromkeys(KEYS, 0), "not_required": 2}


@pytest.mark.parametrize("policy", ["always", "unless_enrolled_in_view"])
def test_wake_word_window_and_rejections_when_required(policy):
    gate = WakeGate(policy, WINDOW_S)
    assert not decide(gate, start=0.0, last=1.0)  # nothing open yet
    assert decide(gate, wake_word=True, start=2.0, last=3.0, speaker="ann")
    assert decide(gate, start=4.0, last=5.0, speaker="ann")  # follow-up inside the window
    assert decide(gate, start=6.0, last=7.0)  # speaker unknown: still the same window
    assert not decide(gate, start=8.0, last=9.0, speaker="bob")  # inside, another person
    assert not decide(gate, start=20.0, last=21.0, speaker="ann")  # after the window
    assert counts(gate) == {
        "not_required": 0,
        "wake_word": 1,
        "window": 2,
        "rejected_no_window": 2,
        "rejected_other_speaker": 1,
        "rejected_no_face_at_voice": 0,
        "rejected_not_facing": 0,
        "rejected_no_facing_data": 0,
        "rejected_music": 0,
        "rejected_media": 0,
        "face_media": 0,
    }


def test_snapshot_is_json_and_keeps_its_other_fields():
    gate = WakeGate("always", WINDOW_S)
    decide(gate, wake_word=True)
    snap = gate.snapshot(0.5)
    assert snap["policy"] == "always" and snap["window_open"] is True
    assert json.loads(json.dumps(snap)) == snap


class Check:
    """A scripted voice-at-face check that counts its calls."""

    def __init__(self, passed):
        self.result, self.calls = AT_FACE if passed else NO_FACE_AT_VOICE, 0

    def __call__(self):
        self.calls += 1
        return self.result


@pytest.mark.parametrize(
    "policy,enrolled", [("never", False), ("never", True), ("unless_enrolled_in_view", True)]
)
def test_not_required_without_a_face_at_the_voice_is_rejected(policy, enrolled):
    gate = WakeGate(policy, WINDOW_S)
    assert not decide(gate, enrolled=enrolled, at_face=Check(False))
    assert decide(gate, enrolled=enrolled, at_face=Check(True))
    assert counts(gate) == {
        **dict.fromkeys(KEYS, 0),
        "not_required": 1,
        "rejected_no_face_at_voice": 1,
    }


@pytest.mark.parametrize("policy", ["never", "always", "unless_enrolled_in_view"])
def test_the_wake_word_always_passes_and_never_runs_the_check(policy):
    gate = WakeGate(policy, WINDOW_S)
    check = Check(False)
    assert decide(gate, wake_word=True, at_face=check)
    assert check.calls == 0


def test_a_window_follow_up_needs_the_face_and_a_rejection_keeps_the_window():
    gate = WakeGate("always", WINDOW_S)
    assert decide(gate, wake_word=True, start=0.0, last=1.0, speaker="ann")
    assert not decide(gate, start=2.0, last=3.0, at_face=Check(False))  # music in the window
    assert decide(gate, start=5.5, last=6.0, at_face=Check(True))  # window still from t=1
    assert not decide(gate, start=12.0, last=13.0, at_face=Check(True))  # past it: no window
    assert counts(gate)["rejected_no_face_at_voice"] == 1
    assert counts(gate)["window"] == 1 and counts(gate)["rejected_no_window"] == 1


def test_other_speaker_and_no_window_are_decided_before_the_check():
    gate = WakeGate("always", WINDOW_S)
    check = Check(True)
    assert not decide(gate, at_face=check)  # no window yet
    decide(gate, wake_word=True, speaker="ann")
    assert not decide(gate, start=1.5, last=2.0, speaker="bob", at_face=check)
    assert check.calls == 0


def test_addressable_asks_without_counting_or_moving_the_window():
    """SR-3: the listener asks at speech start whether the utterance needs the wake word
    (SPEC 8, Realtime speech route); the answer changes nothing."""
    g = WakeGate("always", 3.0)
    assert not g.addressable(enrolled_in_view=True, start=1.0)
    assert g.decide(wake_word=True, enrolled_in_view=False, start=0.0, last_speech=1.0, speaker="a")
    before = g.snapshot(1.5)
    assert g.addressable(enrolled_in_view=False, start=2.0, speaker="a")
    assert not g.addressable(enrolled_in_view=False, start=2.0, speaker="b")
    assert not g.addressable(enrolled_in_view=False, start=5.0, speaker="a")
    assert g.snapshot(1.5) == before
    assert WakeGate("never", 0.0).addressable(enrolled_in_view=False, start=0.0)
    enrolled = WakeGate("unless_enrolled_in_view", 0.0)
    assert enrolled.addressable(enrolled_in_view=True, start=0.0)
    assert not enrolled.addressable(enrolled_in_view=False, start=0.0)


# ------------------------------------------------------------------ music (SPEC 8)


class Music:
    """A scripted music predicate over [start, end] that records what it was asked."""

    def __init__(self, playing=True):
        self.playing, self.asked = playing, []

    def __call__(self, start, end):
        self.asked.append((start, end))
        return self.playing


def test_music_requires_wake_word_with_policy_never():
    gate = WakeGate("never", WINDOW_S, music=Music())
    assert gate.outcome(wake_word=False, enrolled_in_view=False, start=0.0, last_speech=1.0) == (
        "rejected_music"
    )
    assert counts(gate) == {**dict.fromkeys(KEYS, 0), "rejected_music": 1}


def test_music_is_asked_over_the_whole_utterance():
    music = Music(playing=False)
    gate = WakeGate("never", WINDOW_S, music=music)
    decide(gate, start=2.0, last=4.5)
    assert music.asked == [(2.0, 4.5)]


def test_music_closes_follow_up_window():
    music = Music(playing=False)
    gate = WakeGate("always", WINDOW_S, music=music)
    assert decide(gate, wake_word=True, start=0.0, last=1.0)
    assert decide(gate, start=2.0, last=3.0)  # a follow-up while no music plays
    music.playing = True
    out = gate.outcome(wake_word=False, enrolled_in_view=False, start=4.0, last_speech=5.0)
    assert out == "rejected_music"


def test_wake_word_over_music_is_addressed():
    for policy in ("never", "always", "unless_enrolled_in_view"):
        gate = WakeGate(policy, WINDOW_S, music=Music())
        out = gate.outcome(wake_word=True, enrolled_in_view=True, start=0.0, last_speech=1.0)
        assert out == "wake_word"
        assert gate.snapshot(1.0 + WINDOW_S)["window_open"]  # the window opens as today
        assert gate.snapshot(1.0 + WINDOW_S + 0.1)["window_open"] is False


def test_music_wins_over_no_window_and_face():
    check = Check(False)
    gate = WakeGate("always", WINDOW_S, music=Music())
    out = gate.outcome(
        wake_word=False, enrolled_in_view=False, start=0.0, last_speech=1.0, at_face=check
    )
    assert out == "rejected_music" and check.calls == 0
    never = WakeGate("never", WINDOW_S, music=Music())
    assert not decide(never, at_face=check) and check.calls == 0
    assert counts(never)["rejected_music"] == 1


def test_addressable_false_over_music():
    music = Music(playing=False)
    gate = WakeGate("never", WINDOW_S, music=music)
    assert gate.addressable(enrolled_in_view=False, start=1.0, end=3.0)
    assert gate.addressable(enrolled_in_view=False, start=1.0)
    assert music.asked == [(1.0, 3.0), (1.0, 1.0)]  # end None: the start alone
    music.playing = True
    assert not gate.addressable(enrolled_in_view=False, start=1.0, end=3.0)
    assert not gate.addressable(enrolled_in_view=False, start=1.0)
    window = WakeGate("always", WINDOW_S, music=music)
    music.playing = False
    window.decide(wake_word=True, enrolled_in_view=False, start=0.0, last_speech=1.0)
    music.playing = True
    assert not window.addressable(enrolled_in_view=False, start=2.0, end=3.0)
    assert counts(window)["rejected_music"] == 0  # asking counts nothing


def test_no_music_callable_keeps_today():
    with_none = WakeGate("always", WINDOW_S, music=None)
    quiet = WakeGate("always", WINDOW_S, music=Music(playing=False))
    for gate in (with_none, quiet):
        assert not decide(gate, start=0.0, last=1.0)
        assert decide(gate, wake_word=True, start=2.0, last=3.0, speaker="ann")
        assert decide(gate, start=4.0, last=5.0, speaker="ann")
        assert not decide(gate, start=8.0, last=9.0, speaker="bob")
        assert counts(gate) == {
            **dict.fromkeys(KEYS, 0),
            "wake_word": 1,
            "window": 1,
            "rejected_no_window": 1,
            "rejected_other_speaker": 1,
        }


# ------------------------------------------- the face rule over the radio (SPEC 8, 2026-11-20)


def radio_gate(policy="never"):
    return WakeGate(policy, WINDOW_S, media=lambda at: True)


def face_outcome(
    gate, passed, *, face_media=True, wake_word=False, start=0.0, last=1.0, enrolled=False
):
    check = Check(passed)
    outcome = gate.outcome(
        wake_word=wake_word,
        enrolled_in_view=enrolled,
        start=start,
        last_speech=last,
        at_face=check,
        face_media=face_media,
    )
    return outcome, check.calls


@pytest.mark.parametrize(
    ("policy", "enrolled"), [("never", False), ("unless_enrolled_in_view", True)]
)
def test_over_the_radio_a_voice_from_a_face_is_a_face_media_wake(policy, enrolled):
    """Owner decision 2026-11-20: over the radio the voice-at-face check stands in for the wake
    word when a server transcript can confirm it and the policy would not require the wake word
    were the radio silent; it opens the window as a wake does."""
    gate = radio_gate(policy)
    assert face_outcome(gate, True, enrolled=enrolled) == ("face_media", 1)
    assert counts(gate)["face_media"] == 1
    assert gate.snapshot(1.0)["window_open"]
    assert not gate.policy_requires(enrolled)


@pytest.mark.parametrize(
    ("policy", "enrolled"),
    [("always", False), ("always", True), ("unless_enrolled_in_view", False)],
)
def test_a_policy_that_requires_the_wake_word_keeps_rejected_media(policy, enrolled):
    """Security review 2026-11-20: the face rule never sends audio the policy would not send.
    Where the policy requires the wake word, a face over the radio stays ``rejected_media`` and
    the face is not even asked."""
    gate = radio_gate(policy)
    assert face_outcome(gate, True, enrolled=enrolled) == ("rejected_media", 0)
    assert gate.policy_requires(enrolled)


def test_over_the_radio_a_voice_from_no_face_stays_rejected_media():
    gate = radio_gate()
    assert face_outcome(gate, False) == ("rejected_media", 1)
    assert counts(gate)["rejected_media"] == 1 and counts(gate)["face_media"] == 0
    assert not gate.snapshot(1.0)["window_open"]


def test_without_a_confirming_transcript_the_face_is_never_asked_over_the_radio():
    """``face_media`` off (no realtime audio route, the switch off): today's rule."""
    gate = radio_gate()
    assert face_outcome(gate, True, face_media=False) == ("rejected_media", 0)


def test_the_wake_word_over_the_radio_never_asks_the_face():
    gate = radio_gate()
    assert face_outcome(gate, True, wake_word=True) == ("wake_word", 0)


def test_without_the_radio_the_face_media_flag_changes_nothing():
    gate = WakeGate("never", WINDOW_S, media=lambda at: False)
    assert face_outcome(gate, True) == ("not_required", 1)
    assert face_outcome(gate, False) == ("rejected_no_face_at_voice", 1)
