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
