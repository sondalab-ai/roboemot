"""WakeGate decision counters for the debug report (SPEC 8 stage 3, 12A): numbers only."""

from __future__ import annotations

import json

import pytest
from roboemot.voice.wake import WakeGate

KEYS = {"not_required", "wake_word", "window", "rejected_no_window", "rejected_other_speaker"}
WINDOW_S = 5.0


def decide(gate, *, wake_word=False, enrolled=False, start=0.0, last=1.0, speaker=None):
    return gate.decide(
        wake_word=wake_word,
        enrolled_in_view=enrolled,
        start=start,
        last_speech=last,
        speaker=speaker,
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
    }


def test_snapshot_is_json_and_keeps_its_other_fields():
    gate = WakeGate("always", WINDOW_S)
    decide(gate, wake_word=True)
    snap = gate.snapshot(0.5)
    assert snap["policy"] == "always" and snap["window_open"] is True
    assert json.loads(json.dumps(snap)) == snap
