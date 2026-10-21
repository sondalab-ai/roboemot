"""SPEC 11A, 12: the live transcript ring. In memory only, opt-in, bounded by count, age and text
length; nothing is kept while the flag is off; cleared on every trigger; never logged."""

from __future__ import annotations

import logging

import pytest
from roboemot.config.schema import LiveConfig
from roboemot.conversation.transcript import TranscriptRing


def ring(**kw):
    now = [1000.0]
    cfg = LiveConfig(transcript=kw.pop("transcript", True), **kw)
    return TranscriptRing(cfg, clock=lambda: now[0], wall=lambda: 5000.0 + now[0]), now


def test_off_keeps_nothing_at_all():
    r, _ = ring(transcript=False)
    assert r.add("user", "hello", "en") is None
    assert r.snapshot(0)["turns"] == [] and r.snapshot(0)["seq"] == 0
    assert r.size == 0


def test_a_turn_has_the_documented_shape_and_newest_is_last():
    r, now = ring()
    a = r.add("user", "ciao", "it")
    now[0] += 1
    b = r.add("robot", "Ciao a te.", "it")
    turns = r.snapshot(0)["turns"]
    assert [t["seq"] for t in turns] == [a, b] and a < b
    assert turns[0] == {
        "seq": a, "t": 6000.0, "who": "user", "text": "ciao", "language": "it", "interrupted": False
    }  # fmt: skip
    assert turns[1]["who"] == "robot"


def test_the_count_bound_drops_the_oldest():
    r, _ = ring(transcript_max_turns=3)
    for i in range(5):
        r.add("user", f"t{i}", "en")
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["t2", "t3", "t4"]


def test_the_age_bound_drops_old_turns_on_read_and_on_write():
    r, now = ring(transcript_max_age_s=60.0)
    r.add("user", "old", "en")
    now[0] += 30
    r.add("user", "mid", "en")
    now[0] += 40  # old is 70 s, mid is 40 s
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["mid"]
    now[0] += 100
    r.add("user", "new", "en")
    assert r.size == 1


def test_the_text_is_cut_to_the_character_bound():
    r, _ = ring(transcript_max_chars=80)
    r.add("robot", "x" * 500, "en")
    assert len(r.snapshot(0)["turns"][0]["text"]) == 80


def test_the_cursor_returns_only_newer_turns():
    r, _ = ring()
    a = r.add("user", "one", "en")
    b = r.add("robot", "two", "en")
    snap = r.snapshot(a)
    assert [t["seq"] for t in snap["turns"]] == [b] and snap["seq"] == b
    assert r.snapshot(b)["turns"] == []


def test_a_cursor_ahead_of_the_ring_gets_everything():
    r, _ = ring()
    r.add("user", "one", "en")
    snap = r.snapshot(9999)  # ahead of the ring: the page's cursor is from another life
    assert [t["text"] for t in snap["turns"]] == ["one"]


@pytest.mark.parametrize("trigger", ["clear", "flag_off"])
def test_every_trigger_clears_and_changes_the_epoch(trigger):
    r, _ = ring()
    r.add("user", "secret words", "en")
    before = r.snapshot(0)["epoch"]
    if trigger == "clear":
        r.clear()
    else:
        r.apply_config(LiveConfig(transcript=False))
    snap = r.snapshot(0)
    assert snap["turns"] == [] and snap["epoch"] != before and r.size == 0


def test_clearing_does_not_reuse_sequence_numbers():
    r, _ = ring()
    a = r.add("user", "one", "en")
    r.clear()
    assert r.add("user", "two", "en") > a


def test_turning_the_flag_on_starts_empty_and_a_reload_keeps_what_is_kept():
    r, _ = ring(transcript=False)
    r.apply_config(LiveConfig(transcript=True))
    r.add("user", "kept", "en")
    r.apply_config(LiveConfig(transcript=True, transcript_max_turns=10))
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["kept"]


def test_a_smaller_bound_applies_at_once():
    r, _ = ring()
    for i in range(5):
        r.add("user", f"t{i}", "en")
    r.apply_config(LiveConfig(transcript=True, transcript_max_turns=2))
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["t3", "t4"]


def test_a_robot_turn_grows_then_is_marked_interrupted_with_the_heard_part():
    r, _ = ring()
    seq = r.add("robot", "Uno.", "it")
    r.extend(seq, "Due.")
    assert r.snapshot(0)["turns"][0]["text"] == "Uno. Due."
    r.settle(seq, "Uno.", interrupted=True)
    turn = r.snapshot(0)["turns"][0]
    assert turn["text"] == "Uno." and turn["interrupted"] is True


def test_settling_with_nothing_heard_removes_the_turn():
    r, _ = ring()
    seq = r.add("robot", "Never heard.", "it")
    r.settle(seq, "", interrupted=True)
    assert r.snapshot(0)["turns"] == []


def test_extending_or_settling_a_turn_that_is_gone_does_nothing():
    r, _ = ring(transcript_max_turns=1)
    old = r.add("robot", "a", "en")
    r.add("user", "b", "en")
    r.extend(old, "more")
    r.settle(old, "x", interrupted=True)
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["b"]


def test_nothing_is_logged(caplog):
    caplog.set_level(logging.DEBUG)
    r, _ = ring(transcript_max_turns=1)
    seq = r.add("robot", "tokenSECRET1", "en")
    r.extend(seq, "tokenSECRET2")
    r.add("user", "tokenSECRET3", "en")
    r.settle(seq, "tokenSECRET4", interrupted=True)
    r.snapshot(0)
    r.clear()
    assert "tokenSECRET" not in caplog.text
