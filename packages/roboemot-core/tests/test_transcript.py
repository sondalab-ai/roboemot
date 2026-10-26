"""SPEC 11A, 12: the live transcript ring. In memory only, opt-in, bounded by count, age and text
length; nothing is kept while the flag is off; cleared on every trigger; never logged."""

from __future__ import annotations

import logging

import pytest
from roboemot.config.schema import LiveConfig
from roboemot.conversation.transcript import TranscriptRing


def ring(**kw):
    now = [1000.0]
    cfg = LiveConfig(enabled=True, transcript=kw.pop("transcript", True), **kw)
    return TranscriptRing(cfg, clock=lambda: now[0], wall=lambda: 5000.0 + now[0]), now


def test_off_keeps_nothing_at_all():
    r, _ = ring(transcript=False)
    assert r.add("user", "hello", "en") is None
    assert r.snapshot(0)["turns"] == [] and r.snapshot(0)["rev"] == 0
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
    cursor = r.snapshot(0)["rev"]
    b = r.add("robot", "two", "en")
    snap = r.snapshot(cursor)
    assert [t["seq"] for t in snap["turns"]] == [b] and a < b
    assert r.snapshot(snap["rev"])["turns"] == []


def test_a_reply_that_grew_since_the_last_poll_comes_again_with_the_same_seq():
    r, _ = ring()
    seq = r.add("robot", "Uno.", "it")
    first = r.snapshot(0)
    r.extend(seq, "Due.")
    again = r.snapshot(first["rev"])
    assert [(t["seq"], t["text"]) for t in again["turns"]] == [(seq, "Uno. Due.")]
    assert r.snapshot(again["rev"])["turns"] == []


def test_a_cut_reply_reaches_a_page_that_already_has_its_first_sentence():
    r, _ = ring()
    seq = r.add("robot", "Uno.", "it")
    r.extend(seq, "Due.")
    seen = r.snapshot(0)["rev"]
    r.settle(seq, "Uno.", interrupted=True)
    turn = r.snapshot(seen)["turns"][0]
    assert (turn["seq"], turn["text"], turn["interrupted"]) == (seq, "Uno.", True)


def test_a_reply_with_nothing_heard_goes_empty_for_pages_that_had_it_and_is_never_listed_afresh():
    r, _ = ring()
    r.add("user", "q", "it")
    seq = r.add("robot", "Never heard.", "it")
    seen = r.snapshot(0)["rev"]
    r.settle(seq, "", interrupted=True)
    assert [(t["seq"], t["text"]) for t in r.snapshot(seen)["turns"]] == [(seq, "")]
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["q"]


def test_a_cursor_ahead_of_the_ring_gets_everything():
    r, _ = ring()
    r.add("user", "one", "en")
    snap = r.snapshot(9999)  # ahead of the ring: the page's cursor is from another life
    assert [t["text"] for t in snap["turns"]] == ["one"]


def test_a_cursor_from_another_epoch_gets_everything_even_when_it_looks_valid():
    r, _ = ring()
    r.add("user", "one", "en")
    r.add("user", "two", "en")
    epoch = r.snapshot(0)["epoch"]
    cursor = r.snapshot(0)["rev"]
    assert r.snapshot(cursor, epoch)["turns"] == []
    assert [t["text"] for t in r.snapshot(cursor, "deadbeef.0")["turns"]] == ["one", "two"]


@pytest.mark.parametrize("trigger", ["clear", "flag_off"])
def test_every_trigger_clears_and_changes_the_epoch(trigger):
    r, _ = ring()
    r.add("user", "secret words", "en")
    before = r.snapshot(0)["epoch"]
    if trigger == "clear":
        r.clear()
    else:
        r.apply_config(LiveConfig(enabled=True, transcript=False))
    snap = r.snapshot(0)
    assert snap["turns"] == [] and snap["epoch"] != before and r.size == 0


def test_clearing_does_not_reuse_sequence_numbers():
    r, _ = ring()
    a = r.add("user", "one", "en")
    r.clear()
    assert r.add("user", "two", "en") > a


def test_turning_the_flag_on_starts_empty_and_a_reload_keeps_what_is_kept():
    r, _ = ring(transcript=False)
    r.apply_config(LiveConfig(enabled=True, transcript=True))
    r.add("user", "kept", "en")
    r.apply_config(LiveConfig(enabled=True, transcript=True, transcript_max_turns=10))
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["kept"]


def test_a_smaller_bound_applies_at_once():
    r, _ = ring()
    for i in range(5):
        r.add("user", f"t{i}", "en")
    r.apply_config(LiveConfig(enabled=True, transcript=True, transcript_max_turns=2))
    assert [t["text"] for t in r.snapshot(0)["turns"]] == ["t3", "t4"]


def test_a_robot_turn_grows_then_is_marked_interrupted_with_the_heard_part():
    r, _ = ring()
    seq = r.add("robot", "Uno.", "it")
    r.extend(seq, "Due.")
    assert r.snapshot(0)["turns"][0]["text"] == "Uno. Due."
    r.settle(seq, "Uno.", interrupted=True)
    turn = r.snapshot(0)["turns"][0]
    assert turn["text"] == "Uno." and turn["interrupted"] is True


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


def test_the_live_tab_itself_must_be_on():
    r = TranscriptRing(LiveConfig(enabled=False, transcript=True))
    assert r.add("user", "x", "en") is None
    r.apply_config(LiveConfig(enabled=True, transcript=True))
    assert r.add("user", "x", "en") == 1
    r.apply_config(LiveConfig(enabled=False, transcript=True))
    assert r.size == 0


def test_a_reply_generation_from_before_a_clear_is_refused():
    r, _ = ring()
    seq, gen = r.begin("user", "q", "en")
    assert seq == 1 and r.add("robot", "a", "en", generation=gen) is not None
    r.clear()
    assert r.add("robot", "late", "en", generation=gen) is None
    assert r.size == 0
    seq2, gen2 = r.begin("user", "q2", "en")
    assert gen2 != gen and r.add("robot", "ok", "en", generation=gen2) is not None


def test_begin_off_returns_a_generation_nothing_can_match():
    r, _ = ring(transcript=False)
    seq, gen = r.begin("user", "q", "en")
    assert seq is None
    r.apply_config(LiveConfig(enabled=True, transcript=True))
    assert r.add("robot", "a", "en", generation=gen) is None


def test_asleep_records_nothing_and_clears_and_waking_resumes():
    r, _ = ring()
    r.add("user", "x", "en")
    r.set_asleep(True)
    assert (
        r.size == 0 and r.add("user", "y", "en") is None and r.begin("user", "y", "en")[0] is None
    )
    r.set_asleep(False)
    assert r.add("user", "z", "en") is not None


def test_flag_off_clears_once_and_a_repeated_reload_does_not_churn_the_epoch():
    r, _ = ring()
    r.add("user", "x", "en")
    r.apply_config(LiveConfig(enabled=True, transcript=False))
    epoch = r.snapshot(0)["epoch"]
    r.apply_config(LiveConfig(enabled=True, transcript=False))
    assert r.snapshot(0)["epoch"] == epoch


def test_the_age_bound_holds_without_a_read_when_trim_is_called():
    r, now = ring(transcript_max_age_s=60.0)
    r.add("user", "old", "en")
    now[0] += 61
    r.trim()
    assert r.size == 0


def test_a_turn_repr_never_shows_its_text():
    r, _ = ring()
    r.add("user", "tokenSECRET", "en")
    assert "tokenSECRET" not in repr(r._turns[0]) and "tokenSECRET" not in repr(r._turns)


def test_a_full_list_over_the_response_budget_sends_the_newest_turns():
    r, _ = ring(transcript_response_max_chars=1000)
    for i in range(10):
        r.add("user", f"{i}" * 300, "en")
    texts = [t["text"][0] for t in r.snapshot(0)["turns"]]
    assert texts == ["7", "8", "9"] or texts == ["8", "9"] or texts[-1] == "9" and len(texts) <= 4
    assert sum(len(t["text"]) for t in r.snapshot(0)["turns"]) <= 1300


def test_a_single_turn_is_always_sent():
    r, _ = ring(transcript_response_max_chars=1000, transcript_max_chars=2000)
    r.add("user", "x" * 1500, "en")
    assert len(r.snapshot(0)["turns"]) == 1


def test_attach_follows_the_stack_even_when_a_later_listener_raises():
    from roboemot.config.stack import ConfigStack

    stack = ConfigStack(app={"webui": {"live": {"enabled": True, "transcript": True}}})
    r = TranscriptRing(stack.effective().webui.live)
    r.attach(stack)

    def broken(change):
        raise RuntimeError("an earlier reload step failed")

    stack.subscribe(broken)
    r.add("user", "x", "en")
    stack.set("webui.live.transcript", False, source="test", admin=True)
    assert r.size == 0 and r.add("user", "y", "en") is None
