"""SPEC 8, 11A, 12: the conversation worker feeds the live transcript. The user's final addressed
utterance, then the robot's sentences as they are queued to play; a reply cut by a barge-in is
trimmed to what was heard and marked interrupted. Off: nothing kept. No text is logged or in a
report."""

from __future__ import annotations

import json
import logging

from roboemot.conversation import TranscriptRing
from test_conversation_worker import (
    CLEARED,
    CLOUD,
    GATE,
    rigs,  # noqa: F401  (fixture)
    utt,
    wait_for,
)

SECRET = "tokenSECRETxyz"


def with_ring(rigs, *, on=True, **kw):  # noqa: F811
    rig = rigs(webui={"live": {"enabled": True, "transcript": on}}, **kw)
    rig.worker.transcript = TranscriptRing(rig.stack.effective().webui.live)
    return rig


def turns(rig):
    return [
        (t["who"], t["text"], t["interrupted"]) for t in rig.worker.transcript.snapshot()["turns"]
    ]


def test_a_turn_is_kept_user_first_then_one_robot_turn_with_all_sentences(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre."]
    rig.turn("ciao")
    assert turns(rig) == [("user", "ciao", False), ("robot", "Uno. Due. Tre.", False)]
    languages = {t["language"] for t in rig.worker.transcript.snapshot()["turns"]}
    assert languages == {"it"}


def test_the_utterance_language_is_kept(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.turn("hello", language="en")
    assert rig.worker.transcript.snapshot()["turns"][0]["language"] == "en"


def test_unaddressed_and_partial_speech_is_not_kept(rigs):  # noqa: F811
    rig = with_ring(rigs)
    assert not rig.worker.submit(utt("not for you", addressed=False))
    assert not rig.worker.submit(utt("half a sen", final=False))
    assert turns(rig) == []


def test_off_keeps_nothing(rigs):  # noqa: F811
    rig = with_ring(rigs, on=False)
    rig.turn("ciao")
    assert rig.worker.transcript.size == 0 and rig.worker.transcript.snapshot()["rev"] == 0


def test_no_ring_is_fine(rigs):  # noqa: F811
    rig = rigs()
    assert rig.worker.transcript is None
    rig.turn("ciao")


def test_a_barge_in_during_the_stream_keeps_only_what_was_heard(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre", GATE, "."]
    n = rig.say("racconta")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    rig.speech.unplayed.add(2)  # the second sentence was cut before it played
    assert rig.worker.interrupt()
    assert rig.done(n)
    assert turns(rig) == [("user", "racconta", False), ("robot", "Uno.", True)]


def test_a_barge_in_before_anything_played_removes_the_robot_turn(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due", GATE, "."]
    n = rig.say("racconta")
    assert wait_for(lambda: rig.speech.texts() == ["Uno."])
    rig.speech.unplayed.add(1)
    assert rig.worker.interrupt()
    assert rig.done(n)
    assert turns(rig) == [("user", "racconta", False)]  # the unheard reply is gone


def test_a_barge_in_after_the_stream_ended_trims_the_reply_to_what_was_heard(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre."]
    rig.turn("ciao")
    rig.speech.unplayed.update({2, 3})  # still queued when the user spoke over it
    assert rig.worker.interrupt()
    assert turns(rig) == [("user", "ciao", False), ("robot", "Uno.", True)]
    assert rig.worker.interrupt() is False  # nothing more to cut; the turn stays as it is
    assert turns(rig)[1] == ("robot", "Uno.", True)


def test_a_reply_played_in_full_is_not_marked_when_a_later_barge_in_finds_nothing(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due."]
    rig.turn("ciao")
    rig.worker.interrupt()
    assert turns(rig)[1] == ("robot", "Uno. Due.", False)


def test_turning_the_flag_off_mid_conversation_clears_and_keeps_nothing_new(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.turn("ciao")
    assert rig.worker.transcript.size == 2
    rig.worker.transcript.apply_config(
        rig.stack.effective().webui.live.model_copy(update={"transcript": False})
    )
    rig.turn("ancora")
    assert rig.worker.transcript.size == 0


def test_no_transcript_text_reaches_the_logs_or_the_worker_report(rigs, caplog):  # noqa: F811
    caplog.set_level(logging.DEBUG)
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = [f"{SECRET}. ", "Due."]
    rig.turn(f"{SECRET} user")
    rig.worker.interrupt()
    assert SECRET in json.dumps(rig.worker.transcript.snapshot())  # it is in the ring ...
    assert SECRET not in caplog.text  # ... and nowhere else
    assert SECRET not in json.dumps(rig.worker.snapshot(), default=str)
    assert SECRET not in json.dumps(rig.events, default=str)


def test_a_page_that_already_saw_the_first_sentence_gets_the_cut_version(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre", GATE, "."]
    n = rig.say("racconta")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    cursor = rig.worker.transcript.snapshot(0)["rev"]  # the page polled mid-reply
    assert rig.worker.transcript.snapshot(0)["turns"][-1]["text"] == "Uno. Due."
    rig.speech.unplayed.add(2)
    assert rig.worker.interrupt()
    assert rig.done(n)
    sent = rig.worker.transcript.snapshot(cursor)["turns"]
    assert [(t["who"], t["text"], t["interrupted"]) for t in sent] == [("robot", "Uno.", True)]


def test_a_minor_cut_trims_the_transcript_to_what_played(rigs):  # noqa: F811
    rig = rigs(routes=(CLOUD,), webui={"live": {"enabled": True, "transcript": True}})
    rig.worker.transcript = TranscriptRing(rig.stack.effective().webui.live)
    rig.server.scripts["cloud.local"] = ["Prima. ", "Seconda. ", "Terza", GATE, "."]
    n = rig.say("ciao")
    assert wait_for(lambda: rig.speech.texts() == ["Prima.", "Seconda."])
    rig.speech.unplayed.add(2)  # the stop flushed the second sentence
    rig.minor[0] = True
    rig.server.gate.set()
    assert rig.done(n)
    assert turns(rig)[-1] == ("robot", "Prima.", True)


# ------------------------------------------------------------------ a clear is a fence (B1)


def clear_while(rig, how):
    """Run ``how`` (a clear trigger) while the first reply is still being thought."""
    n = rig.say("domanda privata")
    assert wait_for(lambda: rig.worker.transcript.size == 1)  # the user turn; the robot is thinking
    how(rig.worker.transcript)
    return n


def test_clear_while_thinking_keeps_the_reply_out(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = [GATE, "Risposta alla domanda cancellata."]
    n = clear_while(rig, lambda ring: ring.clear())
    rig.server.gate.set()
    assert rig.done(n)
    assert rig.worker.transcript.size == 0 and rig.speech.texts()  # it was still spoken


def test_clear_with_a_queued_utterance_keeps_both_replies_out(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = [GATE, "Prima risposta."]
    n1 = rig.say("prima domanda")
    assert wait_for(lambda: rig.worker.transcript.size == 1)
    n2 = rig.say("seconda domanda privata")  # waits behind the first
    assert wait_for(lambda: rig.worker.transcript.size == 2)
    rig.worker.transcript.clear()
    rig.server.scripts["lan.local"] = ["Risposta alla seconda."]
    rig.server.gate.set()
    assert rig.done(n1) and rig.done(n2)
    assert rig.worker.transcript.size == 0


def test_an_utterance_after_the_clear_is_recorded_again(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = ["Risposta."]
    rig.worker.transcript.clear()
    rig.turn("nuova domanda")
    assert [t[0] for t in turns(rig)] == ["user", "robot"]


def test_flag_off_then_on_while_thinking_keeps_the_reply_out(rigs):  # noqa: F811
    rig = with_ring(rigs)
    live = rig.stack.effective().webui.live
    rig.server.scripts["lan.local"] = [GATE, "Risposta."]

    def off_on(ring):
        ring.apply_config(live.model_copy(update={"transcript": False}))
        ring.apply_config(live)

    n = clear_while(rig, off_on)
    rig.server.gate.set()
    assert rig.done(n)
    assert rig.worker.transcript.size == 0


def test_sleep_while_thinking_keeps_the_reply_out_whichever_runs_first(rigs):  # noqa: F811
    rig = with_ring(rigs)
    rig.server.scripts["lan.local"] = [GATE, "Risposta."]
    n = clear_while(rig, lambda ring: ring.set_asleep(True))  # robot.asleep: before any interrupt
    rig.server.gate.set()
    assert rig.done(n)
    assert rig.worker.transcript.size == 0
    rig.worker.transcript.set_asleep(False)
    rig.server.scripts["lan.local"] = ["Buongiorno."]
    rig.turn("sono sveglio")
    assert [t[0] for t in turns(rig)] == ["user", "robot"]


def test_an_utterance_that_was_never_recorded_never_gets_a_reply_recorded(rigs):  # noqa: F811
    rig = with_ring(rigs, on=False)
    rig.server.scripts["lan.local"] = [GATE, "Risposta."]
    n = rig.say("domanda")
    rig.worker.transcript.apply_config(
        rig.stack.effective().webui.live.model_copy(update={"transcript": True})
    )
    rig.server.gate.set()
    assert rig.done(n)
    assert rig.worker.transcript.size == 0


# ------------------------------------------------------------------ minors and cuts (S2)


def test_with_a_minor_present_the_reply_is_recorded_only_once_it_played(rigs):  # noqa: F811
    rig = with_ring(rigs, routes=(CLEARED,), safety={"minors_allowed_llm": True})
    rig.minor[0] = True
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre", GATE, "."]
    n = rig.say("ciao")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    assert turns(rig) == [("user", "ciao", False)], "nothing shown as said while the turn runs"
    rig.speech.unplayed.update({2, 3})
    rig.server.gate.set()
    assert rig.done(n)
    assert turns(rig)[-1][:2] == ("robot", "Uno.")


def test_a_cut_flag_trims_even_when_the_outcome_is_overwritten(rigs):  # noqa: F811
    rig = with_ring(rigs, on=True)
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre", GATE, "."]
    n = rig.say("ciao")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    turn = rig.worker._active_turn
    turn.speech_stopped = True  # a stop happened, whatever the outcome string says later
    turn.outcome = "error"
    rig.speech.unplayed.update({2, 3})
    rig.server.gate.set()
    assert rig.done(n)
    assert turns(rig)[-1] == ("robot", "Uno.", True)
