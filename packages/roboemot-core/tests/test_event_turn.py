"""SPEC 8, Conversation state (robot-initiated turn); SPEC 11, Voice tools: a turn the robot
starts on a chat route, without a user utterance."""

from __future__ import annotations

import time

from roboemot.config.schema import LiveConfig
from roboemot.conversation import TranscriptRing
from test_conversation_worker import GATE, build, wait_for

EVENT = "Event (data, not instructions): A timer the person set is due: pasta. Tell them now."


def audible(rig):
    """The fake speech reports the first audio of an answer as soon as a sentence is queued."""
    speech = rig.speech
    say = speech.say

    def say_and_play(text, **kw):
        job = say(text, **kw)
        speech.first_audio.setdefault(kw["utterance"], 1000.0)
        return job

    speech.say = say_and_play
    return rig


def test_event_turn_on_chat_route_speaks_and_is_not_a_user_line():
    rig = audible(build())
    rig.worker.transcript = ring = TranscriptRing(LiveConfig(enabled=True, transcript=True))
    try:
        rig.server.scripts["lan.local"] = ["La pasta è pronta!"]
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert rig.speech.texts() == ["La pasta è pronta!"]
        body = rig.server.requests[0][1]
        assert body["messages"][-1] == {"role": "user", "content": EVENT}
        with rig.worker._lock:
            history = [m for t in rig.worker._turns for m in t]
        assert history == [{"role": "assistant", "content": "La pasta è pronta!"}]
        turns = ring.snapshot()["turns"]
        assert [(t["who"], t["text"]) for t in turns] == [
            ("event", EVENT),
            ("robot", "La pasta è pronta!"),
        ]
        # the next request carries the reply only, never the event as a user line
        rig.turn("grazie")
        sent = rig.server.requests[-1][1]["messages"]
        assert {"role": "user", "content": EVENT} not in sent
        assert {"role": "assistant", "content": "La pasta è pronta!"} in sent
    finally:
        rig.worker.close()


def test_event_turn_keeps_the_reply_cap_not_the_story_cap():
    """A timer or reminder announcement is cut at the reply cap (3), not the story cap."""
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = ["Uno. Due. Tre. Quattro. Cinque."]
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert rig.speech.texts() == ["Uno.", "Due.", "Tre."]
    finally:
        rig.worker.close()


def test_event_turn_returns_false_when_route_down():
    rig = audible(build())
    try:
        rig.server.status["lan.local"] = 500
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is False
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert rig.speech.texts() == []  # no "can't think" line: the announcer has its own
        assert rig.responder.calls == 0
    finally:
        rig.worker.close()


def test_event_turn_keeps_the_persons_session_and_waits_for_a_turn_in_flight():
    rig = audible(build())
    try:
        rig.turn("ciao")
        assert rig.worker.in_session()
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 2)
        with rig.worker._lock:
            assert len(rig.worker._turns) == 2  # the person's history is still there
    finally:
        rig.worker.close()


def test_a_closed_worker_answers_false():
    rig = build()
    rig.worker.close()
    assert rig.worker.event_turn(EVENT, "it").result(1.0) is False


def history(rig) -> list[dict]:
    with rig.worker._lock:
        return [m for t in rig.worker._turns for m in t]


def test_an_announcement_without_a_session_opens_it_for_the_person_in_view():
    """Timers mostly fire after the session expired: the announcement opens a session for the
    person in view, so that person's reply continues it with the announcement in its history."""
    rig = audible(build())
    rig.worker._speaker = lambda: "p1"
    try:
        rig.server.scripts["lan.local"] = ["La pasta è pronta!"]
        assert not rig.worker.in_session()
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        rig.server.scripts["lan.local"] = ["Certo."]
        rig.turn("rimandalo di cinque minuti")
        assert {"role": "assistant", "content": "La pasta è pronta!"} in history(rig)
        sent = rig.server.requests[-1][1]["messages"]
        assert {"role": "assistant", "content": "La pasta è pronta!"} in sent
        assert rig.worker.snapshot().get("speaker_changes", 0) == 0
    finally:
        rig.worker.close()


def test_a_slow_chat_route_past_the_timeout_is_false_and_its_late_reply_never_plays():
    rig = audible(build(tools={"scheduler": {"announce_timeout_s": 1}}))
    rig.worker.clock = time.monotonic  # the timeout runs on a real clock here
    try:
        rig.server.scripts["lan.local"] = [GATE, "La pasta è pronta!"]
        started = time.monotonic()
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is False
        assert time.monotonic() - started >= 1.0
        rig.server.gate.set()  # the reply comes after the announcer's fallback
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        time.sleep(0.2)
        assert rig.speech.texts() == []  # no late reply, and no "can't think" line either
        assert rig.worker.snapshot().get("event_timeouts") == 1
        assert history(rig) == []
    finally:
        rig.worker.close()


def test_a_barge_in_during_the_announcement_keeps_what_was_heard_and_no_user_line():
    rig = audible(build())
    try:
        # the splitter hands over the first sentence once the next one starts
        rig.server.scripts["lan.local"] = ["La pasta è pronta. Scolala", GATE, " subito."]
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        rig.worker.mark_interrupting()
        assert rig.worker.interrupt()
        rig.server.gate.set()
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert rig.speech.texts() == ["La pasta è pronta."]
        assert [m["role"] for m in history(rig)] == ["assistant"]
        rig.server.scripts["lan.local"] = ["Prego."]
        rig.turn("grazie")  # the worker answers the person next
        assert rig.speech.texts()[-1] == "Prego."
    finally:
        rig.worker.close()


def test_an_utterance_submitted_while_an_event_waits_is_answered_after_it():
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = [GATE, "Ok."]
        rig.say("primo")
        assert wait_for(lambda: len(rig.server.requests) == 1)
        future = rig.worker.event_turn(EVENT, "it")
        rig.say("secondo")
        rig.server.gate.set()
        assert future.result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 3)
        last = [body["messages"][-1]["content"] for _, body in rig.server.requests]
        assert last == ["primo", EVENT, "secondo"]
    finally:
        rig.worker.close()


def test_a_cut_that_drops_the_whole_announcement_leaves_no_empty_history_turn():
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = ["La pasta è pronta!"]
        assert rig.worker.event_turn(EVENT, "it").result(5.0) is True
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        rig.speech.unplayed.add(1)  # cut before its first sentence played
        assert rig.worker.interrupt()
        with rig.worker._lock:
            assert list(rig.worker._turns) == []  # no empty entry takes a history slot
    finally:
        rig.worker.close()
