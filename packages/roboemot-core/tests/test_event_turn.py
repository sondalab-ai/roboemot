"""SPEC 8, Conversation state (robot-initiated turn); SPEC 11, Voice tools: a turn the robot
starts on a chat route, without a user utterance."""

from __future__ import annotations

from roboemot.config.schema import LiveConfig
from roboemot.conversation import TranscriptRing
from test_conversation_worker import build, wait_for

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
