"""SPEC 8, Robot expression: a reaction turn gives way to a person and never jumps ahead."""

from __future__ import annotations

import time

from roboemot.config.schema import LiveConfig
from roboemot.conversation import TranscriptRing
from test_conversation_worker import GATE, build, utt, wait_for
from test_event_turn import audible

EVENT_PREFIX = "Event (data, not instructions): "
EVENT = EVENT_PREFIX + "A person is showing you something that looks like a cup."


def last_user_texts(rig) -> list[str]:
    return [body["messages"][-1]["content"] for _, body in rig.server.requests]


def history(rig) -> list[dict]:
    with rig.worker._lock:
        return [m for t in rig.worker._turns for m in t]


def test_reaction_turn_refuses_when_an_utterance_is_pending():
    """Review Focus 2: the reaction saw an idle room, then a person spoke before the loop took
    it. The person goes first; the reaction is never sent."""
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = [GATE, "Ok."]
        rig.say("primo")
        assert wait_for(lambda: len(rig.server.requests) == 1)
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert rig.worker.submit(utt("ciao"))
        rig.server.gate.set()
        assert fut.result(timeout=2) == "busy"
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 2)
        assert last_user_texts(rig) == ["primo", "ciao"]
        assert all(EVENT_PREFIX not in t for t in last_user_texts(rig))
        assert rig.worker.snapshot().get("reactions_busy") == 1
    finally:
        rig.worker.close()


def test_a_reaction_held_in_line_past_its_max_age_is_never_said():
    """S1: a person's turn holds the loop longer than the reaction's max age; when the loop
    finally takes the reaction it no longer applies (``fresh``) and gives way: no speech."""
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = [GATE, "Ok."]
        rig.say("primo")
        assert wait_for(lambda: len(rig.server.requests) == 1)
        deadline = time.monotonic() + 0.2  # its max age, on the caller's clock
        fut = rig.worker.reaction_turn(
            EVENT, "it", timeout_s=5, quiet=lambda: True, fresh=lambda: time.monotonic() < deadline
        )
        time.sleep(0.3)  # the person's turn still holds the loop past the max age
        rig.server.gate.set()
        assert fut.result(timeout=5) == "busy"
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        time.sleep(0.05)
        assert last_user_texts(rig) == ["primo"]
        assert rig.speech.texts() == ["Ok."]  # the person's answer only
    finally:
        rig.worker.close()


def test_a_reaction_given_up_before_it_is_taken_is_never_said():
    rig = audible(build())
    try:
        rig.server.scripts["lan.local"] = [GATE, "Ok."]
        rig.say("primo")
        assert wait_for(lambda: len(rig.server.requests) == 1)
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert fut.cancel()  # the expressor closed
        rig.server.gate.set()
        assert wait_for(lambda: rig.worker.snapshot().get("reactions_busy") == 1)
        assert last_user_texts(rig) == ["primo"]
    finally:
        rig.worker.close()


def test_person_turn_and_conversed_follow_the_persons_words():
    """S2, S5: a reaction's own turn is not a person's turn and holds no person's words."""
    rig = audible(build())
    try:
        assert rig.worker.person_turn_at() == float("-inf") and not rig.worker.conversed()
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert fut.result(timeout=5) == "spoken"
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert rig.worker.person_turn_at() == float("-inf") and not rig.worker.conversed()
        rig.turn("ciao")
        assert rig.worker.person_turn_at() == rig.clock[0] and rig.worker.conversed()
    finally:
        rig.worker.close()


def test_the_outer_future_resolves_when_the_turn_fails():
    """The reaction's future always resolves, whatever happens to the turn."""
    rig = audible(build())
    try:

        async def boom(ev):
            raise RuntimeError("x")

        rig.worker._answer_event_turn = boom
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert fut.result(timeout=5) == "no_route"
    finally:
        rig.worker.close()


def test_reaction_turn_refuses_when_room_not_quiet():
    rig = audible(build())
    try:
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: False)
        assert fut.result(timeout=2) == "busy"
        time.sleep(0.05)
        assert rig.server.requests == []
        assert not rig.worker.in_session()  # a refusal opens no session
        assert rig.worker.snapshot().get("turns", 0) == 0
    finally:
        rig.worker.close()


def test_reaction_turn_speaks_and_is_not_a_user_line():
    rig = audible(build())
    rig.worker.transcript = ring = TranscriptRing(LiveConfig(enabled=True, transcript=True))
    try:
        rig.server.scripts["lan.local"] = ["Che bella tazza!"]
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert fut.result(timeout=5) == "spoken"
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        assert history(rig) == [{"role": "assistant", "content": "Che bella tazza!"}]
        turns = ring.snapshot()["turns"]
        assert [(t["who"], t["text"]) for t in turns] == [
            ("event", EVENT),
            ("robot", "Che bella tazza!"),
        ]
    finally:
        rig.worker.close()


def test_reaction_turn_no_route_after_its_own_timeout():
    rig = audible(build(tools={"scheduler": {"announce_timeout_s": 60}}))
    try:
        rig.server.scripts["lan.local"] = [GATE, "Che bella tazza!"]
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=1.0, quiet=lambda: True)
        assert wait_for(lambda: len(rig.server.requests) == 1)
        rig.clock[0] += 1.1  # past its own timeout, well inside announce_timeout_s
        assert fut.result(timeout=5) == "no_route"
        rig.server.gate.set()
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) == 1)
        time.sleep(0.1)
        assert rig.speech.texts() == []  # the late reply never plays
    finally:
        rig.worker.close()


def test_profile_false_leaves_out_the_profile_summary():
    rig = audible(build())
    rig.profile[0] = "PROFILE-7f1e"
    try:
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True, profile=False)
        assert fut.result(timeout=5) == "spoken"
        sent = rig.server.requests[0][1]["messages"]
        assert all("PROFILE-7f1e" not in str(m["content"]) for m in sent)
        fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
        assert fut.result(timeout=5) == "spoken"
        sent = rig.server.requests[1][1]["messages"]
        assert any("PROFILE-7f1e" in str(m["content"]) for m in sent)
    finally:
        rig.worker.close()


def test_a_closed_worker_answers_no_route():
    rig = build()
    rig.worker.close()
    fut = rig.worker.reaction_turn(EVENT, "it", timeout_s=5, quiet=lambda: True)
    assert fut.result(timeout=1) == "no_route"


def test_local_voice_follows_the_built_backend():
    rig = build()
    try:
        assert rig.worker.local_voice() is True
        rig.speech.backend = type("B", (), {"name": "none"})()
        assert rig.worker.local_voice() is False
    finally:
        rig.worker.close()


def test_greeting_name_is_aliased_with_pseudonyms(monkeypatch):
    """SPEC 8 Pseudonyms, 12: with ``privacy.pseudonyms_to_cloud`` the greeting carries the
    registry's whole name (a first name alone is not caught by the map), and the cloud chat
    route receives its alias, never the name."""
    import functools
    import json
    from types import SimpleNamespace

    import test_conversation_worker
    from expressor_rig import Rig as ExpressorRig
    from roboemot.conversation import ConversationWorker
    from roboemot.llm.pseudonyms import NameSource
    from roboemot.runtime.expressor import event_text

    privacy = {"pseudonyms_to_cloud": True}
    cloud = {"name": "cloud", "base_url": "http://api.example.com/v1", "model": "m1"}
    source = NameSource(lambda: ["Marcello Barile"])
    monkeypatch.setattr(
        test_conversation_worker,
        "ConversationWorker",
        functools.partial(ConversationWorker, name_source=source),
    )
    reactions = ExpressorRig(app={"privacy": privacy})
    record = SimpleNamespace(display_name="Marcello Barile", status="active")
    reactions.expressor.registry = SimpleNamespace(get_person=lambda person_id: record)
    request = reactions.request("returning_person", {"wake": ""}, key="p1")
    text = event_text(reactions.expressor._named(request))
    assert "Marcello Barile" in text
    rig = audible(test_conversation_worker.build(routes=(cloud,), privacy=privacy))
    try:
        fut = rig.worker.reaction_turn(text, "it", timeout_s=5, quiet=lambda: True, profile=False)
        assert fut.result(5.0) == "spoken"
        sent = json.dumps([body for _, body in rig.server.requests], ensure_ascii=False)
        assert "Marcello" not in sent and "Barile" not in sent
        assert source.peek().to_alias("Marcello Barile") in sent
    finally:
        rig.worker.close()
