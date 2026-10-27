"""Audio turns in the worker (SPEC 8, Realtime speech route; Conversation state): the server
holds the words of an utterance it heard, so the turn asks for a response only; a server that
answers on its own is cancelled; the items of an audio utterance no realtime response answers
are deleted."""

from __future__ import annotations

import dataclasses

from realtime_helpers import wait_until
from rt_rig import build_rt
from test_conversation_worker import utt
from test_realtime_listener import FACE_RAD, Ear
from turn_server import TurnServer

OPT_IN = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}


def test_a_server_that_answers_on_its_own_is_cancelled_and_the_turn_answers_once():
    with TurnServer(heard=[["ciao Neve"]], vad_answers=True) as server:
        rig = build_rt(server, rt={"input": "audio"}, app=OPT_IN)
        ear = Ear(rig, script=["x"], bearing=FACE_RAD, to_worker=True)
        try:
            ear.say()
            assert ear.settle(lambda: len(rig.events) == 1, 10.0)
            assert rig.route.snapshot()["unrequested"] == 1
            assert len(server.of("response.cancel")) >= 1
            assert len(server.of("response.create")) == 1  # the robot asked once
            update = server.of("session.update")
            burst = server.kinds().index("input_audio_buffer.append")
            assert server.kinds()[burst - 1] == "session.update"  # fresh instructions first
            assert update[-1]["session"]["instructions"]
        finally:
            ear.close()
            rig.close()


def test_a_replaced_audio_utterance_has_its_server_items_deleted():
    """Two utterances wait behind a turn with room for one: the older is never answered, so
    the server must not answer it either."""
    app = {**OPT_IN, "conversation": {"language": "it", "pending_utterances": 1}}
    with TurnServer(["silent"]) as server:
        rig = build_rt(server, rt={"input": "audio"}, app=app)
        try:
            assert rig.route.ready()
            rig.say("prima")
            assert wait_until(lambda: len(server.of("response.create")) == 1)
            gen = rig.route.generation
            heard = dataclasses.replace(utt("seconda"), realtime=(gen, ("in_a", "in_b")))
            assert rig.worker.submit(heard)
            assert rig.worker.submit(utt("terza"))
            assert wait_until(lambda: len(server.of("conversation.item.delete")) == 2)
            assert [e["item_id"] for e in server.of("conversation.item.delete")] == [
                "in_a",
                "in_b",
            ]
        finally:
            rig.close()


def test_the_words_go_as_text_when_the_session_that_heard_them_is_gone():
    with TurnServer() as server:
        rig = build_rt(server, rt={"input": "audio"}, app=OPT_IN)
        try:
            assert rig.route.ready()
            old = rig.route.generation
            rig.route.close_soon("conversation_end")
            heard = dataclasses.replace(utt("come stai"), realtime=(old, ("in_old",)))
            n = rig.worker.snapshot().get("turns", 0)
            assert rig.worker.submit(heard)
            assert wait_until(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)
            items = server.of("conversation.item.create")
            assert len(items) == 1
            assert items[0]["item"]["content"][0]["text"] == "come stai"
            assert server.of("conversation.item.delete") == []  # nothing of the old session
        finally:
            rig.close()


# ------------------------------------------------------------------ fix round 1


def test_an_audio_utterance_the_worker_refuses_has_its_items_deleted():
    """I-3: "stop" consumed after a barge-in: the server must not answer it later."""
    with TurnServer() as server:
        rig = build_rt(server, rt={"input": "audio"}, app=OPT_IN)
        try:
            assert rig.route.ready()
            gen = rig.route.generation
            rig.worker.interrupt()  # a barge-in just cut the robot off
            stop = dataclasses.replace(utt("stop"), realtime=(gen, ("in_stop",)))
            assert not rig.worker.submit(stop)
            assert wait_until(lambda: len(server.of("conversation.item.delete")) == 1)
            assert server.of("conversation.item.delete")[0]["item_id"] == "in_stop"
            rig.worker.close()
            late = dataclasses.replace(utt("ancora"), realtime=(gen, ("in_late",)))
            assert not rig.worker.submit(late)  # closed
            assert wait_until(lambda: len(server.of("conversation.item.delete")) == 2)
        finally:
            rig.close()


def test_a_realtime_turn_holds_the_audio_input_while_it_runs():
    """M-7: the worker holds bursts from the turn's first send to its end."""
    with TurnServer(gap_s=0.1, deltas=5) as server:
        rig = build_rt(server, rt={"input": "audio"}, app=OPT_IN)
        try:
            assert rig.route.ready()
            held = []
            hold, release = rig.route.hold_input, rig.route.release_input
            rig.route.hold_input = lambda: (held.append("hold"), hold())[1]
            rig.route.release_input = lambda: (held.append("release"), release())[1]
            rig.turn("ciao")
            assert held == ["hold", "release"]
            assert not rig.route.input_held()
        finally:
            rig.close()


def test_a_turn_sends_its_response_create_only_on_the_session_that_holds_its_words():
    """M-6: the response.create names the turn's session."""
    with TurnServer() as server:
        rig = build_rt(server, rt={"input": "audio"}, app=OPT_IN)
        try:
            asked = []
            request = rig.route.request
            rig.route.request = lambda *a, **kw: (
                asked.append(kw.get("generation")),
                request(*a, **kw),
            )[1]
            rig.turn("ciao")
            assert asked and asked[0] == rig.route.generation
        finally:
            rig.close()
