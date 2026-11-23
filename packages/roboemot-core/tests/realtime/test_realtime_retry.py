"""A cloud reply lost before the robot said anything is asked again (SPEC 8, Realtime speech
route, Lost reply retry): up to ``retry.max_retries`` more attempts within
``retry.total_budget_s``, the first one announced with the ``retrying`` clip; when all fail the
robot without a local voice plays the ``sorry`` clip and one with a local voice lets the chat
routes answer (SPEC 12A: the ``retry`` counts)."""

from __future__ import annotations

import collections
import dataclasses

import pytest
from rt_rig import build_rt, rt_route, wait_for
from test_conversation_worker import utt
from test_realtime_listener import ears  # noqa: F401 (fixture)
from test_realtime_turns import tooled
from turn_server import REPLY, TurnServer

NO_VOICE = {"voice": {"backend": "none"}}
SORRY_IT = "Scusa, ho perso la connessione. Puoi ripetere?"


class FakeClips:
    """The runtime's voice clips as the worker sees them (``play_clip``): every kind is cached
    unless ``missing`` names it."""

    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.played: collections.Counter[str] = collections.Counter()
        self.missing = missing

    def play(self, kind: str) -> bool:
        if kind in self.missing:
            return False
        self.played[kind] += 1
        return True


@pytest.fixture()
def rigs():
    made = []

    def make(server, clips=None, **kw):
        rig = build_rt(server, **kw)
        rig.clips = clips if clips is not None else FakeClips()
        rig.worker.play_clip = rig.clips.play
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def retry(rig) -> dict:
    return rig.worker.snapshot()["retry"]


def reply_events(rig) -> list[tuple[str, dict]]:
    return [(e["reason"], e) for e in rig.diag if e["kind"] == "reply"]


def test_a_reply_lost_before_any_audio_is_retried_and_answered(rigs):
    with TurnServer(["close_before", "reply"]) as srv:
        rig = rigs(srv)
        rig.turn()
        assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)  # one answer played
        counts = retry(rig)
        assert counts["attempts"] == 1 and counts["recovered_1"] == 1
        assert counts["gave_up"] == 0
        assert rig.chat_hosts() == []
        assert rig.events[-1]["outcome"] == "ok"
        history = rig.history()
        assert [m["role"] for m in history[0]] == ["user", "assistant"]
        assert history[0][1]["content"] == REPLY
        reasons = [r for r, _ in reply_events(rig)]
        assert reasons == ["retry", "recovered"]
        _, first = reply_events(rig)[0]
        assert first["attempt"] == 1 and first["limit"] == 2


def test_the_first_retry_is_announced_with_the_retrying_clip(rigs):
    with TurnServer(["close_before", "close_before", "reply"]) as srv:
        rig = rigs(srv)
        rig.turn()
        assert retry(rig)["recovered_2"] == 1
        assert rig.clips.played == {"retrying": 1}  # the second retry is not announced


def test_the_announcement_can_be_turned_off(rigs):
    with TurnServer(["close_before", "reply"]) as srv:
        rig = rigs(srv, rt={"retry": {"announce_first_retry": False}})
        rig.turn()
        assert retry(rig)["recovered_1"] == 1
        assert rig.clips.played == {}


def test_two_failed_retries_end_with_the_sorry_clip_on_a_robot_without_local_voice(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv, app=NO_VOICE)
        rig.turn()
        counts = retry(rig)
        assert counts["attempts"] == 2 and counts["gave_up"] == 1
        assert rig.clips.played == {"retrying": 1, "sorry": 1}
        assert rig.chat_hosts() == []
        assert rig.cues.played == []  # the clip took the fallback cue's place
        assert rig.events[-1]["outcome"] == "realtime_lost"
        assert [r for r, _ in reply_events(rig)] == ["retry", "retry", "gave_up"]
        assert [e["attempt"] for r, e in reply_events(rig) if r == "retry"] == [1, 2]


def test_without_a_sorry_clip_the_fallback_cue_shows(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv, clips=FakeClips(missing=("sorry",)), app=NO_VOICE)
        rig.turn()
        assert retry(rig)["gave_up"] == 1
        assert rig.cues.played == ["fallback"]


def test_a_robot_with_a_local_voice_tries_the_chat_route_after_the_retries(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv)
        rig.turn()
        counts = retry(rig)
        assert counts["attempts"] == 2 and counts["gave_up"] == 1
        assert rig.chat_hosts() == ["lan.local"]
        assert rig.clips.played == {"retrying": 1}  # an answer beats an apology


def test_the_local_voice_apologizes_when_the_chat_route_fails_too(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv)
        rig.chat.status["lan.local"] = 500
        rig.turn()
        assert retry(rig)["gave_up"] == 1
        assert wait_for(lambda: rig.speech.backend.texts == [SORRY_IT])


def test_no_retry_after_a_skill_ran(rigs):
    play = '{"gesture": "joy"}'
    with TurnServer(["tool", "close_before"], tool_name="play", tool_args=play) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert env.played == ["joy"]  # once
        counts = retry(rig)
        assert counts["attempts"] == 0 and counts["skipped_tool_ran"] == 1
        assert rig.route.snapshot()["cut_mid_turn"] == 1
        assert rig.clips.played == {}


def test_no_retry_after_audio_played(rigs):
    with TurnServer(["close_mid"], delta_ms=1500, gap_s=0.3) as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.route.snapshot()["cut_mid_turn"] == 1
        counts = retry(rig)
        assert counts["attempts"] == 0 and counts["skipped_played"] == 1
        assert rig.chat_hosts() == []


def test_a_retry_on_the_same_session_sends_the_words_as_text(rigs):
    """The server refuses the response (an ``error`` echoing it) and keeps the session: the
    retry runs on it. The server's own items of the audio are deleted first and the words go
    as one ``input_text`` item, so they are not twice in the conversation."""
    with TurnServer(["error", "reply"]) as srv:
        rig = rigs(
            srv,
            rt={"input": "audio"},
            app={"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}},
        )
        assert rig.route.ready()
        gen = rig.route.generation
        heard = dataclasses.replace(utt("come stai"), realtime=(gen, ("in_a",)))
        n = rig.worker.snapshot().get("turns", 0)
        assert rig.worker.submit(heard)
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)
        assert retry(rig)["recovered_1"] == 1
        assert len(srv.connections) == 1 and rig.route.generation == gen
        kinds = srv.kinds()
        assert [e["item_id"] for e in srv.of("conversation.item.delete")] == ["in_a"]
        texts = [
            e["item"]["content"][0]
            for e in srv.of("conversation.item.create")
            if e["item"]["type"] == "message"
        ]
        assert texts == [{"type": "input_text", "text": "come stai"}]
        assert kinds.index("conversation.item.delete") < kinds.index("conversation.item.create")
        assert len(srv.of("response.create")) == 2


def test_a_newer_utterance_wins_over_a_retry(rigs):
    """The first reply stalls while the person speaks again: the newer utterance is answered,
    the lost one is not asked again, and nobody apologises for it."""
    with TurnServer(["silent", "reply"]) as srv:
        rig = rigs(srv, route=rt_route(srv, first_token_timeout_s=1))
        assert rig.route.ready()
        n = rig.say("prima")
        assert wait_for(lambda: len(srv.of("response.create")) == 1)  # in service: not replaced
        rig.say("seconda")  # waits behind the first turn
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n + 1, 10.0)
        counts = retry(rig)
        assert counts["skipped_newer_utterance"] == 1 and counts["attempts"] == 0
        assert rig.clips.played == {}
        assert [e["outcome"] for e in rig.events] == ["realtime_lost", "ok"]
        # the lost one never went to the chat routes (the newer one may, if the session that
        # stalled is not back yet)
        assert len(rig.chat_hosts()) <= 1


def test_the_total_budget_skips_a_retry_that_cannot_start_in_time(rigs):
    """The first retry starts at once and stalls past the budget: the second is skipped."""
    with TurnServer(["close_before", "created_stall"]) as srv:
        rig = rigs(
            srv,
            route=rt_route(srv, first_token_timeout_s=1, stall_timeout_s=1.5),
            rt={"retry": {"total_budget_s": 1}},
            app=NO_VOICE,
        )
        rig.turn()
        counts = retry(rig)
        assert counts["attempts"] == 1 and counts["skipped_budget"] == 1
        assert counts["gave_up"] == 1
        assert rig.clips.played["sorry"] == 1


def test_no_retry_when_the_retry_is_off(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv, rt={"retry": {"enabled": False}})
        rig.turn()
        assert retry(rig)["attempts"] == 0
        assert rig.chat_hosts() == ["lan.local"]
        assert rig.clips.played == {}


def test_the_report_has_every_retry_count_from_the_start(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        assert retry(rig) == dict.fromkeys(
            (
                "attempts",
                "recovered_1",
                "recovered_2",
                "gave_up",
                "skipped_played",
                "skipped_tool_ran",
                "skipped_newer_utterance",
                "skipped_no_words",
                "skipped_budget",
            ),
            0,
        )


def test_the_runtime_wires_the_voice_clips_to_the_worker(tmp_path):
    """``start_voice_clips`` hands ``VoiceClips.play`` to the worker (``play_clip``)."""
    from contextlib import ExitStack
    from types import SimpleNamespace

    from roboemot.config import ConfigStack
    from roboemot.runtime.startup import start_voice_clips

    worker = SimpleNamespace(play_clip=None)
    host = SimpleNamespace(conversation=worker, realtime_route=object(), clips=None)
    stack = ConfigStack(app={"data_dir": str(tmp_path)})
    with ExitStack() as cleanup:
        start_voice_clips(host, stack, cleanup)
        assert host.clips is not None
        assert worker.play_clip == host.clips.play


def test_a_server_turn_lost_before_its_audio_is_retried_by_the_robot_with_its_words(ears):  # noqa: F811
    """Server turns: the server's own response stalls after its words came; the retry is the
    robot's request, the words sent as text (SPEC 8, Lost reply retry)."""
    from test_realtime_server_turns import HEARD, OPT_INS, SERVER

    with TurnServer(["created_stall", "reply"], stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        rig = ear.rt
        assert ear.settle(lambda: rig.worker.snapshot().get("turns", 0) == 1, 15.0)
        assert retry(rig)["recovered_1"] == 1
        assert rig.events[-1]["outcome"] == "ok"
        assert len(server.of("response.create")) == 1  # the retry: the robot asked
        texts = [
            e["item"]["content"][0]["text"]
            for e in server.of("conversation.item.create")
            if e["item"]["type"] == "message"
        ]
        assert texts == [HEARD]


def lost_and_unreachable(rigs, srv, **kw):
    """A rig whose session is up; every later connect is refused, so after the loss the route
    stays not ready (the retry waits for it)."""
    rig = rigs(srv, **kw)
    assert rig.route.ready()
    srv.refuse_connects = 1000
    return rig


def test_a_session_not_back_in_time_is_a_failed_attempt(rigs):
    rt = {"retry": {"max_retries": 1, "reconnect_wait_s": 0.5}}
    with TurnServer(["close_before"]) as srv:
        rig = lost_and_unreachable(rigs, srv, rt=rt, app=NO_VOICE)
        rig.turn()
        counts = retry(rig)
        assert counts["attempts"] == 1 and counts["gave_up"] == 1
        assert srv.of("response.create") and len(srv.of("response.create")) == 1  # no resend
        assert rig.clips.played == {"retrying": 1, "sorry": 1}
        [ended] = [e for e in rig.diag if e["kind"] == "turn"]
        assert ended["reason"] == "realtime_lost" and ended["failure"] == "not_ready"


def test_a_barge_in_during_the_reconnect_wait_ends_the_turn(rigs):
    """The person cuts in while the retry waits for the session: the turn ends interrupted at
    once, says nothing, and holds neither the audio input nor the realtime lock."""
    rt = {"retry": {"reconnect_wait_s": 10}}
    with TurnServer(["close_before"]) as srv:
        rig = lost_and_unreachable(rigs, srv, rt=rt, app=NO_VOICE)
        n = rig.say()
        assert wait_for(lambda: reply_events(rig), 5.0)  # the first retry is waiting
        assert rig.worker.interrupt()
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n, 3.0)
        assert rig.events[-1]["outcome"] == "interrupted"
        counts = retry(rig)
        assert counts["attempts"] == 1 and counts["gave_up"] == 0
        assert rig.clips.played == {"retrying": 1}  # no apology over the person
        assert rig.cues.played == []
        assert not rig.route.input_held()
        assert not rig.worker._rt_lock.locked()
        srv.refuse_connects = 0
        rig.turn("ancora")  # the next turn is served
        assert rig.events[-1]["outcome"] != "interrupted"


def test_a_newer_utterance_during_the_reconnect_wait_ends_the_wait(rigs):
    rt = {"retry": {"reconnect_wait_s": 10}}
    with TurnServer(["close_before"]) as srv:
        rig = lost_and_unreachable(rigs, srv, rt=rt, app=NO_VOICE)
        n = rig.say("prima")
        assert wait_for(lambda: reply_events(rig), 5.0)  # the first retry is waiting
        rig.say("seconda")
        assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n, 3.0)
        counts = retry(rig)
        assert counts["skipped_newer_utterance"] == 1 and counts["gave_up"] == 0
        assert rig.events[0]["outcome"] == "realtime_lost"
        assert "sorry" not in rig.clips.played
