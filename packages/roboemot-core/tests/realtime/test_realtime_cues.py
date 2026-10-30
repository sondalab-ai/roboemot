"""No local voice (``voice.backend: none``): every line only Piper spoke becomes a non-verbal cue,
the confirmation fails closed, and the listening cue follows the utterances whose audio went
(SPEC 8, Non-verbal cues; CV-2)."""

from __future__ import annotations

import time

import pytest
from roboemot.config.schema import RegisterParams, StateEffects
from roboemot.health.speech_cue import SpeechCue
from roboemot.voice.listener import AUDIO_NOT_HEARD, AUDIO_SENT
from rt_rig import build_rt, wait_for
from test_conversation_tools import Store
from test_realtime_listener import FACE_RAD, Ear
from test_realtime_turns import tooled
from turn_server import TurnServer

NO_VOICE = {"voice": {"backend": "none", "fallback_backend": "none"}}


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        app = {**NO_VOICE, **kw.pop("app", {})}
        rig = build_rt(server, app=app, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def spoke_nothing(rig) -> bool:
    snap = rig.speech.snapshot()
    return snap["backend"] == "none" and snap["spoken"] == 0


# ------------------------------------------------------------------ the route cannot answer


def test_the_route_down_at_start_shows_the_fallback_cue_and_asks_no_chat_route(rigs):
    """Pins the current choice (owner decision pending): a chat route is configured (lan) and
    reachable, yet no chat request is sent; the fallback cue shows."""
    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        rig.turn()
        assert rig.chat_hosts() == [] and srv.requests == []  # nobody would hear the answer
        assert rig.cues.played == ["fallback"]
        assert rig.events[-1]["outcome"] == "no_voice"
        assert rig.route.snapshot()["fallbacks_start"] == 1
        assert spoke_nothing(rig) and rig.history() == []


def test_the_network_down_at_start_is_the_offline_reaction_not_the_fallback_cue(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        for _ in range(rig.network.policy.failures_to_offline):
            rig.network.report("lan", False)
        assert not rig.network.online
        rig.turn()
        assert rig.worker.responder.calls == 1  # puzzled gesture and the offline cue
        assert rig.cues.played == [] and rig.chat_hosts() == []
        assert rig.events[-1]["outcome"] == "offline"


def test_fallback_mid_turn_before_first_audio_is_the_cue_not_a_local_answer(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.chat_hosts() == []
        assert rig.cues.played == ["fallback"]
        assert rig.route.snapshot()["fallbacks_mid_turn"] == 1
        assert spoke_nothing(rig)


def test_fallback_mid_turn_after_first_audio_cuts_as_today(rigs):
    with TurnServer(["close_mid"], delta_ms=1500, gap_s=0.3) as srv:
        rig = rigs(srv)
        rig.turn()
        assert rig.chat_hosts() == [] and rig.cues.played == []
        assert rig.route.snapshot()["cut_mid_turn"] == 1
        assert rig.history()[0][-1]["interrupted"] is True


def test_a_loss_after_a_tool_ran_shows_the_fallback_cue_instead_of_the_piper_line(rigs):
    play = '{"gesture": "joy"}'
    with TurnServer(["tool", "close_before"], tool_name="play", tool_args=play) as srv:
        rig, env = tooled(rigs, srv)
        rig.turn()
        assert env.played == ["joy"] and rig.chat_hosts() == []
        assert rig.cues.played == ["fallback"]
        assert spoke_nothing(rig)
        assert rig.worker.snapshot()["lines_unspoken"] == 1


# ------------------------------------------------------------------ fixed lines


def test_a_confirmation_question_the_route_cannot_say_fails_closed(rigs):
    args = '{"key": "colore", "value": "blu"}'
    script = ["tool", "close_before"]
    with TurnServer(script, tool_name="remember_preference", tool_args=args) as srv:
        store = Store()
        rig, _ = tooled(rigs, srv, store=store)
        rig.turn()
        assert rig.worker._confirm is None  # nothing pending: a yes stores nothing
        snap = rig.worker.snapshot()
        assert snap["confirm_unspoken"] == 1 and snap["confirm_asked"] == 1
        rig.turn("sì", stamp=time.monotonic() + 1.0)
        assert store.saved == []
        assert spoke_nothing(rig)


def test_a_confirmation_question_the_server_did_not_voice_fails_closed(rigs):
    args = '{"key": "colore", "value": "blu"}'
    with TurnServer(["tool", "empty"], tool_name="remember_preference", tool_args=args) as srv:
        rig, _ = tooled(rigs, srv, store=Store())
        rig.turn()
        assert rig.worker._confirm is None
        assert rig.worker.snapshot()["confirm_unspoken"] == 1
        assert spoke_nothing(rig)


def test_the_offline_line_becomes_the_offline_cue_when_the_session_drops_before_it(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv)
        assert rig.route.ready()
        assert rig.worker.say_line("Sono senza rete.", "it")
        assert wait_for(lambda: rig.cues.played == ["offline"])
        assert spoke_nothing(rig)


def test_the_offline_line_while_the_route_is_down_is_the_offline_cue(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        cue = SpeechCue(
            lambda: rig.speech,
            lambda: (RegisterParams(), StateEffects(), "joy", 0.5),
            lambda: False,
            rig.worker.say_line,
            no_voice=lambda: True,
            cue=lambda: rig.cues.play("offline"),
        )
        assert cue.say("Sono senza rete.", "it")
        assert rig.cues.played == ["offline"] and spoke_nothing(rig)


# ------------------------------------------------------------------ listening cue


@pytest.fixture
def ears():
    made = []

    def make(server, **kw):
        cfg = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
        rig = build_rt(server, rt={"input": "audio"}, app=cfg)
        ear = Ear(rig, bearing=FACE_RAD, to_worker=True, **kw)
        events: list[str] = []
        ear.r.listener.bind_realtime(ear.input, lambda event, key: events.append(event))
        made.append(ear)
        return ear, events

    yield make
    for ear in made:
        ear.close()
        ear.rt.close()


def test_listening_starts_when_the_audio_leaves_and_the_turn_ends_it(ears):
    with TurnServer(heard=[["ciao Neve come stai"]]) as server:
        ear, events = ears(server, script=["x"])
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert events == [AUDIO_SENT]
        assert wait_for(lambda: ear.rt.worker.snapshot().get("turns", 0) == 1, 10.0)
        assert ear.rt.cues.ended == 1  # once, whether or not audio came


def test_a_transcript_timeout_is_not_heard_and_never_answered(ears):
    with TurnServer(heard=[[]]) as server:
        ear, events = ears(server, script=["x"], decoded="che tempo fa")
        ear.say()
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert events == [AUDIO_SENT, AUDIO_NOT_HEARD]  # the cue shows at the timeout
        assert ear.utterances[0].local_only
        assert wait_for(lambda: len(ear.rt.events) == 1, 10.0)
        assert ear.rt.events[0]["outcome"] == "not_heard"
        assert ear.rt.chat_hosts() == [] and server.of("response.create") == []
        assert ear.rt.cues.ended == 0  # the listener's not_heard ended it, not the worker


def test_the_listeners_audio_events_drive_the_cues():
    from roboemot.runtime.startup import audio_cues
    from roboemot.voice.listener import AUDIO_UNSENT
    from rt_rig import FakeCues

    cues = FakeCues()
    on_audio = audio_cues(cues)
    on_audio(AUDIO_SENT, 1.0)
    on_audio(AUDIO_UNSENT, 1.0)
    assert (cues.begun, cues.ended, cues.played) == (1, 1, [])
    on_audio(AUDIO_SENT, 2.0)
    on_audio(AUDIO_NOT_HEARD, 2.0)
    assert (cues.begun, cues.ended, cues.played) == (2, 2, ["not_heard"])


# ------------------------------------------------------------------ fix round 1


def heard(text="ciao", items=("in_1",), **kw):
    """An utterance whose audio the server transcribed (it holds the listening cue)."""
    from dataclasses import replace

    from test_conversation_worker import utt

    kw.setdefault("stamp", time.monotonic())
    return replace(utt(text, **kw), realtime=(1, items))


def test_a_turn_that_fails_before_it_starts_still_ends_the_listening_cue(rigs, monkeypatch):
    with TurnServer() as srv:
        rig = rigs(srv)

        def broken(cfg):
            raise RuntimeError("boom")

        monkeypatch.setattr(rig.hub, "sync", broken)
        assert rig.worker.submit(heard())
        assert wait_for(lambda: rig.worker.snapshot().get("errors", 0) == 1)
        assert rig.cues.ended == 1


def test_no_voice_follows_the_built_backend_not_a_hot_override(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, extra=False)
        rig.stack.set("voice.backend", "piper", source="test", admin=True)
        assert rig.stack.effective().voice.backend == "piper"
        assert rig.worker._no_voice()  # built with none: still no voice
        rig.turn()
        assert rig.cues.played == ["fallback"] and rig.chat_hosts() == []


def test_a_hot_override_to_none_keeps_the_built_local_voice():
    with TurnServer() as srv:
        rig = build_rt(srv, extra=False)
        try:
            rig.stack.set("voice.backend", "none", source="test", admin=True)
            assert not rig.worker._no_voice()
            rig.turn()
            assert rig.chat_hosts() == ["lan.local"] and rig.cues.played == []
        finally:
            rig.close()


def test_closing_ends_the_listening_cue_of_the_utterances_still_waiting(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        with rig.worker._lock:
            rig.worker._pending.append((heard(stamp=1.0), 0))
            rig.worker._pending.append((heard(stamp=2.0), 0))
        rig.worker.close()
        assert rig.cues.ended == 2


def test_a_skipped_audio_turn_ends_the_listening_cue_and_shows_no_other_cue():
    """CV-3 with CV-2: the real listener, worker and cues; the server calls the skip tool."""
    from roboemot.config.schema import CueConfig
    from roboemot.realtime.turn import SKIP_TOOL
    from roboemot.runtime.startup import audio_cues
    from roboemot.voice.cues import Cues
    from test_cues import prim

    app = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
    with TurnServer(["tool"], tool_name=SKIP_TOOL, heard=[["la TV parla"]]) as srv:
        rig = build_rt(srv, rt={"input": "audio", "skip_tool": True}, app=app)
        cues = Cues(
            CueConfig(listening_onset_s=0.0),
            gesture=lambda name: prim(name),
            sounds={},
        )
        rig.worker.cues = cues
        ear = Ear(rig, script=["x"], bearing=FACE_RAD, to_worker=True)
        ear.r.listener.bind_realtime(ear.input, audio_cues(cues))
        try:
            ear.say()
            assert ear.settle(lambda: cues.snapshot()["listening"]["outstanding"] == 1)
            cues.commands(time.monotonic())  # the onset passed: the gesture shows
            assert ear.settle(lambda: len(rig.events) == 1, 10.0)
            assert rig.events[0]["outcome"] == "skipped"
            snap = cues.snapshot()
            assert snap["listening"] == {"holding": False, "outstanding": 0, "generation": 0}
            assert snap["counts"]["listening"] == {"shown": 1, "released_ended": 1}
            assert not any(snap["counts"][k] for k in ("fallback", "not_heard", "offline"))
        finally:
            ear.close()
            rig.close()
