"""SPEC 8, Robot expression and Non-verbal cues: an addressed utterance the robot did not catch
is asked about in its own words, the did-not-get-it cue keeping its gesture without its sound.
With music or offline the cue stays as it was; a television, an unaddressed stream and a face
wake that heard nothing are never asked about."""

from __future__ import annotations

import pytest
from expressor_rig import Rig as ExpressorRig
from roboemot.runtime.startup import audio_cues
from roboemot.voice.listener import AUDIO_NOT_HEARD, AUDIO_NOT_HEARD_SPOKEN
from rt_rig import FakeCues, wait_for
from test_realtime_listener import FACE_RAD, Ear, say_after_quiet
from test_realtime_server_turns import OPT_INS, SERVER
from turn_server import TurnServer


@pytest.fixture()
def ears():
    """A listener bound to a real realtime input and worker, its audio events driving recorded
    cues through the runtime's ``audio_cues``; the policy addresses every voice at a face."""
    from rt_rig import build_rt

    made = []

    def make(server, *, rt=None, app=None, **kw):
        cfg = {
            "privacy": {"realtime_audio": True},
            "voice": {"wake_word_policy": "never"},
            **(app or {}),
        }
        rig = build_rt(server, rt={"input": "audio", **(rt or {})}, app=cfg)
        kw.setdefault("bearing", FACE_RAD)
        ear = Ear(rig, to_worker=True, **kw)
        cues = FakeCues()
        events: list[str] = []
        on_audio = audio_cues(cues)

        def record(event, key):
            events.append(event)
            on_audio(event, key)

        ear.r.listener.bind_realtime(ear.input, record)
        made.append(ear)
        return ear, cues, events

    yield make
    for ear in made:
        ear.close()
        ear.rt.close()


def transcript_timeout(ear, server):
    """Say something whose transcript never comes, and move past the transcript wait."""
    ear.say()
    assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
    ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
    assert ear.settle(lambda: len(ear.utterances) == 1)


def offers(expressor):
    """``on_not_heard`` through the expressor, recording each code it is called with."""
    calls: list[str] = []

    def offer(why):
        calls.append(why)
        return expressor.offer_not_heard(why)

    return offer, calls


def test_spoken_question_replaces_the_sound_keeps_the_gesture(ears):
    with TurnServer(heard=[[]]) as server:
        ear, cues, events = ears(server, script=["x"], decoded="che tempo fa")
        reactions = ExpressorRig()
        ear.r.listener.on_not_heard, calls = offers(reactions.expressor)
        transcript_timeout(ear, server)
        assert calls == ["no_transcript"]
        assert events[-1] == AUDIO_NOT_HEARD_SPOKEN
        assert cues.reasons == [("not_heard", "no_transcript", {"sound": False})]
        assert reactions.expressor.snapshot()["queued"] == 1
        assert reactions.expressor._take().situation == "didnt_catch"


def test_music_shows_the_cue_gesture_without_its_sound(ears):
    """S6, the owner's hard rule: while the radio plays a reaction is a gesture only. The
    did-not-get-it cue shows its gesture, never its sound, and no second gesture plays."""
    with TurnServer(heard=[[]]) as server:
        ear, cues, events = ears(server, script=["x"], decoded="che tempo fa")
        reactions = ExpressorRig(
            app={"expression": {"reactions": {"didnt_catch": {"gesture": "x"}}}}
        )
        reactions.media.is_playing = True
        ear.r.listener.on_not_heard, calls = offers(reactions.expressor)
        transcript_timeout(ear, server)
        assert calls == ["no_transcript"]
        assert events[-1] == AUDIO_NOT_HEARD_SPOKEN
        assert cues.reasons == [("not_heard", "no_transcript", {"sound": False})]
        assert reactions.host.expression.requests == []  # the cue's gesture is the only one
        assert reactions.expressor.snapshot()["queued"] == 0
        assert reactions.counts("didnt_catch") == {"gesture_only_music": 1}
        assert reactions.media.calls == []


def test_offline_keeps_the_cue_as_today(ears):
    with TurnServer(heard=[[]]) as server:
        ear, cues, events = ears(server, script=["x"], decoded="che tempo fa")
        reactions = ExpressorRig(online=False)
        ear.r.listener.on_not_heard, calls = offers(reactions.expressor)
        transcript_timeout(ear, server)
        assert events[-1] == AUDIO_NOT_HEARD
        assert cues.reasons == [("not_heard", "no_transcript", {})]
        assert reactions.expressor.snapshot()["counts"] == {}


def test_decodes_lost_offers_without_the_audio_event_consumer(ears):
    """The audio was refused (no realtime audio) and the local decode heard no word: the
    reaction is offered once even though nothing consumes the audio events."""
    with TurnServer() as server:
        ear, _, _ = ears(server, script=["x"], decoded="")
        ear.r.listener.bind_realtime(ear.input)  # no on_audio: the cues are not wired
        calls: list[str] = []
        ear.r.listener.on_not_heard = lambda why: calls.append(why) or True
        ear.rt.stack.set("privacy.realtime_audio", False, source="test", admin=True)
        say_after_quiet(ear, 1.0)
        assert ear.settle(lambda: calls)
        assert calls == ["decodes_lost"]
        assert ear.r.listener.snapshot()["realtime_audio"]["decodes_lost"] == 1


def test_unaddressed_stream_that_times_out_offers_nothing(ears):
    """Server turns: a stream with no wake decision whose transcript never came is not asked
    about (a television may have spoken), and its cue is as today."""
    with TurnServer(stream_vad=True, heard=[[]]) as server:
        app = {**OPT_INS, "voice": {"wake_word_policy": "always"}}
        ear, cues, events = ears(server, rt=SERVER, app=app, script=["x"], bearing=None)
        calls: list[str] = []
        ear.r.listener.on_not_heard = lambda why: calls.append(why) or True
        ear.say(0.5)
        assert ear.settle(lambda: "sent" in events)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s + 1
        assert ear.settle(lambda: AUDIO_NOT_HEARD in events or len(ear.utterances) == 1, 10.0)
        assert calls == []
        assert AUDIO_NOT_HEARD_SPOKEN not in events


def test_unanswered_without_a_wake_decision_offers_nothing():
    """A stream no server answer took is asked about only when its words held the robot's
    name: a stream carries no other wake decision (it may be a television)."""
    import dataclasses

    from test_conversation_worker import build, utt

    rig = build()
    try:
        cues = FakeCues()
        rig.worker.cues = cues
        calls: list[str] = []
        rig.worker.on_not_heard = lambda why: calls.append(why) or True
        rig.worker._unanswered(utt("x", addressed=False))
        rig.worker._unanswered(utt("x", addressed=True))  # streamed, no name in its words
        assert calls == []
        assert cues.reasons == [("not_heard", "no_server_answer", {"sound": True})] * 2
        named = dataclasses.replace(utt("Neve, x", addressed=True), wake_word=True)
        rig.worker._unanswered(named)
        assert calls == ["no_server_answer"]
        assert cues.reasons[-1] == ("not_heard", "no_server_answer", {"sound": False})
    finally:
        rig.worker.close()


def test_question_opens_the_follow_up_window():
    """After the spoken question, the person answers without the robot's name (the wake word
    is required otherwise): the answer is addressed and goes to the worker."""
    from roboemot.config.schema import RoboEmotConfig
    from test_voice_listener import Rig as ListenerRig

    cfg = RoboEmotConfig.model_validate({"voice": {"wake_word_policy": "always"}})
    ear = ListenerRig(["la radio"], cfg=cfg, enrolled_in_view=lambda now: True)
    ear.say()
    assert not ear.utterances[-1].addressed  # no name: not for the robot
    reactions = ExpressorRig()
    reactions.host.voice_listener = ear.listener
    assert reactions.expressor.handle(reactions.request("didnt_catch")) == "spoken"
    ear.say()
    assert ear.utterances[-1].addressed and not ear.utterances[-1].wake_word


def test_one_utterance_asks_once(ears):
    """A transcript timeout, then the worker's turn for its local decode (``local_only``):
    one question, never two."""
    with TurnServer(heard=[[]]) as server:
        ear, cues, events = ears(server, script=["x"], decoded="che tempo fa")
        calls: list[str] = []
        ear.r.listener.on_not_heard = lambda why: calls.append(why) or True
        ear.rt.worker.on_not_heard = lambda why: calls.append(why) or True
        transcript_timeout(ear, server)
        assert ear.utterances[0].local_only
        assert wait_for(lambda: ear.rt.worker.snapshot().get("turns", 0) == 1, 10.0)
        assert calls == ["no_transcript"]
        assert [k for k, _, _ in cues.reasons] == ["not_heard"]
