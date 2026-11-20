"""Over the robot's own radio a voice from a face stands in for the wake word, and the server's
transcript decides (SPEC 8, Radio playback and Realtime speech route; owner decision
2026-11-20, ``tools.radio.face_addresses``). The robot (2026-11-20): over music Vosk wrote
"viene" and "ragazzi dopo musica" for "hey neve spegni la musica", so the local wake gate
rejected it and the server's stop rule never ran.

With the rule on, a realtime audio route and the radio audible as an utterance starts, the
local recognizer, its grammar pass and any local decode never run (CPU): the voice-at-face
check alone decides. Passed: its audio goes as a wake the server must confirm; failed:
``rejected_media`` and nothing leaves. Nothing is answered and no tool runs before the
server's words confirm it (Gemini Live's end to end is in ``test_gemini_route.py``)."""

# ruff: noqa: F811  (the ``ears`` fixture is imported and then taken by name)
from __future__ import annotations

import logging
import threading

import pytest
from test_realtime_listener import (  # noqa: F401 (fixture)
    FACE_RAD,
    OFF_FACE_RAD,
    appended,
    ears,
)
from test_realtime_server_turns import OPT_INS, SERVER
from turn_server import TurnServer

NEVE = {"wake_word_policy": "never", "wake_word": "Neve", "facing": {"enabled": True}}
LOOKING = 0.0  # the face's head yaw: it looks at the robot
LYRIC = "ragazzi dopo musica"  # Vosk's words over the radio (never asked with the rule on)


def app(language="it", **radio):
    cfg = {"voice": dict(NEVE), "tools": {"radio": radio}}
    if language != "it":
        cfg["conversation"] = {"language": language}
    return cfg


def on_radio(ear, *, station=True):
    """The robot's radio plays a station and is audible; returns the player calls the listener
    made: ``stop`` reasons and ``hold`` stamps."""
    calls: dict[str, list] = {"stop": [], "hold": []}
    ear.r.listener.direction_over_media = True  # a hardware echo canceller (the host's wiring)
    ear.r.listener.media_audible = lambda at: True
    ear.r.listener.media_set = lambda: station
    ear.r.listener.media_stop = calls["stop"].append
    ear.r.listener.media_hold = calls["hold"].append
    return calls


def events(ear):
    seen: list[tuple[str, str, dict]] = []
    ear.r.listener.events = lambda kind, reason, **fields: seen.append((kind, reason, fields))
    return seen


def spy_grammar(ear):
    """Count the grammar passes the fast path is asked for."""
    asked: list[str] = []
    fast = ear.r.listener._fast
    real = fast.grammar_pass

    def spy(text, *args, **kw):
        asked.append(text)
        return real(text, *args, **kw)

    fast.grammar_pass = spy
    return asked


def radio_ear(make, server, *, bearing=FACE_RAD, yaw=LOOKING, cfg=None, **kw):
    return make(server, script=[LYRIC], bearing=bearing, yaw=yaw, app=cfg or app(), **kw)


def local_checks(ear, grammar):
    """What the local speech check did: recognizer starts, audio it took, grammar passes and
    fallback decodes."""
    asr = ear.r.asr
    return len(asr.started), asr.accepted, len(grammar), asr.decodes


# ------------------------------------------------------------------ the voice-at-face check


@pytest.mark.parametrize(
    ("language", "heard"),
    [
        ("it", "Ehi Neve, spegni la musica."),
        ("it", "spegni la musica"),  # no name: stop trusts the command (PR 283)
        ("en", "Stop the music."),
    ],
)
def test_a_stop_the_music_from_a_face_over_the_radio_stops_it(ears, language, heard):  # noqa: F811
    """The robot's case, Italian and English: the server's words hold a stop and a media word
    of the robot's language, so the station stops and nothing else is answered."""
    with TurnServer(heard=[[heard]]) as server:
        ear = radio_ear(ears, server, cfg=app(language), to_worker=True)
        ear.r.listener._speaker_language = lambda: language
        calls = on_radio(ear)
        seen = events(ear)
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        named = "Neve" in heard
        assert u.addressed is named  # the name confirms: answered (its own radio_stop tool)
        assert u.language == language and u.realtime is not None
        assert calls["stop"] == ([] if named else ["voice_stop"])
        assert len(calls["hold"]) == 1  # its turn held the radio as a wake does
        assert ("media", "face_wake", {}) in seen
        assert local_checks(ear, grammar) == (0, 0, 0, 0)
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["face_wake_media"] == 1 and audio["vosk_skipped"] == 1
        assert audio["voice_stop_media"] == (0 if named else 1)
        decisions = ear.r.listener.snapshot()["wake"]["decisions"]
        assert decisions["face_media"] == 1 and sum(decisions.values()) == 1
        assert len(server.of("input_audio_buffer.commit")) == 1
        if named:
            assert ear.settle(lambda: len(server.of("response.create")) == 1, 10.0)
        else:
            assert ear.settle(lambda: len(server.of("conversation.item.delete")) == 1)
            assert server.of("response.create") == []


@pytest.mark.parametrize("heard", ["Neve, che tempo fa?", "Nivea che tempo fa"])
def test_the_name_in_the_servers_words_is_answered(ears, heard):  # noqa: F811
    with TurnServer(heard=[[heard]]) as server:
        ear = radio_ear(ears, server, to_worker=True)
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].addressed
        assert ear.r.listener.snapshot()["realtime_audio"]["wake_unconfirmed_media"] == 0
        assert ear.settle(lambda: len(server.of("response.create")) == 1, 10.0)


@pytest.mark.parametrize("heard", ["Innamorata musica", "e la nave va", "il robot balla"])
def test_a_face_and_no_name_or_stop_is_dropped_unanswered(ears, heard, caplog):  # noqa: F811
    """Someone in front of the robot sings along: no name, no stop in the server's words. It is
    dropped as an unconfirmed wake: no response, its items deleted, the window closed; no
    words in any log."""
    caplog.set_level(logging.DEBUG)
    with TurnServer(heard=[[heard]]) as server:
        ear = radio_ear(ears, server, to_worker=True)
        calls = on_radio(ear)
        seen = events(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed and calls["stop"] == []
        assert ("unanswered", "media_unconfirmed", {"heard": None}) in seen
        assert ear.r.listener.snapshot()["realtime_audio"]["wake_unconfirmed_media"] == 1
        assert not ear.r.listener._gate.addressable(enrolled_in_view=False, start=ear.r.t)
        assert ear.settle(lambda: len(server.of("conversation.item.delete")) == 1)
        assert server.of("response.create") == []
        assert ear.rt.sink.pushed == [] and ear.rt.history() == []
    assert heard not in caplog.text


def test_a_lyric_from_no_face_is_rejected_media_and_nothing_leaves(ears):  # noqa: F811
    """The robot's own speaker is no face: the check fails on real direction readings, the
    outcome stays ``rejected_media`` and no audio or word leaves the robot."""
    with TurnServer() as server:
        ear = radio_ear(ears, server, bearing=OFF_FACE_RAD)
        calls = on_radio(ear)
        seen = events(ear)
        grammar = spy_grammar(ear)
        ear.say()
        ear.settle(lambda: False, 0.3)
        assert ear.utterances == [] and calls == {"stop": [], "hold": []}
        snap = ear.r.listener.snapshot()
        assert snap["direction"]["checks"]["rejected"] == 1  # the real check ran on readings
        assert snap["wake"]["decisions"]["rejected_media"] == 1
        assert local_checks(ear, grammar) == (0, 0, 0, 0)
        assert [(k, r) for k, r, _ in seen] == [("unanswered", "media")]
        assert seen[0][2]["heard"] is None  # no local words
        assert appended(server) == []


def test_a_face_looking_away_is_rejected_media(ears):  # noqa: F811
    with TurnServer() as server:
        ear = radio_ear(ears, server, yaw=0.7)  # 40 deg away
        on_radio(ear)
        ear.say()
        ear.settle(lambda: False, 0.3)
        snap = ear.r.listener.snapshot()
        assert snap["wake"]["decisions"]["rejected_media"] == 1
        assert appended(server) == []


def test_a_face_wake_whose_transcript_never_comes_is_dropped_never_decoded(ears):  # noqa: F811
    with TurnServer(heard=[[]]) as server:
        ear = radio_ear(ears, server, to_worker=True)
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        assert ear.settle(lambda: not ear.r.listener.awaiting_transcript())
        assert ear.utterances == [] and ear.r.asr.decodes == 0
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["face_wake_unheard"] == 1 and audio["local_fallbacks"] == 0
        assert server.of("response.create") == []
        assert not ear.r.listener._gate.addressable(enrolled_in_view=False, start=ear.r.t)


# ------------------------------------------------------------------ today's path kept


def test_the_switch_off_keeps_the_local_check(ears):  # noqa: F811
    """``tools.radio.face_addresses`` off: Vosk and its grammar pass hear it, the wake word
    is required, and a face does not count (today's ``rejected_media``)."""
    with TurnServer() as server:
        ear = radio_ear(ears, server, cfg=app(face_addresses=False))
        on_radio(ear)
        seen = events(ear)
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        started, accepted, passes, _ = local_checks(ear, grammar)
        assert started == 1 and accepted > 0 and passes == 1
        assert ear.r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1
        assert ("media", "face_wake", {}) not in [(k, r, f) for k, r, f in seen]
        assert appended(server) == []


def test_the_switch_off_keeps_the_name_waking_the_robot(ears):  # noqa: F811
    with TurnServer(heard=[["Neve spegni la musica"]]) as server:
        ear = ears(
            server, script=["neve spegni la musica"], bearing=OFF_FACE_RAD, yaw=0.7,
            app=app(face_addresses=False), to_worker=True,
        )  # fmt: skip
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].addressed and ear.utterances[0].wake_word


def test_without_a_realtime_audio_route_the_local_check_decides(ears):  # noqa: F811
    """A chat route with local ASR only (``input: text``): no server transcript can confirm
    a face, so the wake word is required as today and nothing is skipped."""
    with TurnServer() as server:
        ear = ears(
            server, rt={"input": "text"}, script=[LYRIC], bearing=FACE_RAD, yaw=LOOKING,
            app=app(),
        )  # fmt: skip
        on_radio(ear)
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        assert local_checks(ear, grammar)[0] == 1 and len(grammar) == 1
        assert ear.r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1
        assert appended(server) == []


def test_server_turns_keep_the_local_check_and_send_nothing(ears):  # noqa: F811
    """``turns: server``: the ``media`` gate keeps the segment local, and no burst can go (the
    server would answer it on its own before any confirmation), so the face rule does not
    apply: Vosk decides, the wake word is required, nothing leaves."""
    with TurnServer(stream_vad=True) as server:
        ear = ears(
            server, rt=SERVER, script=[LYRIC], bearing=FACE_RAD, yaw=LOOKING,
            app={**app(), **OPT_INS},
        )  # fmt: skip
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        snap = ear.r.listener.snapshot()
        assert snap["realtime_audio"]["stream_gated_media"] == 1
        assert snap["realtime_audio"]["face_wake_media"] == 0
        assert snap["wake"]["decisions"]["rejected_media"] == 1
        assert appended(server) == [] and server.of("response.create") == []


def policy_app(policy):
    cfg = app()
    cfg["voice"] = {**cfg["voice"], "wake_word_policy": policy}
    return cfg


@pytest.mark.parametrize(
    ("policy", "enrolled"),
    [("always", False), ("always", True), ("unless_enrolled_in_view", False)],
)
def test_a_policy_that_requires_the_wake_word_keeps_the_local_check(  # noqa: F811
    ears, policy, enrolled
):
    """Security review 2026-11-20: the face rule never sends audio the policy would not send.
    Where the policy requires the wake word, a face over the radio is ``rejected_media``,
    nothing leaves, and Vosk with its grammar pass runs as today."""
    with TurnServer() as server:
        ear = ears(server, script=[LYRIC], bearing=FACE_RAD, yaw=LOOKING, app=policy_app(policy))
        ear.r.listener._enrolled_in_view = lambda now: enrolled
        on_radio(ear)
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        started, accepted, passes, _ = local_checks(ear, grammar)
        assert started == 1 and accepted > 0 and passes == 1
        snap = ear.r.listener.snapshot()
        assert snap["wake"]["decisions"]["rejected_media"] == 1
        assert snap["realtime_audio"]["face_wake_media"] == 0
        assert appended(server) == []


@pytest.mark.parametrize("policy", ["always", "unless_enrolled_in_view"])
def test_the_name_still_wakes_the_robot_where_the_policy_requires_it(ears, policy):  # noqa: F811
    with TurnServer(heard=[["Neve spegni la musica"]]) as server:
        ear = ears(
            server, script=["neve spegni la musica"], bearing=OFF_FACE_RAD, yaw=0.7,
            app=policy_app(policy), to_worker=True,
        )  # fmt: skip
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].addressed and ear.utterances[0].wake_word


def test_unless_enrolled_in_view_with_an_enrolled_person_in_view_takes_the_face(  # noqa: F811
    ears,
):
    with TurnServer(heard=[["Spegni la musica"]]) as server:
        ear = ears(
            server, script=[LYRIC], bearing=FACE_RAD, yaw=LOOKING,
            app=policy_app("unless_enrolled_in_view"), to_worker=True,
        )  # fmt: skip
        ear.r.listener._enrolled_in_view = lambda now: True
        calls = on_radio(ear)
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert calls["stop"] == ["voice_stop"]
        assert local_checks(ear, grammar) == (0, 0, 0, 0)
        assert ear.r.listener.snapshot()["realtime_audio"]["face_wake_media"] == 1


def test_without_the_radio_nothing_changes(ears):  # noqa: F811
    with TurnServer(heard=[["che bel video"]]) as server:
        ear = radio_ear(ears, server, to_worker=True)
        ear.r.listener.media_audible = lambda at: False
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.utterances[0].addressed  # policy never, a face: answered without the name
        assert ear.r.listener.snapshot()["realtime_audio"]["face_wake_media"] == 0


# ------------------------------------------------ a line riding a radio wake (policy never)


def held_by_holds(ear):
    """The radio is audible except within ``resume_after_s`` after a recorded hold."""
    holds: list[float] = []
    resume = ear.rt.stack.effective().tools.media.resume_after_s
    ear.r.listener.direction_over_media = True
    ear.r.listener.media_audible = lambda at: not any(h <= at <= h + resume for h in holds)
    ear.r.listener.media_hold = holds.append
    return holds


@pytest.mark.parametrize(
    ("second", "answered"),
    [("e la nave va", False), ("Neve, che tempo fa?", True)],
)
def test_a_line_riding_an_unconfirmed_face_wake_under_never_is_not_answered(
    ears,  # noqa: F811
    second,
    answered,
):
    """The robot's policy is ``never``. A face wake holds the radio from its last speech, so the
    next line starts in the hold and gets the normal rules: ``not_required`` with a face, not a
    window follow-up. It still rides the wake whose verdict is due: when the first line is not
    confirmed, the second is dropped too unless its own words hold the name."""
    release = threading.Event()  # the server's transcripts wait until both were said
    with TurnServer(heard=[["Innamorata musica"], [second]], hold=release) as server:
        ear = radio_ear(ears, server, to_worker=True)
        holds = held_by_holds(ear)
        ear.say()
        ear.say()  # inside the first's hold: the normal rules
        assert len(holds) == 1 and ear.utterances == []
        decisions = ear.r.listener.snapshot()["wake"]["decisions"]
        assert decisions["face_media"] == 1 and decisions["not_required"] == 1
        release.set()
        assert ear.settle(lambda: len(ear.utterances) == 2)
        assert not ear.utterances[0].addressed
        assert ear.utterances[1].addressed is answered
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["wake_unconfirmed_media"] == (1 if answered else 2)
        want = 1 if answered else 0
        assert ear.settle(lambda: len(server.of("response.create")) == want, 10.0)
        assert len(server.of("response.create")) == want


def test_a_line_riding_an_unconfirmed_local_wake_under_never_is_not_answered(ears):  # noqa: F811
    """The same with the switch off: the name Vosk matched in a lyric opened the wake (the
    pre-existing path), and the robot's ``never`` policy answered the next line from a face."""
    release = threading.Event()
    with TurnServer(heard=[["Innamorata"], ["musica"]], hold=release) as server:
        ear = ears(
            server, script=["neve"], bearing=FACE_RAD, yaw=LOOKING,
            app=app(face_addresses=False), to_worker=True,
        )  # fmt: skip
        holds = held_by_holds(ear)
        ear.say()
        ear.say()
        assert len(holds) == 1 and ear.utterances == []
        release.set()
        assert ear.settle(lambda: len(ear.utterances) == 2)
        assert not ear.utterances[0].addressed and not ear.utterances[1].addressed
        assert ear.r.listener.snapshot()["realtime_audio"]["wake_unconfirmed_media"] == 2
        assert server.of("response.create") == []


# ------------------------------------- only a real direction reading over the radio counts (B)


def test_no_direction_over_the_radio_keeps_vosk_and_the_name(ears):
    """No hardware echo canceller (``direction_over_media`` off): the face rule cannot pass, so
    Vosk, its grammar pass and the wake word decide as before; a lyric is ``rejected_media``."""
    with TurnServer() as server:
        ear = radio_ear(ears, server)
        on_radio(ear)
        ear.r.listener.direction_over_media = False
        grammar = spy_grammar(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert not ear.utterances[0].addressed
        started, accepted, passes, _ = local_checks(ear, grammar)
        assert started == 1 and accepted > 0 and passes == 1
        assert ear.r.listener.snapshot()["wake"]["decisions"]["rejected_media"] == 1
        assert appended(server) == []


@pytest.mark.parametrize(
    ("decoded", "addressed"), [(LYRIC, False), ("neve spegni la musica", True)]
)
def test_a_degraded_face_check_over_the_radio_decodes_for_the_name(ears, decoded, addressed):
    """No reading reached the check (``degraded_rule: engaged``, attention on a face): a
    degraded pass is never trusted over the radio. The buffer is decoded for the wake word: a
    lyric is ``rejected_media`` with nothing sent, the name still wakes the robot."""
    cfg = app()
    cfg["voice"] = {**cfg["voice"], "direction": {"degraded_rule": "engaged"}}
    heard = [["Neve spegni la musica"]] if addressed else None
    with TurnServer(heard=heard) as server:
        ear = ears(server, script=[decoded], bearing=FACE_RAD, yaw=LOOKING, app=cfg)
        ear.bearing = None  # no sound direction over the radio
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        snap = ear.r.listener.snapshot()
        assert ear.r.asr.decodes == 1 and snap["realtime_audio"]["face_wake_media"] == 0
        assert snap["direction"]["checks"]["degraded_passed"] == 1
        assert ear.utterances[0].addressed is addressed
        if addressed:
            assert ear.utterances[0].wake_word
        else:
            assert snap["wake"]["decisions"]["rejected_media"] == 1
            assert appended(server) == []


# ------------------------------------------------------------------ refused, late, followed


def test_audio_that_can_no_longer_go_at_the_end_is_decoded_not_held(ears):
    """The session dropped (or the budget ran out, the robot fell asleep) between the start and
    the end: no radio hold, no face wake, no Live event; the buffer is decoded as today."""
    with TurnServer() as server:
        ear = radio_ear(ears, server)
        calls = on_radio(ear)
        seen = events(ear)
        answers = iter([True])
        ear.input.wants = lambda cfg: next(answers, False)
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ear.r.asr.started == [] and ear.r.asr.decodes == 1
        assert calls["hold"] == [] and ("media", "face_wake", {}) not in seen
        assert ear.r.listener.snapshot()["realtime_audio"]["face_wake_media"] == 0
        assert appended(server) == []


@pytest.mark.parametrize(
    ("decoded", "addressed"),
    [("Neve spegni la musica", True), (LYRIC, False)],
    ids=["name", "lyric"],
)
def test_a_face_wake_refused_before_it_left_is_decoded_for_the_wake_word(ears, decoded, addressed):
    """Owner decision 2026-11-20 (no utterance is lost): a face wake whose audio was refused
    before it left (its session never came, asleep, the budget) is decoded from its buffer, and
    over the radio only the name addresses it then; its audio is not offered a second time."""
    from roboemot.realtime.input import InputJob

    with TurnServer() as server:
        ear = radio_ear(ears, server, decoded=decoded)
        on_radio(ear)
        seen = events(ear)
        cues: list[str] = []
        ear.r.listener._on_audio = lambda event, key: cues.append(event)
        asked: list[str] = []  # SPEC 8, Robot expression: a face wake is never asked about
        ear.r.listener.on_not_heard = lambda why: asked.append(why) or True

        def refused(samples, rate_hz, **kw):
            job = InputJob(samples, rate_hz)
            job._end(refused="no_session")
            return job

        ear.input.submit = refused
        ear.say()
        assert ear.settle(lambda: len(ear.utterances) == 1)
        u = ear.utterances[0]
        assert u.text == decoded and u.addressed is addressed and u.realtime is None
        # its audio is never offered again; the person looked at the robot, so a decode with
        # no name ends with the did-not-get-it cue (review of PR 291)
        assert cues == ["sent", "unsent" if addressed else "not_heard"]
        assert asked == []
        assert ear.r.asr.decodes == 1
        audio = ear.r.listener.snapshot()["realtime_audio"]
        assert audio["face_wake_unsent"] == 1 and audio["face_wake_unheard"] == 0
        if not addressed:  # over the radio without the name: the local media rule
            assert ("unanswered", "media") in [(k, r) for k, r, _ in seen]
        assert appended(server) == []


def test_a_follow_up_riding_a_confirmed_face_wake_is_answered(ears):
    release = threading.Event()
    heard = [["Neve, che tempo fa?"], ["e domani?"]]
    with TurnServer(heard=heard, hold=release) as server:
        ear = radio_ear(ears, server, to_worker=True)
        held_by_holds(ear)
        ear.say()
        ear.say()  # inside the face wake's hold: not_required, rides it
        release.set()
        assert ear.settle(lambda: len(ear.utterances) == 2)
        assert ear.utterances[0].addressed and ear.utterances[1].addressed
        assert ear.r.listener.snapshot()["realtime_audio"]["wake_unconfirmed_media"] == 0


def test_a_transcript_after_the_deadline_is_deleted_and_never_answered(ears):
    """``openai_realtime``: the face wake's transcript comes after the listener gave up on it
    (dropped, ``face_wake_unheard``): its items are deleted and nothing asks for a response."""
    release = threading.Event()
    with TurnServer(heard=[["Neve, che tempo fa?"]], hold=release) as server:
        ear = radio_ear(ears, server, to_worker=True)
        on_radio(ear)
        ear.say()
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        ear.r.now += ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        assert ear.settle(lambda: not ear.r.listener.awaiting_transcript())
        assert ear.r.listener.snapshot()["realtime_audio"]["face_wake_unheard"] == 1
        release.set()
        assert ear.settle(lambda: len(server.of("conversation.item.delete")) >= 1, 10.0)
        ear.settle(lambda: False, 0.5)
        assert ear.utterances == [] and server.of("response.create") == []
