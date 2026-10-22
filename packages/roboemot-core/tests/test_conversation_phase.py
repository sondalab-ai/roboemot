"""SPEC 8, 11A, 12: the conversation phase the Live tab shows (idle, listening, thinking with the
route and the elapsed time, speaking). State only: no text, prompt or reply is ever held."""

from __future__ import annotations

import json

from roboemot.conversation import ConversationPhase
from test_conversation_worker import (
    FAIL,
    GATE,
    rigs,  # noqa: F401  (fixture)
    wait_for,
)
from test_voice_listener import Rig as ListenerRig


def clocked(speaking=lambda now: False):
    now = [100.0]
    return ConversationPhase(speaking, clock=lambda: now[0]), now


def test_a_conversation_goes_idle_listening_thinking_speaking_idle():
    spoken = [False]
    phase, now = clocked(lambda t: spoken[0])
    assert phase.snapshot() == {"phase": "idle", "route": None, "elapsed_s": None}
    phase.set_listening(True)
    assert phase.snapshot()["phase"] == "listening"
    phase.set_listening(False)
    phase.begin_thinking()
    assert phase.snapshot()["phase"] == "thinking"
    now[0] += 1.5
    phase.set_route("lan")
    assert phase.snapshot() == {"phase": "thinking", "route": "lan", "elapsed_s": 1.5}
    spoken[0] = True  # the first sentence plays while the stream goes on
    snap = phase.snapshot()
    assert snap["phase"] == "speaking" and snap["route"] == "lan" and snap["elapsed_s"] == 1.5
    phase.end_thinking()
    assert phase.snapshot() == {"phase": "speaking", "route": None, "elapsed_s": None}
    spoken[0] = False
    assert phase.snapshot()["phase"] == "idle"


def test_a_barge_in_goes_speaking_to_listening_and_the_cut_stream_ends():
    spoken = [True]
    phase, _ = clocked(lambda t: spoken[0])
    phase.begin_thinking()
    phase.set_route("lan")
    spoken[0] = False  # the speech was stopped
    phase.set_listening(True)  # the listener resumes on the user's speech
    phase.end_thinking()  # the conversation cancels the stream
    assert phase.snapshot() == {"phase": "listening", "route": None, "elapsed_s": None}


def test_ending_twice_and_a_late_route_change_nothing():
    phase, _ = clocked()
    phase.end_thinking()
    phase.set_route("late")  # no stream in flight: ignored
    phase.begin_thinking()
    phase.end_thinking()
    phase.end_thinking()
    phase.set_route("late")
    assert phase.snapshot()["route"] is None and phase.snapshot()["phase"] == "idle"


def test_a_new_request_forgets_the_previous_route():
    phase, _ = clocked()
    phase.begin_thinking()
    phase.set_route("cloud")
    phase.begin_thinking()
    assert phase.snapshot()["route"] is None


def test_the_phase_holds_no_free_text():
    phase, _ = clocked()
    phase.begin_thinking()
    phase.set_route("lan")
    assert set(phase.snapshot()) == {"phase", "route", "elapsed_s"}


# ------------------------------------------------------------------ the worker drives it


def test_the_worker_is_thinking_with_its_route_while_the_stream_is_in_flight(rigs):  # noqa: F811
    rig = rigs()
    rig.server.scripts["lan.local"] = ["Uno. ", "Due", GATE, "."]
    assert rig.worker.phase.snapshot()["phase"] == "idle"
    n = rig.say("ciao")
    assert wait_for(lambda: rig.worker.phase.snapshot()["route"] == "lan")
    snap = rig.worker.phase.snapshot()
    assert snap["phase"] == "thinking" and snap["elapsed_s"] >= 0.0
    rig.server.gate.set()
    assert rig.done(n)
    assert rig.worker.phase.snapshot() == {"phase": "idle", "route": None, "elapsed_s": None}


def test_the_worker_phase_reports_speaking_from_the_speech_gate(rigs):  # noqa: F811
    rig = rigs()
    spoken = [False]
    rig.worker.phase = ConversationPhase(lambda now: spoken[0])
    rig.server.scripts["lan.local"] = ["Uno. ", "Due. ", "Tre", GATE, "."]
    n = rig.say("ciao")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    spoken[0] = True
    snap = rig.worker.phase.snapshot()
    assert snap["phase"] == "speaking" and snap["route"] == "lan"
    rig.server.gate.set()
    assert rig.done(n)
    spoken[0] = False
    assert rig.worker.phase.snapshot()["phase"] == "idle"


def test_a_barge_in_ends_thinking(rigs):  # noqa: F811
    rig = rigs()
    rig.server.scripts["lan.local"] = ["Uno. ", "Due", GATE, "."]
    n = rig.say("racconta")
    assert wait_for(lambda: rig.worker.phase.snapshot()["phase"] == "thinking")
    assert rig.worker.interrupt()
    assert rig.done(n)
    assert rig.worker.phase.snapshot()["phase"] == "idle"


def test_a_stream_that_breaks_ends_thinking(rigs):  # noqa: F811
    rig = rigs()
    rig.server.scripts["lan.local"] = ["Uno. ", FAIL]
    rig.turn("ciao")
    assert rig.worker.phase.snapshot()["phase"] == "idle"


def test_a_turn_that_raises_ends_thinking(rigs):  # noqa: F811
    rig = rigs()
    rig.worker.hub.sync = lambda cfg: 1 / 0  # fails before the request, in the turn
    assert rig.say("ciao")
    assert wait_for(lambda: rig.worker.snapshot().get("errors") == 1)
    assert rig.worker.phase.snapshot()["phase"] == "idle"


def test_an_unreachable_route_ends_thinking(rigs):  # noqa: F811
    rig = rigs()
    rig.server.status["lan.local"] = 500
    rig.turn("ciao")
    assert rig.worker.phase.snapshot()["phase"] == "idle"


def test_the_snapshot_never_carries_what_was_said(rigs):  # noqa: F811
    rig = rigs()
    rig.server.scripts["lan.local"] = ["Segreto di Pulcinella. ", "Altro", GATE]
    rig.say("parola riservata")
    assert wait_for(lambda: rig.worker.phase.snapshot()["phase"] == "thinking")
    dump = json.dumps(rig.worker.phase.snapshot()) + repr(rig.worker.phase.__dict__)
    assert "riservata" not in dump and "Pulcinella" not in dump
    rig.server.gate.set()


# ------------------------------------------------------------------ the listener reports it


def test_the_listener_reports_an_utterance_in_progress_and_its_end():
    seen: list[bool] = []
    r = ListenerRig(["ciao"], on_activity=seen.append)
    r.say(1.0)
    assert seen == [True, False]


def phase_during_finish(final_text="ciao", fail=False):
    """The Live phase as the recognizer's final runs (2-5 s on the robot), and after it."""
    phase = ConversationPhase()
    r = ListenerRig(["ciao"], on_activity=phase.set_listening)
    during: list[str] = []
    finish = r.asr.finish

    def slow_finish():
        during.append(phase.snapshot()["phase"])
        if fail:
            raise RuntimeError("decoder")
        finish()
        return final_text

    r.asr.finish = slow_finish
    try:
        r.say(1.0)
    except RuntimeError:
        r.listener._abort()  # what the listener's loop does with a chunk that raised
    return during, phase.snapshot()["phase"], r


def test_the_phase_stays_listening_while_the_recognizer_finishes():
    during, after, r = phase_during_finish()
    assert during == ["listening"], "not idle while the final result is computed"
    assert after == "idle" and len(r.utterances) == 1


def test_an_empty_final_still_ends_listening():
    during, after, r = phase_during_finish(final_text="")
    assert during == ["listening"] and after == "idle" and r.utterances == []


def test_a_failing_final_still_ends_listening():
    during, after, _ = phase_during_finish(fail=True)
    assert during == ["listening"] and after == "idle"


def test_the_listener_reports_the_end_when_the_utterance_is_discarded():
    from test_voice_listener import silence, voice

    seen: list[bool] = []
    r = ListenerRig(["ciao"], on_activity=seen.append)
    r.feed(voice(seconds=0.6))
    assert seen == [True]
    r.speech.extend(r.t + 1.0)  # the robot starts to speak: the utterance is dropped
    r.feed(silence(1.0))
    assert seen == [True, False]


def test_a_failing_activity_hook_does_not_stop_the_listener():
    def boom(active: bool) -> None:
        raise RuntimeError("hook")

    r = ListenerRig(["ciao"], on_activity=boom)
    r.say(1.0)
    assert len(r.utterances) == 1
