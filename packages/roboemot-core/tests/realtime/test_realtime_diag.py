"""Decision events of the realtime turn (SPEC 8, 12A; CV-10): the worker and the route against the
fake server publish one ``diag.event`` per decision with the reason code the Live tab explains."""

from __future__ import annotations

import time

import pytest
from roboemot.realtime.turn import SKIP_TOOL
from rt_rig import build_rt, rt_route, wait_for
from test_realtime_cues import NO_VOICE, ears  # noqa: F401 (fixture)
from test_realtime_skip import SKIP
from test_realtime_skip import tooled as skip_tooled
from turn_server import PENDING, TurnServer


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def of(rig, kind: str, reason: str | None = None) -> list[dict]:
    return [e for e in rig.diag if e["kind"] == kind and reason in (None, e["reason"])]


def test_a_refused_answer_shows_the_fallback_cue_with_the_servers_code(rigs):
    """The owner's case: the server refuses the response (function_call_output_pending), the
    session is reset, no local voice: the fallback cue's reason says all of it."""
    with TurnServer(["pending"]) as srv:
        rig = rigs(srv, app=NO_VOICE)
        rig.turn()
        assert rig.cues.played == ["fallback"]
        kind, reason, detail = rig.cues.reasons[-1]
        assert (kind, reason) == ("fallback", "no_voice")
        assert detail["failure"] == "refused" and detail["detail"] == PENDING
        assert detail["reset"] == "pending_calls"  # marked before the run hears the error
        assert wait_for(lambda: of(rig, "session", "reset"))
        reset = of(rig, "session", "reset")[0]
        assert reset["why"] == "pending_calls" and reset["counter"] == "pending_call_resets"
        assert [e["code"] for e in of(rig, "session", "server_error")] == [PENDING]
        assert of(rig, "turn") == []  # the cue explains the turn: no second row


def test_a_route_down_at_start_says_the_route_was_not_ready(rigs):
    with TurnServer() as srv:
        rig = rigs(srv, extra=False, app=NO_VOICE)
        rig.turn()
        kind, reason, detail = rig.cues.reasons[-1]
        assert (kind, reason, detail["failure"]) == ("fallback", "no_voice", "not_ready")
        assert detail["detail"] == rig.route.reason


def test_a_session_lost_mid_turn_is_published_with_its_reason(rigs):
    with TurnServer(["close_before"]) as srv:
        rig = rigs(srv, app=NO_VOICE)
        rig.turn()
        assert rig.cues.reasons[-1][2]["failure"] == "closed"
        assert wait_for(lambda: of(rig, "session", "lost"))


def test_an_answered_turn_publishes_its_outcome_and_voice(rigs):
    with TurnServer() as srv:
        rig = rigs(srv)
        rig.turn()
        assert wait_for(lambda: of(rig, "turn"))
        assert of(rig, "turn")[-1] | {"t": 0} == {
            "kind": "turn", "reason": "ok", "voice": "realtime", "failure": None, "detail": None,
            "t": 0,
        }  # fmt: skip


def test_a_reconnect_after_a_reset_is_published(rigs):
    with TurnServer(["pending", "reply"]) as srv:
        rig = rigs(srv)
        rig.turn()
        assert wait_for(lambda: rig.route.snapshot()["pending_call_resets"] == 1)
        rig.turn("ancora", stamp=time.monotonic() + 1.0)
        assert wait_for(lambda: of(rig, "session", "reconnected"))
        assert of(rig, "session", "reconnected")[0]["after"] == "pending_calls"


def test_a_skip_is_published_with_the_conversation_state(rigs):
    with TurnServer(["tool"], tool_name=SKIP_TOOL) as srv:
        rig = rigs(srv, rt=SKIP)
        rig.turn()
        skipped = of(rig, "unanswered", "skipped")
        assert len(skipped) == 1 and skipped[0]["conversation"] == "none"
        assert of(rig, "turn") == []


def test_a_refused_skip_is_published(rigs):
    names = ["play", SKIP_TOOL]
    with TurnServer(["tools", "reply"], tool_names=names, tool_args='{"gesture": "joy"}') as srv:
        rig, _ = skip_tooled(rigs, srv)
        rig.turn()
        assert len(of(rig, "turn", "skip_refused")) == 1
        assert of(rig, "unanswered") == []


def test_no_event_of_a_turn_carries_its_words_or_the_key(rigs):
    """Scans every event the worker and the route publish over a refused turn, a lost
    session and an answered one (server strings: error codes, close reasons)."""
    words = "Zanzibar quokka telefonata segretissima"
    key = "sk-Live-Secret-Key"
    for script in (["pending"], ["close_before"], ["reply"]):
        with TurnServer(script) as srv:
            rig = rigs(srv, route=rt_route(srv, api_key=key), app=NO_VOICE)
            rig.turn(words)
            assert wait_for(lambda r=rig: of(r, "turn") or of(r, "cue") or r.cues.reasons)
            for ev in rig.diag:
                for name, value in ev.items():
                    assert isinstance(value, bool | int | float | str | type(None)), name
                    if isinstance(value, str):
                        assert " " not in value and value.lower() not in words.lower().split()
                        assert key.lower() not in value.lower()


def test_a_transcript_timeout_is_published_before_the_not_heard_cue(ears):  # noqa: F811
    with TurnServer(heard=[[]]) as server:
        ear, _ = ears(server, script=["x"], decoded="che tempo fa")
        events = []
        ear.r.listener.events = lambda kind, reason, **f: events.append((kind, reason, f))
        ear.say()
        assert ear.settle(lambda: len(server.of("input_audio_buffer.commit")) == 1)
        timeout = ear.rt.stack.effective().voice.realtime_route.transcript_timeout_s
        ear.r.now += timeout
        assert ear.settle(lambda: len(ear.utterances) == 1)
        assert ("audio", "transcript_timeout", {"timeout_s": timeout}) in events
