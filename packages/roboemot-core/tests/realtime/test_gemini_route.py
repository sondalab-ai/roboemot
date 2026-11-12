"""A ``gemini_live`` route end to end (SPEC 8, Gemini Live wire; SPEC 12): the real route,
client, codec, worker and speech output against the scripted Gemini server, over TLS."""

from __future__ import annotations

import time

import pytest
from gemini_server import HEARD, REPLY, GeminiServer
from roboemot.realtime import protocol
from roboemot.realtime.route import _Token
from rt_rig import build_rt, wait_for
from test_conversation_tools import Env
from test_realtime_listener import FACE_RAD, Ear

OPT_IN = {"privacy": {"realtime_audio": True}, "voice": {"wake_word_policy": "never"}}
AUDIO = {"input": "audio"}


@pytest.fixture()
def rigs():
    made = []

    def make(srv, *, rt=None, app=None, **route):
        rig = build_rt(srv, route=gemini_route(srv, **route), rt=rt, app=app)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def gemini_route(srv: GeminiServer, **kw) -> dict:
    return {
        "name": "rt",
        "base_url": srv.base_url(),
        "model": "gemini-3.8-live",
        "kind": "realtime",
        "protocol": "gemini_live",
        "voice": "Kore",
        "locality": "local",
        "api_key": "none",
        "first_token_timeout_s": 3,
        "stall_timeout_s": 3,
        **kw,
    }


def text_turns(srv: GeminiServer) -> list[dict]:
    return [m["clientContent"] for m in srv.of("clientContent")]


def said(rig) -> list[str]:
    return [m["content"] for t in rig.history() for m in t if m.get("role") == "assistant"]


def heard_words(rig) -> list[str]:
    return [m["content"] for t in rig.history() for m in t if m.get("role") == "user"]


# ------------------------------------------------------------------ connect


def test_the_connect_sends_setup_first_and_waits_for_setup_complete(gemini, rigs):
    srv = gemini()
    rig = rigs(srv)
    assert rig.route.ready()
    assert srv.requests[0]["path"] == "/ws/BidiGenerateContent"  # base_url as written
    assert len(srv.setups) == 1 and list(srv.received[0]) == ["setup"]
    setup = srv.setups[0]
    assert setup["model"] == "models/gemini-3.8-live"
    voice = setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice == {"voiceName": "Kore"}
    assert setup["realtimeInputConfig"] == {"automaticActivityDetection": {"disabled": True}}
    snap = rig.route.snapshot()
    assert snap["state"] == "connected" and rig.route.output_rate_hz == 24000
    assert snap["wire"]["resumption_updates"] == 1  # the codec's counts, one key
    assert "wire" not in snap["client"]


# ------------------------------------------------------------------ turns


def test_a_text_turn_is_one_client_content_turn_and_its_answer_plays(gemini, rigs):
    srv = gemini([["audio"]])
    rig = rigs(srv)
    rig.turn("ciao Neve")
    assert wait_for(lambda: rig.events and rig.events[-1]["outcome"] == "ok")
    turns = text_turns(srv)
    assert turns[-1]["turnComplete"] is True
    words = [p["text"] for t in turns[-1]["turns"] for p in t["parts"]]
    assert "ciao Neve" in words
    assert rig.sink.pushed  # the reply played
    assert said(rig) == [REPLY]
    snap = rig.route.snapshot()
    assert snap["foreign_events"] == 0 and snap["client"]["stalls"] == 0
    assert snap["responses"] == {"completed": 1}


def listen(rig, script):
    return Ear(rig, script=script, bearing=FACE_RAD, to_worker=True)


def test_a_robot_audio_turn_notes_the_tone_and_plays_the_held_answer(gemini, rigs):
    srv = gemini([["call:note_tone", "audio"]], call_args={"note_tone": {"tone": "happy"}})
    rig = rigs(srv, rt={**AUDIO, "tone_tool": True}, app=OPT_IN)
    ear = listen(rig, ["x"])
    try:
        ear.say()
        assert ear.settle(lambda: len(rig.events) == 1, 10.0)
        assert rig.events[-1]["outcome"] == "ok"
        declared = srv.setups[0]["tools"][0]["functionDeclarations"]
        assert "note_tone" in [d["name"] for d in declared]  # the setup holds the turn's tools
        assert all("parametersJsonSchema" in d for d in declared)
        assert srv.tool_responses[0] == [
            {"id": "fc_0_note_tone", "name": "note_tone", "response": {"ok": True}}
        ]
        assert rig.emitted == [("speech.tone.happy", {"language": "it"})]
        assert rig.sink.pushed
        assert heard_words(rig) == [HEARD] and said(rig) == [REPLY]
        assert not [t for t in text_turns(srv) if t.get("turnComplete")]  # never a text turn
        assert rig.route.snapshot()["foreign_events"] == 0
    finally:
        ear.close()


def test_a_turn_not_addressed_to_the_robot_is_skipped_and_its_call_answered(gemini, rigs):
    srv = gemini([["call:ignore_utterance"]], heard=["Marco, mi passi il sale?"])
    rig = rigs(srv, rt={**AUDIO, "skip_tool": True}, app=OPT_IN)
    ear = listen(rig, ["x"])
    try:
        ear.say()
        assert ear.settle(lambda: len(rig.events) == 1, 10.0)
        assert rig.events[-1]["outcome"] == "skipped"
        assert ear.settle(lambda: len(srv.tool_responses) == 1, 5.0)
        assert srv.tool_responses[0][0]["name"] == "ignore_utterance"
        assert rig.sink.pushed == [] and rig.history() == []
        snap = rig.route.snapshot()
        assert snap["skipped"] == 1 and snap["items_delete_emulated"] >= 1
        assert snap["items_delete_confirmed"] == 0
    finally:
        ear.close()


def test_a_tool_round_answers_the_call_and_gemini_continues(gemini, rigs):
    srv = gemini([["call:play", "audio"]], call_args={"play": {"gesture": "joy"}})
    rig = rigs(srv)
    env = Env()
    rig.worker.skills = env.registry
    rig.worker._speaker = lambda: "p1"
    rig.turn("fai una faccia felice")
    assert env.played == ["joy"]
    assert [r["name"] for r in srv.tool_responses[0]] == ["play"]
    assert srv.tool_responses[0][0]["id"] == "fc_0_play"
    assert rig.sink.pushed and said(rig)[-1].endswith(REPLY)
    snap = rig.route.snapshot()
    assert snap["tool_calls"] == 1 and snap["responses"] == {"completed": 2}
    assert snap["foreign_events"] == 0


def barge_in(rig):
    rig.worker.mark_interrupting()
    rig.speech.stop()
    return rig.worker.interrupt()


def test_a_barge_in_cancels_locally_and_drops_the_rest_of_the_answer(gemini, rigs):
    srv = gemini([["audio"]], chunks=40, chunk_ms=100, gap_s=0.02)  # faster than it plays
    rig = rigs(srv)
    n = rig.say("raccontami una storia")
    assert wait_for(lambda: rig.sink.pushed and rig.speech.stream_progress(1) is not None)
    time.sleep(0.3)
    assert barge_in(rig)
    assert wait_for(lambda: rig.worker.snapshot().get("turns", 0) >= n)
    played = len(rig.sink.pushed)
    assert wait_for(lambda: rig.route.snapshot()["wire"]["dropped_after_cancel"] > 0)
    assert len(rig.sink.pushed) == played  # nothing of it plays after the cut
    assert rig.history()[0][-1]["interrupted"] is True
    snap = rig.route.snapshot()
    assert snap["items_truncate_emulated"] == 1 and snap["responses"] == {"cancelled": 1}


def test_a_declined_utterance_discards_the_held_answer(gemini, rigs):
    srv = gemini([["audio"]], heard=["stop"])
    rig = rigs(srv, rt=AUDIO, app=OPT_IN)
    ear = listen(rig, ["stop"])
    try:
        rig.worker.interrupt()  # a barge-in just cut the robot off: "stop" is consumed
        ear.say()
        assert ear.settle(lambda: rig.route.snapshot()["items_delete_emulated"] >= 1, 10.0)
        ear.settle(lambda: False, 0.5)
        assert rig.sink.pushed == []  # the answer Gemini gave on its own never played
        assert rig.route.snapshot()["wire"]["held_discarded"] == 1
        assert not [t for t in text_turns(srv) if t.get("turnComplete")]
        rig.turn("ciao")  # the next turn gets its own answer, not the discarded one
        assert rig.events[-1]["outcome"] == "ok" and rig.sink.pushed
    finally:
        ear.close()


def test_a_go_away_reconnects_once_idle_and_the_setup_replays_the_history(gemini, rigs):
    srv = gemini([["audio", "go_away"], ["audio"]])
    rig = rigs(srv)
    rig.turn("mi chiamo Marcello")
    assert wait_for(lambda: len(srv.setups) == 2 and rig.route.connected, 10.0)
    snap = rig.route.snapshot()
    assert snap["go_aways"] == 1 and snap["go_away_reconnects"] == 1
    assert snap["reconnects"] == 1 and rig.route.generation == 2
    first = srv.setups[0]["systemInstruction"]["parts"][0]["text"]
    replay = srv.setups[1]["systemInstruction"]["parts"][0]["text"]
    assert "mi chiamo Marcello" not in first
    assert "mi chiamo Marcello" in replay and REPLY in replay  # the new session lacks them
    rig.turn("come mi chiamo?")
    assert rig.events[-1]["outcome"] == "ok" and srv.connections == 2
    assert rig.route.snapshot()["failures"] == {}


def test_an_idle_close_is_no_loss_and_the_next_turn_answers_on_the_route(gemini, rigs):
    srv = gemini([["audio"]], idle_close_s=0.8)
    rig = rigs(srv)
    assert rig.route.ready()
    assert wait_for(lambda: rig.route.snapshot()["idle_closes"] >= 2 and rig.route.connected, 10.0)
    snap = rig.route.snapshot()
    assert snap["idle_closes"] >= 2 and snap["failures"] == {}  # never a loss
    assert rig.route._resets_in_row == 0 and rig.route._reopen_at == 0.0  # no backoff step
    assert rig.network.realtime.online  # the link never went down
    srv.idle_close_s = None
    assert wait_for(lambda: rig.route.connected, 5.0)
    rig.turn("ciao")
    assert rig.events[-1]["outcome"] == "ok" and rig.chat_hosts() == []


def test_a_1008_on_connect_is_retried_not_an_account_error(gemini, rigs, caplog):
    srv = gemini(refuse=1)
    rig = rigs(srv)
    assert not rig.route.ready()
    assert rig.route.state == "fallback_local" and rig.route.reason == "connect_failed"
    assert wait_for(lambda: rig.route.connected, 10.0)  # the background reconnect
    assert srv.closes == 1 and len(srv.setups) == 2
    assert "close code 1008" in caplog.text and "permission" not in caplog.text


def test_late_usage_corrects_the_estimate_once(gemini, rigs):
    srv = gemini([["audio"]], usage=5000, turn_complete_s=0.8)
    rig = rigs(srv)
    rig.turn("ciao")
    estimate = rig.hub.router.used("rt")
    assert 0 < estimate < 5000  # charged at the response's end
    rig.turn("ancora")  # the first turn's usage comes while this one runs
    assert wait_for(lambda: rig.route.snapshot()["usage_reports"] == 2, 5.0)
    assert rig.hub.router.used("rt") == 10_000


def test_a_usage_report_of_a_gone_session_is_charged_once(gemini, rigs):
    srv = gemini([["audio"]], turn_complete_s=30.0)  # its turnComplete never comes in time
    rig = rigs(srv)
    rig.turn("ciao")
    estimate = rig.hub.router.used("rt")
    rid = next(iter(rig.route._estimated))
    rig.route.close_soon("conversation_end")
    report = protocol.UsageReport(protocol.USAGE_REPORT, (rid,), estimate + 1000)
    rig.route._on_event(report, _Token())  # a client that is gone: the tokens were spent
    rig.route._on_event(report, _Token())  # a repeat charges nothing
    assert rig.hub.router.used("rt") == estimate + 1000
    snap = rig.route.snapshot()
    assert snap["usage_reports"] == 1 and snap["usage_repeats"] == 1


def test_a_usage_below_the_estimate_charges_nothing_more(gemini, rigs):
    srv = gemini([["audio"]], usage=1)
    rig = rigs(srv)
    rig.turn("ciao")
    estimate = rig.hub.router.used("rt")
    assert wait_for(lambda: rig.route.snapshot()["usage_reports"] == 1, 5.0)
    assert rig.hub.router.used("rt") == estimate


def test_a_released_held_answer_never_trips_the_first_token_watchdog(gemini, rigs):
    srv = gemini([["audio"]])
    rig = rigs(srv, rt=AUDIO, app=OPT_IN, first_token_timeout_s=0.5)
    request = rig.route.request

    def late(*args, **kw):  # the whole answer is held before the route asks for it
        assert wait_for(lambda: srv.of("activityEnd"))
        time.sleep(0.3)
        return request(*args, **kw)

    rig.route.request = late
    ear = listen(rig, ["x"])
    try:
        ear.say()
        assert ear.settle(lambda: len(rig.events) == 1, 10.0)
        ear.settle(lambda: False, 1.0)  # past the first-token timeout
        assert rig.events[-1]["outcome"] == "ok" and rig.sink.pushed
        snap = rig.route.snapshot()
        assert snap["client"]["stalls"] == 0 and snap["state"] == "connected"
    finally:
        ear.close()


def test_a_live_answer_moves_the_watchdog_and_a_silent_one_stalls(gemini, rigs):
    srv = gemini([["audio"]], delay_s=0.3)
    rig = rigs(srv, first_token_timeout_s=1.0)
    rig.turn("ciao")
    assert rig.events[-1]["outcome"] == "ok"
    assert rig.route.snapshot()["client"]["stalls"] == 0
    srv.delay_s = 3.0
    rig.turn("ancora")
    assert wait_for(lambda: rig.route.snapshot()["client"]["stalls"] == 1, 5.0)


def test_gemini_cutting_its_own_answer_ends_the_turn_once(gemini, rigs):
    srv = gemini([["audio", "interrupt"]])
    rig = rigs(srv)
    rig.turn("ciao")
    time.sleep(0.5)
    assert len(rig.events) == 1 and rig.events[-1]["outcome"] != "no_text"
    assert len([t for t in text_turns(srv) if t.get("turnComplete")]) == 1  # nothing more asked
    snap = rig.route.snapshot()
    assert snap["server_interrupts"] == 1 and snap["responses"] == {"cancelled": 1}
