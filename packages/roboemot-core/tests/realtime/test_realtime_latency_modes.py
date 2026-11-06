"""S3 (SPEC 13): the realtime route's report shows one latency set per turn mode, side by side
(``latency_ms.by_turns.robot`` and ``.server``): ``turn``, ``first_audio``, ``transcript`` and
the barge-in's ``onset_to_stop``, each with the same percentile shape (count, p50, p95, max). A
turn counts in the mode it ran in (a server turn answers a stream; every other turn is a robot
turn), a transcript in the mode its audio went (a stream or a burst), a stop of the robot's
voice in the mode its barge-in episode started in. The keys QA already reads stay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from roboemot.runtime.startup import bind_barge_in
from test_bargein import World, user_voice
from test_bargein_server import barge, server_world
from test_realtime_listener import ears  # noqa: F401 (fixture)
from test_realtime_server_responses import snap, turns
from test_realtime_server_turns import OPT_INS, SERVER
from turn_server import TurnServer

SHAPE = {"count", "p50", "p95", "max"}
KEYS = {"turn", "first_audio", "transcript", "onset_to_stop"}


def by_turns(ear) -> dict:
    return snap(ear)["latency_ms"]["by_turns"]


def test_both_modes_report_the_same_keys_even_before_any_turn(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, script=["x"])
        latency = snap(ear)["latency_ms"]
        assert {"first_audio", "transcript", "turn", "server"} <= set(latency)  # kept
        modes = latency["by_turns"]
        assert set(modes) == {"robot", "server"}
        for mode in modes.values():
            assert set(mode) == KEYS and all(v == {"count": 0} for v in mode.values())


def test_a_server_turn_counts_in_server_mode_and_a_text_turn_in_robot_mode(ears):  # noqa: F811
    with TurnServer(stream_vad=True, vad_answers=True) as server:
        ear = ears(server, rt=SERVER, app=OPT_INS, script=["x"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: turns(ear) == 1, 10.0)
        assert ear.settle(lambda: by_turns(ear)["server"]["transcript"]["count"] == 1, 10.0)
        modes = by_turns(ear)
        for key in ("turn", "first_audio", "transcript"):
            assert set(modes["server"][key]) == SHAPE
            assert modes["robot"][key] == {"count": 0}
        latency = snap(ear)["latency_ms"]
        assert modes["server"]["turn"] == latency["turn"]  # the same figure, by its mode
    with TurnServer() as server:  # robot turns, text in
        ear2 = ears(server, script=["x"])
        ear2.rt.turn("ciao")
        modes = by_turns(ear2)
        assert modes["robot"]["turn"]["count"] == modes["robot"]["first_audio"]["count"] == 1
        assert modes["server"]["turn"] == {"count": 0}


def test_a_burst_transcript_counts_in_robot_mode(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, script=["ciao Neve"], to_worker=True)
        ear.say()
        assert ear.settle(lambda: by_turns(ear)["robot"]["transcript"]["count"] == 1, 10.0)
        assert by_turns(ear)["server"]["transcript"] == {"count": 0}


def test_the_route_summarises_the_barge_ins_stops_per_mode(ears):  # noqa: F811
    with TurnServer() as server:
        ear = ears(server, script=["x"])
        route = ear.rt.route
        for ms in (120.0, 150.0, 180.0):
            route.note_barge_in("robot", ms / 1000)
        route.note_barge_in("server", 0.9)
        modes = by_turns(ear)
        assert modes["robot"]["onset_to_stop"] == {
            "count": 3, "p50": 150.0, "p95": 180.0, "max": 180.0,
        }  # fmt: skip
        assert modes["server"]["onset_to_stop"]["count"] == 1
        assert modes["server"]["onset_to_stop"]["max"] == pytest.approx(900.0)


def test_a_local_cut_reports_a_robot_stop_and_a_confirmation_a_server_stop():
    w = World(["ciao", "ciao robot come stai"])
    stops: list[tuple[str, float]] = []
    w.bargein.on_stop = lambda mode, seconds: stops.append((mode, seconds))
    w.speak(5.0)
    w.feed(1.0)
    w.feed(2.0, user_voice(1.5, 0.21), user_at=0.5)
    assert [m for m, _ in stops] == ["robot"] and stops[0][1] < 0.2  # SPEC 13
    s, server, _ = server_world()
    s.bargein.on_stop = lambda mode, seconds: stops.append((mode, seconds))
    barge(s)
    s.bargein.confirm()
    s.feed(0.1, user_voice(0.1, 0.21))
    assert [m for m, _ in stops] == ["robot", "server"]
    latency = s.bargein.snapshot()["latency"]["onset_to_stop_server"]
    assert stops[1][1] * 1000 == pytest.approx(latency["last_ms"])


def test_the_runtime_hands_the_barge_ins_stops_to_the_realtime_route():
    noted: list[tuple[str, float]] = []
    route = SimpleNamespace(note_barge_in=lambda mode, s: noted.append((mode, s)))
    audio_in = SimpleNamespace(route=route, on_confirm=None)
    bargein = SimpleNamespace(duck=None, bind=lambda *a, **kw: None, confirm=object())
    bargein.on_stop = None
    listener = SimpleNamespace(
        realtime_input=audio_in, resume=None, stream_playback=None, end_playback=None,
        server_turns=None, playback_job=None,
    )  # fmt: skip
    conv = SimpleNamespace(interrupt=None, mark_interrupting=None)
    bind_barge_in(bargein, SimpleNamespace(stop=None), conv, listener)
    bargein.on_stop("robot", 0.1)
    assert noted == [("robot", 0.1)] and audio_in.on_confirm is bargein.confirm
