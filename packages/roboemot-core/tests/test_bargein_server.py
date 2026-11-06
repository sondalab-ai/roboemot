"""Barge-in with server turns (S2; SPEC 8 stage 8 and Server turns; SPEC 13): at the local
cut-off point the speech is handed to the server (a stream over the robot's voice) and the duck
stays down; the server's ``speech_started`` for that stream confirms it and only then is the
robot stopped. No confirmation within ``voice.barge_in.duck.confirm_ms`` brings the voice back
and ends the stream. The echo guard and the onset rules decide the cut-off point as before."""

from __future__ import annotations

import numpy as np
from roboemot.config.schema import RoboEmotConfig
from test_bargein import World, onset_run, user_voice
from test_voice_listener import FRAME, RATE


class Server:
    """The listener's side of a barge-in handed to the server: what was asked, what was
    ended, and whether it takes the stream."""

    def __init__(self, take: bool = True) -> None:
        self.take = take
        self.asked: list[tuple[int, float]] = []
        self.ended = 0

    def playback(self, samples: np.ndarray, start: float) -> bool:
        self.asked.append((samples.size, start))
        return self.take

    def end_playback(self) -> None:
        self.ended += 1


def server_world(take: bool = True, cfg: RoboEmotConfig | None = None, **kw) -> tuple:
    w = World(["ciao", "ciao robot come stai"], cfg=cfg, **kw)
    server = Server(take)
    resumed: list[float] = []

    def resume(samples, stamp):
        resumed.append(stamp)
        w.rig.listener.resume(samples, stamp)

    w.bargein.bind(
        w.stop,
        w.interrupt,
        resume,
        playback=server.playback,
        end_playback=server.end_playback,
        server_mode=lambda: True,
    )
    return w, server, resumed


def barge(w: World, seconds: float = 0.6) -> float:
    """The robot speaks; the person talks over it. Returns the onset."""
    w.speak(6.0)
    w.feed(1.0)
    onset = w.rig.t + 0.2
    w.feed(seconds, user_voice(seconds, 0.21), user_at=0.2)
    return onset


def ins(w: World) -> dict:
    return w.bargein.snapshot()["barge_ins"]


def test_the_cut_off_point_ducks_and_hands_the_speech_to_the_server_without_a_stop():
    w, server, resumed = server_world()
    onset = barge(w)
    assert len(server.asked) == 1 and w.stops == [] and w.interrupts == 0 and resumed == []
    size, start = server.asked[0]
    assert start <= onset + 0.05 and size > 0  # from the onset: the first words go too
    assert w.bargein.duck.ducking(w.rig.t)  # the voice stays down while the server decides
    assert ins(w)["playback_streams"] == 1 and w.bargein.snapshot()["triggers"] == 0


def test_the_servers_confirmation_stops_the_robot_at_the_next_chunk():
    w, server, resumed = server_world()
    onset = barge(w)
    w.bargein.confirm()  # the client thread: speech_started of the barge-in's stream
    assert w.stops == []  # only the listener thread cuts the robot off
    w.feed(0.1, user_voice(0.1, 0.21))
    assert len(w.stops) == 1 and w.interrupts == 1
    assert resumed == []  # the listener hears the person already: no hand-back
    snap = w.bargein.snapshot()
    assert ins(w)["server_confirmed"] == 1 and snap["duck"]["cut"] == 1
    assert not w.bargein.duck.ducking(w.rig.t)
    latency = snap["latency"]["onset_to_stop_server"]
    assert latency["n"] == 1 and latency["last_ms"] >= (w.stops[0] - onset) * 1000 - 50
    assert "onset_to_stop" not in snap["latency"]  # robot mode's SPEC 13 figure stays its own
    assert snap["recent"][-1]["outcome"] == "cut" and snap["recent"][-1]["confirmed"] == "server"
    assert server.ended == 0


def test_no_confirmation_in_time_restores_the_voice_and_ends_the_stream():
    w, server, _ = server_world()
    barge(w)
    confirm_s = w.cfg.voice.barge_in.duck.confirm_ms / 1000
    w.feed(confirm_s + 0.2)  # the person keeps quiet; the server never confirms
    assert server.ended == 1 and w.stops == [] and w.interrupts == 0
    assert not w.bargein.duck.ducking(w.rig.t)
    b = ins(w)
    assert b["released_unconfirmed"] == 1 and b["false"] == {"robot": 0, "server": 1}
    assert w.bargein.snapshot()["recent"][-1]["released_by"] == "unconfirmed"


def test_the_duck_is_held_until_the_confirmation_not_released_by_a_pause():
    w, server, _ = server_world()
    barge(w, seconds=0.4)
    w.feed(0.5)  # a pause much longer than duck.release_ms: still waiting for the server
    assert w.bargein.duck.ducking(w.rig.t) and server.ended == 0
    assert ins(w)["false"] == {"robot": 0, "server": 0}


def test_a_late_confirmation_after_the_release_still_stops_the_robot():
    """With interrupt_response the server has cancelled its answer: the robot stops and
    truncates rather than play the rest of what it had."""
    w, server, _ = server_world()
    barge(w)
    w.feed(w.cfg.voice.barge_in.duck.confirm_ms / 1000 + 0.2)
    assert server.ended == 1
    w.bargein.confirm()
    w.feed(0.1)
    assert len(w.stops) == 1 and ins(w)["server_confirmed_late"] == 1


def test_a_confirmation_with_nothing_waiting_stops_nothing():
    w, server, _ = server_world()
    w.speak(3.0)
    w.bargein.confirm()
    w.feed(0.5)
    assert w.stops == [] and ins(w)["confirms_idle"] == 1


def test_the_robots_own_voice_hands_nothing_to_the_server():
    w, server, _ = server_world()
    w.speak(3.0)
    w.feed(3.0)
    assert server.asked == [] and w.stops == []
    assert w.bargein.snapshot()["echo_frames"] > 20  # the echo guard held it back


def test_a_refused_stream_cuts_the_robot_locally_as_in_robot_turns():
    w, server, resumed = server_world(take=False)
    onset = barge(w)
    assert len(server.asked) == 1 and len(w.stops) == 1 and w.interrupts == 1
    assert resumed and resumed[0] <= onset + 0.05  # the listener restarts on the person
    assert ins(w)["playback_refused"] == 1 and ins(w)["local_cuts"] == 1


def test_a_person_still_talking_after_a_release_is_cut_off_locally():
    w, server, resumed = server_world()
    barge(w)
    w.feed(w.cfg.voice.barge_in.duck.confirm_ms / 1000 + 0.1)  # released, quiet meanwhile
    w.feed(0.6, user_voice(0.6, 0.21), user_at=0.1)
    assert len(server.asked) == 1  # once per speaking window
    assert len(w.stops) == 1 and resumed and ins(w)["local_cuts"] == 1


def test_the_robots_answer_ending_first_leaves_the_stream_to_go_on():
    w, server, _ = server_world()
    barge(w)
    w.bargein.reset()  # the listener: the robot stopped speaking on its own
    assert ins(w)["ended_by_speech_end"] == 1 and server.ended == 0
    w.bargein.confirm()
    w.speak(2.0)  # a later answer: a confirmation from before it is idle
    w.feed(0.3)
    assert w.stops == [] and ins(w)["confirms_idle"] == 1


def test_the_stop_keyword_still_cuts_while_the_server_decides():
    w, server, _ = server_world()
    barge(w)
    w.bargein._trigger("keyword", w.rig.t, w.rig.t, np.empty(0, np.float32), path="keyword")
    assert len(w.stops) == 1 and w.bargein.snapshot()["keyword_triggers"] == 1
    w.feed(2.0)  # the release never comes: nothing waits any more
    assert server.ended == 0 and ins(w)["released_unconfirmed"] == 0


def test_a_quiet_candidate_released_counts_as_a_false_barge_in_of_its_mode():
    """Robot turns: a duck released after release_ms of quiet with no cut is the false
    barge-in QA A compares with the server's unconfirmed releases."""
    probs = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  # one confident frame: a cough, a click
    stops, bi = onset_run(probs, user_voice(8 * FRAME / RATE, 0.1), keyword_enabled=False)
    assert stops == []
    assert bi.snapshot()["barge_ins"]["false"] == {"robot": 1, "server": 0}


def test_confirm_ms_may_not_outlast_the_ducks_safety_cap():
    import pytest

    with pytest.raises(ValueError, match="confirm_ms"):
        RoboEmotConfig.model_validate(
            {"voice": {"barge_in": {"duck": {"confirm_ms": 4000, "max_episode_s": 3}}}}
        )
