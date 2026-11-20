"""SPEC 8, Robot expression; SPEC 11, Media player: a radio station that drops is said in one
sentence, and the player is only ever read."""

from __future__ import annotations

import pytest
from expressor_rig import Rig

DROP = {"station": "Radio Uno", "reason": "dropped"}
USER_STOPS = (
    "replaced",
    "voice_stop",
    "fist_stop",
    "webui",
    "asleep",
    "nap",
    "disabled",
    "removed",
    "max_play",
    "closed",
    "tool",
)


def attached() -> Rig:
    rig = Rig()
    rig.expressor.attach()
    return rig


def test_dropped_station_is_said_and_player_never_called():
    rig = attached()
    rig.host.bus.publish("media.stopped", DROP, 0.0)
    request = rig.expressor._take()
    assert request is not None and request.situation == "failed_action"
    assert rig.expressor.handle(request) == "spoken"
    assert "Radio Uno" in rig.worker.turns[0]["text"]
    assert rig.media.calls == []


@pytest.mark.parametrize("reason", ["dropped", "error"])
def test_failed_station_is_not_music(reason):
    rig = attached()
    rig.host.bus.publish("media.stopped", {"station": "Radio Uno", "reason": reason}, 0.0)
    assert rig.expressor.handle(rig.expressor._take()) != "gesture_only_music"


@pytest.mark.parametrize("reason", USER_STOPS)
def test_user_stops_do_not_count(reason):
    rig = attached()
    rig.host.bus.publish("media.stopped", {"station": "Radio Uno", "reason": reason}, 0.0)
    assert rig.expressor.snapshot()["queued"] == 0
    assert rig.expressor.snapshot()["counts"] == {}


def test_another_station_playing_by_then_shows_gesture_only():
    rig = attached()
    rig.host.bus.publish("media.stopped", DROP, 0.0)
    rig.media.is_playing = True  # the person asked for another station meanwhile
    assert rig.expressor.handle(rig.expressor._take()) == "gesture_only_music"
    assert rig.host.expression.requests == ["puzzled"]
    assert rig.worker.turns == [] and rig.media.calls == []


def test_station_cooldown():
    rig = attached()
    rig.host.bus.publish("media.stopped", DROP, 0.0)
    assert rig.expressor.handle(rig.expressor._take()) == "spoken"
    rig.clock.t = 60.0
    rig.host.bus.publish("media.stopped", DROP, 60.0)
    assert rig.expressor.snapshot()["queued"] == 0
    assert rig.counts("failed_action") == {"spoken": 1, "dropped_cooldown": 1}


def test_closed_expressor_hears_nothing():
    rig = attached()
    rig.expressor.close()
    rig.host.bus.publish("media.stopped", DROP, 0.0)
    assert rig.expressor.snapshot()["queued"] == 0
