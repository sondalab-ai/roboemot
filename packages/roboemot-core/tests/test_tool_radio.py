"""SPEC 11 (Voice tools, Radio) and SPEC 12 (only listed URLs play, admin opt-in): the radio
skills over ``ToolHost.media``."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, Station
from roboemot.skills.host import ToolHost
from roboemot.skills.plugins import load_plugin_skill_sets
from roboemot.skills.registry import SkillContext, SkillRegistry, execute
from roboemot.skills.schema import ArgError, validate
from roboemot.tools import radio
from roboemot.tools.radio_catalogue import radio_catalogue

NAMES = ["radio_play", "radio_stop", "radio_volume"]
CTX = SkillContext(speaker="p1", language="it", now=10.0)
EXEC = {"timeout_s": 1.0, "max_chars": 50, "enum_max": 20}
T0 = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
JAZZ = {"name": "Jazz FM", "url": "http://jazz.example.test/live", "genres": ["jazz", "swing"]}
ROCK = {"name": "Rock One", "url": "https://rock.example.test/s", "genres": ["rock", "jazz"]}


class FakePlayer:
    def __init__(self, answer="ok"):
        self.answer = answer
        self.played: list[tuple[Station, float | None]] = []
        self.stops: list[str] = []
        self.steps: list[float] = []
        self.level = -12.0
        self.playing: Station | None = None

    def play(self, station, connect_timeout_s=None):
        self.played.append((station, connect_timeout_s))
        if self.answer == "ok":
            self.playing = station
        return self.answer

    def stop(self, reason):
        self.stops.append(reason)
        self.playing = None

    def volume(self, delta_db):
        self.steps.append(delta_db)
        self.level += delta_db
        return self.level

    def now_playing(self):
        return self.playing


def rig(enabled=True, stations=(JAZZ, ROCK), player=None, radio_cfg=None, **conv):
    cfg = {"cfg": RoboEmotConfig.model_validate(
        {
            "tools": {
                "radio": {"enabled": enabled, "stations": list(stations), **(radio_cfg or {})}
            },
            "conversation": conv,
        }
    )}  # fmt: skip
    host = ToolHost(config=lambda: cfg["cfg"], wall_now=lambda: T0, bus=Bus())
    host.media = player if player is not None else FakePlayer()
    fns = {fn.skill.name: fn for fn in radio.skills(host)}
    return host, fns, cfg


def test_core_registers_the_radio_plugin_and_lists_it_by_default():
    host = ToolHost(config=lambda: RoboEmotConfig(), wall_now=lambda: T0, bus=Bus())
    plugins = {p.name: p for p in load_plugin_skill_sets(host)}
    assert plugins["radio"].distribution == "roboemot-core"
    assert [fn.skill.name for fn in plugins["radio"].skills] == NAMES
    assert set(NAMES) <= set(RoboEmotConfig().conversation.llm_skills)


def test_radio_not_offered_when_disabled_or_without_a_player():
    host, fns, cfg = rig(enabled=False)
    assert not any(fn.skill.is_available() for fn in fns.values())
    assert SkillRegistry(list(fns.values())).offered(NAMES) == []
    cfg["cfg"] = RoboEmotConfig.model_validate(  # read live
        {"tools": {"radio": {"enabled": True, "stations": []}}}
    )
    assert fns["radio_play"].skill.is_available()
    assert not fns["radio_stop"].skill.is_available()  # nothing to stop or turn up
    assert not fns["radio_volume"].skill.is_available()
    cfg["cfg"] = RoboEmotConfig.model_validate(
        {"tools": {"radio": {"enabled": True, "stations": [JAZZ]}}}
    )
    assert all(fn.skill.is_available() for fn in fns.values())
    host.media = None  # the run has no player (no speaker that can play media)
    assert not any(fn.skill.is_available() for fn in fns.values())


def test_radio_play_accepts_only_configured_names_or_genres():
    _, fns, _ = rig()
    play = fns["radio_play"].skill
    options = play.resolved_choices()
    assert options["station"] == ["jazz", "swing", "rock", "Jazz FM", "Rock One"]  # deduped
    for bad in (JAZZ["url"], "http://evil.example.test/x", "Classic"):
        with pytest.raises(ArgError) as err:
            validate(play.params, {"station": bad}, options)
        assert err.value.param == "station"
    assert validate(play.params, {"station": "jazz"}, options) == {"station": "jazz"}


def test_a_genre_picks_the_first_station_that_lists_it():
    host, fns, _ = rig()
    assert fns["radio_play"](station="jazz").content == {"status": "ok", "station": "Jazz FM"}
    assert fns["radio_play"](station="rock").content == {"status": "ok", "station": "Rock One"}
    assert fns["radio_play"](station="Rock One").content["station"] == "Rock One"
    assert [s.name for s, _ in host.media.played] == ["Jazz FM", "Rock One", "Rock One"]


def test_radio_play_waits_no_longer_than_the_tool_timeout():
    """The model never hears ``failed`` for a stream that then plays: the connect wait is
    capped by ``conversation.tool_timeout_s`` (5 s by default, below the 8 s connect timeout)."""
    host, fns, _ = rig(tool_timeout_s=3.0)
    fns["radio_play"](station="jazz")
    assert host.media.played[-1][1] == pytest.approx(3.0 - radio.TOOL_TIMEOUT_MARGIN_S)
    assert host.media.played[-1][1] < 3.0  # strictly below: the executor never wins the race
    host, fns, _ = rig(tool_timeout_s=30.0)
    fns["radio_play"](station="jazz")
    media = RoboEmotConfig().tools.media
    assert host.media.played[-1][1] == pytest.approx(media.connect_timeout_s)


def test_an_unreachable_station_is_an_error_result_and_logs_its_name_only(caplog):
    host, fns, _ = rig(player=FakePlayer(answer="unreachable"))
    with caplog.at_level(logging.INFO):
        out = fns["radio_play"](station="Jazz FM").content
    assert out == {"error": "unreachable", "station": "Jazz FM"}
    assert JAZZ["url"] not in caplog.text


def test_radio_stop_and_volume():
    host, fns, _ = rig()
    assert fns["radio_stop"]().content == {"error": "not_playing"}
    fns["radio_play"](station="jazz")
    assert fns["radio_stop"]().content == {"ok": True, "station": "Jazz FM"}
    assert host.media.stops == ["tool"]
    step = RoboEmotConfig().tools.media.volume_step_db
    assert fns["radio_volume"](change="up").content == {"volume_db": -12.0 + step}
    assert fns["radio_volume"](change="down").content == {"volume_db": -12.0}
    assert host.media.steps == [step, -step]


def test_stop_and_volume_stay_available_while_something_plays():
    """Review I3: with the radio turned off (or no station left) while a station still plays,
    "stop the radio" must still have a tool; radio_play does not."""
    host, fns, cfg = rig()
    fns["radio_play"](station="jazz")
    cfg["cfg"] = RoboEmotConfig.model_validate({"tools": {"radio": {"enabled": False}}})
    assert not fns["radio_play"].skill.is_available()
    assert fns["radio_stop"].skill.is_available() and fns["radio_volume"].skill.is_available()
    assert fns["radio_stop"]().content == {"ok": True, "station": "Jazz FM"}
    assert not fns["radio_stop"].skill.is_available()  # nothing plays any more


async def test_radio_on_without_stations_is_offered_and_says_not_configured():
    """Owner decision 2026-11-17: "metti un po' di musica" with the radio on and no station
    must reach ``radio_play``, so the robot can say how to add a station. It plays nothing."""
    host, fns, _ = rig(stations=())
    reg = SkillRegistry(list(fns.values()))
    offered = reg.offered(NAMES)
    assert [s.name for s in offered] == ["radio_play"]
    snap = SkillRegistry.snapshot_choices(offered, EXEC["enum_max"])
    [tool] = SkillRegistry.tools(offered, choices=snap, max_chars=EXEC["max_chars"])
    station = tool["function"]["parameters"]["properties"]["station"]
    assert station["type"] == "string" and "enum" not in station
    assert "not_configured" in tool["function"]["description"]
    out = await execute(reg, NAMES, "radio_play", {"station": "jazz"}, CTX, choices=snap, **EXEC)
    assert json.loads(out.content) == {"error": "not_configured"}
    assert host.media.played == []


def test_with_stations_an_unlisted_name_is_not_found_and_plays_nothing():
    host, fns, _ = rig()
    assert fns["radio_play"](station="http://evil.example.test/x").content == {"error": "not_found"}
    assert host.media.played == []


def test_radio_results_are_known_result_keys():
    from roboemot.skills.registry import RESULT_KEYS

    assert {"station", "volume_db"} <= set(RESULT_KEYS)


def offered_station_enum(fns, enum_max=EXEC["enum_max"]):
    """The ``station`` enum a request shows and the snapshot the executor checks."""
    reg = SkillRegistry(list(fns.values()))
    snap = SkillRegistry.snapshot_choices(reg.offered(NAMES), enum_max)
    [tool] = [t for t in SkillRegistry.tools(reg.offered(NAMES), choices=snap)
              if t["function"]["name"] == "radio_play"]  # fmt: skip
    return reg, snap, tool["function"]["parameters"]["properties"]["station"]["enum"]


def test_the_whole_catalogue_offers_every_genre_and_every_name_genres_first():
    """Robot run 2026-11-18: with the shipped catalogue added, "ambient" and "chill out" were
    not understood: names came first and the global cap (20) cut every genre away."""
    stations = [
        {"name": s.name, "url": s.url, "genres": list(s.genres)}
        for g in radio_catalogue().groups
        for s in g.stations
    ]
    _, fns, cfg = rig(stations=stations)
    names = [s["name"] for s in stations]
    genres = list(dict.fromkeys(x for s in stations for x in s["genres"]))
    assert len(names) > EXEC["enum_max"] and {"ambient", "chillout"} <= set(genres)
    _, _, enum = offered_station_enum(fns)
    assert enum == genres + names
    assert len(enum) <= cfg["cfg"].tools.radio.max_choices


async def test_a_small_radio_cap_keeps_genres_first_and_the_executor_matches_the_cut(caplog):
    _, fns, _ = rig(radio_cfg={"max_choices": 3})
    with caplog.at_level(logging.WARNING):
        reg, snap, enum = offered_station_enum(fns)
    assert enum == ["jazz", "swing", "rock"]
    assert "tools.radio.max_choices" in caplog.text
    ok = await execute(reg, NAMES, "radio_play", {"station": "rock"}, CTX, choices=snap, **EXEC)
    assert json.loads(ok.content) == {"status": "ok", "station": "Rock One"}
    cut = {"station": "Jazz FM"}  # past the cut: not offered, so not accepted
    out = await execute(reg, NAMES, "radio_play", cut, CTX, choices=snap, **EXEC)
    assert out.code == "invalid_arguments"
    out = await execute(reg, NAMES, "radio_play", cut, CTX, **EXEC)  # resolved now: same cut
    assert out.code == "invalid_arguments"


def test_the_radio_cap_replaces_the_global_one_only_for_the_radio():
    _, fns, _ = rig(radio_cfg={"max_choices": 5})
    _, _, enum = offered_station_enum(fns, enum_max=1)
    assert len(enum) == 5


def test_a_genre_equal_to_a_station_name_is_offered_once_and_plays_that_station():
    jazz = {**JAZZ, "genres": ["Rock One"]}
    host, fns, _ = rig(stations=(jazz, ROCK))
    assert radio.names_and_genres(host.config()) == ["Rock One", "rock", "jazz", "Jazz FM"]
    assert fns["radio_play"](station="Rock One").content["station"] == "Rock One"
