"""SPEC 11 (Voice tools), 11A (the Tools tab) and 12 (privacy, the PIN session): the Tools tab.

``GET /api/tools/state`` is a passive poll behind the PIN session: the tools with their switch
and availability, the active timers and reminders, the missed items and what plays (the station
name only, never its URL). Every write (cancel, clear the missed items, stop the radio, a test
play, the place search) is a POST behind the session, the CSRF token and the re-entered PIN. The
place search query travels in the body, and no reminder text, label, place, coordinate or query
reaches a log line.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from roboemot.config.schema import RoboEmotConfig, Station
from roboemot.config.stack import KEYED_LISTS, ConfigStack, GuardrailError
from roboemot.diagnostics import Diagnostics
from roboemot.runtime.tools_state import ToolsError, ToolsState, check_catalogue, folded_name
from roboemot.tools.radio_catalogue import GROUP_IDS, parse_catalogue, radio_catalogue
from roboemot.tools.scheduler import Scheduler
from roboemot.webui import make_server
from roboemot.webui.api import PASSIVE_POLLS, TOOLS_STATE, TOOLS_STATUS
from roboemot.webui.settings import SettingNotAllowed, SettingsEditor
from roboemot.webui.strings import STRINGS, language_for
from roboemot.webui.validation import ValidationError
from test_webui import PIN, Client, FakeService

REMINDER = "pick up Giulia at the dentist"
LABEL = "pasta-secret-label"
PLACE_QUERY = "Borgo Segreto"
PLACE_LAT, PLACE_LON = 45.123456, 9.654321
STATION_URL = "https://stream.example.org/secret-radio.mp3"
ZONE = ZoneInfo("Europe/Rome")
START = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)
DOTTED = "RTL 102.5"  # a catalogue keys this station by its index, not its name


class Mono:
    """A monotonic clock the test moves."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Player:
    """The part of ``MediaPlayer`` the tab uses; records what it was asked to do."""

    def __init__(self) -> None:
        self.station: Station | None = None
        self.played: list[str] = []
        self.stopped: list[str] = []
        self.waits: list[float | None] = []
        self.status = "ok"

    def play(self, station: Station, connect_timeout_s: float | None = None) -> str:
        self.played.append(station.name)
        self.waits.append(connect_timeout_s)
        if self.status == "ok":
            self.station = station
        return self.status

    def stop(self, reason: str) -> None:
        self.stopped.append(reason)
        self.station = None

    def now_playing(self) -> Station | None:
        return self.station

    def volume_db(self) -> float:
        return -12.0


def geocoder(calls: list[httpx.Request]):
    """An HTTP client factory over a mock Open-Meteo place search."""

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        body = {
            "results": [
                {
                    "name": PLACE_QUERY,
                    "admin1": "Lombardia",
                    "country": "Italia",
                    "latitude": PLACE_LAT,
                    "longitude": PLACE_LON,
                }
            ]
        }
        return httpx.Response(200, json=body)

    return lambda: httpx.Client(transport=httpx.MockTransport(handle))


def make_scheduler(tmp_path, cfg: RoboEmotConfig) -> Scheduler:
    sched = Scheduler(
        tmp_path / "schedule.json", lambda: cfg.tools.scheduler, lambda: START, lambda: ZONE
    )
    sched.load(START)
    return sched


def app_config(tmp_path, **extra):
    return {
        "data_dir": str(tmp_path / "data"),
        "webui": {"bind": "127.0.0.1", "port": 0},
        "circadian": {"timezone": "Europe/Rome"},
        **extra,
    }


@pytest.fixture
def served(tmp_path):
    stack = ConfigStack(
        app=app_config(
            tmp_path,
            tools={
                "radio": {
                    "enabled": True,
                    "stations": [
                        {"name": "Jazz", "url": STATION_URL},
                        {"name": DOTTED, "url": "https://stream.example.org/rtl.mp3"},
                    ],
                }
            },
        ),
        audit_path=tmp_path / "audit.jsonl",
    )
    cfg = stack.effective()
    sched = make_scheduler(tmp_path, cfg)
    player = Player()
    calls: list[httpx.Request] = []
    host = SimpleNamespace(scheduler=sched, media=player, circadian_runner=object())
    mono = Mono()
    tools = ToolsState(host, cfg, http=geocoder(calls), clock=mono)
    stack.subscribe(tools.apply_config)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, tools=tools
    )
    srv.set_pin(PIN)
    srv.start()
    yield SimpleNamespace(
        client=Client(srv.port),
        sched=sched,
        player=player,
        calls=calls,
        stack=stack,
        host=host,
        mono=mono,
        editor=editor,
    )
    srv.stop()


def logged(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- the four brief tests


def test_state_needs_session_and_lists_items(served):
    c = served.client
    assert TOOLS_STATE == "/api/tools/state" and TOOLS_STATE in PASSIVE_POLLS
    assert c.request("GET", TOOLS_STATE).status == 401
    timer = served.sched.add_timer(600, LABEL)
    reminder = served.sched.add_reminder(REMINDER, datetime(2026, 11, 18, 8, 30), "daily", ())
    c.login()
    resp = c.request("GET", TOOLS_STATE)
    assert resp.status == 200
    data = resp.json
    assert data["poll_ms"] == 2000
    assert [t["name"] for t in data["tools"]] == ["sleep", "scheduler", "weather", "radio"]
    by = {t["name"]: t for t in data["tools"]}
    assert by["scheduler"] == {"name": "scheduler", "enabled": True, "available": True}
    assert by["weather"] == {"name": "weather", "enabled": False, "available": False}
    assert by["radio"] == {"name": "radio", "enabled": True, "available": True}
    assert data["items"] == [
        {
            "id": timer.id,
            "kind": "timer",
            "text": LABEL,
            "due_local": "2026-11-17T10:10:00",
            "repeat": "once",
            "days": [],
        },
        {
            "id": reminder.id,
            "kind": "reminder",
            "text": REMINDER,
            "due_local": "2026-11-18T08:30:00",
            "repeat": "daily",
            "days": [],
        },
    ]
    assert data["missed"] == [] and data["playing"] is None
    served.player.station = Station(name="Jazz", url=STATION_URL)
    playing = c.request("GET", TOOLS_STATE)
    assert playing.json["playing"] == {"name": "Jazz", "volume_db": -12.0}
    assert STATION_URL not in playing.body.decode(), "a station URL never leaves the editor"


def test_cancel_requires_pin_and_csrf(served):
    c = served.client
    item = served.sched.add_timer(600, LABEL)
    assert c.request("POST", "/api/tools/cancel", {"id": item.id, "pin": PIN}).status == 401
    c.login()
    no_csrf = c.request("POST", "/api/tools/cancel", {"id": item.id, "pin": PIN}, csrf=False)
    assert no_csrf.status == 403 and no_csrf.json == {"error": "csrf"}
    other = c.request(
        "POST",
        "/api/tools/cancel",
        {"id": item.id, "pin": PIN},
        headers={"Origin": "http://evil.example"},
    )
    assert other.status == 403 and other.json == {"error": "origin"}
    assert c.request("POST", "/api/tools/cancel", {"id": item.id}).status == 400
    weird = c.request("POST", "/api/tools/cancel", {"id": "../x", "pin": PIN})
    assert weird.status == 400
    assert [i.id for i in served.sched.items()] == [item.id]
    ok = c.request("POST", "/api/tools/cancel", {"id": item.id, "pin": PIN})
    assert ok.status == 200 and ok.json == {"ok": True}
    assert served.sched.items() == []
    gone = c.request("POST", "/api/tools/cancel", {"id": item.id, "pin": PIN})
    assert gone.status == 404 and gone.json == {"error": "unknown_item"}
    other = served.sched.add_timer(60, None)
    bad = c.request("POST", "/api/tools/cancel", {"id": other.id, "pin": "0000000"})
    assert bad.status == 401 and bad.json == {"error": "bad_pin"}
    assert [i.id for i in served.sched.items()] == [other.id], "a wrong PIN cancels nothing"


def test_geocode_is_post_and_not_logged(served, caplog):
    c = served.client
    c.login()
    caplog.set_level(logging.DEBUG)
    assert c.request("GET", "/api/tools/geocode?query=Borgo").status == 404
    resp = c.request("POST", "/api/tools/geocode", {"query": PLACE_QUERY, "pin": PIN})
    assert resp.status == 200
    assert resp.json == {
        "places": [
            {
                "name": PLACE_QUERY,
                "admin": "Lombardia",
                "country": "Italia",
                "lat": PLACE_LAT,
                "lon": PLACE_LON,
            }
        ]
    }
    (sent,) = served.calls
    assert sent.url.host == "geocoding-api.open-meteo.com"
    assert sent.url.params["name"] == PLACE_QUERY and sent.url.params["count"] == "5"
    page = language_for(served.stack.effective().conversation.language)
    assert sent.url.params["language"] == page
    text = logged(caplog)
    for secret in (PLACE_QUERY, "Borgo", str(PLACE_LAT), str(PLACE_LON), "Lombardia"):
        assert secret not in text
    no_pin = c.request("POST", "/api/tools/geocode", {"query": PLACE_QUERY})
    assert no_pin.status == 400 and len(served.calls) == 1
    empty = c.request("POST", "/api/tools/geocode", {"query": "  ", "pin": PIN})
    assert empty.status == 422 and empty.json["error"] == "query_invalid"
    long = c.request("POST", "/api/tools/geocode", {"query": "x" * 1000, "pin": PIN})
    assert long.status == 422 and len(served.calls) == 1


def test_state_hides_reminder_text_from_logs(served, caplog):
    c = served.client
    caplog.set_level(logging.DEBUG)
    timer = served.sched.add_timer(600, LABEL)
    served.sched.add_reminder(REMINDER, datetime(2026, 11, 17, 23, 0), "once", ())
    c.login()
    assert REMINDER in c.request("GET", TOOLS_STATE).body.decode()
    assert c.request("POST", "/api/tools/cancel", {"id": timer.id, "pin": PIN}).status == 200
    assert c.request("POST", "/api/tools/missed/clear", {"pin": PIN}).status == 200
    text = logged(caplog)
    assert REMINDER not in text and LABEL not in text and "Giulia" not in text


# ---------------------------------------------------------------- more routes and gates


def test_radio_test_plays_a_listed_station_and_stop_stops_it(served):
    c = served.client
    c.login()
    ok = c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN})
    assert ok.status == 200 and ok.json == {"status": "ok"}
    assert served.player.played == ["Jazz"]
    assert served.player.waits == [8.0], "the connect timeout, within the request's work budget"
    unknown = c.request("POST", "/api/tools/radio/test", {"name": "jazz music", "pin": PIN})
    assert unknown.status == 404 and unknown.json == {"error": "unknown_station"}
    served.player.status = "unreachable"
    served.mono.now += 60
    down = c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN})
    assert down.status == 200 and down.json == {"status": "unreachable"}
    stop = c.request("POST", "/api/tools/radio/stop", {"pin": PIN})
    assert stop.status == 200 and served.player.stopped[-1] == "webui"
    assert c.request("POST", "/api/tools/radio/stop", {}).status == 400


def test_radio_test_is_refused_while_radio_is_off(served):
    """SPEC 12: no outbound stream before the admin opt-in."""
    c = served.client
    c.login()
    off = {"path": "tools.radio.enabled", "value": False, "pin": PIN}
    assert c.request("POST", "/api/settings", off).status == 200
    resp = c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN})
    assert resp.status == 409 and resp.json == {"error": "radio_disabled"}
    assert served.player.played == []
    by = {t["name"]: t for t in c.request("GET", TOOLS_STATE).json["tools"]}
    assert by["radio"] == {"name": "radio", "enabled": False, "available": False}


def test_a_test_play_waits_no_longer_than_the_page_request_may_work():
    cfg = RoboEmotConfig(
        tools={
            "radio": {"enabled": True, "stations": [{"name": "Jazz", "url": STATION_URL}]},
            "media": {"connect_timeout_s": 30},
        }
    )
    player = Player()
    ToolsState(
        SimpleNamespace(scheduler=None, media=player, circadian_runner=None), cfg
    ).radio_test("Jazz")
    web = cfg.webui
    assert player.waits == [web.request_deadline_s * web.request_work_share]


def test_missed_items_are_listed_and_cleared(tmp_path):
    cfg = RoboEmotConfig(circadian={"timezone": "Europe/Rome"})
    path = tmp_path / "schedule.json"
    early = Scheduler(path, lambda: cfg.tools.scheduler, lambda: START, lambda: ZONE)
    early.load(START)
    early.add_timer(60, LABEL)
    later = START + timedelta(days=1)
    sched = Scheduler(path, lambda: cfg.tools.scheduler, lambda: later, lambda: ZONE)
    sched.load(later)
    tools = ToolsState(SimpleNamespace(scheduler=sched, media=None, circadian_runner=None), cfg)
    (missed,) = tools.state()["missed"]
    assert missed["text"] == LABEL and missed["kind"] == "timer"
    tools.clear_missed()
    assert tools.state()["missed"] == [] and sched.missed() == []


def test_without_a_scheduler_or_a_player_the_tab_still_answers(tmp_path):
    cfg = RoboEmotConfig()
    tools = ToolsState(SimpleNamespace(scheduler=None, media=None, circadian_runner=None), cfg)
    data = tools.state()
    assert data["items"] == [] and data["missed"] == [] and data["playing"] is None
    by = {t["name"]: t for t in data["tools"]}
    assert by["scheduler"]["available"] is False and by["radio"]["available"] is False
    for call, code in (
        (lambda: tools.cancel("t1"), "scheduler_unavailable"),
        (tools.clear_missed, "scheduler_unavailable"),
        (tools.radio_stop, "radio_unavailable"),
        (lambda: tools.radio_test("Jazz"), "radio_unavailable"),
    ):
        with pytest.raises(ToolsError) as err:
            call()
        assert err.value.code == code


def test_sleep_and_scheduler_switch_with_llm_skills():
    """SPEC 11: ``conversation.llm_skills`` is the switch of sleep and of the scheduler."""
    base = RoboEmotConfig().conversation.llm_skills
    cfg = RoboEmotConfig(conversation={"llm_skills": [s for s in base if s != "timer_set"]})
    sched = SimpleNamespace(items=list, missed=list)
    host = SimpleNamespace(scheduler=sched, media=None, circadian_runner=object())
    by = {t["name"]: t for t in ToolsState(host, cfg).state()["tools"]}
    assert by["sleep"] == {"name": "sleep", "enabled": True, "available": True}
    assert by["scheduler"] == {"name": "scheduler", "enabled": False, "available": False}


def test_weather_is_available_only_with_a_place():
    host = SimpleNamespace(scheduler=None, media=None, circadian_runner=None)
    on = {"weather": {"enabled": True}}
    without = ToolsState(host, RoboEmotConfig(tools=on)).state()["tools"][2]
    assert without == {"name": "weather", "enabled": True, "available": False}
    placed = RoboEmotConfig(tools=on, circadian={"location": {"lat": 45.0, "lon": 9.0}})
    assert ToolsState(host, placed).state()["tools"][2]["available"] is True


def test_radio_without_a_station_keeps_its_hint_while_the_model_still_gets_it():
    """Owner decision 2026-11-17: the model gets ``radio_play`` with no station (it answers
    ``not_configured``), but the tab still shows "Add a station" (``available`` False)."""
    media = SimpleNamespace(now_playing=lambda: None)
    host = SimpleNamespace(scheduler=None, media=media, circadian_runner=None)
    on = RoboEmotConfig(tools={"radio": {"enabled": True}})
    assert ToolsState(host, on).state()["tools"][3] == {
        "name": "radio", "enabled": True, "available": False,
    }  # fmt: skip


def test_a_failed_place_search_names_no_query(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    def broken() -> httpx.Client:
        def handle(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot reach {request.url}")

        return httpx.Client(transport=httpx.MockTransport(handle))

    tools = ToolsState(
        SimpleNamespace(scheduler=None, media=None, circadian_runner=None),
        RoboEmotConfig(),
        http=broken,
    )
    with pytest.raises(ToolsError) as err:
        tools.geocode(PLACE_QUERY)
    assert err.value.code == "geocode_unreachable"
    assert PLACE_QUERY not in logged(caplog) and "Borgo" not in logged(caplog)


def test_the_place_search_result_saves_in_one_batch(tmp_path):
    """The tab writes ``circadian.location`` and ``tools.weather.place_name`` in one batch."""
    stack = ConfigStack(app=app_config(tmp_path), audit_path=tmp_path / "audit.jsonl")
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    result = editor.change_many(
        [
            {"path": "circadian.location", "value": {"lat": PLACE_LAT, "lon": PLACE_LON}},
            {"path": "tools.weather.place_name", "value": PLACE_QUERY},
        ]
    )
    assert result["changed"] is True
    cfg = stack.effective()
    assert cfg.circadian.location == {"lat": PLACE_LAT, "lon": PLACE_LON}
    assert cfg.tools.weather.place_name == PLACE_QUERY


def test_every_refusal_has_a_sentence_in_both_languages():
    codes = {*TOOLS_STATUS, "tools_unavailable", "query_invalid"}
    for lang, strings in STRINGS.items():
        assert sorted(c for c in codes if f"err_{c}" not in strings) == [], lang


def test_the_state_serialises_and_holds_no_url(tmp_path):
    cfg = RoboEmotConfig(
        tools={"radio": {"enabled": True, "stations": [{"name": "Jazz", "url": STATION_URL}]}}
    )
    player = Player()
    player.station = cfg.tools.radio.stations[0]
    tools = ToolsState(SimpleNamespace(scheduler=None, media=player, circadian_runner=None), cfg)
    assert STATION_URL not in json.dumps(tools.state())


# ---------------------------------------------------------------- fix round 1


def test_a_station_whose_name_has_a_dot_test_plays_and_edits(served):
    """The catalogue keys "RTL 102.5" by its index; the page sends the real name (review I1)."""
    rows = {r["path"]: r for r in served.editor.fields()}
    key = next(
        k
        for k in rows["tools.radio.stations"]["value"]
        if rows[f"tools.radio.stations.{k}.name"]["value"] == DOTTED
    )
    assert key != DOTTED, "the path segment is not the name"
    c = served.client
    c.login()
    ok = c.request("POST", "/api/tools/radio/test", {"name": DOTTED, "pin": PIN})
    assert ok.status == 200 and ok.json == {"status": "ok"}
    assert served.player.played == [DOTTED]
    url = "https://stream.example.org/rtl2.mp3"
    change = {"changes": [{"path": f"tools.radio.stations.{key}.url", "value": url}], "pin": PIN}
    assert c.request("POST", "/api/settings/batch", change).status == 200
    station = next(s for s in served.stack.effective().tools.radio.stations if s.name == DOTTED)
    assert station.url == url


def test_place_searches_and_test_plays_keep_a_minimum_interval(served):
    c = served.client
    c.login()
    gap = served.stack.effective().webui.tools.min_interval_s
    assert gap > 0
    first = c.request("POST", "/api/tools/geocode", {"query": "Como", "pin": PIN})
    assert first.status == 200
    again = c.request("POST", "/api/tools/geocode", {"query": "Como", "pin": PIN})
    assert again.status == 429 and again.json == {"error": "tools_too_soon"}
    assert int(again.headers["Retry-After"]) >= 1
    assert len(served.calls) == 1, "the refused search sent nothing"
    play = c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN})
    assert play.status == 200, "the test play has its own interval"
    assert c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN}).status == 429
    served.mono.now += gap
    assert c.request("POST", "/api/tools/geocode", {"query": "Como", "pin": PIN}).status == 200
    assert c.request("POST", "/api/tools/radio/test", {"name": "Jazz", "pin": PIN}).status == 200


def test_the_place_search_waits_no_longer_than_the_page_request_may_work():
    seen: list[object] = []

    def slow() -> httpx.Client:
        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, json={"results": []})

        return httpx.Client(transport=httpx.MockTransport(handle))

    cfg = RoboEmotConfig(tools={"weather": {"timeout_s": 30}})
    host = SimpleNamespace(scheduler=None, media=None, circadian_runner=None)
    assert ToolsState(host, cfg, http=slow).geocode("Como") == []
    budget = cfg.webui.request_deadline_s * cfg.webui.request_work_share
    assert seen and all(v == budget for v in seen[0].values())


# ------------------------------------------------------- station editor: add, remove, restart


def radio_editor(tmp_path, stations=()):
    """A stack with the radio on (``stations`` in app.yaml) and its editor (file ``s.json``)."""
    radio = {"enabled": True, "stations": list(stations)}
    stack = ConfigStack(
        app={"webui": {"bind": "127.0.0.1", "port": 0}, "tools": {"radio": radio}},
        audit_path=tmp_path / "a.jsonl",
    )
    return stack, SettingsEditor(stack, store=tmp_path / "s.json")


def station(name: str, url: str = STATION_URL) -> dict:
    return {"name": name, "url": url, "genres": [], "language": None}


def names(stack: ConfigStack) -> list[str]:
    return [s.name for s in stack.effective().tools.radio.stations]


def restarted(tmp_path, stations=()) -> ConfigStack:
    """A fresh stack with the overrides file restored, as after a restart."""
    stack, editor = radio_editor(tmp_path, stations)
    editor.restore()
    return stack


def segment(editor: SettingsEditor, name: str) -> str:
    """The catalogue path segment of station ``name`` (its index when the name has a dot)."""
    rows = {r["path"]: r for r in editor.fields()}
    box = "tools.radio.stations"
    return next(k for k in rows[box]["value"] if rows[f"{box}.{k}.name"]["value"] == name)


def test_a_removed_station_is_gone_and_a_restart_keeps_every_added_one(tmp_path):
    """Owner report 2026-11-17: removing a station said "removed" but the station stayed, and a
    restart lost every added station (the file held one station, not a list)."""
    stack, editor = radio_editor(tmp_path)
    editor.add_entry("tools.radio.stations", None, {**station("Jazz"), "genres": ["jazz"]})
    editor.add_entry("tools.radio.stations", None, station("Rock", "https://example.com/b.mp3"))
    editor.add_entry("tools.radio.stations", None, station("Pop", "https://example.com/c.mp3"))
    assert names(restarted(tmp_path)) == ["Jazz", "Rock", "Pop"]
    assert editor.remove_entry(f"tools.radio.stations.{segment(editor, 'Jazz')}")["changed"]
    assert names(stack) == ["Rock", "Pop"]
    stored = json.loads((tmp_path / "s.json").read_text())["overrides"]
    assert all(isinstance(v, dict) and v.get("name") for v in stored.values()), stored
    assert sorted(v["name"] for v in stored.values()) == ["Pop", "Rock"]
    assert names(restarted(tmp_path)) == ["Rock", "Pop"]


def test_a_station_whose_name_has_a_dot_adds_edits_removes_and_restarts(tmp_path):
    stack, editor = radio_editor(tmp_path)
    editor.add_entry("tools.radio.stations", None, station("Jazz"))
    editor.add_entry("tools.radio.stations", None, station(DOTTED))
    key = segment(editor, DOTTED)
    assert key != DOTTED, "the path segment is the index"
    url = "https://stream.example.org/rtl2.mp3"
    editor.change_many([{"path": f"tools.radio.stations.{key}.url", "value": url}])
    again = restarted(tmp_path)
    assert names(again) == ["Jazz", DOTTED]
    assert again.effective().tools.radio.stations[1].url == url
    assert editor.remove_entry(f"tools.radio.stations.{key}")["changed"]
    assert names(stack) == ["Jazz"]
    assert names(restarted(tmp_path)) == ["Jazz"]


def test_the_page_adds_edits_and_removes_a_station_named_with_spaces(served, tmp_path):
    """The Tools tab's own routes, with a name that is a path segment as is ("Radio Deejay")."""
    c = served.client
    c.login()
    box, name = "tools.radio.stations", "Radio Deejay"
    added = {"path": box, "value": station(name, "https://example.com/d.mp3"), "pin": PIN}
    assert c.request("POST", "/api/settings/entry", added).status == 200
    key = segment(served.editor, name)
    url = "https://example.com/d2.mp3"
    edit = {"changes": [{"path": f"{box}.{key}.url", "value": url}], "pin": PIN}
    assert c.request("POST", "/api/settings/batch", edit).status == 200
    assert served.stack.effective().tools.radio.stations[2].url == url
    lower = [
        {"name": "Jazz", "url": STATION_URL},
        {"name": DOTTED, "url": "https://stream.example.org/rtl.mp3"},
    ]
    stack = ConfigStack(
        app=app_config(tmp_path, tools={"radio": {"enabled": True, "stations": lower}}),
        audit_path=tmp_path / "b.jsonl",
    )
    SettingsEditor(stack, store=tmp_path / "webui_settings.json").restore()
    assert [(s.name, s.url) for s in stack.effective().tools.radio.stations][2] == (name, url)
    gone = {"path": f"{box}.{key}", "pin": PIN}
    assert c.request("POST", "/api/settings/entry/remove", gone).status == 200
    assert names(served.stack) == ["Jazz", DOTTED]


@pytest.mark.parametrize(
    ("name", "code"),
    [(" Jazz", "name_invalid"), ("Radio  Jazz", "name_invalid"), ("Ra\ndio", "name_invalid")],
)
def test_a_station_name_is_one_plain_line(tmp_path, name, code):
    stack, editor = radio_editor(tmp_path)
    editor.add_entry("tools.radio.stations", None, station("Radio Deejay"))
    with pytest.raises(ValidationError) as exc:
        editor.add_entry("tools.radio.stations", None, station(name))
    assert exc.value.code == code
    assert names(stack) == ["Radio Deejay"]


def test_a_station_of_app_yaml_cannot_be_removed_and_its_edit_survives_a_restart(tmp_path):
    lower = [station("Jazz"), station(DOTTED, "https://stream.example.org/rtl.mp3")]
    stack, editor = radio_editor(tmp_path, lower)
    editor.add_entry("tools.radio.stations", None, station("Rock", "https://example.com/b.mp3"))
    for name in ("Jazz", DOTTED):
        with pytest.raises(ValidationError) as exc:
            editor.remove_entry(f"tools.radio.stations.{segment(editor, name)}")
        assert exc.value.code == "entry_in_lower_layer"
    url = "https://stream.example.org/rtl2.mp3"
    edit = {"path": f"tools.radio.stations.{segment(editor, DOTTED)}.url", "value": url}
    editor.change_many([edit])
    again = restarted(tmp_path, lower)
    assert names(again) == ["Jazz", DOTTED, "Rock"]
    assert again.effective().tools.radio.stations[1].url == url
    assert names(stack) == ["Jazz", DOTTED, "Rock"]


def test_the_broken_station_file_of_an_older_version_restores_its_station(tmp_path, caplog):
    """Before the fix the file held ``{"tools.radio.stations": {one station}}`` (the last one
    added). It is read as that station, the file is rewritten, and no log line names its URL."""
    old = {"v": 1, "overrides": {"tools.radio.stations": station("Rock")}}
    (tmp_path / "s.json").write_text(json.dumps(old))
    (tmp_path / "s.json").chmod(0o600)
    caplog.set_level(logging.DEBUG)
    assert names(restarted(tmp_path)) == ["Rock"]
    assert STATION_URL not in logged(caplog)
    stored = json.loads((tmp_path / "s.json").read_text())["overrides"]
    assert "tools.radio.stations" not in stored, "the file is rewritten in the new shape"
    assert names(restarted(tmp_path)) == ["Rock"]


def test_a_station_field_edit_stored_by_an_older_version_still_lands(tmp_path):
    """Before the fix an edit of an app.yaml station was stored by its catalogue path (an index
    for a dotted name); it still edits that station once the list is merged by name."""
    lower = [station("Jazz"), station(DOTTED, "https://stream.example.org/rtl.mp3")]
    url = "https://stream.example.org/rtl2.mp3"
    old = {
        "v": 1,
        "overrides": {"tools.radio.stations.1.url": url, "tools.radio.stations.Jazz.genres": ["j"]},
    }
    (tmp_path / "s.json").write_text(json.dumps(old))
    (tmp_path / "s.json").chmod(0o600)
    stations = restarted(tmp_path, lower).effective().tools.radio.stations
    assert [(s.name, s.url, s.genres) for s in stations] == [
        ("Jazz", STATION_URL, ["j"]),
        (DOTTED, url, []),
    ]


def test_a_removal_that_removes_nothing_is_refused_and_keeps_the_file(tmp_path, monkeypatch):
    stack, editor = radio_editor(tmp_path)
    editor.add_entry("tools.radio.stations", None, station("Jazz"))
    before = (tmp_path / "s.json").read_text()
    monkeypatch.setattr(stack, "clear", lambda *a, **k: False)
    with pytest.raises(ValidationError) as exc:
        editor.remove_entry("tools.radio.stations.Jazz")
    assert exc.value.code == "entry_not_removed"
    assert (tmp_path / "s.json").read_text() == before
    assert "err_entry_not_removed" in STRINGS["en"] and "err_entry_not_removed" in STRINGS["it"]


def test_every_list_the_web_can_add_to_is_a_merged_list(tmp_path):
    """A list of entries outside ``KEYED_LISTS`` is replaced whole by a layer, so the stack cannot
    take one added entry out of it: no such list may gain an entry from the web UI."""
    _, editor = radio_editor(tmp_path)
    for row in editor.fields():
        if row["kind"] != "keyed_list" or row["path"] in KEYED_LISTS:
            continue
        with pytest.raises((ValidationError, GuardrailError, SettingNotAllowed)):
            editor.add_entry(row["path"], None, {"name": "probe"})


# ------------------------------------------------------- the station catalogue (SPEC 11, Radio)

CATALOGUE = "/api/tools/radio/catalogue"
STATION_KEYS = {
    "name",
    "url",
    "genres",
    "language",
    "operator",
    "policy_url",
    "official",
    "bitrate_kbps",
    "added",
}


def catalogued(data: dict) -> dict[str, dict]:
    return {s["name"]: s for g in data["groups"] for s in g["stations"]}


def test_the_catalogue_needs_the_session_and_is_not_a_poll(served):
    c = served.client
    assert CATALOGUE not in PASSIVE_POLLS
    assert c.request("GET", CATALOGUE).status == 401
    c.login()
    resp = c.request("GET", CATALOGUE)
    assert resp.status == 200
    data = resp.json
    assert [g["id"] for g in data["groups"]] == list(GROUP_IDS)
    for s in catalogued(data).values():
        assert set(s) == STATION_KEYS
        assert s["added"] is False
    assert STATION_URL not in resp.body.decode(), "a configured station's URL is never echoed"


def test_a_catalogue_station_shows_as_added_by_name_or_by_url():
    shipped = [s for g in radio_catalogue().groups for s in g.stations]
    by_name, by_url, other = shipped[0], shipped[1], shipped[2]
    stations = [
        {"name": by_name.name.upper(), "url": STATION_URL},
        {"name": "My own name", "url": by_url.url},
    ]
    cfg = RoboEmotConfig(tools={"radio": {"stations": stations}})
    host = SimpleNamespace(scheduler=None, media=None, circadian_runner=None)
    marks = catalogued(ToolsState(host, cfg).catalogue())
    assert marks[by_name.name]["added"] and marks[by_url.name]["added"]
    assert not marks[other.name]["added"]
    assert sum(s["added"] for s in marks.values()) == 2


def test_every_catalogue_station_adds_through_the_settings_route_once(tmp_path):
    """The Add button's gate is the settings entry route, not only the pydantic model."""
    stack, editor = radio_editor(tmp_path)
    shipped = [s for g in radio_catalogue().groups for s in g.stations]
    for s in shipped:
        assert editor.add_entry("tools.radio.stations", None, s.station())["changed"]
    assert names(stack) == [s.name for s in shipped]
    for s in shipped:
        with pytest.raises(ValidationError) as exc:
            editor.add_entry("tools.radio.stations", None, s.station())
        assert exc.value.code == "entry_exists"


def test_a_catalogue_station_is_added_and_removed_through_the_page_routes(served):
    c = served.client
    c.login()
    pick = next(iter(catalogued(c.request("GET", CATALOGUE).json).values()))
    value = {k: pick[k] for k in ("name", "url", "genres", "language")}
    body = {"path": "tools.radio.stations", "value": value, "pin": PIN}
    assert c.request("POST", "/api/settings/entry", body).status == 200
    assert catalogued(c.request("GET", CATALOGUE).json)[pick["name"]]["added"] is True
    assert names(served.stack) == ["Jazz", DOTTED, pick["name"]]
    again = c.request("POST", "/api/settings/entry", body)
    assert again.status == 422 and again.json["error"] == "entry_exists"
    rows = {r["path"]: r for r in c.request("GET", "/api/settings").json["fields"]}
    box = "tools.radio.stations"
    named = {rows[f"{box}.{k}.name"]["value"]: k for k in rows[box]["removable"]}
    key = named[value["name"]]
    gone = {"path": f"{box}.{key}", "pin": PIN}
    assert c.request("POST", "/api/settings/entry/remove", gone).status == 200
    assert names(served.stack) == ["Jazz", DOTTED]
    assert catalogued(c.request("GET", CATALOGUE).json)[pick["name"]]["added"] is False


def test_added_folds_names_like_the_add_route_shapes_them():
    """NFC, collapsed whitespace and case are folded on both sides of the compare."""
    first = radio_catalogue().groups[0].stations[0]
    spaced = "  ".join(first.name.upper().split(" ")) if " " in first.name else first.name.upper()
    cfg = RoboEmotConfig(tools={"radio": {"stations": [{"name": spaced, "url": STATION_URL}]}})
    host = SimpleNamespace(scheduler=None, media=None, circadian_runner=None)
    assert catalogued(ToolsState(host, cfg).catalogue())[first.name]["added"]
    assert folded_name("Cafe\u0301  Jazz") == folded_name("caf\u00e9 jazz")


def test_a_case_variant_shows_as_added_although_the_route_would_take_it(tmp_path):
    """Documented mismatch (SPEC 11, Radio): the mark is conservative, the route exact. A name
    that differs only in case is a new station for the route, and the catalogue marks it added."""
    first = radio_catalogue().groups[0].stations[0]
    stack, editor = radio_editor(tmp_path)
    editor.add_entry("tools.radio.stations", None, first.station())
    variant = {**first.station(), "name": first.name.lower(), "url": "https://example.org/v.mp3"}
    assert editor.add_entry("tools.radio.stations", None, variant)["changed"]
    assert names(stack) == [first.name, first.name.lower()]
    host = SimpleNamespace(scheduler=None, media=None, circadian_runner=None)
    assert catalogued(ToolsState(host, stack.effective()).catalogue())[first.name]["added"]


def test_a_broken_catalogue_is_logged_at_start_up_without_urls_and_the_page_gets_503(
    served, monkeypatch, caplog
):
    from roboemot.runtime import tools_state

    bad = {"groups": [{"id": "jazz", "stations": [{"name": "X", "url": STATION_URL}]}]}

    def broken():
        return parse_catalogue(bad)

    monkeypatch.setattr(tools_state, "radio_catalogue", broken)
    caplog.set_level(logging.DEBUG)
    assert check_catalogue() is False
    assert "radio station catalogue unusable (ValidationError)" in logged(caplog)
    assert STATION_URL not in logged(caplog)
    c = served.client
    c.login()
    resp = c.request("GET", CATALOGUE)
    assert resp.status == 503 and resp.json["error"] == "radio_catalogue_unavailable"
    assert STATION_URL not in logged(caplog)


def test_the_shipped_catalogue_passes_the_start_up_check():
    assert check_catalogue() is True
