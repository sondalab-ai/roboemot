"""SPEC 11 (Voice tools, Weather) and SPEC 12 (network tools are admin opt-ins, no location in
logs): the ``weather`` skill over Open-Meteo, its cache, its errors, and ``geocode``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
from roboemot.bus import Bus
from roboemot.config.schema import FORECAST_URL_DEFAULT, RoboEmotConfig
from roboemot.config.stack import ConfigStack, GuardrailError
from roboemot.redact import is_private
from roboemot.skills import SkillContext
from roboemot.skills.host import tool_host
from roboemot.skills.plugins import load_plugin_skill_sets
from roboemot.skills.registry import SkillRegistry, execute
from roboemot.tools import weather
from roboemot.webui.settings import SettingsEditor
from test_webui_settings import put, serve

FIXTURE = Path(__file__).parent / "data" / "weather" / "forecast.json"
LAT, LON = 45.4642, 9.1900
PLACE = "Pontedilegno"
CTX = SkillContext(speaker="p1", language="it", now=10.0)
EXEC = {"timeout_s": 1.0, "max_chars": 50, "enum_max": 20}


def config(enabled=True, location=True, timezone=None, **weather_cfg) -> RoboEmotConfig:
    return RoboEmotConfig.model_validate(
        {
            "circadian": {
                "location": {"lat": LAT, "lon": LON} if location else None,
                "timezone": timezone,
            },
            "tools": {"weather": {"enabled": enabled, "place_name": PLACE, **weather_cfg}},
        }
    )


class Rig:
    """A tool host over a config that a test can swap, an Open-Meteo stand-in that counts the
    requests it gets, and a clock the cache reads."""

    def __init__(self, cfg=None, handler=None):
        self.cfg = cfg if cfg is not None else config()
        self.requests: list[httpx.Request] = []
        self.now = 0.0
        self.handler = handler or (
            lambda r: httpx.Response(200, json=json.loads(FIXTURE.read_text()))
        )
        self.host = tool_host(lambda: self.cfg, Bus())
        self.host.http = lambda: httpx.Client(transport=httpx.MockTransport(self._serve))
        (self.fn,) = weather.skills(self.host, clock=lambda: self.now)

    def _serve(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def call(self, **kwargs):
        return self.fn(**kwargs).content


def test_weather_not_offered_when_disabled_or_no_location():
    rig = Rig(config(enabled=False))
    s = rig.fn.skill
    assert s.name == "weather" and s.llm_visible
    assert not s.is_available()
    rig.cfg = config(enabled=True, location=False)  # read live, never at factory time
    assert not s.is_available()
    rig.cfg = config()
    assert s.is_available()
    rig.host.http = None  # the realtime check's throwaway host has no HTTP client
    assert not s.is_available()


def test_weather_is_off_by_default_and_an_admin_opt_in():
    cfg = RoboEmotConfig()
    assert cfg.tools.weather.enabled is False and cfg.tools.weather.place_name == ""
    assert "weather" in cfg.conversation.llm_skills
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("tools.weather.enabled", True)
    stack.set("tools.weather.enabled", True, admin=True)
    assert stack.get("tools.weather.enabled") is True


def test_weather_timeout_must_be_below_the_tool_timeout():
    cfg = RoboEmotConfig()
    assert cfg.tools.weather.timeout_s == 3.0 < cfg.conversation.tool_timeout_s
    config()  # the defaults with weather on load
    with pytest.raises(ValueError, match="tool_timeout_s"):
        RoboEmotConfig.model_validate(
            {
                "conversation": {"tool_timeout_s": 5},
                "tools": {"weather": {"enabled": True, "timeout_s": 5}},
            }
        )
    # with weather off the tool cannot run: a short tool timeout stays allowed
    RoboEmotConfig.model_validate({"conversation": {"tool_timeout_s": 0.1}})


@pytest.mark.parametrize("field", ["base_url", "geocode_url"])
@pytest.mark.parametrize(
    "url",
    [
        "http://api.open-meteo.com/v1/forecast",  # not https
        "https:///v1/forecast",  # no host
        "https://user:pw@api.open-meteo.com/v1/forecast",  # user info
        "https://api.open-meteo.com/v1/forecast?apikey=x",  # query
        "https://api.open-meteo.com/v1/forecast#frag",  # fragment
        "https://api.open-meteo.com/v1/ forecast",  # a space
        "ftp://api.open-meteo.com/v1/forecast",
    ],
)
def test_weather_urls_must_be_plain_https(field, url):
    with pytest.raises(ValueError):
        RoboEmotConfig.model_validate({"tools": {"weather": {field: url}}})


@pytest.mark.parametrize("field", ["base_url", "geocode_url"])
def test_weather_urls_are_admin_only(field):
    path = f"tools.weather.{field}"
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set(path, "https://example.org/v1/forecast")
    stack.set(path, "https://example.org/v1/forecast", admin=True)
    assert stack.get(path) == "https://example.org/v1/forecast"


def test_the_web_ui_refuses_a_forecast_url_that_is_not_plain_https(tmp_path):
    """SPEC 12: the forecast URL receives the location. The settings page writes it only with
    the PIN re-entered (one PIN covers operator and admin: ``test_webui_settings`` pins the
    wrong-PIN refusal), and the load check refuses a URL that is not plain https."""
    stack = ConfigStack(app={"webui": {"bind": "127.0.0.1", "port": 0}})
    srv, client = serve(tmp_path, SettingsEditor(stack, store=tmp_path / "s.json"), stack)
    path = "tools.weather.base_url"
    try:
        assert put(client, path, "http://example.org/v1/forecast").status == 422
        assert put(client, path, "https://example.org/v1/forecast?k=1").status == 422
        assert stack.get(path) == FORECAST_URL_DEFAULT
        assert put(client, path, "https://example.org/v1/forecast").status == 200
        assert stack.get(path) == "https://example.org/v1/forecast"
    finally:
        srv.stop()


def test_place_name_is_private():
    assert is_private("tools.weather.place_name")
    assert not is_private("tools.weather.enabled")


def test_core_registers_the_weather_plugin_and_the_realtime_host_hides_it():
    host = tool_host(lambda: config(), Bus())
    plugins = {p.name: p for p in load_plugin_skill_sets(host)}
    assert plugins["weather"].distribution == "roboemot-core"
    assert [fn.skill.name for fn in plugins["weather"].skills] == ["weather"]
    assert not plugins["weather"].skills[0].skill.is_available()  # http is None


def test_weather_parses_open_meteo_payload():
    rig = Rig()
    out = rig.call(days=3)
    assert out["current"] == {
        "temperature_c": 8.4,
        "humidity_pct": 87,
        "wind_kmh": 6.5,
        "conditions": "slight rain",
    }
    assert out["days"][0] == {
        "date": "2026-11-17",
        "conditions": "slight rain",
        "max_c": 11.2,
        "min_c": 6.9,
        "precipitation_probability_pct": 80,
    }
    assert [d["conditions"] for d in out["days"]] == ["slight rain", "overcast", "clear sky"]
    assert out["days"][2]["precipitation_probability_pct"] is None
    text = json.dumps(out)
    assert PLACE not in text and "45.46" not in text and "9.18" not in text


def test_the_request_carries_only_the_location_and_the_days():
    rig = Rig(config(timezone="Europe/Rome"))  # never sent: Open-Meteo resolves it (auto)
    rig.call(days=2)
    (req,) = rig.requests
    assert str(req.url).startswith("https://api.open-meteo.com/v1/forecast?")
    assert dict(req.url.params) == {
        "latitude": str(LAT),
        "longitude": str(LON),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "forecast_days": "2",
        "timezone": "auto",
    }


def test_days_are_clamped_to_forecast_days():
    rig = Rig(config(forecast_days=3))
    rig.call(days=7)
    assert rig.requests[0].url.params["forecast_days"] == "3"


def test_unknown_wmo_code_has_a_fixed_word():
    assert weather.WMO_WORDS[61] == "slight rain"
    assert weather.conditions(1234) == weather.UNKNOWN_CONDITIONS


def test_weather_cache_avoids_second_call():
    rig = Rig(config(cache_s=600))
    first = rig.call(days=1)
    rig.now = 599.0
    assert rig.call(days=1) == first
    assert len(rig.requests) == 1
    rig.call(days=2)  # another key
    assert len(rig.requests) == 2
    rig.now = 601.0
    rig.call(days=1)
    assert len(rig.requests) == 3


def test_cache_zero_always_asks():
    rig = Rig(config(cache_s=0))
    rig.call(days=1)
    rig.call(days=1)
    assert len(rig.requests) == 2


def test_weather_unreachable_returns_error_code():
    def down(request):
        raise httpx.ConnectError("no route", request=request)

    rig = Rig(handler=down)
    assert rig.call(days=1) == {"error": "unreachable"}
    rig.handler = lambda r: httpx.Response(503)
    assert rig.call(days=1) == {"error": "unreachable"}
    rig.handler = lambda r: httpx.Response(200, json=json.loads(FIXTURE.read_text()))
    assert "current" in rig.call(days=1)  # an error is never cached
    assert len(rig.requests) == 3


def test_the_timeout_reaches_the_request():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json=json.loads(FIXTURE.read_text()))

    Rig(config(timeout_s=4), handler=handler).call(days=1)
    assert seen[0]["connect"] == 4 and seen[0]["read"] == 4


def test_logs_never_carry_coordinates(caplog):
    def down(request):
        raise httpx.ConnectError(f"failed {request.url}", request=request)

    with caplog.at_level(logging.DEBUG):
        rig = Rig()
        rig.call(days=1)
        rig.handler = lambda r: httpx.Response(502)
        rig.now = 10_000.0
        rig.call(days=1)
        rig.handler = down
        rig.now = 20_000.0
        rig.call(days=1)
    text = caplog.text
    # httpx logs each request's full URL at INFO: it must stay quiet in this process.
    assert logging.getLogger("httpx").getEffectiveLevel() > logging.INFO
    assert "weather" in text and "502" in text and "unreachable" in text
    for secret in (str(LAT), str(LON), "45.46", "9.18", PLACE, "latitude", "slight rain"):
        assert secret not in text


async def test_disabling_mid_session_refuses_in_flight_call():
    rig = Rig()
    registry = SkillRegistry([rig.fn])
    assert [s.name for s in registry.offered(["weather"])] == ["weather"]
    rig.cfg = config(enabled=False)  # the admin turns it off after the request was built
    assert registry.offered(["weather"]) == []
    out = await execute(registry, ["weather"], "weather", {"days": 1}, CTX, **EXEC)
    assert out.code == "unavailable" and not out.ok
    assert rig.call(days=1) == {"error": "unavailable"}  # the skill checks again itself
    assert rig.requests == []


def test_geocode_parses_places_and_sends_the_count():
    payload = {
        "results": [
            {
                "id": 3173435,
                "name": "Milano",
                "latitude": 45.46427,
                "longitude": 9.18951,
                "country": "Italia",
                "admin1": "Lombardia",
            },
            {"id": 1, "name": "Milano Marittima", "latitude": 44.27, "longitude": 12.35},
        ],
        "generationtime_ms": 0.5,
    }
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        places = weather.geocode(
            client, weather.GEOCODE_URL_DEFAULT, "Milano", "it", count=5, timeout_s=5
        )
    assert places == [
        weather.Place("Milano", "Lombardia", "Italia", 45.46427, 9.18951),
        weather.Place("Milano Marittima", "", "", 44.27, 12.35),
    ]
    assert dict(seen[0].url.params) == {
        "name": "Milano",
        "count": "5",
        "language": "it",
        "format": "json",
    }


def test_geocode_with_no_match_is_empty():
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"generationtime_ms": 1}))
    ) as client:
        assert weather.geocode(client, weather.GEOCODE_URL_DEFAULT, "zzz", "en", 5, 5) == []
