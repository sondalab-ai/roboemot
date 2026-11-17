"""The weather voice tool through Open-Meteo (SPEC 11, Voice tools, Weather; SPEC 12).

``weather`` returns the current conditions and a daily forecast for ``circadian.location`` as
metric, structured JSON; the model phrases it. It is an admin opt-in (``tools.weather.enabled``)
and is available only with a location and an HTTP client (``ToolHost.http``). A result is cached
by ``(lat, lon, days)`` for ``tools.weather.cache_s``; a network error or a non-2xx answer gives
``{"error": "unreachable"}`` and is never cached. ``geocode`` is the place search of the Tools
tab. The request carries only the location and the number of days (``timezone=auto``: Open-Meteo
resolves the zone from the location). Logs carry the tool name, the status and the HTTP status
code only, never coordinates, a place, a query or a forecast; an httpx error's text may hold the
URL, so it is never logged.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any

import httpx

from ..config.schema import (
    FORECAST_DAYS_CEILING,
    GEOCODE_URL_DEFAULT,
    RoboEmotConfig,
)
from ..skills.host import ToolHost
from ..skills.registry import SkillResult, skill
from ..skills.schema import Desc, Range

log = logging.getLogger(__name__)
# httpx logs every request's full URL at INFO, coordinates included (SPEC 12).
logging.getLogger("httpx").setLevel(logging.WARNING)

__all__ = ["GEOCODE_URL_DEFAULT", "Forecast", "Place", "fetch_forecast", "geocode", "skills"]

CURRENT_FIELDS = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
DAILY_FIELDS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
TIMEZONE_AUTO = "auto"
GEOCODE_FORMAT = "json"
UNREACHABLE = "unreachable"
UNAVAILABLE = "unavailable"
UNKNOWN_CONDITIONS = "unknown"

# WMO weather interpretation codes as Open-Meteo documents them, in English for the model.
WMO_WORDS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

DESCRIPTION = (
    "Get the weather at the robot's home: the current conditions and a daily forecast (today "
    "first), in metric units. Phrase it for the person; ask for more days only when needed"
)
Days = Annotated[
    int, Range(1, FORECAST_DAYS_CEILING), Desc("Days of forecast, today included; 1 is today")
]


class ProviderError(Exception):
    """Open-Meteo could not be reached or answered with an error. ``status`` is the HTTP status
    code when there was an answer. Its message never holds the URL."""

    def __init__(self, status: int | None = None) -> None:
        super().__init__(UNREACHABLE if status is None else f"{UNREACHABLE} ({status})")
        self.status = status


@dataclass(frozen=True)
class Day:
    """One day of forecast: local date, conditions, temperatures in °C and the highest chance
    of precipitation in % (None when Open-Meteo has none for that day)."""

    date: str
    conditions: str
    max_c: float
    min_c: float
    precipitation_probability_pct: float | None


@dataclass(frozen=True)
class Forecast:
    """The current values (°C, %, km/h, conditions) and one :class:`Day` per forecast day."""

    temperature_c: float
    humidity_pct: float
    wind_kmh: float
    conditions: str
    days: tuple[Day, ...]

    def as_result(self) -> dict[str, Any]:
        """The skill's JSON body: ``current`` and ``days``, no location."""
        return {
            "current": {
                "temperature_c": self.temperature_c,
                "humidity_pct": self.humidity_pct,
                "wind_kmh": self.wind_kmh,
                "conditions": self.conditions,
            },
            "days": [asdict(d) for d in self.days],
        }


@dataclass(frozen=True)
class Place:
    """A place search result: name, region (``admin1``), country and coordinates."""

    name: str
    admin: str
    country: str
    lat: float
    lon: float


def conditions(code: int | None) -> str:
    """The English words for a WMO ``weather_code``; ``unknown`` for a code not in the table."""
    return WMO_WORDS.get(code, UNKNOWN_CONDITIONS) if code is not None else UNKNOWN_CONDITIONS


def coordinates(cfg: RoboEmotConfig) -> tuple[float, float] | None:
    """``circadian.location`` as ``(lat, lon)``, or None when it is unset or lacks either."""
    loc = cfg.circadian.location
    if not loc or "lat" not in loc or "lon" not in loc:
        return None
    return loc["lat"], loc["lon"]


def _get(client: httpx.Client, url: str, params: dict[str, Any], timeout_s: float) -> Any:
    """GET ``url`` and return the JSON body; :class:`ProviderError` on a transport error or a
    non-2xx answer. The httpx error is dropped (``from None``): its text may hold the URL."""
    try:
        response = client.get(url, params=params, timeout=timeout_s)
    except httpx.HTTPError:
        raise ProviderError() from None
    if not response.is_success:
        raise ProviderError(response.status_code)
    return response.json()


def fetch_forecast(
    client: httpx.Client, url: str, lat: float, lon: float, days: int, timeout_s: float
) -> Forecast:
    """The Open-Meteo forecast at ``(lat, lon)`` for ``days`` days (metric, the location's own
    time zone). Raises :class:`ProviderError` when it cannot get one."""
    body = _get(
        client,
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "current": CURRENT_FIELDS,
            "daily": DAILY_FIELDS,
            "forecast_days": days,
            "timezone": TIMEZONE_AUTO,
        },
        timeout_s,
    )
    now, daily = body["current"], body["daily"]
    rain = daily.get("precipitation_probability_max") or [None] * len(daily["time"])
    return Forecast(
        temperature_c=now["temperature_2m"],
        humidity_pct=now["relative_humidity_2m"],
        wind_kmh=now["wind_speed_10m"],
        conditions=conditions(now["weather_code"]),
        days=tuple(
            Day(date, conditions(code), high, low, chance)
            for date, code, high, low, chance in zip(
                daily["time"],
                daily["weather_code"],
                daily["temperature_2m_max"],
                daily["temperature_2m_min"],
                rain,
                strict=True,
            )
        ),
    )


def geocode(
    client: httpx.Client, url: str, query: str, language: str, count: int, timeout_s: float
) -> list[Place]:
    """At most ``count`` places matching ``query`` (names in ``language``), best first; empty
    when none matches. Raises :class:`ProviderError` when the search cannot run."""
    body = _get(
        client,
        url,
        {"name": query, "count": count, "language": language, "format": GEOCODE_FORMAT},
        timeout_s,
    )
    return [
        Place(r["name"], r.get("admin1", ""), r.get("country", ""), r["latitude"], r["longitude"])
        for r in body.get("results") or []
    ]


def skills(host: ToolHost, clock: Callable[[], float] = time.monotonic) -> list[Callable[..., Any]]:
    """The weather tool over ``host``: one skill, ``weather``, available while
    ``tools.weather.enabled`` is on, ``circadian.location`` is set and the run has an HTTP client
    (all read on every check). ``clock`` is the cache's monotonic clock. Side-effect free: it only
    builds the skill, so the realtime check may call it with a throwaway host."""
    cache: dict[tuple[float, float, int], tuple[float, dict[str, Any]]] = {}
    lock = threading.Lock()

    def available() -> bool:
        cfg = host.config()
        return cfg.tools.weather.enabled and coordinates(cfg) is not None and host.http is not None

    def cached(key: tuple[float, float, int], ttl_s: float) -> dict[str, Any] | None:
        with lock:
            hit = cache.get(key)
            return hit[1] if hit is not None and clock() - hit[0] < ttl_s else None

    def keep(key: tuple[float, float, int], ttl_s: float, body: dict[str, Any]) -> None:
        with lock:
            now = clock()
            for old in [k for k, (at, _) in cache.items() if now - at >= ttl_s]:
                del cache[old]
            if ttl_s > 0:
                cache[key] = (now, body)

    @skill(llm_visible=True, description=DESCRIPTION, available=available)
    def weather(days: Days = 1) -> SkillResult:
        cfg = host.config()
        where, factory = coordinates(cfg), host.http
        if not cfg.tools.weather.enabled or where is None or factory is None:
            log.info("weather: %s", UNAVAILABLE)
            return SkillResult({"error": UNAVAILABLE})
        wcfg = cfg.tools.weather
        span = min(days, wcfg.forecast_days)
        key = (*where, span)
        body = cached(key, wcfg.cache_s)
        if body is not None:
            log.info("weather: cached")
            return SkillResult(body)
        try:
            with factory() as client:
                forecast = fetch_forecast(client, wcfg.base_url, *where, span, wcfg.timeout_s)
        except ProviderError as e:
            log.info("weather: %s", e)
            return SkillResult({"error": UNREACHABLE})
        body = forecast.as_result()
        keep(key, wcfg.cache_s, body)
        log.info("weather: ok")
        return SkillResult(body)

    return [weather]
