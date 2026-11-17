"""The weather voice tool through Open-Meteo (SPEC 11, Voice tools, Weather; SPEC 12).

``weather`` returns the current conditions and a daily forecast for ``circadian.location`` as
metric, structured JSON; the model phrases it. It is an admin opt-in (``tools.weather.enabled``)
and is available with an HTTP client (``ToolHost.http``). Without a location it is still offered
(owner decision 2026-11-17): it returns ``{"error": "not_configured"}`` and sends no request, so
the robot tells the person to set the place in the web UI's Tools tab. A result is cached
by ``(url, lat, lon, days)`` for ``tools.weather.cache_s``; a network error or a non-2xx answer
(a redirect included: none is followed) gives ``{"error": "unreachable"}``, and a malformed or
oversized answer ``{"error": "failed"}``; an error is never cached. Only numbers checked as
numbers reach the model. ``geocode`` is the place search of the Tools tab. The request carries
only the location and the number of days (``timezone=auto``: Open-Meteo resolves the zone from
the location). Logs carry the tool name, the status and the HTTP status
code only, never coordinates, a place, a query or a forecast; an httpx error's text may hold the
URL, so it is never logged.
"""

from __future__ import annotations

import json
import logging
import math
import re
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

__all__ = [
    "GEOCODE_URL_DEFAULT",
    "BadResponse",
    "Forecast",
    "Place",
    "fetch_forecast",
    "geocode",
    "skills",
]

CURRENT_FIELDS = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
DAILY_FIELDS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
TIMEZONE_AUTO = "auto"
GEOCODE_FORMAT = "json"
UNREACHABLE = "unreachable"
UNAVAILABLE = "unavailable"
FAILED = "failed"
NOT_CONFIGURED = "not_configured"
# A 7-day forecast is about 2 KB and 10 places about 4 KB: a larger body is not Open-Meteo's.
MAX_BODY_BYTES = 65536
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
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
    "first), in metric units. Phrase it for the person; ask for more days only when needed. "
    "The error not_configured means the robot's place is not set yet: tell the person to set "
    "it in the Tools tab of the robot's web UI"
)
Days = Annotated[
    int, Range(1, FORECAST_DAYS_CEILING), Desc("Days of forecast, today included; 1 is today")
]


class BadResponse(Exception):
    """Open-Meteo answered 2xx with a body that is too large, not JSON, or not the expected
    shape and types. The skill gives ``failed``."""

    def __init__(self) -> None:
        super().__init__(FAILED)


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


def _number(value: Any, nullable: bool = False) -> float | None:
    """``value`` when it is a finite JSON number (None too when ``nullable``), else
    :class:`BadResponse`: only checked numbers reach the model."""
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BadResponse()
    return value


def _text(value: Any, pattern: re.Pattern[str] | None = None) -> str:
    """``value`` when it is a string (matching ``pattern`` when given), else
    :class:`BadResponse`."""
    if not isinstance(value, str) or (pattern is not None and not pattern.fullmatch(value)):
        raise BadResponse()
    return value


def _code(value: Any) -> int:
    """A WMO code: a JSON integer, else :class:`BadResponse`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadResponse()
    return value


def _get(client: httpx.Client, url: str, params: dict[str, Any], timeout_s: float) -> Any:
    """GET ``url`` and return the JSON body. :class:`ProviderError` on a transport error or an
    answer that is not 2xx (a redirect is never followed); :class:`BadResponse` for a body over
    ``MAX_BODY_BYTES`` or one that is not JSON. The httpx error is dropped (``from None``): its
    text may hold the URL."""
    try:
        with client.stream(
            "GET", url, params=params, timeout=timeout_s, follow_redirects=False
        ) as response:
            if not response.is_success:
                raise ProviderError(response.status_code)
            data = bytearray()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_BODY_BYTES:
                    raise BadResponse()
    except httpx.HTTPError:
        raise ProviderError() from None
    try:
        return json.loads(data)
    except ValueError:
        raise BadResponse() from None


def _forecast(body: Any) -> Forecast:
    """The :class:`Forecast` in an Open-Meteo answer; :class:`BadResponse` for any other
    shape or a value of the wrong type."""
    try:
        now, daily = body["current"], body["daily"]
        dates = daily["time"]
        rain = daily.get("precipitation_probability_max") or [None] * len(dates)
        rows = zip(
            dates,
            daily["weather_code"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            rain,
            strict=True,
        )
        return Forecast(
            temperature_c=_number(now["temperature_2m"]),
            humidity_pct=_number(now["relative_humidity_2m"]),
            wind_kmh=_number(now["wind_speed_10m"]),
            conditions=conditions(_code(now["weather_code"])),
            days=tuple(
                Day(
                    _text(date, DATE_PATTERN),
                    conditions(_code(code)),
                    _number(high),
                    _number(low),
                    _number(chance, nullable=True),
                )
                for date, code, high, low, chance in rows
            ),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        raise BadResponse() from None


def fetch_forecast(
    client: httpx.Client, url: str, lat: float, lon: float, days: int, timeout_s: float
) -> Forecast:
    """The Open-Meteo forecast at ``(lat, lon)`` for ``days`` days (metric, the location's own
    time zone). Raises :class:`ProviderError` when it cannot get one and :class:`BadResponse`
    when the answer is malformed or too large."""
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
    return _forecast(body)


def geocode(
    client: httpx.Client, url: str, query: str, language: str, count: int, timeout_s: float
) -> list[Place]:
    """At most ``count`` places matching ``query`` (names in ``language``), best first; empty
    when none matches. Raises :class:`ProviderError` when the search cannot run and
    :class:`BadResponse` when the answer is malformed or too large."""
    body = _get(
        client,
        url,
        {"name": query, "count": count, "language": language, "format": GEOCODE_FORMAT},
        timeout_s,
    )
    try:
        return [
            Place(
                _text(r["name"]),
                _text(r.get("admin1", "")),
                _text(r.get("country", "")),
                _number(r["latitude"]),
                _number(r["longitude"]),
            )
            for r in body.get("results") or []
        ]
    except (KeyError, TypeError, AttributeError):
        raise BadResponse() from None


def skills(host: ToolHost, clock: Callable[[], float] = time.monotonic) -> list[Callable[..., Any]]:
    """The weather tool over ``host``: one skill, ``weather``, available while
    ``tools.weather.enabled`` is on and the run has an HTTP client (both read on every check).
    Without ``circadian.location`` it answers ``not_configured`` and sends nothing. ``clock`` is
    the cache's monotonic clock. Side-effect free: it only builds the skill, so the realtime
    check may call it with a throwaway host."""
    cache: dict[tuple[str, float, float, int], tuple[float, dict[str, Any]]] = {}
    lock = threading.Lock()

    def available() -> bool:
        cfg = host.config()
        return cfg.tools.weather.enabled and host.http is not None

    def cached(key: tuple[str, float, float, int], ttl_s: float) -> dict[str, Any] | None:
        with lock:
            hit = cache.get(key)
            return hit[1] if hit is not None and clock() - hit[0] < ttl_s else None

    def keep(key: tuple[str, float, float, int], ttl_s: float, body: dict[str, Any]) -> None:
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
        if not cfg.tools.weather.enabled or factory is None:
            log.info("weather: %s", UNAVAILABLE)
            return SkillResult({"error": UNAVAILABLE})
        if where is None:
            log.info("weather: %s", NOT_CONFIGURED)
            return SkillResult({"error": NOT_CONFIGURED})
        wcfg = cfg.tools.weather
        span = min(days, wcfg.forecast_days)
        key = (wcfg.base_url, *where, span)
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
        except BadResponse:
            log.info("weather: %s", FAILED)
            return SkillResult({"error": FAILED})
        body = forecast.as_result()
        keep(key, wcfg.cache_s, body)
        log.info("weather: ok")
        return SkillResult(body)

    return [weather]
