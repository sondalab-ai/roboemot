"""The web UI's Tools tab: the voice tools' state and its writes (SPEC 11, Voice tools; 11A; 12).

``ToolsState.state()`` is the tab's poll: each tool with its switch and availability, the active
timers and reminders (id, kind, text, local due time, repetition), the missed items and the
station that plays with its volume. Sleep and the scheduler have no ``enabled`` field: their
switch is ``conversation.llm_skills`` (all of the tool's skills listed). A part the run lacks
(no scheduler without a listener, no player without a media-capable speaker) shows as an
unavailable tool and an empty list, never as an error.

Privacy (SPEC 12): the state holds the station name only, never its URL, and never the place
name or the location. It does hold the reminder texts and timer labels, because it is served
behind the PIN session only (SPEC 11A). Nothing here logs a text, a label, a place, a query or a
URL; an error is a ``ToolsError`` code, and a failed place search is logged by type only (the
HTTP error's text quotes the URL with the query).

The writes (the API checks the PIN session, CSRF and the PIN first): ``cancel`` and
``clear_missed`` on the scheduler, ``radio_stop`` and ``radio_test`` on the player, and
``geocode``, the admin's place search. ``catalogue`` is read-only: the shipped station
catalogue with the stations already added marked (SPEC 11, Radio). A test play needs
``tools.radio.enabled`` like the ``radio_play`` skill, so no stream leaves the robot before the
admin opt-in.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roboemot.config.schema import RoboEmotConfig
from roboemot.skills.host import http_client
from roboemot.tools.radio_catalogue import radio_catalogue
from roboemot.tools.schedule import Item
from roboemot.tools.timers import LOCAL_FORMAT
from roboemot.tools.weather import BadResponse, ProviderError, coordinates, geocode
from roboemot.webui.api import ToolsError
from roboemot.webui.strings import language_for

if TYPE_CHECKING:
    import httpx

    from .host import RuntimeHost

log = logging.getLogger(__name__)

# The voice tools of the tab, in card order, with the skills that make up each one (SPEC 11).
TOOL_SKILLS: dict[str, tuple[str, ...]] = {
    "sleep": ("go_to_sleep",),
    "scheduler": ("timer_set", "timer_cancel", "reminder_set", "reminder_cancel", "schedule_list"),
    "weather": ("weather",),
    "radio": ("radio_play", "radio_stop", "radio_volume"),
}
STOP_REASON = "webui"  # ``media.stopped``'s reason for the tab's Stop button


class ToolsState:
    """Builds the Tools tab's payload and runs its writes. ``host`` is the running
    ``RuntimeHost`` (``scheduler``, ``media`` and ``circadian_runner`` are read on every call,
    each may be None); ``cfg`` the configuration, kept current by ``apply_config``; ``http``
    builds the HTTP client of one place search (the caller closes it); ``clock`` is the
    monotonic clock of the minimum interval between two place searches or two test plays."""

    def __init__(
        self,
        host: RuntimeHost,
        cfg: RoboEmotConfig,
        *,
        http: Callable[[], httpx.Client] = http_client,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host, self._cfg, self._http, self._clock = host, cfg, http, clock
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}  # when each outbound action last started

    def _budget_s(self) -> float:
        """How long the work of one page request may take (``webui.request_deadline_s`` times
        ``webui.request_work_share``), so an outbound wait never outlives the request."""
        web = self._cfg.webui
        return web.request_deadline_s * web.request_work_share

    def _pace(self, action: str) -> None:
        """Start ``action`` (``geocode`` or ``radio_test``) only ``webui.tools.min_interval_s``
        after its last start; ``tools_too_soon`` with the wait left otherwise."""
        gap = self._cfg.webui.tools.min_interval_s
        with self._lock:
            now = self._clock()
            last = self._last.get(action)
            if last is not None and now - last < gap:
                raise ToolsError("tools_too_soon", retry_after=gap - (now - last))
            self._last[action] = now

    def apply_config(self, change: Any) -> None:
        """``ConfigStack`` subscriber: keep the settings this view reads (hot reload)."""
        self._cfg = change.config

    # ----------------------------------------------------------------- state

    def state(self) -> dict[str, Any]:
        """The tab's payload: ``poll_ms``, ``tools``, ``items``, ``missed`` and ``playing``."""
        cfg = self._cfg
        sched = self._host.scheduler
        media = self._host.media
        station = media.now_playing() if media is not None else None
        return {
            "poll_ms": cfg.webui.tools.poll_ms,
            "tools": self._tools(cfg),
            "items": [self._item(i) for i in sched.items()] if sched is not None else [],
            "missed": [self._item(i) for i in sched.missed()] if sched is not None else [],
            "playing": (
                None if station is None else {"name": station.name, "volume_db": media.volume_db()}
            ),
        }

    def _tools(self, cfg: RoboEmotConfig) -> list[dict[str, Any]]:
        """Each tool's switch and whether it is ready: the run has its part and, for weather and
        radio, it is set up (a place, a station). A tool that is on but not set up is still
        offered to the model, which answers ``not_configured`` (owner decision 2026-11-17), so
        here it stays not available and the tab keeps its hint ("Set a place first")."""
        listed = set(cfg.conversation.llm_skills)
        host = self._host
        switch = {name: set(skills) <= listed for name, skills in TOOL_SKILLS.items()}
        switch["weather"] = cfg.tools.weather.enabled
        switch["radio"] = cfg.tools.radio.enabled
        has = {
            "sleep": host.circadian_runner is not None,
            "scheduler": host.scheduler is not None,
            "weather": coordinates(cfg) is not None,
            "radio": bool(cfg.tools.radio.stations) and host.media is not None,
        }
        return [
            {"name": name, "enabled": switch[name], "available": switch[name] and has[name]}
            for name in TOOL_SKILLS
        ]

    def catalogue(self) -> dict[str, Any]:
        """The shipped station catalogue (SPEC 11, Radio) for the tab's Add buttons: ``groups``,
        each ``{id, stations}``, every station with its fields and ``added`` when
        ``tools.radio.stations`` already holds a station of that name (case ignored) or URL.
        Served behind the PIN session, like the station editor's rows; the configured stations'
        own URLs are compared here and never returned."""
        stations = self._cfg.tools.radio.stations
        names = {s.name.casefold() for s in stations}
        urls = {s.url for s in stations}
        return {
            "groups": [
                {
                    "id": group.id,
                    "stations": [
                        {
                            **s.model_dump(),
                            "added": s.name.casefold() in names or s.url in urls,
                        }
                        for s in group.stations
                    ],
                }
                for group in radio_catalogue().groups
            ]
        }

    def _item(self, item: Item) -> dict[str, Any]:
        """One timer or reminder as the tab lists it, due time in ``circadian.timezone``."""
        tz = self._host.scheduler.tz()
        return {
            "id": item.id,
            "kind": item.kind,
            "text": item.text,
            "due_local": item.due.astimezone(tz).strftime(LOCAL_FORMAT),
            "repeat": item.repeat,
            "days": list(item.days),
        }

    # ----------------------------------------------------------------- writes

    def _scheduler(self) -> Any:
        sched = self._host.scheduler
        if sched is None:
            raise ToolsError("scheduler_unavailable")
        return sched

    def _media(self) -> Any:
        media = self._host.media
        if media is None:
            raise ToolsError("radio_unavailable")
        return media

    def cancel(self, item_id: str) -> bool:
        """Cancel the active item ``item_id``; False when there is none."""
        return bool(self._scheduler().cancel(item_id))

    def clear_missed(self) -> None:
        """Forget the missed items."""
        self._scheduler().clear_missed()

    def radio_stop(self) -> None:
        """Stop what plays (nothing when nothing does)."""
        self._media().stop(STOP_REASON)

    def radio_test(self, name: str) -> str:
        """Play the listed station named exactly ``name`` (no genre match) and return the
        player's status (``ok``, ``unreachable``, ``disabled``). ``radio_disabled`` while
        ``tools.radio.enabled`` is off, ``unknown_station`` for a name not in the list. The wait
        for the first audio stays within the page request's work budget
        (``webui.request_deadline_s`` times ``webui.request_work_share``), so the answer is sent."""
        media = self._media()
        cfg = self._cfg
        if not cfg.tools.radio.enabled:
            raise ToolsError("radio_disabled")
        station = next((s for s in cfg.tools.radio.stations if s.name == name), None)
        if station is None:
            raise ToolsError("unknown_station")
        self._pace("radio_test")
        return str(media.play(station, min(cfg.tools.media.connect_timeout_s, self._budget_s())))

    def geocode(self, query: str) -> list[dict[str, Any]]:
        """The admin's place search through ``tools.weather.geocode_url``, names in the page
        language: at most ``tools.weather.geocode_results`` places, best first.
        ``geocode_unreachable`` when the search cannot run, ``geocode_failed`` for an answer it
        cannot read. Logged by type only."""
        cfg = self._cfg
        weather = cfg.tools.weather
        self._pace("geocode")
        try:
            with self._http() as client:
                places = geocode(
                    client,
                    url=weather.geocode_url,
                    query=query,
                    language=language_for(cfg.conversation.language),
                    count=weather.geocode_results,
                    timeout_s=min(weather.timeout_s, self._budget_s()),
                )
        except ProviderError as exc:
            log.info("tools tab: place search unreachable (%s)", type(exc).__name__)
            raise ToolsError("geocode_unreachable") from None
        except BadResponse as exc:
            log.info("tools tab: place search failed (%s)", type(exc).__name__)
            raise ToolsError("geocode_failed") from None
        return [
            {"name": p.name, "admin": p.admin, "country": p.country, "lat": p.lat, "lon": p.lon}
            for p in places
        ]
