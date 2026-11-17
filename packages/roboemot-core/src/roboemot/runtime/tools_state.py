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
``geocode``, the admin's place search. A test play needs ``tools.radio.enabled`` like the
``radio_play`` skill, so no stream leaves the robot before the admin opt-in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from roboemot.config.schema import RoboEmotConfig
from roboemot.skills.host import http_client
from roboemot.tools.schedule import Item
from roboemot.tools.timers import LOCAL_FORMAT
from roboemot.tools.weather import BadResponse, ProviderError, coordinates, geocode
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


class ToolsError(Exception):
    """A write was refused; ``code`` is the page's error code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ToolsState:
    """Builds the Tools tab's payload and runs its writes. ``host`` is the running
    ``RuntimeHost`` (``scheduler``, ``media`` and ``circadian_runner`` are read on every call,
    each may be None); ``cfg`` the configuration, kept current by ``apply_config``; ``http``
    builds the HTTP client of one place search (the caller closes it)."""

    def __init__(
        self,
        host: RuntimeHost,
        cfg: RoboEmotConfig,
        *,
        http: Callable[[], httpx.Client] = http_client,
    ) -> None:
        self._host, self._cfg, self._http = host, cfg, http

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
        """Each tool's switch and whether the model may use it now (its skills' ``available``)."""
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
        budget = cfg.webui.request_deadline_s * cfg.webui.request_work_share
        return str(media.play(station, min(cfg.tools.media.connect_timeout_s, budget)))

    def geocode(self, query: str) -> list[dict[str, Any]]:
        """The admin's place search through ``tools.weather.geocode_url``, names in the page
        language: at most ``tools.weather.geocode_results`` places, best first.
        ``geocode_unreachable`` when the search cannot run, ``geocode_failed`` for an answer it
        cannot read. Logged by type only."""
        cfg = self._cfg
        weather = cfg.tools.weather
        try:
            with self._http() as client:
                places = geocode(
                    client,
                    url=weather.geocode_url,
                    query=query,
                    language=language_for(cfg.conversation.language),
                    count=weather.geocode_results,
                    timeout_s=weather.timeout_s,
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
