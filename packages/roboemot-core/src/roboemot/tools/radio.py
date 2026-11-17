"""The radio voice tools (SPEC 11, Voice tools, Radio; SPEC 12).

``radio_play(station)`` plays a station of ``tools.radio.stations`` through the run's media
player (``ToolHost.media``). The model picks a station name or a genre from ``choices``; a genre
plays the first station that lists it. The model never passes a URL: only a listed URL plays.
``radio_stop`` stops it and ``radio_volume(change)`` steps the music volume up or down by
``tools.media.volume_step_db``. ``radio_play`` is available only while ``tools.radio.enabled``
is on, the station list is not empty and the run has a player (read on every check);
``radio_stop`` and ``radio_volume`` are available then too, and also whenever something plays,
so "stop the radio" always has a tool. A stream that does not start gives
``{"error": "unreachable"}``; the player stops on a later drop by itself. The connect wait stays
``TOOL_TIMEOUT_MARGIN_S`` below ``conversation.tool_timeout_s``, so the model is never told
``timeout`` about a stream that then plays. Logs carry the skill, the status and the station
name only, never a URL.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal

from ..config.schema import RoboEmotConfig, Station
from ..skills.host import ToolHost
from ..skills.registry import SkillResult, skill
from ..skills.schema import Desc

log = logging.getLogger(__name__)

PLAY_DESCRIPTION = (
    "Play internet radio through the robot's speaker: pick a station by its name, or a genre "
    "to play the first station of that genre. The music pauses while you speak"
)
STOP_DESCRIPTION = "Stop the radio that is playing"
VOLUME_DESCRIPTION = "Make the radio music louder (up) or quieter (down) by one step"
StationArg = Annotated[str, Desc("A station name or a genre from the list")]
Change = Annotated[Literal["up", "down"], Desc("up: louder; down: quieter")]
UNAVAILABLE = "unavailable"
NOT_PLAYING = "not_playing"
NOT_FOUND = "not_found"
# How far below conversation.tool_timeout_s the connect wait ends: the skill returns its own
# result before the executor abandons it (a margin for the thread hand-off, not a behaviour).
TOOL_TIMEOUT_MARGIN_S = 0.5


def names_and_genres(cfg: RoboEmotConfig) -> list[str]:
    """The ``radio_play`` choices: every station name, then every genre, each once, in the
    order of the list."""
    out: list[str] = []
    for station in cfg.tools.radio.stations:
        if station.name not in out:
            out.append(station.name)
    for station in cfg.tools.radio.stations:
        out.extend(g for g in station.genres if g not in out)
    return out


def pick(cfg: RoboEmotConfig, wanted: str) -> Station | None:
    """The station named ``wanted``, else the first station that lists ``wanted`` as a genre,
    else None."""
    stations = cfg.tools.radio.stations
    named = next((s for s in stations if s.name == wanted), None)
    return named or next((s for s in stations if wanted in s.genres), None)


def skills(host: ToolHost) -> list[Callable[..., Any]]:
    """The radio tools over ``host``: ``radio_play``, ``radio_stop`` and ``radio_volume``,
    available while ``tools.radio.enabled`` is on, ``tools.radio.stations`` is not empty and
    ``host.media`` is set; stop and volume also whenever something plays. Side-effect free: it
    only builds the skills, so the realtime check may call it with a throwaway host."""

    def available() -> bool:
        cfg = host.config()
        return cfg.tools.radio.enabled and bool(cfg.tools.radio.stations) and host.media is not None

    def controllable() -> bool:
        player = host.media
        return available() or (player is not None and player.now_playing() is not None)

    @skill(
        llm_visible=True,
        description=PLAY_DESCRIPTION,
        available=available,
        choices={"station": lambda: names_and_genres(host.config())},
    )
    def radio_play(station: StationArg) -> SkillResult:
        cfg, player = host.config(), host.media
        if player is None or not cfg.tools.radio.enabled:
            log.info("radio_play: %s", UNAVAILABLE)
            return SkillResult({"error": UNAVAILABLE})
        chosen = pick(cfg, station)
        if chosen is None:  # the list changed after the model was shown it
            log.info("radio_play: %s", NOT_FOUND)
            return SkillResult({"error": NOT_FOUND})
        budget_s = cfg.conversation.tool_timeout_s - TOOL_TIMEOUT_MARGIN_S
        wait_s = max(0.0, min(cfg.tools.media.connect_timeout_s, budget_s))
        status = player.play(chosen, connect_timeout_s=wait_s)
        log.info("radio_play: %s %s", status, chosen.name)
        if status == "ok":
            return SkillResult({"status": status, "station": chosen.name})
        return SkillResult({"error": status, "station": chosen.name})

    @skill(llm_visible=True, description=STOP_DESCRIPTION, available=controllable)
    def radio_stop() -> SkillResult:
        player = host.media
        playing = player.now_playing() if player is not None else None
        if player is None or playing is None:
            log.info("radio_stop: %s", NOT_PLAYING)
            return SkillResult({"error": NOT_PLAYING})
        player.stop("tool")
        log.info("radio_stop: ok %s", playing.name)
        return SkillResult({"ok": True, "station": playing.name})

    @skill(llm_visible=True, description=VOLUME_DESCRIPTION, available=controllable)
    def radio_volume(change: Change) -> SkillResult:
        player = host.media
        if player is None:
            log.info("radio_volume: %s", UNAVAILABLE)
            return SkillResult({"error": UNAVAILABLE})
        step = host.config().tools.media.volume_step_db
        level = player.volume(step if change == "up" else -step)
        log.info("radio_volume: %s", change)
        return SkillResult({"volume_db": level})

    return [radio_play, radio_stop, radio_volume]
