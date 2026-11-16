"""What a voice tool may use (SPEC 11, Voice tools): the narrow core object handed to every
``roboemot.skills`` factory.

A part the run does not have is ``None``; a skill that needs it marks itself unavailable
(``@skill(available=...)``). ``config`` is a reader, never a snapshot: an ``available`` check or a
skill calls it each time, so a hot reload applies at once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .. import timezones
from ..bus import Bus
from ..config.schema import RoboEmotConfig

if TYPE_CHECKING:
    from ..tools.scheduler import Scheduler


@dataclass
class ToolHost:
    """The parts a tool may use. ``config`` reads the live merged config; ``wall_now`` is the
    tz-aware wall-clock time in ``circadian.timezone``; ``bus`` is the run's message bus. Later
    tools add their parts as fields that default to None. ``nap`` asks for a requested nap and
    returns its status at once (``RuntimeHost.request_nap``, SPEC 7A); runtime startup sets it.
    ``scheduler`` is the run's timers and reminders store (SPEC 11, Voice tools), set by runtime
    startup."""

    config: Callable[[], RoboEmotConfig]
    wall_now: Callable[[], datetime]
    bus: Bus
    nap: Callable[[], str] | None = None
    scheduler: Scheduler | None = None


def tool_host(config: Callable[[], RoboEmotConfig], bus: Bus) -> ToolHost:
    """A :class:`ToolHost` over ``config`` and ``bus`` whose wall clock reads
    ``circadian.timezone`` from ``config()`` on every call (the system's zone when it is null),
    always tz-aware."""

    def wall_now() -> datetime:
        now = timezones.now(config().circadian.timezone)
        return now if now.tzinfo is not None else now.astimezone()

    return ToolHost(config=config, wall_now=wall_now, bus=bus)
