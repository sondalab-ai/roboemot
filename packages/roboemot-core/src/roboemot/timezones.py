"""IANA timezone names for the robot's wall clock (SPEC 7A, circadian ``timezone``).

The robot's system timezone may differ from the owner's (a robot left on UTC or GMT), and
changing it can affect other software on the robot, so RoboEmot keeps its own setting. The
installer offers these names with completion and defaults to the PC's zone.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

LOCALTIME = Path("/etc/localtime")
ZONEINFO_MARK = "zoneinfo/"


def zone(name: str) -> ZoneInfo:
    """The zone named ``name`` (``Europe/Rome``); ValueError when it is not an IANA name
    the system's or ``tzdata``'s database knows."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as e:  # OSError: a folder (Europe)
        raise ValueError(f"unknown timezone {name!r}, use an IANA name such as Europe/Rome") from e


def names() -> list[str]:
    """Every IANA zone name the database knows, sorted (the installer's completion list)."""
    return sorted(available_timezones())


def now(name: str | None) -> datetime:
    """The wall-clock time in zone ``name``; None: the system's local time."""
    return datetime.now(zone(name) if name else None)


def local_name(localtime: Path = LOCALTIME, env: dict[str, str] | None = None) -> str | None:
    """This machine's IANA zone name: ``TZ`` when it is one, else the target of
    ``/etc/localtime`` (macOS and most Linux); None when neither names a known zone."""
    tz = (os.environ if env is None else env).get("TZ", "").lstrip(":")
    candidates = [tz] if tz else []
    try:
        target = str(localtime.resolve())
    except OSError:
        target = ""
    if ZONEINFO_MARK in target:
        candidates.append(target.rsplit(ZONEINFO_MARK, 1)[1])
    for name in candidates:
        try:
            zone(name)
        except ValueError:
            continue
        return name
    return None
