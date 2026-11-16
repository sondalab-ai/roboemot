"""Timers and reminders as data (SPEC 11, Voice tools, Scheduler store): the item, its
recurrence and its JSON form. Pure: no clock, no file, no lock.

A due time is aware UTC. A repeating reminder keeps its local clock time (``local_time``) and
recurs at that time in the configured zone, so 08:00 stays 08:00 across a DST switch. Full
calendar rules (every second Tuesday, monthly) are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, tzinfo
from typing import Any, Literal

Repeat = Literal["once", "daily", "weekdays", "days"]
Kind = Literal["timer", "reminder"]
REPEATS: tuple[str, ...] = ("once", "daily", "weekdays", "days")
KINDS: tuple[str, ...] = ("timer", "reminder")
ISO_DAYS = frozenset(range(1, 8))  # ISO weekdays: Monday 1 .. Sunday 7
WORK_DAYS = frozenset(range(1, 6))  # Monday to Friday
DAYS_PER_WEEK = 7
CLOCK_FORMAT = "%H:%M"


@dataclass(frozen=True)
class Item:
    """One timer or reminder. ``text`` is the timer's label (empty: none) or the reminder's
    text. ``due`` and ``created`` are aware UTC. ``days`` are ISO weekdays for ``repeat ==
    "days"``. ``local_time`` is the local clock time a repeating reminder recurs at."""

    id: str
    kind: Kind
    text: str
    due: datetime
    created: datetime
    repeat: Repeat = "once"
    days: tuple[int, ...] = ()
    local_time: time | None = None


def matches(item: Item, weekday: int) -> bool:
    """Whether a repeating ``item`` recurs on ISO ``weekday`` (1 Monday .. 7 Sunday)."""
    if item.repeat == "daily":
        return True
    if item.repeat == "weekdays":
        return weekday in WORK_DAYS
    return item.repeat == "days" and weekday in item.days


def next_due(item: Item, after: datetime, tz: tzinfo) -> datetime | None:
    """The first occurrence of ``item.local_time`` in ``tz`` strictly after ``after``, as aware
    UTC; None for a ``once`` item or one without a local time. Walks local dates, never adds
    24 h in UTC, so the local clock time holds across DST. A time a spring-forward gap skips
    resolves the way ``zoneinfo`` does (fold 0)."""
    if item.repeat == "once" or item.local_time is None:
        return None
    start = after.astimezone(tz).date()
    for offset in range(DAYS_PER_WEEK + 1):
        day = start + timedelta(days=offset)
        if not matches(item, day.isoweekday()):
            continue
        at = datetime.combine(day, item.local_time, tzinfo=tz)
        if at > after:
            return at.astimezone(UTC)
    return None


def to_json(item: Item) -> dict[str, Any]:
    """``item`` as a JSON object for the store file (times as ISO text)."""
    return {
        "id": item.id,
        "kind": item.kind,
        "text": item.text,
        "due": item.due.isoformat(),
        "created": item.created.isoformat(),
        "repeat": item.repeat,
        "days": list(item.days),
        "local_time": item.local_time.strftime(CLOCK_FORMAT) if item.local_time else None,
    }


def from_json(raw: Any) -> Item:
    """The :class:`Item` of a store object; ValueError (or a KeyError/TypeError) when it is
    not one. Due and created times without a zone are refused, so a hand-edited file never
    shifts an item silently."""
    if raw["kind"] not in KINDS or raw["repeat"] not in REPEATS:
        raise ValueError("unknown kind or repeat")
    due, created = datetime.fromisoformat(raw["due"]), datetime.fromisoformat(raw["created"])
    if due.tzinfo is None or created.tzinfo is None:
        raise ValueError("naive time")
    days = tuple(int(d) for d in raw.get("days") or ())
    if not set(days) <= ISO_DAYS:
        raise ValueError("bad days")
    local = raw.get("local_time")
    return Item(
        id=str(raw["id"]),
        kind=raw["kind"],
        text=str(raw["text"]),
        due=due.astimezone(UTC),
        created=created.astimezone(UTC),
        repeat=raw["repeat"],
        days=days,
        local_time=datetime.strptime(local, CLOCK_FORMAT).time() if local else None,
    )
