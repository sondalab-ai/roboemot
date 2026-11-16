"""The timer and reminder voice tools (SPEC 11, Voice tools: Scheduler skills, Limits).

``timer_set``, ``timer_cancel``, ``reminder_set``, ``reminder_cancel`` and ``schedule_list``
over ``ToolHost.scheduler``; each is unavailable while the run has no scheduler. Their
descriptions carry the local date, weekday and time in ``circadian.timezone``, so the model
resolves "tomorrow at six" itself. A result is JSON with the item's ``id``, ``kind`` and local
``due`` time; an error is ``{"error": code}``: the store's ``full``, ``too_long``, ``in_past``,
``too_far``, ``bad_days``, ``empty``, and this module's ``bad_date``, ``bad_time`` and
``not_found``. Logs carry skill names, ids and codes only, never a label or a reminder text
(SPEC 12).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from typing import Annotated, Any

from ..config.schema import TIMER_MAX_CEILING_S
from ..skills.host import ToolHost
from ..skills.registry import SkillResult, skill
from ..skills.schema import Desc, Range
from .schedule import ISO_DAYS, Item, Kind, Repeat
from .scheduler import ScheduleError, Scheduler

log = logging.getLogger(__name__)

# ASCII digits only: "²" or "٠٨" are digits to str.isdigit and int(), never to these patterns.
DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
CLOCK_PATTERN = re.compile(r"([0-9]{1,2}):([0-9]{2})")
DAY_PATTERN = re.compile(r"[0-9]+")
DAYS_SEPARATOR = ","
LOCAL_FORMAT = "%Y-%m-%dT%H:%M:%S"
NOW_FORMAT = "%A %Y-%m-%d %H:%M"

Seconds = Annotated[int, Range(1, TIMER_MAX_CEILING_S), Desc("Duration in seconds")]
Label = Annotated[str, Desc("Optional short label, such as pasta")]
Text = Annotated[str, Desc("What to remind, in the person's words")]
Date = Annotated[str, Desc("Local date YYYY-MM-DD; omit for today")]
Clock = Annotated[str, Desc("Local clock time HH:MM, 24-hour")]
RepeatArg = Annotated[Repeat, Desc("once, daily, weekdays (Monday to Friday) or days")]
Days = Annotated[str, Desc("For repeat days: ISO weekdays, 1 Monday to 7 Sunday, as 1,3,5")]
Which = Annotated[str, Desc("The id or the label of the item")]

TIMER_SET = (
    "Start a countdown timer for the person: the duration in seconds (convert minutes and "
    "hours yourself) and an optional label. You announce it when it ends"
)
TIMER_CANCEL = "Cancel a running timer, by its id or label"
REMINDER_SET = (
    "Set a reminder at a local date and clock time, once or repeating. Resolve relative days "
    "(tomorrow, next Friday) yourself from the current local time below. You announce it then"
)
REMINDER_CANCEL = "Cancel a reminder, by its id or text"
SCHEDULE_LIST = "List the active timers and reminders with their local due times"


class BadArgument(ValueError):
    """An argument this module cannot parse; ``code`` is the error result's code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def parse_date(text: str) -> dt.date:
    """``YYYY-MM-DD`` as a date; ``bad_date`` for any other spelling or a day that does not
    exist (2026-13-40)."""
    if not DATE_PATTERN.fullmatch(text.strip()):
        raise BadArgument("bad_date")
    try:
        return dt.date.fromisoformat(text.strip())
    except ValueError:
        raise BadArgument("bad_date") from None


def parse_clock(text: str) -> dt.time:
    """``HH:MM`` (24-hour) as a time; ``bad_time`` for any other spelling or an hour or minute
    out of range."""
    m = CLOCK_PATTERN.fullmatch(text.strip())
    if m is None:
        raise BadArgument("bad_time")
    try:
        return dt.time(int(m.group(1)), int(m.group(2)))
    except ValueError:
        raise BadArgument("bad_time") from None


def parse_days(text: str | None) -> tuple[int, ...]:
    """A weekday list like ``1,3,5`` as ISO weekdays; ``bad_days`` for anything else. None or
    empty gives no days (the store refuses that for ``repeat: days``)."""
    if not text or not text.strip():
        return ()
    parts = [p.strip() for p in text.split(DAYS_SEPARATOR)]
    if not all(DAY_PATTERN.fullmatch(p) for p in parts):
        raise BadArgument("bad_days")
    days = tuple(int(p) for p in parts)
    if not set(days) <= ISO_DAYS:
        raise BadArgument("bad_days")
    return days


def skills(host: ToolHost) -> list[Callable[..., Any]]:
    """The scheduler tools over ``host``: five skills, each available while ``host.scheduler``
    is set (read on every check and call). Side-effect free: it only builds the skills, so the
    realtime check may call it with a throwaway host whose scheduler is None."""

    def store() -> Scheduler:
        sched = host.scheduler
        if sched is None:  # the executor checks availability first: a race with shutdown
            raise RuntimeError("no scheduler")
        return sched

    def local_now() -> dt.datetime:
        sched = host.scheduler
        return sched.now().astimezone(sched.tz()) if sched is not None else host.wall_now()

    def described(base: str) -> Callable[[], str]:
        return lambda: f"{base}. Now: {local_now().strftime(NOW_FORMAT)} local time"

    def available() -> bool:
        return host.scheduler is not None

    def local(at: dt.datetime) -> str:
        return at.astimezone(store().tz()).strftime(LOCAL_FORMAT)

    def created(item: Item) -> SkillResult:
        out: dict[str, Any] = {"id": item.id, "kind": item.kind, "due": local(item.due)}
        if item.repeat != "once":
            out["repeat"] = item.repeat
        if item.days:
            out["days"] = list(item.days)
        return SkillResult(out)

    def refused(name: str, code: str) -> SkillResult:
        log.info("%s: refused: %s", name, code)
        return SkillResult({"error": code})

    def choices(kind: Kind) -> Callable[[], list[str]]:
        def ids_and_texts() -> list[str]:
            sched = host.scheduler
            mine = [i for i in sched.items() if i.kind == kind] if sched is not None else []
            texts = [i.text for i in mine if i.text]
            return [i.id for i in mine] + list(dict.fromkeys(texts))

        return ids_and_texts

    def cancel(name: str, kind: Kind, which: str) -> SkillResult:
        mine = [i for i in store().items() if i.kind == kind]
        hit = next((i for i in mine if i.id == which), None) or next(
            (i for i in mine if i.text == which), None
        )
        if hit is None or not store().cancel(hit.id):
            return refused(name, "not_found")
        return SkillResult({"ok": True, "id": hit.id})

    @skill(llm_visible=True, describe=described(TIMER_SET), available=available)
    def timer_set(seconds: Seconds, label: Label | None = None) -> SkillResult:
        try:
            return created(store().add_timer(seconds, label))
        except ScheduleError as e:
            return refused("timer_set", e.code)

    @skill(
        llm_visible=True,
        describe=described(TIMER_CANCEL),
        available=available,
        choices={"which": choices("timer")},
    )
    def timer_cancel(which: Which) -> SkillResult:
        return cancel("timer_cancel", "timer", which)

    @skill(llm_visible=True, describe=described(REMINDER_SET), available=available)
    def reminder_set(
        text: Text,
        time: Clock,
        date: Date | None = None,
        repeat: RepeatArg = "once",
        days: Days | None = None,
    ) -> SkillResult:
        try:
            clock = parse_clock(time)
            day = parse_date(date) if date else local_now().date()
            weekdays = parse_days(days) if repeat == "days" else ()
            due = dt.datetime.combine(day, clock, tzinfo=store().tz())
            return created(store().add_reminder(text, due, repeat, weekdays))
        except (BadArgument, ScheduleError) as e:
            return refused("reminder_set", e.code)

    @skill(
        llm_visible=True,
        describe=described(REMINDER_CANCEL),
        available=available,
        choices={"which": choices("reminder")},
    )
    def reminder_cancel(which: Which) -> SkillResult:
        return cancel("reminder_cancel", "reminder", which)

    @skill(llm_visible=True, describe=described(SCHEDULE_LIST), available=available)
    def schedule_list() -> SkillResult:
        out = []
        for item in store().items():
            entry: dict[str, Any] = {"id": item.id, "kind": item.kind, "due": local(item.due)}
            if item.text:
                entry["label" if item.kind == "timer" else "text"] = item.text
            if item.repeat != "once":
                entry["repeat"] = item.repeat
            if item.days:
                entry["days"] = list(item.days)
            out.append(entry)
        return SkillResult({"items": out})

    return [timer_set, timer_cancel, reminder_set, reminder_cancel, schedule_list]
