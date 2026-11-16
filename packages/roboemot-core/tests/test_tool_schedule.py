"""SPEC 11 (Voice tools, Scheduler store): the item model and the recurrence of a repeating
reminder at its local clock time in ``circadian.timezone``."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from roboemot.tools.schedule import Item, next_due

ROME = ZoneInfo("Europe/Rome")


def reminder(repeat, local_time=time(8, 0), days=()):
    """A repeating reminder item anchored at ``local_time`` (its own due does not matter to
    ``next_due``)."""
    start = datetime(2026, 10, 1, 6, 0, tzinfo=UTC)
    return Item("r1", "reminder", "x", start, start, repeat, tuple(days), local_time)


def test_daily_reminder_keeps_local_time_across_dst():
    item = reminder("daily")
    after = datetime(2026, 10, 24, 8, 0, tzinfo=ROME)  # +02:00, the day before the switch
    assert after.utcoffset() == timedelta(hours=2)
    first = next_due(item, after, ROME)
    assert first == datetime(2026, 10, 25, 8, 0, tzinfo=ROME)
    assert first.utcoffset() == timedelta(0) and first.tzinfo is UTC  # stored as aware UTC
    assert first.astimezone(ROME).utcoffset() == timedelta(hours=1)  # 08:00+01:00
    second = next_due(item, first, ROME)
    assert second == datetime(2026, 10, 26, 8, 0, tzinfo=ROME)
    assert second - first == timedelta(hours=24)
    assert first - after == timedelta(hours=25)  # the day of the switch has 25 hours


def test_weekdays_skip_weekend():
    friday = datetime(2026, 11, 13, 9, 0, tzinfo=ROME)
    assert friday.isoweekday() == 5
    assert next_due(reminder("weekdays"), friday, ROME) == datetime(2026, 11, 16, 8, 0, tzinfo=ROME)


def test_days_use_iso_weekdays_and_today_counts_when_still_ahead():
    monday_early = datetime(2026, 11, 16, 7, 0, tzinfo=ROME)
    item = reminder("days", days=(1, 3))
    assert next_due(item, monday_early, ROME) == datetime(2026, 11, 16, 8, 0, tzinfo=ROME)
    monday_late = datetime(2026, 11, 16, 8, 0, tzinfo=ROME)  # strictly after: not this one
    assert next_due(item, monday_late, ROME) == datetime(2026, 11, 18, 8, 0, tzinfo=ROME)
    sunday = reminder("days", days=(7,))
    assert next_due(sunday, monday_late, ROME) == datetime(2026, 11, 22, 8, 0, tzinfo=ROME)


def test_once_has_no_next():
    assert next_due(reminder("once"), datetime(2026, 11, 16, tzinfo=ROME), ROME) is None
