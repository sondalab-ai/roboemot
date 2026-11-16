"""SPEC 11 (Voice tools: Scheduler store, Limits, Missed items, Due items) and SPEC 11A
(``tools.scheduler.*``): the persistent store and the runner that publishes ``tools.due``."""

from __future__ import annotations

import json
import logging
import stat
import threading
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from roboemot.bus import TOPICS, Bus
from roboemot.config.schema import RoboEmotConfig, SchedulerConfig
from roboemot.tools.scheduler import ScheduleError, Scheduler, SchedulerRunner

ROME = ZoneInfo("Europe/Rome")
T0 = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)


class Clock:
    """A settable wall clock (aware UTC)."""

    def __init__(self, at=T0):
        self.at = at

    def __call__(self):
        return self.at


def make(tmp_path, clock=None, **cfg):
    clock = clock or Clock()
    conf = SchedulerConfig(**cfg)
    return Scheduler(tmp_path / "tools_schedule.json", lambda: conf, clock, lambda: ROME), clock


def test_defaults_are_the_spec_values():
    c = RoboEmotConfig().tools.scheduler
    assert (c.max_items, c.text_max_chars, c.timer_max_s, c.missed_grace_s) == (20, 200, 86400, 900)
    assert (c.tick_s, c.announce_timeout_s, c.announce_max_wait_s) == (1.0, 8, 60)
    assert (c.repeat_after_s, c.repeat_max) == (30, 2)
    assert "tools.due" in TOPICS


def test_restart_inside_grace_keeps_item_and_outside_moves_it_to_missed(tmp_path):
    s, clock = make(tmp_path)
    item = s.add_timer(60, None)
    due = item.due
    assert due == T0 + timedelta(seconds=60)

    inside, _ = make(tmp_path, Clock(due + timedelta(seconds=30)))
    inside.load(due + timedelta(seconds=30))
    assert [i.id for i in inside.due(due + timedelta(seconds=30))] == [item.id]
    assert inside.missed() == []

    outside, _ = make(tmp_path, Clock(due + timedelta(seconds=901)))
    outside.load(due + timedelta(seconds=901))
    assert outside.due(due + timedelta(seconds=901)) == []
    assert [i.id for i in outside.missed()] == [item.id]
    assert outside.items() == []
    outside.clear_missed()
    assert outside.missed() == []


def test_a_running_process_never_expires_a_due_item(tmp_path):
    s, clock = make(tmp_path)
    item = s.add_timer(10, "pasta")
    clock.at = T0 + timedelta(hours=3)  # a platform sleep: still due, never missed
    assert [i.id for i in s.due(clock.at)] == [item.id]
    assert s.missed() == []


def test_a_missed_repeating_reminder_is_listed_and_moves_on(tmp_path):
    s, clock = make(tmp_path)
    first = datetime(2026, 11, 16, 11, 0, tzinfo=ROME)
    item = s.add_reminder("pills", first, "daily", ())
    later = first + timedelta(days=2, hours=1)
    again, _ = make(tmp_path, Clock(later))
    again.load(later)
    assert [i.id for i in again.missed()] == [item.id]
    (kept,) = again.items()
    assert kept.id == item.id and kept.due == datetime(2026, 11, 19, 11, 0, tzinfo=ROME)


def test_fired_once_removes_and_repeat_moves_forward(tmp_path):
    s, clock = make(tmp_path)
    timer = s.add_timer(30, None)
    daily = s.add_reminder("water", datetime(2026, 11, 16, 10, 30, tzinfo=ROME), "daily", ())
    clock.at = T0 + timedelta(hours=2)
    assert {i.id for i in s.due(clock.at)} == {timer.id, daily.id}
    s.fired(timer.id)
    s.fired(daily.id)
    s.fired("nope")  # an unknown id (already cancelled): nothing happens
    (kept,) = s.items()
    assert kept.id == daily.id
    assert kept.due == datetime(2026, 11, 17, 10, 30, tzinfo=ROME)
    assert s.due(clock.at) == []


def test_a_late_repeating_reminder_moves_past_now(tmp_path):
    s, clock = make(tmp_path)
    daily = s.add_reminder("water", datetime(2026, 11, 16, 10, 30, tzinfo=ROME), "daily", ())
    clock.at = datetime(2026, 11, 18, 12, 0, tzinfo=ROME)  # fired two days late
    s.fired(daily.id)
    (kept,) = s.items()
    assert kept.due == datetime(2026, 11, 19, 10, 30, tzinfo=ROME)


def test_a_repeating_reminder_starts_at_its_first_matching_day(tmp_path):
    saturday = datetime(2026, 11, 21, 7, 0, tzinfo=ROME)
    s, _ = make(tmp_path, Clock(saturday.astimezone(UTC)))
    item = s.add_reminder("work", datetime(2026, 11, 21, 8, 0, tzinfo=ROME), "weekdays", ())
    assert item.due == datetime(2026, 11, 23, 8, 0, tzinfo=ROME)
    assert item.local_time is not None and item.local_time.hour == 8
    past = s.add_reminder("x", datetime(2026, 11, 20, 6, 0, tzinfo=ROME), "daily", ())
    assert past.due == datetime(2026, 11, 22, 6, 0, tzinfo=ROME)  # never in_past


def test_store_survives_reload_and_is_0600(tmp_path):
    s, clock = make(tmp_path)
    timer = s.add_timer(90, "tea")
    rem = s.add_reminder("call", datetime(2026, 11, 20, 18, 0, tzinfo=ROME), "days", (1, 5))
    path = tmp_path / "tools_schedule.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    json.loads(path.read_text())
    again, _ = make(tmp_path)
    again.load(T0)
    assert again.items() == s.items() == sorted([timer, rem], key=lambda i: i.due)
    assert again.cancel(timer.id) is True
    assert again.cancel(timer.id) is False
    third, _ = make(tmp_path)
    third.load(T0)
    assert [i.id for i in third.items()] == [rem.id]
    new = third.add_timer(5, None)
    assert new.id not in {timer.id, rem.id}  # ids stay unique across restarts


def test_a_broken_file_starts_empty_without_logging_its_content(tmp_path, caplog):
    (tmp_path / "tools_schedule.json").write_text('{"items": [{"text": "secret plan"')
    s, _ = make(tmp_path)
    with caplog.at_level(logging.DEBUG):
        s.load(T0)
    assert s.items() == []
    assert "secret plan" not in caplog.text
    bad = {"items": 3, "missed": [{"id": "t9", "text": "secret"}], "seq": "x"}
    (tmp_path / "tools_schedule.json").write_text(json.dumps(bad))
    with caplog.at_level(logging.DEBUG):
        s.load(T0)
    assert s.items() == [] and s.missed() == []
    assert "secret" not in caplog.text


def test_add_rejects_past_full_and_long_text(tmp_path):
    s, clock = make(tmp_path, max_items=2, text_max_chars=10, timer_max_s=100)
    s.add_timer(10, None)
    before = s.items()

    def code(fn, *args):
        with pytest.raises(ScheduleError) as e:
            fn(*args)
        assert s.items() == before
        return e.value.code

    past = datetime(2026, 11, 16, 9, 0, tzinfo=ROME)  # 08:00 UTC, before T0
    assert code(s.add_reminder, "x", past, "once", ()) == "in_past"
    assert code(s.add_timer, 10, "a label far too long") == "too_long"
    assert code(s.add_reminder, "a text far too long", T0 + timedelta(hours=1), "once", ()) == (
        "too_long"
    )
    assert code(s.add_timer, 101, None) == "too_far"
    assert code(s.add_reminder, "x", T0 + timedelta(hours=1), "days", ()) == "bad_days"
    assert code(s.add_reminder, "x", T0 + timedelta(hours=1), "days", (0, 8)) == "bad_days"
    s.add_timer(10, None)
    before = s.items()
    assert code(s.add_timer, 10, None) == "full"
    assert str(ScheduleError("full")) == "full"  # the code only, never a value


def test_logs_carry_ids_and_kinds_never_text(tmp_path, caplog):
    s, clock = make(tmp_path)
    with caplog.at_level(logging.DEBUG):
        t = s.add_timer(10, "secret label")
        r = s.add_reminder("secret text", T0 + timedelta(hours=1), "once", ())
        s.cancel(r.id)
        clock.at = T0 + timedelta(minutes=1)
        s.fired(t.id)
    assert "secret" not in caplog.text


class Collector:
    def __init__(self, bus):
        self.seen: list[list[str]] = []
        bus.subscribe("tools.due", lambda m: self.seen.append(list(m.payload)))


def test_runner_without_a_consumer_logs_and_marks_fired(tmp_path, caplog):
    s, clock = make(tmp_path)
    bus = Bus()
    seen = Collector(bus)
    t = s.add_timer(10, "secret label")
    runner = SchedulerRunner(s, bus, lambda: s.cfg())
    runner.tick()
    assert seen.seen == []  # nothing due: nothing published
    clock.at = T0 + timedelta(seconds=10)
    with caplog.at_level(logging.INFO):
        runner.tick()
    assert seen.seen == [[t.id]]
    assert s.items() == []
    assert t.id in caplog.text and "secret" not in caplog.text


def test_runner_with_a_consumer_republishes_every_tick_until_fired(tmp_path):
    s, clock = make(tmp_path)
    bus = Bus()
    seen = Collector(bus)
    a, b = s.add_timer(10, None), s.add_timer(20, None)
    runner = SchedulerRunner(s, bus, lambda: s.cfg())
    runner.set_consumer(True)
    assert runner.consumer_attached
    clock.at = T0 + timedelta(seconds=30)
    runner.tick()
    runner.tick()
    assert seen.seen == [[a.id, b.id], [a.id, b.id]]  # no defer: again every tick
    s.fired(a.id)
    runner.tick()
    assert seen.seen[-1] == [b.id]
    runner.set_consumer(False)
    runner.tick()
    assert s.items() == []


def test_runner_thread_ticks_and_stops(tmp_path):
    s, clock = make(tmp_path, tick_s=0.2)
    bus = Bus()
    published = threading.Event()
    bus.subscribe("tools.due", lambda m: published.set())
    s.add_timer(1, None)
    clock.at = T0 + timedelta(seconds=5)
    runner = SchedulerRunner(s, bus, lambda: s.cfg())
    runner.start()
    try:
        assert published.wait(5.0)
    finally:
        runner.close()
    assert not runner.alive
