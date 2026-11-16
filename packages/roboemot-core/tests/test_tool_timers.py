"""SPEC 11 (Voice tools: Scheduler skills, Limits) and SPEC 12 (no tool content in logs): the
timer and reminder skills over the scheduler on ``ToolHost.scheduler``."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig, SchedulerConfig
from roboemot.skills.host import ToolHost
from roboemot.skills.plugins import load_plugin_skill_sets
from roboemot.skills.registry import SkillRegistry
from roboemot.tools import timers
from roboemot.tools.scheduler import Scheduler

ROME = ZoneInfo("Europe/Rome")
T0 = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)  # Monday 10:00 in Rome
NAMES = ["timer_set", "timer_cancel", "reminder_set", "reminder_cancel", "schedule_list"]


class Clock:
    def __init__(self, at=T0):
        self.at = at

    def __call__(self):
        return self.at


def rig(tmp_path, with_scheduler=True, **cfg):
    """A tool host whose wall clock and scheduler share one settable clock; the skills by name."""
    clock = Clock()
    conf = SchedulerConfig(**cfg)
    host = ToolHost(
        config=lambda: RoboEmotConfig(),
        wall_now=lambda: clock().astimezone(ROME),
        bus=Bus(),
    )
    if with_scheduler:
        host.scheduler = Scheduler(tmp_path / "s.json", lambda: conf, clock, lambda: ROME)
    fns = {fn.skill.name: fn for fn in timers.skills(host)}
    return host, fns, clock


def content(result):
    return result.content


def test_the_five_skills_are_visible_and_need_a_scheduler(tmp_path):
    host, fns, _ = rig(tmp_path, with_scheduler=False)
    assert list(fns) == NAMES
    assert all(fn.skill.llm_visible for fn in fns.values())
    assert not any(fn.skill.is_available() for fn in fns.values())
    host.scheduler = Scheduler(tmp_path / "s.json", SchedulerConfig, Clock(), lambda: ROME)
    assert all(fn.skill.is_available() for fn in fns.values())  # read live


def test_core_registers_the_timers_plugin_and_lists_it_by_default():
    host = ToolHost(config=lambda: RoboEmotConfig(), wall_now=lambda: T0, bus=Bus())
    plugins = {p.name: p for p in load_plugin_skill_sets(host)}
    assert plugins["timers"].distribution == "roboemot-core"
    assert [fn.skill.name for fn in plugins["timers"].skills] == NAMES
    assert set(NAMES) <= set(RoboEmotConfig().conversation.llm_skills)


def test_description_carries_local_now(tmp_path):
    _, fns, clock = rig(tmp_path)
    text = fns["reminder_set"].skill.text()
    assert "Monday" in text and "2026-11-16" in text and "10:00" in text
    clock.at = T0 + timedelta(days=1, hours=2)
    text = fns["timer_set"].skill.text()
    assert "Tuesday" in text and "2026-11-17" in text and "12:00" in text


def test_timer_set_returns_id_kind_and_local_due(tmp_path):
    host, fns, _ = rig(tmp_path)
    out = content(fns["timer_set"](seconds=90, label="pasta"))
    assert out["kind"] == "timer" and out["id"]
    assert out["due"] == "2026-11-16T10:01:30"
    (item,) = host.scheduler.items()
    assert item.id == out["id"] and item.text == "pasta"


def test_timer_set_beyond_the_limit_is_too_far(tmp_path):
    host, fns, _ = rig(tmp_path, timer_max_s=60)
    assert content(fns["timer_set"](seconds=61, label=None)) == {"error": "too_far"}
    assert host.scheduler.items() == []
    hi = fns["timer_set"].skill.params[0].range.hi
    assert hi == SchedulerConfig.model_fields["timer_max_s"].metadata[-1].le  # the schema's top


def test_reminder_set_bad_date_returns_error_and_no_item(tmp_path):
    host, fns, _ = rig(tmp_path)
    rs = fns["reminder_set"]
    assert content(rs(text="x", time="08:00", date="2026-13-40")) == {"error": "bad_date"}
    assert content(rs(text="x", time="08:00", date="tomorrow")) == {"error": "bad_date"}
    assert content(rs(text="x", time="25:00", date=None)) == {"error": "bad_time"}
    assert content(rs(text="x", time="8 am", date=None)) == {"error": "bad_time"}
    assert content(rs(text="x", time="08:00", repeat="days", days="1,9")) == {"error": "bad_days"}
    assert content(rs(text="x", time="08:00", repeat="days", days="mon")) == {"error": "bad_days"}
    assert content(rs(text="x", time="08:00", repeat="days", days=None)) == {"error": "bad_days"}
    yesterday = content(rs(text="x", time="18:00", date="2026-11-15"))
    assert yesterday == {"error": "in_past"}
    assert host.scheduler.items() == []


def test_reminder_set_resolves_local_date_and_time(tmp_path):
    host, fns, _ = rig(tmp_path)
    rs = fns["reminder_set"]
    out = content(rs(text="dentist", time="18:30", date="2026-11-17"))
    assert out == {"id": out["id"], "kind": "reminder", "due": "2026-11-17T18:30:00"}
    today = content(rs(text="call", time="11:00"))
    assert today["due"] == "2026-11-16T11:00:00"
    # No date and a clock time already gone today: in the past (the model has the local now)
    assert content(rs(text="late", time="09:00")) == {"error": "in_past"}
    weekly = content(rs(text="bins", time="07:00", repeat="days", days="3, 5"))
    assert weekly["due"] == "2026-11-18T07:00:00" and weekly["repeat"] == "days"
    daily = content(rs(text="pills", time="09:00", repeat="daily"))
    assert daily["due"] == "2026-11-17T09:00:00"  # a repeating one moves on, never in_past
    assert len(host.scheduler.items()) == 4


def test_cancel_by_id_or_label_with_choices(tmp_path):
    host, fns, _ = rig(tmp_path)
    t = content(fns["timer_set"](seconds=60, label="pasta"))
    r = content(fns["reminder_set"](text="call mum", time="20:00"))
    tc, rc = fns["timer_cancel"].skill, fns["reminder_cancel"].skill
    assert tc.resolved_choices()["which"] == [t["id"], "pasta"]
    assert rc.resolved_choices()["which"] == [r["id"], "call mum"]
    assert content(fns["timer_cancel"](which="pasta")) == {"ok": True, "id": t["id"]}
    assert content(fns["timer_cancel"](which="pasta")) == {"error": "not_found"}
    assert content(fns["reminder_cancel"](which=r["id"])) == {"ok": True, "id": r["id"]}
    assert host.scheduler.items() == []
    assert tc.resolved_choices()["which"] == []  # nothing to cancel: not offered
    assert SkillRegistry([fns["timer_cancel"]]).offered(["timer_cancel"]) == []


def test_schedule_list_gives_local_times(tmp_path):
    _, fns, _ = rig(tmp_path)
    fns["timer_set"](seconds=60, label=None)
    fns["reminder_set"](text="bins", time="07:00", repeat="weekdays")
    out = content(fns["schedule_list"]())
    assert [i["kind"] for i in out["items"]] == ["timer", "reminder"]
    assert out["items"][0]["due"] == "2026-11-16T10:01:00"
    assert out["items"][1] == {
        "id": out["items"][1]["id"],
        "kind": "reminder",
        "due": "2026-11-17T07:00:00",
        "text": "bins",
        "repeat": "weekdays",
    }


def test_logs_never_carry_reminder_text(tmp_path, caplog):
    _, fns, _ = rig(tmp_path, max_items=2)
    with caplog.at_level(logging.DEBUG):
        t = content(fns["timer_set"](seconds=60, label="secret label"))
        r = content(fns["reminder_set"](text="secret text", time="20:00"))
        fns["reminder_set"](text="secret overflow", time="21:00")  # full
        fns["schedule_list"]()
        fns["timer_cancel"](which="secret label")
        fns["reminder_cancel"](which=r["id"])
    assert t["id"] in caplog.text  # ids and kinds are logged
    assert "secret" not in caplog.text
