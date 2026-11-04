"""The robot's own wall-clock timezone (SPEC 7A, ``circadian.timezone``)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from pydantic import ValidationError
from roboemot import timezones
from roboemot.bus import Bus
from roboemot.circadian.runner import CircadianRunner, SystemClock
from roboemot.config.schema import RoboEmotConfig

UTC_NOW = dt.datetime(2026, 11, 4, 21, 43, 30, tzinfo=dt.UTC)  # the robot's GMT clock read 21:43


class FrozenNow(dt.datetime):
    """``datetime.now(tz)`` at UTC_NOW, in ``tz``; no tz: the naive UTC time (a GMT robot)."""

    @classmethod
    def now(cls, tz=None):
        return UTC_NOW.astimezone(tz) if tz else UTC_NOW.replace(tzinfo=None)


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(timezones, "datetime", FrozenNow)


def with_zone(name: str | None) -> RoboEmotConfig:
    cfg = RoboEmotConfig()
    return cfg.model_copy(update={"circadian": cfg.circadian.model_copy(update={"timezone": name})})


def test_the_configured_zone_gives_the_owners_hour_on_a_utc_robot(frozen):
    assert SystemClock("Europe/Rome").hour() == pytest.approx(22 + 43.5 / 60)


def test_no_zone_keeps_the_system_local_hour(frozen):
    assert SystemClock().hour() == pytest.approx(21 + 43.5 / 60)


def test_an_unknown_zone_is_refused_by_the_config():
    with pytest.raises(ValidationError, match="circadian.timezone"):
        RoboEmotConfig.model_validate({"circadian": {"timezone": "Mars/Olympus"}})


def test_the_runner_points_the_clock_at_the_zone_and_follows_hot_reload(frozen):
    clock = SystemClock()
    r = CircadianRunner(
        with_zone("Europe/Rome"), Bus(), clock=clock, body_yaw=None, affect_intensity=lambda: 0.0
    )
    assert clock.timezone == "Europe/Rome"
    r.apply_config(with_zone(None))
    assert clock.timezone is None


def test_the_local_name_comes_from_tz_or_the_localtime_link(tmp_path: Path):
    link = tmp_path / "localtime"
    link.symlink_to("/usr/share/zoneinfo/Europe/Rome")
    assert timezones.local_name(link, env={}) == "Europe/Rome"
    assert timezones.local_name(link, env={"TZ": ":Asia/Tokyo"}) == "Asia/Tokyo"
    assert timezones.local_name(tmp_path / "missing", env={"TZ": "nope"}) is None
