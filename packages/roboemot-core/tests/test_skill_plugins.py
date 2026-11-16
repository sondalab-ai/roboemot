"""SPEC 11, Voice tools: the ``roboemot.skills`` plugin group, its loader and the registry it
feeds."""

from __future__ import annotations

import importlib.metadata
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.skills import skill
from roboemot.skills.defaults import SkillEnv
from roboemot.skills.host import ToolHost, tool_host
from roboemot.skills.plugins import (
    SKILLS_GROUP,
    build_skill_registry,
    load_plugin_skill_sets,
    load_skill_plugins,
    warn_hidden_core_skills,
)


def fake_ep(name: str, factory: Any, dist: str = "some-plugin") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, group=SKILLS_GROUP, load=lambda: factory, dist=SimpleNamespace(name=dist)
    )


def install(monkeypatch: pytest.MonkeyPatch, *eps: SimpleNamespace) -> None:
    def entry_points(**kw: Any) -> list[SimpleNamespace]:
        return [ep for ep in eps if kw.get("group") == ep.group]

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


def factory_of(*names: str) -> Any:
    def factory(host: ToolHost) -> list[Any]:
        out = []
        for n in names:

            @skill(n, llm_visible=True, inline=True, description=n)
            def tool() -> str:
                return "ok"

            out.append(tool)
        return out

    return factory


def host(cfg: RoboEmotConfig | None = None) -> ToolHost:
    current = cfg or RoboEmotConfig()
    return tool_host(lambda: current, Bus())


def env() -> SkillEnv:
    cfg = RoboEmotConfig()
    return SkillEnv(
        gestures=lambda: list(cfg.affect.labels),
        play=lambda _n: True,
        look_names=list,
        look_at=lambda _t: True,
        affect=SimpleNamespace(cfg=cfg.affect),
    )


def test_plugins_load_in_name_order_and_skip_broken(monkeypatch, caplog):
    def boom(host: ToolHost) -> list[Any]:
        raise RuntimeError("secret text from the plugin")

    install(
        monkeypatch,
        fake_ep("b", factory_of("b_tool")),
        fake_ep("a", factory_of("a_one", "a_two")),
        fake_ep("c", boom),
    )
    with caplog.at_level(logging.WARNING):
        loaded = load_skill_plugins(host())
    assert [fn.skill.name for fn in loaded] == ["a_one", "a_two", "b_tool"]
    assert "'c'" in caplog.text and "RuntimeError" in caplog.text
    assert "secret text" not in caplog.text


def test_a_factory_returning_a_non_skill_or_failing_to_load_is_skipped(monkeypatch, caplog):
    def not_skills(host: ToolHost) -> list[Any]:
        return [factory_of("ok_tool")(host)[0], lambda: None]

    def unloadable() -> Any:
        raise ImportError("missing dependency")

    bad_load = SimpleNamespace(
        name="d", group=SKILLS_GROUP, load=unloadable, dist=SimpleNamespace(name="x")
    )
    install(monkeypatch, fake_ep("e", not_skills), bad_load, fake_ep("f", lambda h: "nope"))
    with caplog.at_level(logging.WARNING):
        assert load_skill_plugins(host()) == []
    assert all(f"'{n}'" in caplog.text for n in "def")


def test_startup_registry_includes_plugins_named_in_llm_skills(monkeypatch):
    install(monkeypatch, fake_ep("weather", factory_of("weather_now")))
    cfg = RoboEmotConfig().model_copy(deep=True)
    allow = [*cfg.conversation.llm_skills, "weather_now"]
    registry = build_skill_registry(env(), host(cfg))
    assert "weather_now" in registry.names()
    assert "play" in registry.names()
    assert "weather_now" in [s.name for s in registry.offered(allow)]
    assert "weather_now" not in [s.name for s in registry.offered(cfg.conversation.llm_skills)]


def test_a_plugin_skill_with_a_taken_name_is_skipped(monkeypatch, caplog):
    install(monkeypatch, fake_ep("x", factory_of("play", "fresh")))
    with caplog.at_level(logging.WARNING):
        registry = build_skill_registry(env(), host())
    assert "fresh" in registry.names()
    assert registry.get("play").fn.__module__ == "roboemot.skills.defaults"
    assert "play" in caplog.text


def test_startup_warns_once_about_core_tools_missing_from_llm_skills(monkeypatch, caplog):
    install(
        monkeypatch,
        fake_ep("core_tools", factory_of("go_to_sleep", "timer_set"), dist="roboemot_core"),
        fake_ep("other", factory_of("third_party"), dist="someone-else"),
    )
    allow = ["play", "timer_set"]
    plugins = load_plugin_skill_sets(host())
    with caplog.at_level(logging.WARNING):
        build_skill_registry(env(), host(), plugins)
    assert "go_to_sleep" not in caplog.text  # building the registry alone never warns
    with caplog.at_level(logging.WARNING):
        assert warn_hidden_core_skills(plugins, allow) == ["go_to_sleep"]
    warnings = [r for r in caplog.records if "llm_skills" in r.getMessage()]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "go_to_sleep" in text
    assert "timer_set" not in text and "third_party" not in text


def test_no_warning_when_every_core_tool_is_listed(monkeypatch, caplog):
    install(monkeypatch, fake_ep("core_tools", factory_of("go_to_sleep"), dist="roboemot-core"))
    plugins = load_plugin_skill_sets(host())
    with caplog.at_level(logging.WARNING):
        assert warn_hidden_core_skills(plugins, ["go_to_sleep"]) == []
    assert "llm_skills" not in caplog.text


def test_wall_now_is_aware_and_reads_the_live_timezone():
    current = {"cfg": RoboEmotConfig.model_validate({"circadian": {"timezone": "Asia/Tokyo"}})}
    h = tool_host(lambda: current["cfg"], Bus())
    assert h.wall_now().utcoffset().total_seconds() == 9 * 3600
    current["cfg"] = RoboEmotConfig.model_validate({"circadian": {"timezone": "UTC"}})
    assert h.wall_now().utcoffset().total_seconds() == 0
    current["cfg"] = RoboEmotConfig()
    now = h.wall_now()
    assert isinstance(now, datetime) and now.tzinfo is not None


def test_the_realtime_check_offers_plugin_skills_named_in_llm_skills(monkeypatch):
    from roboemot.realtime.check import offline_tools

    install(monkeypatch, fake_ep("w", factory_of("weather_now", "hidden_tool")))
    cfg = RoboEmotConfig.model_validate({"conversation": {"llm_skills": ["weather_now"]}})
    assert [t["name"] for t in offline_tools(cfg)] == ["weather_now"]
