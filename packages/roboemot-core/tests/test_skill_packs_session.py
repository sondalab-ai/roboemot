"""The active skill pack state (SPEC 8; design 2026-11-24-skill-packs 5.4, 5.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.prompt import ended_line
from roboemot.skill_packs.session import END_REASONS, SkillPackSession

ROUTE = {"base_url": "http://lan.local/v1", "model": "m1"}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def write(root: Path, name: str, body: str = "Body text.") -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: About {name}.\n---\n{body}")


@pytest.fixture
def env(tmp_path):
    write(tmp_path, "quiz-night", "QUIZ-BODY")
    write(tmp_path, "bedtime", "BED-BODY")
    write(tmp_path, "off-pack")
    state = {"cfg": RoboEmotConfig(skill_packs={"enabled": ["quiz-night", "bedtime"]})}
    lib = SkillPackLibrary(tmp_path, lambda: state["cfg"])
    lib.scan()
    clock = Clock()
    session = SkillPackSession(lib, lambda: state["cfg"], clock)
    return session, state, clock


def test_start_normalises_name(env):
    session, _, _ = env
    assert session.start("  Quiz Night ") == ("started", "QUIZ-BODY")
    assert session.snapshot()["active"] == "quiz-night"
    assert session.start("quiz_night")[0] == "started"


def test_start_unknown_or_disabled_is_not_found(env):
    session, _, _ = env
    assert session.start("nope") == ("not_found", None)
    assert session.start("off-pack") == ("not_found", None)
    assert session.snapshot()["active"] is None


def test_start_replaces_and_queues_end_line(env):
    session, _, _ = env
    session.start("quiz-night")
    session.instructions()
    session.start("bedtime")
    text = session.instructions()
    assert ended_line("quiz-night") in text
    assert "BED-BODY" in text and "QUIZ-BODY" not in text


@pytest.mark.parametrize("reason", sorted(END_REASONS))
def test_each_end_reason(env, reason):
    session, _, _ = env
    assert session.end(reason) is False
    session.start("quiz-night")
    assert session.end(reason) is True
    assert session.snapshot()["active"] is None
    assert session.snapshot()["counts"][f"end_{reason}"] == 1


def test_unknown_end_reason_rejected(env):
    session, _, _ = env
    with pytest.raises(ValueError):
        session.end("whatever")


def test_end_line_sent_once(env):
    session, _, _ = env
    session.start("quiz-night")
    session.end("tool")
    assert ended_line("quiz-night") in session.instructions()
    assert ended_line("quiz-night") not in session.instructions()


def test_instructions_carry_catalogue_and_active_block(env):
    session, _, _ = env
    text = session.instructions()
    assert "bedtime: About bedtime." in text and "quiz-night: About quiz-night." in text
    assert "off-pack" not in text
    session.start("quiz-night")
    assert "QUIZ-BODY" in session.instructions()


def test_idle_end_after_idle_end_s(env):
    session, state, clock = env
    session.start("quiz-night")
    clock.t += state["cfg"].skill_packs.idle_end_s - 1
    session.tick()
    assert session.snapshot()["active"] == "quiz-night"
    clock.t += 2
    session.tick()
    assert session.snapshot()["active"] is None
    assert session.snapshot()["counts"]["end_idle"] == 1


def test_heard_addressed_resets_idle(env):
    session, state, clock = env
    session.start("quiz-night")
    clock.t += state["cfg"].skill_packs.idle_end_s - 1
    session.heard_addressed()
    clock.t += 2
    session.tick()
    assert session.snapshot()["active"] == "quiz-night"


def test_on_config_disable_ends_active(env):
    session, state, _ = env
    session.start("quiz-night")
    state["cfg"] = RoboEmotConfig(skill_packs={"enabled": ["bedtime"]})
    session.on_config(state["cfg"])
    assert session.snapshot()["active"] is None
    assert session.snapshot()["counts"]["end_disabled"] == 1
    assert session.enabled_names() == ["bedtime"]


def test_too_large_pack_skipped(tmp_path, caplog):
    write(tmp_path, "big", "x" * 3000)
    write(tmp_path, "small")
    route = LLMRoute(name="r", **ROUTE, context_budget_tokens=256)
    base = RoboEmotConfig(skill_packs={"enabled": ["big", "small"]})
    from roboemot.llm.prompt import system_tokens_worst_case

    budget = system_tokens_worst_case(base) + base.conversation.turn_reserve_tokens + 300
    route = route.model_copy(update={"context_budget_tokens": budget})
    cfg = base.model_copy(update={"llm_routes": [route]})
    lib = SkillPackLibrary(tmp_path, lambda: cfg)
    lib.scan()
    session = SkillPackSession(lib, lambda: cfg, Clock())
    assert session.enabled_names() == ["small"]
    assert session.too_large() == {"big": "r"}
    assert "big" in caplog.text


def test_instructions_empty_when_nothing_enabled(tmp_path):
    cfg = RoboEmotConfig()
    lib = SkillPackLibrary(tmp_path, lambda: cfg)
    session = SkillPackSession(lib, lambda: cfg, Clock())
    assert session.instructions() == ""
    assert session.start("x") == ("not_found", None)


def test_nothing_usable_without_the_tools_listed(tmp_path):
    write(tmp_path, "quiz-night")
    cfg = RoboEmotConfig(
        skill_packs={"enabled": ["quiz-night"]}, conversation={"llm_skills": ["play"]}
    )
    lib = SkillPackLibrary(tmp_path, lambda: cfg)
    lib.scan()
    session = SkillPackSession(lib, lambda: cfg, Clock())
    assert session.enabled_names() == []
    assert session.instructions() == ""
