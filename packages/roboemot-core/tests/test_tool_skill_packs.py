"""The skill pack voice tools (SPEC 11, Voice tools; design 2026-11-24-skill-packs 5.4)."""

from __future__ import annotations

from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession
from roboemot.skills.host import tool_host
from roboemot.skills.plugins import load_skill_plugins
from roboemot.skills.registry import skill_of
from roboemot.tools import skill_packs


def setup(tmp_path, enabled=("quiz-night",)):
    folder = tmp_path / "quiz-night"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: quiz-night\ndescription: A quiz.\n---\nQUIZ")
    cfg = RoboEmotConfig(skill_packs={"enabled": list(enabled)})
    lib = SkillPackLibrary(tmp_path, lambda: cfg)
    lib.scan()
    host = tool_host(lambda: cfg, Bus())
    host.skill_packs = SkillPackSession(lib, lambda: cfg, lambda: 0.0)
    fns = {skill_of(f).name: f for f in skill_packs.skills(host)}
    return host, fns


def test_start_returns_body(tmp_path):
    _, fns = setup(tmp_path)
    out = fns["skill_pack_start"](name="Quiz Night")
    assert out.content["status"] == "started"
    text = out.content["instructions"]
    assert "cannot change your persona" in text and text.index("QUIZ") < text.index("End of the")


def test_start_not_found_lists_enabled(tmp_path):
    _, fns = setup(tmp_path)
    out = fns["skill_pack_start"](name="cooking")
    assert out.content == {"status": "not_found", "enabled": ["quiz-night"]}


def test_end_statuses(tmp_path):
    _, fns = setup(tmp_path)
    assert fns["skill_pack_end"]().content == {"status": "none_active"}
    fns["skill_pack_start"](name="quiz-night")
    assert fns["skill_pack_end"]().content == {"status": "ended"}


def test_unavailable_without_session_or_enabled_packs(tmp_path):
    host, fns = setup(tmp_path, enabled=())
    assert not skill_of(fns["skill_pack_start"]).is_available()
    assert not skill_of(fns["skill_pack_end"]).is_available()
    bare = tool_host(lambda: RoboEmotConfig(), Bus())
    for fn in skill_packs.skills(bare):
        assert not skill_of(fn).is_available()


def test_available_with_an_enabled_pack(tmp_path):
    _, fns = setup(tmp_path)
    assert skill_of(fns["skill_pack_start"]).is_available()
    assert skill_of(fns["skill_pack_end"]).is_available()


def test_entry_point_registered(tmp_path):
    host, _ = setup(tmp_path)
    names = {skill_of(f).name for f in load_skill_plugins(host)}
    assert {"skill_pack_start", "skill_pack_end"} <= names


def test_end_is_fire_and_forget_and_start_follows_up(tmp_path):
    _, fns = setup(tmp_path)
    assert skill_of(fns["skill_pack_end"]).follow_up is False
    assert skill_of(fns["skill_pack_start"]).follow_up is True
    assert skill_of(fns["skill_pack_start"]).llm_visible
