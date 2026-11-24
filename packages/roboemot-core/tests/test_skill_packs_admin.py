"""The web UI's skill pack admin (SPEC 11A, 12; design 2026-11-24-skill-packs 5.2, 5.3, 5.6)."""

from __future__ import annotations

import base64

import pytest
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.prompt import system_tokens_worst_case
from roboemot.skill_packs.admin import SkillPacksAdmin, SkillPacksError
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession


def skill(name: str, body: str = "BODY") -> bytes:
    return f"---\nname: {name}\ndescription: About {name}.\nauthor: me\n---\n{body}\n".encode()


class Env:
    def __init__(self, tmp_path, **cfg) -> None:
        self.cfg = RoboEmotConfig(**cfg)
        self.writes: list[list[str]] = []
        self.root = tmp_path
        self.library = SkillPackLibrary(tmp_path, lambda: self.cfg)
        self.session = SkillPackSession(self.library, lambda: self.cfg, lambda: 0.0)
        self.admin = SkillPacksAdmin(lambda: self.session, self.write, lambda: self.cfg)

    def write(self, names: list[str]) -> None:
        self.writes.append(names)
        self.cfg = self.cfg.model_copy(
            update={"skill_packs": self.cfg.skill_packs.model_copy(update={"enabled": names})}
        )
        self.session.on_config(self.cfg)

    def upload(self, name: str, body: str = "BODY", replace: bool = False) -> str:
        data = base64.b64encode(skill(name, body)).decode()
        return self.admin.upload(f"{name}.md", data, replace)


def code(fn, *args) -> str:
    with pytest.raises(SkillPacksError) as err:
        fn(*args)
    return err.value.code


def test_list_states_without_body(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    (tmp_path / "broken").mkdir()
    env.admin.rescan()
    listing = env.admin.listing()
    rows = {r["name"]: r for r in listing["packs"]}
    assert rows["quiz"]["state"] == "off" and rows["quiz"]["meta"] == {"author": "me"}
    assert rows["broken"]["state"] == "invalid" and rows["broken"]["reason"] == "no_skill_md"
    assert listing["active"] is None
    assert "BODY" not in str(listing)


def test_missing_enabled_name_is_listed(tmp_path):
    env = Env(tmp_path, skill_packs={"enabled": ["gone"]})
    rows = {r["name"]: r for r in env.admin.listing()["packs"]}
    assert rows["gone"]["state"] == "missing"


def test_detail_has_body(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    assert env.admin.detail("quiz")["body"] == "BODY"
    assert code(env.admin.detail, "nope") == "unknown_pack"


def test_enable_and_disable(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    env.admin.enable("quiz", True)
    assert env.writes[-1] == ["quiz"]
    assert {r["name"]: r["state"] for r in env.admin.listing()["packs"]}["quiz"] == "on"
    env.admin.enable("quiz", False)
    assert env.writes[-1] == []


def test_enable_refusals(tmp_path):
    env = Env(tmp_path, conversation={"llm_skills": ["play"]})
    env.upload("quiz")
    assert code(env.admin.enable, "quiz", True) == "llm_skills_missing"
    env = Env(tmp_path / "b", skill_packs={"max_enabled": 1, "enabled": []})
    env.upload("a-pack")
    env.upload("b-pack")
    env.admin.enable("a-pack", True)
    assert code(env.admin.enable, "b-pack", True) == "max_enabled"
    assert code(env.admin.enable, "nope", True) == "unknown_pack"


def test_enable_too_large_names_the_route(tmp_path):
    base = RoboEmotConfig()
    budget = system_tokens_worst_case(base) + base.conversation.turn_reserve_tokens + 100
    route = LLMRoute(
        name="small", base_url="http://lan.local/v1", model="m", context_budget_tokens=budget
    )
    env = Env(tmp_path, llm_routes=[route])
    env.upload("big", body="x" * 3000)
    with pytest.raises(SkillPacksError) as err:
        env.admin.enable("big", True)
    assert err.value.code == "too_large_for_route" and err.value.extra == {"route": "small"}
    assert env.writes == []


def test_upload_replace_turns_the_pack_off(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    env.admin.enable("quiz", True)
    env.session.start("quiz")
    assert code(env.upload, "quiz") == "exists"
    env.upload("quiz", body="NEW", replace=True)
    assert env.writes[-1] == []
    assert env.session.snapshot()["active"] is None
    assert env.admin.detail("quiz")["body"] == "NEW"


def test_upload_errors(tmp_path):
    env = Env(tmp_path)
    assert code(env.admin.upload, "x.md", "!!not base64!!", False) == "bad_upload"
    with pytest.raises(SkillPacksError) as err:
        env.admin.upload("x.md", base64.b64encode(b"no front matter").decode(), False)
    assert err.value.code == "invalid" and err.value.extra == {"reason": "no_front_matter"}


def test_remove_drops_from_enabled_and_ends(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    env.admin.enable("quiz", True)
    env.session.start("quiz")
    env.admin.remove("quiz")
    assert env.writes[-1] == []
    assert not (tmp_path / "quiz").exists()
    assert env.session.snapshot()["counts"]["end_removed"] == 1
    assert code(env.admin.remove, "quiz") == "unknown_pack"


def test_end_active(tmp_path):
    env = Env(tmp_path)
    env.upload("quiz")
    env.admin.enable("quiz", True)
    env.session.start("quiz")
    assert env.admin.listing()["active"] == "quiz"
    assert env.admin.end() is True
    assert env.session.snapshot()["counts"]["end_admin"] == 1


def test_unavailable_without_session(tmp_path):
    admin = SkillPacksAdmin(lambda: None, lambda names: None, RoboEmotConfig)
    assert code(admin.listing) == "skill_packs_unavailable"
