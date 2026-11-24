"""The ``skill_packs.*`` config fields (SPEC 11, 11A; design 2026-11-24-skill-packs 5.8)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig, SkillPacksConfig

RANGES = {
    "max_enabled": (1, 100),
    "description_max_chars": (20, 500),
    "body_max_chars": (200, 8000),
    "upload_max_bytes": (1024, 4194304),
    "upload_max_files": (1, 500),
    "idle_end_s": (30, 3600),
}


def test_defaults():
    sp = RoboEmotConfig().skill_packs
    assert sp.enabled == []
    assert sp.max_enabled == 20
    assert sp.description_max_chars == 200
    assert sp.body_max_chars == 4000
    assert sp.upload_max_bytes == 262144
    assert sp.upload_max_files == 50
    assert sp.idle_end_s == 300


@pytest.mark.parametrize("field", sorted(RANGES))
def test_ranges_reject_out_of_range(field):
    low, high = RANGES[field]
    SkillPacksConfig(**{field: low})
    SkillPacksConfig(**{field: high})
    for bad in (low - 1, high + 1):
        with pytest.raises(ValidationError):
            SkillPacksConfig(**{field: bad})


@pytest.mark.parametrize("name", ["Quiz", "a/b", "", "-a", "a-", "a--b", "x" * 65])
def test_enabled_rejects_bad_name(name):
    with pytest.raises(ValidationError):
        SkillPacksConfig(enabled=[name])


def test_enabled_accepts_good_names():
    assert SkillPacksConfig(enabled=["quiz-night", "a1"]).enabled == ["quiz-night", "a1"]


def test_enabled_rejects_duplicates_and_over_max():
    with pytest.raises(ValidationError):
        SkillPacksConfig(enabled=["a", "a"])
    with pytest.raises(ValidationError):
        SkillPacksConfig(max_enabled=1, enabled=["a", "b"])


def test_admin_only_fields():
    for name, info in SkillPacksConfig.model_fields.items():
        extra = info.json_schema_extra or {}
        assert bool(extra.get("admin_only")) == (name != "idle_end_s"), name


def test_default_llm_skills_hold_the_two_tools():
    skills = RoboEmotConfig().conversation.llm_skills
    assert "skill_pack_start" in skills
    assert "skill_pack_end" in skills
