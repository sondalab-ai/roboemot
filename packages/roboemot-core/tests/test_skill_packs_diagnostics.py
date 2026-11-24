"""The skill packs section of the debug report (SPEC 12A; design 2026-11-24 5.7)."""

from __future__ import annotations

import json

from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession


def test_report_counts_reasons_and_no_body(tmp_path):
    for name in ("quiz", "bed"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: D.\n---\nSECRET-BODY"
        )
    (tmp_path / "broken").mkdir()
    cfg = RoboEmotConfig(skill_packs={"enabled": ["quiz", "gone"]})
    lib = SkillPackLibrary(tmp_path, lambda: cfg)
    lib.scan()
    session = SkillPackSession(lib, lambda: cfg, lambda: 0.0)
    session.start("quiz")
    report = session.report()
    assert report["installed"] == 3
    assert report["enabled"] == 1
    assert report["invalid"] == {"broken": "no_skill_md"}
    assert report["missing"] == ["gone"]
    assert report["active"] == "quiz"
    assert report["counts"]["starts"] == 1
    assert "SECRET-BODY" not in json.dumps(report)
