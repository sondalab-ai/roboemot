"""The shipped example skill packs pass the checks (SPEC 11; design 2026-11-24 section 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.library import SkillPack, read_pack

EXAMPLES = Path(__file__).resolve().parents[3] / "configs" / "skill_packs"


@pytest.mark.parametrize("name", ["quiz-night", "bedtime-story"])
def test_example_is_valid(name):
    pack = read_pack(EXAMPLES / name, RoboEmotConfig())
    assert isinstance(pack, SkillPack), pack
    assert pack.ignored_files == ()
