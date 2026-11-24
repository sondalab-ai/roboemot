"""Reading and checking skill pack folders (SPEC 11; design 2026-11-24-skill-packs 5.1)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.paths import skill_packs_dir
from roboemot.skill_packs.library import InvalidPack, SkillPack, SkillPackLibrary, read_pack

BODY = "Ask five questions, one at a time. SECRET-BODY-TEXT"


def cfg(**sp) -> RoboEmotConfig:
    return RoboEmotConfig(skill_packs=sp)


def make(root: Path, name: str, text: str | bytes | None, folder: str | None = None) -> Path:
    path = root / (folder or name)
    path.mkdir(parents=True)
    if text is not None:
        data = text.encode() if isinstance(text, str) else text
        (path / "SKILL.md").write_bytes(data)
    return path


def skill(name: str = "quiz-night", desc: str = "A quiz game.", body: str = BODY, extra=""):
    return f"---\nname: {name}\ndescription: {desc}\n{extra}---\n\n{body}\n"


def test_skill_packs_dir_is_under_the_data_folder(tmp_path):
    assert skill_packs_dir(str(tmp_path)) == tmp_path / "skill_packs"


def test_valid_pack_fields(tmp_path):
    folder = make(tmp_path, "quiz-night", skill(extra="author: x\n"))
    (folder / "run.py").write_text("print(1)")
    pack = read_pack(folder, cfg())
    assert isinstance(pack, SkillPack)
    assert pack.name == "quiz-night"
    assert pack.description == "A quiz game."
    assert pack.body == BODY
    assert pack.meta == {"author": "x"}
    assert pack.ignored_files == ("run.py",)
    assert pack.size_bytes > 0


def test_bom_and_crlf_are_accepted(tmp_path):
    text = "﻿" + skill().replace("\n", "\r\n")
    pack = read_pack(make(tmp_path, "quiz-night", text), cfg())
    assert isinstance(pack, SkillPack)
    assert pack.body == BODY


@pytest.mark.parametrize(
    ("folder", "text", "reason", "conf"),
    [
        ("quiz-night", None, "no_skill_md", {}),
        ("quiz-night", b"\xff\xfe\x00bad", "not_utf8", {}),
        ("quiz-night", "just text, no front matter", "no_front_matter", {}),
        ("quiz-night", "---\nname: quiz-night\ndescription: x\n", "no_front_matter", {}),
        ("Quiz", skill(name="Quiz"), "bad_name", {}),
        ("quiz-night", skill(name="other"), "name_mismatch", {}),
        ("quiz-night", skill(desc=""), "no_description", {}),
        ("quiz-night", skill(desc="x" * 21), "description_too_long", {"description_max_chars": 20}),
        ("quiz-night", skill(body="  "), "body_empty", {}),
        ("quiz-night", skill(body="x" * 201), "body_too_long", {"body_max_chars": 200}),
    ],
)
def test_invalid_reasons(tmp_path, folder, text, reason, conf):
    name = folder
    pack = read_pack(make(tmp_path, name, text, folder), cfg(**conf))
    assert isinstance(pack, InvalidPack)
    assert pack.reason == reason
    assert pack.name == folder


def test_scan_skips_dot_folders_and_files(tmp_path):
    make(tmp_path, "b-pack", skill(name="b-pack"))
    make(tmp_path, "a-pack", skill(name="a-pack"))
    (tmp_path / ".incoming").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    lib = SkillPackLibrary(tmp_path, cfg)
    lib.scan()
    assert [p.name for p in lib.packs()] == ["a-pack", "b-pack"]
    assert lib.get("a-pack") is not None
    assert lib.get("missing") is None


def test_scan_of_a_missing_root_is_empty(tmp_path):
    lib = SkillPackLibrary(tmp_path / "none", cfg)
    lib.scan()
    assert lib.packs() == []


def test_enabled_skips_missing_and_invalid(tmp_path, caplog):
    make(tmp_path, "a-pack", skill(name="a-pack"))
    make(tmp_path, "bad", "no front matter")
    lib = SkillPackLibrary(tmp_path, cfg)
    lib.scan()
    with caplog.at_level(logging.INFO):
        got = lib.enabled(cfg(enabled=["bad", "gone", "a-pack"]))
    assert [p.name for p in got] == ["a-pack"]
    assert "gone" in caplog.text


def test_get_returns_only_valid(tmp_path):
    make(tmp_path, "bad", "no front matter")
    lib = SkillPackLibrary(tmp_path, cfg)
    lib.scan()
    assert lib.get("bad") is None


def test_body_never_logged(tmp_path, caplog):
    make(tmp_path, "quiz-night", skill())
    make(tmp_path, "toolong", skill(name="toolong", body="SECRET-BODY-TEXT" * 400))
    with caplog.at_level(logging.DEBUG):
        lib = SkillPackLibrary(tmp_path, lambda: cfg(body_max_chars=200))
        lib.scan()
        lib.enabled(cfg(enabled=["quiz-night", "toolong"], body_max_chars=200))
    assert "SECRET-BODY-TEXT" not in caplog.text
