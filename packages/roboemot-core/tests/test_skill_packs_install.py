"""Installing and removing skill packs from uploads (SPEC 11, 12; design 2026-11-24 5.2)."""

from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.install import InstallError, install, remove


def cfg(**sp) -> RoboEmotConfig:
    return RoboEmotConfig(skill_packs=sp)


def skill(name: str = "quiz-night", body: str = "Ask five questions.") -> str:
    return f"---\nname: {name}\ndescription: A quiz.\n---\n{body}\n"


def zipped(entries: dict[str, str | bytes], links: tuple[str, ...] = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        for name in links:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


def only_dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def code_of(root: Path, data: bytes, filename: str = "p.zip", **conf) -> str:
    with pytest.raises(InstallError) as err:
        install(root, data, filename, cfg(**conf))
    return err.value.code


def test_zip_with_top_folder(tmp_path):
    data = zipped({"quiz-night/SKILL.md": skill(), "quiz-night/run.py": "x"})
    assert install(tmp_path, data, "q.zip", cfg()) == "quiz-night"
    assert (tmp_path / "quiz-night" / "SKILL.md").read_text().startswith("---")
    assert (tmp_path / "quiz-night" / "run.py").exists()


def test_zip_with_files_at_root(tmp_path):
    assert install(tmp_path, zipped({"SKILL.md": skill()}), "q.zip", cfg()) == "quiz-night"


def test_md_upload_uses_front_matter_name(tmp_path):
    assert install(tmp_path, skill("bedtime").encode(), "anything.md", cfg()) == "bedtime"
    assert (tmp_path / "bedtime" / "SKILL.md").is_file()


def test_too_large_compressed(tmp_path):
    data = zipped({"SKILL.md": skill(body="x" * 3000)})
    assert code_of(tmp_path, data + b"\0" * 2000, upload_max_bytes=1024) == "too_large"


def test_zip_bomb_refused(tmp_path):
    data = zipped({"SKILL.md": skill(), "pad.txt": "0" * 200_000})
    assert len(data) < 2048
    assert code_of(tmp_path, data, upload_max_bytes=2048) == "too_large"
    assert only_dirs(tmp_path) in ([], [".incoming"])
    assert not any((tmp_path / ".incoming").glob("*")) if (tmp_path / ".incoming").exists() else 1


def test_too_many_files(tmp_path):
    data = zipped({"SKILL.md": skill(), "a": "1", "b": "2"})
    assert code_of(tmp_path, data, upload_max_files=2) == "too_many_files"


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs.txt", "x/../../evil.txt", "C:/x.txt"])
def test_bad_paths(tmp_path, bad):
    assert code_of(tmp_path, zipped({"SKILL.md": skill(), bad: "x"})) == "bad_path"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_two_top_folders_is_bad_path(tmp_path):
    data = zipped({"a/SKILL.md": skill("a"), "b/x.txt": "x"})
    assert code_of(tmp_path, data) == "bad_path"


def test_symlink_refused(tmp_path):
    assert code_of(tmp_path, zipped({"SKILL.md": skill()}, links=("link",))) == "link"


def test_not_zip(tmp_path):
    assert code_of(tmp_path, b"not a zip at all") == "not_zip"
    assert code_of(tmp_path, b"x", "p.exe") == "not_zip"


def test_no_skill_md(tmp_path):
    assert code_of(tmp_path, zipped({"README.md": "x"})) == "no_skill_md"


def test_invalid_carries_reason(tmp_path):
    with pytest.raises(InstallError) as err:
        install(tmp_path, zipped({"SKILL.md": "no front matter"}), "p.zip", cfg())
    assert err.value.code == "invalid"
    assert err.value.reason == "no_front_matter"


def test_folder_name_must_match_declared_name(tmp_path):
    data = zipped({"other/SKILL.md": skill()})
    assert code_of(tmp_path, data) == "invalid"


def test_failed_install_leaves_nothing(tmp_path):
    for data in (zipped({"SKILL.md": "bad"}), zipped({"SKILL.md": skill(), "../x": "1"})):
        with pytest.raises(InstallError):
            install(tmp_path, data, "p.zip", cfg())
    assert [p for p in only_dirs(tmp_path) if p != ".incoming"] == []
    incoming = tmp_path / ".incoming"
    assert not incoming.exists() or not any(incoming.iterdir())


def test_exists_without_replace(tmp_path):
    install(tmp_path, skill().encode(), "q.md", cfg())
    assert code_of(tmp_path, skill(body="new").encode(), "q.md") == "exists"


def test_replace_swaps_folder(tmp_path):
    install(tmp_path, zipped({"SKILL.md": skill(), "old.txt": "1"}), "q.zip", cfg())
    install(tmp_path, skill(body="new").encode(), "q.md", cfg(), replace=True)
    assert "new" in (tmp_path / "quiz-night" / "SKILL.md").read_text()
    assert not (tmp_path / "quiz-night" / "old.txt").exists()


def test_modes_0600_0700(tmp_path):
    data = zipped({"quiz-night/SKILL.md": skill(), "quiz-night/d/x.txt": "1"})
    install(tmp_path, data, "q.zip", cfg())
    folder = tmp_path / "quiz-night"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "d").stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "SKILL.md").stat().st_mode) == 0o600
    assert stat.S_IMODE((folder / "d" / "x.txt").stat().st_mode) == 0o600


def test_remove(tmp_path):
    install(tmp_path, skill().encode(), "q.md", cfg())
    assert remove(tmp_path, "quiz-night") is True
    assert not (tmp_path / "quiz-night").exists()
    assert remove(tmp_path, "quiz-night") is False
    assert remove(tmp_path, "../etc") is False
