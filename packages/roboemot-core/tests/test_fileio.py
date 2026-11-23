"""SPEC 11A, 12: the shared atomic file write."""

from __future__ import annotations

import stat

from roboemot.fileio import replace_file


def test_replace_file_is_atomic_and_owner_only(tmp_path):
    target = tmp_path / "store.json"
    target.write_text("old")
    target.chmod(0o644)
    replace_file(target, "new", 0o600)
    assert target.read_text() == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["store.json"]  # no temp file left


def test_replace_file_replaces_a_symlink_itself_not_its_target(tmp_path):
    real = tmp_path / "real"
    real.write_text("kept")
    link = tmp_path / "link"
    link.symlink_to(real)
    replace_file(link, "new", 0o600)
    assert real.read_text() == "kept"
    assert not link.is_symlink() and link.read_text() == "new"


def test_replace_file_writes_bytes_as_they_are(tmp_path):
    target = tmp_path / "clip.wav"
    replace_file(target, b"RIFF\x00\xff", 0o600)
    assert target.read_bytes() == b"RIFF\x00\xff"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
