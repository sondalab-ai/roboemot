"""Installing and removing skill packs from web UI uploads (SPEC 11, 12; design 2026-11-24 5.2).

An upload is one ``SKILL.md`` (a ``.md`` file) or a ``.zip`` archive with one top folder or its
files at the root. Every archive rule is checked before a byte is written: the compressed and
the declared uncompressed sizes, the entry count, no absolute or ``..`` paths, no symbolic
links. Entries are then read with a byte cap (a header may lie), written into a private folder
under ``<root>/.incoming``, checked with :func:`read_pack`, and moved into place with
``os.replace``. The temporary folder is always removed, so a refused upload leaves nothing.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from ..config.schema import RoboEmotConfig
from .library import SKILL_FILE, InvalidPack, front_matter, read_pack, valid_name

INCOMING = ".incoming"
DIR_MODE = 0o700
FILE_MODE = 0o600
CHUNK_BYTES = 65536
PLACEHOLDER = "pack"
DRIVE = re.compile(r"^[A-Za-z]:")


class InstallError(Exception):
    """A refused upload: ``code`` names the rule; ``reason`` is the :class:`InvalidPack` reason
    when ``code`` is ``invalid``."""

    def __init__(self, code: str, reason: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.reason = reason


def _private_dir(path: Path) -> None:
    path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    path.chmod(DIR_MODE)


def _write(path: Path, data: bytes) -> None:
    _private_dir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _declared_name(text: bytes) -> str:
    """The front matter ``name`` when it is a valid name, else a placeholder folder name (the
    check then reports the reason)."""
    try:
        parsed = front_matter(text.decode("utf-8-sig").replace("\r\n", "\n"))
    except UnicodeDecodeError:
        return PLACEHOLDER
    name = parsed[0].get("name", "") if parsed else ""
    return name if valid_name(name) else PLACEHOLDER


def _parts(name: str) -> tuple[str, ...]:
    if name.startswith(("/", "\\")) or "\\" in name or DRIVE.match(name):
        raise InstallError("bad_path")
    parts = PurePosixPath(name).parts
    if any(p == ".." for p in parts):
        raise InstallError("bad_path")
    return tuple(p for p in parts if p not in ("", "."))


Entry = tuple[zipfile.ZipInfo, tuple[str, ...]]


def _entries(data: bytes, cfg: RoboEmotConfig) -> tuple[zipfile.ZipFile, list[Entry]]:
    caps = cfg.skill_packs
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise InstallError("not_zip") from exc
    infos = zf.infolist()
    if len(infos) > caps.upload_max_files:
        raise InstallError("too_many_files")
    if sum(i.file_size for i in infos) > caps.upload_max_bytes:
        raise InstallError("too_large")
    out = []
    for info in infos:
        parts = _parts(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise InstallError("link")
        if parts and not info.is_dir():
            out.append((info, parts))
    return zf, out


def _layout(files: list[Entry]) -> int:
    """How many leading path parts to strip: 0 for files at the root, 1 for one top folder."""
    root = [p for _, p in files if len(p) == 1]
    if any(p == (SKILL_FILE,) for p in root):
        return 0
    if root:
        raise InstallError("no_skill_md")
    tops = {p[0] for _, p in files}
    if len(tops) > 1:
        raise InstallError("bad_path")
    if not any(p[1:] == (SKILL_FILE,) for _, p in files):
        raise InstallError("no_skill_md")
    return 1


def _read_capped(zf: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int) -> bytes:
    out = bytearray()
    with zf.open(info) as f:
        while chunk := f.read(CHUNK_BYTES):
            out += chunk
            if len(out) > budget:
                raise InstallError("too_large")
    return bytes(out)


def _unpack_zip(data: bytes, cfg: RoboEmotConfig, work: Path) -> Path:
    zf, files = _entries(data, cfg)
    with zf:
        strip = _layout(files)
        budget = cfg.skill_packs.upload_max_bytes
        contents: list[tuple[tuple[str, ...], bytes]] = []
        for info, parts in files:
            blob = _read_capped(zf, info, budget)
            budget -= len(blob)
            contents.append((parts[strip:], blob))
    if strip:
        folder_name = files[0][1][0]
    else:
        skill = next(blob for rel, blob in contents if rel == (SKILL_FILE,))
        folder_name = _declared_name(skill)
    folder = work / folder_name
    _private_dir(folder)
    for rel, blob in contents:
        _write(folder.joinpath(*rel), blob)
    return folder


def install(
    root: Path, data: bytes, filename: str, cfg: RoboEmotConfig, *, replace: bool = False
) -> str:
    """Install the upload ``data`` (named ``filename``) under ``root`` and return the skill
    pack's name. Raises :class:`InstallError`. An existing skill pack of that name is replaced
    only with ``replace``."""
    if len(data) > cfg.skill_packs.upload_max_bytes:
        raise InstallError("too_large")
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in (".md", ".zip"):
        raise InstallError("not_zip")
    _private_dir(root)
    incoming = root / INCOMING
    _private_dir(incoming)
    work = Path(tempfile.mkdtemp(dir=incoming))
    try:
        if suffix == ".md":
            folder = work / _declared_name(data)
            _private_dir(folder)
            _write(folder / SKILL_FILE, data)
        else:
            folder = _unpack_zip(data, cfg, work)
        pack = read_pack(folder, cfg)
        if isinstance(pack, InvalidPack):
            raise InstallError("invalid", pack.reason)
        target = root / pack.name
        if target.exists():
            if not replace:
                raise InstallError("exists")
            os.replace(target, work / f"{INCOMING}-old")
        os.replace(folder, target)
        return pack.name
    finally:
        shutil.rmtree(work, ignore_errors=True)


def remove(root: Path, name: str) -> bool:
    """Delete the skill pack folder ``name``; False when the name is not valid or not there."""
    if not valid_name(name):
        return False
    target = root / name
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True
