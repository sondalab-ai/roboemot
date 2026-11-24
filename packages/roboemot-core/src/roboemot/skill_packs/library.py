"""Reading and checking skill pack folders (SPEC 11; design 2026-11-24-skill-packs 5.1).

A skill pack is ``<root>/<name>/SKILL.md``: a front matter block (``---`` lines around
``key: value`` lines, with ``name`` and ``description`` required) and a plain-text body. A folder
that breaks a rule is an :class:`InvalidPack` with a reason; it is listed but never reaches a
model. Other files in the folder are kept and ignored.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config.schema import SKILL_PACK_NAME, SKILL_PACK_NAME_MAX_CHARS, RoboEmotConfig

log = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
FENCE = "---"


@dataclass(frozen=True)
class SkillPack:
    """A valid skill pack: its name (the folder name), the one-line description, the body,
    the other front matter keys, the files it holds besides ``SKILL.md`` and its size."""

    name: str
    description: str
    body: str
    meta: dict[str, str]
    ignored_files: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class InvalidPack:
    """A folder that is not a valid skill pack, with the reason the web UI shows."""

    name: str
    reason: str


def valid_name(name: str) -> bool:
    """True for a skill pack name: a-z, 0-9 and single hyphens, at most 64 characters."""
    return len(name) <= SKILL_PACK_NAME_MAX_CHARS and SKILL_PACK_NAME.fullmatch(name) is not None


def _front_matter(text: str) -> tuple[dict[str, str], str] | None:
    """The ``key: value`` pairs between the opening and closing fence, and the body after it;
    None when the text has no complete front matter block."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != FENCE:
        return None
    for end in range(1, len(lines)):
        if lines[end].strip() == FENCE:
            break
    else:
        return None
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = value.strip()
    return meta, "\n".join(lines[end + 1 :]).strip()


def read_pack(folder: Path, cfg: RoboEmotConfig) -> SkillPack | InvalidPack:
    """The skill pack in ``folder``, checked against the ``skill_packs`` caps of ``cfg``."""
    name = folder.name
    path = folder / SKILL_FILE
    if not path.is_file():
        return InvalidPack(name, "no_skill_md")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return InvalidPack(name, "not_utf8")
    parsed = _front_matter(text)
    if parsed is None:
        return InvalidPack(name, "no_front_matter")
    meta, body = parsed
    declared = meta.pop("name", "")
    if not valid_name(declared):
        return InvalidPack(name, "bad_name")
    if declared != name:
        return InvalidPack(name, "name_mismatch")
    description = meta.pop("description", "")
    caps = cfg.skill_packs
    if not description:
        return InvalidPack(name, "no_description")
    if len(description) > caps.description_max_chars:
        return InvalidPack(name, "description_too_long")
    if not body:
        return InvalidPack(name, "body_empty")
    if len(body) > caps.body_max_chars:
        return InvalidPack(name, "body_too_long")
    others = sorted(
        str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p != path
    )
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    return SkillPack(name, description, body, meta, tuple(others), size)


class SkillPackLibrary:
    """The skill packs under ``root``, read by :meth:`scan` (at start, after an upload or a
    removal, and on the web UI's Rescan) with the caps of ``config()`` at that time. Folders
    whose name starts with a dot (``.incoming``) and plain files are skipped."""

    def __init__(self, root: Path, config: Callable[[], RoboEmotConfig]) -> None:
        self.root = root
        self._config = config
        self._lock = threading.Lock()
        self._packs: dict[str, SkillPack | InvalidPack] = {}
        self._missing_logged: set[str] = set()

    def scan(self) -> None:
        """Read every folder again; a skill pack that became invalid is logged with its reason."""
        cfg = self._config()
        found: dict[str, SkillPack | InvalidPack] = {}
        if self.root.is_dir():
            for folder in sorted(self.root.iterdir()):
                if folder.name.startswith(".") or not folder.is_dir():
                    continue
                pack = read_pack(folder, cfg)
                if isinstance(pack, InvalidPack):
                    log.info("skill pack %s invalid: %s", pack.name, pack.reason)
                found[folder.name] = pack
        with self._lock:
            self._packs = found
            self._missing_logged = set()

    def packs(self) -> list[SkillPack | InvalidPack]:
        """Every folder read by the last scan, valid or not, in name order."""
        with self._lock:
            return [self._packs[n] for n in sorted(self._packs)]

    def get(self, name: str) -> SkillPack | None:
        """The valid skill pack called ``name``, else None."""
        with self._lock:
            pack = self._packs.get(name)
        return pack if isinstance(pack, SkillPack) else None

    def enabled(self, cfg: RoboEmotConfig) -> list[SkillPack]:
        """The valid skill packs named in ``cfg.skill_packs.enabled``, in name order. A name with
        no valid folder is skipped and logged once per scan."""
        out: list[SkillPack] = []
        for name in sorted(cfg.skill_packs.enabled):
            pack = self.get(name)
            if pack is not None:
                out.append(pack)
                continue
            with self._lock:
                first = name not in self._missing_logged
                self._missing_logged.add(name)
            if first:
                log.info("skill pack %s is on but missing or invalid: skipped", name)
        return out
