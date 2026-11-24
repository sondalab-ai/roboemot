"""The web UI's skill pack admin (SPEC 11A, 12; design 2026-11-24-skill-packs 5.2, 5.3, 5.6).

The API checks the PIN session, CSRF, the re-entered PIN and the confirmation first; this class
runs the list, the upload, turning on and off, the removal, the rescan and the End button.
Turning a skill pack on writes ``skill_packs.enabled`` through the Settings editor, so the edit is
kept like any other admin-only setting. The list never carries a body; only :meth:`detail` does,
for the admin to read before turning a skill pack on.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from typing import Any

from ..config.schema import RoboEmotConfig
from ..llm.prompt import skill_packs_fit
from .install import InstallError, install, remove
from .library import InvalidPack, SkillPack, valid_name
from .session import SkillPackSession, tools_listed


class SkillPacksError(Exception):
    """A refused admin action: ``code`` for the web UI, ``extra`` with a field it may show
    (``route``, ``reason``)."""

    def __init__(self, code: str, extra: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.extra = extra or {}


class SkillPacksAdmin:
    """``session()`` is the run's :class:`SkillPackSession` (None without a conversation);
    ``write_enabled(names)`` keeps a new ``skill_packs.enabled`` (the Settings editor);
    ``config()`` is the live configuration."""

    def __init__(
        self,
        session: Callable[[], SkillPackSession | None],
        write_enabled: Callable[[list[str]], Any],
        config: Callable[[], RoboEmotConfig],
    ) -> None:
        self._session = session
        self._write = write_enabled
        self._config = config

    def _live(self) -> SkillPackSession:
        session = self._session()
        if session is None:
            raise SkillPacksError("skill_packs_unavailable")
        return session

    def listing(self) -> dict[str, Any]:
        """Every folder and every enabled name with its state, and the active name."""
        session = self._live()
        cfg = self._config()
        enabled = set(cfg.skill_packs.enabled)
        too_large = session.too_large()
        rows: list[dict[str, Any]] = []
        for pack in session.library.packs():
            row: dict[str, Any] = {
                "name": pack.name,
                "reason": None,
                "route": None,
                "enabled": pack.name in enabled,
            }
            if isinstance(pack, InvalidPack):
                row.update(state="invalid", reason=pack.reason)
            else:
                row.update(
                    description=pack.description,
                    meta=pack.meta,
                    ignored_files=list(pack.ignored_files),
                    size_bytes=pack.size_bytes,
                    state="off",
                )
                if pack.name in too_large:
                    row.update(state="too_large", route=too_large[pack.name])
                elif pack.name in enabled:
                    row["state"] = "on"
            rows.append(row)
        listed = {r["name"] for r in rows}
        rows += [
            {"name": n, "state": "missing", "reason": None, "route": None, "enabled": True}
            for n in sorted(enabled - listed)
        ]
        return {
            "packs": rows,
            "active": session.snapshot()["active"],
            "tools_listed": tools_listed(cfg),
        }

    def _pack(self, name: str) -> SkillPack:
        pack = self._live().library.get(name) if valid_name(name) else None
        if pack is None:
            raise SkillPacksError("unknown_pack")
        return pack

    def detail(self, name: str) -> dict[str, Any]:
        """One valid skill pack with its body, for the admin to read."""
        pack = self._pack(name)
        return {"name": pack.name, "description": pack.description, "body": pack.body}

    def enable(self, name: str, on: bool) -> None:
        """Turn ``name`` on or off. Turning on is refused when the tools are not offered, the
        list is full, or a route's budget cannot hold it."""
        cfg = self._config()
        names = [n for n in cfg.skill_packs.enabled if n != name]
        if not on:
            self._write(names)
            return
        pack = self._pack(name)
        if not tools_listed(cfg):
            raise SkillPacksError("llm_skills_missing")
        if len(names) + 1 > cfg.skill_packs.max_enabled:
            raise SkillPacksError("max_enabled")
        others = [p for p in self._live().usable(cfg) if p.name != name]
        route = skill_packs_fit(cfg, [*others, pack])
        if route is not None:
            raise SkillPacksError("too_large_for_route", {"route": route})
        self._write(sorted([*names, name]))

    def upload(self, filename: str, data_b64: str, replace: bool) -> str:
        """Install an upload (base64). The skill pack is always off afterwards, even when its
        name was already on (a replacement, or a missing folder): text reaches the model only
        after the admin turns it on (design G2)."""
        session = self._live()
        try:
            data = base64.b64decode(data_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SkillPacksError("bad_upload") from exc
        try:
            name = install(session.library.root, data, filename, self._config(), replace=replace)
        except InstallError as exc:
            raise SkillPacksError(exc.code, {"reason": exc.reason} if exc.reason else None) from exc
        if session.snapshot()["active"] == name:
            session.end("disabled")
        self._drop(name)
        session.library.scan()
        return name

    def _drop(self, name: str) -> None:
        names = self._config().skill_packs.enabled
        if name in names:
            self._write([n for n in names if n != name])

    def remove(self, name: str) -> None:
        """Delete a skill pack folder; it leaves the enabled list and ends if active."""
        session = self._live()
        if not remove(session.library.root, name):
            raise SkillPacksError("unknown_pack")
        if session.snapshot()["active"] == name:
            session.end("removed")
        self._drop(name)
        session.library.scan()

    def rescan(self) -> None:
        """Read the folders again (after a developer copied one in)."""
        self._live().library.scan()

    def end(self) -> bool:
        """The End button: end the active skill pack."""
        return self._live().end("admin")
