"""The active skill pack (SPEC 8, Conversation state; design 2026-11-24-skill-packs 5.4, 5.5).

Core, not the model, holds which skill pack is active: a realtime reconnect replays only user
and robot text, so the body must come back from here on every turn. :meth:`instructions` is the
text the conversation worker appends to every turn's instructions: the catalogue of usable skill
packs, the active block, then each pending end line once (a model that keeps its history,
Gemini Live, otherwise keeps following an ended skill pack).
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

from ..config.schema import RoboEmotConfig
from ..llm.prompt import skill_packs_fit
from .library import SkillPack, SkillPackLibrary
from .prompt import active_block, catalogue_block, ended_line

log = logging.getLogger(__name__)

END_REASONS = frozenset(
    {"tool", "conversation_end", "asleep", "resting", "disabled", "removed", "admin", "idle"}
)


TOOLS = ("skill_pack_start", "skill_pack_end")


def tools_listed(cfg: RoboEmotConfig) -> bool:
    """Whether ``conversation.llm_skills`` offers both skill pack tools: without them the
    catalogue would point the model at a tool it does not have."""
    return all(t in cfg.conversation.llm_skills for t in TOOLS)


def normalise(name: str) -> str:
    """A model's spelling of a skill pack name in folder form: trimmed, lower case, spaces and
    underscores as hyphens."""
    return "-".join(name.strip().lower().replace("_", " ").split())


class SkillPackSession:
    """The one active skill pack and its end reasons; thread-safe (one lock). ``clock`` is a
    monotonic clock in seconds."""

    def __init__(
        self,
        library: SkillPackLibrary,
        config: Callable[[], RoboEmotConfig],
        clock: Callable[[], float],
    ) -> None:
        self.library = library
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._active: SkillPack | None = None
        self._started_at = 0.0
        self._heard_at = 0.0
        self._ended: list[str] = []
        self._counts: Counter[str] = Counter()
        self._too_large: dict[str, str] = {}

    def usable(self, cfg: RoboEmotConfig) -> list[SkillPack]:
        """The enabled valid skill packs that fit every route's budget, in name order. A skill
        pack that would break a route's budget is skipped and logged once with the route. None
        is usable while ``conversation.llm_skills`` does not offer both skill pack tools."""
        if not tools_listed(cfg):
            return []
        accepted: list[SkillPack] = []
        too_large: dict[str, str] = {}
        for pack in self.library.enabled(cfg):
            route = skill_packs_fit(cfg, [*accepted, pack])
            if route is None:
                accepted.append(pack)
            else:
                too_large[pack.name] = route
        with self._lock:
            for name, route in too_large.items():
                if self._too_large.get(name) != route:
                    log.warning("skill pack %s skipped: too large for route %s", name, route)
            self._too_large = too_large
        return accepted

    def too_large(self) -> dict[str, str]:
        """Enabled skill packs skipped by the budget, each with the route it did not fit."""
        self.usable(self._config())
        with self._lock:
            return dict(self._too_large)

    def enabled_names(self) -> list[str]:
        """Names of the usable skill packs."""
        return [p.name for p in self.usable(self._config())]

    def start(self, name: str) -> tuple[str, str | None]:
        """Make the usable skill pack ``name`` (normalised) active and return ``("started",
        body)``; ``("not_found", None)`` otherwise. A replaced skill pack queues its end line."""
        wanted = normalise(name)
        pack = next((p for p in self.usable(self._config()) if p.name == wanted), None)
        if pack is None:
            return "not_found", None
        now = self._clock()
        with self._lock:
            if self._active is not None and self._active.name != pack.name:
                self._ended.append(self._active.name)
                self._counts["end_replaced"] += 1
            self._active = pack
            self._started_at = self._heard_at = now
            self._counts["starts"] += 1
        log.info("skill pack %s started", pack.name)
        return "started", pack.body

    def end(self, reason: str) -> bool:
        """End the active skill pack for ``reason`` (one of ``END_REASONS``); False when none
        was active."""
        if reason not in END_REASONS:
            raise ValueError(f"unknown skill pack end reason {reason!r}")
        with self._lock:
            pack = self._active
            if pack is None:
                return False
            self._active = None
            self._ended.append(pack.name)
            self._counts[f"end_{reason}"] += 1
        log.info("skill pack %s ended: %s", pack.name, reason)
        return True

    def heard_addressed(self) -> None:
        """Speech addressed to the robot was heard: the idle time starts again."""
        with self._lock:
            self._heard_at = self._clock()

    def tick(self) -> None:
        """End the active skill pack after ``skill_packs.idle_end_s`` without addressed speech."""
        idle_s = self._config().skill_packs.idle_end_s
        with self._lock:
            idle = self._active is not None and self._clock() - self._heard_at > idle_s
        if idle:
            self.end("idle")

    def on_config(self, cfg: RoboEmotConfig) -> None:
        """A config change: the active skill pack ends when it is no longer usable."""
        with self._lock:
            active = self._active
        if active is not None and active.name not in {p.name for p in self.usable(cfg)}:
            self.end("disabled")

    def instructions(self) -> str:
        """The catalogue block, the active block and each pending end line (sent once)."""
        packs = self.usable(self._config())
        with self._lock:
            active, ended = self._active, self._ended
            self._ended = []
        parts = [catalogue_block(packs)]
        if active is not None:
            parts.append(active_block(active))
        parts.extend(ended_line(n) for n in ended if active is None or n != active.name)
        return "\n".join(p for p in parts if p)

    def snapshot(self) -> dict[str, Any]:
        """The debug report's view: the active name, its start time and the counts (no body)."""
        with self._lock:
            active = self._active
            return {
                "active": active.name if active else None,
                "started_at": self._started_at if active else None,
                "counts": dict(self._counts),
            }

    def report(self) -> dict[str, Any]:
        """The debug report section (SPEC 12A): installed, enabled and invalid skill packs with
        the reasons, enabled names with no valid folder, the ones too large for a route, the
        active name and the counts. Names and reasons only, never a body."""
        cfg = self._config()
        packs = self.library.packs()
        valid = {p.name for p in packs if isinstance(p, SkillPack)}
        invalid = {p.name: p.reason for p in packs if not isinstance(p, SkillPack)}
        return {
            "installed": len(packs),
            "enabled": len(self.usable(cfg)),
            "invalid": invalid,
            "missing": sorted(n for n in cfg.skill_packs.enabled if n not in valid),
            "too_large": self.too_large(),
            "tools_listed": tools_listed(cfg),
            **self.snapshot(),
        }
