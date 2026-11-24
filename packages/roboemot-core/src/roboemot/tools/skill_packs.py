"""The skill pack voice tools (SPEC 11, Voice tools; design 2026-11-24-skill-packs 5.4).

``skill_pack_start`` makes an enabled skill pack active and returns its text, so the model acts
on it in the same turn; core then keeps the text in every turn's instructions until the skill
pack ends. The name is free text, not a choices enum: Gemini Live fixes its tool list when a
session opens, so a skill pack turned on later must still be startable. Both tools are offered
only while at least one skill pack is usable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..skills.host import ToolHost
from ..skills.registry import SkillResult, skill

START_DESCRIPTION = (
    "Start an installed skill pack, an activity listed under 'Skill packs installed on you'. "
    "Call it only when the person asks for an activity that matches one of those lines, in "
    "any words and in any language. Pass the name exactly as listed. Then follow the returned "
    "instructions in the same turn"
)
END_DESCRIPTION = (
    "End the active skill pack: call it when the person wants to stop the activity or when "
    "the activity is over"
)


def skills(host: ToolHost) -> list[Callable[..., Any]]:
    """The skill pack tools over ``host.skill_packs`` (read on every check and call). Side-effect
    free: it only builds the skills."""

    def available() -> bool:
        session = host.skill_packs
        return session is not None and bool(session.enabled_names())

    @skill(llm_visible=True, description=START_DESCRIPTION, available=available)
    def skill_pack_start(name: str) -> SkillResult:
        session = host.skill_packs
        if session is None:
            return SkillResult({"status": "not_found", "enabled": []})
        status, body = session.start(name)
        if body is None:
            return SkillResult({"status": status, "enabled": session.enabled_names()})
        return SkillResult({"status": status, "instructions": body})

    @skill(llm_visible=True, description=END_DESCRIPTION, available=available, follow_up=False)
    def skill_pack_end() -> SkillResult:
        session = host.skill_packs
        ended = session is not None and session.end("tool")
        return SkillResult({"status": "ended" if ended else "none_active"})

    return [skill_pack_start, skill_pack_end]
