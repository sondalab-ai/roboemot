"""Skills the LLM may call (SPEC 11, LLM tools)."""

from .registry import (
    ChoiceCap,
    Skill,
    SkillContext,
    SkillRegistry,
    SkillResult,
    ToolOutcome,
    execute,
    skill,
)
from .schema import ArgError, Desc, MaxLen, Range

__all__ = [
    "ArgError",
    "ChoiceCap",
    "Desc",
    "MaxLen",
    "Range",
    "Skill",
    "SkillContext",
    "SkillRegistry",
    "SkillResult",
    "ToolOutcome",
    "execute",
    "skill",
]
