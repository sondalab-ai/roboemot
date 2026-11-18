"""The sleep voice tool (SPEC 11, Voice tools, Sleep; SPEC 7A, Requested nap).

``go_to_sleep`` arms a requested nap through ``ToolHost.nap`` and returns its status at once:
``ok`` (armed: the nap starts after the turn's last reply audio), ``already_resting``,
``disabled`` or ``asleep``. "Wake up" needs no tool: the wake word wakes the robot. It has no
``enabled`` switch; ``conversation.llm_skills`` is its switch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..skills.host import ToolHost
from ..skills.registry import SkillResult, skill

DESCRIPTION = (
    "Take a nap now. Call it whenever the person tells you to sleep, rest or take a nap, in "
    "any words and in any language, even a single word such as 'dormi'. Never answer that you "
    "do not sleep or are always awake: call it and say a short good night in the same turn. "
    "The robot's name wakes you"
)


def skills(host: ToolHost) -> list[Callable[..., Any]]:
    """The sleep tool over ``host``: one skill, ``go_to_sleep``, available while the run has a
    nap part (``host.nap``, read on every check). Side-effect free: it only builds the skill,
    so the realtime check may call it with a throwaway host."""

    @skill(llm_visible=True, description=DESCRIPTION, available=lambda: host.nap is not None)
    def go_to_sleep() -> SkillResult:
        nap = host.nap
        status = nap() if nap is not None else "disabled"
        return SkillResult({"status": status})

    return [go_to_sleep]
