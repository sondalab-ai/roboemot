"""One describe request for one room picture (SPEC 8, Surroundings snapshots; designs
2026-11-24, 5.4; 2026-11-25).

The fixed English prompt of ``llm/prompts/surroundings_room.en.txt`` with the view count and the
span, the JPEG and ``surroundings.describe_max_tokens`` go to one image route (the service
picks it and falls back to the next) through ``Router.describe_image``; the reply comes back on
one line, cut to ``room.max_chars`` like a registry name. No names, no history, no persona: the
request holds nothing about the people the robot knows.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig
from ..conversation.summaries import one_line
from ..llm.prompt import surroundings_prompt


class EmptyDescription(RuntimeError):
    """The route answered with no text."""


async def describe(
    router: Any, route: LLMRoute, jpeg: bytes, views: int, span_deg: int, cfg: RoboEmotConfig
) -> str:
    """The description of ``jpeg`` (``views`` views joined left to right over ``span_deg``) by
    ``route``, one line and capped. Raises ``EmptyDescription`` on a blank reply; the router's
    errors propagate."""
    s = cfg.surroundings
    prompt = surroundings_prompt().format(views=views, span=span_deg)
    reply = await router.describe_image(
        route, prompt, jpeg, s.describe_max_tokens, s.describe_first_token_s
    )
    text = one_line(reply, s.room.max_chars).strip()
    if not text:
        raise EmptyDescription("room")
    return text
