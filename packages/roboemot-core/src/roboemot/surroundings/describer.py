"""One describe request for one snapshot (SPEC 8, Surroundings snapshots; design 5.4).

The fixed English prompt of ``llm/prompts/surroundings_<kind>.en.txt`` (a shown thing's prompt
carries the detector's category as a hint), the JPEG and ``surroundings.describe_max_tokens`` go
to the image route through ``Router.describe_image``; the reply comes back on one line, cut to
``shown.max_chars`` or ``room.max_chars`` like a registry name. No names, no history, no
persona: the request holds nothing about the people the robot knows.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import LLMRoute, RoboEmotConfig
from ..conversation.summaries import one_line
from ..llm.prompt import surroundings_prompt
from .types import Kind


class EmptyDescription(RuntimeError):
    """The route answered with no text."""


async def describe(
    router: Any, route: LLMRoute, kind: Kind, jpeg: bytes, hint: str, cfg: RoboEmotConfig
) -> str:
    """The description of ``jpeg`` by ``route``, one line and capped. Raises
    ``EmptyDescription`` on a blank reply; the router's errors propagate."""
    s = cfg.surroundings
    prompt = surroundings_prompt(kind).replace("{hint}", one_line(hint, s.shown.max_chars))
    reply = await router.describe_image(
        route, prompt, jpeg, s.describe_max_tokens, s.describe_first_token_s
    )
    cap = s.shown.max_chars if kind == "shown" else s.room.max_chars
    text = one_line(reply, cap).strip()
    if not text:
        raise EmptyDescription(kind)
    return text
