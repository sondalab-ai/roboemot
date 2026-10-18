"""The default LLM-visible skills (SPEC 11, LLM tools): play, look_at, express_emotion,
remember_preference, stop_talking.

They take their effects from a :class:`SkillEnv` of callables the app provides, so this module
knows no robot. All five run inline on the conversation's event loop (they only post a request),
so none can outlive its turn. None of them can address a joint: ``play`` names an
expression of the manifest's
map (rendered by the expression renderer under the register, persona, circadian and minor
caps), ``look_at`` a category or a name in view (a hint to the attention selector, so the gaze
controller and the limiter still decide), ``express_emotion`` a label and a strength (an affect
impulse inside the engine's per-turn budget).

Minors (decision 2026-10-18): all five are allowed. ``play``, ``look_at`` and ``express_emotion``
go through the paths that already apply the minor motion caps and the no-approach rule;
``remember_preference`` stays allowed because it only ever acts after the person's spoken
confirmation, and its question passes the minor output filter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from .registry import SkillContext, SkillRegistry, SkillResult, skill
from .schema import Desc, Range

LOOK_CATEGORIES = ("speaker", "person", "animal", "object", "away")
OK = {"ok": True}


@dataclass
class SkillEnv:
    """What the default skills act through. Every callable runs on the conversation's event
    loop and must not block: it posts a request or touches thread-safe state, never a joint."""

    gestures: Callable[[], Sequence[str]]  # affect labels and named gestures the robot can play
    play: Callable[[str], bool]
    look_names: Callable[[], Sequence[str]]  # names of what is in view, besides the categories
    look_at: Callable[[str], bool]
    affect: Any  # AffectEngine: cfg.labels, state(now), llm_impulse(now, v, a)


def default_skills(env: SkillEnv) -> SkillRegistry:
    """A registry with the five default skills, all ``llm_visible``."""

    @skill(
        llm_visible=True,
        inline=True,
        description=(
            "Play one of the robot's named motions. An emotion name (joy, sadness, ...) shows "
            "that feeling with antennas and head; 'nod' means yes and 'shake' means no. Pick "
            "an emotion to show a feeling and a gesture to answer or agree by motion."
        ),
        choices={"gesture": env.gestures},
    )
    def play(gesture: Annotated[str, Desc("Name of the emotion or gesture")]) -> Any:
        return OK if env.play(gesture) else {"error": "not_now"}

    @skill(
        llm_visible=True,
        inline=True,
        description=(
            "Turn the head to look at someone or something: 'speaker' or 'person' for the "
            "people, 'animal', 'object', 'away' to look aside, or a name of what is in view."
        ),
        choices={"target": lambda: [*LOOK_CATEGORIES, *env.look_names()]},
    )
    def look_at(target: Annotated[str, Desc("Who or what to look at")]) -> Any:
        return OK if env.look_at(target) else {"error": "nothing_to_look_at"}

    @skill(
        llm_visible=True,
        inline=True,
        description=(
            "Show a feeling in the robot's own mood: it nudges the mood a little towards the "
            "emotion, never beyond a small amount per turn."
        ),
        choices={"emotion": lambda: list(env.affect.cfg.labels)},
    )
    def express_emotion(
        ctx: SkillContext,
        emotion: Annotated[str, Desc("The emotion to show")],
        intensity: Annotated[float, Range(0.0, 1.0), Desc("How strongly, 0 to 1")] = 0.5,
    ) -> Any:
        label = env.affect.cfg.labels[emotion]
        state = env.affect.state(ctx.now)
        dv, da = env.affect.llm_impulse(
            ctx.now,
            (label.valence - state.valence) * intensity,
            (label.arousal - state.arousal) * intensity,
        )
        return {"ok": True, "applied": not math.isclose(math.hypot(dv, da), 0.0)}

    @skill(
        llm_visible=True,
        inline=True,
        confirm=True,
        description=(
            "Remember a fact about the person who is talking (a preference such as a favourite "
            "colour) for next time. Only call it when they asked you to remember something. "
            "The robot asks them to confirm out loud; nothing is stored until they say yes."
        ),
    )
    def remember_preference(
        ctx: SkillContext,
        key: Annotated[str, Desc("Short name of the preference, for example favourite colour")],
        value: Annotated[str, Desc("The preference itself")],
    ) -> SkillResult:
        status = ctx.ask_confirmation(key, value)
        return SkillResult({"status": status}, end_turn=status == "asked")

    @skill(
        llm_visible=True,
        inline=True,
        description="Stop talking now and end this turn, for example when asked to be quiet.",
    )
    def stop_talking(ctx: SkillContext) -> SkillResult:
        ctx.stop_speech()
        return SkillResult(OK, end_turn=True)

    return SkillRegistry([play, look_at, express_emotion, remember_preference, stop_talking])
