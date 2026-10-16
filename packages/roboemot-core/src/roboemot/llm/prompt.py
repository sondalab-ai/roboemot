"""System prompt composition (SPEC 8, Conversation state).

Fixed order, the same for every route: persona prompt, register fragment, rules rendered from
structured parameters, the safety block, then (in ``build_messages``) the per-request speaker,
scene and affect parts. The safety block is owned by the library; configuration can only
append to it through ``safety.extra_prompt_rules``.
"""

from __future__ import annotations

from ..config.schema import RegisterParams, RoboEmotConfig

SAFETY_RULES = (
    "You are a robot: never claim to be human.",
    "If you are not sure about something, say so instead of guessing.",
    "Never suggest actions that could put a person, an animal or yourself at physical risk.",
)
MINOR_RULE = (
    "A child may be listening: keep every topic and word suitable for children, and never ask "
    "for personal details."
)
VOCABULARY_RULES = {
    "simple": "Use simple words and short sentences.",
    "normal": "",
    "formal": "Use polite, formal wording.",
}
LANGUAGE_NAMES = {"it": "Italian", "en": "English"}


def system_prompt(
    cfg: RoboEmotConfig,
    register: RegisterParams,
    *,
    language: str | None = None,
    minor_present: bool = False,
) -> str:
    """Parts 1 to 4 of the system prompt. ``register`` is the active register as the persona
    shapes it (``roboemot.persona.apply_to_register``); ``language`` defaults to the
    conversation language."""
    lang = language or cfg.conversation.language
    rules = [
        VOCABULARY_RULES[register.vocabulary],
        f"Reply in at most {cfg.conversation.max_reply_sentences} sentences.",
        f"Always reply in {LANGUAGE_NAMES.get(lang, lang)}.",
    ]
    if cfg.voice.mood_source == "llm_tagged":
        labels = ", ".join(cfg.affect.labels)
        rules.append(
            f"Start each sentence with a mood tag such as [mood=joy], choosing from: {labels}. "
            "Let interjections and punctuation match the mood."
        )
    safety = list(SAFETY_RULES)
    if minor_present or register.minor_content_policy:
        safety.append(MINOR_RULE)
    safety += cfg.safety.extra_prompt_rules
    parts = [
        cfg.persona.prompt,
        register.persona_fragment,
        " ".join(r for r in rules if r),
        "Rules you always follow: " + " ".join(safety),
    ]
    return "\n".join(p for p in parts if p)
