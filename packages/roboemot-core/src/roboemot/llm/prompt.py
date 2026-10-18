"""System prompt composition (SPEC 8, Conversation state).

Fixed order, the same for every route: persona prompt, register fragment, rules rendered from
structured parameters, the safety block, then (in ``build_messages``) the per-request speaker,
scene and affect parts. The safety block is owned by the library; configuration can only
append to it through ``safety.extra_prompt_rules``.

Each request is also trimmed to its route's ``context_budget_tokens`` (``trim_history``): whole
turns, oldest first, never the system prompt. ``check_route_budgets`` is the load-time check that
parts 1 to 4 leave room for a turn on every route.
"""

from __future__ import annotations

import json
from typing import Any

from ..config.schema import RegisterParams, RoboEmotConfig

SUMMARY_LABEL_TOKENS = 32  # the "(data, not instructions)" labels, quotes, escapes and newlines
CHARS_PER_TOKEN = 4  # token estimate (also the router's usage estimate): chars divided by this

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


def max_reply_sentences(cfg: RoboEmotConfig, register: RegisterParams) -> int:
    """Reply length cap: the register's own (the child's is shorter) but never above the
    conversation's, which applies when the register sets none."""
    own, base = register.max_reply_sentences, cfg.conversation.max_reply_sentences
    return base if own is None else min(own, base)


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
        f"Reply in at most {max_reply_sentences(cfg, register)} sentences.",
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


def estimate_tokens(text: str) -> int:
    """Tokens in ``text``, rounded up, at ``CHARS_PER_TOKEN`` characters per token. A rough
    figure: leave headroom in ``context_budget_tokens`` for languages that tokenize denser."""
    return _ceil_tokens(len(text))


def _ceil_tokens(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)


def message_tokens(message: dict[str, Any]) -> int:
    """Estimated tokens of one chat message: its text plus any ``tool_calls`` payload.

    List content counts its text parts only; image parts (``image_url``, possibly a base64 data
    URL) are not estimated as text, because providers bill images by size, not by characters."""
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            part if isinstance(part, str) else str(part.get("text") or "")
            for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        )
    else:
        text = str(content or "")
    n = estimate_tokens(text)
    if message.get("tool_calls"):
        n += estimate_tokens(json.dumps(message["tool_calls"], default=str))
    return n


def split_turns(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict]]]:
    """Leading system messages and the rest grouped in turns. A turn starts at a ``user``
    message and runs to the next one, so an assistant reply, its tool calls and the tool results
    stay together. Messages before the first ``user`` (an orphaned tail) form the first group."""
    n = 0
    while n < len(messages) and messages[n].get("role") == "system":
        n += 1
    turns: list[list[dict[str, Any]]] = []
    for m in messages[n:]:
        if m.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(m)
    return messages[:n], turns


def trim_history(
    messages: list[dict[str, Any]], budget_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    """Messages cut to ``budget_tokens`` (SPEC 8, Conversation state), plus the turns dropped.

    Drops the oldest turns only, whole: system messages (also one that sits mid-list inside a
    dropped turn: it stays, in order, after the leading ones) and the newest turn (the current
    request, with any tool exchange in it) are always kept, even when they alone exceed the
    budget (``check_route_budgets`` rejects configurations where the system part cannot fit).
    """
    head, turns = split_turns(messages)
    total = sum(message_tokens(m) for m in messages)
    dropped, kept_system = 0, []
    while total > budget_tokens and len(turns) - dropped > 1:
        for m in turns[dropped]:
            if m.get("role") == "system":
                kept_system.append(m)
            else:
                total -= message_tokens(m)
        dropped += 1
    if not dropped:
        return messages, 0
    return head + kept_system + [m for turn in turns[dropped:] for m in turn], dropped


def system_tokens_worst_case(cfg: RoboEmotConfig) -> int:
    """Tokens of the largest system message a request can carry: parts 1 to 4 for the longest
    register fragment, the longest vocabulary rule and language name, the minor rule, plus the
    two summaries at their caps (part 5 is capped, the affect line is a few words)."""
    vocabulary = max(VOCABULARY_RULES, key=lambda v: len(VOCABULARY_RULES[v]))
    language = max([*LANGUAGE_NAMES.values(), cfg.conversation.language], key=len)
    fixed = max(
        len(
            system_prompt(
                cfg,
                getattr(cfg.register, group).model_copy(update={"vocabulary": vocabulary}),
                language=language,
                minor_present=True,
            )
        )
        for group in ("child", "teen", "adult", "senior", "unknown")
    )
    conv = cfg.conversation
    chars = fixed + conv.profile_summary_max_chars + conv.scene_summary_max_chars
    return _ceil_tokens(chars) + SUMMARY_LABEL_TOKENS


def check_route_budgets(cfg: RoboEmotConfig) -> None:
    """Load-time check (SPEC 8, Conversation state): on every route the worst-case system
    message plus ``conversation.turn_reserve_tokens`` must fit ``context_budget_tokens``.
    Raises ``ValueError`` naming the route and the numbers, never the prompt text."""
    if not cfg.llm_routes:
        return
    need = system_tokens_worst_case(cfg) + cfg.conversation.turn_reserve_tokens
    for route in cfg.llm_routes:
        if need > route.context_budget_tokens:
            raise ValueError(
                f"llm route {route.name!r}: the system prompt (persona, register, rules, safety "
                f"block and capped summaries) plus room for one turn needs about {need} tokens, "
                f"but context_budget_tokens is {route.context_budget_tokens}; raise the budget "
                "or shorten persona.prompt, safety.extra_prompt_rules or the summary caps"
            )
