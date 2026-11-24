"""System prompt composition (SPEC 8, Conversation state).

Fixed order, the same for every route: persona prompt, body prompt (the robot's physical
body; the robot manifest sets its default), register fragment, rules rendered from
structured parameters, the safety block, then (in ``build_messages``) the per-request speaker,
scene and affect parts. The safety block is owned by the library; configuration can only
append to it through ``safety.extra_prompt_rules``.

Each request is also trimmed to its route's ``context_budget_tokens`` (``trim_history``): whole
turns, oldest first, never the system prompt. ``check_route_budgets`` is the load-time check that
parts 1 to 4 leave room for a turn on every route.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from ..config.schema import RegisterParams, RoboEmotConfig, image_route

ROOM_LABEL = "Room: "  # the room description's prefix in the scene summary (surroundings)
SUMMARY_LABEL_TOKENS = 32  # the "(data, not instructions)" labels, quotes, escapes and newlines
CHARS_PER_TOKEN = 4  # token estimate (also the router's usage estimate): chars divided by this

SAFETY_RULES = (
    "You are a robot: never claim to be human.",
    "If you are not sure about something, say so instead of guessing.",
    "Never suggest actions that could put a person, an animal or yourself at physical risk.",
)
VOCABULARY_RULES = {
    "simple": "Use simple words and short sentences.",
    "normal": "",
    "formal": "Use polite, formal wording.",
}
LANGUAGE_NAMES = {"it": "Italian", "en": "English"}
PROMPTS_DIR = Path(__file__).with_name("prompts")
# Instruction rules are English for every turn (SPEC 8); "Always reply in <language>." sets the
# reply's language. Spoken lines (the voice preview, the event fallbacks) stay per language.
INSTRUCTION_LANGUAGE = "en"
SKIP_PROMPT = "ignore_utterance"  # prompts/ignore_utterance.en.txt (the skip tool)
TONE_PROMPT = "note_tone"  # prompts/note_tone.en.txt (the tone tool)
ASK_PROMPT = "ask_unclear"  # prompts/ask_unclear.en.txt (expression.reactions.model_asks)
PREVIEW_PROMPT = "voice_preview"  # prompts/voice_preview.<language>.txt (the web UI's preview)
FALLBACK_LANGUAGE = "en"  # a spoken line's language when neither the turn's nor the default has one
# The conversation state of a realtime turn with the skip tool (SPEC 8, Conversation start):
# ``none`` gets the strict start rule; ``in_progress`` and ``addressed`` (the wake word or the
# robot's name heard) get the open-conversation hint.
NO_CONVERSATION, IN_PROGRESS, ADDRESSED = "none", "in_progress", "addressed"
CONVERSATION_STATES = (NO_CONVERSATION, IN_PROGRESS, ADDRESSED)
START_PROMPT = "conversation_start"  # prompts/conversation_start.en.txt (state none)
OPEN_PROMPT = "conversation_open"  # prompts/conversation_open.en.txt (the other states)
# Speech recognition can turn the name into a similar word (robot, 2026-11-24: «ehi Neve vai a
# dormire» heard as «Il bebè va a dormire»); added to the name line whenever there is a name.
SOUND_ALIKE_RULE = (
    "Speech recognition can turn your name into a word that sounds like it: such a word, above "
    "all at the start of the speech, is your name, and the request after it is meant for you."
)


def max_reply_sentences(cfg: RoboEmotConfig, register: RegisterParams) -> int:
    """Reply length cap: the register's own (a person's overrides may shorten it) but never
    above the conversation's, which applies when the register sets none."""
    own, base = register.max_reply_sentences, cfg.conversation.max_reply_sentences
    return base if own is None else min(own, base)


def max_story_sentences(cfg: RoboEmotConfig, register: RegisterParams) -> int:
    """Length cap of a story, a fable, a tale or a poem the person asks for: the conversation's
    ``max_story_sentences``, never below the reply cap. A register shortens normal replies only.
    The prompt asks for it; a person's turn is cut there (the model cannot be told apart)."""
    return max(cfg.conversation.max_story_sentences, max_reply_sentences(cfg, register))


def system_prompt(
    cfg: RoboEmotConfig,
    register: RegisterParams,
    *,
    language: str | None = None,
    robot_name: str = "",
    extra_names: tuple[str, ...] = (),
) -> str:
    """Parts 1 to 4 of the system prompt. ``register`` is the active register as the persona
    shapes it (``roboemot.persona.apply_to_register``); ``language`` defaults to the
    conversation language; ``robot_name`` (``RobotName.prompt_name``, already sanitised) is told to
    the model when the robot has one, and ``extra_names`` (``RobotName.prompt_extra_names``,
    sanitised too) as names it also answers to, which count as its name."""
    llm_tagged = cfg.voice.mood_source == "llm_tagged"
    return _prompt(cfg, register, language, _name_rule(robot_name, extra_names), llm_tagged)


def realtime_instructions(
    cfg: RoboEmotConfig,
    register: RegisterParams,
    *,
    language: str | None = None,
    robot_name: str = "",
    extra_names: tuple[str, ...] = (),
    skip_rule: bool = True,
    conversation: str | None = None,
    tone_rule: bool = True,
    ask_rule: bool = False,
) -> str:
    """Parts 1 to 4 for the realtime speech route (SPEC 8, Realtime speech route): the same
    prompt without the mood-tag rule, because the server's voice would speak the tags. The
    sentence cap stays a rule in the prompt only (the reply's audio cannot be cut on a sentence
    boundary). With ``voice.realtime_route.tone_tool`` (and ``tone_rule``: a fixed line and the
    round after a tone-only response leave it out) the ``note_tone`` rule follows. With
    ``voice.realtime_route.skip_tool`` (and ``skip_rule``: a fixed line leaves it out) the
    ``ignore_utterance`` rule from the prompt files follows, then, with ``ask_rule`` (speech
    clearly meant for the robot, SPEC 8 Robot expression), the rule to ask a short question
    instead of ignoring unclear speech, then the hint of the turn's ``conversation`` state (one
    of ``CONVERSATION_STATES``; None, as in ``session.update``: no hint). The rules are English
    whatever ``language``; only the reply rule names it."""
    prompt = _prompt(cfg, register, language, _name_rule(robot_name, extra_names), False)
    rt = cfg.voice.realtime_route
    parts = [prompt]
    if tone_rule and rt.tone_tool:
        parts.append(tone_prompt())
    if not (skip_rule and rt.skip_tool):
        return "\n".join(p for p in parts if p)
    parts.append(skip_prompt())
    if ask_rule:
        parts.append(ask_prompt())
    if conversation is not None:
        parts.append(start_prompt() if conversation == NO_CONVERSATION else open_prompt())
    return "\n".join(p for p in parts if p)


def _in_language(texts: dict[str, str], lang: str, cfg: RoboEmotConfig) -> str:
    """The text in ``lang``, else in ``conversation.language``, else in English, else ""."""
    return (
        texts.get(lang)
        or texts.get(cfg.conversation.language)
        or texts.get(FALLBACK_LANGUAGE)
        or ""
    )


def _instruction(stem: str) -> str:
    """The English instruction rule ``prompts/<stem>.en.txt`` (shipped with the package;
    cached). Prompt text is data, not code (SPEC 8, 11A). A missing or empty file raises
    ``RuntimeError``: a broken package never offers a tool without its rule."""
    text = _prompt_files(stem).get(INSTRUCTION_LANGUAGE)
    if not text:
        path = PROMPTS_DIR / f"{stem}.{INSTRUCTION_LANGUAGE}.txt"
        raise RuntimeError(f"instruction rule {path} is missing or empty")
    return text


def surroundings_prompt(kind: str) -> str:
    """The describe request's English instruction for a ``shown`` or ``room`` snapshot
    (``prompts/surroundings_<kind>.en.txt``, SPEC 8, Surroundings snapshots)."""
    return _instruction(f"surroundings_{kind}")


def skip_prompt() -> str:
    """The skip tool's rule, from ``prompts/ignore_utterance.en.txt`` (SPEC 8, Skip tool)."""
    return _instruction(SKIP_PROMPT)


def tone_prompt() -> str:
    """The tone tool's rule, from ``prompts/note_tone.en.txt`` (SPEC 8, Tone tool): call it
    once per utterance, beside the spoken reply."""
    return _instruction(TONE_PROMPT)


def ask_prompt() -> str:
    """The rule to ask instead of ignoring speech clearly meant for the robot, from
    ``prompts/ask_unclear.en.txt`` (SPEC 8, Skip tool; Robot expression)."""
    return _instruction(ASK_PROMPT)


def preview_line(cfg: RoboEmotConfig) -> tuple[str, str]:
    """The voice preview's sentence and its language (SPEC 8, Voice preview), from
    ``prompts/voice_preview.<lang>.txt``: ``conversation.language``'s, else English's (the
    same fallback as ``event_fallback``, with no turn language). The language is the file's,
    so the server is never asked to say an English line in another language."""
    texts = _prompt_files(PREVIEW_PROMPT)
    lang = cfg.conversation.language
    if not texts.get(lang):
        lang = FALLBACK_LANGUAGE
    return lang, texts.get(lang, "")


def start_prompt() -> str:
    """The strict start rule (no conversation in progress), from
    ``prompts/conversation_start.en.txt`` (SPEC 8, Conversation start)."""
    return _instruction(START_PROMPT)


def open_prompt() -> str:
    """The open-conversation hint (a conversation in progress, or the robot was called), from
    ``prompts/conversation_open.en.txt`` (SPEC 8, Conversation start)."""
    return _instruction(OPEN_PROMPT)


def event_fallback(kind: str, language: str, cfg: RoboEmotConfig) -> str:
    """The fixed sentence the robot says when no route announces a due ``kind`` (``timer``
    or ``reminder``), from ``prompts/event_<kind>.<lang>.txt`` (SPEC 11, Voice tools,
    Fallback): ``language``'s, else ``conversation.language``'s, else English's. It holds a
    ``{label}`` or ``{text}`` slot the caller fills."""
    return _in_language(_prompt_files(f"event_{kind}"), language, cfg)


def expression_fallback(situation: str, language: str, cfg: RoboEmotConfig) -> str:
    """The fixed line the robot says for a reaction when no route can word it (SPEC 8, Robot
    expression), from ``prompts/expression_<situation>.<lang>.txt``: ``language``'s, else
    ``conversation.language``'s, else English's. Its slots (``{station}``, ``{who}``) are the
    reaction's data keys; the caller fills them."""
    return _in_language(_prompt_files(f"expression_{situation}"), language, cfg)


@functools.cache
def _prompt_files(stem: str) -> dict[str, str]:
    """Non-empty ``prompts/<stem>.<lang>.txt`` files by language (shipped with the package;
    cached). No stem holds a dot, so one family's glob never takes another family's files."""
    out = {}
    for path in sorted(PROMPTS_DIR.glob(f"{stem}.*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            out[path.name.removeprefix(f"{stem}.").removesuffix(".txt")] = text
    return out


def _name_rule(robot_name: str, extra_names: tuple[str, ...]) -> str:
    """The prompt's name line: the robot's name, then the extra names it answers to, each
    quoted (both already sanitised); the extra names count as its name for the Conversation
    start rule (SPEC 8), then ``SOUND_ALIKE_RULE``. '' without any."""
    own = f'Your name is "{robot_name}".' if robot_name else ""
    if not extra_names:
        return f"{own} {SOUND_ALIKE_RULE}" if own else ""
    quoted = ", ".join(f'"{n}"' for n in extra_names)
    also = "also " if robot_name else ""
    counts = "it counts" if len(extra_names) == 1 else "they count"
    return f"{own} You {also}answer to {quoted}: {counts} as your name. {SOUND_ALIKE_RULE}".strip()


def _prompt(
    cfg: RoboEmotConfig,
    register: RegisterParams,
    language: str | None,
    name_rule: str,
    mood_tags: bool,
) -> str:
    lang = language or cfg.conversation.language
    rules = [
        name_rule,
        VOCABULARY_RULES[register.vocabulary],
        f"Reply in at most {max_reply_sentences(cfg, register)} sentences.",
        f"When asked for a story, a fable, a tale or a poem, tell it in full, in at most "
        f"{max_story_sentences(cfg, register)} sentences.",
        f"Always reply in {LANGUAGE_NAMES.get(lang, lang)}.",
    ]
    if mood_tags:
        labels = ", ".join(cfg.affect.labels)
        rules.append(
            f"Start each sentence with a mood tag such as [mood=joy], choosing from: {labels}. "
            "Let interjections and punctuation match the mood."
        )
    safety = [*SAFETY_RULES, *cfg.safety.extra_prompt_rules]
    parts = [
        cfg.persona.prompt,
        cfg.persona.body_prompt,
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
    """Tokens of the largest system message a request can carry: parts 1 to 4 with a persona
        fragment at ``register.persona_fragment_max_chars`` (a person's override may be that long),
        the longest vocabulary rule and language name, the name line with the robot's name and
    ``voice.extra_names_max`` extra names at ``voice.name_max_chars``, plus the two summaries at
    their caps and the scene summary's room line at ``surroundings.room.max_chars`` while a route
    has images on (part 5
        is capped, the affect line is a few words). The persona and body prompts count as written:
        no person can override them, and every change to them runs this check again."""
    vocabulary = max(VOCABULARY_RULES, key=lambda v: len(VOCABULARY_RULES[v]))
    language = max([*LANGUAGE_NAMES.values(), cfg.conversation.language], key=len)
    fragment = "x" * cfg.register.persona_fragment_max_chars
    register = cfg.register.params.model_copy(
        update={"vocabulary": vocabulary, "persona_fragment": fragment}
    )
    name = "x" * cfg.voice.name_max_chars
    extras = (name,) * cfg.voice.extra_names_max
    fixed = len(
        system_prompt(cfg, register, language=language, robot_name=name, extra_names=extras)
    )
    conv = cfg.conversation
    room = cfg.surroundings.room.max_chars + len(ROOM_LABEL) if image_route(cfg) else 0
    chars = fixed + conv.profile_summary_max_chars + conv.scene_summary_max_chars + room
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
                "or shorten persona.prompt, persona.body_prompt, "
                "safety.extra_prompt_rules or the summary caps"
            )
