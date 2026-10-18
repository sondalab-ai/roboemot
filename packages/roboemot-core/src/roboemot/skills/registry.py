"""Skills: the decorator, the registry and the executor (SPEC 11, LLM tools; 8, OpenAI routes).

``@skill`` marks a function as a skill and derives its parameters from the type hints
(:mod:`roboemot.skills.schema`). A skill the model may call has ``llm_visible=True``; the
registry offers it as an OpenAI ``tool`` only when it is also in ``conversation.llm_skills`` and,
while a minor is present, ``allowed_with_minor``. Skills never take joints: what they do goes
through the same paths as everything else (the expression renderer, the attention selector, the
affect engine's budgeted impulse, the registry's confirmed preferences).

The executor runs a call the model streamed back: unknown or not-offered names, bad arguments,
timeouts and exceptions all become a tool result the model can read (``{"error": code}``), and
none of them carries an argument value or exception text (SPEC 12). Names and counts may be
logged; arguments and results may not.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schema import ArgError, Param, Signature, json_schema, signature_of, validate

log = logging.getLogger(__name__)
Choices = Mapping[str, Callable[[], Sequence[Any]]]


@dataclass
class SkillContext:
    """What a skill may know about the call: who is talking, in which language, whether a
    minor is present, and the few actions a skill needs from the conversation. The actions are
    injected by the worker (so skills stay testable and robot-agnostic)."""

    speaker: str | None
    language: str
    minor: bool
    now: float
    stop_speech: Callable[[], None] = lambda: None
    ask_confirmation: Callable[[str, str], str] = lambda key, value: "unavailable"


@dataclass(frozen=True)
class SkillResult:
    """A skill's return value: ``content`` goes back to the model; ``end_turn`` stops the
    turn there (no further request), for a skill that already said what there is to say."""

    content: Any = "ok"
    end_turn: bool = False


@dataclass(frozen=True)
class Skill:
    name: str
    fn: Callable[..., Any]
    description: str
    llm_visible: bool
    confirm: bool
    allowed_with_minor: bool
    signature: Signature
    choices: Choices = field(default_factory=dict)
    inline: bool = False

    @property
    def params(self) -> tuple[Param, ...]:
        return self.signature.params

    def resolved_choices(self, enum_max: int | None = None) -> dict[str, list[Any]]:
        """The run-time choices of each parameter that has them, cut to ``enum_max`` items."""
        out = {}
        for name, source in self.choices.items():
            items = list(source())
            out[name] = items if enum_max is None else items[:enum_max]
        return out


def skill(
    name: str | None = None,
    *,
    llm_visible: bool = False,
    description: str = "",
    confirm: bool = False,
    allowed_with_minor: bool = True,
    choices: Choices | None = None,
    inline: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark ``fn`` as a skill (``fn.skill`` is its :class:`Skill`). ``name`` defaults to the
    function's; ``description`` to the first line of its docstring. ``confirm`` says the skill
    asks the person before it acts (informational: the skill does the asking). ``choices`` maps a
    parameter to a function giving its allowed values at run time (labels, names in view).
    ``inline`` runs a plain function on the event loop instead of a worker thread: for skills
    that only post a request and return, so nothing of theirs can still be running (and
    touching a finished turn) after a timeout or a cancel. They must not block."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        text = description or (inspect.getdoc(fn) or "").split("\n")[0]
        sig = signature_of(fn, SkillContext)
        for param in choices or {}:
            if param not in {p.name for p in sig.params}:
                raise TypeError(f"skill {fn.__name__}: choices for unknown parameter {param}")
        fn.skill = Skill(  # type: ignore[attr-defined]
            name or fn.__name__,
            fn,
            text,
            llm_visible,
            confirm,
            allowed_with_minor,
            sig,
            dict(choices or {}),
            inline,
        )
        return fn

    return wrap


class SkillRegistry:
    """The skills of an app, by name."""

    def __init__(self, skills: Collection[Callable[..., Any] | Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for s in skills:
            self.add(s)

    def add(self, item: Callable[..., Any] | Skill) -> Skill:
        s = item if isinstance(item, Skill) else getattr(item, "skill", None)
        if s is None:
            raise TypeError("not a skill: decorate it with @skill")
        if s.name in self._skills:
            raise ValueError(f"skill {s.name} is already registered")
        self._skills[s.name] = s
        return s

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return list(self._skills)

    def offered(self, allow: Collection[str], *, minor: bool) -> list[Skill]:
        """The skills the model may call now: visible, listed in ``allow``, and allowed with a
        minor present when there is one. Call it for every request."""
        return [
            s
            for s in self._skills.values()
            if s.llm_visible
            and s.name in allow
            and (s.allowed_with_minor or not minor)
            and all(s.resolved_choices().values())  # a parameter with nothing to choose from
        ]

    @staticmethod
    def snapshot_choices(
        offered: Sequence[Skill], enum_max: int | None = None
    ) -> dict[str, dict[str, list[Any]]]:
        """The run-time choices of ``offered`` skills, resolved once: what the model is shown
        is what the executor checks, even if the world changes while it answers."""
        return {s.name: s.resolved_choices(enum_max) for s in offered}

    @staticmethod
    def tools(
        offered: Sequence[Skill],
        *,
        enum_max: int | None = None,
        max_chars: int | None = None,
        choices: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        """OpenAI ``tools`` entries for ``offered``; choices from ``choices`` (a snapshot) or
        resolved now."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": json_schema(
                        s.params,
                        choices[s.name] if choices is not None else s.resolved_choices(enum_max),
                        max_chars,
                    ),
                },
            }
            for s in offered
        ]


@dataclass(frozen=True)
class ToolOutcome:
    """What the executor hands back: the tool message content, whether the skill ran, and
    whether it ended the turn. ``skill`` is the name (safe to log), ``code`` the error code."""

    content: str
    ok: bool
    end_turn: bool = False
    skill: str = ""
    code: str = ""


def _error(code: str, skill_name: str, detail: str | None = None) -> ToolOutcome:
    body: dict[str, Any] = {"error": code}
    if detail:
        body["detail"] = detail
    return ToolOutcome(json.dumps(body), False, False, skill_name, code)


async def execute(
    registry: SkillRegistry,
    offered: Collection[str],
    name: str,
    args: Any,
    ctx: SkillContext,
    *,
    timeout_s: float,
    max_chars: int,
    enum_max: int,
    choices: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> ToolOutcome:
    """Run the call ``name(args)``. Only a skill in ``offered`` runs (a model may name one it
    was not given). Validation, timeout and exceptions become error results. ``choices`` is the
    snapshot the model was shown (else the choices are resolved now)."""
    s = registry.get(name)
    if s is None or name not in offered:
        return _error("unknown_skill", "")
    try:
        options = choices[name] if choices is not None else s.resolved_choices(enum_max)
        kwargs = validate(s.params, args, options, max_chars)
    except ArgError as exc:
        return _error("invalid_arguments", name, f"{exc.param or 'call'}: {exc.reason}")
    if s.signature.context is not None:
        kwargs[s.signature.context] = ctx
    try:
        if inspect.iscoroutinefunction(s.fn):
            value = await asyncio.wait_for(s.fn(**kwargs), timeout_s)
        elif s.inline:
            value = s.fn(**kwargs)
        else:
            value = await asyncio.wait_for(asyncio.to_thread(s.fn, **kwargs), timeout_s)
    except TimeoutError:
        log.warning("skill %s: timed out", name)
        return _error("timeout", name)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # the text may hold what the user said: type only
        log.warning("skill %s: %s", name, type(exc).__name__)
        return _error("failed", name)
    result = value if isinstance(value, SkillResult) else SkillResult(value)
    content = result.content if isinstance(result.content, str) else json.dumps(result.content)
    return ToolOutcome(content, True, result.end_turn, name)
