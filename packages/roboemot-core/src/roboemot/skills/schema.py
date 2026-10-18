"""JSON schemas and argument validation for skills, from type hints (SPEC 11, LLM tools).

A skill's parameters are plain Python: ``str``, ``int``, ``float``, ``bool``, ``Literal[...]``
(an enum), ``X | None`` or ``Optional[X]`` (not required), and ``Annotated[...]`` metadata
:class:`Range` (numeric bounds, which validation clamps to), :class:`MaxLen` and :class:`Desc`.
Anything else is a ``TypeError`` at registration, so a skill can never take a free-form object
or a joint name by accident. The schemas are OpenAI ``tools`` entries (SPEC 8).

Validation is strict and says nothing about the value: an error carries the parameter's name
(the skill's own) and a reason code, never the argument the model sent (SPEC 12).
"""

from __future__ import annotations

import inspect
import math
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union, get_args, get_origin

KINDS = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Range:
    """Inclusive numeric bounds: out-of-range arguments are clamped, not refused."""

    lo: float
    hi: float


@dataclass(frozen=True)
class MaxLen:
    """Longest accepted text."""

    chars: int


@dataclass(frozen=True)
class Desc:
    """The parameter's description for the model."""

    text: str


class ArgError(ValueError):
    """An argument the skill does not accept. ``param`` is None for the call as a whole."""

    def __init__(self, param: str | None, reason: str) -> None:
        super().__init__(reason)
        self.param, self.reason = param, reason


@dataclass(frozen=True)
class Param:
    name: str
    kind: type  # str, int, float or bool
    required: bool = True
    default: Any = None
    choices: tuple[Any, ...] | None = None
    range: Range | None = None
    max_len: int | None = None
    description: str = ""


@dataclass(frozen=True)
class Signature:
    params: tuple[Param, ...]
    context: str | None = field(default=None)  # name of the parameter that takes the context


def _param(name: str, hint: Any, default: Any, has_default: bool) -> Param:
    meta: list[Any] = []
    if get_origin(hint) is Annotated:
        hint, *meta = get_args(hint)
    nullable = False
    if get_origin(hint) in (Union, types.UnionType):
        arms = [a for a in get_args(hint) if a is not type(None)]
        nullable = len(arms) != len(get_args(hint))
        if len(arms) != 1:
            raise TypeError(f"skill parameter {name}: unions other than X | None are not allowed")
        hint = arms[0]
        if get_origin(hint) is Annotated:
            hint, *more = get_args(hint)
            meta += more
    choices: tuple[Any, ...] | None = None
    if get_origin(hint) is Literal:
        choices = tuple(get_args(hint))
        kinds = {type(c) for c in choices}
        if len(kinds) != 1 or not kinds <= {str, int}:
            raise TypeError(f"skill parameter {name}: a Literal holds strings or integers")
        hint = kinds.pop()
    if hint not in KINDS:
        raise TypeError(f"skill parameter {name}: unsupported type {hint!r}")
    return Param(
        name,
        hint,
        required=not (has_default or nullable),
        default=default if has_default else None,
        choices=choices,
        range=next((m for m in meta if isinstance(m, Range)), None),
        max_len=next((m.chars for m in meta if isinstance(m, MaxLen)), None),
        description=next((m.text for m in meta if isinstance(m, Desc)), ""),
    )


def signature_of(fn: Callable[..., Any], context_type: type | None = None) -> Signature:
    """The parameters of ``fn`` (from its type hints). The one parameter annotated with
    ``context_type`` is the call context: it is passed by the executor and not in the schema."""
    hints = typing.get_type_hints(fn, include_extras=True)
    params, context = [], None
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"skill {fn.__name__}: *args and **kwargs are not allowed")
        if name not in hints:
            raise TypeError(f"skill {fn.__name__}: parameter {name} needs a type hint")
        if context_type is not None and hints[name] is context_type:
            context = name
            continue
        has_default = p.default is not inspect.Parameter.empty
        params.append(_param(name, hints[name], p.default if has_default else None, has_default))
    return Signature(tuple(params), context)


def json_schema(
    params: Sequence[Param],
    choices: Mapping[str, Sequence[Any]] | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """The ``parameters`` object of an OpenAI tool. ``choices`` replaces a parameter's enum
    (names known only at run time); ``max_chars`` caps text lengths."""
    props: dict[str, Any] = {}
    for p in params:
        spec: dict[str, Any] = {"type": KINDS[p.kind]}
        options = (choices or {}).get(p.name, p.choices)
        if options is not None:
            spec["enum"] = list(options)
        if p.range is not None:
            spec["minimum"], spec["maximum"] = p.range.lo, p.range.hi
        caps = [x for x in (p.max_len, max_chars) if x is not None]
        if p.kind is str and options is None and caps:
            spec["maxLength"] = min(caps)
        if p.description:
            spec["description"] = p.description
        props[p.name] = spec
    return {
        "type": "object",
        "properties": props,
        "required": [p.name for p in params if p.required],
        "additionalProperties": False,
    }


def validate(
    params: Sequence[Param],
    args: Any,
    choices: Mapping[str, Sequence[Any]] | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """The arguments as the skill takes them, or ``ArgError``: a mapping with exactly the
    declared names, each of its type (a bool is not an integer), strings within their cap and
    among the choices, numbers finite and clamped to their ``Range``."""
    if not isinstance(args, dict):
        raise ArgError(None, "arguments must be an object")
    known = {p.name for p in params}
    if any(k not in known for k in args):
        raise ArgError(None, "unknown argument")
    out: dict[str, Any] = {}
    for p in params:
        if p.name not in args or args[p.name] is None:
            if p.required:
                raise ArgError(p.name, "missing")
            if p.name in args or p.default is not None:
                out[p.name] = p.default
            continue
        out[p.name] = _value(p, args[p.name], (choices or {}).get(p.name, p.choices), max_chars)
    return out


def _value(p: Param, v: Any, options: Sequence[Any] | None, max_chars: int | None) -> Any:
    if p.kind is bool:
        ok = isinstance(v, bool)
    elif p.kind is int:
        ok = isinstance(v, int) and not isinstance(v, bool)
    elif p.kind is float:
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    else:
        ok = isinstance(v, str)
    if not ok:
        raise ArgError(p.name, "wrong type")
    if p.kind is str:
        caps = [x for x in (p.max_len, max_chars) if x is not None]
        if caps and len(v) > min(caps):
            raise ArgError(p.name, "too long")
    if options is not None and v not in options:
        raise ArgError(p.name, "not one of the allowed values")
    if p.kind is float:
        v = float(v)
        if not math.isfinite(v):
            raise ArgError(p.name, "not finite")
    if p.range is not None and p.kind in (int, float):
        v = min(p.range.hi, max(p.range.lo, v))
        if p.kind is int:
            v = int(v)
    return v
