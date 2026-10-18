"""The field catalogue: one JSON descriptor per config field, for the web UI (SPEC 11A, 12).

``build_catalogue`` walks the ``RoboEmotConfig`` schema together with the effective values and
returns ``FieldDescriptor`` dicts. This shape is a frozen contract (the robot plugin and the
settings page are built against it); ``DESCRIPTOR_KEYS`` and ``KINDS`` pin it in a test.

Keys of every descriptor (all always present, ``None`` when not applicable):

- ``path``: dotted config path (a dict key or a keyed-list entry name/event is one segment).
- ``kind``: one of ``KINDS``. Scalars ``number`` (float), ``int``, ``bool``, ``text``, ``enum``
  (``choices``); ``tuple`` (fixed length ``length``, items ``item_kind`` number/int/text);
  collections ``string_list``, ``enum_list`` (``choices``), ``tuple_list``, ``string_dict``,
  ``number_dict``; ``json`` for any other shape (an absent optional section too).
  ``dict_of_model``, ``keyed_list`` and ``string_list_dict`` are containers: ``value`` is the list
  of entry keys and each entry follows as descriptors under ``<path>.<key>`` (its fields, or for
  ``string_list_dict`` a ``string_list``). ``keyed_list`` names
  its identity field in ``key``.
- ``nullable``: the field accepts null (any kind but the containers).
- ``value`` / ``default``: effective value, and the value without the runtime layer (what a
  reset would give); plain JSON.
- ``source``: layer that decides the value (``ConfigStack.source_of``): ``runtime``, ``derived``,
  ``app``, ``preset``, ``manifest``, ``defaults`` or ``schema``.
- ``ge`` / ``le`` (numeric limits), ``unit``, ``restart``, ``admin_only``, ``opt_in``,
  ``conservative``, ``description``: from the schema (``field_meta`` for the flags).
- ``secret``: the value is a secret or secret-shaped. ``value`` and ``default`` are then
  ``"***"`` (or null when absent) and the field is never writable from the web.
- ``writable``: the web UI may write it today.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypedDict, get_args, get_origin

from pydantic import BaseModel

from roboemot.diagnostics import REDACTED, is_secret_key, secret_shaped

from .schema import RoboEmotConfig
from .stack import KEYED_LISTS, _strip_optional, field_meta, get_path

KINDS = (
    "number",
    "int",
    "bool",
    "text",
    "enum",
    "tuple",
    "string_list",
    "enum_list",
    "tuple_list",
    "string_dict",
    "number_dict",
    "string_list_dict",
    "json",
    "dict_of_model",
    "keyed_list",
)
CONTAINERS = ("dict_of_model", "keyed_list", "string_list_dict")
TEXT_KINDS = ("text", "enum", "string_list", "string_dict", "json")
DEFAULT_LIST_KEY = "name"


class FieldDescriptor(TypedDict):
    """One catalogue entry; see the module docstring for the meaning of each key."""

    path: str
    kind: str
    nullable: bool
    value: Any
    default: Any
    source: str
    ge: float | None
    le: float | None
    unit: str | None
    restart: bool
    admin_only: bool
    opt_in: bool
    conservative: str | None
    description: str
    choices: list[Any] | None
    item_kind: str | None
    length: int | None
    key: str | None
    secret: bool
    writable: bool


DESCRIPTOR_KEYS = frozenset(FieldDescriptor.__annotations__)


def _scalar_kind(annotation: Any) -> str:
    if annotation is bool:
        return "bool"
    if annotation is float:
        return "number"
    if annotation is int:
        return "int"
    if annotation is str:
        return "text"
    return "json"


def classify(annotation: Any) -> tuple[str, dict[str, Any]]:
    """Kind of a non-model annotation and its extras (``choices``, ``item_kind``, ``length``).
    Anything outside the known shapes is ``json``."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return "enum", {"choices": list(args)}
    if origin is tuple:
        kinds = {_scalar_kind(a) for a in args}
        item = kinds.pop() if len(kinds) == 1 else "json"
        return (
            ("tuple", {"item_kind": item, "length": len(args)})
            if item != "json"
            else (
                "json",
                {},
            )
        )
    if origin is list and args:
        inner, inner_args = args[0], get_args(args[0])
        if inner is str:
            return "string_list", {}
        if get_origin(inner) is Literal:
            return "enum_list", {"choices": list(inner_args)}
        if get_origin(inner) is tuple:
            kind, extra = classify(inner)
            return ("tuple_list", extra) if kind == "tuple" else ("json", {})
    if origin is dict and len(args) == 2 and args[0] is str:
        if args[1] is str:
            return "string_dict", {}
        if args[1] is float:
            return "number_dict", {}
        if get_origin(args[1]) is list and get_args(args[1]) == (str,):
            return "string_list_dict", {}
    kind = _scalar_kind(annotation)
    return kind, {}


def _limits(info: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {"ge": None, "le": None}
    for item in info.metadata:
        for name in out:
            if hasattr(item, name):
                out[name] = getattr(item, name)
    return out


def _model_of(annotation: Any) -> type[BaseModel] | None:
    return (
        annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None
    )


def _segment(item: Any, key: str, index: int, taken: set[str]) -> str:
    """Path segment of a keyed-list entry: its identity value when unique and not numeric,
    else its index (``get_path`` accepts both)."""
    ident = item.get(key) if isinstance(item, dict) else None
    usable = isinstance(ident, str) and ident and not ident.isdigit() and "." not in ident
    if usable and ident not in taken:
        taken.add(ident)
        return ident
    return str(index)


class _Walker:
    """Walks the schema with the effective JSON tree and the layered-default JSON tree."""

    def __init__(
        self,
        values: dict[str, Any],
        defaults: dict[str, Any],
        source_of: Callable[[str], str],
        writable: Callable[[str], bool],
    ) -> None:
        self.values, self.defaults = values, defaults
        self.source_of, self.writable = source_of, writable
        self.out: list[FieldDescriptor] = []

    def model(self, model: type[BaseModel], prefix: str, values: dict[str, Any]) -> None:
        for name, info in model.model_fields.items():
            if name in values:
                self.field(f"{prefix}{name}", name, info, values[name])

    def field(self, path: str, name: str, info: Any, value: Any) -> None:
        annotation = _strip_optional(info.annotation)
        nullable = annotation is not info.annotation
        origin, args = get_origin(annotation), get_args(annotation)
        sub = _model_of(annotation)
        if sub is not None and value is not None:
            self.model(sub, path + ".", value)
            return
        if origin is dict and len(args) == 2 and _model_of(args[1]) is not None:
            self.container(path, "dict_of_model", info, list(value), None)
            for key, entry in value.items():
                self.model(_model_of(args[1]), f"{path}.{key}.", entry)  # type: ignore[arg-type]
            return
        if origin is list and args and _model_of(args[0]) is not None:
            key = KEYED_LISTS.get(path, DEFAULT_LIST_KEY)
            taken: set[str] = set()
            segments = [_segment(item, key, n, taken) for n, item in enumerate(value)]
            self.container(path, "keyed_list", info, segments, key)
            for segment, entry in zip(segments, value, strict=True):
                self.model(_model_of(args[0]), f"{path}.{segment}.", entry)  # type: ignore[arg-type]
            return
        kind, extra = classify(annotation)
        if kind == "string_list_dict":
            self.container(path, kind, info, list(value), None)
            for key, words in value.items():
                self.leaf(f"{path}.{key}", name, info, "string_list", {}, words, False)
            return
        self.leaf(path, name, info, kind, extra, value, nullable or sub is not None)

    def base(self, path: str, info: Any, kind: str) -> FieldDescriptor:
        meta = field_meta(path)
        return {
            "path": path,
            "kind": kind,
            "nullable": False,
            "value": None,
            "default": None,
            "source": self.source_of(path),
            **_limits(info),  # type: ignore[typeddict-item]
            "unit": meta.get("unit"),
            "restart": bool(meta.get("restart")),
            "admin_only": bool(meta.get("admin_only")),
            "opt_in": bool(meta.get("opt_in")),
            "conservative": meta.get("conservative"),
            "description": info.description or "",
            "choices": None,
            "item_kind": None,
            "length": None,
            "key": None,
            "secret": False,
            "writable": False,
        }

    def container(self, path: str, kind: str, info: Any, keys: list[str], key: str | None) -> None:
        row = self.base(path, info, kind)
        row["value"], row["key"] = list(keys), key
        self.out.append(row)

    def leaf(
        self,
        path: str,
        name: str,
        info: Any,
        kind: str,
        extra: dict[str, Any],
        value: Any,
        nullable: bool,
    ) -> None:
        row = self.base(path, info, kind)
        try:
            default = get_path(self.defaults, path)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            default = None
        row.update(extra)  # type: ignore[typeddict-item]
        row["nullable"] = nullable
        row["secret"] = (kind in TEXT_KINDS and is_secret_key(name)) or secret_shaped(
            [value, default]
        )
        if row["secret"]:
            value = REDACTED if value is not None else None
            default = REDACTED if default is not None else None
        row["value"], row["default"] = value, default
        row["writable"] = not row["secret"] and self.writable(path)
        self.out.append(row)


def build_catalogue(
    stack: Any, writable: Callable[[str], bool] = lambda path: False
) -> list[FieldDescriptor]:
    """Descriptors for every field of the effective config of ``stack`` (a ``ConfigStack``),
    in schema order. ``writable(path)`` says which paths the web UI may write; secrets never are.
    Needs ``effective()``, ``merged_tree(include_runtime=False)`` and ``source_of()``."""
    values = stack.effective().model_dump(mode="json")
    layered = RoboEmotConfig.model_validate(stack.merged_tree(include_runtime=False))
    walker = _Walker(values, layered.model_dump(mode="json"), stack.source_of, writable)
    walker.model(RoboEmotConfig, "", values)
    return walker.out


__all__ = ["DESCRIPTOR_KEYS", "KINDS", "FieldDescriptor", "build_catalogue", "classify"]
