"""The field catalogue: one JSON descriptor per config field, for the web UI (SPEC 11A, 12).

``build_catalogue`` walks the ``RoboEmotConfig`` schema together with the effective values and
returns ``FieldDescriptor`` dicts. This shape is a frozen contract (the robot plugin and the
settings page are built against it); ``DESCRIPTOR_KEYS`` and ``KINDS`` pin it in a test.
``descriptor()`` builds one row with every key filled, for read-only rows a robot plugin adds.

Keys of every descriptor (all always present, ``None`` when not applicable):

- ``path``: dotted config path. A dict key or a keyed-list entry (its name, else its index) is one
  segment; dict keys containing "." or secret-shaped are not listed.
- ``kind``: one of ``KINDS``. Scalars ``number`` (float), ``int``, ``bool``, ``text``, ``enum``
  (``choices``); ``tuple`` (fixed length ``length``, items ``item_kind`` number/int/text);
  collections ``string_list``, ``enum_list`` (``choices``), ``tuple_list``, ``string_dict``,
  ``number_dict``; ``json`` for any other shape (an absent optional section too).
  ``dict_of_model``, ``keyed_list`` and ``string_list_dict`` are containers: ``value`` is the list
  of entry keys and each entry follows as descriptors under ``<path>.<key>`` (its fields, or for
  ``string_list_dict`` a ``string_list``). ``keyed_list`` names its identity field in ``key``.
- ``entry``: for ``dict_of_model`` and ``keyed_list`` only: descriptors of a default-built entry
  with paths relative to the entry, to build an "add entry" form (required fields are null).
- ``nullable``: the field accepts null (any kind but the containers).
- ``value`` / ``default``: effective value, and the value without the runtime layer (what a
  reset would give); plain JSON.
- ``source``: layer that decides the value (``ConfigStack.source_of``): ``runtime``, ``derived``,
  ``app``, ``preset``, ``manifest``, ``defaults`` or ``schema``.
- ``ge`` / ``le`` (numeric limits), ``unit``, ``restart``, ``admin_only``, ``opt_in``,
  ``conservative``, ``description``: from the schema (``field_meta`` for the flags).
- ``secret``: the value is a secret or secret-shaped (a field is fully secret if any element of
  its value or default is secret-shaped). ``value`` and ``default`` are then ``"***"`` (or null
  when absent) and the field is never writable from the web, except a route ``base_url`` (see
  ``write_only``): it is replaced whole, never shown.
- ``private``: privacy and location settings (``PRIVATE_PATHS``), admin only (the PIN).
  Unless the catalogue is built with ``reveal_private``, ``value`` and ``default`` are ``"***"``;
  the field itself stays writable. No private path is a container.
  ``admin_only`` is true on every private field. A secret stays redacted even when revealed.
- ``writable``: the web UI may write it today.
- ``removable``: for ``dict_of_model`` and ``keyed_list`` only (else ``None``): the entry keys the
  web UI may remove, i.e. those no layer below the runtime one holds (a merge cannot mask them).
  The page shows Remove from this list only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal, TypedDict, get_args, get_origin

from pydantic import BaseModel

from roboemot.diagnostics import jsonable
from roboemot.redact import (
    REDACTED,
    is_private,
    is_secret_key,
    secret_shaped,
)

from .schema import RoboEmotConfig
from .stack import (
    KEYED_LISTS,
    entry_identity,
    field_meta,
    get_path,
    resolve_in_layer,
    strip_optional,
)

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
ENTRY_KINDS = ("dict_of_model", "keyed_list")
TEXT_KINDS = ("text", "enum", "string_list", "string_dict", "json")
STRUCTURED_KINDS = ("string_dict", "json")
DEFAULT_LIST_KEY = "name"
ENTRY_PLACEHOLDER = "_"
BLANK_SOURCE = "schema"


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
    entry: list[FieldDescriptor] | None
    secret: bool
    private: bool
    writable: bool
    removable: list[str] | None


DESCRIPTOR_KEYS = frozenset(FieldDescriptor.__annotations__)


def _plain(value: Any) -> Any:
    """``value`` as plain JSON (models dumped, tuples as lists)."""
    return json.loads(json.dumps(value, default=lambda o: o.model_dump(mode="json")))


def _is_secret(kind: str, name: str, value: Any, default: Any) -> bool:
    """Secret by field name (text kinds only: ``max_tokens`` is not a token), by value shape, or,
    for free-form maps, by a credential-named inner key (what ``jsonable`` would redact)."""
    if kind in TEXT_KINDS and is_secret_key(name):
        return True
    if secret_shaped([value, default]):
        return True
    return kind in STRUCTURED_KINDS and jsonable([value, default]) != _plain([value, default])


def descriptor(path: str, kind: str, value: Any = None, **fields: Any) -> FieldDescriptor:
    """A descriptor with every key filled (``None``, ``False``, ``""`` or source ``schema``),
    overridden by ``fields``; for read-only rows a robot plugin contributes. Secret and private
    redaction apply as for the catalogue's own rows. Unknown keys or kinds raise ``ValueError``."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    unknown = set(fields) - DESCRIPTOR_KEYS
    if unknown:
        raise ValueError(f"unknown descriptor keys {sorted(unknown)}")
    row: FieldDescriptor = {
        "path": path,
        "kind": kind,
        "nullable": False,
        "value": value,
        "default": None,
        "source": BLANK_SOURCE,
        "ge": None,
        "le": None,
        "unit": None,
        "restart": False,
        "admin_only": False,
        "opt_in": False,
        "conservative": None,
        "description": "",
        "choices": None,
        "item_kind": None,
        "length": None,
        "key": None,
        "entry": None,
        "secret": False,
        "private": False,
        "writable": False,
        "removable": None,
    }
    row.update(fields)  # type: ignore[typeddict-item]
    if kind not in CONTAINERS:
        seal(row, path.rsplit(".", 1)[-1])
    return row


WRITE_ONLY_URL = re.compile(r"llm_routes\.[^.]+\.base_url")


def write_only(path: str) -> bool:
    """A route ``base_url``: secret when it holds credentials, yet replaceable whole from the web
    (the stored value is never sent, a new value may not hold credentials)."""
    return WRITE_ONLY_URL.fullmatch(path) is not None


def seal(row: FieldDescriptor, name: str, reveal_private: bool = False) -> None:
    """Set ``secret`` and ``private`` of a leaf row and redact its ``value`` and ``default``
    (a private value stays visible with ``reveal_private``; a secret never does). A private field
    is admin only; a secret field is never writable (a route ``base_url`` apart)."""
    row["secret"] = _is_secret(row["kind"], name, row["value"], row["default"])
    row["private"] = is_private(row["path"])
    row["admin_only"] = row["admin_only"] or row["private"]
    hide = row["secret"] or (row["private"] and not reveal_private)
    if hide:
        row["value"] = REDACTED if row["value"] is not None else None
        row["default"] = REDACTED if row["default"] is not None else None
    if row["secret"] and not write_only(row["path"]):
        row["writable"] = False


def _scalar_kind(annotation: Any) -> str:
    kinds = {bool: "bool", float: "number", int: "int", str: "text"}
    return kinds.get(annotation, "json")


def classify(annotation: Any) -> tuple[str, dict[str, Any]]:
    """Kind of a non-model annotation and its extras (``choices``, ``item_kind``, ``length``).
    Anything outside the known shapes is ``json``."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        return "enum", {"choices": list(args)}
    if origin is tuple:
        kinds = {_scalar_kind(a) for a in args}
        item = kinds.pop() if len(kinds) == 1 else "json"
        if item == "json":
            return "json", {}
        return "tuple", {"item_kind": item, "length": len(args)}
    if origin is list and args:
        inner = args[0]
        if inner is str:
            return "string_list", {}
        if get_origin(inner) is Literal:
            return "enum_list", {"choices": list(get_args(inner))}
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
    return _scalar_kind(annotation), {}


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


def _addressable(key: Any) -> bool:
    """A dict key that survives as a path segment: no dot, not secret-shaped."""
    return isinstance(key, str) and "." not in key and not secret_shaped(key)


def _segment(item: Any, key: str, index: int, taken: set[str]) -> str:
    """Path segment of a keyed-list entry: its identity value when unique, plain and not numeric,
    else its index (``get_path`` accepts both; layers are matched by identity, not by this)."""
    ident = item.get(key) if isinstance(item, dict) else None
    usable = isinstance(ident, str) and ident and not ident.isdigit() and _addressable(ident)
    if usable and ident not in taken:
        taken.add(ident)
        return ident
    return str(index)


def _blank(model: type[BaseModel]) -> dict[str, Any]:
    """Plain defaults of a model; a required field is null."""
    built = model.model_construct()
    return {
        name: None if info.is_required() else _plain(getattr(built, name))
        for name, info in model.model_fields.items()
    }


class _Walker:
    """Walks the schema with the effective JSON tree and the layered-default JSON tree."""

    def __init__(
        self,
        values: dict[str, Any],
        base: dict[str, Any] | None,
        source_of: Callable[[str], str],
        writable: Callable[[str], bool],
        reveal_private: bool = False,
    ) -> None:
        self.reveal = reveal_private
        self.values, self.base = values, base
        self.source_of, self.writable = source_of, writable
        self.out: list[FieldDescriptor] = []

    def model(self, model: type[BaseModel], prefix: str, values: dict[str, Any]) -> None:
        for name, info in model.model_fields.items():
            if name in values:
                self.field(f"{prefix}{name}", name, info, values[name])

    def field(self, path: str, name: str, info: Any, value: Any) -> None:
        annotation = strip_optional(info.annotation)
        nullable = annotation is not info.annotation
        origin, args = get_origin(annotation), get_args(annotation)
        sub = _model_of(annotation)
        if sub is not None and value is not None:
            self.model(sub, path + ".", value)
            return
        if value is not None and origin is dict and len(args) == 2 and _model_of(args[1]):
            entries = {k: v for k, v in value.items() if _addressable(k)}
            held = self.lower(path)
            fresh = [k for k in entries if isinstance(held, dict) and k not in held]
            self.container(
                path, "dict_of_model", info, list(entries), None, _model_of(args[1]), fresh
            )
            for key, entry in entries.items():
                self.model(_model_of(args[1]), f"{path}.{key}.", entry)  # type: ignore[arg-type]
            return
        if value is not None and origin is list and args and _model_of(args[0]):
            key = KEYED_LISTS.get(path, DEFAULT_LIST_KEY)
            taken: set[str] = set()
            segments = [_segment(item, key, n, taken) for n, item in enumerate(value)]
            held = self.lower(path)
            fresh = [
                seg
                for seg, item in zip(segments, value, strict=True)
                if isinstance(held, list)
                and all(entry_identity(path, e) != entry_identity(path, item) for e in held)
            ]
            self.container(path, "keyed_list", info, segments, key, _model_of(args[0]), fresh)
            for segment, entry in zip(segments, value, strict=True):
                self.model(_model_of(args[0]), f"{path}.{segment}.", entry)  # type: ignore[arg-type]
            return
        kind, extra = classify(annotation)
        if kind == "string_list_dict" and value is not None:
            entries = {k: v for k, v in value.items() if _addressable(k)}
            self.container(path, kind, info, list(entries), None, None)
            for key, words in entries.items():
                self.leaf(f"{path}.{key}", name, info, "string_list", {}, words, False)
            return
        self.leaf(path, name, info, kind, extra, value, nullable or sub is not None)

    def lower(self, path: str) -> Any:
        """The value at ``path`` without the runtime layer; ``None`` when there is none (or in a
        blank entry, which has no lower layer)."""
        if self.base is None:
            return None
        try:
            return get_path(self.base, path)
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    def base_row(self, path: str, info: Any, kind: str) -> FieldDescriptor:
        meta = field_meta(path)
        return descriptor(
            path,
            kind,
            source=self.source_of(path),
            unit=meta.get("unit"),
            restart=bool(meta.get("restart")),
            admin_only=bool(meta.get("admin_only")),
            opt_in=bool(meta.get("opt_in")),
            conservative=meta.get("conservative"),
            description=info.description or "",
            **_limits(info),
        )

    def container(
        self,
        path: str,
        kind: str,
        info: Any,
        keys: list[str],
        key: str | None,
        model: type[BaseModel] | None,
        removable: list[str] | None = None,
    ) -> None:
        row = self.base_row(path, info, kind)
        row["private"] = is_private(path)
        row["admin_only"] = row["admin_only"] or row["private"]
        row["key"] = key
        if kind in ENTRY_KINDS:
            row["removable"] = removable or []
        row["value"] = list(keys)
        if kind in ENTRY_KINDS and model is not None:
            row["entry"] = self.entry(path, model)
        self.out.append(row)

    def entry(self, path: str, model: type[BaseModel]) -> list[FieldDescriptor]:
        """Descriptors of a default-built entry, paths relative to the entry."""
        prefix = f"{path}.{ENTRY_PLACEHOLDER}."
        walker = _Walker(_blank(model), None, lambda p: BLANK_SOURCE, lambda p: False, self.reveal)
        walker.model(model, prefix, walker.values)
        rows = walker.out
        for row in rows:
            row["path"] = row["path"][len(prefix) :]
        return rows

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
        row = self.base_row(path, info, kind)
        if self.base is None:  # a blank entry: its value is its default
            default = value
        else:
            found, default = resolve_in_layer(self.base, self.values, path)
            default = default if found else None
        row.update(extra)  # type: ignore[typeddict-item]
        row["nullable"] = nullable
        row["value"], row["default"] = value, default
        row["writable"] = self.writable(path)
        seal(row, name, self.reveal)
        self.out.append(row)


def build_catalogue(
    stack: Any,
    writable: Callable[[str], bool] = lambda path: False,
    reveal_private: bool = False,
) -> list[FieldDescriptor]:
    """Descriptors for every field of the effective config of ``stack`` (a ``ConfigStack``),
    in schema order. ``writable(path)`` says which paths the web UI may write; secrets never are.
    ``reveal_private`` shows private values (the caller has checked the PIN). Needs ``effective()``,
    ``merged_tree(include_runtime=False)`` and ``source_of(path, tree)``."""
    values = stack.effective().model_dump(mode="json")
    layered = RoboEmotConfig.model_validate(stack.merged_tree(include_runtime=False))
    walker = _Walker(
        values,
        layered.model_dump(mode="json"),
        lambda path: stack.source_of(path, values),
        writable,
        reveal_private,
    )
    walker.model(RoboEmotConfig, "", values)
    return walker.out


__all__ = [
    "CONTAINERS",
    "DESCRIPTOR_KEYS",
    "KINDS",
    "FieldDescriptor",
    "build_catalogue",
    "classify",
    "write_only",
    "descriptor",
    "seal",
]
