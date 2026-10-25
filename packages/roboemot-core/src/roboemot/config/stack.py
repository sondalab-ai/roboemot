"""Layered configuration with validation, guardrails, TTL overrides, audit and hot reload.

Layers, lowest to highest precedence (SPEC 11A, Layers):

1. ``defaults``  library defaults (the schema defaults plus ``configs/defaults/*.yaml``)
2. ``manifest``  behaviour-relevant settings carried by the robot manifest
3. ``preset``    personality preset, resolved through ``extends``
4. ``app``       the application's config file, then ``derived``: values the library computes
                 in place of app values (persona traits rated by an LLM), never written to a file
5. ``person``    per-person overrides, supplied per query
6. ``runtime``   API / web UI overrides, optionally with a TTL

Lists of dicts with a natural key are merged by that key instead of replaced
(``appraisal.rules`` by ``event``, ``llm_routes`` by ``name``).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ValidationError

from ..redact import scrub, scrub_text, shown
from .schema import (
    SCHEMA_VERSION,
    WITHDRAW_RULE_FIELDS,
    WITHDRAW_SHARED_FIELDS,
    RoboEmotConfig,
    VoiceConfig,
    rule_identity,
)

log = logging.getLogger(__name__)

LAYER_ORDER = ("defaults", "manifest", "preset", "app", "derived", "person", "runtime")
KEYED_LISTS = {"appraisal.rules": "event", "llm_routes": "name"}
AUDIT_FILE_MODE = 0o600
APP_FILE_MODE = 0o600  # app.yaml: owner only, it may hold a literal api_key (SPEC 12)
_MOVED = yaml.safe_load(Path(__file__).with_name("moved_settings.yaml").read_text())
RENAMED_SETTINGS: dict[str, str] = _MOVED["renamed"]  # old path -> new path (web UI file only)
MIGRATED_SETTINGS: dict[str, str] = _MOVED["migrated"]  # old path -> new path (everywhere)
REMOVED_SETTINGS: dict[str, str] = _MOVED["removed"]  # old path (or prefix) -> what replaces it
DROPPED_SETTINGS: dict[str, str] = _MOVED["dropped"]  # old path (``*``: any entry) -> why
# keyed list path -> {identity of an entry that no longer exists: why}
DROPPED_ENTRIES: dict[str, dict[str, str]] = _MOVED["dropped_entries"]
# keyed list path -> {a ``when`` key no event payload carries any more: why}
DROPPED_WHEN: dict[str, dict[str, str]] = _MOVED["dropped_when"]
DROP = object()  # ``without_dropped``: the value is a dropped setting as a whole
_dropped_logged: set[tuple[str, str]] = set()  # (source, path) pairs already logged
NO_KEY = ("", "none")  # api_key values that carry no key
PATCH = "#patch"  # runtime slot suffix: edits of existing entries of a merged list, by identity


def slot_base(slot: str) -> str:
    """The config path a runtime slot belongs to (a patch slot shares its list's path)."""
    return slot.removesuffix(PATCH)


def _plain(value: Any) -> Any:
    """``value`` as plain JSON (models dumped, tuples as lists); ``str`` for anything else."""

    def dump(obj: Any) -> Any:
        return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else str(obj)

    return json.loads(json.dumps(value, default=dump))


class ConfigError(ValueError):
    """Invalid configuration (unknown key, out of range, bad file)."""


class GuardrailError(PermissionError):
    """A change would loosen a safety parameter or enable a privacy option without consent.

    ``violations`` are the structured refusals (empty when raised from elsewhere); the message is
    built from redacted values, so it is safe to log.
    """

    def __init__(self, message: str = "", violations: Iterable[Violation] = ()) -> None:
        super().__init__(message)
        self.violations = list(violations)


@dataclass
class AuditEntry:
    time: float
    source: str
    key: str
    old: Any
    new: Any


@dataclass
class _RuntimeValue:
    value: Any
    expires_at: float | None
    # Guarded values an admin approved for this runtime override only; they lapse with it.
    approved: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigChange:
    changed_keys: list[str]
    restart_required: list[str]
    config: RoboEmotConfig


# ----------------------------------------------------------------------------- dict helpers


def deep_merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        p = f"{path}.{key}" if path else key
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value, p)
        elif p in KEYED_LISTS and isinstance(value, list) and isinstance(out.get(key), list):
            out[key] = _merge_keyed(out[key], value, KEYED_LISTS[p])
        else:
            out[key] = copy.deepcopy(value)
    return out


def _merge_id(item: dict, key: str) -> Any:
    """Merge identity of a keyed-list entry. Appraisal rules also carry their ``when``
    condition, so two rules for one event with different conditions stay two rules."""
    if item.get(key) is None:
        return None
    if key == "event":
        return rule_identity(item[key], item.get("when"))
    return item[key]


def _merge_keyed(base: list, over: list, key: str) -> list:
    result = [copy.deepcopy(x) for x in base]
    index: dict[Any, int] = {}
    for i, x in enumerate(result):
        if isinstance(x, dict):
            index.setdefault(_merge_id(x, key), i)  # duplicates: the first entry is the one
    for item in over:
        k = _merge_id(item, key) if isinstance(item, dict) else None
        if k is not None and k in index:
            result[index[k]] = deep_merge(result[index[k]], item)
        else:
            result.append(copy.deepcopy(item))
    return result


def _patch_keyed(base: list, patches: list, key: str) -> list:
    """``base`` with each patch merged into the entry of the same identity; a patch whose entry is
    gone is dropped (it must not turn into an entry of its own)."""
    result = [copy.deepcopy(x) for x in base]
    index: dict[Any, int] = {}
    for i, x in enumerate(result):
        if isinstance(x, dict):
            index.setdefault(_merge_id(x, key), i)
    for item in patches:
        at = index.get(_merge_id(item, key)) if isinstance(item, dict) else None
        if at is not None:
            result[at] = deep_merge(result[at], item)
    return result


def _list_item(items: list, part: str, dotted: str) -> Any:
    """List element by index, or by natural key (``llm_routes.cloud``, ``appraisal.rules.x``)."""
    if part.isdigit():
        return items[int(part)]
    for item in items:
        key = item if isinstance(item, dict) else getattr(item, "__dict__", {})
        if part in (key.get("name"), key.get("event")):
            return item
    raise KeyError(f"{dotted}: no list entry named {part!r}")


def entry_identity(list_path: str, entry: Any) -> Any:
    """Stable identity of a keyed-list entry (``None`` when it has none): its ``name`` or, for
    appraisal rules, the event plus its normalised ``when`` (``rule_identity``). Index and natural
    key in a path are only addresses; this is what stays the same across layers."""
    key = KEYED_LISTS.get(list_path, "name")
    return _merge_id(entry, key) if isinstance(entry, dict) else None


def resolve_in_layer(layer: Any, effective: Any, path: str) -> tuple[bool, Any]:
    """Find ``path`` of the effective tree inside one layer's tree: ``(found, value)``.

    Dict steps are plain keys. A list step (index or natural key, as ``get_path`` takes) picks the
    entry of the *effective* tree, takes its ``entry_identity`` and then looks for the entry with
    the same identity in the layer's list, wherever it sits there. So an app rule appended at
    index 0 does not stand for the default rule at index 0. ``found`` is False when the layer does
    not hold the path, or an entry has no identity.
    """
    eff, lay, walked = effective, layer, ""
    for part in path.split("."):
        if isinstance(eff, list):
            try:
                item = _list_item(eff, part, path)
            except (KeyError, IndexError):
                return False, None
            ident = entry_identity(walked, item)
            if ident is None or not isinstance(lay, list):
                return False, None
            match = [x for x in lay if entry_identity(walked, x) == ident]
            if not match:
                return False, None
            eff, lay = item, match[0]
        elif isinstance(eff, dict) and isinstance(lay, dict):
            if part not in eff or part not in lay:
                return False, None
            eff, lay = eff[part], lay[part]
        else:
            return False, None
        walked = f"{walked}.{part}" if walked else part
    return True, lay


def ident_fields(list_path: str, entry: dict) -> dict[str, Any]:
    """The fields of a keyed-list entry that make up its identity (``name``; ``event`` plus a
    non-empty ``when``), enough to address that entry in any layer."""
    key = KEYED_LISTS.get(list_path, "name")
    out = {key: entry[key]}
    if key == "event" and entry.get("when"):
        out["when"] = copy.deepcopy(entry["when"])
    return out


def keyed_ref(
    tree: Any, path: str, *, named: bool = False
) -> tuple[str, dict[str, Any], str] | None:
    """``(list path, identity fields, rest)`` when ``path`` of ``tree`` reaches into an entry of a
    ``KEYED_LISTS`` list by *index* (``appraisal.rules.3.valence``), else ``None``. A name in the
    path is already stable, so it is only resolved with ``named`` (callers that want one identity
    form for both); an index is only where the entry sits now."""
    parts, node, walked = path.split("."), tree, ""
    for n, part in enumerate(parts):
        if isinstance(node, list):
            if walked not in KEYED_LISTS or not (part.isdigit() or named):
                return None
            try:
                item = node[int(part)] if part.isdigit() else _list_item(node, part, path)
            except (IndexError, KeyError):
                return None
            if not isinstance(item, dict) or item.get(KEYED_LISTS[walked]) is None:
                return None
            return walked, ident_fields(walked, item), ".".join(parts[n + 1 :])
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
        walked = f"{walked}.{part}" if walked else part
    return None


def keyed_index(tree: Any, list_path: str, ident: dict[str, Any]) -> int | None:
    """Where the entry with identity ``ident`` sits in the list at ``list_path`` of ``tree``."""
    try:
        items = get_path(tree, list_path)
    except (KeyError, IndexError, ValueError, TypeError):
        return None
    key = KEYED_LISTS[list_path]
    want = _merge_id(ident, key)
    for n, item in enumerate(items):
        if isinstance(item, dict) and _merge_id(item, key) == want:
            return n
    return None


def nest(dotted: str, value: Any) -> dict:
    """``value`` wrapped in dicts along ``dotted``: the layer tree that holds just that path."""
    for part in reversed(dotted.split(".")):
        value = {part: value}
    return value


def _pop_leaf(entry: dict, rest: str) -> tuple[bool, Any]:
    """Remove ``rest`` (a dotted path) from ``entry`` and the parents it leaves empty."""
    parts = rest.split(".")
    chain = [entry]
    try:
        for part in parts[:-1]:
            chain.append(chain[-1][part])
        old = chain[-1].pop(parts[-1])
    except (KeyError, AttributeError, TypeError):
        return False, None
    for n in range(len(parts) - 2, -1, -1):
        if chain[n + 1]:
            break
        chain[n].pop(parts[n])
    return True, old


def set_in_layer(layer: dict, effective: Any, dotted: str, value: Any) -> None:
    """``set_path`` for one layer's own tree: a list entry is found in the layer by identity
    (``entry_identity``), not by the index it has in the effective tree. A merged list
    (``KEYED_LISTS``) gets the entry as an identity-only skeleton when the layer lacks it; any other
    list is copied whole from the effective tree first (a layer's list replaces the lower one)."""
    parts = dotted.split(".")
    eff, lay, walked = effective, layer, ""
    for n, part in enumerate(parts):
        last = n == len(parts) - 1
        if isinstance(eff, list):
            item = _list_item(eff, part, dotted)
            ident = entry_identity(walked, item)
            if ident is None or not isinstance(lay, list):
                raise ConfigError(f"{dotted}: list entry has no identity")
            match = next((x for x in lay if entry_identity(walked, x) == ident), None)
            if match is None:
                match = ident_fields(walked, item) if walked in KEYED_LISTS else copy.deepcopy(item)
                lay.append(match)
            if last:
                match.clear()
                match.update(value)
                return
            eff, lay = item, match
        else:
            if not isinstance(lay, dict):
                raise ConfigError(f"{dotted}: {part} is not a section")
            if last:
                lay[part] = value
                return
            child = eff.get(part) if isinstance(eff, dict) else None
            if isinstance(child, list):
                lay = lay.setdefault(
                    part,
                    [] if f"{walked}.{part}".lstrip(".") in KEYED_LISTS else copy.deepcopy(child),
                )
            else:
                lay = lay.setdefault(part, {})
            eff = child
        walked = f"{walked}.{part}" if walked else part
        if not isinstance(lay, dict | list):
            raise ConfigError(f"{dotted}: {part} is not a section")


def set_path(tree: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = tree
    for part in parts[:-1]:
        if isinstance(node, list):
            try:
                node = _list_item(node, part, dotted)
            except KeyError as exc:
                raise ConfigError(str(exc)) from exc
        else:
            node = node.setdefault(part, {})
        if not isinstance(node, (dict, list)):
            raise ConfigError(f"{dotted}: {part} is not a section")
    if isinstance(node, list):
        raise ConfigError(f"{dotted}: address a list entry by name, then a field")
    node[parts[-1]] = value


def get_path(tree: Any, dotted: str) -> Any:
    node = tree
    for part in dotted.split("."):
        if isinstance(node, BaseModel):
            node = getattr(node, part)
        elif isinstance(node, dict):
            node = node[part]
        elif isinstance(node, list):
            node = _list_item(node, part, dotted)
        else:
            raise KeyError(dotted)
    return node


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, dict):
        out: dict[str, Any] = {}
        for k, v in tree.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: tree}


INHERITED_META = ("admin_only", "opt_in")  # a guard on a container covers everything inside it


def field_meta(dotted: str) -> dict[str, Any]:
    """Metadata (unit, restart, conservative, opt_in, admin_only) of a path, {} if unknown.
    ``admin_only`` and ``opt_in`` set on a container (a dict or model field) apply to every
    leaf inside it.

    Dict-valued fields (``voice.moods``) and list fields (``llm_routes``) consume one extra
    path element (the key or the index).
    """
    parts = dotted.split(".")
    model: Any = RoboEmotConfig
    meta: dict[str, Any] = {}
    inherited: dict[str, Any] = {}
    i = 0
    while i < len(parts):
        fields = getattr(model, "model_fields", None)
        if not fields or parts[i] not in fields:
            return {}
        info = fields[parts[i]]
        extra = info.json_schema_extra
        meta = {**inherited, **(extra if isinstance(extra, dict) else {})}
        inherited = {k: v for k, v in meta.items() if k in INHERITED_META and v}
        annotation = strip_optional(info.annotation)
        i += 1
        if get_origin(annotation) in (dict, list):
            i += 1  # skip the key or index
        model = _inner_model(annotation)
        if model is None:
            return meta if i >= len(parts) else {}
    return meta


def strip_optional(annotation: Any) -> Any:
    args = [a for a in (get_args(annotation) or ()) if a is not type(None)]
    if get_origin(annotation) in (Union, UnionType) and len(args) == 1:
        return args[0]
    return annotation


def _inner_model(annotation: Any) -> Any:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in getattr(annotation, "__args__", ()) or ():
        found = _inner_model(arg)
        if found is not None:
            return found
    return None


def load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    data, found = without_dropped("", data)
    log_dropped(str(path), found)
    for old, moved in migrate_paths(data):
        log_migrated(str(path), old, moved=moved)
    return data


def migrate_paths(data: dict[str, Any]) -> list[tuple[str, bool]]:
    """Move, in place, every ``MIGRATED_SETTINGS`` path found in the nested ``data`` to its new
    path and drop the sections it leaves empty. A value already at the new path wins: the old
    one is then dropped. Returns ``(old path, moved)`` for each old path found."""
    found: list[tuple[str, bool]] = []
    for old, new in MIGRATED_SETTINGS.items():
        *parents, leaf = old.split(".")
        node: Any = data
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            continue
        value = node.pop(leaf)
        try:
            get_path(data, new)
            found.append((old, False))
        except (KeyError, AttributeError, IndexError, TypeError):
            set_path(data, new, value)
            found.append((old, True))
        _prune(data, parents)
    return found


def _prune(data: dict[str, Any], parents: list[str]) -> None:
    """Remove the empty mappings along ``parents``, deepest first."""
    for depth in range(len(parents), 0, -1):
        node: Any = data
        for part in parents[: depth - 1]:
            node = node.get(part) if isinstance(node, dict) else None
        key = parents[depth - 1]
        if isinstance(node, dict) and node.get(key) == {}:
            del node[key]


def log_migrated(source: str, path: str, *, moved: bool) -> None:
    """One warning per ``source`` and migrated ``path``, naming the paths and never the value:
    moved to its new path, or ignored because a value was already set there."""
    if (source, path) not in _dropped_logged:
        _dropped_logged.add((source, path))
        new = MIGRATED_SETTINGS[path]
        if moved:
            log.warning("%s: setting %s moved to %s", source, path, new)
        else:
            log.warning("%s: setting %s ignored: %s already set", source, path, new)


def without_dropped(path: str, value: Any) -> tuple[Any, list[str]]:
    """``value`` stored at ``path`` (``""`` for a whole file) without the dropped settings it
    holds (``DROPPED_SETTINGS``), removed in place, and their paths as found. ``DROP`` when
    ``path`` is itself a dropped setting or lies under one; a ``*`` in ``path`` stands for any
    list entry (the identity of a web UI file key). Entries of a keyed list whose identity is in
    ``DROPPED_ENTRIES`` are removed too, found as ``list[identity]``."""
    here = [p for p in path.split(".") if p]
    found: list[str] = []
    for pattern in DROPPED_SETTINGS:
        parts = pattern.split(".")
        if not all(p in ("*", h) for p, h in zip(parts, here, strict=False)):
            continue
        if len(here) >= len(parts):
            return DROP, [path]
        _strip(value, parts[len(here) :], here, found)
    for list_path in sorted(DROPPED_ENTRIES.keys() | DROPPED_WHEN.keys()):
        parts = list_path.split(".")
        if here == parts[: len(here)]:
            _strip_entries(value, parts[len(here) :], list_path, found)
    return value, found


def dropped_entry(list_path: str, ident: dict[str, Any]) -> bool:
    """Whether the keyed-list entry ``ident`` of ``list_path`` no longer exists: its identity is
    in ``DROPPED_ENTRIES``, or its ``when`` names a key in ``DROPPED_WHEN`` (a rule that could
    never match again)."""
    if list_path not in KEYED_LISTS:
        return False
    when = ident.get("when")
    gone = DROPPED_WHEN.get(list_path, {})
    return ident.get(KEYED_LISTS[list_path]) in DROPPED_ENTRIES.get(list_path, {}) or (
        isinstance(when, dict) and any(k in gone for k in when)
    )


def _strip_entries(node: Any, parts: list[str], list_path: str, found: list[str]) -> None:
    """Delete, in place, the entries of the keyed list ``parts`` names under ``node`` that
    ``dropped_entry`` matches, appending ``list_path[identity]`` for each to ``found``."""
    for part in parts:
        node = node.get(part) if isinstance(node, dict) else None
    if not isinstance(node, list):
        return
    key = KEYED_LISTS[list_path]
    gone = [e for e in node if isinstance(e, dict) and dropped_entry(list_path, e)]
    node[:] = [e for e in node if not any(e is g for g in gone)]
    found += [f"{list_path}[{e[key]}]" for e in gone]


def _strip(node: Any, parts: list[str], trail: list[str], found: list[str]) -> None:
    """Delete the keys ``parts`` names under ``node`` (``*``: every list entry or mapping
    value), appending the full path of each deleted key (``trail`` is ``node``'s) to ``found``."""
    head, rest = parts[0], parts[1:]
    if head == "*":
        items: list[tuple[Any, Any]] = list(enumerate(node)) if isinstance(node, list) else []
        items += list(node.items()) if isinstance(node, dict) else []
    else:
        items = [(head, node[head])] if isinstance(node, dict) and head in node else []
    for key, child in items:
        if rest:
            before = len(found)
            _strip(child, rest, [*trail, str(key)], found)
            if len(found) > before and child == {} and isinstance(node, dict):
                del node[key]  # a section that held only dropped settings goes with them
        elif isinstance(node, dict):
            del node[key]
            found.append(".".join([*trail, str(key)]))


def log_dropped(source: str, paths: Iterable[str]) -> None:
    """One warning per ``source`` and dropped ``path``, naming the path and never its value."""
    for path in paths:
        if (source, path) not in _dropped_logged:
            _dropped_logged.add((source, path))
            log.warning("%s: setting %s ignored: removed from the configuration", source, path)


def resolve_preset(name: str, preset_dirs: Iterable[Path], _seen: tuple[str, ...] = ()) -> dict:
    """Load preset ``name`` and everything it ``extends``, base first."""
    if name in _seen:
        raise ConfigError(f"preset cycle: {' -> '.join((*_seen, name))}")
    for d in preset_dirs:
        p = Path(d) / f"{name}.yaml"
        if p.exists():
            data = load_yaml(p)
            parent = data.pop("extends", None)
            data.setdefault("personality", name)
            if parent:
                base = resolve_preset(parent, preset_dirs, (*_seen, name))
                base.pop("personality", None)
                return deep_merge(base, data)
            return data
    raise ConfigError(f"preset not found: {name}")


# ----------------------------------------------------------------------------- guardrails

MANIFEST_BEHAVIOUR_KEYS = frozenset(
    {"expression_map", "gesture_map", "circadian_motion", "speech_motion_map", "voice", "motion"}
)  # what a robot manifest may set
MANIFEST_MOTION_KEYS = frozenset({"app_tick_hz", "arbiter"})  # the rates of ``motion``
MANIFEST_ARBITER_KEYS = frozenset({"loop_hz"})  # the rate of ``motion.arbiter``
MANIFEST_VOICE_KEYS = frozenset(
    {"default_name", "keywords", "base", "limits"}
)  # the name fields and the pitch of ``voice``
MANIFEST_VOICE_PITCH_KEYS = frozenset(
    {"pitch_st"}
)  # all that ``voice.base`` / ``voice.limits`` take
MANIFEST_KEYWORD_KEYS = frozenset({"name_aliases"})  # per language under ``voice.keywords``
PERSON_SECTIONS = ("register", "voice", "conversation", "persona.traits")


@dataclass(frozen=True)
class Violation:
    """A leaf that is less safe than approved. ``check`` says which rule found it: ``field`` (the
    field's own direction, opt-in or admin flag) or one of the cross-field rules ``reflex``,
    ``locality``, ``secret``, which a caller can refuse whatever its authority."""

    kind: Literal["safety", "privacy", "admin"]
    path: str
    approved: Any
    candidate: Any
    check: Literal["field", "reflex", "locality", "secret"] = "field"

    def __str__(self) -> str:
        return (
            f"{self.path} ({shown(self.path, self.approved)} -> {shown(self.path, self.candidate)})"
        )


def _guarded(meta: dict[str, Any]) -> bool:
    return bool(meta.get("conservative") or meta.get("opt_in") or meta.get("admin_only"))


def guarded_leaves(cfg: RoboEmotConfig) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Every safety-relevant leaf of a config, routes and reflex rules keyed by name, not index."""
    tree = cfg.model_dump(mode="python")
    tree["llm_routes"] = {r["name"]: r for r in tree.get("llm_routes", [])}
    rules = tree.get("reflexes", {}).get("rules", [])
    tree["reflexes"]["rules"] = {r["name"]: r for r in rules}  # by name: a removal shifts nothing
    out = {}
    for path, value in flatten(tree).items():
        meta = field_meta(path)
        if path.endswith(".max_reply_sentences") and path.startswith("register.") and value is None:
            value = cfg.conversation.max_reply_sentences  # unset means the conversation default
        if _guarded(meta):
            out[path] = (value, meta)
    return out


NEW_RULE_REFUSED = "reflexes.rules."


def _name_chars_ceiling() -> int:
    """The schema's upper bound of ``voice.name_max_chars``: the longest name any valid
    configuration can allow."""
    meta = VoiceConfig.model_fields["name_max_chars"].metadata
    return max(m.le for m in meta if hasattr(m, "le"))


def check_manifest_motion(motion: Any) -> None:
    """A manifest may set the loop rates of its hardware (``motion.app_tick_hz`` and
    ``motion.arbiter.loop_hz``) and nothing else under ``motion``: limits and blending stay
    with the schema and the owner."""
    if motion is None:
        return
    if not isinstance(motion, dict) or set(motion) - MANIFEST_MOTION_KEYS:
        extra = sorted(set(motion) - MANIFEST_MOTION_KEYS) if isinstance(motion, dict) else motion
        raise ConfigError(
            f"manifest behaviour motion may set {sorted(MANIFEST_MOTION_KEYS)}, not {extra}"
        )
    arbiter = motion.get("arbiter", {})
    if not isinstance(arbiter, dict) or set(arbiter) - MANIFEST_ARBITER_KEYS:
        extra = (
            sorted(set(arbiter) - MANIFEST_ARBITER_KEYS) if isinstance(arbiter, dict) else arbiter
        )
        raise ConfigError(
            f"manifest behaviour motion.arbiter may set {sorted(MANIFEST_ARBITER_KEYS)}, "
            f"not {extra}"
        )


def check_manifest_voice(voice: Any) -> None:
    """A manifest may name the robot (``voice.default_name``, per-language
    ``voice.keywords.<lang>.name_aliases``) and give it a voice pitch (``voice.base.pitch_st``,
    ``voice.limits.pitch_st`` as a [lo, hi] pair holding the base) and nothing else under
    ``voice``. Each name must
    survive ``sanitise_name`` unchanged and fit the schema's maximum of ``voice.name_max_chars``,
    not the owner's own setting: a tight owner cap must not make the shipped manifest unloadable
    (the names are only matched, never shown, so the cap has nothing to cut)."""
    from ..voice.keywords import normalise
    from ..voice.robot_name import sanitise_name

    if voice is None:
        return
    cap = _name_chars_ceiling()

    def check_name(where: str, value: Any) -> None:
        if not isinstance(value, str) or sanitise_name(value, cap) != value:
            raise ConfigError(
                f"manifest behaviour voice.{where}: {value!r} must be letters, digits, spaces "
                f"and '.- only, at most {cap} characters"
            )

    if not isinstance(voice, dict) or set(voice) - MANIFEST_VOICE_KEYS:
        extra = sorted(set(voice) - MANIFEST_VOICE_KEYS) if isinstance(voice, dict) else voice
        raise ConfigError(
            f"manifest behaviour voice may set {sorted(MANIFEST_VOICE_KEYS)}, not {extra}"
        )
    if "default_name" in voice:
        check_name("default_name", voice["default_name"])
    check_manifest_voice_pitch(voice)
    keywords = voice.get("keywords", {})
    if not isinstance(keywords, dict):
        raise ConfigError("manifest behaviour voice.keywords must map languages to name_aliases")
    for language, entry in keywords.items():
        where = f"keywords.{language}"
        if not isinstance(entry, dict) or set(entry) - MANIFEST_KEYWORD_KEYS:
            raise ConfigError(
                f"manifest behaviour voice.{where} may set {sorted(MANIFEST_KEYWORD_KEYS)} only"
            )
        aliases = entry.get("name_aliases", [])
        if not isinstance(aliases, list):
            raise ConfigError(f"manifest behaviour voice.{where}.name_aliases must be a list")
        for alias in aliases:
            check_name(f"{where}.name_aliases", alias)
            if not normalise(alias):
                raise ConfigError(f"manifest behaviour voice.{where}.name_aliases: empty alias")


def check_manifest_voice_pitch(voice: dict) -> None:
    """``voice.base`` and ``voice.limits`` take ``pitch_st`` only; the limits are a [lo, hi] pair
    and must hold the base (each side falls back to the schema default when the manifest sets
    only the other)."""
    for block in ("base", "limits"):
        entry = voice.get(block, {})
        if not isinstance(entry, dict) or set(entry) - MANIFEST_VOICE_PITCH_KEYS:
            extra = (
                sorted(set(entry) - MANIFEST_VOICE_PITCH_KEYS) if isinstance(entry, dict) else entry
            )
            raise ConfigError(
                f"manifest behaviour voice.{block} may set {sorted(MANIFEST_VOICE_PITCH_KEYS)}, "
                f"not {extra}"
            )
    if not voice.get("base") and not voice.get("limits"):
        return
    limits = voice.get("limits", {}).get("pitch_st", VoiceConfig().limits.pitch_st)
    if (
        not isinstance(limits, (list, tuple))
        or len(limits) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in limits)
    ):
        raise ConfigError("manifest behaviour voice.limits.pitch_st must be a [lo, hi] pair")
    lo, hi = limits
    base = voice.get("base", {}).get("pitch_st", VoiceConfig().base.pitch_st)
    if not isinstance(base, (int, float)) or isinstance(base, bool) or not lo <= base <= hi:
        raise ConfigError(
            f"manifest behaviour voice.limits.pitch_st {[lo, hi]} must hold "
            f"voice.base.pitch_st {base!r}"
        )


def violations(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> list[Violation]:
    """Leaves where ``candidate`` is less safe than ``approved`` (SPEC 11A, Guardrails)."""
    base = guarded_leaves(approved)
    found = []
    for path, (value, meta) in guarded_leaves(candidate).items():
        direction = meta.get("conservative")
        if path.startswith(NEW_RULE_REFUSED) and path not in base:
            # A reflex rule (new or renamed) that the approved config does not have is a new motion
            # the approval never saw: refused whole, not compared with anything (SPEC 10, 12).
            found.append(Violation("safety", path, None, value, "reflex"))
            continue
        if path in base:
            ref = base[path][0]
        else:  # e.g. a new LLM route: compare against the most conservative value
            ref = {"false": False, "true": True}.get(direction, value)
            if meta.get("opt_in"):
                ref = False
        ok = {
            "lower": lambda v, r: v <= r,
            "higher": lambda v, r: v >= r,
            "false": lambda v, r: (not v) or v == r,
            "true": lambda v, r: bool(v) or v == r,
            None: lambda v, r: True,
        }[direction](value, ref)
        if meta.get("admin_only") and value != ref:
            found.append(Violation("admin", path, ref, value))
            if not ok:  # an admin may approve the change, never a loosening of the direction
                found.append(Violation("safety", path, ref, value))
        elif meta.get("opt_in") and value and not ref:
            found.append(Violation("privacy", path, ref, value))
        elif not ok:
            found.append(Violation("safety", path, ref, value))
    cand = guarded_leaves(candidate)
    for path, (ref, meta) in base.items():  # an admin-only entry (a language's list) removed
        if meta.get("admin_only") and path not in cand and not path.startswith(NEW_RULE_REFUSED):
            found.append(Violation("admin", path, ref, None))
    return (
        found
        + reflex_violations(approved, candidate)
        + locality_violations(approved, candidate)
        + secret_violations(approved, candidate)
    )


def locality_violations(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> list[Violation]:
    """Routes that would see real names without the approval having trusted them (SPEC 8, 11A,
    12). A route is local by its effective locality (``route_is_local``: the ``locality``
    field and the host together), so a new ``auto`` route on a LAN host, a ``base_url`` moved
    to a ``*.local`` host and ``locality: local`` are all the same loosening. A local route
    needs an approved route of the same name that was local with the same ``base_url``;
    anything that makes a route cloud is stricter and passes. The values are exact (approval
    matches on them); ``Violation.__str__`` redacts."""
    from ..llm.pseudonyms import route_is_local  # lazy: llm imports the config package

    trusted = {r.name: r.base_url for r in approved.llm_routes if route_is_local(r)}
    return [
        Violation(
            "privacy",
            f"llm_routes.{r.name}.locality",
            f"local {trusted[r.name]}" if r.name in trusted else "cloud or absent",
            f"local {r.base_url} ({r.locality})",
            "locality",
        )
        for r in candidate.llm_routes
        if route_is_local(r) and trusted.get(r.name) != r.base_url
    ]


def endpoint_key(url: str) -> tuple[Any, ...]:
    """Scheme, host and port of a URL: where a key would be sent."""
    try:
        parts = urlsplit(url)
        return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port
    except ValueError:
        return (url,)


def secret_violations(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> list[Violation]:
    """Routes that would send a key somewhere the approval did not trust (SPEC 8, 11A, 12). A
    route with a key (a ``${secret:name}`` reference or a literal; not empty or ``none``) is
    trusted with exactly its approved endpoint (scheme, host, port) and ``api_key``. A candidate
    route with a key that is new, has another endpoint (an ``http`` downgrade included) or
    another ``api_key`` needs the admin: otherwise a runtime change could point a key at a host
    of its choosing. The values are exact; ``Violation.__str__`` redacts them."""
    trusted = {r.name: r for r in approved.llm_routes if r.api_key not in NO_KEY}

    def same(r: Any) -> bool:
        ref = trusted.get(r.name)
        return (
            ref is not None
            and endpoint_key(ref.base_url) == endpoint_key(r.base_url)
            and ref.api_key == r.api_key
        )

    return [
        Violation(
            "admin",
            f"llm_routes.{r.name}.api_key",
            f"{trusted[r.name].base_url} {trusted[r.name].api_key}"
            if r.name in trusted
            else "absent",
            f"{r.base_url} {r.api_key}",
            "secret",
        )
        for r in candidate.llm_routes
        if r.api_key not in NO_KEY and not same(r)
    ]


def reflex_violations(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> list[Violation]:
    """Reflex changes that no single field's direction can express (SPEC 10, 11A, 12).

    Rules cannot be removed, ``trigger`` and ``primitive`` are their identity, a disabled rule
    cannot be enabled, and the threshold is safer upward for a sound or a voice (fires less) and
    downward for a close face (the withdraw fires earlier).

    A withdraw rule is protective: every field that affects it is frozen at runtime, in both
    directions (``WITHDRAW_RULE_FIELDS``, and the shared settings in ``WITHDRAW_SHARED_FIELDS``
    that the engine reads for it, plus ``audio`` when a sound drives it). It can only be changed
    in the base config.
    """
    found: list[Violation] = []

    def refuse(path: str, was: Any, now: Any) -> None:
        found.append(Violation("safety", path, was, now, "reflex"))

    base = {r.name: r for r in approved.reflexes.rules}
    cand = {r.name: r for r in candidate.reflexes.rules}
    for name in base:
        if name not in cand:
            refuse(f"reflexes.rules.{name}", "present", "removed")
    protective = [r for r in base.values() if r.primitive == "withdraw" and r.enabled]
    ar, cr = approved.reflexes, candidate.reflexes
    if protective:
        shared = list(WITHDRAW_SHARED_FIELDS)
        if any(r.trigger == "loud_onset" for r in protective):
            shared.append("audio")
        for attr in shared:
            if getattr(ar, attr) != getattr(cr, attr):
                refuse(f"reflexes.{attr}", getattr(ar, attr), getattr(cr, attr))
    for rule in candidate.reflexes.rules:
        ref = base.get(rule.name)
        if ref is None:
            continue  # refused whole by the path check
        at = f"reflexes.rules.{rule.name}"
        if ref.primitive == "withdraw":
            for attr in WITHDRAW_RULE_FIELDS:
                if getattr(rule, attr) != getattr(ref, attr):
                    refuse(f"{at}.{attr}", getattr(ref, attr), getattr(rule, attr))
            continue
        for attr in ("trigger", "primitive"):
            if getattr(rule, attr) != getattr(ref, attr):
                refuse(f"{at}.{attr}", getattr(ref, attr), getattr(rule, attr))
        if rule.trigger != ref.trigger:
            continue
        if rule.enabled and not ref.enabled:
            refuse(f"{at}.enabled", False, True)
        if rule.threshold < ref.threshold:  # a sound or a voice: lower fires more
            refuse(f"{at}.threshold", ref.threshold, rule.threshold)
    return found


# ----------------------------------------------------------------------------- the stack


def _mutator(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A method that edits the stack: whatever happens inside, the cached effective configuration
    is dropped on the way out (a safety net over the targeted invalidations)."""

    @functools.wraps(fn)
    def wrapper(self: ConfigStack, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        finally:
            with self._lock:
                self._invalidate()

    return wrapper


# Attributes whose rebinding changes the merged result: they drop the cached effective config.
_CACHE_INPUTS = frozenset(
    {"defaults", "manifest", "preset", "app", "derived", "channels", "clock", "_runtime"}
)


@dataclass
class ConfigStack:
    defaults: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    preset: dict = field(default_factory=dict)
    app: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)
    clock: Callable[[], float] = time.time
    audit_path: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)
    preset_dirs: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._cached: RoboEmotConfig | None = None
        self._cached_until: float | None = None
        self._runtime: dict[str, _RuntimeValue] = {}
        self.channels: frozenset[str] | None = None
        self.body_yaw: str | None = None  # refused in speech_motion_map (set with the channels)
        self._listeners: list[Callable[[ConfigChange], None]] = []
        self._lock = threading.RLock()
        self._mtimes: dict[str, float] = {}
        self.audit: list[AuditEntry] = []
        self.last_error: str | None = None
        # Listener calls that raised; a writer compares it before and after (R3 SF3)
        self.listener_failures = 0
        self._last = self.effective()
        self._check_route_budgets(self._last)
        # The configuration installed by the owner is the initial approved baseline.
        self._approved = self._last

    def __setattr__(self, name: str, value: Any) -> None:
        """Rebinding a layer, the runtime store, the channels or the clock drops the cached
        effective configuration (in-place edits call :meth:`_invalidate` themselves)."""
        object.__setattr__(self, name, value)
        if name in _CACHE_INPUTS:
            self._invalidate()

    def _invalidate(self) -> None:
        """Drop the cached effective configuration: the next ``effective()`` rebuilds it."""
        if "_cached" in self.__dict__:
            self._cached = None
            self._cached_until = None

    # ---- construction from files

    @classmethod
    def from_files(
        cls,
        *,
        defaults: Path | None = None,
        preset: str | None = None,
        preset_dirs: Iterable[Path] = (),
        app: Path | None = None,
        manifest_behaviour: dict | None = None,
        audit_path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> ConfigStack:
        dirs = [Path(p) for p in preset_dirs]
        files = {k: Path(v) for k, v in {"defaults": defaults, "app": app}.items() if v}
        stack = cls(
            defaults=load_yaml(defaults) if defaults else {},
            manifest=manifest_behaviour or {},
            preset=resolve_preset(preset, dirs) if preset else {},
            app=load_yaml(app) if app else {},
            clock=clock,
            audit_path=audit_path,
            files=files,
            preset_dirs=dirs,
        )
        if preset:
            stack.files["preset"] = Path("preset:" + preset)
        stack._remember_mtimes()
        return stack

    @_mutator
    def set_manifest_behaviour(
        self, behaviour: dict, channels: Collection[str], *, body_yaw: str | None
    ) -> None:
        """Install the robot manifest's ``behaviour`` block (layer 2) once the manifest is loaded.

        ``channels`` are the dof names expressions may drive (``Manifest.channels``): from now on
        every configuration this stack builds is refused when a map or pose names another one.
        ``body_yaw`` (``GazeMap.body_yaw``, None when the robot cannot turn its body) is refused
        in ``speech_motion_map`` on top of that: the speaking motion never turns the body. It is
        required (keyword), so a caller cannot forget it and silently accept a body turn.
        The block becomes part of the installed configuration, so it also resets the approved
        baseline.
        """
        unknown = sorted(set(behaviour) - MANIFEST_BEHAVIOUR_KEYS)
        if unknown:
            raise ConfigError(
                f"manifest behaviour may set {sorted(MANIFEST_BEHAVIOUR_KEYS)}, not {unknown}"
            )
        with self._lock:
            check_manifest_voice(behaviour.get("voice"))
            check_manifest_motion(behaviour.get("motion"))
            saved, saved_channels, saved_body = self.manifest, self.channels, self.body_yaw
            self.manifest, self.channels = behaviour, frozenset(channels)
            self.body_yaw = body_yaw
            try:
                defaults = RoboEmotConfig()
                layer = RoboEmotConfig.model_validate(
                    deep_merge(defaults.model_dump(mode="python"), behaviour)
                )
                found = violations(defaults, layer)  # the manifest layer alone, vs the defaults
                if found:
                    raise GuardrailError(
                        "manifest behaviour would loosen: " + "; ".join(map(str, found))
                    )
                baseline = RoboEmotConfig.model_validate(self.merged_tree(include_runtime=False))
                self._check_channels(baseline)
                self._check_route_budgets(baseline)
            except (ValidationError, GuardrailError, ConfigError) as exc:
                self.manifest, self.channels = saved, saved_channels
                self.body_yaw = saved_body
                if isinstance(exc, ValidationError):
                    raise ConfigError(f"manifest behaviour: {_explain(exc)}") from exc
                raise
            self._last, self._approved = self.effective(), baseline
            self._record("manifest", "behaviour", None, sorted(behaviour))

    # ---- reading

    def merged_tree(self, person: dict | None = None, *, include_runtime: bool = True) -> dict:
        with self._lock:
            # Start from the schema defaults so keyed lists (appraisal.rules) merge with them.
            tree: dict = RoboEmotConfig().model_dump(mode="python")
            person = self._sanitize_person(person) if person else {}
            for layer in (
                self.defaults,
                self.manifest,
                self.preset,
                self.app,
                self.derived,
                person,
            ):
                tree = deep_merge(tree, layer)
            if include_runtime:
                now = self.clock()
                for key, rv in self._runtime.items():  # in write order: the latest write wins
                    if rv.expires_at is not None and rv.expires_at <= now:
                        continue  # expired; removed (with a change event) by poll()
                    value = copy.deepcopy(rv.value)
                    if key.endswith(PATCH):
                        try:
                            current = get_path(tree, slot_base(key))
                        except (KeyError, IndexError, ValueError, TypeError):
                            continue
                        if isinstance(current, list):
                            patched = _patch_keyed(current, value, KEYED_LISTS[slot_base(key)])
                            set_path(tree, slot_base(key), patched)
                        continue
                    try:
                        current = get_path(tree, key)
                    except (KeyError, IndexError, ValueError, TypeError):
                        current = None
                    if isinstance(value, dict) and isinstance(current, dict):
                        value = deep_merge(current, value)  # a section override keeps the rest
                    elif (
                        key in KEYED_LISTS and isinstance(value, list) and isinstance(current, list)
                    ):
                        value = _merge_keyed(current, value, KEYED_LISTS[key])
                    set_path(tree, key, value)
            return tree

    def effective(self, person: dict | None = None) -> RoboEmotConfig:
        """The merged, validated configuration. Without ``person`` it is cached until a layer,
        the runtime store, the channels change or the earliest runtime TTL lapses (SPEC 11A), so
        callers get the same read-only object: never mutate it. A ``person`` call is rebuilt every
        time and never cached."""
        if person:
            return self._build(person)
        with self._lock:
            until = self._cached_until
            if self._cached is not None and (until is None or self.clock() < until):
                return self._cached
            now = self.clock()
            cfg = self._build(None)
            pending = [
                rv.expires_at
                for rv in self._runtime.values()
                if rv.expires_at is not None and rv.expires_at > now
            ]
            self._cached, self._cached_until = cfg, min(pending, default=None)
            return cfg

    def _build(self, person: dict | None) -> RoboEmotConfig:
        tree = self.merged_tree(person)
        version = tree.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version {version} not supported (expected "
                f"{SCHEMA_VERSION}); run the migration"
            )
        try:
            cfg = RoboEmotConfig.model_validate(tree)
        except ValidationError as exc:
            raise ConfigError(_explain(exc)) from exc
        self._check_channels(cfg)
        return cfg

    def _check_channels(self, cfg: RoboEmotConfig) -> None:
        """Refuse a channel the loaded manifest has no dof for (SPEC 7, 7A, 11A). Unchecked until
        :meth:`set_manifest_behaviour` has told the stack which dofs the robot has."""
        if self.channels is None:
            return
        named = {f"expression_map.{k}": e.channels for k, e in cfg.expression_map.items()}
        named |= {f"gesture_map.{k}": e.channels for k, e in cfg.gesture_map.items()}
        motion = cfg.circadian_motion
        for label in ("yawn", "stretch"):
            if (entry := getattr(motion, label)) is not None:
                named[f"circadian_motion.{label}"] = entry.channels
        named["circadian_motion.rest_pose_deg"] = motion.rest_pose_deg
        for role, names in cfg.speech_motion_map.channels().items():
            named[f"speech_motion_map.{role}"] = names
            if self.body_yaw is not None and self.body_yaw in names:
                raise ConfigError(
                    f"speech_motion_map.{role}: {self.body_yaw!r} turns the body; the speaking "
                    "motion never does"
                )
        for where, names in named.items():
            for name in names:
                if name not in self.channels:
                    raise ConfigError(
                        f"{where}: unknown channel {name!r}, expected one of "
                        f"{sorted(self.channels)}"
                    )

    def get(self, key: str, person: dict | None = None) -> Any:
        return get_path(self.effective(person), key)

    def source_of(self, path: str, tree: dict | None = None) -> str:
        """The layer that decides ``path``: ``runtime`` (a live override that holds the path: on
        it, in a section above it that contains it, or below it), else the highest of
        ``derived``, ``app``, ``preset``, ``manifest``, ``defaults`` that holds it, else
        ``schema`` (the built-in default). Keyed-list entries are matched by identity
        (``resolve_in_layer``), not by position. ``tree`` is the effective tree, ``merged_tree()``
        when omitted (pass it when asking for many paths). The per-call ``person`` layer is not
        stored and never reported."""
        with self._lock:
            now = self.clock()
            effective = tree if tree is not None else self.merged_tree()
            for key, rv in self._runtime.items():
                if rv.expires_at is not None and rv.expires_at <= now:
                    continue
                base = slot_base(key)
                if path == base or base.startswith(path + "."):
                    return "runtime"
                if base in KEYED_LISTS and isinstance(rv.value, list):
                    if (
                        path.startswith(base + ".")
                        and resolve_in_layer(nest(base, rv.value), effective, path)[0]
                    ):
                        return "runtime"
                    continue
                if path.startswith(key + "."):
                    try:
                        get_path(rv.value, path[len(key) + 1 :])
                    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
                        continue
                    return "runtime"
            for name in reversed(LAYER_ORDER[:-1]):
                layer = getattr(self, name, None)
                if layer and resolve_in_layer(layer, effective, path)[0]:
                    return name
        return "schema"

    # ---- writing

    @_mutator
    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_s: float | None = None,
        source: str = "api",
        admin: bool = False,
        layer: str = "runtime",
        must_exist: bool = False,
        refuse: frozenset[str] = frozenset(),
        drop_runtime: bool = False,
        exempt: frozenset[str] = frozenset(),
    ) -> RoboEmotConfig:
        """Set a parameter or a whole section. ``must_exist`` refuses (``ConfigError``) a key the
        effective configuration does not have yet, e.g. a language that was removed meanwhile.
        ``exempt`` lists violation paths that ``refuse`` does not block (one route's ``api_key``
        finding, for an acknowledged endpoint move); other paths keep every check.
        ``refuse`` names cross-field checks (``Violation.check``) that stay refused whatever
        ``admin`` says. ``drop_runtime`` (app layer) removes the runtime override of the same key
        in the same transaction, so it cannot shadow the new value and a refusal undoes both.

        ``layer="runtime"`` (default): may only make safety more conservative; a privacy
        opt-in or an ``admin_only`` prompt change needs ``admin=True``. An entry of a merged list
        (``KEYED_LISTS``) addressed by index is stored by identity, so it stays that entry.
        ``layer="app"``: persistent edit of the app config (written back to its file);
        loosening safety or changing prompt text needs ``admin=True`` (the admin PIN, SPEC 11A).
        ``layer="derived"``: a computed value just above the app config, in memory only.
        """
        with self._lock:
            self._check_key(key, must_exist)
            old = self._safe_get(key)
            persist = None
            runtime_keys: list[str] = []
            if layer == "runtime":
                slot, undo = self._stage_runtime(key, value, ttl_s)
                runtime_keys.append(slot)
                allow = {"privacy", "admin"} if admin else set()
            elif layer == "app":
                saved_app, saved_rt = copy.deepcopy(self.app), copy.deepcopy(self._runtime)
                set_in_layer(self.app, self.merged_tree(), key, copy.deepcopy(value))
                self._invalidate()
                if drop_runtime:
                    self._take_runtime(key, False)
                persist = self._persist_app

                def undo() -> None:
                    self.app = saved_app
                    self._runtime = saved_rt

                allow = {"privacy", "safety", "admin"} if admin else set()
            elif layer == "derived":
                saved_derived = copy.deepcopy(self.derived)
                set_in_layer(self.derived, self.merged_tree(), key, copy.deepcopy(value))
                self._invalidate()

                def undo() -> None:
                    self.derived = saved_derived

                allow = set()
            else:
                raise ConfigError(f"layer {layer!r} is not writable")
            cfg = self._commit(undo, allow, source, runtime_keys, refuse, persist, exempt=exempt)
            self._record(source, key, old, value)
            self._notify(cfg)
            return cfg

    @_mutator
    def set_many(
        self,
        changes: dict[str, Any],
        *,
        source: str = "api",
        admin: bool = False,
        must_exist: bool = False,
        refuse: frozenset[str] = frozenset(),
        exempt: frozenset[str] = frozenset(),
    ) -> RoboEmotConfig:
        """Set several runtime overrides as one change: validated and guarded once on the final
        state, so a rule that spans fields does not depend on their order. All or nothing."""
        with self._lock:
            for key in changes:
                self._check_key(key, must_exist)
            before = {k: self._safe_get(k) for k in changes}
            slots: list[str] = []
            undos: list[Callable[[], None]] = []
            try:
                for key, value in changes.items():
                    slot, undo = self._stage_runtime(key, value, None)
                    slots.append(slot)
                    undos.append(undo)
            except Exception:
                for undo in reversed(undos):
                    undo()
                raise

            def undo_all() -> None:
                for undo in reversed(undos):
                    undo()

            allow = {"privacy", "admin"} if admin else set()
            cfg = self._commit(undo_all, allow, source, slots, refuse, exempt=exempt)
            for key, value in changes.items():
                self._record(source, key, before[key], value)
            self._notify(cfg)
            return cfg

    @_mutator
    def preview(self, changes: dict[str, Any]) -> RoboEmotConfig:
        """The effective configuration with ``changes`` laid over the runtime layer, nothing kept
        (no guardrail, no event). ``ConfigError`` when the result is invalid."""
        with self._lock:
            saved = copy.deepcopy(self._runtime)
            try:
                for key, value in changes.items():
                    self._stage_runtime(key, value, None)
                return self.effective()
            finally:
                self._runtime = saved

    @_mutator
    def clear(
        self,
        key: str,
        *,
        source: str = "api",
        layer: str = "runtime",
        subtree: bool = False,
        admin: bool = False,
        refuse: frozenset[str] = frozenset(),
    ) -> bool:
        """Remove an override from the runtime layer, or a value from the derived layer; True when
        something was removed. ``subtree`` also removes the overrides under ``key`` (a whole added
        entry). An override of a merged-list entry addressed by index or name is found by
        identity. Removing an override can loosen things (a locality reset), so the result goes
        through the same guardrails as ``set`` (``admin``, ``refuse``) and must validate, else
        nothing is removed. A state that was already invalid may be cleared."""
        with self._lock:
            removal = self._removal(key, layer, subtree)
            if removal is None:
                return False
            old, undo = removal
            allow = {"privacy", "admin"} if admin else set()
            try:
                cfg = self._commit(undo, allow, source, (), refuse)
            except ConfigError:
                if self._valid():  # the clear itself broke a valid state: refused, undone
                    raise
                removal = self._removal(key, layer, subtree)  # it was invalid already
                assert removal is not None
                cfg = self._commit(removal[1], allow, source, (), refuse, lenient=True)
            self._record(source, key, old, None)
            if cfg is not None:
                self._notify(cfg)
            return True

    def _removal(
        self, key: str, layer: str, subtree: bool
    ) -> tuple[Any, Callable[[], None]] | None:
        """Take ``key`` out of the runtime or derived layer: ``(old value, undo)``, or ``None``
        when there was nothing to remove."""
        if layer == "runtime":
            saved = copy.deepcopy(self._runtime)
            found, old = self._take_runtime(key, subtree)
            if not found:
                return None

            def undo_runtime() -> None:
                self._runtime = saved

            return old, undo_runtime
        if layer == "derived":
            try:
                old = get_path(self.derived, key)
            except (KeyError, AttributeError, IndexError):
                return None
            parent, _, leaf = key.rpartition(".")
            saved_derived = copy.deepcopy(self.derived)
            (get_path(self.derived, parent) if parent else self.derived).pop(leaf)
            self._invalidate()

            def undo_derived() -> None:
                self.derived = saved_derived

            return old, undo_derived
        return None

    def note(self, source: str, key: str, old: Any, new: Any) -> None:
        """Add an audit entry for something the stack did not write itself (values are scrubbed)."""
        with self._lock:
            self._record(source, key, old, new)

    @_mutator
    def apply_preset(self, name: str, *, source: str = "api", admin: bool = False) -> None:
        """Switch personality. A preset that loosens safety or privacy needs ``admin=True``."""
        with self._lock:
            saved = self.preset, self.files.get("preset")
            self.preset = resolve_preset(name, self.preset_dirs)

            def undo() -> None:
                self.preset = saved[0]

            self._commit(undo, {"privacy", "safety", "admin"} if admin else set(), source)
            self.files["preset"] = Path("preset:" + name)
            self._record(source, "personality", self._last.personality, name)
            self._notify()

    def _commit(
        self,
        undo: Callable[[], None],
        allow: set[str],
        source: str,
        runtime_keys: Iterable[str] = (),
        refuse: frozenset[str] = frozenset(),
        persist: Callable[[], None] | None = None,
        lenient: bool = False,
        exempt: frozenset[str] = frozenset(),
    ) -> RoboEmotConfig | None:
        """Validate the tentative state and check guardrails; roll back with ``undo`` on
        failure. ``lenient`` (a clear from a state that was already invalid) skips only the
        validation: the guardrails still run on what can be built (``_buildable``), and the result
        is ``None`` when nothing could.

        Two configurations are checked against the approved baseline: the layers without
        runtime overrides (``base``) and the effective one. An admin-approved loosening of a
        persistent layer (app, preset) makes ``base`` the new baseline; an admin-approved
        runtime override approves those exact values for that override only and lapses when it
        is cleared or expires, so the baseline never absorbs runtime values. A violation whose
        ``check`` is in ``refuse`` blocks whatever ``allow`` says.
        """
        keys = list(runtime_keys)
        try:
            cfg: RoboEmotConfig | None = self.effective()
            base: RoboEmotConfig | None = (
                cfg  # no runtime override: both are the same configuration
                if not self._runtime
                else RoboEmotConfig.model_validate(self.merged_tree(include_runtime=False))
            )
            self._check_route_budgets(cfg)
        except (ConfigError, ValidationError) as exc:
            if not lenient:
                undo()
                if isinstance(exc, ValidationError):
                    raise ConfigError(_explain(exc)) from exc
                raise
            cfg, base = self._buildable()
        now = self.clock()
        live_approved = [
            (path, value)
            for rv in self._runtime.values()
            if rv.expires_at is None or rv.expires_at > now
            for path, value in rv.approved.items()
        ]
        base_found = violations(self._approved, base) if base is not None else []
        base_paths = {v.path for v in base_found}
        rt_found = [
            v
            for v in (
                violations(self._approved, cfg) if cfg is not None and cfg is not base else []
            )
            if v.path not in base_paths and (v.path, v.candidate) not in live_approved
        ]
        blocked = [
            v
            for v in base_found + rt_found
            if v.kind not in allow or (v.check in refuse and v.path not in exempt)
        ]
        if blocked:
            undo()
            raise GuardrailError(
                "change refused, it would loosen: "
                + "; ".join(str(v) for v in blocked)
                + ". Safety: edit the app config with admin confirmation. Privacy: admin opt-in."
                + " Prompt text: admin confirmation.",
                blocked,
            )
        if persist is not None:
            try:  # the file is the truth: unwritable means nothing changes and nothing is logged
                persist()
            except OSError:
                undo()
                raise
        if base_found:
            self._approved = base
        if rt_found and keys:
            for v in rt_found:  # each approved value lapses with the override that covers it
                owner = next(
                    (
                        k
                        for k in keys
                        if v.path == slot_base(k) or v.path.startswith(slot_base(k) + ".")
                    ),
                    keys[0],
                )
                rv = self._runtime[owner]
                rv.approved = rv.approved | {v.path: copy.deepcopy(v.candidate)}
        if base_found or rt_found:
            self._record(source, "approved", None, [str(v) for v in base_found + rt_found])
        return cfg

    @staticmethod
    def _check_route_budgets(cfg: RoboEmotConfig) -> None:
        """Load-time check (SPEC 8, Conversation state) on a configuration without person
        layers. It runs when a file loads or a change commits, not on every ``effective()``
        call, so per-person lookups never pay for it or fail on it."""
        from ..llm.prompt import check_route_budgets  # lazy: llm imports the config package

        try:
            check_route_budgets(cfg)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    def subscribe(self, fn: Callable[[ConfigChange], None]) -> Callable[[], None]:
        """Call ``fn`` on every change; subscribing the same ``fn`` twice keeps one entry.
        Returns a function that removes it again (safe to call twice)."""
        if fn not in self._listeners:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsubscribe

    # ---- hot reload (SPEC 11A: applies within 1 s)

    def poll(self) -> ConfigChange | None:
        """Expire TTL overrides and hot-reload changed files (SPEC 11A: within 1 s).

        A reloaded file that is invalid or would loosen safety/privacy is refused: the last
        valid configuration stays in effect and the reason is kept in ``last_error``.
        """
        with self._lock:
            changed = self._expire()
            for layer, path in self.files.items():
                if str(path).startswith("preset:"):
                    continue
                mtime = _mtime(path)
                if mtime == self._mtimes.get(layer):
                    continue
                self._mtimes[layer] = mtime
                saved = getattr(self, layer)
                try:
                    setattr(self, layer, load_yaml(path))

                    def undo(layer: str = layer, saved: dict = saved) -> None:
                        setattr(self, layer, saved)

                    self._commit(undo, set(), f"reload:{layer}")
                    self.last_error = None
                    changed = True
                except (ConfigError, GuardrailError) as exc:
                    setattr(self, layer, saved)
                    self.last_error = scrub_text(f"{path}: {exc}")
                    self._record(f"reload:{layer}", "refused", None, scrub_text(str(exc)))
            return self._notify() if changed else None

    def start_watcher(self, interval_s: float = 0.5) -> threading.Event:
        stop = threading.Event()

        def run() -> None:
            while not stop.wait(interval_s):
                # keep the last valid config on a bad file; `roboemot config validate` reports it
                with contextlib.suppress(ConfigError):
                    self.poll()

        threading.Thread(target=run, name="config-watcher", daemon=True).start()
        return stop

    # ---- internals

    def _buildable(self) -> tuple[RoboEmotConfig | None, RoboEmotConfig | None]:
        """The effective and the persistent-layers configuration as far as they can be built when
        the state is invalid: the effective one without the first runtime override whose removal
        makes it valid (an override that cannot be in effect anyway), else ``None``."""
        try:
            base = RoboEmotConfig.model_validate(self.merged_tree(include_runtime=False))
        except ValidationError:
            base = None
        saved = dict(self._runtime)
        try:
            for slot in saved:
                self._runtime = {k: v for k, v in saved.items() if k != slot}
                try:
                    return self.effective(), base
                except ConfigError:
                    continue
        finally:
            self._runtime = saved
        return None, base

    def _check_key(self, key: str, must_exist: bool) -> None:
        """``ConfigError`` for a parameter that is unknown, or (``must_exist``) not there now."""
        if not field_meta(key) and not self._known_path(key):
            raise ConfigError(f"unknown parameter: {key}")
        if must_exist and not self._known_path(key):
            raise ConfigError(f"parameter does not exist: {key}")

    def _valid(self) -> bool:
        try:
            self.effective()
        except ConfigError:
            return False
        return True

    def _safe_get(self, key: str) -> Any:
        try:
            return self.get(key)
        except (KeyError, AttributeError, IndexError):
            return None

    def _stage_runtime(
        self, key: str, value: Any, ttl_s: float | None
    ) -> tuple[str, Callable[[], None]]:
        """Put ``value`` for ``key`` in the runtime layer; returns the runtime slot it landed in
        and the undo.

        An entry of a merged list (``KEYED_LISTS``), addressed by index or by name, is stored as
        a partial entry (identity fields plus the change) in the list's patch slot, or, for an entry
        added at runtime, in the added-entries slot itself (write order cannot then drop the
        edit), so one edit has one representation and a patch whose entry vanishes does nothing.
        A list of entries written to the list itself is merged into the list's own slot (added
        entries). Neither can expire (``ttl_s`` is refused)."""
        expires = self.clock() + ttl_s if ttl_s else None
        ref = keyed_ref(self.merged_tree(), key, named=True)
        if ref is not None or (key in KEYED_LISTS and isinstance(value, list)):
            if ttl_s:
                raise ConfigError(f"{key}: a list entry cannot expire")
            if ref is not None:
                list_path, ident, rest = ref
                slot, partial = list_path + PATCH, copy.deepcopy(ident)
                added = self._runtime.get(list_path)
                if (
                    added is not None
                    and keyed_index(nest(list_path, added.value), list_path, ident) is not None
                    and keyed_index(self.merged_tree(include_runtime=False), list_path, ident)
                    is None
                ):
                    slot = list_path  # an entry added at runtime: its edits join it
                if rest:
                    set_path(partial, rest, copy.deepcopy(value))
                else:
                    partial = copy.deepcopy(value)
                items = [partial]
            else:
                list_path, slot, items = key, key, copy.deepcopy(value)
            held = self._runtime.get(slot)
            merged = _merge_keyed(held.value if held else [], items, KEYED_LISTS[list_path])
            new = _RuntimeValue(merged, None, dict(held.approved) if held else {})
        else:
            slot, new = key, _RuntimeValue(copy.deepcopy(value), expires)
        before = dict(self._runtime)
        self._runtime.pop(slot, None)
        self._runtime[slot] = new  # last in order: the latest write wins
        self._invalidate()

        def undo() -> None:
            self._runtime.clear()
            self._runtime.update(before)
            self._invalidate()

        return slot, undo

    def _take_runtime(self, key: str, subtree: bool) -> tuple[bool, Any]:
        """Remove the runtime override of ``key`` (and those under it with ``subtree``).

        Returns ``(found, old)``. An entry of a merged list is looked up by identity in the patch
        slot and in the added-entries slot; emptied parents and slots are dropped. Override keys
        spelled with the path are swept as well."""
        found, old = False, None
        ref = keyed_ref(self.merged_tree(), key, named=True)
        if ref is not None:
            list_path, ident, rest = ref
            for slot in (list_path + PATCH, list_path):
                held = self._runtime.get(slot)
                at = keyed_index(nest(list_path, held.value), list_path, ident) if held else None
                if held is None or at is None:
                    continue
                entry = held.value[at]
                if rest:
                    hit, was = _pop_leaf(entry, rest)
                    if hit:
                        found, old = True, was
                    if hit and slot.endswith(PATCH) and set(entry) <= set(ident):
                        held.value.pop(at)
                else:
                    found, old = True, held.value.pop(at)
                if not held.value:
                    self._runtime.pop(slot)
        gone = [k for k in self._runtime if k == key or (subtree and k.startswith(key + "."))]
        if gone:
            old = self._runtime[key].value if key in self._runtime else old
            for k in gone:
                self._runtime.pop(k)
        self._invalidate()
        return found or bool(gone), old

    def _known_path(self, key: str) -> bool:
        try:
            get_path(self.effective(), key)
            return True
        except (KeyError, AttributeError, IndexError, ValueError):
            return False

    def _sanitize_person(self, person: dict) -> dict:
        """Per-person overrides may shape register, voice, conversation and persona traits only
        (never the persona prompt), and never a guarded (safety / privacy) value;
        anything else is dropped and audited."""
        clean: dict = {}
        dropped: list[str] = []
        cap = None
        for path, leaf in flatten(person).items():
            meta = field_meta(path)
            allowed = any(path == s or path.startswith(s + ".") for s in PERSON_SECTIONS)
            too_long = False
            if path.endswith(".persona_fragment") and isinstance(leaf, str):
                if cap is None:
                    cap = self.merged_tree()["register"]["persona_fragment_max_chars"]
                too_long = len(leaf) > cap
            if path == "register.persona_fragment_max_chars" or too_long:
                dropped.append(path)  # a person may not lengthen the system prompt
            elif not allowed or _guarded(meta):
                dropped.append(path)
            else:
                set_path(clean, path, leaf)
        if dropped:
            self.audit.append(AuditEntry(self.clock(), "person", "dropped", None, dropped))
        return clean

    def _persist_app(self) -> None:
        path = self.files.get("app")
        if path and not str(path).startswith("preset:"):
            _replace_file(Path(path), yaml.safe_dump(self.app, sort_keys=False))
            self._mtimes["app"] = _mtime(path)

    def _expire(self) -> bool:
        now = self.clock()
        expired = [k for k, v in self._runtime.items() if v.expires_at and v.expires_at <= now]
        for key in expired:
            old = self._runtime.pop(key).value
            self._invalidate()
            self.audit.append(AuditEntry(now, "ttl", key, scrub(key, _plain(old)), None))
        return bool(expired)

    def _record(self, source: str, key: str, old: Any, new: Any) -> None:
        entry = AuditEntry(
            self.clock(), source, key, scrub(key, _plain(old)), scrub(key, _plain(new))
        )
        self.audit.append(entry)
        if self.audit_path:
            fd = os.open(self.audit_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, AUDIT_FILE_MODE)
            os.fchmod(fd, AUDIT_FILE_MODE)  # the mode of O_CREAT applies to a new file only
            with os.fdopen(fd, "a") as fh:
                fh.write(json.dumps(entry.__dict__, default=str) + "\n")

    def _notify(self, new: RoboEmotConfig | None = None) -> ConfigChange | None:
        """Tell the listeners what changed; ``new`` is the effective config if the caller has it."""
        new = new if new is not None else self.effective()
        before, after = flatten(self._last.model_dump()), flatten(new.model_dump())
        keys = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        self._last = new
        if not keys:
            return None
        restart = [k for k in keys if field_meta(k).get("restart")]
        change = ConfigChange(keys, restart, new)
        for fn in list(self._listeners):
            try:
                fn(change)
            except Exception as exc:  # one failing listener must not starve the others
                # visible, not silent: the value is in effect() but a part may hold the old one
                kind = type(exc).__name__
                log.warning("config listener failed: %s", kind)
                self.listener_failures += 1
                self.last_error = f"config listener failed: {kind}"
                self._record("listener", "listener_failed", None, kind)
        return change

    def _remember_mtimes(self) -> None:
        for layer, path in self.files.items():
            if not str(path).startswith("preset:"):
                self._mtimes[layer] = _mtime(path)


def _replace_file(path: Path, text: str) -> None:
    """Write ``text`` over ``path`` atomically (SPEC 11A, 12): a temp file in the same folder,
    owner-only (``APP_FILE_MODE``: the file may hold a literal ``api_key``) whatever the old mode
    was, synced, renamed over the target, then the folder synced as a best effort. A kill at any
    point leaves the old file or the new one, never a truncated one. A symlink keeps pointing at
    its target, which is the file replaced: planting one needs write access to the data folder,
    which is already the app user's, so following it grants nothing new."""
    target = path.resolve()
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            os.fchmod(handle.fileno(), APP_FILE_MODE)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    with contextlib.suppress(OSError):  # the new file is in place: durability is best effort
        dir_fd = os.open(target.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _mtime(path: Path) -> float:
    try:
        return os.stat(path).st_mtime_ns / 1e9
    except OSError:
        return -1.0


def _moved_hint(path: str) -> str:
    """ " (renamed to X)" or " (removed: use Y)" when ``path`` is a setting that moved."""
    if path in RENAMED_SETTINGS:
        return f" (renamed to {RENAMED_SETTINGS[path]})"
    for old, instead in REMOVED_SETTINGS.items():
        if path == old or path.startswith(old + "."):
            return f" (removed: use {instead})"
    return ""


def _explain(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"{loc}: {err['msg']}{_moved_hint(loc)}")
    return "invalid configuration:\n  " + "\n  ".join(lines)
