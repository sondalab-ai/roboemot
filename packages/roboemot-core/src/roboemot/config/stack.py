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
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
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
    rule_identity,
)

LAYER_ORDER = ("defaults", "manifest", "preset", "app", "derived", "person", "runtime")
KEYED_LISTS = {"appraisal.rules": "event", "llm_routes": "name"}
AUDIT_FILE_MODE = 0o600
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
    return data


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
    {"expression_map", "gesture_map", "circadian_motion"}
)  # what a robot manifest may set
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


def _filter_keys(entries: Any) -> set[tuple[str, bool]]:
    """Entries of an output-filter list as (normalised text, is prefix), the forms the matcher
    compares, so an accent, case or invisible-character swap is a different entry."""
    from ..conversation.output_filter import PREFIX, clean  # lazy: conversation imports config
    from ..voice.keywords import normalise

    keys = set()
    for entry in entries if isinstance(entries, list) else []:
        text = clean(str(entry)).strip()
        keys.add((normalise(text.rstrip(PREFIX + " ")), text.endswith(PREFIX)))
    return keys


def keeps_every_entry(candidate: Any, approved: Any) -> bool:
    """``superset`` direction: the candidate list has every approved entry (normalised)."""
    return _filter_keys(approved) <= _filter_keys(candidate)


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
            "superset": keeps_every_entry,
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
        if meta.get("conservative") == "superset" and path not in cand:
            found.append(Violation("safety", path, ref, None))  # a whole list removed
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


def _endpoint_key(url: str) -> tuple[Any, ...]:
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
            and _endpoint_key(ref.base_url) == _endpoint_key(r.base_url)
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
        self._runtime: dict[str, _RuntimeValue] = {}
        self._listeners: list[Callable[[ConfigChange], None]] = []
        self._lock = threading.RLock()
        self._mtimes: dict[str, float] = {}
        self.audit: list[AuditEntry] = []
        self.last_error: str | None = None
        self._last = self.effective()
        self._check_route_budgets(self._last)
        # The configuration installed by the owner is the initial approved baseline.
        self._approved = self._last

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

    def set_manifest_behaviour(self, behaviour: dict) -> None:
        """Install the robot manifest's ``behaviour`` block (layer 2) once the manifest is loaded.

        It becomes part of the installed configuration, so it also resets the approved baseline.
        """
        unknown = sorted(set(behaviour) - MANIFEST_BEHAVIOUR_KEYS)
        if unknown:
            raise ConfigError(
                f"manifest behaviour may set {sorted(MANIFEST_BEHAVIOUR_KEYS)}, not {unknown}"
            )
        with self._lock:
            saved = self.manifest
            self.manifest = behaviour
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
                self._check_route_budgets(baseline)
            except (ValidationError, GuardrailError, ConfigError) as exc:
                self.manifest = saved
                if isinstance(exc, ValidationError):
                    raise ConfigError(_explain(exc)) from exc
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
        tree = self.merged_tree(person)
        version = tree.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ConfigError(
                f"schema_version {version} not supported (expected "
                f"{SCHEMA_VERSION}); run the migration"
            )
        try:
            return RoboEmotConfig.model_validate(tree)
        except ValidationError as exc:
            raise ConfigError(_explain(exc)) from exc

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
    ) -> RoboEmotConfig:
        """Set a parameter or a whole section. ``must_exist`` refuses (``ConfigError``) a key the
        effective configuration does not have yet, e.g. a language that was removed meanwhile.
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

                def undo() -> None:
                    self.derived = saved_derived

                allow = set()
            else:
                raise ConfigError(f"layer {layer!r} is not writable")
            cfg = self._commit(undo, allow, source, runtime_keys, refuse, persist)
            self._record(source, key, old, value)
            self._notify()
            return cfg

    def set_many(
        self,
        changes: dict[str, Any],
        *,
        source: str = "api",
        admin: bool = False,
        must_exist: bool = False,
        refuse: frozenset[str] = frozenset(),
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
            cfg = self._commit(undo_all, allow, source, slots, refuse)
            for key, value in changes.items():
                self._record(source, key, before[key], value)
            self._notify()
            return cfg

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
            valid = self._valid()
            if layer == "runtime":
                saved = copy.deepcopy(self._runtime)
                found, old = self._take_runtime(key, subtree)
                if not found:
                    return False

                def undo() -> None:
                    self._runtime = saved

            elif layer == "derived":
                try:
                    old = get_path(self.derived, key)
                except (KeyError, AttributeError, IndexError):
                    return False
                parent, _, leaf = key.rpartition(".")
                saved_derived = copy.deepcopy(self.derived)
                (get_path(self.derived, parent) if parent else self.derived).pop(leaf)

                def undo() -> None:
                    self.derived = saved_derived

            else:
                return False
            allow = {"privacy", "admin"} if admin else set()
            self._commit(undo, allow, source, (), refuse, lenient=not valid)
            self._record(source, key, old, None)
            if self._valid():
                self._notify()
            return True

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
            base: RoboEmotConfig | None = RoboEmotConfig.model_validate(
                self.merged_tree(include_runtime=False)
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
            for v in (violations(self._approved, cfg) if cfg is not None else [])
            if v.path not in base_paths and (v.path, v.candidate) not in live_approved
        ]
        blocked = [v for v in base_found + rt_found if v.kind not in allow or v.check in refuse]
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

    def subscribe(self, fn: Callable[[ConfigChange], None]) -> None:
        self._listeners.append(fn)

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
        edit), so one edit has one representation and a patch whose entry vanishes does nothing. A list of entries
        written to the list itself is merged into the list's own slot (added entries). Neither
        can expire (``ttl_s`` is refused)."""
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

        def undo() -> None:
            self._runtime.clear()
            self._runtime.update(before)

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
        return found or bool(gone), old

    def _known_path(self, key: str) -> bool:
        try:
            get_path(self.effective(), key)
            return True
        except (KeyError, AttributeError, IndexError, ValueError):
            return False

    def _sanitize_person(self, person: dict) -> dict:
        """Per-person overrides may shape register, voice, conversation and persona traits only
        (never the persona prompt), and never a guarded (safety / minors / privacy) value;
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
            path.write_text(yaml.safe_dump(self.app, sort_keys=False))
            self._mtimes["app"] = _mtime(path)

    def _expire(self) -> bool:
        now = self.clock()
        expired = [k for k, v in self._runtime.items() if v.expires_at and v.expires_at <= now]
        for key in expired:
            old = self._runtime.pop(key).value
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

    def _notify(self) -> ConfigChange | None:
        new = self.effective()
        before, after = flatten(self._last.model_dump()), flatten(new.model_dump())
        keys = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        self._last = new
        if not keys:
            return None
        restart = [k for k in keys if field_meta(k).get("restart")]
        change = ConfigChange(keys, restart, new)
        for fn in list(self._listeners):
            fn(change)
        return change

    def _remember_mtimes(self) -> None:
        for layer, path in self.files.items():
            if not str(path).startswith("preset:"):
                self._mtimes[layer] = _mtime(path)


def _mtime(path: Path) -> float:
    try:
        return os.stat(path).st_mtime_ns / 1e9
    except OSError:
        return -1.0


def _explain(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "invalid configuration:\n  " + "\n  ".join(lines)
