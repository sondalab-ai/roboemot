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

import yaml
from pydantic import BaseModel, ValidationError

from .schema import SCHEMA_VERSION, RoboEmotConfig, rule_identity

LAYER_ORDER = ("defaults", "manifest", "preset", "app", "derived", "person", "runtime")
KEYED_LISTS = {"appraisal.rules": "event", "llm_routes": "name"}


class ConfigError(ValueError):
    """Invalid configuration (unknown key, out of range, bad file)."""


class GuardrailError(PermissionError):
    """A change would loosen a safety parameter or enable a privacy option without consent."""


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
    index = {_merge_id(x, key): i for i, x in enumerate(result) if isinstance(x, dict)}
    for item in over:
        k = _merge_id(item, key) if isinstance(item, dict) else None
        if k is not None and k in index:
            result[index[k]] = deep_merge(result[index[k]], item)
        else:
            result.append(copy.deepcopy(item))
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


def field_meta(dotted: str) -> dict[str, Any]:
    """Metadata (unit, restart, conservative, opt_in, admin_only) of a path, {} if unknown.

    Dict-valued fields (``voice.moods``) and list fields (``llm_routes``) consume one extra
    path element (the key or the index).
    """
    parts = dotted.split(".")
    model: Any = RoboEmotConfig
    meta: dict[str, Any] = {}
    i = 0
    while i < len(parts):
        fields = getattr(model, "model_fields", None)
        if not fields or parts[i] not in fields:
            return {}
        info = fields[parts[i]]
        extra = info.json_schema_extra
        meta = dict(extra) if isinstance(extra, dict) else {}
        annotation = _strip_optional(info.annotation)
        i += 1
        if get_origin(annotation) in (dict, list):
            i += 1  # skip the key or index
        model = _inner_model(annotation)
        if model is None:
            return meta if i >= len(parts) else {}
    return meta


def _strip_optional(annotation: Any) -> Any:
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
    {"expression_map", "circadian_motion"}
)  # what a robot manifest may set
PERSON_SECTIONS = ("register", "voice", "conversation", "persona.traits")


@dataclass(frozen=True)
class Violation:
    kind: Literal["safety", "privacy", "admin"]
    path: str
    approved: Any
    candidate: Any

    def __str__(self) -> str:
        return f"{self.path} ({self.approved!r} -> {self.candidate!r})"


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
        if _guarded(meta):
            out[path] = (value, meta)
    return out


NEW_RULE_REFUSED = "reflexes.rules."


def violations(approved: RoboEmotConfig, candidate: RoboEmotConfig) -> list[Violation]:
    """Leaves where ``candidate`` is less safe than ``approved`` (SPEC 11A, Guardrails)."""
    base = guarded_leaves(approved)
    found = []
    for path, (value, meta) in guarded_leaves(candidate).items():
        direction = meta.get("conservative")
        if path.startswith(NEW_RULE_REFUSED) and path not in base:
            # A reflex rule (new or renamed) that the approved config does not have is a new motion
            # the approval never saw: refused whole, not compared with anything (SPEC 10, 12).
            found.append(Violation("safety", path, None, value))
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
        elif meta.get("opt_in") and value and not ref:
            found.append(Violation("privacy", path, ref, value))
        elif not ok:
            found.append(Violation("safety", path, ref, value))
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
            except (ValidationError, GuardrailError) as exc:
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
                for key, rv in self._runtime.items():
                    if rv.expires_at is not None and rv.expires_at <= now:
                        continue  # expired; removed (with a change event) by poll()
                    value = copy.deepcopy(rv.value)
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
    ) -> RoboEmotConfig:
        """Set a parameter or a whole section.

        ``layer="runtime"`` (default): may only make safety more conservative; a privacy
        opt-in or an ``admin_only`` prompt change needs ``admin=True``.
        ``layer="app"``: persistent edit of the app config (written back to its file);
        loosening safety or changing prompt text needs ``admin=True`` (the admin PIN, SPEC 11A).
        ``layer="derived"``: a computed value just above the app config, in memory only.
        """
        with self._lock:
            if not field_meta(key) and not self._known_path(key):
                raise ConfigError(f"unknown parameter: {key}")
            try:
                old = self.get(key)
            except (KeyError, AttributeError, IndexError):
                old = None
            if layer == "runtime":
                prev = self._runtime.get(key)
                expires = self.clock() + ttl_s if ttl_s else None
                self._runtime[key] = _RuntimeValue(copy.deepcopy(value), expires)

                def undo() -> None:
                    if prev is None:
                        self._runtime.pop(key, None)
                    else:
                        self._runtime[key] = prev

                allow = {"privacy", "admin"} if admin else set()
            elif layer == "app":
                saved_app = copy.deepcopy(self.app)
                set_path(self.app, key, copy.deepcopy(value))

                def undo() -> None:
                    self.app = saved_app

                allow = {"privacy", "safety", "admin"} if admin else set()
            elif layer == "derived":
                saved_derived = copy.deepcopy(self.derived)
                set_path(self.derived, key, copy.deepcopy(value))

                def undo() -> None:
                    self.derived = saved_derived

                allow = set()
            else:
                raise ConfigError(f"layer {layer!r} is not writable")
            cfg = self._commit(undo, allow, source, runtime_key=key if layer == "runtime" else None)
            if layer == "app":
                self._persist_app()
            self._record(source, key, old, value)
            self._notify()
            return cfg

    def clear(self, key: str, *, source: str = "api", layer: str = "runtime") -> None:
        """Remove an override from the runtime layer, or a value from the derived layer."""
        with self._lock:
            if layer == "runtime" and key in self._runtime:
                old = self._runtime.pop(key).value
            elif layer == "derived":
                try:
                    old = get_path(self.derived, key)
                except (KeyError, AttributeError, IndexError):
                    return
                parent, _, leaf = key.rpartition(".")
                (get_path(self.derived, parent) if parent else self.derived).pop(leaf)
            else:
                return
            self._record(source, key, old, None)
            self._notify()

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
        runtime_key: str | None = None,
    ) -> RoboEmotConfig:
        """Validate the tentative state and check guardrails; roll back with ``undo`` on
        failure.

        Two configurations are checked against the approved baseline: the layers without
        runtime overrides (``base``) and the effective one. An admin-approved loosening of a
        persistent layer (app, preset) makes ``base`` the new baseline; an admin-approved
        runtime override approves those exact values for that override only and lapses when it
        is cleared or expires, so the baseline never absorbs runtime values.
        """
        try:
            cfg = self.effective()
            base = RoboEmotConfig.model_validate(self.merged_tree(include_runtime=False))
        except (ConfigError, ValidationError) as exc:
            undo()
            if isinstance(exc, ValidationError):
                raise ConfigError(_explain(exc)) from exc
            raise
        now = self.clock()
        live_approved = [
            (path, value)
            for rv in self._runtime.values()
            if rv.expires_at is None or rv.expires_at > now
            for path, value in rv.approved.items()
        ]
        base_found = violations(self._approved, base)
        base_paths = {v.path for v in base_found}
        rt_found = [
            v
            for v in violations(self._approved, cfg)
            if v.path not in base_paths and (v.path, v.candidate) not in live_approved
        ]
        blocked = [v for v in base_found + rt_found if v.kind not in allow]
        if blocked:
            undo()
            raise GuardrailError(
                "change refused, it would loosen: "
                + "; ".join(str(v) for v in blocked)
                + ". Safety: edit the app config with admin confirmation. Privacy: admin opt-in."
                + " Prompt text: admin confirmation."
            )
        if base_found:
            self._approved = base
        if rt_found and runtime_key is not None:
            rv = self._runtime[runtime_key]
            rv.approved = rv.approved | {v.path: copy.deepcopy(v.candidate) for v in rt_found}
        if base_found or rt_found:
            self._record(source, "approved", None, [str(v) for v in base_found + rt_found])
        return cfg

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
                    self.last_error = f"{path}: {exc}"
                    self._record(f"reload:{layer}", "refused", None, str(exc))
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
        for path, leaf in flatten(person).items():
            meta = field_meta(path)
            allowed = any(path == s or path.startswith(s + ".") for s in PERSON_SECTIONS)
            if not allowed or _guarded(meta):
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
            self.audit.append(AuditEntry(now, "ttl", key, old, None))
        return bool(expired)

    def _record(self, source: str, key: str, old: Any, new: Any) -> None:
        entry = AuditEntry(self.clock(), source, key, old, new)
        self.audit.append(entry)
        if self.audit_path:
            with open(self.audit_path, "a") as fh:
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
