"""The web-editable settings: the generic write path over the field catalogue (SPEC 11A, 12).

Every catalogue leaf that is not a secret can be written (``fields()`` marks it ``writable``);
a path that is not such a leaf is refused. A value is cleaned from its descriptor (kind and
limits, ``setting_clean.clean_value``) or by the per-path hook that owns it (wake word, child-safe
lines, wake phrases, blocklist: ``setting_clean.HOOKS``), then written through
``ConfigStack.set``, so pydantic validation, the guardrails, audit (source ``webui:<ip>``) and hot
reload apply unchanged. A secret field and a secret-shaped value (also inside an added entry) are
never accepted.

Layers: runtime by default. When the stack refuses a change only because it loosens a safety
value (a field's own direction; never the reflex, locality or secret rules, which stay refused
on both layers), it is retried once on the app layer, which writes the app YAML and the audit.
Entries of a merged list are addressed by identity; entries can be added, and removed when only the
runtime layer holds them (a lower layer cannot be masked by a merge, so that is refused). Each
accepted runtime edit is kept in a 0600 overrides file in the data folder and replayed as one batch
at startup (``restore``). The PIN is the caller's job (``WebApi``); this module never sees a PIN
or a secret.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from roboemot import paths
from roboemot.config.catalogue import (
    CONTAINERS,
    ENTRY_KINDS,
    FieldDescriptor,
    build_catalogue,
)
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import (
    KEYED_LISTS,
    ConfigError,
    GuardrailError,
    deep_merge,
    entry_identity,
    get_path,
    ident_fields,
    keyed_index,
    keyed_ref,
    set_path,
)
from roboemot.redact import secret_shaped

from .setting_clean import (
    WORDS_KIND,
    clean_hooked,
    clean_key,
    clean_value,
    hook_for,
    hook_schema,
)
from .validation import ValidationError

log = logging.getLogger(__name__)
SOURCE = "webui"
RESTORE_SOURCE = "webui:restore"
FILE_VERSION = 1
MAX_FILE_BYTES = 1_000_000  # bounds what a damaged or hostile file can make the app read
FILE_MODE = 0o600
GROUP_OTHER_BITS = 0o077
NOT_WRITABLE = frozenset({"schema_version"})
LANGUAGE_MAPS = frozenset({"voice.keywords", "safety.minor_lines"})  # keyed by language
REFUSED_CHECKS = frozenset({"reflex", "locality", "secret"})  # never granted, on any layer
APP_FILE = "app"


class StackPort(Protocol):
    """The part of ``roboemot.config.ConfigStack`` this module uses."""

    files: dict[str, Path]

    def effective(self) -> RoboEmotConfig: ...

    def merged_tree(self, *, include_runtime: bool = True) -> dict[str, Any]: ...

    def source_of(self, path: str, tree: dict[str, Any] | None = None) -> str: ...

    def set(self, key: str, value: Any, **kwargs: Any) -> RoboEmotConfig: ...

    def set_many(self, changes: dict[str, Any], **kwargs: Any) -> RoboEmotConfig: ...

    def preview(self, changes: dict[str, Any]) -> RoboEmotConfig: ...

    def clear(self, key: str, **kwargs: Any) -> bool: ...


class SettingNotAllowed(LookupError):
    """The path is not a writable catalogue field (or entry container)."""


def _writable(path: str) -> bool:
    return path not in NOT_WRITABLE


def plain(value: Any) -> Any:
    """``value`` as plain JSON types (models dumped, tuples as lists)."""
    return json.loads(json.dumps(value, default=lambda o: o.model_dump(mode="json")))


def loosening_only(exc: GuardrailError) -> bool:
    """Whether every refusal is a field's own safety direction: the only case that may be retried
    on the app layer. Reflex, locality and secret findings, privacy and admin ones, and an error
    without structured findings are not."""
    return bool(exc.violations) and all(
        v.kind == "safety" and v.check == "field" for v in exc.violations
    )


class OverridesFile:
    """The persisted edits, ``{"v": 1, "overrides": {path: value}}``, mode 0600 in the data folder.

    Written like the secret store (temp file in the same folder, fsync, rename, folder fsync); a
    read refuses a file that is not a regular file owned by this user or that group or others
    can access, or that is oversized or not the expected shape. Nothing here logs a value.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        """The stored edits; {} when there is no file. Raises ``ValueError`` (reason only, no
        content) when the file is unsafe or malformed."""
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise ValueError(f"cannot open ({exc.strerror})") from exc
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("not a regular file owned by this user")
            if stat.S_IMODE(info.st_mode) & GROUP_OTHER_BITS:
                raise ValueError("readable by group or others")
            if info.st_size > MAX_FILE_BYTES:
                raise ValueError("too large")
            raw = handle.read(MAX_FILE_BYTES + 1)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ValueError("not valid JSON") from exc
        found = data.get("overrides") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("v") != FILE_VERSION:
            raise ValueError("unexpected shape")
        if not isinstance(found, dict):
            raise ValueError("unexpected shape")
        return found

    def save(self, overrides: dict[str, Any]) -> None:
        """Replace the file atomically; the data folder is created private when missing."""
        folder = paths.data_dir(str(self.path.parent))
        fd, name = tempfile.mkstemp(dir=folder, prefix=self.path.name + ".")  # 0600
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), FILE_MODE)
                handle.write(
                    json.dumps(
                        {"v": FILE_VERSION, "overrides": overrides}, indent=2, sort_keys=True
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            dir_fd = os.open(folder, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise


def okey(tree: dict[str, Any], path: str) -> str:
    """The key an edit has in the overrides file: ``path``, except that an index into a merged
    list becomes ``list[{identity}].rest``, so a replay finds the same entry wherever it sits."""
    ref = keyed_ref(tree, path, named=True)
    if ref is None:
        return path
    return entry_key(*ref)


def entry_key(list_path: str, ident: dict[str, Any], rest: str = "") -> str:
    """``list[{identity}]`` plus ``.rest`` when given: the file key of a merged-list entry."""
    text = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return f"{list_path}[{text}]" + (f".{rest}" if rest else "")


def parse_okey(key: str) -> tuple[str, dict[str, Any], str] | None:
    """Inverse of ``okey`` for a merged-list key: ``(list path, identity, rest)``, else ``None``."""
    for list_path in KEYED_LISTS:
        if not key.startswith(list_path + "["):
            continue
        try:
            ident, end = json.JSONDecoder().raw_decode(key, len(list_path) + 1)
        except ValueError:
            return None
        if not isinstance(ident, dict) or key[end : end + 1] != "]":
            return None
        rest = key[end + 1 :]
        if rest and not rest.startswith("."):
            return None
        return list_path, ident, rest[1:]
    return None


class SettingsEditor:
    """Lists the writable settings and applies validated changes: one value, an entry added or
    removed, or a runtime value reset."""

    def __init__(
        self,
        stack: StackPort,
        *,
        store: Path | None = None,
    ):
        self.stack = stack
        self._file = OverridesFile(store) if store else None
        self._overrides: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ reading

    def fields(self) -> list[FieldDescriptor]:
        """The full field catalogue, ``writable`` true for every non-secret leaf. Private values
        (minors, privacy, location) are hidden."""
        return build_catalogue(self.stack, _writable)

    def reveal(self) -> list[FieldDescriptor]:
        """Only the private fields, with their real values (secrets stay redacted). The caller
        has verified the PIN."""
        every = build_catalogue(self.stack, _writable, reveal_private=True)
        return [f for f in every if f["private"]]

    def describe(self, fields: list[FieldDescriptor] | None = None) -> list[dict[str, Any]]:
        """The writable fields the current page can render (the hooked ones: wake word, phrases,
        child-safe lines, blocklist) in its shape: catalogue value and default, JSON-Schema-style
        limits and admin flag. Goes away with the new page. Pass the ``fields()`` result to avoid
        building the catalogue twice."""
        cfg = self.stack.effective()
        rows = []
        for field in sorted(
            fields if fields is not None else self.fields(), key=lambda f: f["path"]
        ):
            hook = hook_for(field["path"])
            if not field["writable"] or hook is None:
                continue
            rows.append(
                {
                    "path": field["path"],
                    "value": field["value"],
                    "default": field["default"],
                    "schema": hook_schema(hook, field["path"], cfg),
                    "description": field["description"],
                    "needs_admin": field["admin_only"],
                    "private": field["private"],
                }
            )
        return rows

    # ------------------------------------------------------------------ writing

    def change(self, path: object, value: object, source: str = SOURCE) -> dict[str, Any]:
        """Validate ``value`` for ``path``, write it and keep it in the overrides file; the
        stack's guardrails have the last word.

        Raises ``SettingNotAllowed``, ``ValidationError`` (code, no value echoed) and the
        stack's ``GuardrailError`` / ``ConfigError``. An unchanged value writes nothing. The result
        names the ``layer`` written (``runtime``, or ``app`` for a safety loosening), whether
        the field ``restart_required`` and ``persisted`` (False when the file could not be
        written: the edit is live until restart).
        """
        with self._lock:
            result = self._apply(path, value, source, self._rows(), self.stack.effective())
            key = result.pop("okey", None)
            if result["changed"] and result["layer"] == "runtime":
                self._remember(key, result["value"])
                result["persisted"] = self._persist()
            elif result["changed"] and self._overrides.pop(key, None) is not None:
                result["persisted"] = self._persist()  # the app value replaced a runtime edit
            return result

    def add_entry(
        self, path: object, key: object, value: object, source: str = SOURCE
    ) -> dict[str, Any]:
        """Add an entry to the collection at ``path`` (a dict of models by ``key``, a keyed list
        by the identity inside ``value``) on the runtime layer. A language is never added.

        Raises like ``change``; ``entry_exists`` when the key or identity is taken,
        ``entry_invalid`` for a field the entry type does not have or a secret field.
        """
        with self._lock:
            rows, cfg = self._rows(), self.stack.effective()
            box = rows.get(path) if isinstance(path, str) else None
            if (
                box is None
                or box["kind"] not in ENTRY_KINDS
                or box["secret"]
                or box["path"] in LANGUAGE_MAPS
            ):
                raise SettingNotAllowed("path")
            if secret_shaped([key, value]):
                raise ValidationError("value_type", "value")
            entry = self._clean_entry(box, value, cfg)
            tree = self.stack.merged_tree()
            if box["kind"] == "dict_of_model":
                name = clean_key(key, cfg)
                if name in get_path(tree, box["path"]):
                    raise ValidationError("entry_exists", "key")
                target, stored, written = f"{box['path']}.{name}", f"{box['path']}.{name}", entry
            else:
                target, stored, written = self._list_add(box, entry, tree, cfg)
            self.stack.set(
                target, written, source=source, admin=True, layer="runtime", refuse=REFUSED_CHECKS
            )
            self._overrides[stored] = entry
            return {
                "path": box["path"],
                "changed": True,
                "layer": "runtime",
                "restart_required": self._restart(box, entry_rows=box["entry"]),
                "persisted": self._persist(),
            }

    def remove_entry(self, path: object, source: str = SOURCE) -> dict[str, Any]:
        """Remove the entry at ``path`` (``container.key`` or a list entry) if only the runtime
        layer holds it. A lower layer's entry cannot be masked (the merge only adds and overrides),
        so that raises ``entry_in_lower_layer``; an unknown entry raises ``SettingNotAllowed``."""
        with self._lock:
            rows, tree = self._rows(), self.stack.merged_tree()
            box_path, _, name = path.rpartition(".") if isinstance(path, str) else ("", "", "")
            box = rows.get(box_path)
            if box is None or box["kind"] not in ENTRY_KINDS or name not in box["value"]:
                raise SettingNotAllowed("path")
            base = self.stack.merged_tree(include_runtime=False)
            if self._in_tree(base, box, path, tree):
                raise ValidationError("entry_in_lower_layer", "path")
            key = okey(tree, path)
            self.stack.clear(path, source=source, subtree=True, admin=True, refuse=REFUSED_CHECKS)
            for stored in [k for k in self._overrides if k == key or k.startswith(key + ".")]:
                del self._overrides[stored]
            return {
                "path": path,
                "changed": True,
                "layer": "runtime",
                "restart_required": self._restart(box, entry_rows=box["entry"]),
                "persisted": self._persist(),
            }

    def reset(self, path: object, source: str = SOURCE) -> dict[str, Any]:
        """Drop the runtime value of the leaf ``path`` so the lower layer shows again, and its
        entry in the overrides file. A value of the app layer is not touched. ``changed`` is False
        when there was no runtime value; a field inside an entry added at runtime cannot be reset
        (``not_resettable``)."""
        with self._lock:
            rows = self._rows()
            row = rows.get(path) if isinstance(path, str) else None
            if row is None or row["kind"] in CONTAINERS or not row["writable"]:
                raise SettingNotAllowed("path")
            key = okey(self.stack.merged_tree(), path)
            hook = hook_for(path)
            if (
                self._added_parent(key) is not None
                or row["conservative"] == "superset"
                or (hook is not None and hook.kind == WORDS_KIND)
            ):  # the blocklist only grows from the web; an added entry's field has no lower value
                raise ValidationError("not_resettable", "path")
            changed = self.stack.clear(path, source=source, admin=True, refuse=REFUSED_CHECKS)
            stale = self._overrides.pop(key, None) is not None
            return {
                "path": path,
                "changed": changed,
                "value": plain(get_path(self.stack.effective(), path)),
                "layer": "runtime",
                "restart_required": self._restart(row),
                "persisted": self._persist() if changed or stale else True,
            }

    def restore(self) -> None:
        """Replay the overrides file at startup as one batch: every entry is cleaned against the
        configuration the whole file would give and the stack validates the final state once, so a
        rule that spans fields does not depend on their order.

        An unsafe or malformed file is ignored. An entry that is not writable, no longer valid,
        refused by a guardrail or for a language that is gone is dropped; warnings name the path
        only. If the batch is refused as a whole, the entries are tried one by one. The file is
        rewritten without dropped entries.
        """
        if self._file is None:
            return
        try:
            stored = self._file.load()
        except ValueError as exc:
            log.warning("web UI settings file ignored: %s", exc)
            return
        with self._lock:
            tree, rows = self.stack.merged_tree(), self._rows()
            planned: dict[str, tuple[str, Any, bool]] = {}  # file key -> (stack path, value, add)
            for key, value in stored.items():
                found = self._plan(key, value, tree, rows)
                if found is None:
                    log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
                    continue
                planned[key] = found
            try:
                cfg = self.stack.preview(combine((p, v) for p, v, _ in planned.values()))
            except (ConfigError, ValueError):
                cfg = self.stack.effective()
            cleaned: dict[str, tuple[str, Any]] = {}  # file key -> (stack path, stack value)
            for key, (path, value, add) in planned.items():
                try:
                    cleaned[key] = self._clean_planned(key, path, value, add, rows, cfg, tree)
                except (ValidationError, SettingNotAllowed, GuardrailError):
                    log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
            self._overrides = self._commit_batch(cleaned, stored)
            if set(self._overrides) != set(stored):
                self._persist()

    # ------------------------------------------------------------------ write internals

    def _rows(self) -> dict[str, FieldDescriptor]:
        every = build_catalogue(self.stack, _writable, reveal_private=True)
        return {row["path"]: row for row in every}

    def _apply(
        self,
        path: object,
        value: object,
        source: str,
        rows: dict[str, FieldDescriptor],
        cfg: RoboEmotConfig,
    ) -> dict[str, Any]:
        row = rows.get(path) if isinstance(path, str) else None
        if row is None or not isinstance(path, str) or not self._leaf_writable(rows, row):
            raise SettingNotAllowed("path")
        if secret_shaped(value):
            raise ValidationError("value_type", "value")
        hook = hook_for(path)
        clean = clean_hooked(hook, path, value, cfg) if hook else clean_value(row, value, cfg)
        old = plain(get_path(cfg, path))
        if clean == old:
            return {"path": path, "changed": False, "value": old}
        # a runtime dict merges, so a value that would drop a key cannot be expressed there
        if isinstance(clean, dict) and deep_merge(old or {}, clean) != clean:
            raise ValidationError("value_merged", "value")
        tree = self.stack.merged_tree()
        stored = okey(tree, path)
        layer = self._write(path, clean, source, retry_app=self._added_parent(stored) is None)
        return {
            "path": path,
            "changed": True,
            "value": clean,
            "layer": layer,
            "restart_required": self._restart(row),
            "okey": stored,
        }

    @staticmethod
    def _leaf_writable(rows: dict[str, FieldDescriptor], row: FieldDescriptor) -> bool:
        """A non-secret leaf that is not the identity of a keyed-list entry (renaming one would
        move it to another entry)."""
        if row["kind"] in CONTAINERS or not row["writable"] or row["secret"]:
            return False
        entry, _, name = row["path"].rpartition(".")
        box = rows.get(entry.rpartition(".")[0])
        return not (
            box is not None
            and box["kind"] == "keyed_list"
            and name in (box["key"], "when" if box["key"] == "event" else box["key"])
        )

    def _write(self, key: str, value: Any, source: str, *, retry_app: bool = True) -> str:
        """Write on the runtime layer; if the stack refuses it only for a safety loosening, write
        it on the app layer instead (app YAML and audit) and drop the runtime value that would
        shadow it. Returns the layer. Every other refusal propagates, and so does that one for a
        field of an entry added at runtime (``retry_app`` False): the app file has no such entry.
        """
        try:
            self.stack.set(
                key, value, source=source, admin=True, must_exist=True, refuse=REFUSED_CHECKS
            )
            return "runtime"
        except GuardrailError as exc:
            if not retry_app or not loosening_only(exc):
                raise
        self.stack.set(
            key,
            value,
            source=source,
            admin=True,
            layer="app",
            must_exist=True,
            refuse=REFUSED_CHECKS,
        )
        self.stack.clear(key, source=source, admin=True, refuse=REFUSED_CHECKS)
        return "app"

    def _remember(self, key: str, value: Any) -> None:
        """Keep a runtime edit for the next start; inside an entry added at runtime it joins the
        entry's own value."""
        parent = self._added_parent(key)
        if parent is None:
            self._overrides[key] = value
            return
        entry = json.loads(json.dumps(self._overrides[parent]))
        set_path(entry, key[len(parent) + 1 :], value)
        self._overrides[parent] = entry

    def _added_parent(self, key: str) -> str | None:
        """The file key of an entry added at runtime that holds ``key``."""
        return next(
            (
                k
                for k, v in self._overrides.items()
                if key.startswith(k + ".") and isinstance(v, dict)
            ),
            None,
        )

    def _persist(self) -> bool:
        if self._file is None:
            return True
        try:
            self._file.save(self._overrides)
        except OSError as exc:
            log.warning("web UI settings not saved (%s)", type(exc).__name__)
            return False
        return True

    @staticmethod
    def _restart(row: FieldDescriptor, entry_rows: list[FieldDescriptor] | None = None) -> bool:
        """Whether the change needs a restart: the field's ``restart`` flag, for a collection any
        field of its entries."""
        return row["restart"] or any(r["restart"] for r in entry_rows or [])

    # ------------------------------------------------------------------ entries

    def _clean_entry(
        self, box: FieldDescriptor, value: object, cfg: RoboEmotConfig
    ) -> dict[str, Any]:
        """An entry for ``box`` cleaned field by field from the blank entry's descriptors. A field
        the entry type does not have, or a secret field with a value, is ``entry_invalid``."""
        if not isinstance(value, dict):
            raise ValidationError("value_type", "value")
        rows = {r["path"]: r for r in box["entry"] or []}
        out: dict[str, Any] = {}

        def walk(node: dict[str, Any], prefix: str, into: dict[str, Any]) -> None:
            for name, item in node.items():
                path = f"{prefix}{name}"
                row = rows.get(path)
                if row is not None and row["kind"] not in CONTAINERS:
                    if row["secret"]:
                        if item not in (None, "", row["default"]):
                            raise ValidationError("entry_invalid", "value")
                        continue
                    into[name] = clean_value(row, item, cfg)
                elif isinstance(item, dict) and any(r.startswith(path + ".") for r in rows):
                    into[name] = {}
                    walk(item, path + ".", into[name])
                else:
                    raise ValidationError("entry_invalid", "value")

        walk(value, "", out)
        return out

    def _list_add(
        self, box: FieldDescriptor, entry: dict[str, Any], tree: dict[str, Any], cfg: RoboEmotConfig
    ) -> tuple[str, str, Any]:
        """``(stack path, file key, stack value)`` for a new keyed-list entry. A merged list takes
        just the entry; any other list is replaced by the old entries plus the new one."""
        id_field = box["key"]
        ident = entry.get(id_field)
        clean_key(ident, cfg, dots=id_field == "event")
        path = box["path"]
        current = get_path(tree, path)
        if any(entry_identity(path, e) == entry_identity(path, entry) for e in current):
            raise ValidationError("entry_exists", "value")
        if path in KEYED_LISTS:
            return path, entry_key(path, ident_fields(path, entry)), [entry]
        return path, path, [*current, entry]

    def _in_tree(
        self, base: dict[str, Any], box: FieldDescriptor, path: str, tree: dict[str, Any]
    ) -> bool:
        """Whether the entry at ``path`` of the effective tree exists in ``base`` (the layers
        without runtime): by key for a dict, by identity for a list."""
        if box["kind"] == "dict_of_model":
            return path.rpartition(".")[2] in get_path(base, box["path"])
        entry = get_path(tree, path)
        return any(
            entry_identity(box["path"], e) == entry_identity(box["path"], entry)
            for e in get_path(base, box["path"])
        )

    # ------------------------------------------------------------------ restore internals

    def _plan(
        self, key: str, value: Any, tree: dict[str, Any], rows: dict[str, FieldDescriptor]
    ) -> tuple[str, Any, bool] | None:
        """Where a stored edit lands now: ``(stack path, stack value, is an added entry)``, or
        ``None`` when its field, entry or language is gone."""
        if secret_shaped(value):
            return None
        parsed = parse_okey(key)
        base = self.stack.merged_tree(include_runtime=False)
        if parsed is not None:
            list_path, ident, rest = parsed
            if not rest:  # an added entry: its identity is its own, and no lower layer has it
                if not isinstance(value, dict) or value.get(KEYED_LISTS[list_path]) is None:
                    return None
                if (
                    ident != ident_fields(list_path, value)
                    or keyed_index(base, list_path, ident) is not None
                ):
                    return None
                return (list_path, [value], True)
            at = keyed_index(tree, list_path, ident)
            if at is None:
                return None
            name = ident.get(KEYED_LISTS[list_path])
            for segment in (name, str(at)):  # the catalogue names an entry by its name when it can
                path = f"{list_path}.{segment}.{rest}"
                if isinstance(segment, str) and path in rows:
                    return path, value, False
            return None
        parent = rows.get(key.rpartition(".")[0])
        if parent is not None and parent["kind"] == "dict_of_model" and isinstance(value, dict):
            absent = key.rpartition(".")[2] not in get_path(base, parent["path"])
            return (key, value, True) if absent else None
        return (key, value, False) if key in rows else None

    def _clean_planned(
        self,
        key: str,
        path: str,
        value: Any,
        add: bool,
        rows: dict[str, FieldDescriptor],
        cfg: RoboEmotConfig,
        tree: dict[str, Any],
    ) -> tuple[str, Any]:
        """Clean one planned edit with the live-edit rules, against ``cfg`` (the whole file
        applied). Returns the stack path and value."""
        if add:
            parsed = parse_okey(key)
            box = rows.get(parsed[0] if parsed else key.rpartition(".")[0])
            if box is None or box["kind"] not in ENTRY_KINDS or box["path"] in LANGUAGE_MAPS:
                raise SettingNotAllowed("path")
            entry = self._clean_entry(box, value[0] if parsed else value, cfg)
            return path, ([entry] if parsed else entry)
        row = rows.get(path)
        if row is None or not self._leaf_writable(rows, row):
            raise SettingNotAllowed("path")
        hook = hook_for(path)
        return path, clean_hooked(hook, path, value, cfg) if hook else clean_value(row, value, cfg)

    def _commit_batch(
        self, cleaned: dict[str, tuple[str, Any]], stored: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply the cleaned edits as one change; if the stack refuses the whole, one by one.
        Returns the file entries that were applied."""
        if not cleaned:
            return {}

        def file_value(key: str) -> Any:
            parsed = parse_okey(key)
            value = cleaned[key][1]
            return value[0] if parsed and not parsed[2] else value

        try:
            self.stack.set_many(
                combine(cleaned.values()),
                source=RESTORE_SOURCE,
                admin=True,
                refuse=REFUSED_CHECKS,
            )
            return {k: file_value(k) for k in cleaned}
        except (GuardrailError, ConfigError):
            pass
        kept: dict[str, Any] = {}
        for key, (path, value) in cleaned.items():
            try:
                self.stack.set_many(
                    {path: value}, source=RESTORE_SOURCE, admin=True, refuse=REFUSED_CHECKS
                )
            except (GuardrailError, ConfigError):
                log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
                continue
            kept[key] = file_value(key)
        return kept


def combine(edits: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """``{stack path: value}`` of a batch; entries added to one merged list share its path, so
    their lists are joined rather than the last one winning."""
    out: dict[str, Any] = {}
    for path, value in edits:
        if path in KEYED_LISTS and isinstance(value, list):
            out.setdefault(path, []).extend(value)
        else:
            out[path] = value
    return out


def _shown(key: str) -> str:
    """A file key for a log line: the path only (a merged-list entry shows its list)."""
    return key.split("[", 1)[0]
