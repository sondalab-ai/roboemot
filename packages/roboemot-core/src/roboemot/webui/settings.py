"""The web-editable settings: an allowlist, validation and the write path (SPEC 11A, 12).

``SETTINGS`` is the one place that says what the web UI may change (the generic write path
replaces it later). A path that no entry expands to is refused. What the UI may *see* is the
whole field catalogue (``roboemot.config.catalogue``): ``fields()`` returns it, with ``writable``
set from ``SETTINGS``, and secrets are never writable. The per-kind cleaners below are the
per-path validator hooks of the write path. Values are cleaned here (length, count and
character limits come from ``webui.settings_*`` and ``voice.name_max_chars``), then written
through ``ConfigStack.set`` on the runtime layer with no expiry, so the stack's guardrails,
audit (source ``webui:<ip>``) and hot reload apply unchanged. Each accepted edit is also kept
in a 0600 overrides file in the data folder and replayed through the same checks at startup
(``restore``); the app YAML is never touched. The PIN confirmation is the caller's job
(``WebApi``); this module never sees a PIN or a secret.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import stat
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from roboemot import paths
from roboemot.config.catalogue import FieldDescriptor, build_catalogue
from roboemot.config.schema import MinorLines, RoboEmotConfig
from roboemot.config.stack import (
    ConfigError,
    GuardrailError,
    field_meta,
    get_path,
)
from roboemot.conversation.output_filter import OutputFilter
from roboemot.redact import secret_shaped
from roboemot.voice.keywords import normalise
from roboemot.voice.robot_name import sanitise_name

from .validation import ValidationError

log = logging.getLogger(__name__)
SOURCE = "webui"
RESTORE_SOURCE = "webui:restore"
FILE_VERSION = 1
MAX_FILE_BYTES = 1_000_000  # bounds what a damaged or hostile file can make the app read
NAME_PLACEHOLDER = "{name}"
PHRASE_PATTERN = re.compile(r"[^\W_]+(?: [^\W_]+)*")
WORD_PATTERN = re.compile(r"[^\W_]+(?: [^\W_]+)*\*?")
LINE_BREAKS = "\u2028\u2029"  # categories Zl and Zp: line breaks the C* check does not cover
FILE_MODE = 0o600
GROUP_OTHER_BITS = 0o077
LIST_KIND, NAME_KIND, LINE_KIND, WORDS_KIND = "phrases", "name", "line", "words"


class StackPort(Protocol):
    """The part of ``roboemot.config.ConfigStack`` this module uses."""

    def effective(self) -> RoboEmotConfig: ...

    def merged_tree(self, *, include_runtime: bool = True) -> dict[str, Any]: ...

    def source_of(self, path: str, tree: dict[str, Any] | None = None) -> str: ...

    def set(
        self, key: str, value: Any, *, source: str, admin: bool, must_exist: bool
    ) -> RoboEmotConfig: ...


class SettingNotAllowed(LookupError):
    """The path is not on the allowlist."""


@dataclass(frozen=True)
class Setting:
    """An allowlist entry. ``template`` may hold ``{lang}``, expanded over the keys of the
    dict at ``container`` in the effective config (a language is never created from the web)."""

    template: str
    kind: str
    container: str | None = None
    allow_empty: bool = False


SETTINGS: tuple[Setting, ...] = (
    Setting("voice.keywords.{lang}.wake_phrases", LIST_KIND, "voice.keywords"),
    Setting("voice.wake_word", NAME_KIND),
    *(
        Setting(
            f"safety.minor_lines.{{lang}}.{name}",
            LINE_KIND,
            "safety.minor_lines",
            allow_empty=name == "robot_unnamed",
        )
        for name in MinorLines.model_fields
    ),
    Setting("safety.minor_output_blocklist.{lang}", WORDS_KIND, "safety.minor_output_blocklist"),
)


def expand(cfg: RoboEmotConfig) -> dict[str, Setting]:
    """Every editable path of ``cfg`` mapped to its allowlist entry."""
    out: dict[str, Setting] = {}
    for entry in SETTINGS:
        if entry.container is None:
            out[entry.template] = entry
            continue
        for lang in get_path(cfg, entry.container):
            out[entry.template.format(lang=lang)] = entry
    return out


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


def _text(value: object, field: str = "value") -> str:
    if not isinstance(value, str):
        raise ValidationError("value_type", field)
    return unicodedata.normalize("NFC", value).strip()


def _blocked_by_filter(cfg: RoboEmotConfig, text: str) -> bool:
    """Whether the minors output filter (the effective blocklist) blocks ``text``."""
    return OutputFilter(cfg.safety.minor_output_blocklist).blocked(text)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValidationError("value_type", "value")
    if any(unicodedata.category(c).startswith("C") for x in value for c in x):
        raise ValidationError("phrase_invalid", "value")
    return [" ".join(unicodedata.normalize("NFC", x).split()) for x in value]


class SettingsEditor:
    """Lists the allowlisted settings and applies one validated change."""

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

    def fields(self) -> list[FieldDescriptor]:
        """The full field catalogue, ``writable`` true for the allowlisted, non-secret paths.
        Private values (minors, privacy, location) are hidden."""
        allowed = expand(self.stack.effective())
        return build_catalogue(self.stack, lambda path: path in allowed)

    def reveal(self) -> list[FieldDescriptor]:
        """Only the private fields, with their real values (secrets stay redacted). The caller
        has verified the PIN."""
        allowed = expand(self.stack.effective())
        every = build_catalogue(self.stack, lambda path: path in allowed, reveal_private=True)
        return [f for f in every if f["private"]]

    def describe(self, fields: list[FieldDescriptor] | None = None) -> list[dict[str, Any]]:
        """The writable fields in the shape the current page renders: catalogue value and
        default, JSON-Schema-style limits and admin flag. Goes away with the new page. Pass the
        ``fields()`` result to avoid building the catalogue twice."""
        cfg = self.stack.effective()
        entries = expand(cfg)
        rows = []
        for field in sorted(
            fields if fields is not None else self.fields(), key=lambda f: f["path"]
        ):
            if not field["writable"]:
                continue
            rows.append(
                {
                    "path": field["path"],
                    "value": field["value"],
                    "default": field["default"],
                    "schema": self._schema(entries[field["path"]], cfg),
                    "description": field["description"],
                    "needs_admin": field["admin_only"],
                    "private": field["private"],
                }
            )
        return rows

    def change(self, path: object, value: object, source: str = SOURCE) -> dict[str, Any]:
        """Validate ``value`` for ``path``, write it and keep it in the overrides file; the
        stack's guardrails have the last word.

        Raises ``SettingNotAllowed``, ``ValidationError`` (code, no value echoed) and the
        stack's ``GuardrailError`` / ``ConfigError``. An unchanged value writes nothing.
        ``persisted`` is False when the file could not be written (the edit is live until
        restart).
        """
        with self._lock:
            result = self._apply(path, value, source)
            if result["changed"]:
                self._overrides[result["path"]] = result["value"]
                result["persisted"] = self._persist()
            return result

    def restore(self) -> None:
        """Replay the overrides file through the same checks as a live edit (at startup).

        An unsafe or malformed file is ignored. An entry that is off the allowlist, no longer
        valid, refused by a guardrail or for a language that is gone is dropped; warnings name
        the path only. The file is rewritten without dropped entries.
        """
        if self._file is None:
            return
        try:
            stored = self._file.load()
        except ValueError as exc:
            log.warning("web UI settings file ignored: %s", exc)
            return
        order = sorted(stored, key=lambda p: not p.startswith("safety.minor_output_blocklist."))
        with self._lock:
            for path in order:
                try:
                    result = self._apply(path, stored[path], RESTORE_SOURCE)
                except (SettingNotAllowed, ValidationError, GuardrailError, ConfigError):
                    log.warning("web UI setting %s dropped: no longer acceptable", path)
                    continue
                self._overrides[path] = result["value"]
            if set(self._overrides) != set(stored):
                self._persist()

    def _apply(self, path: object, value: object, source: str) -> dict[str, Any]:
        cfg = self.stack.effective()
        entry = expand(cfg).get(path) if isinstance(path, str) else None
        if entry is None or not isinstance(path, str):
            raise SettingNotAllowed("path")
        if secret_shaped(value):
            raise ValidationError("value_type", "value")
        clean = self._clean(entry, value, cfg)
        old = get_path(cfg, path)
        if clean == old:
            return {"path": path, "changed": False, "value": old}
        self.stack.set(
            path,
            clean,
            source=source,
            admin=bool(field_meta(path).get("admin_only")),
            must_exist=True,  # a language removed since `expand` is not recreated
        )
        return {"path": path, "changed": True, "value": clean}

    def _persist(self) -> bool:
        if self._file is None:
            return True
        try:
            self._file.save(self._overrides)
        except OSError as exc:
            log.warning("web UI settings not saved (%s)", type(exc).__name__)
            return False
        return True

    # ------------------------------------------------------------------ schema and cleaning

    def _schema(self, entry: Setting, cfg: RoboEmotConfig) -> dict[str, Any]:
        web = cfg.webui
        phrase = {"type": "string", "maxLength": web.settings_phrase_max_chars}
        if entry.kind in (LIST_KIND, WORDS_KIND):
            return {
                "type": "array",
                "items": phrase,
                "minItems": 1,
                "maxItems": web.settings_list_max_items,
            }
        if entry.kind == NAME_KIND:
            return {"type": ["string", "null"], "maxLength": cfg.voice.name_max_chars}
        return {
            "type": "string",
            "minLength": 0 if entry.allow_empty else 1,
            "maxLength": web.settings_line_max_chars,
        }

    def _clean(self, entry: Setting, value: object, cfg: RoboEmotConfig) -> Any:
        return {
            LIST_KIND: self._phrases,
            WORDS_KIND: self._words,
            NAME_KIND: self._name,
            LINE_KIND: self._line,
        }[entry.kind](entry, value, cfg)

    @staticmethod
    def _items(value: object, cfg: RoboEmotConfig, pattern: re.Pattern[str]) -> list[str]:
        """Distinct, whitespace-normalised items of a list within the count, length and
        character limits."""
        web = cfg.webui
        items = list(dict.fromkeys(_string_list(value)))
        if not items:
            raise ValidationError("list_empty", "value")
        if len(items) > web.settings_list_max_items:
            raise ValidationError("list_too_long", "value")
        for item in items:
            if len(item) > web.settings_phrase_max_chars:
                raise ValidationError("phrase_too_long", "value")
            if pattern.fullmatch(item) is None or not normalise(item.rstrip("*")):
                raise ValidationError("phrase_invalid", "value")
        return items

    def _phrases(self, entry: Setting, value: object, cfg: RoboEmotConfig) -> list[str]:
        return self._items(value, cfg, PHRASE_PATTERN)

    def _words(self, entry: Setting, value: object, cfg: RoboEmotConfig) -> list[str]:
        return self._items(value, cfg, WORD_PATTERN)

    @staticmethod
    def _name(entry: Setting, value: object, cfg: RoboEmotConfig) -> str | None:
        """The robot's name under the prompt-name rules (``sanitise_name``): a value the
        sanitiser would alter is refused rather than silently changed; empty or null resets to
        the platform's name."""
        if value is None:
            return None
        text = " ".join(_text(value).split())
        if not text:
            return None
        if len(text) > cfg.voice.name_max_chars:
            raise ValidationError("name_too_long", "value")
        if sanitise_name(text, cfg.voice.name_max_chars) != text:
            raise ValidationError("name_invalid", "value")
        if _blocked_by_filter(cfg, text):
            raise ValidationError("name_blocked", "value")
        return text

    @staticmethod
    def _line(entry: Setting, value: object, cfg: RoboEmotConfig) -> str:
        """A spoken line: one line of printable text; ``{name}`` only where the field has it."""
        text = _text(value)
        if not text:
            if entry.allow_empty:
                return ""
            raise ValidationError("line_empty", "value")
        if len(text) > cfg.webui.settings_line_max_chars:
            raise ValidationError("line_too_long", "value")
        if any(unicodedata.category(c).startswith("C") or c in LINE_BREAKS for c in text):
            raise ValidationError("line_invalid", "value")
        allowed = NAME_PLACEHOLDER if entry.template.endswith(".robot_name") else ""
        if "{" in text.replace(allowed, "") or "}" in text.replace(allowed, ""):
            raise ValidationError("line_invalid", "value")
        name = sanitise_name(cfg.voice.wake_word or "", cfg.voice.name_max_chars)
        if _blocked_by_filter(cfg, text.replace(NAME_PLACEHOLDER, name)):
            raise ValidationError("line_blocked", "value")
        return text
