"""The web-editable settings: an allowlist, validation and the write path (SPEC 11A, 12).

``SETTINGS`` is the one place that says what the web UI may change. A path that no entry
expands to is refused. Values are cleaned here (length, count and character limits come from
``webui.settings_*`` and ``voice.name_max_chars``), then written through ``ConfigStack.set`` on
the runtime layer with no expiry, so the stack's guardrails, audit (source ``webui``) and hot
reload apply unchanged. The PIN confirmation is the caller's job (``WebApi``); this module
never sees a PIN or a secret.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from roboemot.config.schema import MinorLines, RoboEmotConfig
from roboemot.config.stack import _inner_model, _strip_optional, field_meta, get_path
from roboemot.voice.keywords import normalise
from roboemot.voice.robot_name import sanitise_name

from .validation import ValidationError

SOURCE = "webui"
NAME_PLACEHOLDER = "{name}"
PHRASE_PATTERN = re.compile(r"[^\W_][\w .'’\-]*")
WORD_PATTERN = re.compile(r"[^\W_]+(?: [^\W_]+)*\*?")
LIST_KIND, NAME_KIND, LINE_KIND, WORDS_KIND = "phrases", "name", "line", "words"


class StackPort(Protocol):
    """The part of ``roboemot.config.ConfigStack`` this module uses."""

    def effective(self) -> RoboEmotConfig: ...

    def set(self, key: str, value: Any, *, source: str, admin: bool) -> RoboEmotConfig: ...


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


def _field_description(path: str) -> str:
    """Schema description of the field a dotted path ends in ('' when unknown)."""
    model: Any = RoboEmotConfig
    parts, i, desc = path.split("."), 0, ""
    while i < len(parts) and model is not None:
        info = getattr(model, "model_fields", {}).get(parts[i])
        if info is None:
            return ""
        desc = info.description or ""
        annotation = _strip_optional(info.annotation)
        i += 1
        if getattr(annotation, "__origin__", None) in (dict, list):
            i += 1
        model = _inner_model(annotation)
    return desc


def _text(value: object, field: str = "value") -> str:
    if not isinstance(value, str):
        raise ValidationError("value_type", field)
    return unicodedata.normalize("NFC", value).strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValidationError("value_type", "value")
    if any(unicodedata.category(c).startswith("C") for x in value for c in x):
        raise ValidationError("phrase_invalid", "value")
    return [" ".join(unicodedata.normalize("NFC", x).split()) for x in value]


class SettingsEditor:
    """Lists the allowlisted settings and applies one validated change."""

    def __init__(
        self, stack: StackPort, *, defaults: Callable[[], RoboEmotConfig] = RoboEmotConfig
    ):
        self.stack = stack
        self._defaults = defaults

    def describe(self) -> list[dict[str, Any]]:
        """Each editable field with value, default, JSON-Schema-style limits and admin flag."""
        cfg, defaults = self.stack.effective(), self._defaults()
        rows = []
        for path, entry in sorted(expand(cfg).items()):
            try:
                default = get_path(defaults, path)
            except (KeyError, AttributeError, IndexError, ValueError):
                default = None
            rows.append(
                {
                    "path": path,
                    "value": get_path(cfg, path),
                    "default": default,
                    "schema": self._schema(entry, cfg),
                    "description": _field_description(path),
                    "needs_admin": bool(field_meta(path).get("admin_only")),
                }
            )
        return rows

    def change(self, path: object, value: object) -> dict[str, Any]:
        """Validate ``value`` for ``path`` and write it; the stack's guardrails have the last word.

        Raises ``SettingNotAllowed``, ``ValidationError`` (code, no value echoed) and the
        stack's ``GuardrailError`` / ``ConfigError``. An unchanged value writes nothing.
        """
        cfg = self.stack.effective()
        entry = expand(cfg).get(path) if isinstance(path, str) else None
        if entry is None or not isinstance(path, str):
            raise SettingNotAllowed("path")
        clean = self._clean(entry, value, cfg)
        old = get_path(cfg, path)
        if clean == old:
            return {"path": path, "changed": False, "value": old}
        self.stack.set(path, clean, source=SOURCE, admin=bool(field_meta(path).get("admin_only")))
        return {"path": path, "changed": True, "value": clean}

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
        if any(unicodedata.category(c).startswith("C") for c in text):
            raise ValidationError("line_invalid", "value")
        allowed = NAME_PLACEHOLDER if entry.template.endswith(".robot_name") else ""
        if "{" in text.replace(allowed, "") or "}" in text.replace(allowed, ""):
            raise ValidationError("line_invalid", "value")
        return text
