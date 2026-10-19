"""Value cleaning for the settings write path (SPEC 11A, 12).

Two layers. ``clean_value`` validates any catalogue leaf from its descriptor (kind, nullable,
``ge``/``le``, choices, tuple length) and the ``webui.settings_*`` caps; the stack's pydantic
validation stays the final check. ``HOOKS`` attach the per-path rules that need more than the
kind: the wake word sanitiser, the child-safe lines, the wake phrases and the output blocklist.
Blocklist words can be removed: the stack refuses it on the runtime layer and the settings editor
retries it on the app layer (admin PIN). Every refusal is a ``ValidationError`` with a stable code
and never echoes the value.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from roboemot.config.catalogue import PATH_WILDCARD, FieldDescriptor
from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.output_filter import OutputFilter
from roboemot.voice.keywords import normalise
from roboemot.voice.robot_name import sanitise_name

from .validation import ValidationError

NAME_PLACEHOLDER = "{name}"
PHRASE_PATTERN = re.compile(r"[^\W_]+(?: [^\W_]+)*")
WORD_PATTERN = re.compile(r"[^\W_]+(?: [^\W_]+)*\*?")
KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
EVENT_PATTERN = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")
LINE_BREAKS = "  "  # categories Zl and Zp: line breaks the C* check does not cover
TEXT_CONTROLS = "\n\t"  # the only control characters a free text may hold
LIST_KIND, NAME_KIND, LINE_KIND, WORDS_KIND = "phrases", "name", "line", "words"


def replace(row: FieldDescriptor, **fields: Any) -> FieldDescriptor:
    """A copy of the descriptor ``row`` with ``fields`` changed."""
    return cast(FieldDescriptor, {**row, **fields})


def _fail(code: str) -> ValidationError:
    return ValidationError(code, "value")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise _fail("value_type")
    return unicodedata.normalize("NFC", value).strip()


def _has_control(text: str, allowed: str = "") -> bool:
    return any(unicodedata.category(c).startswith("C") and c not in allowed for c in text)


def _blocked_by_filter(cfg: RoboEmotConfig, text: str) -> bool:
    """Whether the minors output filter (the effective blocklist) blocks ``text``."""
    return OutputFilter(cfg.safety.minor_output_blocklist).blocked(text)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise _fail("value_type")
    if any(_has_control(x) for x in value):
        raise _fail("phrase_invalid")
    return [" ".join(unicodedata.normalize("NFC", x).split()) for x in value]


# ------------------------------------------------------------------ per-path hooks


@dataclass(frozen=True)
class Hook:
    """A path pattern (``*`` matches one segment) and the cleaner kind that applies to it."""

    pattern: str
    kind: str


HOOKS: tuple[Hook, ...] = (
    Hook("voice.keywords.*.wake_phrases", LIST_KIND),
    Hook("voice.wake_word", NAME_KIND),
    Hook("safety.minor_lines.*.*", LINE_KIND),
    Hook("safety.minor_output_blocklist.*", WORDS_KIND),
)


def hook_for(path: str) -> Hook | None:
    """The hook whose pattern matches ``path`` segment by segment, if any."""
    parts = path.split(".")
    for hook in HOOKS:
        want = hook.pattern.split(".")
        if len(want) == len(parts) and all(
            w in (PATH_WILDCARD, p) for w, p in zip(want, parts, strict=True)
        ):
            return hook
    return None


def _items(value: object, cfg: RoboEmotConfig, pattern: re.Pattern[str]) -> list[str]:
    """Distinct, whitespace-normalised items of a list within the count, length and character
    limits."""
    web = cfg.webui
    items = list(dict.fromkeys(_string_list(value)))
    if not items:
        raise _fail("list_empty")
    if len(items) > web.settings_list_max_items:
        raise _fail("list_too_long")
    for item in items:
        if len(item) > web.settings_phrase_max_chars:
            raise _fail("phrase_too_long")
        if pattern.fullmatch(item) is None or not normalise(item.rstrip("*")):
            raise _fail("phrase_invalid")
    return items


def _name(value: object, cfg: RoboEmotConfig) -> str | None:
    """The robot's name under the prompt-name rules (``sanitise_name``): a value the sanitiser
    would alter is refused rather than silently changed; empty or null resets to the platform's
    name."""
    if value is None:
        return None
    text = " ".join(_text(value).split())
    if not text:
        return None
    if len(text) > cfg.voice.name_max_chars:
        raise _fail("name_too_long")
    if sanitise_name(text, cfg.voice.name_max_chars) != text:
        raise _fail("name_invalid")
    if _blocked_by_filter(cfg, text):
        raise _fail("name_blocked")
    return text


def _line(path: str, value: object, cfg: RoboEmotConfig) -> str:
    """A spoken line: one line of printable text; ``{name}`` only where the field has it."""
    text = _text(value)
    if not text:
        if path.endswith(".robot_unnamed"):
            return ""
        raise _fail("line_empty")
    if len(text) > cfg.webui.settings_line_max_chars:
        raise _fail("line_too_long")
    if _has_control(text) or any(c in LINE_BREAKS for c in text):
        raise _fail("line_invalid")
    allowed = NAME_PLACEHOLDER if path.endswith(".robot_name") else ""
    if "{" in text.replace(allowed, "") or "}" in text.replace(allowed, ""):
        raise _fail("line_invalid")
    name = sanitise_name(cfg.voice.wake_word or "", cfg.voice.name_max_chars)
    if _blocked_by_filter(cfg, text.replace(NAME_PLACEHOLDER, name)):
        raise _fail("line_blocked")
    return text


def clean_hooked(hook: Hook, path: str, value: object, cfg: RoboEmotConfig) -> Any:
    """Apply a hook's cleaner to ``value``."""
    if hook.kind == LIST_KIND:
        return _items(value, cfg, PHRASE_PATTERN)
    if hook.kind == WORDS_KIND:
        return _items(value, cfg, WORD_PATTERN)
    if hook.kind == NAME_KIND:
        return _name(value, cfg)
    return _line(path, value, cfg)


# ------------------------------------------------------------------ generic, by kind

Cleaner = Callable[[FieldDescriptor, Any, RoboEmotConfig], Any]


def _bounded(row: FieldDescriptor, number: float) -> None:
    if not math.isfinite(number):
        raise _fail("value_type")
    if (row["ge"] is not None and number < row["ge"]) or (
        row["le"] is not None and number > row["le"]
    ):
        raise _fail("value_range")


def _number(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail("value_type")
    _bounded(row, value)
    return float(value)


def _int(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail("value_type")
    _bounded(row, value)
    return value


def _bool(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> bool:
    if not isinstance(value, bool):
        raise _fail("value_type")
    return value


def free_text(value: Any, cfg: RoboEmotConfig) -> str:
    """NFC text without control characters (newline and tab allowed) within the text cap."""
    if not isinstance(value, str):
        raise _fail("value_type")
    text = unicodedata.normalize("NFC", value)
    if _has_control(text, TEXT_CONTROLS):
        raise _fail("text_invalid")
    if len(text) > cfg.webui.settings_text_max_chars:
        raise _fail("text_too_long")
    return text


def _plain_text(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> str:
    return free_text(value, cfg)


def _enum(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> Any:
    if not any(type(value) is type(c) and value == c for c in row["choices"] or []):
        raise _fail("value_choice")
    return value


def _item_row(row: FieldDescriptor) -> FieldDescriptor:
    return replace(row, kind=row["item_kind"] or "json", nullable=False)


def _tuple(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> list[Any]:
    if not isinstance(value, list | tuple) or len(value) != row["length"]:
        raise _fail("value_type")
    item = _item_row(row)
    return [CLEANERS[item["kind"]](item, v, cfg) for v in value]


def _counted(value: Any, cfg: RoboEmotConfig) -> list[Any]:
    if not isinstance(value, list):
        raise _fail("value_type")
    if len(value) > cfg.webui.settings_list_max_items:
        raise _fail("list_too_long")
    return value


def _line_item(text: Any, cfg: RoboEmotConfig) -> str:
    if not isinstance(text, str):
        raise _fail("value_type")
    text = unicodedata.normalize("NFC", text)
    if _has_control(text):
        raise _fail("phrase_invalid")
    if len(text) > cfg.webui.settings_line_max_chars:
        raise _fail("phrase_too_long")
    return text


def _string_list_kind(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> list[str]:
    return [_line_item(v, cfg) for v in _counted(value, cfg)]


def _enum_list(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> list[Any]:
    return [_enum(row, v, cfg) for v in _counted(value, cfg)]


def _tuple_list(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> list[Any]:
    return [_tuple(row, v, cfg) for v in _counted(value, cfg)]


def _mapping(value: Any, cfg: RoboEmotConfig) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise _fail("value_type")
    if len(value) > cfg.webui.settings_list_max_items:
        raise _fail("list_too_long")
    for key in value:
        _line_item(key, cfg)
    return value


def _string_dict(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> dict[str, str]:
    return {k: free_text(v, cfg) for k, v in _mapping(value, cfg).items()}


def _number_dict(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> dict[str, float]:
    plain = replace(row, ge=None, le=None)
    return {k: _number(plain, v, cfg) for k, v in _mapping(value, cfg).items()}


def _json(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> Any:
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _fail("value_type") from exc
    if len(text) > cfg.webui.settings_text_max_chars:
        raise _fail("text_too_long")
    return json.loads(text)


CLEANERS: dict[str, Cleaner] = {
    "number": _number,
    "int": _int,
    "bool": _bool,
    "text": _plain_text,
    "enum": _enum,
    "tuple": _tuple,
    "string_list": _string_list_kind,
    "enum_list": _enum_list,
    "tuple_list": _tuple_list,
    "string_dict": _string_dict,
    "number_dict": _number_dict,
    "json": _json,
}


def clean_value(row: FieldDescriptor, value: Any, cfg: RoboEmotConfig) -> Any:
    """``value`` validated and normalised for the leaf ``row`` (a catalogue descriptor): its kind,
    nullability, limits and choices, within the caps of ``webui.settings_*``. Containers are not
    leaves and raise ``value_type``. The stack's own validation still has the last word."""
    if value is None:
        if row["nullable"]:
            return None
        raise _fail("value_type")
    cleaner = CLEANERS.get(row["kind"])
    if cleaner is None:
        raise _fail("value_type")
    return cleaner(row, value, cfg)


def clean_key(key: object, cfg: RoboEmotConfig, *, dots: bool = False) -> str:
    """A new dict key or entry identity: short, plain characters, so it is a safe path segment
    (``dots`` also allows ``.``, for bus event names)."""
    if (
        not isinstance(key, str)
        or (EVENT_PATTERN if dots else KEY_PATTERN).fullmatch(key) is None
        or len(key) > cfg.webui.settings_phrase_max_chars
    ):
        raise ValidationError("key_invalid", "key")
    return key
