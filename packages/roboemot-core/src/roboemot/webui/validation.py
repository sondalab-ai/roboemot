"""Strict input validation for the web UI API (SPEC 6, 12).

Every request body is a JSON object whose keys are checked against an allow-list; strings are
length-bounded and limited to letters, digits and a few name punctuation marks.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from roboemot.config.schema import WebUIConfig
from roboemot.registry.records import CONSENT_SCOPES

NAME_PATTERN = re.compile(r"[^\W_][\w .'’\-]*")
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class ValidationError(ValueError):
    """A request field is missing or invalid; the message is a stable error code."""

    def __init__(
        self, code: str, field: str | None = None, extra: dict[str, str] | None = None
    ) -> None:
        super().__init__(code)
        self.code = code
        self.field = field
        self.extra = extra or {}


def only_keys(body: dict[str, Any], allowed: set[str]) -> None:
    """Reject unknown keys."""
    extra = sorted(set(body) - allowed)
    if extra:
        raise ValidationError("unknown_field", extra[0])


def name(value: object, field: str, cfg: WebUIConfig) -> str:
    """A trimmed person, animal or consenting-party name."""
    if not isinstance(value, str):
        raise ValidationError("name_invalid", field)
    text = unicodedata.normalize("NFC", value).strip()  # composed form: one spelling
    if not text:
        raise ValidationError("name_required", field)
    if len(text) > cfg.name_max_chars:
        raise ValidationError("name_too_long", field)
    if NAME_PATTERN.fullmatch(text) is None:
        raise ValidationError("name_invalid", field)
    return text


def record_id(value: object, field: str = "id") -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise ValidationError("id_invalid", field)
    return value


def quarantine_key(value: object) -> str:
    if not isinstance(value, str) or KEY_PATTERN.fullmatch(value) is None:
        raise ValidationError("id_invalid", "key")
    return value


def confirmed(body: dict[str, Any]) -> bool:
    return body.get("confirm") is True


def consent_scope(value: object) -> tuple[str, ...]:
    """A non-empty list of known scopes, in canonical order, without duplicates."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValidationError("scope_invalid", "scope")
    if not value or not set(value) <= set(CONSENT_SCOPES):
        raise ValidationError("scope_invalid", "scope")
    return tuple(s for s in CONSENT_SCOPES if s in value)


def enrollment_form(body: dict[str, Any], cfg: WebUIConfig) -> dict[str, Any]:
    """The complete-enrollment form as clean values.

    The person consenting defaults to the person enrolled. No age group, gender or guardian is
    asked (owner decision 2026-10-23); a body that still sends them is refused as unknown keys.
    """
    only_keys(body, {"display_name", "consent_who", "scope", "agreed", "pending_id"})
    if body.get("agreed") is not True:
        raise ValidationError("consent_not_agreed", "agreed")
    display = name(body.get("display_name"), "display_name", cfg)
    who_raw = body.get("consent_who")
    who = name(display if who_raw in (None, "") else who_raw, "consent_who", cfg)
    return {
        "display_name": display,
        "who": who,
        "scope": consent_scope(body.get("scope")),
        "pending_id": (
            None
            if body.get("pending_id") in (None, "")
            else record_id(body["pending_id"], "pending_id")
        ),
    }
