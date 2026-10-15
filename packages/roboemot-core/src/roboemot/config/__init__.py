"""Layered, typed configuration (SPEC 11A)."""

from .schema import SCHEMA_VERSION, RoboEmotConfig
from .stack import (
    AuditEntry,
    ConfigChange,
    ConfigError,
    ConfigStack,
    GuardrailError,
    Violation,
    field_meta,
    resolve_preset,
    violations,
)

__all__ = [
    "SCHEMA_VERSION",
    "AuditEntry",
    "ConfigChange",
    "ConfigError",
    "ConfigStack",
    "GuardrailError",
    "RoboEmotConfig",
    "Violation",
    "field_meta",
    "resolve_preset",
    "violations",
]
