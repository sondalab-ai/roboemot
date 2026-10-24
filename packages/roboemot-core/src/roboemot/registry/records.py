"""Registry records for people, animals and objects (SPEC 6, Person record).

Embeddings are model-specific, so every record names the embedding model that produced them
and the matcher ignores records from another model. Animal and object records have no consent
or register fields. People carry no age group or gender (owner decision 2026-10-23); a record
stored before that still loads, its ``age_group``, ``gender`` and consent ``guardian`` ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import ValidationError

from ..config.schema import RegisterParams

CONSENT_SCOPES: tuple[str, ...] = ("face", "voice", "refresh")
ENTITY_KINDS: tuple[str, ...] = ("animal", "object")
SCHEMA_VERSION = 1

Status = Literal["active", "pending"]
Origin = Literal["enrolled", "xmp", "auto"]
EntityKind = Literal["animal", "object"]


def check_register_overrides(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of a person's partial register, or raise ``ValueError`` if a key is not a
    register parameter or a value is out of range (checked when the record is written)."""
    overrides = dict(value or {})
    try:
        RegisterParams.model_validate(overrides)
    except ValidationError as exc:
        raise ValueError(
            f"register_overrides is not a valid partial register: {exc.error_count()} error(s)"
        ) from None
    return overrides


def check_entity_kind(value: str) -> str:
    """Return ``value`` if it is an entity kind (animal or object), else raise ``ValueError``."""
    if value not in ENTITY_KINDS:
        raise ValueError(f"kind must be one of {ENTITY_KINDS}")
    return value


def _iso(when: datetime | None) -> str | None:
    return None if when is None else when.isoformat()


def _when(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


def _no_embeddings() -> np.ndarray:
    return np.zeros((0, 0), np.float32)


def _check_version(meta: dict[str, Any]) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")


@dataclass(frozen=True)
class RecordSummary:
    """What ``Registry.list`` returns: no embeddings, no consent, no preferences."""

    id: str
    display_name: str
    status: Status
    kind: Literal["person", "animal", "object"]
    last_seen: datetime | None


class ConsentError(ValueError):
    """Raised when a consent record is invalid or missing for what is being stored."""


@dataclass(frozen=True)
class Consent:
    """Who consented, when and to what.

    ``scope`` is a non-empty subset of ``CONSENT_SCOPES``: ``face`` and ``voice`` allow storing
    those embeddings, ``refresh`` allows adding embeddings after enrollment (SPEC 12, Data).
    """

    who: str
    when: datetime
    scope: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", tuple(self.scope))
        if not self.who.strip():
            raise ConsentError("consent needs the name of who consented")
        if self.when.tzinfo is None or self.when.utcoffset() is None:
            raise ConsentError("consent time must be timezone-aware")
        if not self.scope or not set(self.scope) <= set(CONSENT_SCOPES):
            raise ConsentError(f"consent scope must be a non-empty subset of {CONSENT_SCOPES}")

    def to_json(self) -> dict[str, Any]:
        return {
            "who": self.who,
            "when": self.when.isoformat(),
            "scope": list(self.scope),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Consent:
        """Read a stored consent; a ``guardian`` key written before 2026-10-23 is ignored."""
        when = datetime.fromisoformat(data["when"])
        return cls(data["who"], when, tuple(data["scope"]))


@dataclass
class PersonRecord:
    """An enrolled or pending person. Pending people never hold biometric data."""

    person_id: str
    display_name: str
    status: Status = "pending"
    embedding_model_id: str | None = None
    face_embeddings: np.ndarray = field(default_factory=_no_embeddings, repr=False)
    n_enrolled: int = 0
    voice_embedding: np.ndarray | None = field(default=None, repr=False)
    voice_model_id: str | None = None
    register_overrides: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict, repr=False)
    consent: Consent | None = field(default=None, repr=False)
    last_seen: datetime | None = None

    def meta(self) -> dict[str, Any]:
        """JSON-safe metadata; the embeddings are stored as arrays beside it."""
        return {
            "schema_version": SCHEMA_VERSION,
            "person_id": self.person_id,
            "display_name": self.display_name,
            "status": self.status,
            "embedding_model_id": self.embedding_model_id,
            "n_enrolled": self.n_enrolled,
            "voice_model_id": self.voice_model_id,
            "register_overrides": self.register_overrides,
            "preferences": self.preferences,
            "consent": None if self.consent is None else self.consent.to_json(),
            "last_seen": _iso(self.last_seen),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if len(self.face_embeddings):
            out["face"] = self.face_embeddings
        if self.voice_embedding is not None:
            out["voice"] = self.voice_embedding
        return out

    @classmethod
    def from_stored(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> PersonRecord:
        """Rebuild a stored person; the ``age_group`` and ``gender`` of an older record are
        ignored (no migration: they go when the record is next written)."""
        consent = meta["consent"]
        _check_version(meta)
        return cls(
            person_id=meta["person_id"],
            display_name=meta["display_name"],
            status=meta["status"],
            embedding_model_id=meta["embedding_model_id"],
            face_embeddings=arrays.get("face", _no_embeddings()),
            n_enrolled=meta["n_enrolled"],
            voice_embedding=arrays.get("voice"),
            voice_model_id=meta["voice_model_id"],
            register_overrides=meta["register_overrides"],
            preferences=meta["preferences"],
            consent=None if consent is None else Consent.from_json(consent),
            last_seen=_when(meta["last_seen"]),
        )

    def export(self) -> dict[str, Any]:
        """Everything held about this person, embeddings included (SPEC 12, access request)."""
        data = self.meta()
        data["face_embeddings"] = self.face_embeddings.tolist()
        voice = self.voice_embedding
        data["voice_embedding"] = None if voice is None else voice.tolist()
        return data


@dataclass
class EntityRecord:
    """An animal or object known by appearance embeddings (SPEC 6)."""

    entity_id: str
    kind: EntityKind
    display_name: str
    category: str = ""
    status: Status = "pending"
    origin: Origin = "enrolled"
    embedding_model_id: str | None = None
    embeddings: np.ndarray = field(default_factory=_no_embeddings, repr=False)
    description: str | None = None
    last_seen: datetime | None = None
    photo_keys: list[str] = field(default_factory=list, repr=False)

    def meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "entity_id": self.entity_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "category": self.category,
            "status": self.status,
            "origin": self.origin,
            "embedding_model_id": self.embedding_model_id,
            "description": self.description,
            "last_seen": _iso(self.last_seen),
            "photo_keys": list(self.photo_keys),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {"appearance": self.embeddings} if len(self.embeddings) else {}

    @classmethod
    def from_stored(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> EntityRecord:
        _check_version(meta)
        return cls(
            entity_id=meta["entity_id"],
            kind=meta["kind"],
            display_name=meta["display_name"],
            category=meta["category"],
            status=meta["status"],
            origin=meta["origin"],
            embedding_model_id=meta["embedding_model_id"],
            embeddings=arrays.get("appearance", _no_embeddings()),
            description=meta["description"],
            last_seen=_when(meta["last_seen"]),
            photo_keys=[str(k) for k in meta.get("photo_keys", [])],
        )
