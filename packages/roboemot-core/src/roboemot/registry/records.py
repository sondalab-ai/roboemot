"""Registry records for people, animals and objects (SPEC 6, Person record).

Embeddings are model-specific, so every record names the embedding model that produced them
and the matcher ignores records from another model. Animal and object records have no consent,
age or register fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np

AGE_GROUPS: tuple[str, ...] = ("child", "teen", "adult", "senior", "unknown")
MINOR_AGE_GROUPS = frozenset({"child", "teen", "unknown"})
GENDERS: tuple[str, ...] = ("female", "male", "other", "unknown")

Status = Literal["active", "pending"]
Source = Literal["declared", "estimated"]
Origin = Literal["enrolled", "xmp", "auto"]
EntityKind = Literal["animal", "object"]


def possible_minor(age_group: str | None) -> bool:
    """True for a child, a teen or an unknown age group (SPEC 5, Rules for estimated attributes)."""
    return age_group is None or age_group in MINOR_AGE_GROUPS


def _iso(when: datetime | None) -> str | None:
    return None if when is None else when.isoformat()


def _when(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


def _no_embeddings() -> np.ndarray:
    return np.zeros((0, 0), np.float32)


@dataclass(frozen=True)
class Attribute:
    """An age group or gender value and whether it was declared or estimated."""

    value: str
    source: Source


@dataclass(frozen=True)
class Consent:
    """Who consented, when and to what; the guardian is required for minors (SPEC 12, Data)."""

    who: str
    when: datetime
    scope: tuple[str, ...]
    guardian: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "who": self.who,
            "when": self.when.isoformat(),
            "scope": list(self.scope),
            "guardian": self.guardian,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Consent:
        when = datetime.fromisoformat(data["when"])
        return cls(data["who"], when, tuple(data["scope"]), data["guardian"])


def _attr(data: dict[str, str] | None) -> Attribute | None:
    return None if data is None else Attribute(data["value"], data["source"])  # type: ignore[arg-type]


def _attr_json(attr: Attribute | None) -> dict[str, str] | None:
    return None if attr is None else {"value": attr.value, "source": attr.source}


@dataclass
class PersonRecord:
    """An enrolled or pending person. Pending people never hold biometric data."""

    person_id: str
    display_name: str
    status: Status = "pending"
    embedding_model_id: str | None = None
    face_embeddings: np.ndarray = field(default_factory=_no_embeddings)
    n_enrolled: int = 0
    voice_embedding: np.ndarray | None = None
    voice_model_id: str | None = None
    age_group: Attribute | None = None
    gender: Attribute | None = None
    register_overrides: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    consent: Consent | None = None
    last_seen: datetime | None = None

    @property
    def possible_minor(self) -> bool:
        return possible_minor(None if self.age_group is None else self.age_group.value)

    def meta(self) -> dict[str, Any]:
        """JSON-safe metadata; the embeddings are stored as arrays beside it."""
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "status": self.status,
            "embedding_model_id": self.embedding_model_id,
            "n_enrolled": self.n_enrolled,
            "voice_model_id": self.voice_model_id,
            "age_group": _attr_json(self.age_group),
            "gender": _attr_json(self.gender),
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
        consent = meta["consent"]
        return cls(
            person_id=meta["person_id"],
            display_name=meta["display_name"],
            status=meta["status"],
            embedding_model_id=meta["embedding_model_id"],
            face_embeddings=arrays.get("face", _no_embeddings()),
            n_enrolled=meta["n_enrolled"],
            voice_embedding=arrays.get("voice"),
            voice_model_id=meta["voice_model_id"],
            age_group=_attr(meta["age_group"]),
            gender=_attr(meta["gender"]),
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
    embeddings: np.ndarray = field(default_factory=_no_embeddings)
    description: str | None = None
    last_seen: datetime | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "category": self.category,
            "status": self.status,
            "origin": self.origin,
            "embedding_model_id": self.embedding_model_id,
            "description": self.description,
            "last_seen": _iso(self.last_seen),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {"appearance": self.embeddings} if len(self.embeddings) else {}

    @classmethod
    def from_stored(cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> EntityRecord:
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
        )
