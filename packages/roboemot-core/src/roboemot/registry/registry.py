"""The registry of known people, animals and objects (SPEC 6, 11, 12).

People are enrolled only with a recorded consent; without one they stay ``pending`` and hold
no biometric data. Nothing here enrolls a person automatically. Embeddings and profile fields
are removed synchronously by ``forget``. Never log embeddings or crops (SPEC 12, Privacy).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np

from ..config.schema import IdentityConfig
from .records import (
    Attribute,
    Consent,
    EntityKind,
    EntityRecord,
    Origin,
    PersonRecord,
)
from .store import RegistryStore

PEOPLE = "people"
ENTITIES = "entities"
GalleryKind = Literal["face", "voice", "entity"]
GUARDIAN_AGE_GROUPS = frozenset({"child", "teen"})


class ConsentError(ValueError):
    """Raised when a person would be enrolled without a valid consent record."""


class HistoryLink(Protocol):
    """A store holding per-person data elsewhere (the opt-in world history, SPEC 5A)."""

    def export(self, person_id: str) -> Any: ...

    def forget(self, person_id: str) -> None: ...


def _matrix(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("embeddings must be a non-empty (n, dim) matrix")
    return array


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.where(norms == 0, 1, norms)


class Registry:
    """People, animals and objects over a ``RegistryStore``, with an in-memory index."""

    def __init__(
        self,
        store: RegistryStore,
        identity: IdentityConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.identity = identity
        self._clock = clock or (lambda: datetime.now(UTC))
        self._people: dict[str, PersonRecord] = {}
        self._entities: dict[str, EntityRecord] = {}
        self._history: list[HistoryLink] = []
        self._gallery_cache: dict[tuple[GalleryKind, str], dict[str, np.ndarray]] = {}
        self.version = 0
        for key in store.keys(PEOPLE):
            self._people[key] = PersonRecord.from_stored(*store.get(PEOPLE, key))
        for key in store.keys(ENTITIES):
            self._entities[key] = EntityRecord.from_stored(*store.get(ENTITIES, key))

    def attach_history(self, link: HistoryLink) -> None:
        """Include ``link`` in ``export`` and ``forget`` (the world history, SPEC 5A)."""
        self._history.append(link)

    def _save_person(self, record: PersonRecord) -> None:
        self.store.put(PEOPLE, record.person_id, record.meta(), record.arrays())
        self._people[record.person_id] = record
        self._changed()

    def _save_entity(self, record: EntityRecord) -> None:
        self.store.put(ENTITIES, record.entity_id, record.meta(), record.arrays())
        self._entities[record.entity_id] = record
        self._changed()

    def _changed(self) -> None:
        self.version += 1
        self._gallery_cache.clear()

    def _consent_valid(self, consent: Consent | None, age_group: str | None) -> bool:
        if consent is None:
            return False
        return not (age_group in GUARDIAN_AGE_GROUPS and not consent.guardian)

    def _check_enrollment(self, embeddings: np.ndarray) -> None:
        low, high = self.identity.enroll_min_embeddings, self.identity.max_embeddings_per_person
        if not low <= len(embeddings) <= high:
            raise ValueError(f"enrollment needs {low} to {high} embeddings, got {len(embeddings)}")

    # ------------------------------------------------------------------ people

    def enroll(
        self,
        display_name: str,
        face_embeddings: Sequence[Sequence[float]] | np.ndarray,
        embedding_model_id: str,
        *,
        consent: Consent | None = None,
        voice_embedding: Sequence[float] | np.ndarray | None = None,
        voice_model_id: str | None = None,
        age_group: str | None = None,
        gender: str | None = None,
        register_overrides: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> PersonRecord:
        """Enroll a person; without a valid consent the record stays ``pending`` and empty.

        A minor needs a guardian in the consent. Embeddings given without consent are
        discarded, never stored. The age group and gender passed here are declared values.
        """
        record = PersonRecord(
            person_id=str(uuid.uuid4()),
            display_name=display_name,
            age_group=None if age_group is None else Attribute(age_group, "declared"),
            gender=None if gender is None else Attribute(gender, "declared"),
            register_overrides=dict(register_overrides or {}),
            preferences=dict(preferences or {}),
        )
        if self._consent_valid(consent, age_group):
            self._activate(
                record,
                face_embeddings,
                embedding_model_id,
                consent,
                voice_embedding,
                voice_model_id,
            )
        self._save_person(record)
        return record

    def create_pending_person(self, display_name: str) -> PersonRecord:
        """A name-only record (catalogue import, SPEC 6); it holds no biometric data."""
        record = PersonRecord(person_id=str(uuid.uuid4()), display_name=display_name)
        self._save_person(record)
        return record

    def complete_enrollment(
        self,
        person_id: str,
        face_embeddings: Sequence[Sequence[float]] | np.ndarray,
        embedding_model_id: str,
        consent: Consent,
        *,
        voice_embedding: Sequence[float] | np.ndarray | None = None,
        voice_model_id: str | None = None,
    ) -> PersonRecord:
        """Turn a pending person active once consent exists and the faces are captured."""
        record = self._people[person_id]
        if record.status != "pending":
            raise ValueError(f"{person_id} is already enrolled")
        declared = None if record.age_group is None else record.age_group.value
        if not self._consent_valid(consent, declared):
            raise ConsentError("a valid consent record is required (guardian for minors)")
        self._activate(
            record, face_embeddings, embedding_model_id, consent, voice_embedding, voice_model_id
        )
        self._save_person(record)
        return record

    def _activate(
        self,
        record: PersonRecord,
        face_embeddings: Sequence[Sequence[float]] | np.ndarray,
        model_id: str,
        consent: Consent | None,
        voice: Sequence[float] | np.ndarray | None,
        voice_model_id: str | None,
    ) -> None:
        faces = _matrix(face_embeddings)
        self._check_enrollment(faces)
        if voice is not None and not voice_model_id:
            raise ValueError("a voice embedding needs its voice_model_id")
        record.face_embeddings = faces
        record.n_enrolled = len(faces)
        record.embedding_model_id = model_id
        record.voice_embedding = None if voice is None else np.asarray(voice, dtype=np.float32)
        record.voice_model_id = voice_model_id if voice is not None else None
        record.consent = consent
        record.status = "active"

    def add_refresh_embedding(
        self, person_id: str, embedding: Sequence[float] | np.ndarray, model_id: str
    ) -> bool:
        """Add a face embedding after a high-confidence match; True when it was stored.

        Skipped when the model differs, when it nearly duplicates a stored one, or when the
        per-person cap is reached and no refreshed embedding exists to replace. At the cap the
        oldest refreshed one is dropped; enrollment embeddings are never replaced.
        """
        record = self._people.get(person_id)
        row = np.asarray(embedding, dtype=np.float32)[None, :]
        if record is None or record.status != "active" or record.embedding_model_id != model_id:
            return False
        if row.shape[1] != record.face_embeddings.shape[1]:
            return False
        sims = _unit(record.face_embeddings) @ _unit(row)[0]
        if float(sims.max()) >= self.identity.refresh_max_similarity:
            return False
        faces = record.face_embeddings
        if len(faces) >= self.identity.max_embeddings_per_person:
            if len(faces) <= record.n_enrolled:
                return False
            faces = np.delete(faces, record.n_enrolled, axis=0)
        record.face_embeddings = np.vstack([faces, row])
        self._save_person(record)
        return True

    def mark_seen(self, record_id: str) -> None:
        """Set ``last_seen`` of a person or entity to now."""
        now = self._clock()
        if record_id in self._people:
            self._people[record_id].last_seen = now
            self._save_person(self._people[record_id])
        elif record_id in self._entities:
            self._entities[record_id].last_seen = now
            self._save_entity(self._entities[record_id])
        else:
            raise KeyError(record_id)

    def get_person(self, person_id: str) -> PersonRecord:
        return self._people[person_id]

    def list(self) -> list[PersonRecord | EntityRecord]:
        """All people then all entities, in registration order of the store keys."""
        return [*self._people.values(), *self._entities.values()]

    def forget(self, person_id: str) -> bool:
        """Remove every embedding, field and file of a person, and their attached history.

        Synchronous: when it returns, nothing about the person is left in memory or on disk.
        Returns False when the person was unknown.
        """
        known = self._people.pop(person_id, None) is not None
        self.store.delete(PEOPLE, person_id)
        for link in self._history:
            link.forget(person_id)
        self._changed()
        return known

    def export(self, person_id: str) -> dict[str, Any]:
        """The person's whole record, embeddings and attached history, as JSON-safe data."""
        data = self._people[person_id].export()
        data["history"] = [link.export(person_id) for link in self._history]
        return data

    # ---------------------------------------------------------------- entities

    def enroll_entity(
        self,
        kind: EntityKind,
        display_name: str,
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        embedding_model_id: str,
        *,
        category: str = "",
        origin: Origin = "enrolled",
        description: str | None = None,
    ) -> EntityRecord:
        """Register an animal or object with its appearance embeddings; no consent applies."""
        record = EntityRecord(
            entity_id=str(uuid.uuid4()),
            kind=kind,
            display_name=display_name,
            category=category,
            origin=origin,
            description=description,
            embedding_model_id=embedding_model_id,
            embeddings=_matrix(embeddings),
            status="active",
        )
        self._save_entity(record)
        return record

    def create_pending_entity(
        self,
        kind: EntityKind,
        display_name: str,
        *,
        category: str = "",
        origin: Origin = "xmp",
        description: str | None = None,
    ) -> EntityRecord:
        """A name-only entity (catalogue import) that gets its appearance once confirmed."""
        record = EntityRecord(
            entity_id=str(uuid.uuid4()),
            kind=kind,
            display_name=display_name,
            category=category,
            origin=origin,
            description=description,
        )
        self._save_entity(record)
        return record

    def confirm_entity(
        self,
        entity_id: str,
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        embedding_model_id: str,
    ) -> EntityRecord:
        """Give a pending entity its appearance (the owner confirmed "this is Mia")."""
        record = self._entities[entity_id]
        record.embeddings = _matrix(embeddings)
        record.embedding_model_id = embedding_model_id
        record.status = "active"
        self._save_entity(record)
        return record

    def get_entity(self, entity_id: str) -> EntityRecord:
        return self._entities[entity_id]

    def rename_entity(self, entity_id: str, display_name: str) -> EntityRecord:
        record = self._entities[entity_id]
        record.display_name = display_name
        self._save_entity(record)
        return record

    def merge_entities(self, keep_id: str, drop_id: str) -> EntityRecord:
        """Fold ``drop_id`` into ``keep_id``: embeddings are pooled and the dropped one deleted."""
        keep, drop = self._entities[keep_id], self._entities[drop_id]
        if len(drop.embeddings):
            if len(keep.embeddings) and keep.embedding_model_id != drop.embedding_model_id:
                raise ValueError("cannot merge entities from different embedding models")
            keep.embedding_model_id = drop.embedding_model_id
            pooled = (
                [keep.embeddings, drop.embeddings] if len(keep.embeddings) else [drop.embeddings]
            )
            keep.embeddings = np.vstack(pooled)
            keep.status = "active"
        self._save_entity(keep)
        self.delete_entity(drop_id)
        return keep

    def delete_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self.store.delete(ENTITIES, entity_id)
        self._changed()

    # ----------------------------------------------------------------- matching

    def gallery(self, kind: GalleryKind, model_id: str) -> dict[str, np.ndarray]:
        """Unit-length embeddings per active owner for ``model_id``; other models are skipped."""
        cached = self._gallery_cache.get((kind, model_id))
        if cached is not None:
            return cached
        out: dict[str, np.ndarray] = {}
        if kind == "entity":
            for e in self._entities.values():
                if e.status == "active" and e.embedding_model_id == model_id and len(e.embeddings):
                    out[e.entity_id] = _unit(e.embeddings)
        else:
            for p in self._people.values():
                if p.status != "active":
                    continue
                if kind == "face" and p.embedding_model_id == model_id and len(p.face_embeddings):
                    out[p.person_id] = _unit(p.face_embeddings)
                elif (
                    kind == "voice"
                    and p.voice_model_id == model_id
                    and p.voice_embedding is not None
                ):
                    out[p.person_id] = _unit(p.voice_embedding[None, :])
        self._gallery_cache[(kind, model_id)] = out
        return out


__all__ = ["ConsentError", "GalleryKind", "HistoryLink", "Registry"]
