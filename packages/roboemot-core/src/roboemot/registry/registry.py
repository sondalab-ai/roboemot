"""The registry of known people, animals and objects (SPEC 6, 11, 12).

People are enrolled only with a recorded consent whose scope covers what is stored, and a
possible minor (anything not declared adult or senior) needs a guardian. Nothing here enrolls a
person automatically; ``create_pending_person`` makes a name-only record with no biometrics.
Records are replaced, never edited in place, and memory changes only after the store write
succeeded. ``forget`` is synchronous and complete. One re-entrant lock guards all state; the
matcher shares it. Never log embeddings or crops (SPEC 12, Privacy).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np

from ..config.schema import IdentityConfig
from .records import (
    Attribute,
    Consent,
    ConsentError,
    EntityKind,
    EntityRecord,
    Origin,
    PersonRecord,
    RecordSummary,
    check_age_group,
    check_entity_kind,
    check_gender,
    possible_minor,
)
from .store import RegistryStore

log = logging.getLogger(__name__)

PEOPLE = "people"
ENTITIES = "entities"
GalleryKind = Literal["face", "voice", "entity"]
Vectors = Sequence[Sequence[float]] | np.ndarray


class HistoryLink(Protocol):
    """A store holding per-person data elsewhere (the opt-in world history, SPEC 5A)."""

    def export(self, person_id: str) -> Any: ...

    def forget(self, person_id: str) -> None: ...


def _usable(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all() and (np.linalg.norm(array, axis=-1) > 0).all())


def _matrix(vectors: Vectors) -> np.ndarray:
    """A non-empty (n, dim) float32 copy; NaN, infinite and zero-norm rows are rejected."""
    array = np.array(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("embeddings must be a non-empty (n, dim) matrix")
    if not _usable(array):
        raise ValueError("embeddings must be finite and non-zero")
    return array


def _vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    return _matrix(np.asarray(vector, dtype=np.float32)[None, :])[0]


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


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
        self.lock = threading.RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._people: dict[str, PersonRecord] = {}
        self._entities: dict[str, EntityRecord] = {}
        self._history: list[HistoryLink] = []
        self._forget_listeners: list[Callable[[str], None]] = []
        self._gallery_cache: dict[tuple[GalleryKind, str], dict[str, np.ndarray]] = {}
        self._dirty: set[str] = set()
        self._last_refresh: dict[str, datetime] = {}
        self._last_flush = self._clock()
        self.version = 0
        self._load(PEOPLE, PersonRecord, self._people, "person_id")
        self._load(ENTITIES, EntityRecord, self._entities, "entity_id")

    def _load(self, collection: str, cls: Any, target: dict[str, Any], id_field: str) -> None:
        """Read every item; one that fails to parse is quarantined and logged by key only."""
        for key in self.store.keys(collection):
            try:
                record = cls.from_stored(*self.store.get(collection, key))
                if getattr(record, id_field) != key:
                    raise ValueError("key does not match the stored id")
                target[key] = record
            except Exception as exc:
                log.error("quarantined %s item %s (%s)", collection, key, type(exc).__name__)
                self.store.quarantine(collection, key)

    def attach_history(self, link: HistoryLink) -> None:
        """Include ``link`` in ``export`` and ``forget`` (the world history, SPEC 5A)."""
        with self.lock:
            self._history.append(link)

    def add_forget_listener(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(person_id)`` during ``forget`` (the matcher purges its votes)."""
        with self.lock:
            self._forget_listeners.append(listener)

    def _changed(self) -> None:
        self.version += 1
        self._gallery_cache.clear()

    def _commit_person(self, record: PersonRecord, *, new: bool = False) -> None:
        """Write the record, then publish it; an id that was forgotten meanwhile raises."""
        with self.lock:
            if not new and record.person_id not in self._people:
                raise KeyError(record.person_id)
            self.store.put(PEOPLE, record.person_id, record.meta(), record.arrays())
            self._people[record.person_id] = record
            self._changed()

    def _commit_entity(self, record: EntityRecord, *, new: bool = False) -> None:
        with self.lock:
            if not new and record.entity_id not in self._entities:
                raise KeyError(record.entity_id)
            self.store.put(ENTITIES, record.entity_id, record.meta(), record.arrays())
            self._entities[record.entity_id] = record
            self._changed()

    # ------------------------------------------------------------------ people

    def _require_consent(
        self, consent: Consent | None, age_group: str | None, needs: tuple[str, ...]
    ) -> Consent:
        if consent is None:
            raise ConsentError("a consent record is required")
        if possible_minor(age_group) and not consent.guardian:
            raise ConsentError("a guardian is required unless the person is declared adult")
        missing = [scope for scope in needs if scope not in consent.scope]
        if missing:
            raise ConsentError(f"consent scope lacks {missing}")
        return consent

    def _with_biometrics(
        self,
        record: PersonRecord,
        faces: Vectors,
        model_id: str,
        consent: Consent | None,
        voice: Sequence[float] | np.ndarray | None,
        voice_model_id: str | None,
    ) -> PersonRecord:
        """A copy of ``record`` that is active and holds the checked biometrics."""
        age = None if record.age_group is None else record.age_group.value
        needs = ("face",) if voice is None else ("face", "voice")
        granted = self._require_consent(consent, age, needs)
        matrix = _matrix(faces)
        low, high = self.identity.enroll_min_embeddings, self.identity.max_embeddings_per_person
        if not low <= len(matrix) <= high:
            raise ValueError(f"enrollment needs {low} to {high} embeddings, got {len(matrix)}")
        if voice is not None and not voice_model_id:
            raise ValueError("a voice embedding needs its voice_model_id")
        return replace(
            record,
            status="active",
            consent=granted,
            embedding_model_id=model_id,
            face_embeddings=matrix,
            n_enrolled=len(matrix),
            voice_embedding=None if voice is None else _vector(voice),
            voice_model_id=voice_model_id if voice is not None else None,
        )

    def enroll(
        self,
        display_name: str,
        face_embeddings: Vectors,
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
        """Enroll a person with consent; raises ``ConsentError`` without a valid one.

        The consent scope must include ``face`` (and ``voice`` when a voice embedding is given);
        a guardian is required unless ``age_group`` is declared adult or senior. Age group and
        gender passed here are declared values. For a name-only record use
        ``create_pending_person``.
        """
        declared_age = None if age_group is None else check_age_group(age_group)
        declared_gender = None if gender is None else check_gender(gender)
        base = PersonRecord(
            person_id=str(uuid.uuid4()),
            display_name=display_name,
            age_group=None if declared_age is None else Attribute(declared_age, "declared"),
            gender=None if declared_gender is None else Attribute(declared_gender, "declared"),
            register_overrides=dict(register_overrides or {}),
            preferences=dict(preferences or {}),
        )
        record = self._with_biometrics(
            base, face_embeddings, embedding_model_id, consent, voice_embedding, voice_model_id
        )
        self._commit_person(record, new=True)
        return record

    def create_pending_person(self, display_name: str) -> PersonRecord:
        """A name-only record (catalogue import, SPEC 6); it holds no biometric data."""
        record = PersonRecord(person_id=str(uuid.uuid4()), display_name=display_name)
        self._commit_person(record, new=True)
        return record

    def complete_enrollment(
        self,
        person_id: str,
        face_embeddings: Vectors,
        embedding_model_id: str,
        consent: Consent,
        *,
        voice_embedding: Sequence[float] | np.ndarray | None = None,
        voice_model_id: str | None = None,
        display_name: str | None = None,
        age_group: str | None = None,
        gender: str | None = None,
    ) -> PersonRecord:
        """Turn a pending person active once consent exists and the faces are captured.

        The same consent rules as ``enroll`` apply; a pending person has no declared age group
        unless one is passed here (declared values, like ``enroll``), so a guardian is normally
        required. ``display_name`` replaces the imported name.
        """
        with self.lock:
            pending = self._people[person_id]
            if pending.status != "pending":
                raise ValueError(f"{person_id} is already enrolled")
            if display_name is not None:
                pending = replace(pending, display_name=display_name)
            if age_group is not None:
                pending = replace(
                    pending, age_group=Attribute(check_age_group(age_group), "declared")
                )
            if gender is not None:
                pending = replace(pending, gender=Attribute(check_gender(gender), "declared"))
            record = self._with_biometrics(
                pending,
                face_embeddings,
                embedding_model_id,
                consent,
                voice_embedding,
                voice_model_id,
            )
            self._commit_person(record)
            return record

    def add_refresh_embedding(
        self, person_id: str, embedding: Sequence[float] | np.ndarray, model_id: str
    ) -> bool:
        """Add a face embedding after a high-confidence match; True when it was stored.

        Requires the ``refresh`` consent scope and ``refresh_min_interval_s`` since the last
        one. To stop template drift the embedding must be at least ``refresh_min_similarity``
        close to an *enrollment* embedding; refreshed ones do not count. Near-duplicates of any
        stored embedding are skipped. At the cap the oldest refreshed embedding is replaced and
        enrollment embeddings never are.
        """
        with self.lock:
            record = self._people.get(person_id)
            if (
                record is None
                or record.status != "active"
                or record.embedding_model_id != model_id
                or record.consent is None
                or "refresh" not in record.consent.scope
            ):
                return False
            row = np.array(embedding, dtype=np.float32)
            if row.ndim != 1 or row.shape[0] != record.face_embeddings.shape[1] or not _usable(row):
                return False
            sims = _unit(record.face_embeddings) @ (row / np.linalg.norm(row))
            if float(sims[: record.n_enrolled].max()) < self.identity.refresh_min_similarity:
                return False
            if float(sims.max()) >= self.identity.refresh_max_similarity:
                return False
            now = self._clock()
            last = self._last_refresh.get(person_id)
            if last is not None and (now - last).total_seconds() < (
                self.identity.refresh_min_interval_s
            ):
                return False
            faces = record.face_embeddings
            if len(faces) >= self.identity.max_embeddings_per_person:
                if len(faces) <= record.n_enrolled:
                    return False
                faces = np.delete(faces, record.n_enrolled, axis=0)
            self._commit_person(replace(record, face_embeddings=np.vstack([faces, row[None, :]])))
            self._last_refresh[person_id] = now
            return True

    def mark_seen(self, record_id: str) -> None:
        """Set ``last_seen`` to now in memory; the disk write is throttled (see ``flush``)."""
        with self.lock:
            record = self._people.get(record_id) or self._entities.get(record_id)
            if record is None:
                raise KeyError(record_id)
            now = self._clock()
            record.last_seen = now
            self._dirty.add(record_id)
            elapsed = (now - self._last_flush).total_seconds()
            if elapsed >= self.identity.last_seen_flush_interval_s:
                self.flush()

    def flush(self) -> None:
        """Write buffered ``last_seen`` updates; call it on shutdown. Failures are re-raised."""
        errors: list[Exception] = []
        with self.lock:
            for record_id in sorted(self._dirty):
                person = self._people.get(record_id)
                record = person or self._entities.get(record_id)
                if record is None:
                    self._dirty.discard(record_id)
                    continue
                collection = PEOPLE if person else ENTITIES
                try:
                    self.store.put(collection, record_id, record.meta(), record.arrays())
                    self._dirty.discard(record_id)
                except Exception as exc:
                    errors.append(exc)
            self._last_flush = self._clock()
        if errors:
            raise ExceptionGroup("last_seen flush incomplete", errors)

    def quarantined(self) -> list[str]:
        """Keys of unreadable people items set aside at startup; ``forget(key)`` purges them."""
        with self.lock:
            return self.store.quarantined(PEOPLE)

    def get_person(self, person_id: str) -> PersonRecord:
        with self.lock:
            return self._people[person_id]

    def list(self) -> list[RecordSummary]:
        """Summaries without biometrics: people by name, then animals and objects by name."""
        with self.lock:
            people = sorted(self._people.values(), key=lambda r: (r.display_name, r.person_id))
            entities = sorted(self._entities.values(), key=lambda r: (r.display_name, r.entity_id))
            rows = [
                RecordSummary(
                    p.person_id,
                    p.display_name,
                    p.status,
                    "person",
                    p.last_seen,
                    None if p.age_group is None else p.age_group.value,
                )
                for p in people
            ]
            rows += [
                RecordSummary(e.entity_id, e.display_name, e.status, e.kind, e.last_seen)
                for e in entities
            ]
            return rows

    def rename_person(self, person_id: str, display_name: str) -> PersonRecord:
        """Change the display name of a person; ``KeyError`` when the id is unknown."""
        with self.lock:
            record = replace(self._people[person_id], display_name=display_name)
            self._commit_person(record)
            return record

    def forget(self, person_id: str) -> bool:
        """Remove every embedding, field and file of a person, and everything attached to them.

        Order: disk first, then memory (the gallery cache is cleared even if the disk delete
        fails), then forget listeners (the matcher purges votes), then every history link. A
        failing step never skips the rest; failures are raised together afterwards as an
        ``ExceptionGroup``. Quarantined keys are purged too. Returns False if nothing was held.
        """
        errors: list[Exception] = []
        with self.lock:
            known = person_id in self._people or person_id in self.store.quarantined(PEOPLE)
            try:
                self.store.delete(PEOPLE, person_id)
            except Exception as exc:
                errors.append(exc)
            finally:
                self._people.pop(person_id, None)
                self._dirty.discard(person_id)
                self._last_refresh.pop(person_id, None)
                self._changed()
            for notify in [*self._forget_listeners, *(link.forget for link in self._history)]:
                try:
                    notify(person_id)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("forget incomplete", errors)
        return known

    def export(self, person_id: str) -> dict[str, Any]:
        """The person's whole record, embeddings and attached history, as JSON-safe data."""
        with self.lock:
            data = self._people[person_id].export()
            data["history"] = [link.export(person_id) for link in self._history]
            return data

    # ---------------------------------------------------------------- entities

    def enroll_entity(
        self,
        kind: EntityKind,
        display_name: str,
        embeddings: Vectors,
        embedding_model_id: str,
        *,
        category: str = "",
        origin: Origin = "enrolled",
        description: str | None = None,
    ) -> EntityRecord:
        """Register an animal or object with its appearance embeddings; no consent applies."""
        record = EntityRecord(
            entity_id=str(uuid.uuid4()),
            kind=check_entity_kind(kind),  # type: ignore[arg-type]
            display_name=display_name,
            category=category,
            origin=origin,
            description=description,
            embedding_model_id=embedding_model_id,
            embeddings=_matrix(embeddings),
            status="active",
        )
        self._commit_entity(record, new=True)
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
            kind=check_entity_kind(kind),  # type: ignore[arg-type]
            display_name=display_name,
            category=category,
            origin=origin,
            description=description,
        )
        self._commit_entity(record, new=True)
        return record

    def confirm_entity(
        self,
        entity_id: str,
        embeddings: Vectors,
        embedding_model_id: str,
        *,
        keep_existing: bool = False,
        category: str | None = None,
    ) -> EntityRecord:
        """Give a pending entity its appearance (the owner confirmed "this is Mia").

        With ``keep_existing`` the rows already attached (catalogue photos) are pooled with the
        new ones when they come from the same model. ``category`` fills in an empty category.
        """
        with self.lock:
            current = self._entities[entity_id]
            rows = _matrix(embeddings)
            if (
                keep_existing
                and len(current.embeddings)
                and current.embedding_model_id == embedding_model_id
            ):
                rows = np.vstack([current.embeddings, rows])
            record = replace(
                current,
                embeddings=rows,
                embedding_model_id=embedding_model_id,
                status="active",
                category=current.category or category or "",
            )
            self._commit_entity(record)
            return record

    def attach_pending_embeddings(
        self,
        entity_id: str,
        embeddings: Vectors,
        embedding_model_id: str,
        max_total: int,
        keys: Sequence[str],
    ) -> int:
        """Store appearance rows on a pending entity; it stays pending and out of matching.

        Used by the catalogue import with photos (SPEC 6). ``keys`` name where each row came
        from (hash of the photo plus the region): a row whose key is already on the record is
        skipped, so repeating an import adds nothing. Rows of another model replace the old
        ones; at ``max_total`` the oldest rows go. Returns the number of rows added.
        ``ValueError`` for an active entity or when ``keys`` does not match the rows.
        """
        with self.lock:
            current = self._entities[entity_id]
            if current.status != "pending":
                raise ValueError("entity is not pending")
            rows = _matrix(embeddings)
            if len(keys) != len(rows):
                raise ValueError("one key per row")
            same_model = (
                len(current.embeddings) > 0 and current.embedding_model_id == embedding_model_id
            )
            seen = set(current.photo_keys) if same_model else set()
            fresh: list[int] = []
            for i, key in enumerate(keys):
                if key not in seen:
                    seen.add(key)
                    fresh.append(i)
            if not fresh:
                return 0
            rows = rows[fresh]
            kept_keys = list(current.photo_keys) if same_model else []
            if same_model:
                rows = np.vstack([current.embeddings, rows])
            kept_keys = (kept_keys + [keys[i] for i in fresh])[-max_total:]
            record = replace(
                current,
                embeddings=rows[-max_total:],
                embedding_model_id=embedding_model_id,
                photo_keys=kept_keys,
            )
            self._commit_entity(record)
            return len(fresh)

    def entities(self) -> list[EntityRecord]:
        """All animals and objects, by name; the records hold embeddings, do not publish them."""
        with self.lock:
            return sorted(self._entities.values(), key=lambda r: (r.display_name, r.entity_id))

    def people_named(self, display_name: str) -> list[PersonRecord]:
        """People whose name equals ``display_name`` ignoring case."""
        with self.lock:
            key = display_name.casefold()
            return [p for p in self._people.values() if p.display_name.casefold() == key]

    def get_entity(self, entity_id: str) -> EntityRecord:
        with self.lock:
            return self._entities[entity_id]

    def rename_entity(self, entity_id: str, display_name: str) -> EntityRecord:
        with self.lock:
            record = replace(self._entities[entity_id], display_name=display_name)
            self._commit_entity(record)
            return record

    def merge_entities(self, keep_id: str, drop_id: str) -> EntityRecord:
        """Fold ``drop_id`` into ``keep_id``: embeddings pooled, newest ``last_seen`` kept.

        Both must exist, differ, and share kind and category and (when both have embeddings)
        the embedding model. The kept description wins; the dropped one fills a gap.
        """
        with self.lock:
            keep, drop = self._entities[keep_id], self._entities[drop_id]
            if keep_id == drop_id:
                raise ValueError("cannot merge an entity into itself")
            if (keep.kind, keep.category) != (drop.kind, drop.category):
                raise ValueError("cannot merge entities of different kind or category")
            seen = [t for t in (keep.last_seen, drop.last_seen) if t is not None]
            merged = replace(
                keep,
                description=keep.description or drop.description,
                last_seen=max(seen) if seen else None,
            )
            if len(drop.embeddings):
                if len(keep.embeddings) and keep.embedding_model_id != drop.embedding_model_id:
                    raise ValueError("cannot merge entities from different embedding models")
                pooled = (
                    [keep.embeddings, drop.embeddings]
                    if len(keep.embeddings)
                    else [drop.embeddings]
                )
                merged = replace(
                    merged,
                    embedding_model_id=drop.embedding_model_id,
                    embeddings=np.vstack(pooled),
                    status="active",
                )
            self._commit_entity(merged)
            self.delete_entity(drop_id)
            return merged

    def delete_entity(self, entity_id: str) -> None:
        """Remove an animal or object; attached history links forget it too, like ``forget``.

        A failing step never skips the rest; failures are raised together afterwards.
        """
        errors: list[Exception] = []
        with self.lock:
            try:
                self.store.delete(ENTITIES, entity_id)
            except Exception as exc:
                errors.append(exc)
            finally:
                self._entities.pop(entity_id, None)
                self._dirty.discard(entity_id)
                self._changed()
            for link in self._history:
                try:
                    link.forget(entity_id)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("delete incomplete", errors)

    # ----------------------------------------------------------------- matching

    def gallery(self, kind: GalleryKind, model_id: str) -> dict[str, np.ndarray]:
        """Unit-length embeddings per active owner for ``model_id``; other models are skipped."""
        with self.lock:
            cached = self._gallery_cache.get((kind, model_id))
            if cached is not None:
                return cached
            out: dict[str, np.ndarray] = {}
            if kind == "entity":
                for e in self._entities.values():
                    ok = e.status == "active" and e.embedding_model_id == model_id
                    if ok and len(e.embeddings):
                        out[e.entity_id] = _unit(e.embeddings)
            else:
                for p in self._people.values():
                    if p.status != "active":
                        continue
                    if kind == "face" and p.embedding_model_id == model_id:
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
