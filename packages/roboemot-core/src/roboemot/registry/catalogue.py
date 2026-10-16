"""Catalogue import and confirmation of pending records (SPEC 6, Catalogue import; SPEC 5A).

``CatalogueService`` is what the web UI talks to for the catalogue:

- ``import_sidecar`` parses one XMP sidecar (and optionally its photo) and creates *pending*
  records. People are name-only and never get biometric data from an import: their regions are
  not even cropped. Animals and objects get their appearance from the photo's tagged regions
  (embedded in memory, the image is dropped at once) but stay pending, outside matching, until
  the owner confirms.
- ``confirm`` is "this is Mia": the owner picks a world entry that is currently seen and its
  appearance becomes the record's (pooled with the photo rows), activating it. A record that
  already has photo rows can be activated without an entry.
- ``list_entities`` and ``live_entries`` give the UI summaries: never embeddings or images.

Photos and sidecars are untrusted input with size, pixel and count limits (``catalogue.*``).
Nothing is written to disk except the registry records; no image is ever stored or logged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..config.schema import CatalogueConfig
from .images import ImageError, decode_bgr
from .records import EntityRecord
from .registry import Registry
from .xmp import CatalogueItem, Conflict, XmpError, parse_xmp

log = logging.getLogger(__name__)

ASPECT_TOLERANCE = 0.02
PHOTO_NONE = "none"
PHOTO_USED = "used"
PHOTO_NO_EMBEDDER = "no_embedder"
PHOTO_DIMS_MISMATCH = "dims_mismatch"
LIVE_KINDS = ("animal", "object")


class CatalogueError(Exception):
    """A catalogue request cannot be done; ``code`` is stable and safe to show."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class AppearanceSource(Protocol):
    """The appearance embedder (DINOv2); ``embed`` is serialised by the embedder itself."""

    model_id: str

    def embed(self, image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray: ...


class LiveWorld(Protocol):
    """The part of ``WorldModel`` the catalogue needs."""

    def get(self, entry_id: str) -> Any: ...

    @property
    def entries(self) -> list[Any]: ...

    def embedding_of(self, entry_id: str) -> np.ndarray | None: ...

    def set_identity(self, entry_id: str, identity: str) -> bool: ...


@dataclass(frozen=True)
class ReportItem:
    """One record touched by an import. ``embeddings`` counts rows attached from the photo."""

    id: str
    display_name: str
    kind: str
    category: str
    created: bool
    embeddings: int = 0


@dataclass
class ImportReport:
    """What an import did; names, ids and counts only."""

    items: list[ReportItem] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    photo: str = PHOTO_NONE
    truncated: bool = False


@dataclass(frozen=True)
class EntitySummary:
    """A pending or auto-enrolled animal or object, without its embeddings."""

    id: str
    display_name: str
    kind: str
    category: str
    status: str
    origin: str
    has_appearance: bool
    last_seen: datetime | None


@dataclass(frozen=True)
class LiveSummary:
    """A world entry that could be confirmed as a record."""

    entry_id: str
    kind: str
    category: str
    age_s: float
    identified: bool


class CatalogueService:
    """Import, list and confirm catalogue records over a registry and the live world."""

    def __init__(
        self,
        registry: Registry,
        cfg: CatalogueConfig,
        *,
        embedder: Callable[[], AppearanceSource | None] = lambda: None,
        world: LiveWorld | None = None,
        validate_name: Callable[[str], str] | None = None,
        now: Callable[[], float] | None = None,
        purge_candidates: Callable[[], int] | None = None,
    ) -> None:
        self.registry = registry
        self.cfg = cfg
        self._embedder = embedder
        self.world = world
        self._validate = validate_name
        self._now = now
        self._purge = purge_candidates

    # ------------------------------------------------------------------ import

    def import_sidecar(self, xmp: bytes | str, image: bytes | None = None) -> ImportReport:
        """Create pending records from one sidecar and, with ``image``, attach appearances.

        ``CatalogueError`` for an unreadable sidecar or photo (nothing is created then).
        Records that already exist (same name and kind, category equal or unset) are reported
        as not created and left alone, except that a pending one still receives photo rows.
        """
        cfg = self.cfg
        kwargs: dict[str, Any] = {
            "max_bytes": cfg.max_xmp_bytes,
            "max_items": cfg.max_items,
            "max_regions": cfg.max_regions,
            "max_pixels": cfg.max_image_pixels,
        }
        if self._validate is not None:
            kwargs["validate_name"] = self._validate
        try:
            parsed = parse_xmp(xmp, **kwargs)
        except XmpError as exc:
            raise CatalogueError(f"xmp_{exc.code}") from exc
        report = ImportReport(
            conflicts=parsed.conflicts, skipped=dict(parsed.skipped), truncated=parsed.truncated
        )
        decoded = self._decode(image, parsed.applied_dims, report) if image is not None else None
        try:
            for item in parsed.items:
                report.items.append(self._import_item(item, decoded, report))
        finally:
            del decoded
        return report

    def _decode(
        self, image: bytes, applied: tuple[int, int] | None, report: ImportReport
    ) -> np.ndarray | None:
        try:
            pixels = decode_bgr(
                image, max_bytes=self.cfg.max_image_bytes, max_pixels=self.cfg.max_image_pixels
            )
        except ImageError as exc:
            raise CatalogueError(exc.code) from exc
        if self._embedder() is None:
            report.photo = PHOTO_NO_EMBEDDER
            return None
        height, width = pixels.shape[:2]
        if applied is not None and abs(applied[0] / applied[1] - width / height) > (
            ASPECT_TOLERANCE * width / height
        ):
            report.photo = PHOTO_DIMS_MISMATCH
            return None
        report.photo = PHOTO_USED
        return pixels

    def _find(self, item: CatalogueItem) -> EntityRecord | None:
        for rec in self.registry.entities():
            same = rec.kind == item.kind and rec.display_name.casefold() == item.name.casefold()
            if same and (not rec.category or not item.category or rec.category == item.category):
                return rec
        return None

    def _import_item(
        self, item: CatalogueItem, pixels: np.ndarray | None, report: ImportReport
    ) -> ReportItem:
        if item.kind == "person":
            known = self.registry.people_named(item.name)
            person = known[0] if known else self.registry.create_pending_person(item.name)
            return ReportItem(person.person_id, person.display_name, "person", "", not known)
        record = self._find(item)
        created = record is None
        if record is None:
            record = self.registry.create_pending_entity(
                item.kind,  # type: ignore[arg-type]
                item.name,
                category=item.category,
                origin="xmp",
            )
        attached = 0
        if pixels is not None and record.status == "pending" and item.regions:
            attached = self._attach(record.entity_id, pixels, item, report)
        return ReportItem(
            record.entity_id, record.display_name, record.kind, record.category, created, attached
        )

    def _attach(
        self, entity_id: str, pixels: np.ndarray, item: CatalogueItem, report: ImportReport
    ) -> int:
        embedder = self._embedder()
        if embedder is None:
            return 0
        height, width = pixels.shape[:2]
        rows: list[np.ndarray] = []
        for region in item.regions:
            x1, x2 = (region.cx - region.w / 2) * width, (region.cx + region.w / 2) * width
            y1, y2 = (region.cy - region.h / 2) * height, (region.cy + region.h / 2) * height
            box = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            if min(box[2] - box[0], box[3] - box[1]) < self.cfg.min_region_px:
                report.skipped["region_small"] = report.skipped.get("region_small", 0) + 1
                continue
            try:
                rows.append(np.asarray(embedder.embed(pixels, box), dtype=np.float32))
            except Exception as exc:
                log.warning("catalogue region embedding failed: %s", type(exc).__name__)
                report.skipped["embed_failed"] = report.skipped.get("embed_failed", 0) + 1
        if not rows:
            return 0
        try:
            self.registry.attach_pending_embeddings(
                entity_id, np.vstack(rows), embedder.model_id, self.cfg.max_embeddings_per_entity
            )
        except ValueError:
            report.skipped["embed_failed"] = report.skipped.get("embed_failed", 0) + 1
            return 0
        finally:
            rows.clear()
        return min(len(item.regions), self.cfg.max_embeddings_per_entity)

    # ------------------------------------------------------------------ listing

    def list_entities(self) -> list[EntitySummary]:
        """Pending and auto-enrolled animals and objects: the ones the owner has to look at."""
        return [
            EntitySummary(
                r.entity_id,
                r.display_name,
                r.kind,
                r.category,
                r.status,
                r.origin,
                len(r.embeddings) > 0,
                r.last_seen,
            )
            for r in self.registry.entities()
            if r.status == "pending" or r.origin == "auto"
        ]

    def live_entries(self) -> list[LiveSummary]:
        """Animals and objects the robot currently has in its map (no embeddings)."""
        if self.world is None:
            return []
        now = self._now() if self._now is not None else None
        out = []
        for e in self.world.entries:
            if e.kind in LIVE_KINDS:
                age = 0.0 if now is None else max(0.0, now - e.last_seen)
                out.append(
                    LiveSummary(
                        e.entry_id, e.kind, e.category, age, e.identity not in ("", "unknown")
                    )
                )
        return out

    def purge_candidates(self) -> int:
        """Delete the auto-enrollment candidates (appearance clusters); returns how many."""
        return 0 if self._purge is None else self._purge()

    # ------------------------------------------------------------------ confirmation

    def confirm(self, record_id: str, entry_id: str | None = None) -> EntityRecord:
        """Activate a pending record with what the robot sees ("this is Mia").

        Raises ``KeyError`` for an unknown record and ``CatalogueError`` with ``not_pending``,
        ``entry_gone``, ``kind_mismatch``, ``category_mismatch`` or ``no_appearance``. Without
        ``entry_id`` the record must already hold photo rows.
        """
        record = self.registry.get_entity(record_id)
        if record.status != "pending":
            raise CatalogueError("not_pending")
        if entry_id is None:
            if not len(record.embeddings) or record.embedding_model_id is None:
                raise CatalogueError("no_appearance")
            return self.registry.confirm_entity(
                record_id, record.embeddings, record.embedding_model_id
            )
        embedder = self._embedder()
        entry = None if self.world is None else self.world.get(entry_id)
        if entry is None:
            raise CatalogueError("entry_gone")
        if entry.kind != record.kind:
            raise CatalogueError("kind_mismatch")
        if record.category and entry.category != record.category:
            raise CatalogueError("category_mismatch")
        assert self.world is not None
        row = self.world.embedding_of(entry_id)
        if embedder is None or row is None:
            raise CatalogueError("no_appearance")
        try:
            confirmed = self.registry.confirm_entity(
                record_id,
                np.asarray(row, dtype=np.float32)[None, :],
                embedder.model_id,
                keep_existing=True,
                category=entry.category,
            )
        except ValueError as exc:
            raise CatalogueError("no_appearance") from exc
        self.world.set_identity(entry_id, record_id)
        return confirmed
