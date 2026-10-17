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

Imports run one at a time (a second one is refused as ``import_busy``), the photo is decoded
only when it can be used (an embedder exists and a sidecar item has regions), and embedding
stops once ``catalogue.import_budget_s`` is spent: the records are still created, the report
says what was skipped, and repeating the import finishes the rest. Each photo row carries a key
(hash of the photo bytes and the region), so repeating an import never adds a row twice.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
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
KEY_HASH_CHARS = 16
PHOTO_NONE = "none"
PHOTO_USED = "used"
PHOTO_PARTIAL = "partial"
PHOTO_NO_REGIONS = "no_regions"
PHOTO_NO_EMBEDDER = "no_embedder"
PHOTO_DIMS_MISMATCH = "dims_mismatch"
LIVE_KINDS = ("animal", "object")
SKIP_DELETED = "deleted"


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

    def skip(self, code: str) -> None:
        self.skipped[code] = self.skipped.get(code, 0) + 1


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


@dataclass
class _Photo:
    """A decoded photo and the key prefix of its rows, alive for one import only."""

    pixels: np.ndarray
    key: str
    deadline: float


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
        clock: Callable[[], float] = time.monotonic,
        budget_cap: Callable[[], float] = lambda: float("inf"),
    ) -> None:
        self.registry = registry
        self.cfg = cfg
        self._embedder = embedder
        self.world = world
        self._validate = validate_name
        self._now = now
        self._purge = purge_candidates
        self._clock = clock
        self._budget_cap = budget_cap
        self._busy = threading.Lock()

    # ------------------------------------------------------------------ import

    def import_sidecar(self, xmp: bytes | str, image: bytes | None = None) -> ImportReport:
        """Create pending records from one sidecar and, with ``image``, attach appearances.

        ``CatalogueError`` for an unreadable sidecar or photo (nothing is created then) and
        ``import_busy`` while another import runs. Records that already exist (same name and
        kind, category equal or unset) are reported as not created and left alone, except that
        a pending one still receives photo rows it does not have yet.
        """
        if not self._busy.acquire(blocking=False):
            raise CatalogueError("import_busy")
        try:
            return self._import(xmp, image)
        finally:
            self._busy.release()

    def _import(self, xmp: bytes | str, image: bytes | None) -> ImportReport:
        cfg = self.cfg
        deadline = self._clock() + min(cfg.import_budget_s, self._budget_cap())
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
        photo = None
        if image is not None:
            photo = self._open_photo(image, parsed.items, parsed.applied_dims, report, deadline)
        try:
            for item in parsed.items:
                done = self._import_item(item, photo, report)
                if done is not None:
                    report.items.append(done)
        finally:
            photo = None
        return report

    def _open_photo(
        self,
        image: bytes,
        items: list[CatalogueItem],
        applied: tuple[int, int] | None,
        report: ImportReport,
        deadline: float,
    ) -> _Photo | None:
        """Decode the photo only if something can use it; refuse a bad one before any write."""
        if not any(i.kind != "person" and i.regions for i in items):
            report.photo = PHOTO_NO_REGIONS
            return None
        if self._embedder() is None:
            report.photo = PHOTO_NO_EMBEDDER
            return None
        try:
            pixels = decode_bgr(
                image, max_bytes=self.cfg.max_image_bytes, max_pixels=self.cfg.max_image_pixels
            )
        except ImageError as exc:
            if exc.code == "image_unsupported":  # no OpenCV here: same as no recognition
                report.photo = PHOTO_NO_EMBEDDER
                return None
            raise CatalogueError(exc.code) from exc
        height, width = pixels.shape[:2]
        if applied is not None and abs(applied[0] / applied[1] - width / height) > (
            ASPECT_TOLERANCE * width / height
        ):
            report.photo = PHOTO_DIMS_MISMATCH
            return None
        report.photo = PHOTO_USED
        key = hashlib.sha256(image).hexdigest()[:KEY_HASH_CHARS]
        return _Photo(pixels, key, deadline)

    def _find(self, item: CatalogueItem) -> EntityRecord | None:
        for rec in self.registry.entities():
            same = rec.kind == item.kind and rec.display_name.casefold() == item.name.casefold()
            if same and (not rec.category or not item.category or rec.category == item.category):
                return rec
        return None

    def _import_item(
        self, item: CatalogueItem, photo: _Photo | None, report: ImportReport
    ) -> ReportItem | None:
        """One item of the sidecar; None when the owner deleted this pending animal or object."""
        if item.kind == "person":
            known = self.registry.people_named(item.name)
            person = known[0] if known else self.registry.create_pending_person(item.name)
            return ReportItem(person.person_id, person.display_name, "person", "", not known)
        record = self._find(item)
        created = record is None
        if record is None and self.registry.tombstones.has(
            item.kind, item.category or "", item.name
        ):
            report.skip(SKIP_DELETED)
            return None
        if record is None:
            record = self.registry.create_pending_entity(
                item.kind,  # type: ignore[arg-type]
                item.name,
                category=item.category,
                origin="xmp",
            )
        attached = 0
        if photo is not None and record.status == "pending" and item.regions:
            attached = self._attach(record, photo, item, report)
        return ReportItem(
            record.entity_id, record.display_name, record.kind, record.category, created, attached
        )

    def _attach(
        self, record: EntityRecord, photo: _Photo, item: CatalogueItem, report: ImportReport
    ) -> int:
        """Embed the item's regions not yet on the record; returns the rows actually added."""
        embedder = self._embedder()
        if embedder is None:
            return 0
        height, width = photo.pixels.shape[:2]
        have = set(record.photo_keys) if record.embedding_model_id == embedder.model_id else set()
        rows: list[np.ndarray] = []
        keys: list[str] = []
        for region in item.regions:
            key = f"{photo.key}:{region.cx:.4f},{region.cy:.4f},{region.w:.4f},{region.h:.4f}"
            if key in have or key in keys:
                continue
            if len(rows) >= self.cfg.max_embeddings_per_entity:
                report.skip("cap")
                break
            x1, x2 = (region.cx - region.w / 2) * width, (region.cx + region.w / 2) * width
            y1, y2 = (region.cy - region.h / 2) * height, (region.cy + region.h / 2) * height
            left, top = max(0.0, x1), max(0.0, y1)
            right, bottom = min(float(width), x2), min(float(height), y2)
            if min(right - left, bottom - top) < self.cfg.min_region_px:
                report.skip("region_small")
                continue
            if self._clock() > photo.deadline:
                report.skip("budget")
                report.photo = PHOTO_PARTIAL
                continue
            try:
                rows.append(
                    np.asarray(embedder.embed(photo.pixels, (left, top, right, bottom)), np.float32)
                )
                keys.append(key)
            except Exception as exc:
                log.warning("catalogue region embedding failed: %s", type(exc).__name__)
                report.skip("embed_failed")
        if not rows:
            return 0
        try:
            return self.registry.attach_pending_embeddings(
                record.entity_id,
                np.vstack(rows),
                embedder.model_id,
                self.cfg.max_embeddings_per_entity,
                keys,
            )
        except ValueError:
            report.skip("embed_failed")
            return 0
        finally:
            rows.clear()

    # ------------------------------------------------------------------ tombstones

    def tombstone_count(self) -> int:
        """How many deleted pending records an import skips (names are not kept)."""
        return self.registry.tombstones.count()

    def clear_tombstones(self) -> int:
        """Let an import bring deleted pending records back; returns how many were cleared."""
        return self.registry.tombstones.clear()

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
