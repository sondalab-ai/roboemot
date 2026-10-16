"""Auto-enrollment of recurring unknown animals and objects (SPEC 5A, Known individuals).

``AutoEnroller.observe`` sees every ``percept.entity`` observation with its world result
(``feed_world``'s ``on_result``). Animals and objects only; a person observation, or anything
categorised ``person``, is refused at runtime and nothing about it is kept.

Algorithm (online, per ``(kind, category)``):

1. An observation whose embedding is close enough (cosine >= ``known_match_threshold``) to an
   active registry record of the same kind (and category, when the record has one) attaches
   that identity to its world entry; nothing is collected.
2. Otherwise it is a *sighting*, at most one per world entry per ``sighting_min_interval_s``
   (so a track seen for a minute is one sighting, not hundreds).
3. A sighting joins the nearest candidate cluster of its kind and category whose centroid is
   within ``max_spread`` (cosine distance), else it starts a cluster. A cluster keeps the last
   ``max_members`` unit embeddings, the number of sightings, and the set of distinct local
   days it was seen on.
4. A cluster with at least ``min_days`` days, ``min_sightings`` sightings and a spread
   (max over retained members of 1 - cosine to the centroid) <= ``max_spread`` is enrolled as
   ``origin=auto`` with a placeholder name (``cat 1``, localized), and its candidate is deleted.
   A cluster that mixes two look-alikes spreads past the gate and never enrolls: the owner
   sees nothing, which is the safe failure.

Candidates persist in the registry store (collection ``candidates``: atomic, 0600, same data
folder) even without the history opt-in, as SPEC 5A says. They are bounded (``max_clusters``,
``max_members``), purged when enrolled and after ``retention_days`` without a sighting, and
removed by ``purge``. Embeddings are never logged. Embeddings are model-specific: candidates
from another model id are deleted at load.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..config.schema import WorldAutoenrollConfig
from ..registry import Registry
from ..registry.store import RegistryStore
from .types import Observation, ObserveResult

log = logging.getLogger(__name__)

COLLECTION = "candidates"
SCHEMA_VERSION = 1
ALLOWED_KINDS = ("animal", "object")
NEVER = "person"
DISPLAY_IT: dict[str, str] = {
    "cat": "gatto",
    "dog": "cane",
    "bird": "uccello",
    "horse": "cavallo",
    "cup": "tazza",
    "bottle": "bottiglia",
    "book": "libro",
    "chair": "sedia",
    "couch": "divano",
    "plant": "pianta",
    "clock": "orologio",
    "vase": "vaso",
    "glass": "bicchiere",
    "teddy": "orsacchiotto",
    "bear": "orso",
    "phone": "telefono",
    "laptop": "portatile",
}
NAME_CLEAN = re.compile(r"[^\w ]+|_")


class IdentityTarget(Protocol):
    """The part of ``WorldModel`` that takes an identity for an entry."""

    def set_identity(self, entry_id: str, identity: str) -> bool: ...


def display_category(category: str, language: str) -> str:
    """The category as the owner reads it: Italian word when known, else the label itself."""
    base = DISPLAY_IT.get(category, category) if language == "it" else category
    return NAME_CLEAN.sub(" ", base).strip() or "item"


@dataclass
class _Cluster:
    cluster_id: str
    kind: str
    category: str
    members: np.ndarray
    days: set[int] = field(default_factory=set)
    sightings: int = 0
    last_day: int = 0
    seq: int = 0

    def centroid(self) -> np.ndarray:
        mean = self.members.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def spread(self) -> float:
        return float(1.0 - (self.members @ self.centroid()).min())


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    row = np.asarray(vector, dtype=np.float32)
    if row.ndim != 1 or not row.size or not np.isfinite(row).all():
        return None
    norm = float(np.linalg.norm(row))
    return None if norm == 0 else row / norm


class AutoEnroller:
    """Collects sightings of unknown animals and objects and enrolls recurring ones."""

    def __init__(
        self,
        registry: Registry,
        store: RegistryStore,
        cfg: WorldAutoenrollConfig,
        model_id: str,
        *,
        world: IdentityTarget | None = None,
        language: Callable[[], str] = lambda: "en",
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.store = store
        self.cfg = cfg
        self.model_id = model_id
        self.world = world
        self._language = language
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._mono = monotonic
        self._lock = threading.RLock()
        self._clusters: dict[str, _Cluster] = {}
        self._dirty: set[str] = set()
        self._last_seen: dict[str, float] = {}
        self._seq = 0
        self._last_flush = monotonic()
        self._load()

    # ------------------------------------------------------------------ public

    def apply_config(self, cfg: WorldAutoenrollConfig) -> None:
        """Hot reload: the new limits apply from the next observation."""
        with self._lock:
            self.cfg = cfg

    def observe(self, obs: Observation, result: ObserveResult) -> str | None:
        """Fold one observation in; returns the registry id attached or enrolled, if any.

        Never raises for bad input: unusable observations are ignored.
        """
        cfg = self.cfg
        if not cfg.enabled or obs.kind == NEVER or obs.category == NEVER:
            return None
        if obs.kind not in ALLOWED_KINDS or obs.kind not in cfg.kinds:
            return None
        row = _unit(obs.embedding)
        if row is None or result.entry is None or not obs.category:
            return None
        if obs.identity not in (None, "", "unknown"):
            return None
        entry_id = result.entry.entry_id
        with self._lock:
            known = self._known_match(obs.kind, obs.category, row)
            if known is not None:
                self._attach(entry_id, known)
                return known
            if not self._throttle(entry_id, obs.stamp):
                return None
            enrolled = self._sight(obs.kind, obs.category, row, entry_id)
            self._maintain()
            return enrolled

    def flush(self) -> None:
        """Write changed candidates now (shutdown)."""
        with self._lock:
            self._write_dirty()

    def purge(self, category: str | None = None) -> int:
        """Delete candidates (all, or one category) from memory and disk; returns the count."""
        with self._lock:
            doomed = [c for c in self._clusters.values() if category in (None, c.category)]
            for cluster in doomed:
                self._drop(cluster.cluster_id)
            return len(doomed)

    def stats(self) -> dict[str, int]:
        """Counts only: candidate clusters and the sightings they hold."""
        with self._lock:
            return {
                "clusters": len(self._clusters),
                "sightings": sum(c.sightings for c in self._clusters.values()),
            }

    # ------------------------------------------------------------------ internals

    def _today(self) -> int:
        return self._clock().astimezone().date().toordinal()

    def _known_match(self, kind: str, category: str, row: np.ndarray) -> str | None:
        best_id, best = None, self.cfg.known_match_threshold
        for owner, rows in self.registry.gallery("entity", self.model_id).items():
            if rows.shape[1] != row.shape[0]:
                continue
            sim = float((rows @ row).max())
            if sim < best:
                continue
            try:
                record = self.registry.get_entity(owner)
            except KeyError:
                continue
            if record.kind == kind and record.category in ("", category):
                best_id, best = owner, sim
        return best_id

    def _attach(self, entry_id: str, owner_id: str) -> None:
        if self.world is not None:
            self.world.set_identity(entry_id, owner_id)
        try:
            self.registry.mark_seen(owner_id)
        except KeyError:
            pass

    def _throttle(self, entry_id: str, stamp: float) -> bool:
        last = self._last_seen.get(entry_id)
        if last is not None and stamp - last < self.cfg.sighting_min_interval_s:
            return False
        self._last_seen[entry_id] = stamp
        if len(self._last_seen) > self.cfg.max_tracked_entries:
            oldest = sorted(self._last_seen, key=self._last_seen.__getitem__)
            for key in oldest[: len(oldest) - self.cfg.max_tracked_entries]:
                del self._last_seen[key]
        return True

    def _sight(self, kind: str, category: str, row: np.ndarray, entry_id: str) -> str | None:
        cfg = self.cfg
        today = self._today()
        cluster = self._nearest(kind, category, row)
        if cluster is None:
            cluster = _Cluster(uuid.uuid4().hex, kind, category, row[None, :])
            self._clusters[cluster.cluster_id] = cluster
        else:
            cluster.members = np.vstack([cluster.members, row])[-cfg.max_members :]
        cluster.sightings += 1
        cluster.days.add(today)
        cluster.last_day = today
        self._seq += 1
        cluster.seq = self._seq
        self._dirty.add(cluster.cluster_id)
        self._cap_clusters()
        if (
            len(cluster.days) >= cfg.min_days
            and cluster.sightings >= cfg.min_sightings
            and cluster.spread() <= cfg.max_spread
        ):
            return self._enroll(cluster, entry_id)
        return None

    def _nearest(self, kind: str, category: str, row: np.ndarray) -> _Cluster | None:
        best, best_dist = None, self.cfg.max_spread
        for cluster in self._clusters.values():
            if (cluster.kind, cluster.category) != (kind, category):
                continue
            if cluster.members.shape[1] != row.shape[0]:
                continue
            dist = float(1.0 - cluster.centroid() @ row)
            if dist <= best_dist:
                best, best_dist = cluster, dist
        return best

    def _enroll(self, cluster: _Cluster, entry_id: str) -> str | None:
        if cluster.kind == NEVER or cluster.kind not in ALLOWED_KINDS:
            raise ValueError("people are never auto-enrolled")
        try:
            record = self.registry.enroll_entity(
                cluster.kind,  # type: ignore[arg-type]
                self._placeholder(cluster),
                cluster.members,
                self.model_id,
                category=cluster.category,
                origin="auto",
            )
        except Exception as exc:
            log.warning("auto-enrollment failed: %s", type(exc).__name__)
            return None
        self._drop(cluster.cluster_id)
        self._attach(entry_id, record.entity_id)
        log.info("auto-enrolled a new %s", cluster.kind)
        return record.entity_id

    def _placeholder(self, cluster: _Cluster) -> str:
        """``<category> <n>`` with n one above the highest number any auto record of that
        name already has (a count would collide after a delete)."""
        base = display_category(cluster.category, self._language())
        pattern = re.compile(rf"{re.escape(base)} (\d+)")
        taken = [
            int(m.group(1))
            for r in self.registry.entities()
            if r.origin == "auto" and (m := pattern.fullmatch(r.display_name))
        ]
        return f"{base} {max(taken, default=0) + 1}"

    def _cap_clusters(self) -> None:
        extra = len(self._clusters) - self.cfg.max_clusters
        if extra > 0:
            for cluster in sorted(self._clusters.values(), key=lambda c: c.seq)[:extra]:
                self._drop(cluster.cluster_id)

    def _maintain(self) -> None:
        """Purge expired candidates and write changes, at most once per persist interval."""
        now = self._mono()
        if now - self._last_flush < self.cfg.persist_interval_s:
            return
        limit = self._today() - self.cfg.retention_days
        for cluster in [c for c in self._clusters.values() if c.last_day < limit]:
            self._drop(cluster.cluster_id)
        self._write_dirty()

    def _drop(self, cluster_id: str) -> None:
        self._clusters.pop(cluster_id, None)
        self._dirty.discard(cluster_id)
        try:
            self.store.delete(COLLECTION, cluster_id)
        except Exception as exc:
            log.warning("candidate delete failed: %s", type(exc).__name__)

    def _write_dirty(self) -> None:
        for cluster_id in sorted(self._dirty):
            cluster = self._clusters.get(cluster_id)
            if cluster is None:
                continue
            meta: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "cluster_id": cluster_id,
                "kind": cluster.kind,
                "category": cluster.category,
                "model_id": self.model_id,
                "sightings": cluster.sightings,
                "days": sorted(cluster.days),
                "last_day": cluster.last_day,
            }
            try:
                self.store.put(COLLECTION, cluster_id, meta, {"members": cluster.members})
                self._dirty.discard(cluster_id)
            except Exception as exc:
                log.warning("candidate write failed: %s", type(exc).__name__)
        self._last_flush = self._mono()

    def _load(self) -> None:
        loaded: list[_Cluster] = []
        for key in self.store.keys(COLLECTION):
            try:
                meta, arrays = self.store.get(COLLECTION, key)
                members = np.asarray(arrays["members"], dtype=np.float32)
                valid = (
                    meta["schema_version"] == SCHEMA_VERSION
                    and meta["model_id"] == self.model_id
                    and meta["cluster_id"] == key
                    and meta["kind"] in ALLOWED_KINDS
                    and members.ndim == 2
                    and len(members)
                    and np.isfinite(members).all()
                )
                if not valid:
                    raise ValueError("stale candidate")
                members = members / np.linalg.norm(members, axis=1, keepdims=True)
                loaded.append(
                    _Cluster(
                        key,
                        meta["kind"],
                        str(meta["category"]),
                        members,
                        {int(d) for d in meta["days"]},
                        int(meta["sightings"]),
                        int(meta["last_day"]),
                    )
                )
            except Exception as exc:
                log.info("candidate %s dropped at load (%s)", key, type(exc).__name__)
                self.store.delete(COLLECTION, key)
        for cluster in sorted(loaded, key=lambda c: c.last_day):
            self._seq += 1
            cluster.seq = self._seq
            self._clusters[cluster.cluster_id] = cluster
