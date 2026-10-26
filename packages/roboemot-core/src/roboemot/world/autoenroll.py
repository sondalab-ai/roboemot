"""Auto-enrollment of recurring unknown animals and objects (SPEC 5A, Known individuals).

``AutoEnroller.observe`` sees every ``percept.entity`` observation with its world result
(``feed_world``'s ``on_result``). Animals and objects only; a person observation, or anything
categorised ``person``, is refused at runtime and nothing about it is kept.

Algorithm (online, per ``(kind, category)``):

1. An observation whose embedding is close enough (cosine >= ``known_match_threshold``) to an
   active registry record of the same kind (and category, when the record has one) attaches
   that identity to its world entry; nothing is collected.
2. Otherwise it is a *sighting*, at most one per world entry per ``sighting_min_interval_s``
   and at most one per candidate cluster per the same interval (so a track seen for a minute,
   or an entry that keeps being recreated, is one sighting, not hundreds).
3. A sighting joins the nearest candidate cluster of its kind and category whose centroid is
   within ``join_radius`` (cosine distance; tighter than ``max_spread``), else it starts a
   cluster. A cluster keeps the last ``max_members`` unit embeddings, each with the local day
   it was seen on. Sightings and days are counted from the retained members only, so a member
   that rolled out takes its day with it.
4. A cluster with at least ``min_days`` days, ``min_sightings`` members and a spread
   (max over members of 1 - cosine to the centroid) <= ``max_spread`` is enrolled as
   ``origin=auto`` with a placeholder name (``cat 1``, localized), and its candidate is deleted.

Why ``join_radius`` and not the spread gate alone: two look-alikes at cosine distance 0.23 have
a spread of about 0.06 around their midpoint, so the spread gate would let them merge. Joining
only within ``join_radius`` of the current centroid keeps them in separate clusters; the spread
gate then only guards against slow drift of one cluster. The cost is slower enrollment of one
animal whose photos vary a lot (it fragments into several clusters).

Candidates persist in the registry store (collection ``candidates``: atomic, 0600, same data
folder) even without the history opt-in, as SPEC 5A says. They are bounded (``max_clusters``,
``max_members``) and deleted when enrolled, after ``retention_days`` without a sighting (checked
at load, on every ``tick`` and ``flush``), when auto-enrollment is disabled or their kind is
no longer listed, and by ``purge``. Embeddings are never logged. Embeddings are model-specific:
candidates from another model id are deleted at load. Store writes and deletes happen outside
the enroller's lock.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..config.schema import WorldAutoenrollConfig
from ..registry import Registry
from ..registry.store import RegistryStore
from .types import Observation, ObserveResult

log = logging.getLogger(__name__)

COLLECTION = "candidates"
SCHEMA_VERSION = 2
ALLOWED_KINDS = ("animal", "object")
NEVER = "person"
DISPLAY_IT: dict[str, str] = {
    "cat": "gatto",
    "dog": "cane",
    "bird": "uccello",
    "horse": "cavallo",
    "sheep": "pecora",
    "cow": "mucca",
    "cup": "tazza",
    "bottle": "bottiglia",
    "book": "libro",
    "chair": "sedia",
    "couch": "divano",
    "potted plant": "pianta",
    "bowl": "ciotola",
    "cell phone": "telefono",
    "teddy bear": "orsacchiotto",
    "laptop": "portatile",
    "remote": "telecomando",
}
NAME_CLEAN = re.compile(r"[^\w ]+|_")
LANGUAGE_SPLIT = re.compile(r"[-_]")


class _HasModelId(Protocol):
    model_id: str


class IdentityTarget(Protocol):
    """The part of ``WorldModel`` that takes an identity for an entry."""

    def set_identity(self, entry_id: str, identity: str) -> bool: ...


def display_category(category: str, language: str) -> str:
    """The category as the owner reads it: Italian word when known, else the label itself.

    The language is matched by its primary subtag (``it-IT`` and ``it_IT`` mean ``it``).
    """
    primary = LANGUAGE_SPLIT.split(language.strip().lower())[0]
    base = DISPLAY_IT.get(category, category) if primary == "it" else category
    return NAME_CLEAN.sub(" ", base).strip() or "item"


@dataclass
class _Cluster:
    """Unit embeddings and the local day (ordinal) of each, oldest first."""

    cluster_id: str
    kind: str
    category: str
    members: np.ndarray
    days: np.ndarray
    seq: int = 0

    @property
    def sightings(self) -> int:
        return len(self.members)

    @property
    def distinct_days(self) -> int:
        return len({int(d) for d in self.days})

    @property
    def last_day(self) -> int:
        return int(self.days.max())

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


def _delete_quiet(store: RegistryStore, key: str) -> None:
    """Delete a candidate; a failure is logged by type and never raised."""
    try:
        store.delete(COLLECTION, key)
    except Exception as exc:
        log.warning("candidate delete failed: %s", type(exc).__name__)


def purge_stored(store: RegistryStore) -> int:
    """Delete every stored candidate, whether or not an enroller exists; returns the count."""
    keys = store.keys(COLLECTION)
    for key in keys:
        _delete_quiet(store, key)
    return len(keys)


_Write = tuple[str, dict[str, Any], dict[str, np.ndarray]]


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
        self._doomed: list[str] = []
        self._last_seen: dict[str, float] = {}
        self._cluster_last: dict[str, float] = {}
        self._seq = 0
        self._last_tick = monotonic()
        self._load()
        self._settle()

    # ------------------------------------------------------------------ public

    def apply_config(self, cfg: WorldAutoenrollConfig) -> None:
        """Hot reload: new limits apply from the next observation; candidates the new
        configuration no longer allows (disabled, kind removed) are deleted now."""
        with self._lock:
            self.cfg = cfg
            self._expire_locked()
        self._settle()

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
        known = self._known_match(obs.kind, obs.category, row)
        if known is not None:
            self._attach(entry_id, known)
            return known
        with self._lock:
            ready = self._sight(obs, row, entry_id) if self._throttle(entry_id, obs.stamp) else None
        owner = None if ready is None else self._enroll(ready, entry_id)
        self.tick()
        return owner

    def tick(self, *, force: bool = False) -> None:
        """Periodic upkeep: retention and configuration purge, then write changed candidates.

        Cheap to call every loop; it works at most once per ``persist_interval_s`` (always with
        ``force``). Independent of observations, so a robot that sees nothing still expires.
        """
        with self._lock:
            now = self._mono()
            if force or now - self._last_tick >= self.cfg.persist_interval_s:
                self._last_tick = now
                self._expire_locked()
                writes = self._take_writes_locked()
            else:
                writes = []
        self._settle(writes)

    def flush(self) -> None:
        """Expire, then write changed candidates now (shutdown)."""
        self.tick(force=True)

    def purge(self, category: str | None = None) -> int:
        """Delete candidates (all, or one category) from memory and disk; returns the count."""
        with self._lock:
            doomed = [
                c.cluster_id for c in self._clusters.values() if category in (None, c.category)
            ]
            for cluster_id in doomed:
                self._forget_locked(cluster_id)
        self._settle()
        return len(doomed)

    def stats(self) -> dict[str, int]:
        """Counts only: candidate clusters and the sightings they hold."""
        with self._lock:
            return {
                "clusters": len(self._clusters),
                "sightings": sum(c.sightings for c in self._clusters.values()),
            }

    # ------------------------------------------------------------------ matching

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
        with contextlib.suppress(KeyError):
            self.registry.mark_seen(owner_id)

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

    # ------------------------------------------------------------------ clusters

    def _sight(self, obs: Observation, row: np.ndarray, entry_id: str) -> _Cluster | None:
        """Record one sighting; returns the cluster to enroll (already detached) if it is ripe."""
        cfg = self.cfg
        today = self._today()
        cluster = self._nearest(obs.kind, obs.category, row)
        if cluster is None:
            cluster = _Cluster(
                uuid.uuid4().hex,
                obs.kind,
                obs.category,
                row[None, :],
                np.array([today], dtype=np.int32),
            )
            self._clusters[cluster.cluster_id] = cluster
        else:
            last = self._cluster_last.get(cluster.cluster_id)
            if last is not None and obs.stamp - last < cfg.sighting_min_interval_s:
                return None
            keep = cfg.max_members
            cluster.members = np.vstack([cluster.members, row])[-keep:]
            cluster.days = np.append(cluster.days, np.int32(today))[-keep:]
        self._cluster_last[cluster.cluster_id] = obs.stamp
        self._seq += 1
        cluster.seq = self._seq
        self._dirty.add(cluster.cluster_id)
        self._cap_clusters_locked()
        ripe = (
            cluster.distinct_days >= cfg.min_days
            and cluster.sightings >= cfg.min_sightings
            and cluster.spread() <= cfg.max_spread
        )
        if not ripe:
            return None
        self._clusters.pop(cluster.cluster_id, None)
        self._dirty.discard(cluster.cluster_id)
        self._doomed.append(cluster.cluster_id)
        return cluster

    def _nearest(self, kind: str, category: str, row: np.ndarray) -> _Cluster | None:
        best, best_dist = None, self.cfg.join_radius
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
            with self._lock:  # keep collecting; the file is rewritten at the next tick
                self._clusters[cluster.cluster_id] = cluster
                self._dirty.add(cluster.cluster_id)
            return None
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

    def _cap_clusters_locked(self) -> None:
        extra = len(self._clusters) - self.cfg.max_clusters
        if extra > 0:
            for cluster in sorted(self._clusters.values(), key=lambda c: c.seq)[:extra]:
                self._forget_locked(cluster.cluster_id)

    def _expire_locked(self) -> None:
        """Forget clusters past retention, or that the configuration no longer allows."""
        cfg = self.cfg
        limit = self._today() - cfg.retention_days
        for cluster in list(self._clusters.values()):
            stale = cluster.last_day < limit
            if stale or not cfg.enabled or cluster.kind not in cfg.kinds:
                self._forget_locked(cluster.cluster_id)

    def _forget_locked(self, cluster_id: str) -> None:
        """Drop a cluster from memory now; its file is deleted by the next ``_settle``."""
        self._clusters.pop(cluster_id, None)
        self._dirty.discard(cluster_id)
        self._cluster_last.pop(cluster_id, None)
        self._doomed.append(cluster_id)

    # ------------------------------------------------------------------ storage

    def _take_writes_locked(self) -> list[_Write]:
        """Snapshot the changed clusters (copies), so the writes can run outside the lock."""
        writes: list[_Write] = []
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
            }
            arrays = {"members": cluster.members.copy(), "days": cluster.days.copy()}
            writes.append((cluster_id, meta, arrays))
        self._dirty.clear()
        return writes

    def _settle(self, writes: list[_Write] | None = None) -> None:
        """Run the pending deletes and the given writes; no lock is held during store I/O."""
        with self._lock:
            doomed, self._doomed = self._doomed, []
        for cluster_id in doomed:
            _delete_quiet(self.store, cluster_id)
        for cluster_id, meta, arrays in writes or []:
            try:
                self.store.put(COLLECTION, cluster_id, meta, arrays)
            except Exception as exc:
                log.warning("candidate write failed: %s", type(exc).__name__)
                with self._lock:
                    if cluster_id in self._clusters:
                        self._dirty.add(cluster_id)

    def _load(self) -> None:
        cfg = self.cfg
        limit = self._today() - cfg.retention_days
        loaded: list[_Cluster] = []
        for key in self.store.keys(COLLECTION):
            try:
                meta, arrays = self.store.get(COLLECTION, key)
                members = np.asarray(arrays["members"], dtype=np.float32)
                days = np.asarray(arrays["days"], dtype=np.int32)
                valid = (
                    meta["schema_version"] == SCHEMA_VERSION
                    and meta["model_id"] == self.model_id
                    and meta["cluster_id"] == key
                    and meta["kind"] in ALLOWED_KINDS
                    and members.ndim == 2
                    and len(members)
                    and days.shape == (len(members),)
                    and np.isfinite(members).all()
                )
                if not valid:
                    raise ValueError("stale candidate")
                members = members / np.linalg.norm(members, axis=1, keepdims=True)
                cluster = _Cluster(key, meta["kind"], str(meta["category"]), members, days)
                expired = cluster.last_day < limit
                if expired or not cfg.enabled or cluster.kind not in cfg.kinds:
                    raise ValueError("expired or not allowed")
                loaded.append(cluster)
            except Exception as exc:
                log.info("candidate %s dropped at load (%s)", key, type(exc).__name__)
                self._doomed.append(key)
        for cluster in sorted(loaded, key=lambda c: c.last_day):
            self._seq += 1
            cluster.seq = self._seq
            self._clusters[cluster.cluster_id] = cluster


def open_autoenroll(
    registry: Registry | None,
    appearance: _HasModelId | None,
    cfg: WorldAutoenrollConfig,
    *,
    world: IdentityTarget | None = None,
    language: Callable[[], str] = lambda: "en",
) -> AutoEnroller | None:
    """The app's auto-enroller, or None when there is nothing to enroll with.

    It needs a registry (its store holds the candidates) and the appearance embedder (its
    ``model_id`` keys the candidates). Without the embedder, or when auto-enrollment is
    disabled, any candidates left on disk are deleted at startup: nothing may keep them. A
    disabled enroller is still built (it observes nothing and keeps nothing), so turning
    ``world.autoenroll.enabled`` on is a hot reload, not a restart (SPEC 11A).
    """
    if registry is None:
        return None
    if appearance is None or not cfg.enabled:
        purge_stored(registry.store)
    if appearance is None:
        return None
    return AutoEnroller(
        registry, registry.store, cfg, appearance.model_id, world=world, language=language
    )
