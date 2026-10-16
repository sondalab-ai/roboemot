"""World model (SPEC 5A, Entries, matching and fading).

- ``observe`` rotates the bearing into the base frame, then matches, first match wins: same
  identity; else same kind and category within ``match_gate_deg`` (and ``range_gate_m`` when
  both have a distance); else a new entry. With several candidates the appearance embedding
  (cosine) decides, otherwise the angularly nearest.
- Bearing-only observations of kind ``unknown`` may confirm an entry but never create one.
- ``tick`` decays confidence with a half-life per kind, slower by ``out_of_view_decay_factor``
  while the entry is out of view (decided by an injected predicate), and removes entries
  below ``forget_confidence``. Callers tick regularly; decay is applied from the entry's
  last update to ``now``.
- Events go to the bus (SPEC 11): appeared, moved, gone, and snapshot on request.

Time is passed in explicitly, so the model is deterministic and replayable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np

from ..bus import Bus
from ..config.schema import WorldConfig
from .frame import FrameSpec, angular_distance, to_base_frame, wrap_angle
from .types import UNKNOWN_IDENTITY, Entry, EntryKind, Observation, ObserveResult

InViewFn = Callable[[float, float], bool]
"""Whether a base-frame bearing (azimuth, elevation, radians) is inside the sensors' view."""


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else None


class WorldModel:
    """Egocentric map of entries; see the module docstring for the rules."""

    def __init__(
        self,
        cfg: WorldConfig,
        frame: FrameSpec | None = None,
        bus: Bus | None = None,
    ):
        self.cfg = cfg
        self.frame = frame or FrameSpec()
        self.bus = bus
        self._entries: dict[str, Entry] = {}
        self._updated: dict[str, float] = {}
        self._announced: dict[str, tuple[float, float]] = {}
        self._counter = 0

    @property
    def entries(self) -> list[Entry]:
        """Current entries (live objects; treat as read-only)."""
        return list(self._entries.values())

    def observe(
        self, obs: Observation, joint_state: Mapping[str, float] | None = None
    ) -> ObserveResult:
        """Fold one observation into the map and return what happened to it."""
        az, el = to_base_frame(obs.azimuth_rad, obs.elevation_rad, joint_state, self.frame)
        if obs.kind == "unknown":
            return self._confirm_bearing(obs, az, el)
        entry = self._match(obs, az, el)
        if entry is None:
            return ObserveResult(self._create(obs, az, el), created=True)
        self._update(entry, obs, az, el)
        return ObserveResult(entry)

    def tick(self, now: float, in_view_fn: InViewFn) -> list[Entry]:
        """Apply decay up to ``now``; remove and announce entries below the forget level."""
        gone: list[Entry] = []
        for entry in list(self._entries.values()):
            dt = max(0.0, now - self._updated[entry.entry_id])
            rate = math.log(2) / getattr(self.cfg.half_life_s, entry.kind)
            if not in_view_fn(entry.azimuth_rad, entry.elevation_rad):
                rate *= self.cfg.out_of_view_decay_factor
            entry.confidence *= math.exp(-rate * dt)
            self._updated[entry.entry_id] = now
            if entry.confidence < self.cfg.forget_confidence:
                self._remove(entry)
                gone.append(entry)
                self._publish("world.entity_gone", entry.public(), now)
        return gone

    def snapshot(self, now: float) -> list[Entry]:
        """Publish ``world.snapshot`` with all entries (no embeddings) and return them."""
        entries = [e.public() for e in self._entries.values()]
        self._publish("world.snapshot", entries, now)
        return entries

    def _angle(self, entry: Entry, az: float, el: float) -> float:
        return angular_distance(entry.azimuth_rad, entry.elevation_rad, az, el)

    def _in_gate(self, entry: Entry, obs: Observation, az: float, el: float) -> bool:
        if self._angle(entry, az, el) > math.radians(self.cfg.match_gate_deg):
            return False
        if entry.distance_m is not None and obs.distance_m is not None:
            return abs(entry.distance_m - obs.distance_m) <= self.cfg.range_gate_m
        return True

    def _match(self, obs: Observation, az: float, el: float) -> Entry | None:
        if obs.identity and obs.identity != UNKNOWN_IDENTITY:
            for entry in self._entries.values():
                if entry.identity == obs.identity:
                    return entry
        candidates = [
            e
            for e in self._entries.values()
            if e.kind == obs.kind
            and e.category == obs.category
            and e.identity in (UNKNOWN_IDENTITY, obs.identity or UNKNOWN_IDENTITY)
            and self._in_gate(e, obs, az, el)
        ]
        if not candidates:
            return None

        def rank(e: Entry) -> tuple[float, float]:
            sim = _cosine(obs.embedding, e.embedding)
            return (sim if sim is not None else -1.0, -self._angle(e, az, el))

        return max(candidates, key=rank)

    def _confirm_bearing(self, obs: Observation, az: float, el: float) -> ObserveResult:
        gate = math.radians(self.cfg.match_gate_deg)
        near = [e for e in self._entries.values() if self._angle(e, az, el) <= gate]
        if not near:
            return ObserveResult(None, unexplained_bearing=(az, el))
        entry = min(near, key=lambda e: self._angle(e, az, el))
        self._gain(entry, obs)
        return ObserveResult(entry)

    def _gain(self, entry: Entry, obs: Observation) -> None:
        gain = self.cfg.confidence_gain * obs.confidence
        entry.confidence += gain * (1.0 - entry.confidence)
        entry.last_seen = max(entry.last_seen, obs.stamp)
        entry.sources.add(obs.source)
        self._updated[entry.entry_id] = max(self._updated[entry.entry_id], obs.stamp)

    def _create(self, obs: Observation, az: float, el: float) -> Entry:
        self._counter += 1
        kind: EntryKind = obs.kind  # type: ignore[assignment]
        entry = Entry(
            entry_id=f"{kind}-{self._counter}",
            kind=kind,
            category=obs.category,
            identity=obs.identity or UNKNOWN_IDENTITY,
            azimuth_rad=az,
            elevation_rad=el,
            distance_m=obs.distance_m,
            uncertainty_rad=math.radians(self.cfg.initial_uncertainty_deg),
            confidence=self.cfg.initial_confidence * obs.confidence,
            first_seen=obs.stamp,
            last_seen=obs.stamp,
            sources={obs.source},
            embedding=None if obs.embedding is None else np.array(obs.embedding, dtype=float),
        )
        self._entries[entry.entry_id] = entry
        self._updated[entry.entry_id] = obs.stamp
        self._announced[entry.entry_id] = (az, el)
        self._publish("world.entity_appeared", entry.public(), obs.stamp)
        return entry

    def _update(self, entry: Entry, obs: Observation, az: float, el: float) -> None:
        w = self.cfg.position_blend
        offset = self._angle(entry, az, el)
        entry.azimuth_rad = wrap_angle(entry.azimuth_rad + w * wrap_angle(az - entry.azimuth_rad))
        entry.elevation_rad += w * (el - entry.elevation_rad)
        if obs.distance_m is not None:
            old = entry.distance_m
            entry.distance_m = obs.distance_m if old is None else old + w * (obs.distance_m - old)
        floor = math.radians(self.cfg.min_uncertainty_deg)
        entry.uncertainty_rad = max(floor, (1.0 - w) * entry.uncertainty_rad + w * offset)
        if obs.identity and obs.identity != UNKNOWN_IDENTITY:
            entry.identity = obs.identity
        if obs.embedding is not None:
            emb = np.asarray(obs.embedding, dtype=float)
            old_emb = entry.embedding
            entry.embedding = emb if old_emb is None else (1.0 - w) * old_emb + w * emb
        self._gain(entry, obs)
        anchor = self._announced[entry.entry_id]
        moved = angular_distance(anchor[0], anchor[1], entry.azimuth_rad, entry.elevation_rad)
        if moved > math.radians(self.cfg.moved_threshold_deg):
            self._announced[entry.entry_id] = (entry.azimuth_rad, entry.elevation_rad)
            self._publish("world.entity_moved", entry.public(), obs.stamp)

    def _remove(self, entry: Entry) -> None:
        del self._entries[entry.entry_id]
        del self._updated[entry.entry_id]
        del self._announced[entry.entry_id]

    def _publish(self, topic: str, payload: object, stamp: float) -> None:
        if self.bus is not None:
            self.bus.publish(topic, payload, stamp)
