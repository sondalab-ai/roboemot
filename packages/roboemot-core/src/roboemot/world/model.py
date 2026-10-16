"""World model (SPEC 5A, Entries, matching and fading).

- ``observe`` rotates the bearing into the base frame, then matches, first match wins: same
  identity; else same kind and category within ``match_gate_deg`` (and ``range_gate_m`` when
  both have a distance), skipping entries with a conflicting known identity; else a new
  entry. The nearest candidate wins; candidates within ``tie_tolerance_deg`` of it are
  ranked by appearance (cosine; a missing similarity is neutral).
- Bearing-only observations of kind ``unknown`` may confirm an entry but never create one.
- Decay: confidence halves every ``half_life_s`` of the entry's kind, slower by
  ``out_of_view_decay_factor`` while out of view. ``tick`` decides in-view with the injected
  predicate and remembers the result per entry; ``observe`` first decays the gap since the
  entry's last update with that remembered state, then gains, and marks the entry in view.
  Entries below ``forget_confidence`` are removed by ``tick``.
- Events go to the bus (SPEC 11): appeared, moved, gone, snapshot on request. They are
  published after the lock is released; a failing subscriber is logged and skipped.
- Thread safety: all state is guarded by one re-entrant lock; entries handed out are copies.

Time is passed in explicitly, so the model is deterministic and replayable.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Mapping

import numpy as np

from ..bus import Bus
from ..config.schema import WorldConfig
from .frame import FrameSpec, angular_distance, to_base_frame, wrap_angle
from .types import UNKNOWN_IDENTITY, Entry, EntryKind, Observation, ObserveResult

log = logging.getLogger(__name__)

InViewFn = Callable[[float, float], bool]
"""Whether a base-frame bearing (azimuth, elevation, radians) is inside the sensors' view."""

Event = tuple[str, object, float]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else None


def _known(identity: str | None) -> bool:
    return bool(identity) and identity != UNKNOWN_IDENTITY


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
        self._lock = threading.RLock()
        self._entries: dict[str, Entry] = {}
        self._updated: dict[str, float] = {}
        self._in_view: dict[str, bool] = {}
        self._announced: dict[str, tuple[float, float]] = {}
        self._counter = 0

    @property
    def entries(self) -> list[Entry]:
        """Copies of the current entries (no embeddings)."""
        with self._lock:
            return [e.public() for e in self._entries.values()]

    def get(self, entry_id: str) -> Entry | None:
        """A copy of one entry, or None when it is gone."""
        with self._lock:
            e = self._entries.get(entry_id)
            return e.public() if e else None

    def observe(
        self, obs: Observation, joint_state: Mapping[str, float] | None = None
    ) -> ObserveResult:
        """Fold one observation into the map and return what happened to it."""
        events: list[Event] = []
        with self._lock:
            az, el = to_base_frame(obs.azimuth_rad, obs.elevation_rad, joint_state, self.frame)
            if obs.kind == "unknown":
                result = self._confirm_bearing(obs, az, el)
            else:
                entry = self._match(obs, az, el)
                if entry is not None:
                    self._update(entry, obs, az, el, events)
                    result = ObserveResult(entry.public())
                else:
                    created = self._create(obs, az, el, events)
                    result = (
                        ObserveResult(None, rejected=True)
                        if created is None
                        else ObserveResult(created.public(), created=True)
                    )
        self._flush(events)
        return result

    def tick(self, now: float, in_view_fn: InViewFn) -> list[Entry]:
        """Decay every entry up to ``now``; remove and announce those below the forget level."""
        gone: list[Entry] = []
        events: list[Event] = []
        with self._lock:
            for entry in list(self._entries.values()):
                self._in_view[entry.entry_id] = in_view_fn(entry.azimuth_rad, entry.elevation_rad)
                self._decay(entry, now)
                if entry.confidence < self.cfg.forget_confidence:
                    self._remove(entry)
                    gone.append(entry.public())
                    events.append(("world.entity_gone", entry.public(), now))
        self._flush(events)
        return gone

    def snapshot(self, now: float) -> list[Entry]:
        """Publish ``world.snapshot`` with all entries (no embeddings) and return them."""
        with self._lock:
            entries = [e.public() for e in self._entries.values()]
        self._flush([("world.snapshot", entries, now)])
        return entries

    def _decay(self, entry: Entry, now: float) -> None:
        """Decay up to ``now`` using the entry's remembered view state; never goes back."""
        last = self._updated[entry.entry_id]
        if now <= last:
            return
        rate = math.log(2) / getattr(self.cfg.half_life_s, entry.kind)
        if not self._in_view[entry.entry_id]:
            rate *= self.cfg.out_of_view_decay_factor
        entry.confidence *= math.exp(-rate * (now - last))
        self._updated[entry.entry_id] = now

    def _angle(self, entry: Entry, az: float, el: float) -> float:
        return angular_distance(entry.azimuth_rad, entry.elevation_rad, az, el)

    def _in_gate(self, entry: Entry, obs: Observation, az: float, el: float) -> bool:
        if self._angle(entry, az, el) > math.radians(self.cfg.match_gate_deg):
            return False
        # The range gate is skipped when either side has no distance.
        if entry.distance_m is not None and obs.distance_m is not None:
            return abs(entry.distance_m - obs.distance_m) <= self.cfg.range_gate_m
        return True

    def _match(self, obs: Observation, az: float, el: float) -> Entry | None:
        if _known(obs.identity):
            for entry in self._entries.values():
                if entry.identity == obs.identity:
                    return entry
        candidates = [
            e
            for e in self._entries.values()
            if e.kind == obs.kind
            and e.category == obs.category
            and not (_known(e.identity) and _known(obs.identity) and e.identity != obs.identity)
            and self._in_gate(e, obs, az, el)
        ]
        if not candidates:
            return None
        nearest = min(self._angle(e, az, el) for e in candidates)
        tol = math.radians(self.cfg.tie_tolerance_deg)
        near = [e for e in candidates if self._angle(e, az, el) <= nearest + tol]

        def rank(e: Entry) -> tuple[float, float]:
            sim = _cosine(obs.embedding, e.embedding)
            return (sim if sim is not None else 0.0, -self._angle(e, az, el))

        return max(near, key=rank)

    def _confirm_bearing(self, obs: Observation, az: float, el: float) -> ObserveResult:
        gate = math.radians(self.cfg.match_gate_deg)
        near = [e for e in self._entries.values() if self._angle(e, az, el) <= gate]
        if not near:
            return ObserveResult(None, unexplained_bearing=(az, el))
        entry = min(near, key=lambda e: self._angle(e, az, el))
        self._gain(entry, obs)
        return ObserveResult(entry.public())

    def _gain(self, entry: Entry, obs: Observation) -> None:
        """Decay the gap up to the observation, then raise confidence; entry is now in view."""
        self._decay(entry, obs.stamp)
        gain = self.cfg.confidence_gain * obs.confidence
        entry.confidence += gain * (1.0 - entry.confidence)
        entry.last_seen = max(entry.last_seen, obs.stamp)
        entry.sources.add(obs.source)
        self._in_view[entry.entry_id] = True

    def _create(self, obs: Observation, az: float, el: float, events: list[Event]) -> Entry | None:
        confidence = self.cfg.initial_confidence * obs.confidence
        if confidence <= self.cfg.forget_confidence:
            return None
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
            confidence=confidence,
            first_seen=obs.stamp,
            last_seen=obs.stamp,
            sources={obs.source},
            embedding=None if obs.embedding is None else np.array(obs.embedding, dtype=float),
        )
        eid = entry.entry_id
        self._entries[eid] = entry
        self._updated[eid] = obs.stamp
        self._in_view[eid] = True
        self._announced[eid] = (az, el)
        events.append(("world.entity_appeared", entry.public(), obs.stamp))
        return entry

    def _update(
        self, entry: Entry, obs: Observation, az: float, el: float, events: list[Event]
    ) -> None:
        self._gain(entry, obs)
        w = self.cfg.position_blend
        offset = self._angle(entry, az, el)
        entry.azimuth_rad = wrap_angle(entry.azimuth_rad + w * wrap_angle(az - entry.azimuth_rad))
        entry.elevation_rad += w * (el - entry.elevation_rad)
        if obs.distance_m is not None:
            old = entry.distance_m
            entry.distance_m = obs.distance_m if old is None else old + w * (obs.distance_m - old)
        floor = math.radians(self.cfg.min_uncertainty_deg)
        entry.uncertainty_rad = max(floor, (1.0 - w) * entry.uncertainty_rad + w * offset)
        if _known(obs.identity):
            entry.identity = obs.identity  # type: ignore[assignment]
        if obs.embedding is not None:
            emb = np.array(obs.embedding, dtype=float)
            old_emb = entry.embedding
            if old_emb is None or old_emb.shape != emb.shape:
                entry.embedding = emb
            else:
                entry.embedding = (1.0 - w) * old_emb + w * emb
        anchor = self._announced[entry.entry_id]
        moved = angular_distance(anchor[0], anchor[1], entry.azimuth_rad, entry.elevation_rad)
        if moved > math.radians(self.cfg.moved_threshold_deg):
            self._announced[entry.entry_id] = (entry.azimuth_rad, entry.elevation_rad)
            events.append(("world.entity_moved", entry.public(), obs.stamp))

    def _remove(self, entry: Entry) -> None:
        eid = entry.entry_id
        del self._entries[eid]
        del self._updated[eid]
        del self._in_view[eid]
        del self._announced[eid]

    def _flush(self, events: list[Event]) -> None:
        """Publish collected events outside the lock; one failing subscriber aborts nothing."""
        if self.bus is None:
            return
        for topic, payload, stamp in events:
            try:
                self.bus.publish(topic, payload, stamp)
            except Exception as exc:
                log.warning("subscriber for %s failed: %s", topic, type(exc).__name__)
