"""Attention selection (SPEC 5 Connection to the motor layer, SPEC 5A Consumers).

Picks one target from world entries and an injected speaker hint, by the categories of
``attention.priority_order``: active speaker, enrolled person, nearest face, most salient
moving object. Switching is held back by ``switch_hysteresis_s`` unless the current target is
gone (absent, not confirmed for ``stale_s``, or no longer qualifying). Time is the ``now`` and
the stamps passed in, never a wall clock, so a replay gives the same targets.

Salient motion has no field on ``Entry``; the proxy is the angular speed of an entry's bearing
over ``motion_window_s``, compared with ``motion_min_dps``. Only animals and objects count: a
moving person is already covered by the face categories.

``attention.target`` carries no embeddings (SPEC 12). It is published when the target changes
(or its bearing moves by ``publish_move_deg``) and otherwise at ``publish_hz``; ``None`` is
published once when the target is lost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..bus import Bus
from ..config.schema import AttentionConfig
from ..world import Entry, angular_distance
from ..world.types import UNKNOWN_IDENTITY

SPEAKER_KEY = "speaker-bearing"
TOPIC = "attention.target"
MOVING_KINDS = ("animal", "object")


@dataclass(frozen=True)
class SpeakerHint:
    """The active speaker, as flagged by the audio side (M3): an entry id, a bearing, or both.

    The bearing is in the robot base frame (azimuth positive left, elevation positive up).
    """

    stamp: float
    entry_id: str | None = None
    azimuth_rad: float | None = None
    elevation_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_id is None and self.azimuth_rad is None:
            raise ValueError("a speaker hint needs an entry id or a bearing")


@dataclass(frozen=True)
class AttentionTarget:
    """What to look at: a world entry (or a bare bearing) in the base frame.

    ``priority`` is the category that selected it; ``reason`` says why it is the target now
    (``new``, ``switched``, ``tracking``, ``held``). ``entry_id`` is None for a bearing-only
    speaker.
    """

    entry_id: str | None
    kind: str
    azimuth_rad: float
    elevation_rad: float
    priority: str
    reason: str
    stamp: float

    @property
    def key(self) -> str:
        return self.entry_id or SPEAKER_KEY


@dataclass
class _Cand:
    key: str
    kind: str
    az: float
    el: float
    entry_id: str | None


@dataclass
class _Motion:
    az: float
    el: float
    stamp: float
    speed_dps: float = 0.0


class AttentionSelector:
    """Stateful selector: call :meth:`update` on every tick with the current entries."""

    def __init__(self, cfg: AttentionConfig, bus: Bus | None = None) -> None:
        self.cfg = cfg
        self.bus = bus
        self._hint: SpeakerHint | None = None
        self._motion: dict[str, _Motion] = {}
        self._current: AttentionTarget | None = None
        self._since = 0.0
        self._published: AttentionTarget | None = None
        self._published_at = -math.inf

    def hint_speaker(self, hint: SpeakerHint) -> None:
        """Inject the active speaker; it stays valid for ``speaker_hold_s`` after its stamp."""
        self._hint = hint

    def update(self, now: float, entries: Sequence[Entry]) -> AttentionTarget | None:
        """Choose the target at ``now``, publish it when due and return it (None: no target)."""
        fresh = [e for e in entries if now - e.last_seen <= self.cfg.stale_s]
        by_id = {e.entry_id: e for e in fresh}
        self._track_motion(now, fresh)
        ranked = self._ranked(now, fresh, by_id)
        qualified: dict[str, str] = {}
        cands_by_key: dict[str, _Cand] = {}
        for category, cands in ranked:
            for c in cands:
                qualified.setdefault(c.key, category)
                cands_by_key.setdefault(c.key, c)
        best = next(((cat, cands[0]) for cat, cands in ranked if cands), None)

        cur = self._current
        target: AttentionTarget | None
        if cur is not None and cur.key in qualified:
            better = best is not None and best[1].key != cur.key
            if better and now - self._since >= self.cfg.switch_hysteresis_s:
                target = self._switch(now, best, "switched")  # type: ignore[arg-type]
            else:
                reason = "held" if better else "tracking"
                target = self._make(now, cands_by_key[cur.key], qualified[cur.key], reason)
        elif best is not None:
            target = self._switch(now, best, "switched" if cur is not None else "new")
        else:
            target = None
        self._current = target
        self._publish(now, target)
        return target

    def _switch(self, now: float, best: tuple[str, _Cand], reason: str) -> AttentionTarget:
        self._since = now
        return self._make(now, best[1], best[0], reason)

    @staticmethod
    def _make(now: float, c: _Cand, category: str, reason: str) -> AttentionTarget:
        return AttentionTarget(c.entry_id, c.kind, c.az, c.el, category, reason, now)

    def _track_motion(self, now: float, entries: Sequence[Entry]) -> None:
        """Update each entry's angular speed once per window; forget entries that vanished."""
        window = self.cfg.motion_window_s
        for e in entries:
            m = self._motion.get(e.entry_id)
            if m is None:
                self._motion[e.entry_id] = _Motion(e.azimuth_rad, e.elevation_rad, now)
            elif now - m.stamp >= window:
                moved = angular_distance(m.az, m.el, e.azimuth_rad, e.elevation_rad)
                m.speed_dps = math.degrees(moved) / (now - m.stamp)
                m.az, m.el, m.stamp = e.azimuth_rad, e.elevation_rad, now
        for gone in set(self._motion) - {e.entry_id for e in entries}:
            del self._motion[gone]

    def _ranked(
        self, now: float, fresh: Sequence[Entry], by_id: dict[str, Entry]
    ) -> list[tuple[str, list[_Cand]]]:
        """Candidates per category, best first, in ``priority_order``."""
        persons = [e for e in fresh if e.kind == "person"]
        enrolled = [e for e in persons if e.identity and e.identity != UNKNOWN_IDENTITY]
        moving = [
            e
            for e in fresh
            if e.kind in MOVING_KINDS
            and self._motion[e.entry_id].speed_dps >= self.cfg.motion_min_dps
        ]
        pools: dict[str, list[_Cand]] = {
            "active_speaker": self._speaker(now, fresh, by_id),
            "enrolled_person": _cands(sorted(enrolled, key=lambda e: (-e.confidence, e.entry_id))),
            "nearest_face": _cands(sorted(persons, key=_nearest_key)),
            "salient_motion": _cands(
                sorted(moving, key=lambda e: (-self._motion[e.entry_id].speed_dps, e.entry_id))
            ),
        }
        return [(cat, pools[cat]) for cat in self.cfg.priority_order]

    def _speaker(self, now: float, fresh: Sequence[Entry], by_id: dict[str, Entry]) -> list[_Cand]:
        """The hinted speaker: its entry, else the nearest entry within the gate, else the bare
        bearing."""
        h = self._hint
        if h is None or now - h.stamp > self.cfg.speaker_hold_s:
            return []
        if h.entry_id is not None and h.entry_id in by_id:
            return _cands([by_id[h.entry_id]])
        if h.azimuth_rad is None:
            return []
        gate = math.radians(self.cfg.speaker_match_deg)
        scored = [
            (angular_distance(h.azimuth_rad, h.elevation_rad, e.azimuth_rad, e.elevation_rad), e)
            for e in fresh
        ]
        near = [(d, e.entry_id, e) for d, e in scored if d <= gate]
        if near:
            return _cands([min(near)[2]])
        return [_Cand(SPEAKER_KEY, "unknown", h.azimuth_rad, h.elevation_rad, None)]

    def _publish(self, now: float, target: AttentionTarget | None) -> None:
        """Publish on change, on a bearing move, or when ``publish_hz`` says it is due."""
        if self.bus is None:
            return
        last = self._published
        if target is None:
            if last is not None:
                self.bus.publish(TOPIC, None, now)
                self._published = None
            return
        changed = last is None or last.key != target.key or last.priority != target.priority
        if not changed:
            shift = angular_distance(
                last.azimuth_rad,  # type: ignore[union-attr]
                last.elevation_rad,  # type: ignore[union-attr]
                target.azimuth_rad,
                target.elevation_rad,
            )
            changed = shift >= math.radians(self.cfg.publish_move_deg)
        due = now - self._published_at >= 1.0 / self.cfg.publish_hz
        if changed or due:
            self.bus.publish(TOPIC, target, now)
            self._published, self._published_at = target, now


def _nearest_key(e: Entry) -> tuple[bool, float, float, str]:
    return (e.distance_m is None, e.distance_m or 0.0, -e.confidence, e.entry_id)


def _cands(entries: Sequence[Entry]) -> list[_Cand]:
    return [_Cand(e.entry_id, e.kind, e.azimuth_rad, e.elevation_rad, e.entry_id) for e in entries]
