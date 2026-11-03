"""Joint attention: interest in a thing a person shows, against the pull of their face (SPEC 5A).

An animal or object entry is *shown* when it lies in a person's region (beside the face or below
it, in face-box heights, ``attention.shown.region_*``) and it moved or first appeared there within
``recent_s``. The caller leaves habituated faces out (a poster holds nothing). A shown entry
keeps its person until it is ``region_exit_faces`` outside the region (edge hysteresis), and its
state survives ``region_grace_s`` outside the region or the view, so coming back within it gives
no fresh show gain. Each shown entry keeps an interest value:

- rises once when it is shown (``show_gain``) and with each measured movement (``move_gain_per_s``
  times the span of the movement), both times its novelty and the state's ``rise_x``. Movement has
  diminishing returns: its gain falls by e for every ``move_habituation_s`` of movement already
  seen, so a thing that never stops still runs out of interest;
- novelty: 1 for an unknown thing, ``familiar_novelty`` for a known individual (a registry animal
  or object), divided by 1 + the recent exposure of its category (one per showing, halving every
  ``exposure_half_life_s``). A thing the registry recognises after it was shown (identity
  votes take a few object frames) has its novelty and interest scaled by ``familiar_novelty``
  then;
- fades while the robot looks at it (``decay_per_s`` times the state's ``decay_x``), and only then.

One social pull stands for the faces: ``pull_base`` plus an extra that grows by
``pull_rate_per_s`` while the robot looks at a shown thing and falls by ``pull_recover_per_s``
while it looks at a person. A shown thing holds the gaze while its interest exceeds the pull.
Nothing else ends the look: there is no maximum duration (owner decision 2026-11-03). The robot's
state shapes the rates through :func:`interest_mod` (affect arousal and label, circadian state,
persona traits). Time is the ``now`` and the stamps passed in, never a wall clock.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..config.schema import ShownObjectConfig, StateEffects
from ..persona import PersonaEffects
from ..world import Entry, angular_distance
from ..world.frame import wrap_angle
from ..world.types import UNKNOWN_IDENTITY

CURIOSITY = "curiosity"


@dataclass(frozen=True)
class InterestMod:
    """How the robot's state scales the interest model: gains by ``rise_x``, decay by
    ``decay_x``. The neutral value changes nothing."""

    rise_x: float = 1.0
    decay_x: float = 1.0


def interest_mod(
    cfg: ShownObjectConfig,
    *,
    arousal: float,
    label: str | None,
    circadian: StateEffects,
    persona: PersonaEffects,
) -> InterestMod:
    """The modulation from affect ``arousal`` and ``label``, the circadian state's effects and the
    persona's effects: arousal and curiosity raise the gains and slow the decay, a sleepy state
    speeds the decay (``interest_decay_scale``), the traits act through their curves."""
    curious = label == CURIOSITY
    rise = (1.0 + cfg.arousal_rise_gain * arousal) * persona.interest_gain_x
    decay = (
        (1.0 - cfg.arousal_decay_gain * arousal)
        * circadian.interest_decay_scale
        * persona.interest_decay_x
    )
    if curious:
        rise *= cfg.curiosity_rise_x
        decay *= cfg.curiosity_decay_x
    return InterestMod(max(0.0, rise), max(0.0, decay))


@dataclass
class Shown:
    """One shown entry: its interest, the person it is shown by, its novelty, the stamp of its
    last stimulus, the movement already seen (seconds, for the diminishing returns) and whether
    it was already a known individual when the novelty was set."""

    interest: float
    holder: str
    novelty: float
    stimulus: float
    moved_s: float = 0.0
    known: bool = False


def in_region(cfg: ShownObjectConfig, thing: Entry, face: Entry, margin: float = 0.0) -> bool:
    """Whether ``thing`` lies in ``face``'s region: within ``region_width_faces`` face heights to
    either side, from ``region_above_faces`` above the face centre to ``region_below_faces``
    below it, each widened by ``margin`` face heights. A face without a box size has no
    region."""
    size = face.extent_rad
    if size is None or size <= 0:
        return False
    dx = abs(wrap_angle(thing.azimuth_rad - face.azimuth_rad))
    dy = thing.elevation_rad - face.elevation_rad
    return (
        dx <= (cfg.region_width_faces + margin) * size
        and -(cfg.region_below_faces + margin) * size
        <= dy
        <= (cfg.region_above_faces + margin) * size
    )


class InterestModel:
    """Interest per shown entry and the social pull; :meth:`step` once per selector tick."""

    def __init__(self, cfg: ShownObjectConfig) -> None:
        self.cfg = cfg
        self.shown: dict[str, Shown] = {}
        self._parked: dict[str, tuple[Shown, float]] = {}  # left the region: (state, since)
        self.pull_extra = 0.0
        self._exposure: dict[str, tuple[float, float]] = {}  # category -> (value, stamp)

    @property
    def pull(self) -> float:
        """The social pull of the faces now."""
        return self.cfg.pull_base + self.pull_extra

    def clear(self) -> None:
        """Forget every shown entry and the extra pull (the robot fell asleep)."""
        self.shown.clear()
        self._parked.clear()
        self.pull_extra = 0.0

    def holds(self, entry_id: str, *, looking: bool) -> bool:
        """Whether a shown entry's interest exceeds the pull (``looking``: the robot looks at
        it already), or the pull plus ``start_margin`` of it (a look would start)."""
        s = self.shown.get(entry_id)
        margin = 0.0 if looking else self.cfg.start_margin
        return s is not None and s.interest > self.pull * (1.0 + margin)

    def recent(self, entry_id: str, now: float) -> bool:
        """Whether a shown entry had a stimulus (shown or moved) within ``recent_s``."""
        s = self.shown.get(entry_id)
        return s is not None and now - s.stimulus <= self.cfg.recent_s

    def step(
        self,
        now: float,
        dt: float,
        things: Sequence[Entry],
        people: Sequence[Entry],
        moves: dict[str, float],
        *,
        watched: str | None,
        looking: str | None,
        mod: InterestMod,
    ) -> None:
        """Advance by ``dt`` seconds to ``now``.

        ``things``: the fresh animals and objects; ``people``: the person entries a region may
        belong to (the caller leaves out habituated faces); ``moves``: entry id to the span (s)
        of a movement measured this tick;
        ``watched``: the entry the robot looked at over ``dt`` (its interest fades);
        ``looking``: ``"thing"`` while it looked at a shown thing, ``"person"`` at a person
        (the pull grows or falls), None otherwise."""
        cfg = self.cfg
        if watched in self.shown:
            self.shown[watched].interest *= math.exp(-cfg.decay_per_s * mod.decay_x * dt)
        if looking == "thing":
            self.pull_extra += cfg.pull_rate_per_s * dt
        elif looking == "person":
            self.pull_extra = max(0.0, self.pull_extra - cfg.pull_recover_per_s * dt)
        kept: dict[str, Shown] = {}
        for e in things:
            s = self._observe(now, e, people, moves.get(e.entry_id), mod)
            if s is not None:
                kept[e.entry_id] = s
        for eid, s in self.shown.items():  # left the region or the view: kept for a grace
            if eid not in kept:
                self._parked.setdefault(eid, (s, now))
        self._parked = {
            eid: (s, since)
            for eid, (s, since) in self._parked.items()
            if eid not in kept and now - since <= cfg.region_grace_s
        }
        self.shown = kept

    def _observe(
        self,
        now: float,
        e: Entry,
        people: Sequence[Entry],
        span: float | None,
        mod: InterestMod,
    ) -> Shown | None:
        """The entry's state after this tick: None once it is in no person's region, or when it
        is not shown yet and had no stimulus."""
        cfg = self.cfg
        s = self.shown.get(e.entry_id)
        if s is None and e.entry_id in self._parked:
            s = self._parked[e.entry_id][0]  # back within the grace: same state, no new gain
        holder = self._holder(e, people, s.holder if s is not None else None)
        if holder is None:
            if s is not None:
                self._parked.setdefault(e.entry_id, (s, now))
            return None
        if s is None:
            appeared = now - e.first_seen <= cfg.recent_s
            if span is None and not appeared:
                return None
            novelty = self._novelty(e, now)
            self._expose(e.category, now)
            s = Shown(cfg.show_gain * novelty * mod.rise_x, holder, novelty, e.last_seen)
            s.known = _known(e)
        elif not s.known and _known(e):
            # the registry recognised it after it was shown (identity votes take a few frames)
            s.known = True
            s.novelty *= cfg.familiar_novelty
            s.interest *= cfg.familiar_novelty
        s.holder = holder
        if span is not None:
            fresh = math.exp(-s.moved_s / cfg.move_habituation_s)
            s.interest += cfg.move_gain_per_s * span * s.novelty * mod.rise_x * fresh
            s.moved_s += span
            s.stimulus = max(s.stimulus, e.last_seen)
        return s

    def _holder(self, e: Entry, people: Iterable[Entry], previous: str | None) -> str | None:
        """The person whose region holds ``e``: the previous one while ``e`` is within its region
        widened by ``region_exit_faces`` (edge hysteresis), else the nearest face."""
        cfg = self.cfg
        if any(
            p.entry_id == previous and in_region(cfg, e, p, cfg.region_exit_faces) for p in people
        ):
            return previous
        inside = [p for p in people if in_region(cfg, e, p)]
        if not inside:
            return None
        return min(
            inside,
            key=lambda p: (
                angular_distance(p.azimuth_rad, p.elevation_rad, e.azimuth_rad, e.elevation_rad),
                p.entry_id,
            ),
        ).entry_id

    def _novelty(self, e: Entry, now: float) -> float:
        """1 for an unknown thing, ``familiar_novelty`` for a known one, over 1 + the category's
        exposure."""
        base = self.cfg.familiar_novelty if _known(e) else 1.0
        return base / (1.0 + self._exposure_at(e.category, now))

    def _exposure_at(self, category: str, now: float) -> float:
        value, stamp = self._exposure.get(category, (0.0, now))
        return value * 0.5 ** (max(0.0, now - stamp) / self.cfg.exposure_half_life_s)

    def _expose(self, category: str, now: float) -> None:
        self._exposure[category] = (self._exposure_at(category, now) + 1.0, now)


def _known(e: Entry) -> bool:
    """Whether ``e`` is a known individual (a registry animal or object)."""
    return bool(e.identity) and e.identity != UNKNOWN_IDENTITY
