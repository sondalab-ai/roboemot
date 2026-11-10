"""Attention selection (SPEC 5 Connection to the motor layer, SPEC 5A Consumers).

Picks one target from world entries and an injected speaker hint, by the categories of
``attention.priority_order``: active speaker, shown object, enrolled person, nearest face, most
salient moving object. Switching is held back by ``switch_hysteresis_s`` unless the current
target is gone (absent, not confirmed for ``stale_s``, or no longer qualifying). Time is the
``now`` and the stamps passed in, never a wall clock, so a replay gives the same targets.

Salient motion has no field on ``Entry``; the proxy is the angular speed of an entry's bearing
between its own observations (``last_seen`` stamps) at least ``motion_window_s`` apart, compared
with ``motion_min_dps``. The speed holds until the next measurement, so an object detected once a
second reads as moving on every tick in between, never as still. Only animals and objects count:
a moving person is already covered by the face categories.

Shown object (joint attention, ``interest``): an animal or object a person shows the robot holds
the gaze while its interest exceeds the social pull of the faces; the end of the look emerges
from the model, never from a timer. A look starts only on a recent stimulus (shown or moved
within ``attention.shown.recent_s``), and only near a face that has not habituated: a poster
or a photo is nobody to show anything. While it lasts the holder's face counts as seen (the head
looks down at the thing, the face may leave the camera) and does not habituate, so the gaze has
a face to come back to. ``engaged`` (scanning yields, company, presence, conversation) is true
for a shown object only while its holder was really detected within ``stale_s``, not merely
excused; ``episode_active`` alone holds off the scan for the look. Each episode is counted with
how it ended (``END_REASONS``) and its duration for the debug report (``snapshot``).

Habituation: a face whose bearing stays within ``habituation_move_deg`` of where it settled for
``habituation_s`` stops qualifying as an enrolled person or a nearest face (a photo or a poster
holds the gaze otherwise). A larger move, a speaker hint on the face, or a conversation in
progress (``update(conversing=True)``) restarts its clock, and so does a look at a thing the
face holds (the holder of the current episode). A habituated face can still be the active
speaker.

``attention.target`` carries no embeddings (SPEC 12). It is published when the target changes
(or its bearing moves by ``publish_move_deg``) and otherwise at ``publish_hz``; ``None`` is
published once when the target is lost.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..bus import Bus
from ..config.schema import ATTENTION_CATEGORIES, AttentionConfig
from ..stats import SUMMARY_LEVELS, summarise
from ..world import Entry, angular_distance
from ..world.types import UNKNOWN_IDENTITY
from .interest import InterestMod, InterestModel
from .types import SHOWN, SPEAKER, SPEAKER_KEY, AttentionTarget

TOPIC = "attention.target"
MOVING_KINDS = ("animal", "object")
# How a shown-object episode ended: the face's pull won, a speaker took the gaze, the thing left
# the person's region or the view, the robot fell asleep, another shown thing won.
END_REASONS = ("face_pull", "speaker", "left_region", "lost", "asleep", "other_object")


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


@dataclass
class _Cand:
    key: str
    kind: str
    az: float
    el: float
    entry_id: str | None
    score: float | None = None  # higher is better; None when the category has no scalar score
    category: str = ""


@dataclass
class _Settled:
    az: float
    el: float
    since: float


@dataclass
class _Motion:
    az: float  # bearing at the anchor observation
    el: float
    stamp: float  # capture stamp of the anchor observation
    speed_dps: float = 0.0


@dataclass
class _Episode:
    entry_id: str
    holder: str
    start: float


class AttentionSelector:
    """Stateful selector: call :meth:`update` on every tick with the current entries."""

    def __init__(self, cfg: AttentionConfig, bus: Bus | None = None) -> None:
        self.cfg = cfg
        self.bus = bus
        self._hint: SpeakerHint | None = None
        self._motion: dict[str, _Motion] = {}
        self._settled: dict[str, _Settled] = {}
        self._current: AttentionTarget | None = None
        self._min_hold_s = 0.0
        self._since = 0.0
        self._published: AttentionTarget | None = None
        self._published_at = -math.inf
        self._interest = InterestModel(cfg.shown)
        self._last_now: float | None = None
        self._episode: _Episode | None = None
        self._excused: tuple[str, float] | None = None  # (holder, last tick of its episode)
        self._present = False  # the person behind the target was really seen within stale_s
        self._completed = False  # the last update ran to its end
        self._stats_lock = threading.Lock()
        self._started = 0
        self._ended = dict.fromkeys(END_REASONS, 0)
        self._durations: deque[float] = deque(maxlen=cfg.shown.report_window)

    def hint_speaker(self, hint: SpeakerHint) -> None:
        """Inject the active speaker; it stays valid for ``speaker_hold_s`` after its stamp."""
        self._hint = hint

    def speaker_active(self, now: float) -> bool:
        """Whether a speaker hint is still valid at ``now`` (someone spoke within
        ``speaker_hold_s``), whether or not the speaker is in view."""
        h = self._hint
        return h is not None and now - h.stamp <= self.cfg.speaker_hold_s

    @property
    def target(self) -> AttentionTarget | None:
        """The target chosen by the last ``update`` (None: no target)."""
        return self._current

    @property
    def engaged(self) -> bool:
        """Whether the current target is the active speaker, or a person (or a thing a person
        shows) while that person was really detected within ``stale_s``, not only excused by the
        look-down (scanning yields, the alone watch counts company, the circadian presence and
        the conversation session hold)."""
        t = self._current
        if t is None:
            return False
        if t.priority == SPEAKER:
            return True
        return (t.kind == "person" or t.priority == SHOWN) and self._present

    @property
    def episode_active(self) -> bool:
        """Whether a shown object was the target of the last update, and that update completed
        (the object stage runs at its active rate; a failed tick clears it)."""
        return self._completed and self._episode is not None

    def sleep(self, now: float) -> None:
        """The robot fell asleep: end a shown-object episode (``asleep``) and forget the shown
        things and the extra pull. The tick does not run while asleep."""
        if self._episode is not None:
            self._end_episode(now, "asleep")
        self._interest.clear()
        self._last_now = None

    def update(
        self,
        now: float,
        entries: Sequence[Entry],
        *,
        conversing: bool = False,
        mod: InterestMod | None = None,
        min_hold_s: float = 0.0,
    ) -> AttentionTarget | None:
        """Choose the target at ``now``, publish it when due and return it (None: no target).
        ``conversing``: a conversation is in progress, so no face habituates. ``mod``: how the
        robot's state scales the shown-object interest (``interest_mod``; None: neutral).
        ``min_hold_s``: least time on a target before a switch, when longer than the hysteresis
        (the dance over room music, SPEC 7)."""
        self._min_hold_s = min_hold_s
        self._completed = False
        fresh = [e for e in entries if now - self._seen(e) <= self.cfg.stale_s]
        by_id = {e.entry_id: e for e in fresh}
        moves = self._track_motion(fresh)
        self._step_interest(now, fresh, moves, mod or InterestMod())
        self._track_settled(now, fresh, conversing)
        ranked = self._ranked(now, fresh, by_id)
        speaker = dict(ranked).get(SPEAKER)
        if speaker and (s := self._settled.get(speaker[0].key)) is not None:
            s.since = now  # a face that speaks is not a photo
        qualified: dict[str, str] = {}
        cands_by_key: dict[str, _Cand] = {}
        for category, cands in ranked:
            for c in cands:
                qualified.setdefault(c.key, category)
                cands_by_key.setdefault(c.key, c)
        best = next(((cat, cands[0]) for cat, cands in ranked if cands), None)

        cur = self._current
        target: AttentionTarget | None
        if best is None:
            target = None
        elif cur is None or cur.key not in qualified:
            target = self._switch(now, best, "new" if cur is None else "switched")
        elif best[1].key == cur.key:
            target = self._make(now, cands_by_key[cur.key], qualified[cur.key], "tracking")
        elif self._may_switch(now, best, qualified[cur.key], cands_by_key[cur.key]):
            target = self._switch(now, best, "switched")
        else:
            target = self._make(now, cands_by_key[cur.key], qualified[cur.key], "held")
        self._current = target
        self._track_episode(now, target, by_id)
        self._present = self._really_seen(now, entries, target)
        self._publish(now, target)
        self._completed = True
        return target

    def snapshot(self) -> dict[str, Any]:
        """The shown-object episodes for the debug report (SPEC 12A): started, ended by reason,
        duration p50 and p90 over the last ``attention.shown.report_window``. Numbers only."""
        with self._stats_lock:
            return {
                "shown_object": {
                    "active": self._episode is not None,
                    "episodes": self._started,
                    "ended": dict(self._ended),
                    "duration_s": summarise(list(self._durations), SUMMARY_LEVELS),
                }
            }

    def _may_switch(
        self, now: float, best: tuple[str, _Cand], cur_category: str, cur: _Cand
    ) -> bool:
        """A new active speaker wins at once. Otherwise the hysteresis time must have passed,
        and a candidate of the same category must beat the current one by ``switch_margin``
        (so noisy, similar scores do not make the head flip-flop)."""
        category, cand = best
        if category == SPEAKER:
            return True
        if now - self._since < max(self.cfg.switch_hysteresis_s, self._min_hold_s):
            return False
        if category != cur_category or cand.score is None or cur.score is None:
            return True
        return cur.score <= 0 or cand.score / cur.score - 1.0 >= self.cfg.switch_margin

    def _switch(self, now: float, best: tuple[str, _Cand], reason: str) -> AttentionTarget:
        self._since = now
        return self._make(now, best[1], best[0], reason)

    @staticmethod
    def _make(now: float, c: _Cand, category: str, reason: str) -> AttentionTarget:
        return AttentionTarget(c.entry_id, c.kind, c.az, c.el, category, reason, now, c.category)

    def _seen(self, e: Entry) -> float:
        """An entry's last sighting; the holder of a shown thing counts as seen up to the last
        tick of its episode (the robot looked down at the thing, not away from the person)."""
        ex = self._excused
        if ex is not None and ex[0] == e.entry_id:
            return max(e.last_seen, ex[1])
        return e.last_seen

    def _really_seen(
        self, now: float, entries: Sequence[Entry], target: AttentionTarget | None
    ) -> bool:
        """Whether the person behind ``target`` (the person itself, or the holder of a shown
        thing) was really detected within ``stale_s``: its own ``last_seen``, not the look-down
        excuse."""
        if target is None:
            return False
        if target.priority == SHOWN:
            who = self._episode.holder if self._episode is not None else None
        else:
            who = target.entry_id
        person = next((e for e in entries if e.entry_id == who), None)
        return person is not None and now - person.last_seen <= self.cfg.stale_s

    def _step_interest(
        self, now: float, fresh: Sequence[Entry], moves: dict[str, float], mod: InterestMod
    ) -> None:
        """Advance the interest model over the time since the last tick, with what the robot
        looked at meanwhile."""
        model = self._interest
        model.cfg = self.cfg.shown
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        cur = self._current
        looking = None
        if cur is not None and cur.priority == SHOWN:
            looking = "thing"
        elif cur is not None and cur.kind == "person":
            looking = "person"
        holder = self._episode.holder if self._episode is not None else None
        people = [
            e
            for e in fresh
            if e.kind == "person" and (e.entry_id == holder or not self._habituated(now, e))
        ]
        model.step(
            now,
            dt,
            [e for e in fresh if e.kind in MOVING_KINDS],
            people,
            moves,
            watched=cur.entry_id if cur is not None else None,
            looking=looking,
            mod=mod,
        )

    def _track_episode(
        self, now: float, target: AttentionTarget | None, by_id: dict[str, Entry]
    ) -> None:
        """End the episode on the tick its thing stops being the shown-object target, with the
        reason; start one when a shown object becomes the target; keep its holder excused."""
        ep = self._episode
        if ep is not None and (
            target is None or target.priority != SHOWN or target.entry_id != ep.entry_id
        ):
            if target is not None and target.priority == SPEAKER:
                reason = "speaker"
            elif ep.entry_id not in by_id:
                reason = "lost"
            elif ep.entry_id not in self._interest.shown:
                reason = "left_region"
            elif target is not None and target.priority == SHOWN:
                reason = "other_object"
            else:
                reason = "face_pull"
            self._end_episode(now, reason)
        if target is None or target.priority != SHOWN:
            return
        shown = self._interest.shown[target.key]
        if self._episode is None:
            self._episode = _Episode(target.key, shown.holder, now)
            with self._stats_lock:
                self._started += 1
        self._episode.holder = shown.holder
        self._excused = (shown.holder, now)

    def _end_episode(self, now: float, reason: str) -> None:
        ep = self._episode
        assert ep is not None
        self._episode = None
        with self._stats_lock:
            self._ended[reason] += 1
            self._durations.append(now - ep.start)

    def _track_motion(self, entries: Sequence[Entry]) -> dict[str, float]:
        """Measure each entry's angular speed from its own observations: once its ``last_seen``
        is at least ``motion_window_s`` past the anchor observation, the bearing change over
        that span is the speed, held until the next measurement. Forget entries that vanished.

        Returns the entries measured moving this tick (at least ``motion_min_dps``) with the
        span of the measurement in seconds (at most ``stale_s``)."""
        window = self.cfg.motion_window_s
        moves: dict[str, float] = {}
        for e in entries:
            m = self._motion.get(e.entry_id)
            if m is None:
                self._motion[e.entry_id] = _Motion(e.azimuth_rad, e.elevation_rad, e.last_seen)
                continue
            span = e.last_seen - m.stamp
            if span < window:
                continue
            moved = angular_distance(m.az, m.el, e.azimuth_rad, e.elevation_rad)
            m.speed_dps = math.degrees(moved) / span
            m.az, m.el, m.stamp = e.azimuth_rad, e.elevation_rad, e.last_seen
            if m.speed_dps >= self.cfg.motion_min_dps:
                moves[e.entry_id] = min(span, self.cfg.stale_s)
        for gone in set(self._motion) - {e.entry_id for e in entries}:
            del self._motion[gone]
        return moves

    def _track_settled(self, now: float, entries: Sequence[Entry], conversing: bool) -> None:
        """Restart a face's habituation clock when it moved past ``habituation_move_deg`` from
        where it settled, while ``conversing``, or while the robot looks at a thing it holds (the
        holder of the current episode); forget faces that vanished."""
        limit = math.radians(self.cfg.habituation_move_deg)
        persons = [e for e in entries if e.kind == "person"]
        holders = {self._episode.holder} if self._episode is not None else set()
        for e in persons:
            s = self._settled.get(e.entry_id)
            if s is None:
                self._settled[e.entry_id] = _Settled(e.azimuth_rad, e.elevation_rad, now)
            elif (
                conversing
                or e.entry_id in holders
                or angular_distance(s.az, s.el, e.azimuth_rad, e.elevation_rad) > limit
            ):
                s.az, s.el, s.since = e.azimuth_rad, e.elevation_rad, now
        for gone in set(self._settled) - {e.entry_id for e in persons}:
            del self._settled[gone]

    def _habituated(self, now: float, e: Entry) -> bool:
        """Whether face ``e`` has been still for ``habituation_s`` (never when it is 0)."""
        after = self.cfg.habituation_s
        s = self._settled.get(e.entry_id)
        return after > 0 and s is not None and now - s.since >= after

    def _ranked(
        self, now: float, fresh: Sequence[Entry], by_id: dict[str, Entry]
    ) -> list[tuple[str, list[_Cand]]]:
        """Candidates per category, best first, in ``priority_order``."""
        persons = [e for e in fresh if e.kind == "person"]
        watched = [e for e in persons if not self._habituated(now, e)]
        enrolled = [e for e in watched if e.identity and e.identity != UNKNOWN_IDENTITY]
        moving = [
            e
            for e in fresh
            if e.kind in MOVING_KINDS
            and self._motion[e.entry_id].speed_dps >= self.cfg.motion_min_dps
        ]
        cur = self._current
        model = self._interest
        looking = cur is not None and cur.priority == SHOWN
        shown = [
            by_id[eid]
            for eid in model.shown
            if eid in by_id
            and (
                model.holds(eid, looking=True)
                if looking and cur is not None and cur.key == eid
                else model.holds(eid, looking=False) and model.recent(eid, now)
            )
        ]
        pools: dict[str, list[_Cand]] = {
            SPEAKER: self._speaker(now, fresh, by_id),
            SHOWN: _cands(
                sorted(shown, key=lambda e: (-model.shown[e.entry_id].interest, e.entry_id)),
                lambda e: model.shown[e.entry_id].interest,
            ),
            "enrolled_person": _cands(
                sorted(enrolled, key=lambda e: (-e.confidence, e.entry_id)),
                lambda e: e.confidence,
            ),
            "nearest_face": _cands(
                sorted(watched, key=_nearest_key),
                lambda e: 1.0 / e.distance_m if e.distance_m else None,
            ),
            "salient_motion": _cands(
                sorted(moving, key=lambda e: (-self._motion[e.entry_id].speed_dps, e.entry_id)),
                lambda e: self._motion[e.entry_id].speed_dps,
            ),
        }
        assert set(pools) == set(ATTENTION_CATEGORIES)
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
            if e.kind == "person"
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
        if last is not None and not changed:
            shift = angular_distance(
                last.azimuth_rad, last.elevation_rad, target.azimuth_rad, target.elevation_rad
            )
            changed = shift >= math.radians(self.cfg.publish_move_deg)
        due = now - self._published_at >= 1.0 / self.cfg.publish_hz
        if changed or due:
            self.bus.publish(TOPIC, target, now)
            self._published, self._published_at = target, now


def _nearest_key(e: Entry) -> tuple[bool, float, float, str]:
    return (e.distance_m is None, e.distance_m or 0.0, -e.confidence, e.entry_id)


def _cands(
    entries: Sequence[Entry], score: Callable[[Entry], float | None] = lambda e: None
) -> list[_Cand]:
    return [
        _Cand(e.entry_id, e.kind, e.azimuth_rad, e.elevation_rad, e.entry_id, score(e), e.category)
        for e in entries
    ]
