"""Bus to appraisal events (SPEC 7, Appraisal events).

The adapter listens to the perception topics and names what happened; the affect engine's
appraisal rules decide what it means. Events (payload fields usable in a rule's ``when``):

- ``person.known``: any ``person.recognized``; also exactly one of ``person.known_child``,
  ``person.known_teen``, ``person.known_adult``, ``person.known_senior`` or
  ``person.known_unknown`` (one per registry age group). Payload: ``person_id``,
  ``age_group``, ``possible_minor``.
- ``person.unknown_appears``: a new face track that has no owner after
  ``appraisal.unknown_face_after_s``; not for a face recognised inside that window (one
  recognised later has already emitted it). Payload: ``track_id``, ``age_group``,
  ``possible_minor``.
- ``animal.appears`` (named ``animal.appeared`` in early drafts): ``world.entity_appeared`` of
  kind ``animal``. Payload: ``entity_id``, ``category``.
- ``sound.loud``: a ``sound.loud`` message from the reflex engine's audio intake (a level scalar,
  never audio). Payload: empty.
- ``idle.long``: no face, attention target or recognised person for ``appraisal.idle_after_s``;
  once, until interaction resumes. Payload: empty.

Person ids travel in payloads for rules but are never logged at info level (SPEC 12).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..bus import Bus, Message
from ..config.schema import AppraisalConfig
from ..registry.records import AGE_GROUPS

Emit = Callable[[str, float, dict[str, Any]], None]
UNKNOWN = "unknown"


@dataclass
class _Pending:
    first: float
    last: float
    age_group: str
    possible_minor: bool


@dataclass
class AppraisalAdapter:
    """Turns bus messages into named appraisal events passed to ``emit(event, now, payload)``.

    Handlers run on whichever thread publishes; :meth:`tick` (the app loop) checks the two
    time-based events. State is guarded by one lock and ``emit`` is called outside it.
    """

    bus: Bus
    cfg: AppraisalConfig
    emit: Emit
    start: float = 0.0
    detection_period: Callable[[], float] = lambda: math.inf  # noqa: E731
    _pending: dict[int, _Pending] = field(default_factory=dict)
    _announced: dict[int, float] = field(default_factory=dict)  # track -> last seen
    _last_interaction: float = 0.0
    _idle_fired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _unsubscribe: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._last_interaction = self.start
        for topic, handler in (
            ("person.recognized", self._on_recognized),
            ("percept.face", self._on_face),
            ("world.entity_appeared", self._on_entity),
            ("attention.target", self._on_attention),
            ("sound.loud", self._on_loud),
        ):
            self._unsubscribe.append(self.bus.subscribe(topic, handler))

    def close(self) -> None:
        """Stop listening."""
        for fn in self._unsubscribe:
            fn()
        self._unsubscribe.clear()

    def tick(self, now: float) -> None:
        """Fire the time-based events: unknown faces past the decision window and idle.long."""
        events: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            lost = self._lost_s()
            self._announced = {t: seen for t, seen in self._announced.items() if now - seen < lost}
            for track, p in list(self._pending.items()):
                if now - p.last >= lost:
                    del self._pending[track]  # gone before it was decided: no event
                elif now - p.first >= self.cfg.unknown_face_after_s:
                    del self._pending[track]
                    self._announced[track] = p.last
                    payload = {
                        "track_id": track,
                        "age_group": p.age_group,
                        "possible_minor": p.possible_minor,
                    }
                    events.append(("person.unknown_appears", payload))
            if not self._idle_fired and now - self._last_interaction >= self.cfg.idle_after_s:
                self._idle_fired = True
                events.append(("idle.long", {}))
        for name, payload in events:
            self.emit(name, now, payload)

    def _lost_s(self) -> float:
        """How long a face track may go unseen: at least ``track_lost_s``, and
        ``track_lost_periods`` detection periods at the rate the governor allows now."""
        period = self.detection_period()
        lost = self.cfg.track_lost_s
        return max(lost, self.cfg.track_lost_periods * period) if math.isfinite(period) else lost

    def _interaction(self, stamp: float) -> None:
        """Note an interaction (caller holds the lock): re-arms idle.long."""
        self._last_interaction = max(self._last_interaction, stamp)
        self._idle_fired = False

    def _on_face(self, msg: Message) -> None:
        p = msg.payload
        with self._lock:
            self._interaction(msg.stamp)
            if p.owner_id is not None:
                self._pending.pop(p.track_id, None)
                return
            if p.track_id in self._announced:
                self._announced[p.track_id] = msg.stamp
                return
            pend = self._pending.get(p.track_id)
            if pend is not None:
                pend.last = msg.stamp
                pend.age_group, pend.possible_minor = p.age_group, p.possible_minor
            elif p.new_track:
                self._pending[p.track_id] = _Pending(
                    msg.stamp, msg.stamp, p.age_group, p.possible_minor
                )

    def _on_recognized(self, msg: Message) -> None:
        p = msg.payload
        with self._lock:
            self._interaction(msg.stamp)
            self._pending.pop(p.track_id, None)
            self._announced.pop(p.track_id, None)
        payload = {
            "person_id": p.person_id,
            "age_group": p.age_group,
            "possible_minor": p.possible_minor,
        }
        group = p.age_group if p.age_group in AGE_GROUPS else UNKNOWN
        for name in ("person.known", f"person.known_{group}"):
            self.emit(name, msg.stamp, dict(payload))

    def _on_entity(self, msg: Message) -> None:
        if msg.payload.kind == "animal":
            self.emit(
                "animal.appears",
                msg.stamp,
                {"entity_id": msg.payload.entry_id, "category": msg.payload.category},
            )

    def _on_loud(self, msg: Message) -> None:
        self.emit("sound.loud", msg.stamp, {})

    def _on_attention(self, msg: Message) -> None:
        if msg.payload is None:  # the target was cleared: nobody to interact with
            return
        with self._lock:
            self._interaction(msg.stamp)
