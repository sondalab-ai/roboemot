"""The empathic register in effect (SPEC 7, Register selection).

:class:`RegisterTracker` keeps the set of people in view from ``percept.face`` and
:func:`resolve_register` turns that set into one register:

1. per person, first match wins: the record's ``register_overrides`` (laid over the group's
   register, guarded fields only ever tighter), the declared age group, the estimated age group
   (the pipeline already withholds an estimate below ``identity.estimate_confidence_min``), then
   ``unknown``;
2. the most restrictive register of the people present (see :func:`most_restrictive`);
3. the persona's traits shape it once (``apply_to_register``);
4. a possible minor within the minor-present hold applies the minor content policy.

Nobody in view resolves to ``unknown``, the fail-closed default, like
:class:`~roboemot.vision.minor.MinorPresence`. The result is published on ``register.active``
when it changes; it carries no person ids. Person ids stay inside the tracker, which uses them
only to look up the record's overrides (SPEC 12).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..bus import Bus, Message
from ..config.schema import ExpressionConfig, PersonaConfig, RegisterConfig, RegisterParams
from ..persona import REACTION_LEVELS, PersonaEffects, apply_to_register
from ..persona import effects as persona_effects

log = logging.getLogger(__name__)

TOPIC = "register.active"
UNKNOWN = "unknown"
GROUPS = ("child", "teen", "adult", "senior", UNKNOWN)
# Tie-break between two equally restrictive people: the stricter default group first.
GROUP_ORDER = (UNKNOWN, "child", "teen", "adult", "senior")


@dataclass(frozen=True)
class Person:
    """One person in view, as the resolver needs them. ``overrides`` is the record's partial
    register (empty for a stranger). No id."""

    age_group: str
    age_source: str
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ActiveRegister:
    """The register in effect: the group it came from, the shaped parameters and how many
    people it covers."""

    group: str
    params: RegisterParams
    people: int

    def summary(self) -> dict[str, Any]:
        """Group and parameters for the debug report; no person id, no free text."""
        params = self.params.model_dump(mode="json", exclude={"persona_fragment"})
        return {"group": self.group, "people": self.people, **params}


def _guarded(name: str) -> str | None:
    """The safe direction of a guarded register field (``false``/``true``), else None."""
    extra = RegisterParams.model_fields[name].json_schema_extra
    return extra.get("conservative") if isinstance(extra, dict) else None


def apply_overrides(base: RegisterParams, overrides: Mapping[str, Any]) -> RegisterParams:
    """Lay a person's partial register over ``base``.

    A guarded field (``approach_allowed``, ``minor_content_policy``) moves only toward its safe
    value, so a record cannot loosen what the group's register says (SPEC 11A, Guardrails). An
    invalid override is ignored whole and logged without the person or the values.
    """
    if not overrides:
        return base
    try:
        patched = RegisterParams.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        log.warning("register override ignored: %d invalid field(s)", exc.error_count())
        return base
    tightened: dict[str, bool] = {}
    for name in overrides:
        match _guarded(name):
            case "false":
                tightened[name] = bool(getattr(base, name) and getattr(patched, name))
            case "true":
                tightened[name] = bool(getattr(base, name) or getattr(patched, name))
    return patched.model_copy(update=tightened)


def resolve_person(person: Person, cfg: RegisterConfig) -> tuple[str, RegisterParams]:
    """Group and register of one person: declared age, else a confident estimate, else
    ``unknown``; then the person's overrides."""
    group = person.age_group
    if person.age_source not in ("declared", "estimated") or group not in GROUPS:
        group = UNKNOWN
    return group, apply_overrides(getattr(cfg, group), person.overrides)


def _rank(group: str, params: RegisterParams) -> tuple[bool, bool, float, int, int]:
    """Larger is more restrictive: minor policy, no approach, lower amplitude, lower reaction
    intensity, then the stricter default group."""
    return (
        params.minor_content_policy,
        not params.approach_allowed,
        -params.motion_amplitude,
        -REACTION_LEVELS.index(params.reaction_intensity),
        -GROUP_ORDER.index(group),
    )


def most_restrictive(registers: Sequence[tuple[str, RegisterParams]]) -> tuple[str, RegisterParams]:
    """Combine the registers of everyone present (SPEC 7).

    Each safety-relevant field takes its most restrictive value across the people: the minor
    content policy if any person has it, approach only if every person allows it, the lowest
    motion amplitude, the lowest reaction intensity. The fields with no restrictiveness order
    (speech rate, pitch, vocabulary, persona fragment) come from the most restrictive person,
    ranked by those four criteria in that order, ties going to the stricter default group; that
    person's group names the result.
    """
    group, lead = max(registers, key=lambda gp: _rank(*gp))
    params = lead.model_copy(
        update={
            "minor_content_policy": any(p.minor_content_policy for _, p in registers),
            "approach_allowed": all(p.approach_allowed for _, p in registers),
            "motion_amplitude": min(p.motion_amplitude for _, p in registers),
            "reaction_intensity": REACTION_LEVELS[
                min(REACTION_LEVELS.index(p.reaction_intensity) for _, p in registers)
            ],
        }
    )
    return group, params


def resolve_register(
    people: Sequence[Person],
    cfg: RegisterConfig,
    persona: PersonaConfig | PersonaEffects,
    *,
    minor_present: bool = False,
) -> ActiveRegister:
    """The register for ``people``: per person, most restrictive, persona-shaped once.

    ``minor_present`` is the motion loop's minor flag (``MinorPresence``): while it is set the
    minor content policy applies even if everyone in view is an adult. The minor motion cap is
    not applied here: it always wins at the arbiter, after this register scaled the motion.
    """
    fx = persona if isinstance(persona, PersonaEffects) else persona_effects(persona)
    resolved = [resolve_person(p, cfg) for p in people] or [(UNKNOWN, cfg.unknown)]
    group, merged = most_restrictive(resolved)
    shaped = apply_to_register(merged, fx)
    if minor_present and not shaped.minor_content_policy:
        shaped = shaped.model_copy(update={"minor_content_policy": True})
    return ActiveRegister(group, shaped, len(people))


@dataclass
class _Seen:
    age_group: str
    age_source: str
    owner_id: str | None
    stamp: float


class RegisterTracker:
    """People in view, from ``percept.face``, resolved to the register in effect.

    A face track stays in the set for ``max(expression.register_hold_s, register_hold_periods x
    detection period)`` after its last sighting, so a throttled detector does not drop a person
    between two frames. ``overrides_for(person_id)`` returns the record's ``register_overrides``
    (``{}`` for an unknown id). Handlers run on the publishing thread; :meth:`tick` runs on the
    app loop. One lock guards the set.
    """

    def __init__(
        self,
        bus: Bus,
        cfg: ExpressionConfig,
        *,
        overrides_for: Callable[[str], Mapping[str, Any]] = lambda _id: {},
        detection_period: Callable[[], float] = lambda: math.inf,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self._overrides_for = overrides_for
        self._period = detection_period
        self._seen: dict[int, _Seen] = {}
        self._lock = threading.Lock()
        self.active: ActiveRegister | None = None
        self._unsubscribe = bus.subscribe("percept.face", self._on_face)

    def close(self) -> None:
        """Stop listening."""
        self._unsubscribe()

    def lifetime_s(self) -> float:
        """How long a track survives unseen at the current detection rate."""
        period = self._period()
        hold = self.cfg.register_hold_s
        return max(hold, self.cfg.register_hold_periods * period) if math.isfinite(period) else hold

    def _on_face(self, msg: Message) -> None:
        p = msg.payload
        with self._lock:
            self._seen[p.track_id] = _Seen(p.age_group, p.age_source, p.owner_id, msg.stamp)

    def people(self, now: float) -> list[Person]:
        """The people in view now; expired tracks are dropped."""
        lifetime = self.lifetime_s()
        with self._lock:
            self._seen = {t: s for t, s in self._seen.items() if now - s.stamp < lifetime}
            seen = list(self._seen.values())
        return [
            Person(s.age_group, s.age_source, self._lookup(s.owner_id) if s.owner_id else {})
            for s in seen
        ]

    def _lookup(self, person_id: str) -> Mapping[str, Any]:
        try:
            return self._overrides_for(person_id)
        except Exception as exc:  # a missing record or registry must not stop the app tick
            log.warning("register overrides unavailable: %s", type(exc).__name__)
            return {}

    def tick(
        self,
        now: float,
        cfg: RegisterConfig,
        persona: PersonaConfig | PersonaEffects,
        *,
        minor_present: bool = False,
    ) -> ActiveRegister:
        """Resolve the register for this tick; publish it on ``register.active`` if it changed."""
        active = resolve_register(self.people(now), cfg, persona, minor_present=minor_present)
        if active != self.active:
            self.active = active
            self.bus.publish(TOPIC, active, now)
        return active
