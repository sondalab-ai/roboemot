"""The empathic register in effect (SPEC 7, Register selection).

The robot has one register for everyone (``register``; no age groups, owner decision
2026-10-23). :class:`RegisterTracker` keeps the set of people in view from ``percept.face`` and
:func:`resolve_register` turns that set into the register in effect:

1. per person: the record's ``register_overrides`` laid over the register (guarded fields only
   ever tighter); a stranger gets the register as it is;
2. the most restrictive register of the people present (see :func:`most_restrictive`);
3. the persona's traits shape it once (``apply_to_register``).

Nobody in view resolves to the register itself. The result is published on ``register.active``
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
from ..config.stack import DROP, without_dropped
from ..persona import REACTION_LEVELS, PersonaEffects, apply_to_register
from ..persona import effects as persona_effects

log = logging.getLogger(__name__)

TOPIC = "register.active"


@dataclass(frozen=True)
class Person:
    """One person in view, as the resolver needs them. ``overrides`` is the record's partial
    register (empty for a stranger). No id."""

    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ActiveRegister:
    """The register in effect: the shaped parameters and how many people it covers. ``params``
    includes the persona fragment: it is app configuration text (not about a person) that the
    prompt builder needs with the rest of the register, and nothing here carries a person id."""

    params: RegisterParams
    people: int

    def summary(self) -> dict[str, Any]:
        """Parameters for the debug report; no person id, no free text."""
        params = self.params.model_dump(mode="json", exclude={"persona_fragment"})
        return {"people": self.people, **params}


def _guarded(name: str) -> str | None:
    """The safe direction of a guarded register field (``false``/``true``), else None."""
    extra = RegisterParams.model_fields[name].json_schema_extra
    return extra.get("conservative") if isinstance(extra, dict) else None


def _valid_guarded(base: RegisterParams, overrides: Mapping[str, Any]) -> dict[str, bool]:
    """The guarded fields of an otherwise invalid override that still tighten ``base``: a
    restriction is never lost because another field of the same override is wrong."""
    out: dict[str, bool] = {}
    for name, value in overrides.items():
        if name in RegisterParams.model_fields and isinstance(value, bool):
            match _guarded(name):
                case "false":
                    out[name] = bool(getattr(base, name) and value)
                case "true":
                    out[name] = bool(getattr(base, name) or value)
    return out


def apply_overrides(base: RegisterParams, overrides: Mapping[str, Any]) -> RegisterParams:
    """Lay a person's partial register over ``base``.

    A guarded field (``approach_allowed``) moves only toward its safe value, so a record cannot
    loosen what the register says (SPEC 11A, Guardrails). A field removed from the register
    (``moved_settings.yaml``, ``dropped``) is ignored. An invalid override is ignored whole and
    logged without the person or the values.
    """
    overrides, _ = without_dropped("register", dict(overrides))
    if overrides is DROP or not overrides:
        return base
    try:
        patched = RegisterParams.model_validate({**base.model_dump(), **overrides})
    except ValidationError as exc:
        log.warning("register override partly ignored: %d invalid field(s)", exc.error_count())
        return base.model_copy(update=_valid_guarded(base, overrides))
    tightened: dict[str, bool] = {}
    for name in overrides:
        match _guarded(name):
            case "false":
                tightened[name] = bool(getattr(base, name) and getattr(patched, name))
            case "true":
                tightened[name] = bool(getattr(base, name) or getattr(patched, name))
    return patched.model_copy(update=tightened)


def _rank(params: RegisterParams) -> tuple[bool, float, int]:
    """Larger is more restrictive: no approach, lower amplitude, lower reaction intensity."""
    return (
        not params.approach_allowed,
        -params.motion_amplitude,
        -REACTION_LEVELS.index(params.reaction_intensity),
    )


def most_restrictive(registers: Sequence[RegisterParams]) -> RegisterParams:
    """Combine the registers of everyone present (SPEC 7).

    Each safety-relevant field takes its most restrictive value across the people: approach
    only if every person allows it, the lowest motion amplitude, the lowest reaction intensity,
    the shortest reply cap that is set. The fields with no restrictiveness order (speech rate,
    pitch, vocabulary, persona fragment) come from the most restrictive person, ranked by those
    three criteria in that order, ties going to the first.
    """
    lead = max(registers, key=_rank)
    return lead.model_copy(
        update={
            "approach_allowed": all(p.approach_allowed for p in registers),
            "motion_amplitude": min(p.motion_amplitude for p in registers),
            "reaction_intensity": REACTION_LEVELS[
                min(REACTION_LEVELS.index(p.reaction_intensity) for p in registers)
            ],
            "max_reply_sentences": min(
                (p.max_reply_sentences for p in registers if p.max_reply_sentences is not None),
                default=None,
            ),
        }
    )


def resolve_register(
    people: Sequence[Person],
    cfg: RegisterConfig,
    persona: PersonaConfig | PersonaEffects,
) -> ActiveRegister:
    """The register for ``people``: each person's overrides over the register, the most
    restrictive of them, persona-shaped once."""
    fx = persona if isinstance(persona, PersonaEffects) else persona_effects(persona)
    base = cfg.params
    merged = most_restrictive([apply_overrides(base, p.overrides) for p in people] or [base])
    return ActiveRegister(apply_to_register(merged, fx), len(people))


@dataclass
class _Seen:
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
            self._seen[p.track_id] = _Seen(p.owner_id, msg.stamp)

    def people(self, now: float) -> list[Person]:
        """The people in view now; expired tracks are dropped."""
        lifetime = self.lifetime_s()
        with self._lock:
            self._seen = {t: s for t, s in self._seen.items() if now - s.stamp < lifetime}
            seen = list(self._seen.values())
        return [self._person(s) for s in seen]

    def _person(self, s: _Seen) -> Person:
        if s.owner_id is None:
            return Person({})
        try:
            return Person(self._overrides_for(s.owner_id))
        except Exception as exc:  # a missing record or a registry error must not stop the tick
            log.warning("register overrides unavailable: %s", type(exc).__name__)
            return Person({})

    def tick(
        self, now: float, cfg: RegisterConfig, persona: PersonaConfig | PersonaEffects
    ) -> ActiveRegister:
        """Resolve the register for this tick; publish it on ``register.active`` if it changed."""
        active = resolve_register(self.people(now), cfg, persona)
        if active != self.active:
            self.active = active
            self.bus.publish(TOPIC, active, now)
        return active
