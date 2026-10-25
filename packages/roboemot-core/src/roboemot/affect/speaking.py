"""Speaking motion: small movements while the robot speaks (SPEC 7, Speaking motion; SPEC 10).

The renderer reads the speech timeline (``roboemot.voice.envelope``) at ``now + latency_ms`` and
plays one additive offset at EXPRESSION priority, fed from the app tick right after the
expression renderer. Roles, bound to channels by the manifest's ``speech_motion_map`` (core names
no joint):

- ``beat``: follows the energy of the speech above the sentence's median, through a two-pole
  smoother (``attack_ms``, ``release_ms``): a small dip on stressed stretches, not on syllables;
- ``phrase_nod``: a nod in the pause after a phrase or a statement, for a seeded share
  (``nod_share``) of them, so the same reply moves the same way;
- ``question_tilt``: at the end of a question, held at ``hold_at_s`` until the listener hears an
  utterance, ``question_hold_s`` passes or a barge-in; then its return plays;
- ``accent``: a flick on emphasis peaks, at most one per the preset's
  ``emphasis_min_interval_s``, scaled by ``mood_gain`` of the sentence's mood;
- ``breath``: an inhale lift before each sentence (not before a continued piece);
- ``sway``: a very slow drift that changes side at each phrase.

Gestures lead their event by the time of their peak (the tilt by ``hold_at_s``), so the accent
peaks on the stress and the tilt is held when the question ends; the nod starts in the pause.
An event that reaches the layer late (a streamed voice writes a phrase gap when the voice
resumes) still plays up to ``event_max_late_s`` after its stamp, a phrase nod only while its
pause lasts.

Amplitude, re-read every tick so a stricter register lowers it at once::

    preset degrees x register.motion_amplitude x circadian factor x (1 + arousal_gain x arousal)

the accent also x ``mood_gain[mood]``; the product of the factors is capped by
``expression.max_amplitude`` and each offset is clamped to the manifest range about home.

Offsets ramp in and out over ``fade_ms`` (the arbiter cross-fades only on a change of winner, so
an additive source that appeared or vanished would step). A barge-in (the timeline's last cut
moved), a disabled layer or a circadian factor of 0 freezes the offsets and ramps them to zero.
Channels the playing expression drives fade out the same way, so the two do not double up.
Never moves the body yaw or a translation: the map is refused at load, and skipped here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..config.schema import (
    ExpressionEntry,
    SpeechMotionConfig,
    SpeechMotionLevel,
    SpeechMotionMap,
    SpeechTiltEntry,
)
from ..manifest import Dof
from ..motion.core import Command, Priority
from ..motion.curves import min_jerk
from ..motion.primitives import Primitive
from ..voice.envelope import SpeechEvent, SpeechSentence, SpeechTimeline
from . import AffectState
from .expression import build_primitive
from .register import ActiveRegister

SOURCE = "speaking"
MS_PER_S = 1000.0
GESTURES = ("phrase_nod", "question_tilt", "accent", "breath")
ROLE_DEGREES = {
    "phrase_nod": "phrase_nod_deg",
    "question_tilt": "question_tilt_deg",
    "accent": "accent_deg",
    "breath": "breath_deg",
}


@dataclass(frozen=True)
class _Shape:
    """A gesture's primitive (radians of the map's shape), its largest offset and when."""

    prim: Primitive
    peak: float
    peak_t: float
    hold_at: float | None


@dataclass
class _Play:
    """One gesture playing, on the timeline clock; ``gain`` is radians per unit of the shape."""

    shape: _Shape
    start: float
    gain: float
    released: float | None = None

    def at(self, t: float) -> float:
        e, hold = t - self.start, self.shape.hold_at
        if hold is None or e <= hold:
            return e
        if self.released is None:
            return hold
        return hold + max(0.0, t - max(self.released, self.start + hold))

    def held_for(self, t: float) -> float:
        """How long the tilt has been held at ``hold_at`` (0 before it got there)."""
        hold = self.shape.hold_at
        return 0.0 if hold is None else max(0.0, t - self.start - hold)

    def done(self, t: float) -> bool:
        return self.at(t) >= self.shape.prim.duration

    def offsets(self, t: float) -> dict[str, float]:
        return {n: v * self.gain for n, v in self.shape.prim.sample(self.at(t)).items()}


@dataclass
class _TwoPole:
    """Two cascaded one-pole filters with a rise and a fall time constant."""

    y1: float = 0.0
    y2: float = 0.0

    def step(self, x: float, dt: float, rise_s: float, fall_s: float) -> float:
        self.y1 = _toward(self.y1, x, dt, rise_s if x > self.y1 else fall_s)
        self.y2 = _toward(self.y2, self.y1, dt, rise_s if self.y1 > self.y2 else fall_s)
        return self.y2


def _toward(y: float, x: float, dt: float, tau_s: float) -> float:
    if dt <= 0:
        return y
    return x + (y - x) * math.exp(-dt / tau_s)


@dataclass
class _State:
    """Everything that is reset once the offsets are back at zero."""

    gain: float = 0.0
    beat: _TwoPole = field(default_factory=_TwoPole)
    beat_t: float | None = None
    x: float = 0.0
    sway: _TwoPole = field(default_factory=_TwoPole)
    sway_t: float | None = None
    side: int = 1
    frozen: dict[str, float] | None = None
    raw: dict[str, float] = field(default_factory=dict)
    duck: dict[str, float] = field(default_factory=dict)


class SpeakingRenderer:
    """Speaking motion (see the module docstring). Call :meth:`set_register` every app tick,
    :meth:`set_affect` with each new affect state and :meth:`commands` every app tick.

    ``timeline`` returns the speech timeline, or None while there is no speech output (it is
    built after assembly, so it is read lazily). ``listening`` says whether the listener has an
    utterance in progress; ``ducked`` returns the channels the playing expression drives.
    ``circadian_factor`` is a hook like the expression renderer's. Single-threaded: all calls
    come from the app loop.
    """

    def __init__(
        self,
        cfg: SpeechMotionConfig,
        speech_map: SpeechMotionMap,
        dofs: Mapping[str, Dof],
        *,
        body_yaw: str | None,
        max_amplitude: float,
        timeline: Callable[[], SpeechTimeline | None],
        listening: Callable[[], bool] = lambda: False,
        ducked: Callable[[], frozenset[str]] = frozenset,
    ) -> None:
        self.dofs, self.body_yaw = dofs, body_yaw
        self.timeline, self.listening, self.ducked = timeline, listening, ducked
        self.circadian_factor: Callable[[], float] = lambda: 1.0
        self._motion_amplitude = 0.0  # no motion before the first register
        self._arousal = 0.0
        self._last_now: float | None = None
        self._cut_seen: float | None = None
        self._plays: dict[str, _Play] = {}
        self._handled: dict[tuple, float] = {}
        self._last_accent = -math.inf
        self._norm_key: tuple[int, float] | None = None
        self._norm: tuple[float, float] | None = None
        self._s = _State()
        self.counts = dict.fromkeys((*GESTURES, "late", "skipped", "cuts"), 0)
        self.apply_config(cfg, speech_map, max_amplitude)

    # ------------------------------------------------------------------ config and inputs

    def apply_config(
        self, cfg: SpeechMotionConfig, speech_map: SpeechMotionMap, max_amplitude: float
    ) -> None:
        """Hot reload: the preset and every field apply from the next tick; a gesture already
        playing keeps its shape. ValueError when the map names the body yaw or a translation."""
        for role, names in speech_map.channels().items():
            for name in names:
                dof = self.dofs.get(name)
                if name == self.body_yaw or (dof is not None and dof.kind != "angle"):
                    raise ValueError(f"speech_motion_map.{role}: {name!r} may not move")
        self.cfg, self.max_amplitude = cfg, max_amplitude
        self._beat_signs = self._signs(speech_map.beat)
        self._sway_signs = self._signs(speech_map.sway)
        self._shapes = {
            role: self._shape(role, entry)
            for role in GESTURES
            if (entry := getattr(speech_map, role)) is not None
        }

    def set_register(self, register: ActiveRegister) -> None:
        """Take the register in effect now; it scales the very next offsets."""
        self._motion_amplitude = register.params.motion_amplitude

    def set_affect(self, state: AffectState) -> None:
        """Take the latest affect state (its arousal scales the amplitude)."""
        self._arousal = state.arousal

    def _signs(self, raw: Mapping[str, int]) -> dict[str, int]:
        return {n: s for n, s in raw.items() if n in self.dofs and self.dofs[n].kind == "angle"}

    def _shape(self, role: str, entry: ExpressionEntry) -> _Shape | None:
        prim = build_primitive(role, entry, self.dofs, body_yaw=self.body_yaw, body=False)
        if prim is None:
            return None
        k = max(prim.keyframes, key=lambda f: max(abs(v) for v in f.values.values()))
        peak = max(abs(v) for v in k.values.values())
        if peak <= 0:
            return None
        hold = entry.hold_at_s if isinstance(entry, SpeechTiltEntry) else None
        return _Shape(prim, peak, k.t, hold)

    def scale(self, level: SpeechMotionLevel | None = None) -> float:
        """The amplitude factor now: register x circadian x arousal, capped."""
        level = level or self.cfg.level()
        arousal = max(0.0, 1 + level.arousal_gain * self._arousal)
        raw = self._motion_amplitude * self.circadian_factor() * arousal
        return max(0.0, min(self.max_amplitude, raw))

    # ------------------------------------------------------------------ the tick

    def commands(self, now: float) -> list[Command]:
        """The speaking offset for this tick, to feed to the motion loop (empty when idle)."""
        cfg, s = self.cfg, self._s
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        tl = self.timeline()
        t = now + cfg.latency_ms / MS_PER_S
        cut = tl.last_cut() if tl is not None else None
        if cut != self._cut_seen:
            self._cut_seen = cut
            if cut is not None:
                self.counts["cuts"] += 1
                self._freeze()
        level = cfg.level()
        live = cfg.enabled and tl is not None and self.scale(level) > 0
        if not live:
            self._freeze()
        if s.frozen is not None or tl is None:
            raw, active = dict(s.frozen or {}), False
        else:
            raw, active = self._render(tl, t, level)
        fade_s = cfg.fade_ms / MS_PER_S
        step = dt / fade_s if fade_s > 0 else 1.0
        target = 1.0 if active else 0.0
        s.gain = min(target, s.gain + step) if target > s.gain else max(target, s.gain - step)
        if s.gain <= 0:
            self._s = _State()
            self._plays.clear()
            return []
        s.raw = raw
        ducked = self.ducked()
        w = min_jerk(s.gain)
        out: dict[str, float] = {}
        for n, v in raw.items():
            d = s.duck.get(n, 1.0)
            d = max(0.0, d - step) if n in ducked else min(1.0, d + step)
            s.duck[n] = d
            out[n] = self._clamp(n, v * w * d)
        if not out:
            return []
        return [
            Command(
                SOURCE,
                Priority.EXPRESSION,
                out,
                now,
                expires=now + cfg.command_ttl_s,
                additive=True,
            )
        ]

    def _freeze(self) -> None:
        """Stop following the timeline: drop the gestures, fade the last offsets out."""
        s = self._s
        if s.frozen is None:
            s.frozen = dict(s.raw)
        self._plays.clear()

    def _render(
        self, tl: SpeechTimeline, t: float, level: SpeechMotionLevel
    ) -> tuple[dict[str, float], bool]:
        """The offsets at timeline time ``t`` before the fade, and whether speech is active."""
        sentence = tl.sentence_at(t)
        self._events(tl, t, level)
        raw: dict[str, float] = {}

        def add(name: str, value: float) -> None:
            raw[name] = raw.get(name, 0.0) + value

        x = self._beat_step(tl, t, sentence)
        for n, sign in self._beat_signs.items():
            add(n, sign * math.radians(level.beat_deg) * x)
        if self._sway_signs:
            want = self._s.side * math.radians(level.sway_deg) if sentence is not None else 0.0
            tau = self.cfg.sway_smoothing_ms / MS_PER_S
            dt = self._sway_dt(t)
            y = self._s.sway.step(want, dt, tau, tau)
            for n, sign in self._sway_signs.items():
                add(n, sign * y)
        for role, play in list(self._plays.items()):
            if play.shape.hold_at is not None and play.released is None:
                held = play.held_for(t)
                if self.listening() or held >= self.cfg.question_hold_s:
                    play.released = t
            if play.done(t):
                del self._plays[role]
                continue
            for n, v in play.offsets(t).items():
                add(n, v)
        k = self.scale(level)
        return {n: v * k for n, v in raw.items()}, sentence is not None or bool(self._plays)

    def _sway_dt(self, t: float) -> float:
        last, self._s.sway_t = self._s.sway_t, t
        return 0.0 if last is None else max(0.0, t - last)

    # ------------------------------------------------------------------ beat

    def _beat_step(self, tl: SpeechTimeline, t: float, sentence: SpeechSentence | None) -> float:
        """Advance the beat smoother over the frames since the last tick, at frame rate."""
        s, cfg = self._s, self.cfg
        rise, fall = cfg.attack_ms / MS_PER_S, cfg.release_ms / MS_PER_S
        if s.beat_t is None or t < s.beat_t:
            s.beat_t = t
        norm = self._normaliser(tl, sentence)
        cur = s.beat_t
        for start, db in tl.frames_between(s.beat_t, t):
            s.beat.step(s.x, start - cur, rise, fall)
            cur, s.x = start, self._normalise(db, norm)
        y = s.beat.step(s.x, t - cur, rise, fall)
        s.beat_t = t
        if tl.level_at(t) is None:
            s.x = 0.0
        return y

    def _normaliser(
        self, tl: SpeechTimeline, sentence: SpeechSentence | None
    ) -> tuple[float, float] | None:
        """``(median, span)`` of the sentence's voiced frames, cached while it does not grow."""
        if sentence is None:
            return None
        key = (sentence.id, sentence.end_s)
        if key != self._norm_key:
            frames = tl.frames_between(sentence.start_s, sentence.end_s)
            voiced = sorted(db for _, db in frames if db >= self.cfg.silence_dbfs)
            self._norm_key, self._norm = key, None
            if voiced:
                n = len(voiced)
                mid = n // 2
                median = voiced[mid] if n % 2 else (voiced[mid - 1] + voiced[mid]) / 2
                self._norm = (median, max(voiced[-1] - median, self.cfg.emphasis_db))
        return self._norm

    @staticmethod
    def _normalise(db: float, norm: tuple[float, float] | None) -> float:
        if norm is None:
            return 0.0
        median, span = norm
        return max(0.0, min(1.0, (db - median) / span))

    # ------------------------------------------------------------------ gestures

    def _events(self, tl: SpeechTimeline, t: float, level: SpeechMotionLevel) -> None:
        late = self.cfg.event_max_late_s
        leads = [sh.hold_at or sh.peak_t for sh in self._shapes.values() if sh is not None]
        horizon = t + max(leads, default=0.0)
        self._handled = {k: at for k, at in self._handled.items() if at >= t - late}
        for e in tl.events_between(t - late, horizon + self._frame_s()):
            if e.kind in ("phrase_gap", "sentence_start") and e.at_s <= t:
                self._once(("sway", e), e, self._flip)
            role, lead, sentence = self._role(tl, e)
            if role is None:
                continue
            key = (role, e)
            if key in self._handled or t < e.at_s - lead:
                continue
            deadline = e.at_s + late
            if e.kind == "phrase_gap":
                deadline = min(deadline, e.until_s)
            if t > deadline:
                self._handled[key] = e.at_s
                self.counts["late"] += 1
                continue
            self._handled[key] = e.at_s
            self._start(role, e, t, level, sentence)

    def _frame_s(self) -> float:
        return self.cfg.envelope_frame_ms / MS_PER_S

    def _once(self, key: tuple, e: SpeechEvent, action: Callable[[], None]) -> None:
        if key not in self._handled:
            self._handled[key] = e.at_s
            action()

    def _flip(self) -> None:
        self._s.side = -self._s.side

    def _role(
        self, tl: SpeechTimeline, e: SpeechEvent
    ) -> tuple[str | None, float, SpeechSentence | None]:
        """The gesture ``e`` starts, how long before its stamp, and its sentence."""
        sentence = tl.sentence(e.sentence)
        if sentence is None or sentence.cut:
            return None, 0.0, None
        role: str | None = None
        if e.kind == "phrase_gap":
            role = "phrase_nod"
        elif e.kind == "sentence_end":
            role = "question_tilt" if sentence.question else "phrase_nod"
        elif e.kind == "sentence_start" and not sentence.continued:
            role = "breath"
        elif e.kind == "emphasis":
            role = "accent"
        shape = self._shapes.get(role) if role is not None else None
        if shape is None:
            return None, 0.0, None
        lead = 0.0 if role == "phrase_nod" else (shape.hold_at or shape.peak_t)
        return role, lead, sentence

    def _start(
        self,
        role: str,
        e: SpeechEvent,
        t: float,
        level: SpeechMotionLevel,
        sentence: SpeechSentence | None,
    ) -> None:
        shape = self._shapes[role]
        deg = getattr(level, ROLE_DEGREES[role])
        if shape is None or deg <= 0:
            return
        if role == "accent":
            if e.at_s - self._last_accent < level.emphasis_min_interval_s:
                return
            self._last_accent = e.at_s
        if role == "phrase_nod" and not self._chosen(e, level.nod_share):
            return
        if role in self._plays:
            self.counts["skipped"] += 1
            return
        gain = math.radians(deg) / shape.peak
        if role == "accent":
            mood = sentence.mood if sentence is not None else None
            gain *= self.cfg.mood_gain.get(mood, 1.0) if mood is not None else 1.0
        self._plays[role] = _Play(shape, t, gain)
        self.counts[role] += 1

    @staticmethod
    def _chosen(e: SpeechEvent, share: float) -> bool:
        """Seeded by the reply and the phrase: the same reply nods at the same places."""
        return random.Random(f"{e.utterance}:{e.kind}:{e.sentence}:{e.index}").random() < share

    # ------------------------------------------------------------------ output

    def _clamp(self, name: str, offset: float) -> float:
        dof = self.dofs[name]
        return max(dof.min - dof.home, min(dof.max - dof.home, offset))

    @property
    def moving(self) -> bool:
        """Whether the layer is feeding offsets right now."""
        return self._s.gain > 0

    def snapshot(self) -> dict[str, int]:
        """Counts since start for the debug report (SPEC 12A): gestures per role, late events,
        gestures skipped because one was playing, cuts. Integers only."""
        return dict(self.counts)


__all__ = ["SOURCE", "SpeakingRenderer"]
