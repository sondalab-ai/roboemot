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

the accent also x ``mood_gain[mood]``; the product of the factors (mood included) is capped by
``expression.max_amplitude`` and each offset is clamped to the manifest range about home.

Offsets ramp in and out over ``fade_ms`` (the arbiter cross-fades only on a change of winner, so
an additive source that appeared or vanished would step). A barge-in (the timeline's last cut
moved), a disabled layer or a circadian factor of 0 freezes the offsets and ramps them to zero.
The tick the ramp reaches zero sends one last command with every channel at 0, so the arbiter
holds no residue until the previous command's TTL. Channels the playing expression drives fade
out the same way, so the two do not double up. Never moves the body yaw or a translation: a map
naming one is refused at load and by :meth:`SpeakingRenderer.apply_config`.

Threads: the app tick calls everything except :meth:`SpeakingRenderer.apply_config`, which the
config reload calls from the thread that changed the configuration. It swaps one immutable
bundle; each tick reads the bundle once, so a tick never mixes old and new settings.
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
from ..voice.envelope import SpeechEvent, SpeechTimeline, TimelineView
from . import AffectState
from .expression import build_primitive
from .register import ActiveRegister

SOURCE = "speaking"
MS_PER_S = 1000.0
GAIN_EPSILON = 1e-9  # a fade gain below this is zero (float rounding of the per-tick steps)
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


@dataclass(frozen=True)
class _Settings:
    """Everything :meth:`SpeakingRenderer.apply_config` sets, swapped as one object."""

    cfg: SpeechMotionConfig
    max_amplitude: float
    beat: dict[str, int]
    sway: dict[str, int]
    shapes: dict[str, _Shape | None]

    def lead(self, role: str) -> float:
        """How long before its event a gesture starts: its peak time (the tilt: its hold)."""
        shape = self.shapes.get(role)
        if shape is None or role == "phrase_nod":
            return 0.0
        return shape.hold_at if shape.hold_at is not None else shape.peak_t


@dataclass
class _Play:
    """One gesture playing, on the timeline clock; ``gain`` is radians per unit of the shape and
    ``mood`` the accent's mood gain (1 for the others), applied under the amplitude cap."""

    shape: _Shape
    start: float
    gain: float
    mood: float = 1.0
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
    ``circadian_factor`` is a hook like the expression renderer's.
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
        self._sent: frozenset[str] = frozenset()  # channels of the last command sent
        self._s = _State()
        self.counts = dict.fromkeys((*GESTURES, "late", "skipped", "cuts"), 0)
        self.apply_config(cfg, speech_map, max_amplitude)

    # ------------------------------------------------------------------ config and inputs

    def apply_config(
        self, cfg: SpeechMotionConfig, speech_map: SpeechMotionMap, max_amplitude: float
    ) -> None:
        """Hot reload, from any thread: the preset and every field apply from the next tick; a
        gesture already playing keeps its shape. ValueError, and the old settings kept, when the
        map names the body yaw or a translation."""
        for role, names in speech_map.channels().items():
            for name in names:
                dof = self.dofs.get(name)
                if name == self.body_yaw or (dof is not None and dof.kind != "angle"):
                    raise ValueError(f"speech_motion_map.{role}: {name!r} may not move")
        self._b = _Settings(
            cfg,
            max_amplitude,
            self._signs(speech_map.beat),
            self._signs(speech_map.sway),
            {
                role: self._shape(role, entry)
                for role in GESTURES
                if (entry := getattr(speech_map, role)) is not None
            },
        )

    @property
    def cfg(self) -> SpeechMotionConfig:
        return self._b.cfg

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

    def scale(self, b: _Settings | None = None) -> float:
        """The amplitude factor now: register x circadian x arousal, capped."""
        b = b or self._b
        arousal = max(0.0, 1 + b.cfg.level().arousal_gain * self._arousal)
        raw = self._motion_amplitude * self.circadian_factor() * arousal
        return max(0.0, min(b.max_amplitude, raw))

    # ------------------------------------------------------------------ the tick

    def commands(self, now: float) -> list[Command]:
        """The speaking offset for this tick, to feed to the motion loop (empty when idle)."""
        b, s = self._b, self._s
        cfg = b.cfg
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        tl = self.timeline()
        t = now + cfg.latency_ms / MS_PER_S
        view = self._view(tl, t, b) if tl is not None else None
        cut = view.last_cut if view is not None else None
        if cut != self._cut_seen:
            self._cut_seen = cut
            if cut is not None:
                self.counts["cuts"] += 1
                self._last_accent = -math.inf
                self._freeze()
        live = cfg.enabled and view is not None and self.scale(b) > 0
        if not live:
            self._freeze()
        if s.frozen is not None or view is None:
            raw, active = dict(s.frozen or {}), False
        else:
            raw, active = self._render(view, b)
        fade_s = cfg.fade_ms / MS_PER_S
        step = dt / fade_s if fade_s > 0 else 1.0
        target = 1.0 if active else 0.0
        s.gain = min(target, s.gain + step) if target > s.gain else max(target, s.gain - step)
        if s.gain <= GAIN_EPSILON:
            self._s = _State()
            self._plays.clear()
            return self._emit(now, cfg, {})  # a last zero for every channel still held
        s.raw = raw
        ducked = self.ducked()
        w = min_jerk(s.gain)
        out: dict[str, float] = {}
        for n, v in raw.items():
            d = s.duck.get(n, 1.0)
            d = max(0.0, d - step) if n in ducked else min(1.0, d + step)
            s.duck[n] = d
            out[n] = self._clamp(n, v * w * d)
        return self._emit(now, cfg, out)

    def _emit(self, now: float, cfg: SpeechMotionConfig, out: dict[str, float]) -> list[Command]:
        """The command for ``out``, with 0 for a channel the previous command set and this one
        does not: the arbiter keeps a source's last command until it expires, so a channel left
        out would hold its last offset for the TTL and then step."""
        targets = dict.fromkeys(self._sent - set(out), 0.0) | out
        self._sent = frozenset(out)
        if not targets:
            return []
        return [
            Command(
                SOURCE,
                Priority.EXPRESSION,
                targets,
                now,
                expires=now + cfg.command_ttl_s,
                additive=True,
            )
        ]

    def _view(self, tl: SpeechTimeline, t: float, b: _Settings) -> TimelineView:
        s, late = self._s, b.cfg.event_max_late_s
        lead = max((b.lead(role) for role in b.shapes), default=0.0)
        return tl.view(
            t,
            frames_from=s.beat_t if s.beat_t is not None and s.beat_t <= t else t,
            events_from=t - late,
            events_to=t + lead + b.cfg.envelope_frame_ms / MS_PER_S,
            known=self._norm_key,
        )

    def _freeze(self) -> None:
        """Stop following the timeline: drop the gestures, fade the last offsets out."""
        s = self._s
        if s.frozen is None:
            s.frozen = dict(s.raw)
        self._plays.clear()

    def _render(self, view: TimelineView, b: _Settings) -> tuple[dict[str, float], bool]:
        """The offsets at timeline time ``view.t`` before the fade, and whether speech is
        active."""
        t, level = view.t, b.cfg.level()
        self._events(view, b, level)
        k = self.scale(b)
        raw: dict[str, float] = {}

        def add(name: str, value: float) -> None:
            raw[name] = raw.get(name, 0.0) + value

        x = self._beat_step(view, b)
        for n, sign in b.beat.items():
            add(n, sign * math.radians(level.beat_deg) * x * k)
        if b.sway:
            want = self._s.side * math.radians(level.sway_deg) if view.sentence else 0.0
            tau = b.cfg.sway_smoothing_ms / MS_PER_S
            y = self._s.sway.step(want, self._sway_dt(t), tau, tau)
            for n, sign in b.sway.items():
                add(n, sign * y * k)
        for role, play in list(self._plays.items()):
            holding = play.shape.hold_at is not None and play.released is None
            if holding and (self.listening() or play.held_for(t) >= b.cfg.question_hold_s):
                play.released = t
            if play.done(t):
                del self._plays[role]
                continue
            factor = min(b.max_amplitude, k * play.mood)  # mood under the cap too
            for n, v in play.offsets(t).items():
                add(n, v * factor)
        return raw, view.sentence is not None or bool(self._plays)

    def _sway_dt(self, t: float) -> float:
        last, self._s.sway_t = self._s.sway_t, t
        return 0.0 if last is None else max(0.0, t - last)

    # ------------------------------------------------------------------ beat

    def _beat_step(self, view: TimelineView, b: _Settings) -> float:
        """Advance the beat smoother over the frames since the last tick, at frame rate."""
        s, cfg, t = self._s, b.cfg, view.t
        rise, fall = cfg.attack_ms / MS_PER_S, cfg.release_ms / MS_PER_S
        if s.beat_t is None or t < s.beat_t:
            s.beat_t = t
        norm = self._normaliser(view, cfg)
        cur = s.beat_t
        for start, db in view.frames:
            if start < cur:
                continue
            s.beat.step(s.x, start - cur, rise, fall)
            cur, s.x = start, self._normalise(db, norm)
        y = s.beat.step(s.x, t - cur, rise, fall)
        s.beat_t = t
        if view.level is None:
            s.x = 0.0
        return y

    def _normaliser(
        self, view: TimelineView, cfg: SpeechMotionConfig
    ) -> tuple[float, float] | None:
        """``(median, span)`` of the sentence's voiced frames, cached while it does not grow."""
        sentence = view.sentence
        if sentence is None:
            return None
        key = (sentence.id, sentence.end_s)
        if key != self._norm_key and view.sentence_dbfs is not None:
            voiced = sorted(db for db in view.sentence_dbfs if db >= cfg.silence_dbfs)
            self._norm_key, self._norm = key, None
            if voiced:
                n = len(voiced)
                mid = n // 2
                median = voiced[mid] if n % 2 else (voiced[mid - 1] + voiced[mid]) / 2
                self._norm = (median, max(voiced[-1] - median, cfg.emphasis_db))
        return self._norm

    @staticmethod
    def _normalise(db: float, norm: tuple[float, float] | None) -> float:
        if norm is None:
            return 0.0
        median, span = norm
        return max(0.0, min(1.0, (db - median) / span))

    # ------------------------------------------------------------------ gestures

    def _events(self, view: TimelineView, b: _Settings, level: SpeechMotionLevel) -> None:
        t, late = view.t, b.cfg.event_max_late_s
        self._handled = {k: at for k, at in self._handled.items() if at >= t - late}
        for e in view.events:
            if e.kind in ("phrase_gap", "sentence_start") and e.at_s <= t:
                self._once(("sway", e), e, self._flip)
            role = self._role(view, b, e)
            if role is None:
                continue
            key = (role, e)
            if key in self._handled or t < e.at_s - b.lead(role):
                continue
            deadline = e.at_s + late
            if e.kind == "phrase_gap":
                deadline = min(deadline, e.until_s)
            self._handled[key] = e.at_s
            if t > deadline:
                self.counts["late"] += 1
                continue
            self._start(role, e, t, b, level, view)

    def _once(self, key: tuple, e: SpeechEvent, action: Callable[[], None]) -> None:
        if key not in self._handled:
            self._handled[key] = e.at_s
            action()

    def _flip(self) -> None:
        self._s.side = -self._s.side

    @staticmethod
    def _role(view: TimelineView, b: _Settings, e: SpeechEvent) -> str | None:
        """The gesture ``e`` starts (None: none, or the map has no shape for it)."""
        sentence = view.sentences.get(e.sentence)
        if sentence is None or sentence.cut:
            return None
        role: str | None = None
        if e.kind == "phrase_gap":
            role = "phrase_nod"
        elif e.kind == "sentence_end":
            role = "question_tilt" if sentence.question else "phrase_nod"
        elif e.kind == "sentence_start" and not sentence.continued:
            role = "breath"
        elif e.kind == "emphasis":
            role = "accent"
        if role is None or b.shapes.get(role) is None:
            return None
        return role

    def _start(
        self,
        role: str,
        e: SpeechEvent,
        t: float,
        b: _Settings,
        level: SpeechMotionLevel,
        view: TimelineView,
    ) -> None:
        shape = b.shapes.get(role)
        deg = getattr(level, ROLE_DEGREES[role])
        if shape is None or deg <= 0:
            return
        if role == "accent" and e.at_s - self._last_accent < level.emphasis_min_interval_s:
            return
        if role == "phrase_nod" and not self._chosen(e, level.nod_share):
            return
        if role in self._plays:
            self.counts["skipped"] += 1
            return
        mood = 1.0
        if role == "accent":
            self._last_accent = e.at_s
            label = view.sentences[e.sentence].mood
            mood = b.cfg.mood_gain.get(label, 1.0) if label is not None else 1.0
        self._plays[role] = _Play(shape, t, math.radians(deg) / shape.peak, mood)
        self.counts[role] += 1

    @staticmethod
    def _chosen(e: SpeechEvent, share: float) -> bool:
        """Seeded by the reply and the phrase: the same reply nods at the same places."""
        return random.Random(f"{e.utterance}:{e.kind}:{e.sentence}:{e.index}").random() < share

    # ------------------------------------------------------------------ output

    def _clamp(self, name: str, offset: float) -> float:
        dof = self.dofs[name]
        return max(dof.min - dof.home, min(dof.max - dof.home, offset))

    def snapshot(self) -> dict[str, int]:
        """Counts since start for the debug report (SPEC 12A): gestures per role, late events,
        gestures skipped because one was playing, cuts. Integers only."""
        return dict(self.counts)


__all__ = ["SOURCE", "SpeakingRenderer"]
