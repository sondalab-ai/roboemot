"""Reflex engine (SPEC 10, Reflexes; budgets in SPEC 13; safety in SPEC 12).

Interim home: SPEC 10 compiles reflexes into ``roboemot-rt``. Until the C++ process and its bridge
land (M0) the rules, in the YAML format of ``reflexes`` (SPEC 11A), run here, on the motion loop's
tick, so a reflex waits for neither the 20 Hz app tick nor the affect engine. They are ported to
C++ later with the same semantics.

Flow. Producers on any thread offer scalars: a microphone level (``offer_audio_level``), a voice
bearing and a face size, the last two read off the bus (``attach``). Nothing but scalars is kept
(SPEC 12: no frame, audio or embedding is stored or logged). A rule's trigger is a rising edge of
its threshold; the trigger waits in a small inbox. ``on_tick`` runs inside ``MotionLoop.step_once``
(the loop's lock is held, so it feeds ``core`` directly and never calls the loop's methods): it
drops triggers older than ``reflexes.max_age_ms`` (``max_age_vision_ms`` for faces: the frame
stamp is the capture time), applies each rule's cooldown and minor
suppression, and starts a ``Player`` at ``Priority.REFLEX`` that expires after the rule's duration.

Safety. The commands go through the arbiter and the limiter like every other source: the minor
band and the manifest ranges still apply, and REFLEX is not additive. Nothing tips the head forward:
``startle`` and ``withdraw`` only deflect the head away (``ReflexMap.pitch_away_sign``);
``turn_toward`` answers a voice, stays in the front half-plane, within ``amplitude_deg`` of where
the head points, inside the manifest range and the head-body ``yaw_delta`` reach, is never started
while a possible minor is present, and is cut off on the first tick after one appears. The pose a
reflex starts from is un-scaled by the minor factor, because the arbiter scales the absolute target
about home: otherwise a head already tipped back would be pulled forward by the band.

Latency. Each firing records the time from the producer's stamp to the command being fed (the
SPEC 13 budget) and to the first pose change of the commanded output. Producers stamp at read
time, so a poll interval or the SDK stream delay before the read is not included.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ..attention.gaze import GazeMap, yaw_reach
from ..bus import Bus, Message
from ..config.schema import ReflexConfig, ReflexRule
from ..manifest import Constraint, Dof
from ..motion import Keyframe, MotionCore, Player, Primitive, Priority
from ..world.types import Observation

MS_PER_S = 1000.0
P50 = 0.5
P95 = 0.95
INBOX_MAX = 64
HALF_PLANE = math.pi / 2
SOURCE_PREFIX = "reflex."


@dataclass(frozen=True)
class SoundLoud:
    """Payload of ``sound.loud``: scalars only, never audio."""

    ratio: float  # level over the running floor
    level: float  # RMS, full scale 1.0


@dataclass(frozen=True)
class ReflexMap:
    """Which manifest dofs the primitives move; supplied by the robot plugin.

    ``antennas`` maps a dof to the sign of a lift (the startle raises each by that sign).
    ``pitch_away_sign`` is the sign of a head pitch that tips the head back, away from what is
    in front (negative on Reachy Mini: a positive pitch points the nose down).
    ``voice_sources`` names the ``Observation.source`` values that are speech directions.
    """

    head_yaw: str
    head_pitch: str
    body_yaw: str | None = None
    antennas: Mapping[str, float] = field(default_factory=dict)
    pitch_away_sign: float = -1.0
    voice_sources: tuple[str, ...] = ()


@dataclass
class _Pending:
    rule: str
    stamp: float
    value: float  # onset ratio, base-frame azimuth of the voice, or face height fraction


@dataclass
class _RuleStats:
    fired: int = 0
    stale: int = 0
    cooldown: int = 0
    minor: int = 0
    no_motion: int = 0


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _quantile(sorted_ms: list[float], q: float) -> float:
    return sorted_ms[max(0, math.ceil(q * len(sorted_ms)) - 1)]


class ReflexEngine:
    """Rule evaluation and reflex playback; see the module docstring."""

    def __init__(
        self,
        cfg: ReflexConfig,
        rmap: ReflexMap,
        dofs: Mapping[str, Dof],
        constraints: list[Constraint],
        *,
        margin_rad: float = 0.0,
        bus: Bus | None = None,
    ) -> None:
        names = [rmap.head_yaw, rmap.head_pitch, *rmap.antennas]
        names += [rmap.body_yaw] if rmap.body_yaw else []
        if missing := [n for n in names if n not in dofs]:
            raise KeyError(f"reflex dofs not in the manifest: {missing}")
        self.rmap, self.dofs, self.constraints, self.bus = rmap, dofs, constraints, bus
        self._gaze = GazeMap(rmap.head_yaw, rmap.head_pitch, rmap.body_yaw)
        self.audio_status: Callable[[], dict[str, Any]] | None = None
        self.bad_payloads = 0
        self._lock = threading.Lock()
        self._inbox: deque[_Pending] = deque(maxlen=INBOX_MAX)
        self._unsubscribe: list[Callable[[], None]] = []
        # audio
        self._floor: float | None = None
        self._audio_stamp = 0.0
        self._prev_ratio = 0.0
        self._last_event = -math.inf
        # voice and face
        self._last_voice: float | None = None
        self._face_stamp = -math.inf
        self._face_h = 0.0
        self._face_new = False
        self._face_armed: dict[str, bool] = {}
        # playback (loop thread only)
        self._players: dict[str, Player] = {}
        self._last_fire: dict[str, float] = {}
        self._moving: dict[str, tuple[float, str, dict[str, float]]] = {}
        self._stats: dict[str, _RuleStats] = {}
        self._lat: dict[str, dict[str, deque[float]]] = {}
        self.cfg = cfg
        self.margin_rad = margin_rad
        self._rules: list[ReflexRule] = list(cfg.rules)

    # ---- configuration

    def apply_config(self, cfg: ReflexConfig, margin_rad: float) -> None:
        """Hot reload (SPEC 11A): new rules and limits; running reflexes finish as started."""
        with self._lock:
            self.cfg, self.margin_rad, self._rules = cfg, margin_rad, list(cfg.rules)

    # ---- inputs (any thread; scalars only)

    def attach(self, bus: Bus) -> None:
        """Listen for voice bearings (``percept.entity``) and faces (``percept.face``, and
        ``percept.face_frame``: a processed frame without a face is a height of 0, which re-arms
        a close-face rule). Attach after ``MinorPresence``, so a failure here can never come
        before the minor latch in the bus's subscriber order."""
        self._unsubscribe += [
            bus.subscribe("percept.entity", self._on_entity),
            bus.subscribe("percept.face", self._on_face),
            bus.subscribe("percept.face_frame", self._on_face_frame),
        ]

    def close(self) -> None:
        """Stop listening."""
        for fn in self._unsubscribe:
            fn()
        self._unsubscribe.clear()

    def offer_audio_level(self, rms: float, stamp: float) -> None:
        """One microphone chunk's RMS (full scale 1.0) captured at ``stamp``.

        The onset is the level over a running floor that falls at once and rises with a half-life
        (``reflexes.audio``). Triggers are rising edges of each loud-onset rule's threshold;
        ``sound.loud`` goes on the bus on a rising edge of ``audio.loud_event_ratio``.
        """
        if not (math.isfinite(rms) and math.isfinite(stamp)) or rms < 0:
            return
        event: SoundLoud | None = None
        with self._lock:
            audio = self.cfg.audio
            floor = max(self._floor if self._floor is not None else rms, audio.min_rms)
            ratio = rms / floor
            prev = self._prev_ratio
            for rule in self._rules:
                if rule.enabled and rule.trigger == "loud_onset" and ratio >= rule.threshold > prev:
                    self._inbox.append(_Pending(rule.name, stamp, ratio))
            edge = ratio >= audio.loud_event_ratio > prev
            if edge and stamp - self._last_event >= audio.loud_event_refractory_s:
                self._last_event = stamp
                event = SoundLoud(ratio, rms)
            if rms <= floor:
                self._floor = rms
            else:
                dt = max(0.0, stamp - self._audio_stamp)
                alpha = 1.0 - 0.5 ** (dt / audio.floor_half_life_s)
                self._floor = floor + alpha * (rms - floor)
            self._audio_stamp, self._prev_ratio = stamp, ratio
        if event is not None and self.bus is not None:
            self.bus.publish("sound.loud", event, stamp)

    def offer_voice(self, stamp: float, azimuth_rad: float, head_yaw_rad: float = 0.0) -> None:
        """A speech direction: ``azimuth_rad`` is head-relative (positive left), ``head_yaw_rad``
        the head's yaw at capture, so their sum is the bearing in the base frame. A voice after
        a silence of at least a ``new_voice`` rule's threshold fires it."""
        if not all(map(math.isfinite, (stamp, azimuth_rad, head_yaw_rad))):
            return
        with self._lock:
            gap = math.inf if self._last_voice is None else stamp - self._last_voice
            self._last_voice = stamp if self._last_voice is None else max(self._last_voice, stamp)
            for rule in self._rules:
                if rule.enabled and rule.trigger == "new_voice" and gap >= rule.threshold:
                    self._inbox.append(_Pending(rule.name, stamp, azimuth_rad + head_yaw_rad))

    def offer_face_height(self, stamp: float, height: float) -> None:
        """A face's box height as a fraction of the frame; several faces of one frame (same
        stamp) count as the tallest."""
        if not (math.isfinite(stamp) and math.isfinite(height)):
            return
        with self._lock:
            if stamp > self._face_stamp:
                self._face_stamp, self._face_h, self._face_new = stamp, height, True
            elif stamp == self._face_stamp:
                self._face_h = max(self._face_h, height)

    # Bus handlers run inside the publisher (the vision thread): a bad payload is counted and
    # dropped, never raised into ``Bus.publish``.

    def _on_entity(self, msg: Message) -> None:
        try:
            obs = msg.payload
            if isinstance(obs, Observation) and obs.source in self.rmap.voice_sources:
                joints = msg.meta.get("joint_state") or {}
                head_yaw = float(joints.get(self.rmap.head_yaw, 0.0))
                self.offer_voice(obs.stamp, obs.azimuth_rad, head_yaw)
        except Exception:
            self.bad_payloads += 1

    def _on_face(self, msg: Message) -> None:
        try:
            _, y1, _, y2 = msg.payload.box
            self.offer_face_height(float(msg.stamp), float(y2) - float(y1))
        except Exception:
            self.bad_payloads += 1

    def _on_face_frame(self, msg: Message) -> None:
        try:
            self.offer_face_height(float(msg.stamp), 0.0)
        except Exception:
            self.bad_payloads += 1

    # ---- the motion loop's tick

    def on_tick(self, now: float, core: MotionCore) -> None:
        """Start due reflexes and advance the running ones. Called with the loop's lock held:
        it only uses ``core`` and never the loop's own methods."""
        triggers, cfg = self._collect(now)
        if not cfg.enabled or core.holding:
            # Disabled, or the watchdog holds the robot (the app stalled): no new reflex, and the
            # running ones end. The hold is at SAFETY priority and must stay (SPEC 10, 12).
            self._stop_all(core)
            return
        if core.minor_present:
            self._stop_minor_sensitive(core)
        for p in triggers:
            self._start(p, now, core, cfg)
        self._advance(now, core, cfg)

    def _collect(self, now: float) -> tuple[list[_Pending], ReflexConfig]:
        """Drain the inbox (dropping stale triggers) and add the close-face edges."""
        with self._lock:
            cfg = self.cfg
            if self._face_new:
                self._face_new = False
                for rule in self._rules:
                    if rule.enabled and rule.trigger == "proximity":
                        self._face_edge(rule, cfg)
            taken = sorted(self._inbox, key=lambda p: p.stamp)
            self._inbox.clear()
        fresh = []
        for p in taken:
            rule = self._rule(p.rule)
            vision = rule is not None and rule.trigger == "proximity"
            limit = cfg.max_age_vision_ms if vision else cfg.max_age_ms
            age = now - p.stamp
            if age > limit / MS_PER_S or age < -cfg.max_future_ms / MS_PER_S:
                self._stat(p.rule).stale += 1
            else:
                fresh.append(p)
        return fresh, cfg

    def _face_edge(self, rule: ReflexRule, cfg: ReflexConfig) -> None:
        armed = self._face_armed.get(rule.name, True)
        h = self._face_h
        if armed and h >= rule.threshold:
            self._face_armed[rule.name] = False
            self._inbox.append(_Pending(rule.name, self._face_stamp, h))
        elif not armed and h < rule.threshold * cfg.proximity_rearm_ratio:
            self._face_armed[rule.name] = True

    def _stat(self, rule: str) -> _RuleStats:
        return self._stats.setdefault(rule, _RuleStats())

    def _rule(self, name: str) -> ReflexRule | None:
        return next((r for r in self._rules if r.name == name), None)

    def _start(self, p: _Pending, now: float, core: MotionCore, cfg: ReflexConfig) -> None:
        rule = self._rule(p.rule)
        if rule is None or not rule.enabled:
            return
        stats = self._stat(rule.name)
        last = self._last_fire.get(rule.name)
        if last is not None and now - last < rule.cooldown_s:
            stats.cooldown += 1
            return
        if (rule.suppress_with_minor or rule.primitive == "turn_toward") and core.minor_present:
            stats.minor += 1
            return
        seed = self._seed(core)
        peak = self._peak(rule, p.value, seed, core, cfg)
        if not peak:
            stats.no_motion += 1
            return
        duration = rule.duration_ms / MS_PER_S
        start = {k: seed[k] for k in peak}
        frames = [Keyframe(0.0, start), Keyframe(cfg.attack_fraction * duration, peak)]
        if rule.primitive == "turn_toward":
            frames.append(Keyframe(duration, peak))
        else:
            frames += [
                Keyframe((1.0 - cfg.release_fraction) * duration, peak),
                Keyframe(duration, start),
            ]
        source = SOURCE_PREFIX + rule.name
        prim = Primitive(source, tuple(frames))
        self._players[rule.name] = Player(prim, now, source=source, priority=Priority.REFLEX)
        self._last_fire[rule.name] = now
        stats.fired += 1
        self._record(rule.trigger, "command", max(0.0, now - p.stamp))
        self._moving[rule.name] = (p.stamp, rule.trigger, start)

    def _seed(self, core: MotionCore) -> dict[str, float]:
        """The pose a reflex starts from, un-scaled by the minor factor.

        The arbiter scales an absolute target about home by the factor, and the limiter's pose
        is already scaled, so seeding from the pose would pull it back to home (forward, for a
        head tipped back) on every reflex while a minor is present.
        """
        factor = core.minor_factor if core.minor_present else 1.0
        out = {}
        for name, dof in self.dofs.items():
            pos = core.limiter.pos.get(name, dof.home)
            out[name] = _clamp(dof.home + (pos - dof.home) / factor, dof.min, dof.max)
        return out

    def _peak(
        self,
        rule: ReflexRule,
        value: float,
        seed: dict[str, float],
        core: MotionCore,
        cfg: ReflexConfig,
    ) -> dict[str, float]:
        """Targets at the top of the motion; empty when there is nothing safe to do."""
        amp = math.radians(min(rule.amplitude_deg, cfg.max_amplitude_deg))
        if amp <= 0:
            return {}
        rmap = self.rmap
        if rule.primitive == "turn_toward":
            return self._turn(value, amp, seed, core, math.radians(cfg.motion_eps_deg))
        pitch = self.dofs[rmap.head_pitch]
        away = rmap.pitch_away_sign
        out: dict[str, float] = {}
        eps = math.radians(cfg.motion_eps_deg)  # less than this is not a movement
        target = _clamp(seed[rmap.head_pitch] + away * amp, pitch.min, pitch.max)
        # The arbiter scales the target about home by the minor factor. If that lands the head
        # forward of where it is now (a pose still outside the band), the pitch is left alone:
        # a reflex never tips the head forward, and one that cannot move it is no_motion.
        factor = core.minor_factor if core.minor_present else 1.0
        scaled = pitch.home + (target - pitch.home) * factor
        now_pitch = core.limiter.pos.get(rmap.head_pitch, pitch.home)
        if abs(target - seed[rmap.head_pitch]) > eps and (scaled - now_pitch) * away >= 0:
            out[rmap.head_pitch] = target
        if rule.primitive == "startle":
            for name, sign in rmap.antennas.items():
                d = self.dofs[name]
                lift = _clamp(seed[name] + sign * amp * cfg.antenna_ratio, d.min, d.max)
                if abs(lift - seed[name]) > eps:
                    out[name] = lift
        return out

    def _turn(
        self, azimuth: float, amp: float, seed: dict[str, float], core: MotionCore, eps: float
    ) -> dict[str, float]:
        rmap = self.rmap
        head = self.dofs[rmap.head_yaw]
        lo, hi = head.min, head.max
        if rmap.body_yaw is not None:
            body_now = core.limiter.pos.get(rmap.body_yaw, 0.0)
            reach = yaw_reach(self._gaze, self.constraints, self.margin_rad)
            lo, hi = max(lo, body_now - reach), min(hi, body_now + reach)
        now_yaw = seed[rmap.head_yaw]
        aim = _clamp(azimuth, -HALF_PLANE, HALF_PLANE)
        target = _clamp(now_yaw + _clamp(aim - now_yaw, -amp, amp), lo, hi)
        if abs(target - now_yaw) <= eps:
            return {}
        return {rmap.head_yaw: target}

    def _advance(self, now: float, core: MotionCore, cfg: ReflexConfig) -> None:
        ttl = cfg.command_ttl_ms / MS_PER_S
        eps = math.radians(cfg.motion_eps_deg)
        for name, player in list(self._players.items()):
            cmd = player.command(now)
            if cmd is None:
                core.arbiter.clear(player.source)
                del self._players[name]
                self._moving.pop(name, None)
                continue
            core.feed(replace(cmd, expires=now + ttl), liveness=False)
            moving = self._moving.get(name)
            if moving is not None:
                stamp, trigger, start = moving
                if any(abs(core.limiter.pos.get(k, v) - v) > eps for k, v in start.items()):
                    self._record(trigger, "motion", max(0.0, now - stamp))
                    del self._moving[name]

    def _stop_minor_sensitive(self, core: MotionCore) -> None:
        """A minor appeared while a reflex that must not play near one is running: end it now."""
        for name, player in list(self._players.items()):
            rule = self._rule(name)
            if rule is None or rule.suppress_with_minor or rule.primitive == "turn_toward":
                core.arbiter.clear(player.source)
                del self._players[name]
                self._moving.pop(name, None)
                self._stat(name).minor += 1

    def _stop_all(self, core: MotionCore) -> None:
        for player in self._players.values():
            core.arbiter.clear(player.source)
        self._players.clear()
        self._moving.clear()

    def _record(self, trigger: str, kind: str, seconds: float) -> None:
        window = self.cfg.latency_window
        series = self._lat.setdefault(trigger, {}).setdefault(kind, deque(maxlen=window))
        if series.maxlen != window:
            series = self._lat[trigger][kind] = deque(series, maxlen=window)
        series.append(seconds * MS_PER_S)

    # ---- diagnostics (SPEC 12A)

    def active(self) -> list[str]:
        """Rules whose reflex is playing now."""
        return sorted(self._players)

    def snapshot(self) -> dict[str, Any]:
        """Debug-report section: counts, latencies and the audio source's state; scalars only."""
        latency: dict[str, dict[str, Any]] = {}
        for trigger, kinds in list(self._lat.items()):
            latency[trigger] = {}
            for kind, series in list(kinds.items()):
                ms = sorted(list(series))
                if ms:
                    latency[trigger][f"{kind}_ms"] = {
                        "count": len(ms),
                        "p50": _quantile(ms, P50),
                        "p95": _quantile(ms, P95),
                        "max": ms[-1],
                    }
        rules = {}
        for r in list(self._rules):
            s = self._stats.get(r.name) or _RuleStats()
            rules[r.name] = {
                "trigger": r.trigger,
                "primitive": r.primitive,
                "enabled": r.enabled,
                "fired": s.fired,
                "dropped_stale": s.stale,
                "dropped_cooldown": s.cooldown,
                "suppressed_minor": s.minor,
                "no_motion": s.no_motion,
            }
        audio = self.audio_status() if self.audio_status is not None else {"state": "off"}
        return {
            "enabled": self.cfg.enabled,
            "active": self.active(),
            "rules": rules,
            "latency": latency,
            "audio": audio,
        }
