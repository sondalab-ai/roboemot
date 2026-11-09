"""The robot's inner state for the web UI's Live tab (SPEC 11A, 12).

``LiveState.state()`` builds one small JSON-able dict from parts that already keep their state in
memory: the affect engine, the sleep gate, the network monitor, the register tracker, the world
model with the attention selector, the reflex engine, the motion loop and the conversation phase.
It never calls the heavy debug sources and never reads the configuration stack: the settings it
needs come from a copy kept up to date by ``apply_config`` (a stack subscriber), because the stack
validates the whole configuration on every read. State only: no audio, text, embeddings, registry
ids, bearings or positions, and nothing is logged or stored. The preview frame is ``LiveVideo``'s.

``music`` and ``dance`` feed the beat card: numbers and codes only, never audio, a spectrum or an
envelope (SPEC 12). Ages are computed here because a monotonic stamp means nothing to the browser.
Each block is None without its part (no voice listener, music off, no dance renderer).

``activity`` is the "What's happening" panel (``webui.live.activity``): a status code for what the
robot is doing now, the conditions that hold (offline, the cloud route down with its reason code)
and the last decision events of the runtime's ``EventRing`` (``roboemot.diag``), newest first.
Codes and numbers only: the page builds the sentences in its language. A host without voice has
no route or listener (None): the status then comes from the other parts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import psutil

from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.sources import entry_name
from roboemot.diag import ROUTE_REASONS, code

from .levels import AudioLevels
from .live_video import LiveVideo

MS_PER_S = 1000
ROUND_DIGITS = 3  # decimals kept in the numbers sent to the page
# what the robot is doing, most telling first ("What's happening" status line)
STATUSES = (
    "asleep", "answering", "cut", "waiting", "sending", "listening", "offline", "cloud_down",
    "napping", "resting", "alone", "idle",
)  # fmt: skip
ROUTE_DOWN = ("fallback_local", "blocked")  # realtime route states that mean it cannot answer
ROUTE_UP_REASONS = (None, "off")  # a blocked route with these reasons is just not in use
NO_TRANSCRIPT = {
    "enabled": False, "epoch": "", "rev": 0, "min_seq": 0, "max_turns": 0, "turns": []
}  # fmt: skip


def cpu_temperature_c() -> float | None:
    """The hottest current sensor reading in degrees C, or None where the OS exposes none."""
    try:
        sensors = psutil.sensors_temperatures()
    except Exception:  # not available on this OS (AttributeError) or unreadable
        return None
    values = [e.current for entries in sensors.values() for e in entries if e.current is not None]
    return max(values) if values else None


class LiveState:
    """Builds the Live tab's payload. ``host`` and ``p`` are the running ``RuntimeHost`` and
    ``Pipeline``; ``clock`` is the motion loop's clock, the one the reflex stamps use."""

    def __init__(
        self,
        host: Any,
        p: Any,
        cfg: RoboEmotConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        temperature: Callable[[], float | None] = cpu_temperature_c,
        audio_egress: Callable[[], bool] = lambda: False,
    ) -> None:
        self._host, self._p, self._clock, self._read_temp = host, p, clock, temperature
        self._cfg = cfg
        self._egress = audio_egress  # microphone audio may leave for the realtime route
        self._temp: tuple[float, float | None] | None = None  # (read at, value)
        self.video = LiveVideo(host, p, cfg, clock=clock)

    def apply_config(self, change: Any) -> None:
        """``ConfigStack`` subscriber: keep the settings this view reads (hot reload)."""
        self._cfg = change.config
        self.video.apply_config(change)

    def video_frame(self, client: str) -> tuple[bytes | None, str | None]:
        """The preview JPEG for ``client`` (an IP), or ``(None, code)``; see ``LiveVideo``."""
        return self.video.frame(client)

    def video_viewers(self) -> dict[str, Any]:
        """The banner's answer: ``active``, ``count`` and the page's poll interval."""
        return self.video.viewers()

    @property
    def enabled(self) -> bool:
        return self._cfg.webui.live.enabled

    def state(self) -> dict[str, Any] | None:
        """The payload, or None while ``webui.live.enabled`` is off (the API answers
        ``live_disabled``)."""
        cfg = self._cfg
        if not cfg.webui.live.enabled:
            return None
        now = self._clock()
        rest = self._rest()  # one read per poll: the modes badge and the status line agree
        return {
            "poll_ms": cfg.webui.live.poll_ms,
            "mood": self._mood(now),
            "modes": self._modes(cfg, rest),
            "conversation": self._host.phase.snapshot(),
            "audio_egress": bool(self._egress()),
            "attention": self._attention(cfg, now),
            "reflexes": self._reflexes(now),
            "motion": self._motion(),
            "music": self._music(cfg, now),
            "dance": self._dance(now),
            "system": self._system(cfg, now),
            "activity": self._activity(cfg, rest) if cfg.webui.live.activity else None,
            "video": {
                "enabled": cfg.webui.live.video,
                "poll_ms": round(MS_PER_S / cfg.webui.live.video_fps),
                "idle_stop_ms": round(MS_PER_S * cfg.webui.live.video_idle_stop_s),
            },
        }

    def audio(self, after: int = 0, epoch: str = "") -> dict[str, Any] | None:
        """The audio poll's payload: the levels in dBFS and the transcript turns added or
        changed after revision ``after`` (all of them if ``epoch`` is not the ring's); None while
        ``webui.live.enabled`` is off. Reads cached scalars and the ring: it never starts or reads
        the microphone, and while asleep or paused reports ``paused``."""
        cfg = self._cfg
        if not cfg.webui.live.enabled:
            return None
        live = cfg.webui.live
        levels, ring = self._host.levels, self._host.transcript
        if ring is not None and not live.transcript:
            ring.apply_config(live)  # the flag is off here: whatever it still holds goes
        parts = self._p.sleep.paused_parts()
        paused = self._p.sleep.asleep or "listener" in parts
        return {
            "audio_poll_ms": live.audio_poll_ms,
            "levels": (levels or AudioLevels()).snapshot(
                paused=paused, max_age_s=live.level_max_age_s, window_ms=live.level_window_ms
            ),
            "transcript": (
                ring.snapshot(after, epoch)
                if ring is not None and live.transcript
                else NO_TRANSCRIPT
            ),
        }

    def clear_transcript(self) -> bool:
        """The Clear button. False while ``webui.live.enabled`` is off (the API answers
        ``live_disabled``)."""
        if not self._cfg.webui.live.enabled:
            return False
        ring = self._host.transcript
        if ring is not None:
            ring.clear()
        return True

    def _activity(self, cfg: RoboEmotConfig, rest: str | None) -> dict[str, Any]:
        """The status code, the conditions that hold and the decision events (newest first)."""
        notes = self._notes(cfg)
        ring = self._host.activity
        return {
            "status": self._status(cfg, notes, rest),
            "notes": notes,
            "events": ring.events() if ring is not None else [],
        }

    def _notes(self, cfg: RoboEmotConfig) -> list[dict[str, Any]]:
        """The conditions that hold now: the network down, the realtime route down (its
        reason code)."""
        notes: list[dict[str, Any]] = []
        if not self._p.network.online:
            notes.append({"code": "offline", "reason": None})
        route = self._host.realtime_route
        if route is not None and cfg.voice.realtime_route.enabled:
            state, reason = route.state, route.reason
            if state in ROUTE_DOWN and reason not in ROUTE_UP_REASONS:
                notes.append({"code": "cloud_down", "reason": code(reason, ROUTE_REASONS)})
        return notes

    def _status(self, cfg: RoboEmotConfig, notes: list[dict[str, Any]], rest: str | None) -> str:
        """One of ``STATUSES``, from the parts' own state (the first that holds)."""
        if self._p.sleep.asleep:
            return "asleep"
        phase = self._host.phase.snapshot()["phase"]
        if phase == "speaking":
            return "answering"
        ring = self._host.activity
        age = ring.age_of("bargein", "cut") if ring is not None else None
        if age is not None and age <= cfg.webui.live.activity_cut_hold_s:
            return "cut"
        if phase == "thinking":
            return "waiting"
        listener = self._host.voice_listener
        if listener is not None and listener.awaiting_transcript():
            return "sending"
        if phase == "listening":
            return "listening"
        if notes:
            return notes[0]["code"]
        if rest in ("napping", "resting"):
            return rest
        return "alone" if self._p.alone.alone else "idle"

    def _rest(self) -> str | None:
        """The circadian rest the modes card and the status line show (SPEC 7A): ``napping``
        for a nap, ``resting`` for any other rest, ``drowsy``, or None while awake or with the
        rhythm off. Not the platform sleep (``asleep``: the motors are off, SPEC 12A)."""
        circ = self._host.circadian
        if circ is None:
            return None
        if circ.state == "resting":
            return "napping" if circ.reason == "nap" else "resting"
        return "drowsy" if circ.state == "drowsy" else None

    def _mood(self, now: float) -> dict[str, Any]:
        s = self._p.runner.engine.state(now)
        circ = self._p.circ.snapshot() if self._host.circadian is not None else None
        energy = None
        if circ is not None:
            sleepiness = circ.get("sleepiness")
            energy = {
                "state": circ["state"],
                "reason": circ.get("reason"),
                "sleepiness": None if sleepiness is None else round(sleepiness, ROUND_DIGITS),
            }
        return {
            "label": s.label,
            "valence": round(s.valence, ROUND_DIGITS),
            "arousal": round(s.arousal, ROUND_DIGITS),
            "intensity": round(s.intensity, ROUND_DIGITS),
            "circadian": energy,
        }

    def _modes(self, cfg: RoboEmotConfig, rest: str | None) -> dict[str, Any]:
        p = self._p
        active = p.tracker.active
        return {
            "asleep": p.sleep.asleep,
            "rest": rest,
            "alone": p.alone.alone,
            "paused": p.sleep.paused_parts(),
            "online": p.network.online,
            "people": active.people if active is not None else 0,
            "persona": cfg.personality,
        }

    def _attention(self, cfg: RoboEmotConfig, now: float) -> dict[str, Any]:
        p = self._p
        target = p.selector.target
        target_id = target.entry_id if target is not None else None
        registry = p.perception.registry
        fresh = [
            e
            for e in p.world.entries
            if e.kind == "person" and now - e.last_seen <= cfg.attention.stale_s
        ]
        fresh.sort(key=lambda e: (e.entry_id != target_id, -e.last_seen))
        faces = [
            {"name": entry_name(registry, e), "target": e.entry_id == target_id}
            for e in fresh[: cfg.webui.live.max_tracked]
        ]
        return {"count": len(fresh), "faces": faces}

    def _reflexes(self, now: float) -> dict[str, Any]:
        last = self._p.reflexes.last_fired(now)
        if last is None:
            return {"last": None}
        return {"last": {"name": last[0], "age_s": round(last[1], ROUND_DIGITS)}}

    def _motion(self) -> dict[str, Any]:
        loop = self._p.loop  # two flags read without the loop's lock: never waits for a tick
        off = loop.motors_off or self._p.sleep.asleep
        if off:
            state = "motors_off"
        elif loop.external:
            state = "external"
        else:  # holding on an unreachable target (SPEC 12A): the loop writes nothing there
            state = "held" if loop.unreachable_held else "writing"
        return {"state": state}

    def _music(self, cfg: RoboEmotConfig, now: float) -> dict[str, Any] | None:
        """The detector's lock-free state, or None without a detector or with ``music.enabled``
        off. ``beat_phase_s`` is already the time since the last beat."""
        listener = self._host.voice_listener
        detector = listener.music if listener is not None else None
        if detector is None or not cfg.music.enabled:
            return None
        m = detector.state(now)
        return {
            "playing": m.playing,
            "tempo_bpm": None if m.tempo_bpm is None else round(m.tempo_bpm, ROUND_DIGITS),
            "confidence": round(float(m.confidence), ROUND_DIGITS),
            "beat_age_s": None if m.beat_phase_s is None else round(m.beat_phase_s, ROUND_DIGITS),
        }

    def _dance(self, now: float) -> dict[str, Any] | None:
        """The dance layer's last tick, or None without a renderer: ``state`` is ``active`` or
        ``idle``, ``bob`` and ``sway`` each role's share of its amplitude."""
        renderer = self._host.dance
        if renderer is None:
            return None
        d = renderer.live_snapshot(now)
        age = d.last_accent_age_s(now)
        return {
            "state": "active" if d.active else "idle",
            "source": d.source,
            "bob": round(d.bob, ROUND_DIGITS),
            "sway": round(d.sway, ROUND_DIGITS),
            "last_accent_age_s": None if age is None else round(age, ROUND_DIGITS),
        }

    def _system(self, cfg: RoboEmotConfig, now: float) -> dict[str, Any]:
        cached = self._temp
        if cached is None or now - cached[0] >= cfg.webui.live.system_cache_s:
            cached = self._temp = (now, self._read_temp())
        value = cached[1]
        return {"cpu_temp_c": None if value is None else round(value, 1)}
