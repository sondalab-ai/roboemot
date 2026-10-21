"""The robot's inner state for the web UI's Live tab (SPEC 11A, 12).

``LiveState.state()`` builds one small JSON-able dict from parts that already keep their state in
memory: the affect engine, the sleep gate, the network monitor, the register tracker, the world
model with the attention selector, the reflex engine, the motion loop and the conversation phase.
It never calls the heavy debug sources and never reads the configuration stack: the settings it
needs come from a copy kept up to date by ``apply_config`` (a stack subscriber), because the stack
validates the whole configuration on every read. State only: no frames, audio, text, embeddings,
registry ids, bearings or positions, and nothing is logged or stored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import psutil

from roboemot.config.schema import RoboEmotConfig
from roboemot.conversation.sources import entry_name

ROUND_DIGITS = 3  # decimals kept in the numbers sent to the page


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
    ) -> None:
        self._host, self._p, self._clock, self._read_temp = host, p, clock, temperature
        self._cfg = cfg
        self._temp: tuple[float, float | None] | None = None  # (read at, value)

    def apply_config(self, change: Any) -> None:
        """``ConfigStack`` subscriber: keep the settings this view reads (hot reload)."""
        self._cfg = change.config

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
        return {
            "poll_ms": cfg.webui.live.poll_ms,
            "mood": self._mood(now),
            "modes": self._modes(cfg),
            "conversation": self._host.phase.snapshot(),
            "attention": self._attention(cfg, now),
            "reflexes": self._reflexes(now),
            "motion": self._motion(),
            "system": self._system(cfg, now),
        }

    def _mood(self, now: float) -> dict[str, Any]:
        s = self._p.runner.engine.state(now)
        circ = self._p.circ.snapshot() if self._host.circadian is not None else None
        energy = None
        if circ is not None:
            sleepiness = circ.get("sleepiness")
            energy = {
                "state": circ["state"],
                "sleepiness": None if sleepiness is None else round(sleepiness, ROUND_DIGITS),
            }
        return {
            "label": s.label,
            "valence": round(s.valence, ROUND_DIGITS),
            "arousal": round(s.arousal, ROUND_DIGITS),
            "intensity": round(s.intensity, ROUND_DIGITS),
            "circadian": energy,
        }

    def _modes(self, cfg: RoboEmotConfig) -> dict[str, Any]:
        p = self._p
        active = p.tracker.active
        return {
            "asleep": p.sleep.asleep,
            "paused": p.sleep.paused_parts(),
            "online": p.network.online,
            "minor_present": self._host.minor_present,
            "register": active.group if active is not None else None,
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
        state = "motors_off" if loop.motors_off else "external" if loop.external else "writing"
        return {"state": state}

    def _system(self, cfg: RoboEmotConfig, now: float) -> dict[str, Any]:
        cached = self._temp
        if cached is None or now - cached[0] >= cfg.webui.live.system_cache_s:
            cached = self._temp = (now, self._read_temp())
        value = cached[1]
        return {"cpu_temp_c": None if value is None else round(value, 1)}
