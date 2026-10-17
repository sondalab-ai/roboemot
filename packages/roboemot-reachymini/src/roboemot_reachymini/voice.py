"""Wiring of the voice listener into the app (SPEC 8 stages 2-4, 5A attention hint).

``build_listener`` assembles Silero VAD, Vosk, the fast path and the listener over the audio
hub's voice queue, with the app-side lookups they need: the sound bearing in the base frame, the
attended person, whether an enrolled person is in view, and the speaker's language preference.
It returns None, with a warning, when the ``voice`` extra or the models are missing: the robot
then runs without voice.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roboemot.attention import AttentionSelector
from roboemot.bus import Bus, Message
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import VoskASR
from roboemot.voice.fastpath import Emit, FastPath
from roboemot.voice.listener import Utterance, VoiceListener
from roboemot.voice.vad import Endpointer, SileroVAD
from roboemot.world import FrameSpec, Observation, WorldModel, to_base_frame
from roboemot.world.types import UNKNOWN_IDENTITY

from .audio import ReachyAudio
from .world import SOUND_SOURCE

log = logging.getLogger(__name__)

LANGUAGE_PREFERENCE = "language"


class LatestBearing:
    """The newest sound direction in the base frame, from the DoA source's observations.

    The source publishes sensor-frame bearings on ``percept.entity`` with the joint state at
    capture; this rotates them with the same frame as the world model and keeps the latest.
    """

    def __init__(self, bus: Bus, frame: FrameSpec) -> None:
        self._frame = frame
        self._lock = threading.Lock()
        self._latest: tuple[float, float] | None = None
        self._unsubscribe = bus.subscribe("percept.entity", self._on_entity)

    def close(self) -> None:
        """Stop listening."""
        self._unsubscribe()

    def _on_entity(self, msg: Message) -> None:
        obs = msg.payload
        if not isinstance(obs, Observation) or obs.source != SOUND_SOURCE:
            return
        az, _ = to_base_frame(
            obs.azimuth_rad, obs.elevation_rad, msg.meta.get("joint_state"), self._frame
        )
        with self._lock:
            self._latest = (obs.stamp, az)

    def get(self, now: float, max_age_s: float) -> float | None:
        """The base-frame azimuth if the newest reading is at most ``max_age_s`` old."""
        with self._lock:
            latest = self._latest
        return latest[1] if latest is not None and now - latest[0] <= max_age_s else None


def build_listener(
    cfg: RoboEmotConfig,
    audio: ReachyAudio,
    speech: SpeechActivity,
    bus: Bus,
    emit: Emit,
    selector: AttentionSelector,
    world: WorldModel,
    registry: Any,
    frame: FrameSpec,
    on_utterance: Callable[[Utterance], None],
) -> tuple[VoiceListener, Callable[[], None]] | None:
    """The listener and a ``close`` for its bus subscription, or None when voice can't run."""
    root = voice_models.models_dir(cfg.voice.models_dir, cfg.data_dir)
    try:
        vad_path = voice_models.model_file(root, "silero-vad")
        asr_paths: dict[str, Path] = {
            lang: voice_models.model_file(root, name) for lang, name in cfg.voice.asr.models.items()
        }
        vad = SileroVAD(vad_path, cfg.voice.vad.threads)
        asr = VoskASR(asr_paths, cfg.voice.sample_rate_hz)
    except (ModelError, RuntimeError, OSError) as exc:
        log.warning(
            "voice: not listening (%s: %s). Install the `voice` extra and run "
            "`roboemot models fetch --voice`",
            type(exc).__name__,
            exc,
        )
        return None
    bearing = LatestBearing(bus, frame)
    max_age_s = cfg.voice.fastpath.doa_max_age_ms / 1000.0

    def attended_person() -> Any:
        """The world entry of the person the robot is attending to, or None."""
        target = selector.target
        entry = world.get(target.entry_id) if target is not None and target.entry_id else None
        return entry if entry is not None and entry.kind == "person" else None

    def speaker_language() -> str | None:
        entry = attended_person()
        if entry is None or entry.identity in ("", UNKNOWN_IDENTITY) or registry is None:
            return None
        try:
            value = registry.get_person(entry.identity).preferences.get(LANGUAGE_PREFERENCE)
        except Exception:  # no such record, registry error: use the default language
            return None
        return value if isinstance(value, str) else None

    def enrolled_in_view(now: float) -> bool:
        stale = cfg.attention.stale_s
        return any(
            e.kind == "person"
            and e.identity not in ("", UNKNOWN_IDENTITY)
            and now - e.last_seen <= stale
            for e in world.entries
        )

    fast = FastPath(
        cfg.voice,
        emit,
        selector,
        bearing=lambda now: bearing.get(now, max_age_s),
        target_entry=lambda: getattr(attended_person(), "entry_id", None),
    )
    listener = VoiceListener(
        cfg,
        audio.read,
        Endpointer(vad, cfg.voice.vad),
        asr,
        speech,
        fast,
        on_utterance,
        bus,
        speaker_language=speaker_language,
        enrolled_in_view=enrolled_in_view,
    )
    return listener, bearing.close
