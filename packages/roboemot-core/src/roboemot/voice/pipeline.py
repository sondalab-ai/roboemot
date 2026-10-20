"""Wiring of the voice listener from core parts (SPEC 8 stages 2-4, 5A attention hint).

``build_listener`` assembles Silero VAD, Vosk, the fast path and the listener over the audio
hub's voice queue, with the app-side lookups they need: the robot's sound bearing in the base
frame (a ``BearingSource``), the attended person, whether an enrolled person is in view, and the
speaker's language preference.
The barge-in detector (SPEC 8 stage 8) is built with the listener, on its own Silero instance, and
hooked to the listener's speaking chunks; the app binds its actions (stop the speech, interrupt the
conversation, resume the listener) once the speech output and the conversation exist. It returns
None, with a warning, when the ``voice`` extra or the models are missing: the robot then runs
without voice.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roboemot.attention import AttentionSelector
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.robot_platform import AudioHub, BearingSource
from roboemot.vision.models import ModelError
from roboemot.voice import models as voice_models
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.asr import VoskASR, resolve_language
from roboemot.voice.bargein import BargeIn, PlaybackReference
from roboemot.voice.fastpath import Emit, FastPath
from roboemot.voice.listener import Utterance, VoiceListener
from roboemot.voice.robot_name import RobotName
from roboemot.voice.vad import Endpointer, SileroVAD
from roboemot.world import Entry, WorldModel
from roboemot.world.types import UNKNOWN_IDENTITY

log = logging.getLogger(__name__)

LANGUAGE_PREFERENCE = "language"


def build_listener(
    cfg: RoboEmotConfig,
    audio: AudioHub,
    speech: SpeechActivity,
    bus: Bus,
    emit: Emit,
    selector: AttentionSelector,
    world: WorldModel,
    registry: Any,
    bearing: BearingSource,
    on_utterance: Callable[[Utterance], None],
    reference: PlaybackReference | None = None,
    names: RobotName | None = None,
    on_answer: Callable[[str, float], None] | None = None,
) -> tuple[VoiceListener, Callable[[], None], BargeIn] | None:
    """The listener, a ``close`` for the bearing source and the barge-in detector (not yet bound
    to its actions), or None when voice can't run. ``bearing`` is the robot's sound direction in
    the base frame (``RobotSession.bearing_source``)."""
    root = voice_models.models_dir(cfg.voice.models_dir, cfg.data_dir)
    try:
        vad_path = voice_models.model_file(root, "silero-vad")
        asr_paths: dict[str, Path] = {
            lang: voice_models.model_file(root, name) for lang, name in cfg.voice.asr.models.items()
        }
        vad = SileroVAD(vad_path, cfg.voice.vad.threads)
        barge_vad = SileroVAD(vad_path, cfg.voice.vad.threads)  # its own recurrent state
        asr = VoskASR(asr_paths, cfg.voice.sample_rate_hz)
    except (ModelError, RuntimeError, OSError) as exc:
        log.warning(
            "voice: not listening (%s: %s). Install the `voice` extra and run "
            "`roboemot models fetch --voice`",
            type(exc).__name__,
            exc,
        )
        return None
    names = names if names is not None else RobotName()
    names.attach_vocabulary(asr.missing_words)
    max_age_s = cfg.voice.fastpath.doa_max_age_ms / 1000.0

    def attended_person() -> Entry | None:
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

    def speaker_id() -> str | None:
        """The attended person's registry id, when it is a known person (else undeterminable)."""
        entry = attended_person()
        return None if entry is None or entry.identity in ("", UNKNOWN_IDENTITY) else entry.identity

    def enrolled_in_view(now: float) -> bool:
        # The policy default (SPEC 8 stage 2): any enrolled person in view drops the wake-word
        # requirement for every voice, including a stranger's or a television's. Kept as
        # specified; see the voice.wake_word_policy help for the consequence.
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
        target_entry=lambda: person.entry_id if (person := attended_person()) else None,
        names=names,
        on_answer=on_answer,
    )
    bargein = BargeIn(
        cfg,
        barge_vad,
        asr,
        reference,
        language=lambda: resolve_language(
            asr.languages, speaker_language(), cfg.conversation.language
        ),
        emit=emit,
        keyword_background=True,
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
        speaker_id=speaker_id,
        on_speaking_chunk=bargein.on_chunk,
        on_speaking_end=bargein.reset,
        before_asr_reset=bargein.handback_asr,
        enrolled_in_view=enrolled_in_view,
    )

    def close() -> None:
        bargein.close()
        bearing.close()

    return listener, close, bargein
