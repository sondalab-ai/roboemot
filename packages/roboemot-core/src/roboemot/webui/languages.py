"""The languages the web UI offers for ``conversation.language`` (SPEC 8 Voice pipeline, 11A).

A choice, not free text: a language is offered when the voice stack supports it (the voice model
manifest has an ASR and a TTS model for it), the configuration lists it for both
(``voice.asr.models`` and ``voice.voices``) and its model files are installed. The current value
is always listed so the control can show it, but it is only kept, never newly accepted, when its
models are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from roboemot.config.schema import RoboEmotConfig
from roboemot.vision.models import status as model_status
from roboemot.voice.models import ASR_ROLE, TTS_ROLE, ModelError, load_manifest, models_dir
from roboemot.voice.models import supported_languages as manifest_languages

log = logging.getLogger(__name__)
LANGUAGE_PATH = "conversation.language"
USABLE = frozenset({"ok", "stale-ort"})  # stale-ort is a verified file that still loads


def installed_languages(cfg: RoboEmotConfig, root: Path | None = None) -> frozenset[str]:
    """Languages usable right now: in the manifest for ASR and TTS, configured for both, with every
    model of those two roles present and verified under ``root`` (default: the voice models folder
    of ``cfg``). Empty when the manifest cannot be read."""
    try:
        specs = load_manifest()
        wanted = manifest_languages() & set(cfg.voice.asr.models) & set(cfg.voice.voices)
    except ModelError:
        return frozenset()
    root = root or models_dir(cfg.voice.models_dir, cfg.data_dir)
    usable: set[str] = set()
    for lang in wanted:
        roles = {ASR_ROLE + lang, TTS_ROLE + lang}
        needed = [spec for spec in specs.values() if spec.role in roles]
        try:
            if needed and all(model_status(spec, root) in USABLE for spec in needed):
                usable.add(lang)
        except OSError as exc:
            log.warning("voice models for %s not readable (%s)", lang, type(exc).__name__)
    return frozenset(usable)
