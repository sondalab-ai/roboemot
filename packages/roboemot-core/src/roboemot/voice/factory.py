"""Build the configured speech output (SPEC 8, TTS backends)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..bus import Bus
from ..config.schema import RoboEmotConfig
from ..llm.router import Router
from . import models as voice_models
from .openai_speech import OpenAISpeechBackend
from .piper_backend import PiperBackend, manifest_voice_files
from .speech import AudioSink, SpeechOutput
from .tts import NO_VOICE, FallbackTTS, NoVoice, TTSBackend


def piper_python(cfg: RoboEmotConfig) -> Path:
    """Interpreter of the Piper virtualenv: ``voice.piper.python``, else the installer's."""
    return (
        Path(cfg.voice.piper.python) if cfg.voice.piper.python else paths.piper_python(cfg.data_dir)
    )


def build_backend(cfg: RoboEmotConfig, router: Callable[[], Router]) -> TTSBackend:
    """``voice.backend`` with ``voice.fallback_backend`` behind it (Piper is the only local one).
    Voice files come from the voice model registry, so only verified files are loaded.
    ``none`` builds no Piper and reads no voice file: as the backend, a :class:`NoVoice` (the
    output refuses sentences); as the fallback, a remote backend without one (SPEC 8, CV-2)."""
    voice = cfg.voice
    if "kokoro" in {voice.backend, voice.fallback_backend}:
        raise ValueError("the kokoro TTS backend is not implemented")
    if voice.backend == NO_VOICE:
        return NoVoice()
    local: TTSBackend | None = None
    if voice.backend == "piper" or voice.fallback_backend == "piper":
        root = voice_models.models_dir(voice.models_dir, cfg.data_dir)
        local = PiperBackend(voice.piper, piper_python(cfg), manifest_voice_files(root))
    if voice.backend == "piper":
        assert local is not None
        return local
    return FallbackTTS(OpenAISpeechBackend(voice.openai_speech, router), local)


def build_speech(
    cfg: RoboEmotConfig,
    sink: AudioSink,
    router: Callable[[], Router],
    bus: Bus | None = None,
) -> SpeechOutput:
    """The speech output over ``sink``; ``router`` is a getter (the app rebuilds its router)."""
    return SpeechOutput(cfg, sink, build_backend(cfg, router), bus)
