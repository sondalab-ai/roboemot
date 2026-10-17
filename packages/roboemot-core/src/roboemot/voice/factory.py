"""Build the configured speech output (SPEC 8, TTS backends)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..bus import Bus
from ..config.schema import RoboEmotConfig
from ..llm.router import Router
from ..vision import models
from .openai_speech import OpenAISpeechBackend
from .piper_backend import PiperBackend
from .speech import AudioSink, SpeechOutput
from .tts import FallbackTTS, TTSBackend


def piper_locations(cfg: RoboEmotConfig) -> tuple[Path, Path]:
    """Interpreter of the Piper virtualenv and the folder of the voice files, from the config
    (empty fields resolve to the installer's locations in the app data folder)."""
    python = Path(cfg.voice.piper.python) if cfg.voice.piper.python else None
    voices = Path(cfg.voice.piper.voices_dir) if cfg.voice.piper.voices_dir else None
    return (
        python or paths.piper_python(cfg.data_dir),
        voices or models.models_dir(cfg.vision.models_dir, cfg.data_dir) / "voice" / "piper",
    )


def build_backend(cfg: RoboEmotConfig, router: Router) -> TTSBackend:
    """``voice.backend`` with ``voice.fallback_backend`` behind it (Piper is the only local one)."""
    chosen = {cfg.voice.backend, cfg.voice.fallback_backend}
    if "kokoro" in chosen:
        raise ValueError("the kokoro TTS backend is not implemented")
    python, voices = piper_locations(cfg)
    local = PiperBackend(cfg.voice.piper, python, voices)
    if cfg.voice.backend == "piper":
        return local
    return FallbackTTS(OpenAISpeechBackend(cfg.voice.openai_speech, router), local)


def build_speech(
    cfg: RoboEmotConfig,
    sink: AudioSink,
    router: Router,
    bus: Bus | None = None,
    minor_present: Callable[[], bool] = lambda: False,
) -> SpeechOutput:
    """The speech output over ``sink``; ``minor_present`` is read when each sentence starts."""
    return SpeechOutput(cfg, sink, build_backend(cfg, router), bus, minor_present)
