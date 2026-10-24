"""Listening aid: one fixed reply rendered in every mood to WAV (SPEC 8, `roboemot voice sample`).

Each file holds the same short reply (a greeting, a statement, a question) spoken with the
configured voice and delivery: neutral, then once per ``voice.moods`` entry at the given affect
intensity, with the sentence variation and the question raise of a real reply (fixed seed, so two
runs give the same files). Sentences are joined with the reply pause (``sentence_pause_ms`` times
the mood's ``pause_scale``). The backend's chunks carry their own rate (the dsp pitch shift moves
it per sentence), so every chunk is resampled to ``SAMPLE_RATE_HZ`` before writing.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config.schema import RoboEmotConfig
from ..persona import effects
from .prosody import Delivery, compose_delivery
from .speech import ends_with_question
from .tts import TTSBackend

SAMPLE_RATE_HZ = 22050  # the rate of the medium Piper voices
SAMPLE_SEED = 1  # the reply id the variation is drawn from
NEUTRAL = "neutral"
INT16_MAX = 32767
MS_PER_S = 1000.0
SAMPLE_TEXT = {
    "it": ("Ciao!", "Oggi ho visto una cosa che non mi aspettavo.", "Vuoi sapere che cos'è?"),
    "en": ("Hi!", "Today I saw something I did not expect.", "Do you want to know what it is?"),
}


@dataclass(frozen=True)
class Sample:
    mood: str
    path: Path
    deliveries: list[Delivery]
    seconds: float


def _resample(samples: np.ndarray, in_hz: int, out_hz: int) -> np.ndarray:
    if in_hz == out_hz or samples.size == 0:
        return samples
    n = max(1, round(samples.size * out_hz / in_hz))
    return np.interp(np.arange(n) * (in_hz / out_hz), np.arange(samples.size), samples)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE_HZ)
        w.writeframes(pcm.tobytes())


async def render_samples(
    cfg: RoboEmotConfig,
    backend: TTSBackend,
    out_dir: Path,
    *,
    language: str,
    intensity: float,
) -> list[Sample]:
    """Write ``<out_dir>/<language>-<mood>.wav`` for neutral and every configured mood; returns
    what was written. The mood source is forced to ``auto`` so each file has its own mood."""
    if language not in SAMPLE_TEXT:
        raise ValueError(f"no sample text for language {language!r} (have {sorted(SAMPLE_TEXT)})")
    voice = cfg.voice.model_copy(update={"mood_source": "auto"})
    persona = effects(cfg.persona)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mood in [NEUTRAL, *voice.moods]:
        parts: list[np.ndarray] = []
        deliveries = []
        for index, sentence in enumerate(SAMPLE_TEXT[language]):
            d = compose_delivery(
                voice,
                register=cfg.register.params,
                circadian=cfg.circadian.states["awake"],
                mood=None if mood == NEUTRAL else mood,
                intensity=intensity,
                persona=persona,
                seed=SAMPLE_SEED,
                index=index,
                question=ends_with_question(sentence),
            )
            if index:
                pause_s = voice.sentence_pause_ms / MS_PER_S * d.pause_scale
                parts.append(np.zeros(round(pause_s * SAMPLE_RATE_HZ), np.float32))
            async for chunk in backend.synthesize(sentence, d, language=language, voice=voice):
                parts.append(_resample(chunk.samples, chunk.rate_hz, SAMPLE_RATE_HZ))
            deliveries.append(d)
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        path = out_dir / f"{language}-{mood}.wav"
        _write_wav(path, audio)
        written.append(Sample(mood, path, deliveries, audio.size / SAMPLE_RATE_HZ))
    return written


__all__ = ["SAMPLE_RATE_HZ", "SAMPLE_TEXT", "Sample", "render_samples"]
