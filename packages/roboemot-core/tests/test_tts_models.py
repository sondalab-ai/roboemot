"""SPEC 8, stage 7: the real Piper worker (opt in with ``-m models``).

Needs a Piper virtualenv (``ROBOEMOT_PIPER_PYTHON``, else the installer's default under the app
data folder) and the voice files (``ROBOEMOT_PIPER_VOICES``, else ``voice/piper`` in the models
folder). Skipped when either is missing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from roboemot import paths
from roboemot.config.schema import PiperConfig, RoboEmotConfig
from roboemot.vision import models
from roboemot.voice.piper_backend import PiperBackend
from roboemot.voice.prosody import Delivery

pytestmark = pytest.mark.models

CFG = RoboEmotConfig()
SENTENCES = {
    "it": "Ciao, sono il tuo piccolo robot. Oggi mi sento molto curioso!",
    "en": "Hello, I am your little robot. Today I feel very curious!",
}
FIRST_AUDIO_BUDGET_S = 2.0  # SPEC 13: end of speech to first TTS audio, of which Piper is a part


def piper_setup() -> tuple[Path, Path]:
    python = Path(os.environ.get("ROBOEMOT_PIPER_PYTHON") or paths.piper_python(CFG.data_dir))
    voices = Path(
        os.environ.get("ROBOEMOT_PIPER_VOICES")
        or models.models_dir(CFG.vision.models_dir, CFG.data_dir) / "voice" / "piper"
    )
    if not python.exists():
        pytest.skip(f"no Piper interpreter at {python}")
    missing = [v for v in CFG.voice.voices.values() if not (voices / f"{v}.onnx").exists()]
    if missing:
        pytest.skip(f"voice files missing in {voices}: {missing}")
    return python, voices


@pytest.mark.parametrize("language", ["it", "en"])
async def test_real_piper_worker_speaks_and_reports_the_voice_rate(language, capsys):
    python, voices = piper_setup()
    backend = PiperBackend(PiperConfig(), python, voices)
    delivery = Delivery(1.0, 0.0, 0.0, 0.6, 1.0)
    try:
        t0 = time.monotonic()
        await backend.warm(language, CFG.voice)
        warm_s = time.monotonic() - t0
        timings = []
        for _ in range(3):
            t0, first, chunks = time.monotonic(), None, []
            async for c in backend.synthesize(
                SENTENCES[language], delivery, language=language, voice=CFG.voice
            ):
                first = first if first is not None else time.monotonic() - t0
                chunks.append(c)
            timings.append(first)
        audio = np.concatenate([c.samples for c in chunks])
        rate = chunks[0].rate_hz
        assert rate in (16000, 22050, 24000) and {c.rate_hz for c in chunks} == {rate}
        assert audio.dtype == np.float32 and 1.0 < audio.size / rate < 15.0
        assert 0.05 < np.abs(audio).max() <= 1.0
        assert backend.restarts == 0
        assert min(timings) < FIRST_AUDIO_BUDGET_S
        with capsys.disabled():
            print(
                f"\n[piper {language}] rate {rate} Hz, load {warm_s:.2f} s, first audio "
                f"{', '.join(f'{t * 1000:.0f}' for t in timings)} ms, "
                f"{audio.size / rate:.1f} s of audio, peak {np.abs(audio).max():.2f}"
            )
    finally:
        await backend.aclose()
