"""Reachy Mini's non-verbal cue sounds (SPEC 8, Non-verbal cues; SPEC 4A).

Registered in the ``roboemot.cue_sounds`` entry-point group under ``reachy-mini-wireless``. Serves
the package data ``data/cues/<kind>.wav``: short blirps synthesized by
``scripts/make_cue_sounds.py`` with RoboEmot's own soothe synth (no third-party audio; seeds,
shapes and sha256 in ``VENDORED.json`` beside them). The gestures are the manifest's
``gesture_map`` entries ``cue_<kind>``; core validates the sounds when the library is loaded.
"""

from __future__ import annotations

import wave
from functools import cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
from roboemot.voice.cues import CUE_KINDS, CueSound

CUES_DIR = Path(__file__).parent / "data" / "cues"
PCM_FULL_SCALE = 32768.0


def read_wav(path: Path) -> CueSound:
    """A 16-bit mono WAV as float samples in [-1, 1]."""
    with wave.open(str(path), "rb") as src:
        if src.getnchannels() != 1 or src.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit mono")
        pcm = np.frombuffer(src.readframes(src.getnframes()), "<i2")
        return CueSound((pcm / PCM_FULL_SCALE).astype(np.float32), src.getframerate())


@cache
def _read() -> MappingProxyType[str, CueSound]:
    return MappingProxyType(
        {kind: read_wav(path) for kind in CUE_KINDS if (path := CUES_DIR / f"{kind}.wav").exists()}
    )


def library() -> MappingProxyType[str, CueSound]:
    """The cue sounds by kind (read once per process)."""
    return _read()
