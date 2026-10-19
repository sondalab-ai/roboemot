"""Voice model registry (SPEC 8 stages 2, 3, 7): Silero VAD, Vosk, Piper voices.

Same manifest format, fetcher and verification as the vision models
(``roboemot.vision.models``); this module only names the voice manifest and folder. The folder
is ``--voice-dir``, ``voice.models_dir`` or, by default, ``voice-models/`` in the app data
folder, so uninstalling the app removes the models with it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import httpx

from .. import paths
from ..vision import models as _models
from ..vision.models import FetchReport, ModelError, ModelFile, ModelSpec

MANIFEST = Path(__file__).with_name("models.yaml")
ASR_ROLE, TTS_ROLE = "asr_", "tts_"


def load_manifest() -> dict[str, ModelSpec]:
    """Parse ``voice/models.yaml`` into ``{name: spec}`` (cached, treat as read-only)."""
    return _models.load_manifest(MANIFEST)


@functools.cache
def supported_languages() -> frozenset[str]:
    """Languages the voice stack can listen to and speak: those with both an ``asr_<lang>`` and a
    ``tts_<lang>`` model in the manifest (cached: the manifest ships with the package)."""
    roles = {spec.role for spec in load_manifest().values()}
    heard = {r.removeprefix(ASR_ROLE) for r in roles if r.startswith(ASR_ROLE)}
    spoken = {r.removeprefix(TTS_ROLE) for r in roles if r.startswith(TTS_ROLE)}
    return frozenset(heard & spoken)


def models_dir(configured: str = "", data_dir: str = "") -> Path:
    """The voice models folder (``~`` expanded; empty: ``voice-models/`` in the data folder)."""
    return Path(configured).expanduser() if configured else paths.voice_models_subdir(data_dir)


def fetch(
    names: list[str],
    root: Path,
    *,
    client: httpx.Client | None = None,
    quantizer: _models.Quantizer = _models.quantize_dynamic_file,
    requantize: bool = False,
) -> FetchReport:
    """Fetch the named voice models into ``root`` (see ``roboemot.vision.models.fetch``)."""
    return _models.fetch(
        names, root, client=client, quantizer=quantizer, requantize=requantize, manifest=MANIFEST
    )


def model_file(root: Path, name: str, suffix: str = "") -> Path:
    """Verified path of the first file of voice model ``name`` whose path ends with ``suffix``.

    Raises ``ModelError`` when the model is unknown or its marker is missing or stale, so a
    backend never loads an unverified file.
    """
    spec = load_manifest().get(name)
    if spec is None:
        raise ModelError(f"unknown voice model {name!r}; known: {sorted(load_manifest())}")
    for f in spec.files:
        if f.path.endswith(suffix):
            return _models.check_file(root, f)
    raise ModelError(f"{name}: no file ending with {suffix!r}")


__all__ = [
    "MANIFEST",
    "FetchReport",
    "ModelError",
    "ModelFile",
    "ModelSpec",
    "fetch",
    "load_manifest",
    "model_file",
    "models_dir",
    "supported_languages",
]
