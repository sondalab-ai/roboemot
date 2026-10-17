"""Voice model registry (SPEC 8 stages 2, 3, 7): Silero VAD, Vosk, Piper voices.

Same manifest format, fetcher and verification as the vision models
(``roboemot.vision.models``); this module only names the voice manifest and folder. The folder
is ``--voice-dir``, ``voice.models_dir`` or, by default, ``voice-models/`` in the app data
folder, so uninstalling the app removes the models with it.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..vision import models as _models
from ..vision.models import FetchReport, ModelError, ModelFile, ModelSpec

MANIFEST = Path(__file__).with_name("models.yaml")


def load_manifest() -> dict[str, ModelSpec]:
    """Parse ``voice/models.yaml`` into ``{name: spec}`` (cached, treat as read-only)."""
    return _models.load_manifest(MANIFEST)


def models_dir(configured: str = "", data_dir: str = "") -> Path:
    """The voice models folder (``~`` expanded; empty: ``voice-models/`` in the data folder)."""
    return Path(configured).expanduser() if configured else paths.voice_models_subdir(data_dir)


def fetch(names: list[str], root: Path, **kwargs) -> FetchReport:
    """Fetch the named voice models into ``root`` (see ``roboemot.vision.models.fetch``)."""
    return _models.fetch(names, root, manifest=MANIFEST, **kwargs)


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
]
