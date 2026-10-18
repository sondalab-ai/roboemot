"""Where the app keeps its data on disk (SPEC 4A Installer, 12 Data).

One folder holds everything RoboEmot writes: the registry (``registry/``), the persona trait
ratings (``llm_traits.json``) and, unless ``vision.models_dir`` says otherwise, the downloaded
models (``models/``, and ``voice-models/`` for the voice ones). Keeping it in one
place lets the installer record it and the uninstaller remove it with a single path.

The folder is, in order: the ``data_dir`` config value, ``$ROBOEMOT_DATA_DIR``, or
``~/.local/share/roboemot``. It is private (0o700); parents the call has to create get the
process umask like any ``mkdir -p``.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

log = logging.getLogger(__name__)

ENV_DIR = "ROBOEMOT_DATA_DIR"
DEFAULT_REL = Path(".local") / "share" / "roboemot"
DIR_MODE = 0o700
GROUP_OTHER_BITS = 0o077
REGISTRY_SUBDIR = "registry"
MODELS_SUBDIR = "models"
VOICE_MODELS_SUBDIR = "voice-models"
TRAITS_CACHE_FILE = "llm_traits.json"
ROUTE_CHECKS_FILE = "route_checks.json"
SECRETS_FILE = "secrets.json"
SETTINGS_OVERRIDES_FILE = "webui_settings.json"


def data_path(configured: str = "") -> Path:
    """The data folder without touching the filesystem (``~`` expanded, made absolute)."""
    if configured:
        return Path(configured).expanduser().absolute()
    env = os.environ.get(ENV_DIR)
    if env:
        return Path(env).expanduser().absolute()
    return Path.home() / DEFAULT_REL


def data_dir(configured: str = "") -> Path:
    """The data folder, created private (0o700) when missing and tightened when it is not.

    An existing folder owned by this user with group or other access gets its mode lowered
    (logged); one owned by someone else is left alone and raises ``PermissionError``.
    """
    path = data_path(configured)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(mode=DIR_MODE)
            path.chmod(DIR_MODE)  # mkdir's mode is masked by the umask
            return path
        except FileExistsError:  # another process created it first: check it below
            pass
    info = path.stat()
    if info.st_uid != os.getuid():
        raise PermissionError(f"data folder {path} is not owned by this user")
    if stat.S_IMODE(info.st_mode) & GROUP_OTHER_BITS:
        log.warning("data folder %s was accessible to others; mode set to 0o700", path)
        path.chmod(DIR_MODE)
    return path


def registry_dir(configured: str = "") -> Path:
    """``<data folder>/registry`` (not created here)."""
    return data_path(configured) / REGISTRY_SUBDIR


def models_subdir(configured: str = "") -> Path:
    """``<data folder>/models``, the default models folder (not created here)."""
    return data_path(configured) / MODELS_SUBDIR


def voice_models_subdir(configured: str = "") -> Path:
    """``<data folder>/voice-models``, the default voice models folder (not created here)."""
    return data_path(configured) / VOICE_MODELS_SUBDIR


def traits_cache_path(configured: str = "") -> Path:
    """``<data folder>/llm_traits.json``, the persona trait ratings cache (not created here).
    Inside the data folder, so the installer's uninstall removes it with the folder."""
    return data_path(configured) / TRAITS_CACHE_FILE


def route_checks_path(configured: str = "") -> Path:
    """``<data folder>/route_checks.json``, the last conformance report per LLM route (SPEC 8;
    not created here). Inside the data folder, so uninstall removes it with the folder."""
    return data_path(configured) / ROUTE_CHECKS_FILE


def secrets_path(configured: str = "") -> Path:
    """``<data folder>/secrets.json``, the secret store (SPEC 8, 12 Data; not created here). Mode
    0o600; inside the data folder, so uninstall removes it with the folder."""
    return data_path(configured) / SECRETS_FILE


def settings_overrides_path(configured: str = "") -> Path:
    """``<data folder>/webui_settings.json``, the web UI's persisted settings edits (SPEC 11A,
    12 Data; not created here). Mode 0o600; inside the data folder, so uninstall removes it."""
    return data_path(configured) / SETTINGS_OVERRIDES_FILE


PIPER_VENV_SUBDIR = "piper-venv"


def piper_venv(configured: str = "") -> Path:
    """The virtualenv the installer creates for Piper (GPL-3.0, so it runs as its own process)."""
    return data_path(configured) / PIPER_VENV_SUBDIR


def piper_python(configured: str = "") -> Path:
    """Interpreter of that virtualenv: the default for ``voice.piper.python``."""
    return piper_venv(configured) / "bin" / "python"


__all__ = [
    "DEFAULT_REL",
    "ENV_DIR",
    "MODELS_SUBDIR",
    "PIPER_VENV_SUBDIR",
    "REGISTRY_SUBDIR",
    "ROUTE_CHECKS_FILE",
    "SECRETS_FILE",
    "SETTINGS_OVERRIDES_FILE",
    "TRAITS_CACHE_FILE",
    "VOICE_MODELS_SUBDIR",
    "data_dir",
    "data_path",
    "models_subdir",
    "piper_python",
    "piper_venv",
    "registry_dir",
    "route_checks_path",
    "secrets_path",
    "settings_overrides_path",
    "traits_cache_path",
    "voice_models_subdir",
]
