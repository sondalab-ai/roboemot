"""Recorded expression clips of Reachy Mini (SPEC 4A, Recorded emotions; SPEC 7).

Registered in the ``roboemot.expression_clips`` entry-point group under ``reachy-mini-wireless``.
Serves the package data ``data/emotions/clips.yaml`` (Pollen's emotions library converted by
``scripts/import_pollen_emotions.py``, Apache-2.0, see ``LICENSE`` beside it) as a read-only
mapping of clip name to keyframe entry. The file is parsed once per process; core validates
every clip against the manifest when the library is loaded.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:  # PyYAML built without libyaml
    from yaml import SafeLoader as _Loader

CLIPS_FILE = Path(__file__).parent / "data" / "emotions" / "clips.yaml"


@cache
def _read() -> MappingProxyType[str, Any]:
    with CLIPS_FILE.open(encoding="utf-8") as handle:
        return MappingProxyType(yaml.load(handle, Loader=_Loader))  # noqa: S506 (safe loader)


def library() -> MappingProxyType[str, Any]:
    """The recorded clips by name: ``{"channels": {channel: [[t_s, offset_deg], ...]}}`` each."""
    return _read()
