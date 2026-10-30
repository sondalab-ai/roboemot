"""Reachy Mini's non-verbal cue sounds (SPEC 8, Non-verbal cues; SPEC 4A, Hierarchy gate)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from importlib.metadata import entry_points

import numpy as np
import pytest
from roboemot.manifest import load_manifest
from roboemot.voice.cues import CUE_KINDS, ENTRY_POINT_GROUP, load_cue_sounds
from roboemot_reachymini import app as app_module
from roboemot_reachymini import cues as cues_module

MANIFEST = load_manifest(
    app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
)
QUIET_PEAK = 0.13  # about -18 dBFS: the cues are made quiet


def test_the_library_is_registered_under_the_manifest_robot_id():
    found = {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}
    assert found[MANIFEST.robot.id] == "roboemot_reachymini.cues:library"


def test_every_cue_has_a_short_quiet_sound_core_accepts():
    sounds = load_cue_sounds(MANIFEST)
    assert set(sounds) == set(CUE_KINDS)
    for kind, sound in sounds.items():
        assert sound.rate_hz == 16000 and sound.duration_s < 1.0, kind
        peak = float(np.max(np.abs(sound.samples)))
        assert 0.0 < peak <= QUIET_PEAK, kind
        assert sound.samples[0] == 0.0 and sound.samples[-1] == 0.0  # no click


def test_the_library_is_read_once_and_read_only():
    lib = cues_module.library()
    assert lib is cues_module.library()
    with pytest.raises(TypeError):
        lib["x"] = None  # type: ignore[index]


def test_the_vendored_sounds_match_their_record_and_a_fresh_render():
    """No third-party audio: each WAV is the soothe synth's render recorded in VENDORED.json."""
    record = json.loads((cues_module.CUES_DIR / "VENDORED.json").read_text(encoding="utf-8"))
    for kind, entry in record["cues"].items():
        data = (cues_module.CUES_DIR / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], kind
    path = cues_module.CUES_DIR.parents[5] / "scripts" / "make_cue_sounds.py"
    spec = importlib.util.spec_from_file_location("make_cue_sounds", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--check"]) == 0
