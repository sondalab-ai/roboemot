"""The Pollen emotions import: conversion rules, drift check and the vendored data (SPEC 4A, 7).

Offline: the conversion is tested on synthetic moves, the vendored files against their manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from roboemot.affect.clips import validate_clip_library
from roboemot.config.schema import EXPRESSION_MAX_CYCLE_S
from roboemot.manifest import load_manifest
from roboemot_reachymini import app as app_module
from roboemot_reachymini.driver import head_matrix

ROOT = Path(__file__).resolve().parents[3]


def _script():
    spec = importlib.util.spec_from_file_location(
        "import_pollen_emotions", ROOT / "scripts/import_pollen_emotions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMP = _script()
MANIFEST = load_manifest(
    app_module.DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True
)
HOMES = IMP.manifest_homes()
ANGLES = {n for n, d in MANIFEST.dofs.items() if d.kind == "angle"}
TOL = IMP.RDP_TOLERANCE_DEG + IMP.ROUND_DEG  # simplification plus rounding, in degrees
STEP = 0.02


def move(duration=2.0, head=None, antennas=None, body=None, translation=(0, 0, 0)) -> bytes:
    """A synthetic upstream move at 50 Hz; each argument is a function of t (deg), or None."""
    times = [round(i * STEP, 6) for i in range(int(duration / STEP) + 1)]
    frames = []
    for t in times:
        roll, pitch, yaw = head(t) if head else (0.0, 0.0, 0.0)
        pose = head_matrix(
            {
                "head.roll": math.radians(roll),
                "head.pitch": math.radians(pitch),
                "head.yaw": math.radians(yaw),
                "head.x": translation[0],
                "head.y": translation[1],
                "head.z": translation[2],
            }
        )
        right, left = antennas(t) if antennas else (HOMES["antenna_right"], HOMES["antenna_left"])
        frames.append(
            {
                "head": pose.tolist(),
                "antennas": [math.radians(right), math.radians(left)],
                "body_yaw": math.radians(body(t)) if body else 0.0,
                "check_collision": False,
            }
        )
    return json.dumps(
        {"description": "synthetic", "time": times, "set_target_data": frames}
    ).encode()


def convert(raw: bytes) -> dict[str, list[list[float]]]:
    return IMP.convert(raw, "synthetic", HOMES)[0]


def at(frames: list[list[float]], t: float) -> float:
    return float(np.interp(t, [f[0] for f in frames], [f[1] for f in frames]))


# ------------------------------------------------------------------ conversion


def test_homes_come_from_the_manifest():
    assert HOMES["antenna_right"] == -10 and HOMES["antenna_left"] == 10
    assert HOMES["head.pitch"] == 0 and HOMES["body_yaw"] == 0


def test_a_robot_at_home_gives_no_channel():
    assert convert(move()) == {}


def test_head_angles_follow_the_drivers_euler_convention():
    times, series, _ = IMP.parse_move(move(head=lambda t: (10.0, -20.0, 30.0)), "synthetic")
    assert series["head.roll"][0] == pytest.approx(10.0)
    assert series["head.pitch"][0] == pytest.approx(-20.0)
    assert series["head.yaw"][0] == pytest.approx(30.0)
    assert times[0] == 0 and times[-1] == pytest.approx(2.0)


def test_a_rotation_becomes_its_roll_pitch_and_yaw_offsets():
    out = convert(move(head=lambda t: (10 * math.sin(math.pi * t), 0.0, 0.0)))
    assert set(out) == {"head.roll"}  # pitch and yaw never leave home
    shift = IMP.EASE_S
    for t in (0.5, 1.0, 1.5):
        assert at(out["head.roll"], t + shift) == pytest.approx(10 * math.sin(math.pi * t), abs=TOL)


def test_translation_is_dropped():
    out = convert(move(translation=(0.02, -0.03, 0.01)))
    assert out == {}
    out = convert(move(head=lambda t: (0.0, 15.0, 0.0), translation=(0.02, 0.0, 0.0)))
    assert set(out) == {"head.pitch"}


def test_antennas_are_offsets_about_their_manifest_home_in_sdk_order():
    right_only = convert(
        move(antennas=lambda t: (HOMES["antenna_right"] + 20.0, HOMES["antenna_left"]))
    )
    assert set(right_only) == {"antenna_right"}  # index 0 is the right antenna
    assert at(right_only["antenna_right"], 1.0) == pytest.approx(20.0, abs=TOL)
    left_only = convert(
        move(antennas=lambda t: (HOMES["antenna_right"], HOMES["antenna_left"] - 15.0))
    )
    assert set(left_only) == {"antenna_left"}
    assert at(left_only["antenna_left"], 1.0) == pytest.approx(-15.0, abs=TOL)


def test_offsets_are_about_home_not_about_the_first_frame():
    out = convert(move(head=lambda t: (0.0, 15.0, 0.0)))  # a lowered head all the way through
    assert at(out["head.pitch"], 1.0) == pytest.approx(15.0, abs=TOL)


def test_body_yaw_is_kept():
    out = convert(move(body=lambda t: 25.0 * math.sin(math.pi * t / 2)))
    assert set(out) == {"body_yaw"} and max(v for _, v in out["body_yaw"]) > 20


def test_every_channel_eases_in_from_zero_and_out_to_zero_and_ends_together():
    raw = move(
        head=lambda t: (8.0, 12.0, -6.0),
        antennas=lambda t: (HOMES["antenna_right"] + 30, HOMES["antenna_left"] - 30),
        body=lambda t: 10.0,
    )
    out = convert(raw)
    assert len(out) == 6
    end = 2.0 + 2 * IMP.EASE_S
    for name, frames in out.items():
        assert frames[0] == [0.0, 0.0] and frames[-1][1] == 0.0, name
        assert frames[-1][0] == pytest.approx(end), name
        assert frames[1][0] > 0 and frames[1][1] != 0  # it ramps, it does not jump
    assert len({f[-1][0] for f in out.values()}) == 1


def test_times_are_milliseconds_and_strictly_increasing_even_with_repeated_timestamps():
    doc = json.loads(move(head=lambda t: (0.0, 10 * math.sin(3 * t), 0.0)))
    doc["time"][10] = doc["time"][9]  # two samples in one tick, as in a few upstream moves
    out = convert(json.dumps(doc).encode())
    for name, frames in out.items():
        times = [f[0] for f in frames]
        assert all(b > a for a, b in zip(times, times[1:], strict=False)), name
        assert all(round(t, 3) == t for t in times), name


def test_simplification_stays_within_the_tolerance_of_the_dense_signal():
    def wiggle(t):
        return 14 * math.sin(2 * math.pi * 1.5 * t) + 5 * math.sin(2 * math.pi * 4 * t)

    out = convert(move(duration=4.0, head=lambda t: (0.0, wiggle(t), 0.0)))
    times, series, _ = IMP.parse_move(move(duration=4.0, head=lambda t: (0.0, wiggle(t), 0.0)), "s")
    t_dense, v_dense = IMP.eased_channel(times, series["head.pitch"])
    err = max(abs(at(out["head.pitch"], t) - v) for t, v in zip(t_dense, v_dense, strict=True))
    assert err <= TOL + 1e-6
    assert len(out["head.pitch"]) < len(t_dense) * 0.6  # and it is simpler than the recording


def test_values_are_rounded_to_a_tenth_of_a_degree():
    out = convert(move(head=lambda t: (0.0, 10 * math.sin(t) + 0.0123, 0.0)))
    for _, v in out["head.pitch"]:
        assert round(v, 1) == v


def test_a_channel_that_never_leaves_the_tolerance_is_dropped():
    out = convert(move(head=lambda t: (0.3 * math.sin(5 * t), 12.0, 0.0)))
    assert set(out) == {"head.pitch"}


def test_a_clip_over_the_cycle_limit_is_refused():
    with pytest.raises(ValueError, match="EXPRESSION_MAX_CYCLE_S"):
        convert(move(duration=EXPRESSION_MAX_CYCLE_S, head=lambda t: (0.0, 10.0, 0.0)))


def test_untrusted_input_is_checked_not_trusted():
    with pytest.raises(ValueError):
        IMP.parse_move(b"x" * (IMP.MAX_FILE_BYTES + 1), "big")
    with pytest.raises(ValueError, match="differ in length"):
        IMP.parse_move(json.dumps({"time": [0, 1], "set_target_data": []}).encode(), "short")
    with pytest.raises(ValueError, match="non-finite"):
        bad = json.loads(move())
        bad["time"][3] = float("nan")
        IMP.parse_move(json.dumps(bad).encode(), "nan")


# ------------------------------------------------------------------ drift check and the pins


def test_the_vendored_data_matches_the_manifest_and_the_pins(capsys):
    assert IMP.check() == 0, capsys.readouterr().out


def test_the_pins_are_a_pinned_revision_and_sha256_digests():
    assert len(IMP.REVISION) == 40 and int(IMP.REVISION, 16) >= 0
    assert IMP.SHA256 and all(n.endswith(".json") for n in IMP.SHA256)
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in IMP.SHA256.values())
    doc = json.loads((IMP.OUT_DIR / "VENDORED.json").read_text())
    assert doc["revision"] == IMP.REVISION and doc["license"] == "Apache-2.0"
    assert doc["files"] == IMP.SHA256
    assert (
        doc["clips_sha256"] == hashlib.sha256((IMP.OUT_DIR / "clips.yaml").read_bytes()).hexdigest()
    )


@pytest.fixture
def copy(tmp_path, monkeypatch):
    folder = tmp_path / "emotions"
    shutil.copytree(IMP.OUT_DIR, folder)
    monkeypatch.setattr(IMP, "OUT_DIR", folder)
    return folder


def test_an_edited_clips_file_fails_the_check(copy, capsys):
    with (copy / "clips.yaml").open("a") as handle:
        handle.write("# edited\n")
    assert IMP.check() == 1
    assert "clips.yaml" in capsys.readouterr().out


def test_a_changed_manifest_fails_the_check(copy, capsys):
    doc = json.loads((copy / "VENDORED.json").read_text())
    doc["revision"] = "0" * 40
    (copy / "VENDORED.json").write_text(json.dumps(doc))
    assert IMP.check() == 1
    assert "revision" in capsys.readouterr().out


def test_a_changed_licence_or_a_stray_file_fails_the_check(copy, capsys):
    (copy / "LICENSE").write_text("not apache")
    (copy / "extra.txt").write_text("x")
    assert IMP.check() == 1
    out = capsys.readouterr().out
    assert "LICENSE" in out and "extra.txt" in out


def test_a_source_that_differs_from_its_pin_is_refused(tmp_path):
    name = next(iter(IMP.SHA256))
    (tmp_path / name).write_bytes(b"{}")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        IMP.read_sources(tmp_path)


# ------------------------------------------------------------------ the vendored data


LIB = yaml.safe_load((IMP.OUT_DIR / "clips.yaml").read_text())


def test_the_licence_is_the_apache_text_beside_the_clips():
    text = (IMP.OUT_DIR / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    header = (IMP.OUT_DIR / "clips.yaml").read_text().splitlines()[:4]
    assert IMP.REVISION in header[0] and "Apache-2.0" in header[1]


def test_every_upstream_move_is_a_clip_of_the_manifest_dofs():
    assert set(LIB) == {n.removesuffix(".json") for n in IMP.SHA256}
    assert len(LIB) >= 81
    validated = validate_clip_library(LIB, MANIFEST.channels)
    assert set(validated) == set(LIB)
    for name, entry in validated.items():
        assert set(entry.channels) <= ANGLES, name  # no translation, no unknown dof
        assert "head.x" not in entry.channels


def test_every_clip_is_within_the_cycle_limit_and_starts_and_ends_at_zero():
    for name, entry in LIB.items():
        ends = set()
        for channel, frames in entry["channels"].items():
            assert frames[0] == [0.0, 0.0] and frames[-1][1] == 0.0, (name, channel)
            times = [f[0] for f in frames]
            assert all(b > a for a, b in zip(times, times[1:], strict=False)), (name, channel)
            assert times[-1] <= EXPRESSION_MAX_CYCLE_S, name
            ends.add(times[-1])
        assert len(ends) == 1, name  # all channels end together
        assert next(iter(ends)) > 2 * IMP.EASE_S, name


def test_the_clip_names_the_shipped_mapping_uses_are_all_vendored():
    behaviour = MANIFEST.model.behaviour
    used = {v.clip for e in behaviour["expression_map"].values() for v in _variants(e)}
    used |= {
        v.clip for k in ("yawn", "stretch") for v in _variants(behaviour["circadian_motion"][k])
    }
    assert used <= set(LIB)


def _variants(entry):
    from roboemot.config.schema import ExpressionEntry

    return ExpressionEntry.model_validate(entry).variants
