"""SPEC 4, 4A, 12: manifest parsing, limits, capabilities, negotiation."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml
from roboemot.cli import main as cli
from roboemot.manifest import ManifestError, load_manifest, negotiate

MANIFESTS = Path(__file__).resolve().parents[3] / "configs" / "manifests"
REACHY = MANIFESTS / "reachy-mini-wireless.yaml"
DESK = MANIFESTS / "desk-head-v1.yaml"


def test_reachy_manifest_rejected_until_dynamics_measured():
    with pytest.raises(ManifestError, match="not measured"):
        load_manifest(REACHY)


def test_reachy_manifest_dev_mode_dofs_and_units():
    m = load_manifest(REACHY, allow_unmeasured_dynamics=True)
    assert set(m.dofs) == {
        "head.roll",
        "head.pitch",
        "head.yaw",
        "head.x",
        "head.y",
        "head.z",
        "body_yaw",
        "antenna_right",
        "antenna_left",
    }
    pitch = m.dofs["head.pitch"]
    assert pitch.min == pytest.approx(math.radians(-40)) and not pitch.measured
    assert m.dofs["head.z"].max == pytest.approx(0.015) and m.dofs["head.z"].kind == "translation"
    assert m.dofs["antenna_right"].home == pytest.approx(math.radians(-10))
    assert m.constraints[0].max_deg == 65


def test_reachy_measured_dynamics_pass():
    raw = yaml.safe_load(REACHY.read_text())
    raw["dynamics"] = {
        "head": {
            "max_vel_dps": 200,
            "max_acc_dps2": 1000,
            "max_lin_vel_mps": 0.05,
            "max_lin_acc_mps2": 0.5,
        },
        "body_yaw": {"max_vel_dps": 120, "max_acc_dps2": 500},
        "antennas": {"max_vel_dps": 400, "max_acc_dps2": 3000},
    }
    m = load_manifest(raw)
    assert all(d.measured for d in m.dofs.values())


def test_capabilities_and_negotiation():
    reachy = load_manifest(REACHY, allow_unmeasured_dynamics=True)
    caps = reachy.capabilities()
    assert {"camera", "microphone", "doa", "speaker", "imu", "teach_mode", "motion"} <= caps
    assert "estop" not in caps and "leds" not in caps
    neg = negotiate(caps)
    assert "mirror" in neg.started and neg.started["mirror"] == {"teach_mode"}
    assert neg.disabled == {}

    desk = load_manifest(DESK)
    dcaps = desk.capabilities()
    assert {"leds", "estop", "proximity"} <= dcaps
    no_cam = negotiate(dcaps - {"camera"})
    assert no_cam.disabled["vision"] == {"camera"}
    assert "mirror" in no_cam.disabled


def test_joint_without_dynamics_rejected():
    raw = yaml.safe_load(DESK.read_text())
    bad = copy.deepcopy(raw)
    del bad["joints"][0]["limits"]["max_vel_dps"]
    with pytest.raises(ManifestError, match="velocity and acceleration"):
        load_manifest(bad)


def test_bad_references_and_unknown_keys_rejected():
    raw = yaml.safe_load(REACHY.read_text())
    bad = copy.deepcopy(raw)
    bad["constraints"][0]["a"] = "head.nope"
    with pytest.raises(ManifestError, match="unknown dof"):
        load_manifest(bad, allow_unmeasured_dynamics=True)
    bad = copy.deepcopy(raw)
    bad["typo_section"] = {}
    with pytest.raises(ManifestError):
        load_manifest(bad, allow_unmeasured_dynamics=True)


def test_cli_manifest_check(capsys):
    assert cli(["manifest", "check", str(REACHY)]) == 1
    assert cli(["manifest", "check", str(REACHY), "--dev"]) == 0
    assert "reachy-mini-wireless" in capsys.readouterr().out


def test_inverted_or_non_positive_limits_rejected():
    raw = yaml.safe_load(DESK.read_text())
    bad = copy.deepcopy(raw)
    bad["joints"][0]["limits"].update(min_deg=90, max_deg=-90)
    with pytest.raises(ManifestError):
        load_manifest(bad)
    bad = copy.deepcopy(raw)
    bad["joints"][0]["limits"]["max_vel_dps"] = -30
    with pytest.raises(ManifestError):
        load_manifest(bad)
    reachy = yaml.safe_load(REACHY.read_text())
    reachy["pose_groups"]["head"]["limits"]["z_mm"] = [15, -55]
    with pytest.raises(ManifestError):
        load_manifest(reachy, allow_unmeasured_dynamics=True)


def test_home_outside_limits_rejected():
    raw = yaml.safe_load(DESK.read_text())
    raw["joints"][0]["home_deg"] = 500
    with pytest.raises(ManifestError, match="home_deg"):
        load_manifest(raw)
